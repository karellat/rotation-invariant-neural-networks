import torch 
import math
from einops import rearrange
from loguru import logger
from typing import Sequence, Union


from hippy2d.complex_invariants_2d import get_complex_monomial
from hippy2d.utils import tukey_2d, get_default_complex, get_circular_mask, SafeAtan2

BASIS_P0 = 1
BASIS_Q0 = 0
KERNEL_SIZE = 15
N_RINGS = 5
MAX_ORDER = 4

import torch



def escnn_style_rings_sigmas(kernel_size: int, n_rings: int):
    """
    Rings: linearly spaced from 0 to floor((k-1)/2).
    Sigmas: 0.005 at the center (if present), 0.6 on interior, 0.4 on outermost.
    """
    assert kernel_size % 2 == 1
    rmax = (kernel_size - 1) // 2
    rings = torch.linspace(0.0, float(rmax), steps=n_rings).tolist()
    if rings[0] == 0.0:
        sigma = [0.005] + [0.6] * (n_rings - 2) + [0.4] if n_rings > 1 else [0.4]
    else:
        sigma = [0.6] * (n_rings - 1) + [0.4]
    return rings, sigma


class RadialGaussianConv2d(torch.nn.Module):
    """
    2D convolution with a radially symmetric kernel parameterized as a
    linear combination of Gaussian rings.

    Kernel(x, y) = sum_j coeff[out, in, j] * exp(-(r - rings[j])^2 / (2*sigma[j]^2)),
    where r = sqrt(x^2 + y^2) on the kernel grid.

    Simplified: stride=1, padding=0 (valid), groups=1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rings: Sequence[float],
        sigma: Union[float, Sequence[float]],
    ):
        super().__init__()
        if isinstance(sigma, (int, float)):
            sigma = [float(sigma)] * len(rings)
        assert len(rings) == len(sigma) and len(rings) > 0, "rings and sigma must match and be non-empty"
        for r in rings:
            assert r >= 0.0
        for s in sigma:
            assert s > 0.0


        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.padding = (self.kernel_size - 1) // 2

        # Learnable ring coefficients: (out, in, R)
        R = len(rings)
        self.coeff = torch.nn.Parameter(torch.zeros(out_channels, in_channels, R))

        # Fixed ring parameters (buffers)
        self.register_buffer("rings", torch.tensor(rings, dtype=torch.get_default_dtype()))   # (R,)
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.get_default_dtype()))   # (R,)

        # Pre-sample the radial basis on the kernel grid: (R, k, k)
        basis = self._build_radial_basis(self.kernel_size, self.rings, self.sigma)
        self.register_buffer("_basis", basis)  # (R, k, k)

        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.in_channels * self.coeff.shape[-1]  # in_channels * n_rings
        bound = 1.0 / math.sqrt(max(1, fan_in))
        torch.nn.init.uniform_(self.coeff, -bound, bound)

    @staticmethod
    def _radius_grid(k: int, device=None, dtype=None) -> torch.Tensor:
        """Radius r = sqrt(x^2 + y^2) over a k×k grid centered at 0."""
        ys = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
        xs = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.sqrt(yy ** 2 + xx ** 2)  # (k, k)

    @classmethod
    def _build_radial_basis(cls, k: int, rings: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        r = cls._radius_grid(k, device=rings.device, dtype=rings.dtype)  # (k, k)
        rings = rings.view(-1, 1, 1)   # (R,1,1)
        sigma = sigma.view(-1, 1, 1)   # (R,1,1)
        return torch.exp(-0.5 * (r.unsqueeze(0) - rings) ** 2 / (sigma ** 2))  # (R,k,k)

    def _assemble_weight(self, dtype=None) -> torch.Tensor:
        # (out, in, R) × (R, k, k) -> (out, in, k, k)
        basis = self._basis.to(dtype=dtype if dtype is not None else self._basis.dtype)
        return torch.einsum("oir,rhw->oihw", self.coeff, basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._assemble_weight(dtype=x.dtype)
        # Valid convolution: stride=1, padding=0, dilation=1, groups=1, bias=None
        return torch.nn.functional.conv2d(x, w, bias=None, stride=1, padding=self.padding, dilation=1, groups=1)

    @torch.no_grad()
    def current_kernel(self) -> torch.Tensor:
        """Get the assembled kernel: (out_channels, in_channels, k, k)."""
        return self._assemble_weight()

class ComplexInvariantConv2D(torch.nn.Module):
    def __init__(self,
                 kernel_size:int,
                 in_channels: int,
                 input_size: int,
                 out_channels:int,
                 max_order:int = MAX_ORDER, 
                 basis_p0:int = 1,
                 basis_q0:int = 0, 
                 circular_padding:str ="tukey",
                 padding: str = "same", 
                 prenormalize: str = "none",
                 preserve_energy: bool = False,
                 invariant_norm: str = "none", # none, rayleigh, gauss
                 magnitude_normalization = "copy", # flussers r^z, copy r, one 1 
                 zero_out_middles: bool = False,
                 eps=1e-8):
        super(ComplexInvariantConv2D, self).__init__()
        self.filter_size = kernel_size
        self.max_order = max_order
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basis_p0 = basis_p0
        self.basis_q0 = basis_q0
        self.eps = eps
        assert magnitude_normalization in ["one", "copy", "flussers"] 
        self.magnitude_normalization = magnitude_normalization
        assert invariant_norm in ["none", "rayleigh", "gauss"]
        self.invariant_norm = invariant_norm

        # Asserts 
        assert kernel_size % 2 == 1, "Filter size must be odd"
        assert max_order >= 0, "Max order must be non-negative"
        assert in_channels > 0, "Number of input channels must be positive"
        assert out_channels > 0, "Number of output channels must be positive"
        assert kernel_size >= 5, "Filters must be decent size for complex invariants"

        assert self.basis_p0 >= 0 and self.basis_q0 >= 0, "Basis indices must be non-negative integers"
        assert self.basis_p0 + self.basis_q0 <= max_order, "Basis indices must not exceed the maximum order"
        assert self.basis_p0 - self.basis_q0 == 1, "Basis indices must differ by 1 for the normalization term"

        # Prepare the fixed filters corresponding to the complex monomials
        filters = []
        ind = []
        types = []
        # Add the p0q0 term
        filters.append(get_complex_monomial(self.filter_size,
                                             self.basis_q0,
                                             self.basis_p0,
                                             dtype=torch.get_default_dtype()))
        types.append(self.basis_q0 - self.basis_p0)
        ind.append((self.basis_q0, self.basis_p0))
        # Add the complex monomials up to the max order
        for p in range(0, self.max_order + 1):
            for q in range(0, min(self.max_order + 1-p, p + 1)):
                filters.append(get_complex_monomial(self.filter_size, p, q, dtype=torch.get_default_dtype()))
                ind.append((p, q))
                types.append(self.basis_q0 - self.basis_p0)
        self.ind = ind

        # NOTE: This part can be shared by all the layers, that can save memory 
        filters = rearrange(filters, 'n h w -> n 1 h w')
        if zero_out_middles:
            for idx, type in enumerate(types):
                # Mask middle 
                if type != 0:
                    filters[idx, 0, kernel_size//2, kernel_size//2] = 0

        Ch, _, _, _ = filters.shape
        self.complex_conv_groups = Ch
        # Radial padding
        if circular_padding == "tukey":
            # Use Tukey window for circular padding
            mask = torch.from_numpy(tukey_2d(self.filter_size, 0.5)).to(dtype=torch.get_default_dtype())
        elif circular_padding == "circular":
            # Use circular padding
            mask = get_circular_mask(self.filter_size, dtype=torch.get_default_dtype())
        elif circular_padding == "none":
            # No padding, just use the filters as they are
            mask = 1
        else:
            raise ValueError(f"Unknown circular padding type: {circular_padding}. Use 'tukey' or 'none'.")
        filters = filters * mask
        if preserve_energy:
            filters = filters / filters.abs().sum(axis=(-2, -1), keepdim=True)

        filters = torch.cat(dim=0, tensors=[filters.real, filters.imag])

        # Normalization factor
        self.magnitude_normalization = magnitude_normalization

        self.padding = padding 
        self.register_buffer('filters', filters)
        self.exponents = torch.tensor([p-q for (p,q) in ind], dtype=torch.int64) # Skip the normalization and scaling term
        self.exponents = self.exponents[1:] # Skip the normalization term
        self.exponents = torch.nn.Parameter(self.exponents[None, :, None, None], requires_grad=False) # Broadcasting dimension
        self.ind = torch.tensor(ind, dtype=torch.uint16)
        self.num_invariants = self.exponents.shape[1] # Number of invariants and skip the normalization and scaling term
        # Prenormalization 
        if prenormalize == "none": 
            self.norm = torch.nn.Identity()
        elif prenormalize == "batch":
            self.norm = torch.nn.BatchNorm2d(self.num_invariants*self.in_channels*2,
                                             affine=False,
                                             dtype=torch.get_default_dtype())
        elif prenormalize == "layer":
            self.norm = torch.nn.LayerNorm(normalized_shape=[self.num_invariants*self.in_channels*2, input_size, input_size],
                                           bias=False,
                                           elementwise_affine=False)
        else: 
            raise ValueError(f"Unknown prenormalization type: {prenormalize}. Use 'none', 'batch', or 'layer'.")
            
        self.conv1x1 = torch.nn.Conv2d(in_channels=self.num_invariants*self.in_channels*2,
                                       out_channels=self.out_channels,
                                       kernel_size=1, 
                                       dtype=torch.get_default_dtype())

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complex invariant convolution layer.
        :param x: Input tensor of shape (batch_size, in_channels, height, width)
        :return: Output tensor of shape (batch_size, out_channels, height', width')
        """

        # Check input shape, but only for debug
        if __debug__:
            assert x.dim() == 4, "Input must be a 4D tensor"
            assert x.shape[1] == self.in_channels, f"Input channels {x.shape[1]} do not match expected {self.in_channels}"
            assert x.dtype == torch.get_default_dtype(), f"Input dtype {x.dtype} does not match expected {torch.get_default_dtype()}"
        
        # Act on batch and channels together
        x = rearrange(x, 'b c h w -> (b c) 1 h w', c=self.in_channels)

        # Apply the complex invariant convolution
        moments = torch.nn.functional.conv2d(x,
                                             weight=self.filters,
                                             padding=self.padding)

        moments = rearrange(moments, 'b (co ch) h w -> b co ch h w', 
                            ch=self.complex_conv_groups, co=2)
        if __debug__:
            assert moments.dtype == torch.get_default_dtype(), f"Output dtype {moments.dtype} does not match expected {torch.get_default_dtype()}"
            assert torch.all(~torch.isnan(moments)), "Output contains NaN values"

        normalization_moment = moments[:, :, 0]
        moments = moments[:, :, 1:]
        # Moivre's Theorem
        # Note: Sqrt of 0 has infty gradient, so we use eps to avoid it
        # Same for angles: atan2(0, 0) is also problematic
        norm_magnitude = torch.norm(normalization_moment, dim=1)  # Calculate the norm of the normalization moment

        # Avoid NaN output and NaN gradients
        norm_angle = SafeAtan2.apply(normalization_moment[:, 1:2],
                                     normalization_moment[:, 0:1],
                                     self.eps)
        if __debug__:
            assert torch.all(~torch.isnan(norm_angle)), "Normalization angle contains NaN values"
            assert torch.all(~torch.isinf(norm_angle)), "Normalization angle contains Inf values"
            # Magnitudes 
            assert torch.all(~torch.isnan(norm_magnitude)), "Normalization magnitude contains NaN values"
        
        if self.magnitude_normalization == "one": 
            powers = 1
        elif self.magnitude_normalization == "copy": 
            powers = norm_magnitude[:, None] 
        elif self.magnitude_normalization == "flussers":  
            powers = norm_magnitude[:, None] ** (self.exponents)
        else: 
            raise NotImplementedError(f"Unknown magnitude normalization {self.magnitude_normalization}")

        norm_angle = norm_angle * self.exponents
        normalization_factor_real = powers * torch.cos(norm_angle)
        normalization_factor_imag = powers * torch.sin(norm_angle)

        result = torch.stack(tensors=[
            moments[:, 0] * normalization_factor_real - moments[:, 1] * normalization_factor_imag,
            moments[:, 0] * normalization_factor_imag + moments[:, 1] * normalization_factor_real
        ], dim=1)  # Stack along the channel dimension

        if __debug__:
            assert result.dtype == torch.get_default_dtype(), f"Output dtype {result.dtype} does not match expected {torch.get_default_dtype()}"
            assert torch.all(~torch.isnan(result)), "Output contains NaN values"

        if self.invariant_norm == "rayleigh":
            result_mag = torch.norm(result, dim=1, keepdim=True)
            sigma_hat = torch.sqrt(((result_mag*result_mag).mean(dim=1, keepdim=True)) / 2)
            mean_rayleigh = sigma_hat * torch.sqrt(torch.tensor(torch.pi / 2, device=result.device))
            std_rayleigh = sigma_hat * torch.sqrt(torch.tensor((4 - torch.pi) / 2, device=result.device))
            result = (result - mean_rayleigh) / (std_rayleigh + self.eps)
        elif self.invariant_norm == "gauss":
            # Real, Imag independently normalized
            mean = torch.mean(result, dim=(0, -2, -1), keepdim=True)
            std = torch.std(result, dim=(0, -2, -1), keepdim=True)
            result = (result - mean) / (std + self.eps)
        elif self.invariant_norm == "none":
            pass
        else:
            raise ValueError(f"Unknown invariant normalization type: {self.invariant_norm}. Use 'none', 'rayleigh', or 'gauss'.")
        result = rearrange(result, '(b ch) co n h w -> b (ch co n) h w', ch=self.in_channels)
        #result = self.norm(result)
        features = self.conv1x1(result) # Convert back to real
        return features

class ComplexInvariantConv2DR(torch.nn.Module):
    def __init__(self,
                 kernel_size:int,
                 in_channels: int,
                 input_size: int,
                 out_channels:int,
                 max_order:int = MAX_ORDER, 
                 n_rings: int = N_RINGS,
                 basis_p0:int = 1,
                 basis_q0:int = 0, 
                 middle_masking = True,
                 padding: str = "same", 
                 eps=1e-8):
        super(ComplexInvariantConv2DR, self).__init__()
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.max_order = max_order
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basis_p0 = basis_p0
        self.basis_q0 = basis_q0
        self.eps = eps

        # Asserts 
        assert kernel_size % 2 == 1, "Filter size must be odd"
        assert max_order >= 0, "Max order must be non-negative"
        assert in_channels > 0, "Number of input channels must be positive"
        assert out_channels > 0, "Number of output channels must be positive"
        assert kernel_size >= 5, "Filters must be decent size for complex invariants"

        assert self.basis_p0 >= 0 and self.basis_q0 >= 0, "Basis indices must be non-negative integers"
        assert self.basis_p0 + self.basis_q0 <= max_order, "Basis indices must not exceed the maximum order"
        assert self.basis_p0 - self.basis_q0 == 1, "Basis indices must differ by 1 for the normalization term"

        # Prepare the fixed filters corresponding to the complex monomials
        filters = []
        ind = []
        types = []
        # Add the p0q0 term
        filters.append(get_complex_monomial(self.kernel_size,
                                             self.basis_q0,
                                             self.basis_p0,
                                             dtype=torch.get_default_dtype()))
        ind.append((self.basis_q0, self.basis_p0))
        types.append(self.basis_q0 - self.basis_p0)
        # Add the complex monomials up to the max order
        for p in range(0, self.max_order + 1):
            for q in range(0, min(self.max_order + 1-p, p + 1)):
                filters.append(get_complex_monomial(self.kernel_size, p, q, dtype=torch.get_default_dtype()))
                ind.append((p, q))
                types.append(p - q)

        # NOTE: This part can be shared by all the layers, that can save memory 
        # Polynomials 
        filters = torch.stack(filters)
        M, _, _  = filters.shape
        Ch = self.in_channels


        if middle_masking:
            for idx, type in enumerate(types):
                if type != 0: 
                    filters[idx, kernel_size//2, kernel_size//2] = 0

        self.complex_conv_groups = M
        filters = torch.stack(dim=0, tensors=[filters.real, filters.imag])
        self.register_buffer('filters', filters)
        self.register_buffer('types', torch.tensor(types, dtype=torch.int8))
        # Radial basis functions
        #   per each polynomial - non-learnable normalization term * input channels 
        rings, sigma = escnn_style_rings_sigmas(self.kernel_size, n_rings=n_rings)
        R = len(rings)
        self.coeff = torch.nn.Parameter(torch.ones(Ch,M,R))
        self.register_buffer("rings", torch.tensor(rings, dtype=torch.get_default_dtype()))   # (R,)
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.get_default_dtype()))

        basis = self._build_radial_basis(self.kernel_size, self.rings, self.sigma)
        self.register_buffer("_basis", basis)  # (R, k, k)

        self.padding = padding 
        self.exponents = torch.tensor([p-q for (p,q) in ind], dtype=torch.int64) # Skip the normalization and scaling term
        self.exponents = self.exponents[1:] # Skip the normalization term
        self.exponents = torch.nn.Parameter(self.exponents[None, :,None, None], requires_grad=False) # Broadcasting dimension [B, Moments, H, W]
        self.ind = torch.tensor(ind, dtype=torch.uint16)
        self.num_invariants = self.exponents.shape[1] # Number of invariants and skip the normalization and scaling term
        self.conv1x1 = torch.nn.Conv2d(in_channels=self.num_invariants*self.in_channels*2,
                                       out_channels=self.out_channels,
                                       kernel_size=1, 
                                       dtype=torch.get_default_dtype())

    @staticmethod
    def _radius_grid(k: int, device=None, dtype=None) -> torch.Tensor:
        """Radius r = sqrt(x^2 + y^2) over a k×k grid centered at 0."""
        ys = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
        xs = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.sqrt(yy ** 2 + xx ** 2)  # (k, k)
    
    @classmethod
    def _build_radial_basis(cls, k: int, rings: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        r = cls._radius_grid(k, device=rings.device, dtype=rings.dtype)  # (k, k)
        rings = rings.view(-1, 1, 1)   # (R,1,1)
        sigma = sigma.view(-1, 1, 1)   # (R,1,1)
        return torch.exp(-0.5 * (r.unsqueeze(0) - rings) ** 2 / (sigma ** 2))  # (R,k,k)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complex invariant convolution layer.
        :param x: Input tensor of shape (batch_size, in_channels, height, width)
        :return: Output tensor of shape (batch_size, out_channels, height', width')
        """

        # Check input shape, but only for debug
        if __debug__:
            assert x.dim() == 4, "Input must be a 4D tensor"
            assert x.shape[1] == self.in_channels, f"Input channels {x.shape[1]} do not match expected {self.in_channels}"
            assert x.dtype == torch.get_default_dtype(), f"Input dtype {x.dtype} does not match expected {torch.get_default_dtype()}"
        
        # [b c h w]
        # Act on batch and channels together
        # TODO: Fix the normalization radial part 
        radial_part =  torch.einsum("cmr,rhw->cmhw", self.coeff, self._basis)# [1 Moments InChannels H W]

        # Apply weights on the filters
        filters = self.filters * radial_part[:, None]
        filters = rearrange(filters,'ch co m h w -> (ch co m) 1 h w', co=2, m=self.complex_conv_groups, ch=self.in_channels)
        # Apply the complex invariant convolution
        moments = torch.nn.functional.conv2d(x,
                                             weight=filters,
                                             padding=self.padding, 
                                             groups=self.in_channels
                                             )

        moments = rearrange(moments, 'b (ch co m) h w -> b ch co m h w', 
                            ch=self.in_channels,
                            m=self.complex_conv_groups,
                            co=2)
        if __debug__:
            assert moments.dtype == torch.get_default_dtype(), f"Output dtype {moments.dtype} does not match expected {torch.get_default_dtype()}"
            assert torch.all(~torch.isnan(moments)), "Output contains NaN values"

        normalization_moment = moments[:, :, :, 0]
        moments = moments[:, :, :, 1:]

        # Moivre's Theorem
        # Note: Sqrt of 0 has infty gradient, so we use eps to avoid it
        # Same for angles: atan2(0, 0) is also problematic
        norm_magnitude = torch.norm(normalization_moment, dim=2)  # Calculate the norm of the normalization moment

        # Avoid NaN output and NaN gradients
        norm_angle = SafeAtan2.apply(normalization_moment[:,:, 1:2],
                                     normalization_moment[:,:, 0:1],
                                     self.eps)
        if __debug__:
            assert torch.all(~torch.isnan(norm_angle)), "Normalization angle contains NaN values"
            assert torch.all(~torch.isinf(norm_angle)), "Normalization angle contains Inf values"
            # Magnitudes 
            assert torch.all(~torch.isnan(norm_magnitude)), "Normalization magnitude contains NaN values"
        norm_angle = norm_angle * self.exponents
        normalization_factor_real = norm_magnitude[:, :, None] * torch.cos(norm_angle)
        normalization_factor_imag = norm_magnitude[:, :,None] * torch.sin(norm_angle)

        result = torch.stack(tensors=[
            moments[:,:, 0] * normalization_factor_real - moments[:,:, 1] * normalization_factor_imag,
            moments[:,:, 0] * normalization_factor_imag + moments[:,:, 1] * normalization_factor_real
        ], dim=2)  # Stack along the channel dimension

        if __debug__:
            assert result.dtype == torch.get_default_dtype(), f"Output dtype {result.dtype} does not match expected {torch.get_default_dtype()}"
            assert torch.all(~torch.isnan(result)), "Output contains NaN values"
        result = rearrange(result, 'b ch co m h w -> b (ch co m) h w', ch=self.in_channels,m=self.num_invariants)
        features = self.conv1x1(result) # Convert back to real
        return features
# Create a block 
