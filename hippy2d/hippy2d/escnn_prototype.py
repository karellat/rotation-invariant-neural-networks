
from typing import Optional
import torch 
import escnn
from hippy2d.learnable import complex_power_moivre, flusser_basis_orders
from hippy2d.utils import SafeAtan2
from einops import rearrange, repeat
from collections import defaultdict

from escnn import gspaces
from escnn.nn.modules.masking_module import build_mask as escnn_build_mask
import numpy as np
from escnn.group import Representation
from escnn.nn.modules.basismanager import BlocksBasisExpansion
from escnn.nn.modules.conv.rd_convolution import get_grid_coords
from escnn.nn.modules.conv.r2convolution import compute_basis_params

def compute_padding(padding, kernel_size):
    """
    Convert padding='same' or 'valid' (or int) to an integer pad
    for square kernels, dilation=1, stride=1.

    Returns an int usable directly in nn.Conv2d(..., padding=pad).
    """
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == "same":
            # works for odd kernels only
            return (kernel_size - 1) // 2
        elif padding == "valid":
            return 0
        else:
            raise ValueError("padding must be 'same', 'valid', or int")
    elif isinstance(padding, int):
        return padding
    else:
        raise TypeError("padding must be str or int")


def flusser_basis(max_total_degree: int):
    return [
        (p0, q0)
        for p0 in range(max_total_degree + 1)
        for q0 in range(p0 + 1)
    ]


def complex_polynomials(p: int,
                        q: int,
                        size: int = 15,
                        extent: float = 1.0,
                        supersample: int = 1,
                        use_escnn_mask: bool = True,
                        mask_margin: float = 0.0,
                        mask_sigma: float = 2.0) -> torch.Tensor:
    """
    Build normalized complex basis term z^p * conj(z)^q on a square grid.
    This mirrors the utility used in notebooks/38_fixed_kernels.ipynb.
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    if supersample <= 0 or int(supersample) != supersample:
        raise ValueError("supersample must be a positive integer")
    supersample = int(supersample)

    hi = size * supersample
    ax_hi = np.linspace(-extent, extent, hi)
    x_hi, y_hi = np.meshgrid(ax_hi, ax_hi)

    z_hi = x_hi + 1j * y_hi
    v_hi = (z_hi ** p) * (np.conj(z_hi) ** q)
    if supersample > 1:
        v = v_hi.reshape(size, supersample, size, supersample).mean(axis=(1, 3))
    else:
        v = v_hi

    ax = np.linspace(-extent, extent, size)
    x, y = np.meshgrid(ax, ax)
    if use_escnn_mask:
        if escnn_build_mask is not None:
            mask = escnn_build_mask(
                size,
                dim=2,
                margin=mask_margin,
                sigma=mask_sigma,
                dtype=torch.get_default_dtype(),
            )[0, 0].cpu().numpy()
        else:
            r = np.sqrt(x ** 2 + y ** 2)
            mask = (r <= extent).astype(v.real.dtype)
        v = v * mask

    s = np.max(np.abs(v))
    if s > 0:
        v = v / s
    return torch.from_numpy(v)

def _complex_mul(x, y, complex_dim=3):
    xr = x.select(complex_dim, 0)
    xi = x.select(complex_dim, 1)
    yr = y.select(complex_dim, 0)
    yi = y.select(complex_dim, 1)
    real = xr * yr - xi * yi
    imag = xr * yi + xi * yr
    return torch.stack((real, imag), dim=complex_dim)

def _complex_mul_real(x, y, complex_dim=3):
    return (x.select(complex_dim, 0) * y.select(complex_dim, 0) 
            - 
            x.select(complex_dim, 1) * y.select(complex_dim, 1))

def _complex_mul_real_parts(x_r, x_i, y_r, y_i):
    return x_r * y_r - x_i * y_i


def _complex_mul_real_polar_parts(magnitude_a, angle_a, magnitude_b, angle_b):
    return (magnitude_a * magnitude_b) * torch.cos(angle_a + angle_b)


def _parse_magnitude_func(magnitude_func: str):
    if magnitude_func.lower() == "none":
        return torch.nn.Identity()
    elif magnitude_func.lower() == "sigmoid":
        return torch.sigmoid
    else:
        raise ValueError(f"magnitude_func '{magnitude_func}' not recognized")


def _rotate_moments(moments: torch.Tensor,
                    exponents: torch.Tensor,
                    eps=1e-8,
                    magnitude_func='copy',
                    ) -> torch.Tensor:
    if magnitude_func == 'copy':
        magnitude = torch.norm(moments, dim=-3)
        angle = SafeAtan2.apply(moments[..., 1, :, :], moments[..., 0, :, :], eps)
        new_magnitude = magnitude * torch.ones_like(exponents)
        new_angle = angle * exponents 

        result_real = new_magnitude * torch.cos(new_angle)
        result_imag = new_magnitude * torch.sin(new_angle)
        return torch.stack([result_real, result_imag], dim=-3)
    elif magnitude_func == 'normed':
        norm_moment = moments / torch.clamp(torch.norm(moments, dim=-3, keepdim=True), min=eps)
        new_magnitude = torch.norm(moments, dim=-3) # Keep zeros
        angle = SafeAtan2.apply(norm_moment[..., 1, :, :], norm_moment[..., 0, :, :], eps)
        new_angle = angle * exponents
        result_real = new_magnitude * torch.cos(new_angle)
        result_imag = new_magnitude * torch.sin(new_angle)

        return torch.stack([result_real, result_imag], dim=-3)
    else: 
        raise NotImplementedError(f"magnitude_func '{magnitude_func}' not implemented")

class MomentLayer(torch.nn.Module):
    def __init__(self,
                 max_order: int,
                 orders: list[int],
                 in_channels: int, 
                 groups: Optional[int]=1,
                 padding: str='same',
                 preserve_energy: bool=False,
                 kernel_size: int=15):
        super().__init__()
        # Parameters
        self.orders = orders
        self.max_order = max_order
        self.groups = groups if groups is not None else in_channels
        self.in_size = in_channels // self.groups
        self.padding = compute_padding(padding, kernel_size)
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        basis_filter, self.rings, self.sigma, _ = compute_basis_params(kernel_size) 
        gspace = gspaces.rot2dOnR2(N=-1)      
        self.in_type = escnn.nn.FieldType(gspace, self.in_size*[gspace.trivial_repr])
        irreps_per_input_channel = [gspace.irrep(order) for order in sorted(self.orders)]
        # repeat the list of irreps for each input channel (avoid starred unpack in comprehension)
        out_irreps = irreps_per_input_channel * in_channels
        self.out_type = escnn.nn.FieldType(gspace, out_irreps)

        # Basis generator
        def basis_2d_generator(in_repr: Representation, out_repr: Representation):
            return gspace.build_kernel_basis(in_repr, 
                                             out_repr,
                                             rings=self.rings,
                                             sigma=self.sigma, 
                                             maximum_frequency=max_order)


        self.basisexpansion = BlocksBasisExpansion(self.in_type.representations, 
                                            self.out_type.representations,
                                            basis_generator=basis_2d_generator,
                                            points=get_grid_coords(d=2, kernel_size=kernel_size, dilation=1),
                                            basis_filter=basis_filter)

        # Learnable parameters
        self.weights = torch.nn.Parameter(torch.zeros(self.basisexpansion.dimension()), requires_grad=True)
        escnn.nn.init.generalized_he_init(self.weights.data, self.basisexpansion)

        # Caching filter for inference
        self.register_buffer("filter", self.expand_parameters())

    def expand_parameters(self):
        _filter = self.basisexpansion(self.weights)
        _filter = _filter.reshape(_filter.shape[0], _filter.shape[1], self.kernel_size, self.kernel_size)                      

        return _filter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _filter = self.expand_parameters()
        moments = torch.nn.functional.conv2d(x,
                                         _filter,
                                          bias=None,
                                          stride=1,
                                          groups=self.groups, 
                                          padding=self.padding) 
        moments = rearrange(moments, 'b (ch o) h w -> b ch o h w', ch=self.in_channels)
        # TODO: 
        return moments

    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            _filter = self.expand_parameters()
            self.register_buffer("filter", _filter)

        return super().train(mode)


class FixedFlusserMomentLayer(torch.nn.Module):
    """
    Moment layer with fixed complex polynomial kernels (Flusser basis),
    matching the construction in notebooks/38_fixed_kernels.ipynb.
    """

    def __init__(self,
                 orders: list[int],
                 basis_qp: list[tuple[int, int]],
                 max_order: int,
                 in_channels: int,
                 groups: Optional[int] = 1,
                 padding: str = "same",
                 preserve_energy: bool = True, 
                 kernel_size: int = 11,
                 extent: float = 1.0,
                 supersample: int = 1,):
        super().__init__()
        assert groups == 1, "We are using group convolution, so groups must be None or equal to in_channels"
        self.in_channels = in_channels
        self.group = in_channels
        self.orders = orders
        self.groups = groups if groups is not None else in_channels
        self.in_size = in_channels // self.groups
        self.padding = compute_padding(padding, kernel_size)
        self.kernel_size = kernel_size
        self.preserve_energy = preserve_energy

        if self.in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if any(order < 0 for order in self.orders):
            raise ValueError("FixedFlusserMomentLayer supports non-negative orders only")

        _filters = []
        for order, (p, q) in zip(self.orders, basis_qp):
            if order != p - q:
                raise ValueError(f"Order {order} does not match basis_qp {p}-{q}")
            _filter = complex_polynomials(p, q, 
                                          size=kernel_size,
                                          extent=extent,
                                          supersample=supersample)
            if preserve_energy:
                _filter = _filter / torch.sum(_filter.abs())
            if order == 0:
                # For the trivial representation, we can use a real filter (the imaginary part is zero)
                _filter = _filter.real.unsqueeze(0)  # shape (1, H, W)
            else: 
                # For non-trivial representations, we need to stack the real and imaginary parts to form a complex filter
                _filter = torch.stack([_filter.real, _filter.imag], dim=0)  # shape (2, H, W)
            _filters.append(_filter)

        self.output_orders = orders
        # Count trivials as 1 and non-trivials as 2 (real and imag) for each input channel
        self.moments_per_input = sum(1 if order == 0 else 2 for order in self.output_orders)

        self.register_buffer("filter",
                            repeat(torch.concat(_filters, dim=0), 'c h w -> (i c) 1 h w', i=in_channels)
                            )  # shape (num_moments, 1, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moments = torch.nn.functional.conv2d(
            x,
            self.filter.to(dtype=x.dtype, device=x.device),
            bias=None,
            stride=1,
            groups=self.in_channels,
            padding=self.padding,
        )
        moments = rearrange(moments, 'b (ch o) h w -> b ch o h w', ch=self.in_channels, o=self.moments_per_input)
        return moments

class InvariantsLayer(torch.nn.Module):
    def __init__(self,
                 orders: list[int],
                 in_channels: int,
                 groups: int):
        super().__init__()
        for i in range(len(orders)-1):
            assert orders[i] <= orders[i+1], "Orders must be sorted in increasing order"
        # Parameters
        self.orders = orders
        self.in_channels = in_channels
        self.groups = groups
        self.in_size = in_channels // groups
        self.trivial_idx = np.sum(np.array(self.orders) == 0)
        self.diagonal_idx = np.sum(np.array(self.orders) == 1) - 1
        self.register_buffer("exponents", -torch.tensor(self.orders[self.trivial_idx+1:], dtype=torch.int32)[:, None, None])
        self.out_channels = (self.in_channels * (self.trivial_idx + self.diagonal_idx + 2*(len(self.orders) - self.trivial_idx - self.diagonal_idx - 1)))

    def forward(self, moments: torch.Tensor) -> torch.Tensor:
        trivial = moments[:, :, :self.trivial_idx, :, :]
        non_trivial = moments[:, :, self.trivial_idx:, :, :]
        
        # Non-trivial to complex 
        non_trivial = rearrange(non_trivial, 
                                'b ch (o c) h w -> b ch o c h w',
                                 o=non_trivial.shape[2]//2,
                                 c=2)
        # Norm
        norm = non_trivial[:, :, 0:1] 
        # Rotate the norm with a sigmoid on norm magnitude  
        norm_magnitude = torch.linalg.vector_norm(norm, dim=-3)
        magnitude = torch.sigmoid(norm_magnitude) 
         
        angle = SafeAtan2.apply(norm[..., 1, :, :], norm[..., 0, :, :], 1e-8)
        new_magnitude = magnitude * torch.ones_like(self.exponents)
        new_angle = angle * self.exponents 

        result_real = new_magnitude * torch.cos(new_angle)
        result_imag = new_magnitude * torch.sin(new_angle)
        norm = torch.stack([result_real, result_imag], dim=-3)

        non_trivial = _complex_mul(non_trivial[:, :, 1:], norm) 
        diagonal = non_trivial[:, :, :self.diagonal_idx, 0]
        non_trivial = rearrange(non_trivial[:, :, self.diagonal_idx:],
                                'b ch o c h w -> b ch (o c) h w')
        invariants = rearrange(torch.cat([trivial, diagonal, non_trivial], dim=2),
                               'b ch o h w -> b (ch o) h w') 
        return invariants

class InvariantLayerMagReal(torch.nn.Module):
    def __init__(self,
                 orders: list[int],
                 in_channels: int,
                 groups: int,
                 norm_mag_func: str):
        super().__init__()
        for i in range(len(orders)-1):
            assert orders[i] <= orders[i+1], "Orders must be sorted in increasing order"
        # Parameters
        self.orders = orders
        self.in_channels = in_channels
        self.groups = groups
        self.in_size = in_channels // groups
        self.trivial_idx = np.sum(np.array(self.orders) == 0)
        self.non_trivials_count = len(self.orders) - self.trivial_idx
        self.register_buffer("exponents", -torch.tensor(self.orders[self.trivial_idx+1:], dtype=torch.int32)[:, None, None])
        # Calculate output channels for this layer
        self.magnitude_func = _parse_magnitude_func(norm_mag_func)
        self.out_channels = (self.in_channels 
                             * 
                             (self.trivial_idx + 1
                              + 2*(len(self.orders) - self.trivial_idx - 1)))

    def forward(self, moments: torch.Tensor) -> torch.Tensor:
        trivial = moments[:, :, :self.trivial_idx, :, :]
        non_trivial = moments[:, :, self.trivial_idx:, :, :]
        
        # Non-trivial to complex 
        non_trivial = rearrange(non_trivial, 
                                'b ch (o c) h w -> b ch o c h w',
                                 o=non_trivial.shape[2]//2,
                                 c=2)
        # Calculate the magnitude for all moments
        all_magnitudes = torch.linalg.vector_norm(non_trivial, dim=-3)
        # Norm
        norm = non_trivial[:, :, 0:1] 
        norm_magnitude = all_magnitudes[:, :, 0:1]
        
        magnitude = self.magnitude_func(norm_magnitude)
        angle = SafeAtan2.apply(norm[..., 1, :, :], norm[..., 0, :, :], 1e-8)
        new_angle = angle * self.exponents 

        result_real = magnitude * torch.cos(new_angle)
        result_imag = magnitude * torch.sin(new_angle)

        non_trivial = _complex_mul_real_parts(
            non_trivial[:, :, 1:, 0],
            non_trivial[:, :, 1:, 1],
            result_real,
            result_imag,
        )
        
        invariants = rearrange(torch.cat([trivial, all_magnitudes, non_trivial], dim=2),
                               'b ch o h w -> b (ch o) h w') 
        return invariants
    
    def parse_output(self, invariants: torch.Tensor) -> dict[int, torch.Tensor]:
        # This is a utility function to parse the output of the layer back into a dictionary of moments by order.
        invariants = rearrange(invariants, 'b (ch o) h w -> b ch o h w', ch=self.in_channels)
        trivials, non_trivials = invariants[:, :, :self.trivial_idx, :, :], invariants[:, :, self.trivial_idx:, :, :]
        all_magnitudes, real_part = non_trivials[:, :, :self.non_trivials_count, :, :], non_trivials[:, :, self.non_trivials_count:, :, :]
        return dict(trivials=trivials, magnitudes=all_magnitudes, real=real_part)

class InvariantLayerMagNormReal(torch.nn.Module):
    def __init__(self,
                 orders: list[int],
                 in_channels: int,
                 groups: int,
                 norm_mag_func: str=torch.sigmoid,
                 ):
        super().__init__()
        for i in range(len(orders)-1):
            assert orders[i] <= orders[i+1], "Orders must be sorted in increasing order"
        # Parameters
        self.orders = orders
        self.in_channels = in_channels
        self.groups = groups
        self.in_size = in_channels // groups
        self.trivial_idx = np.sum(np.array(self.orders) == 0)
        self.register_buffer("exponents", -torch.tensor(self.orders[self.trivial_idx+1:], dtype=torch.int32)[:, None, None])
        # Calculate output channels for this layer
        self.out_channels = (self.in_channels 
                             * 
                             (self.trivial_idx + 1
                              + 2*(len(self.orders) - self.trivial_idx - 1)))
        self.magnitude_func = norm_mag_func

    def forward(self, moments: torch.Tensor) -> torch.Tensor:
        trivial = moments[:, :, :self.trivial_idx, :, :]
        non_trivial = moments[:, :, self.trivial_idx:, :, :]
        
        # Non-trivial to complex 
        non_trivial = rearrange(non_trivial, 
                                'b ch (o c) h w -> b ch o c h w',
                                 o=non_trivial.shape[2]//2,
                                 c=2)
        # Calculate the magnitude for all moments
        all_magnitudes = torch.linalg.vector_norm(non_trivial, dim=-3)
        # Norm
        norm = non_trivial[:, :, 0:1] 
        norm_magnitude = all_magnitudes[:, :, 0:1]
        
        magnitude = self.magnitude_func(norm_magnitude)
        angle = SafeAtan2.apply(norm[..., 1, :, :], norm[..., 0, :, :], 1e-8)
        new_angle = angle * self.exponents 

        result_real = magnitude * torch.cos(new_angle)
        result_imag = magnitude * torch.sin(new_angle)

        non_trivial = _complex_mul_real_parts(
            non_trivial[:, :, 1:, 0],
            non_trivial[:, :, 1:, 1],
            result_real,
            result_imag,
        )

        # Normalize the non-trivial invariants by the magnitudes 
        non_trivial = non_trivial / torch.clamp(all_magnitudes[:, :, 1:] * magnitude, min=1e-8)
        
        invariants = rearrange(torch.cat([trivial, all_magnitudes, non_trivial], dim=2),
                               'b ch o h w -> b (ch o) h w') 
        return invariants

class InvariantLayerMag(torch.nn.Module):
    def __init__(self,
                 orders: list[int],
                 in_channels: int,
                 groups: int):
        super().__init__()
        for i in range(len(orders)-1):
            assert orders[i] <= orders[i+1], "Orders must be sorted in increasing order"
        self.orders = orders
        self.in_channels = in_channels
        self.groups = groups
        self.in_size = in_channels // groups
        self.trivial_idx = np.sum(np.array(self.orders) == 0)
        self.non_trivial_count = len(self.orders) - self.trivial_idx
        self.out_channels = self.in_channels * (self.trivial_idx + self.non_trivial_count)

    def forward(self, moments: torch.Tensor) -> torch.Tensor:
        trivial = moments[:, :, :self.trivial_idx, :, :]
        non_trivial = moments[:, :, self.trivial_idx:, :, :]

        non_trivial = rearrange(non_trivial,
                                'b ch (o c) h w -> b ch o c h w',
                                o=non_trivial.shape[2]//2,
                                c=2)
        magnitudes = torch.linalg.vector_norm(non_trivial, dim=-3)

        invariants = rearrange(torch.cat([trivial, magnitudes], dim=2),
                               'b ch o h w -> b (ch o) h w')
        return invariants
# Refactored
class FlexibleInvariantLayer(torch.nn.Module):
    def __init__(self,
                 orders: list[int],
                 in_channels: int,
                 groups: int, 
                 magnitude_func: str='none',
                 max_b_exponent: Optional[int]=4):
        super().__init__()
        # Parameters
        self.orders = orders
        self.in_channels = in_channels
        self.groups = groups
        self.max_b_exponent = max_b_exponent
        self.in_size = in_channels // groups
        self.trivial_idx = np.sum(np.array(self.orders) == 0)
        # Magnitude func 
        if magnitude_func.lower() == "none":
            self.magnitude_func = torch.nn.Identity()
        elif magnitude_func.lower() == "sigmoid":
            self.magnitude_func = torch.sigmoid
        else:
            raise ValueError(f"magnitude_func '{magnitude_func}' not recognized")

        nontrivial_count = len(self.orders) - self.trivial_idx
        nontrivial_orders = torch.tensor(self.orders[self.trivial_idx:], dtype=torch.int32)
        exp_a = torch.zeros(nontrivial_count, nontrivial_count, dtype=torch.int32)
        exp_b = torch.zeros(nontrivial_count, nontrivial_count, dtype=torch.int32)
        # Select exponents 
        for idx_a, a in enumerate(self.orders[self.trivial_idx:]):
            for idx_b, b in enumerate(self.orders[self.trivial_idx:]):
                gcd = np.gcd(a, b)
                exp_a[idx_a, idx_b] = b // gcd
                exp_b[idx_a, idx_b] = -a // gcd
        # Many things are symmetric within the flexible basis. 
        tril_rows, tril_cols = torch.tril_indices(nontrivial_count, nontrivial_count, offset=-1)
        # Filter out higher orders 
        if self.max_b_exponent is not None:
            # Filter by original order on row-indexed `a` (before gcd reduction).
            keep = nontrivial_orders[tril_cols] <= self.max_b_exponent
            tril_rows = tril_rows[keep]
            tril_cols = tril_cols[keep]
        # Select exponents 
        exp_a = exp_a[tril_rows, tril_cols]
        exp_b = exp_b[tril_rows, tril_cols]
        self.register_buffer("exp_a", exp_a)
        self.register_buffer("exp_b", exp_b)
        # tril_rows and tril_cols can be used to index into the nontrivial moments
        self.register_buffer("tril_rows", tril_rows)
        self.register_buffer("tril_cols", tril_cols)
        self.register_buffer("nontrivial_orders", nontrivial_orders)
        self.num_invariants = (len(orders) + len(exp_a))
        self.out_channels = self.in_channels * self.num_invariants

    def forward(self, moments: torch.Tensor) -> torch.Tensor:
        trivials = moments[:, :, :self.trivial_idx, :, :]
        non_trivials = moments[:, :, self.trivial_idx:, :, :]
        
        # Non-trivial to complex 
        non_trivials = rearrange(non_trivials, 
                                'b ch (o c) h w -> b ch o c h w',
                                 o=non_trivials.shape[2]//2,
                                 c=2)
        # Calculate norm & all the angles
        magnitudes = torch.linalg.vector_norm(non_trivials, dim=-3)
        if self.max_b_exponent == 0:
            stacked_invariants = torch.cat([trivials, magnitudes], dim=2)
        else:
            angles = SafeAtan2.apply(non_trivials[..., 1, :, :], 
                                    non_trivials[..., 0, :, :], 1e-8)
            magnitudes = self.magnitude_func(magnitudes)
            mag_a  = torch.index_select(magnitudes, 2, self.tril_rows)
            mag_b  = torch.index_select(magnitudes, 2, self.tril_cols) 
            angle_a = torch.index_select(angles, 2, self.tril_rows) * self.exp_a[..., None, None]
            angle_b = torch.index_select(angles, 2, self.tril_cols) * self.exp_b[..., None, None]
            nontrivial_real = mag_a * mag_b * torch.cos(angle_a + angle_b)
            stacked_invariants = torch.cat([trivials, magnitudes, nontrivial_real], dim=2)
        return rearrange(stacked_invariants, 'b ch o h w -> b (ch o) h w') 

# Testing different phase functions 
class InvariantFuncMagReal(torch.nn.Module):
    def _parse_magnitude_func(self, magnitude_func: str):
        if magnitude_func.lower() == "prod":
            return lambda mag_a, mag_b: mag_a * mag_b
        elif magnitude_func.lower() == "sqrt_prod":
            # Avoiding nans 
            return lambda mag_a, mag_b: torch.sqrt(torch.clamp(mag_a * mag_b, min=1e-8))
        elif magnitude_func.lower() == "nick":
            return lambda mag_a, mag_b: (mag_a * mag_b) / (mag_a * mag_b + 1)
        else: 
            raise ValueError(f"magnitude_func '{magnitude_func}' not recognized")
    def __init__(self,
                 orders: list[int],
                 in_channels: int,
                 groups: int,
                 norm_mag_func: str,
                 spatial_size: int,
                 magnitude_only: bool=False,
                 norm_per_inv_type: str = 'prod'):
        super().__init__()
        for i in range(len(orders)-1):
            assert orders[i] <= orders[i+1], "Orders must be sorted in increasing order"
        # Parameters
        self.orders = orders
        self.in_channels = in_channels
        self.groups = groups
        self.in_size = in_channels // groups
        self.trivial_idx = np.sum(np.array(self.orders) == 0)
        self.non_trivials_count = len(self.orders) - self.trivial_idx
        self.register_buffer("exponents", -torch.tensor(self.orders[self.trivial_idx+1:], dtype=torch.int32)[:, None, None])
        # Calculate output channels for this layer
        self.magnitude_func = self._parse_magnitude_func(norm_mag_func)
        if norm_per_inv_type.lower() == 'none':
            self.trivial_norm = torch.nn.Identity()
            self.mag_norm = torch.nn.Identity()
            self.real_norm = torch.nn.Identity()
        elif norm_per_inv_type.lower() == 'layer':
            self.trivial_norm = torch.nn.LayerNorm((self.in_channels * self.trivial_idx, spatial_size, spatial_size),
                                                   elementwise_affine=False)
            self.mag_norm = torch.nn.LayerNorm((self.in_channels * self.non_trivials_count, spatial_size, spatial_size),
                                               elementwise_affine=False)
            self.real_norm = torch.nn.LayerNorm((self.in_channels * (len(self.orders) - self.trivial_idx - 1), spatial_size, spatial_size),
                                               elementwise_affine=False)
        elif norm_per_inv_type.lower() == 'batch': 
            self.trivial_norm = torch.nn.BatchNorm2d(self.in_channels * self.trivial_idx, affine=False)
            self.mag_norm = torch.nn.BatchNorm2d(self.in_channels * self.non_trivials_count, affine=False)
            self.real_norm = torch.nn.BatchNorm2d(self.in_channels * (len(self.orders) - self.trivial_idx - 1), affine=False)

        self.magnitude_only = magnitude_only

        if self.magnitude_only:
            self.out_channels = (self.in_channels 
                                * 
                                (self.trivial_idx 
                                + (len(self.orders) - self.trivial_idx)))
        else:
            self.out_channels = (self.in_channels 
                                * 
                                (self.trivial_idx + 1
                                + 2*(len(self.orders) - self.trivial_idx - 1)))

    def forward(self, moments: torch.Tensor) -> torch.Tensor:
        trivial = moments[:, :, :self.trivial_idx, :, :]
        non_trivial = moments[:, :, self.trivial_idx:, :, :]
        
        # Non-trivial to complex 
        non_trivial = rearrange(non_trivial, 
                                'b ch (o c) h w -> b ch o c h w',
                                 o=non_trivial.shape[2]//2,
                                 c=2)
        # Calculate the magnitude for all moments
        all_magnitudes = torch.linalg.vector_norm(non_trivial, dim=-3)
        all_angles = SafeAtan2.apply(non_trivial[..., 1, :, :], non_trivial[..., 0, :, :], 1e-8)
        # Normalizer
        norm_magnitude = all_magnitudes[:, :, 0:1]
        norm_angle = all_angles[:, :, 0:1]
        # Moments
        moments = non_trivial[:, :, 1:]
        moment_magnitudes = all_magnitudes[:, :, 1:]
        moment_angles = all_angles[:, :, 1:]
        # 
        trivial = rearrange(trivial, 'b ch o h w -> b (ch o) h w')
        all_magnitudes = rearrange(all_magnitudes, 'b ch o h w -> b (ch o) h w')
        # Matching spin of moments
        if not self.magnitude_only:
            norm_angle = norm_angle * self.exponents 
            # Calculating real part of the non-trivial invariants with a magnitude function on the normalizer   
            magnitude = self.magnitude_func(norm_magnitude, moment_magnitudes)
            non_trivial = magnitude * torch.cos(moment_angles + norm_angle)
            non_trivial = rearrange(non_trivial, 'b ch o h w -> b (ch o) h w')
            non_trivial = self.real_norm(non_trivial)
        # Trivials
        trivial = self.trivial_norm(trivial)
        all_magnitudes = self.mag_norm(all_magnitudes)
        
        if not self.magnitude_only:
            invariants = torch.cat([trivial, all_magnitudes, non_trivial],
                                dim=1)
        else:
            invariants = torch.cat([trivial, all_magnitudes],
                                dim=1)
        return invariants
    
    def parse_output(self, invariants: torch.Tensor) -> dict[int, torch.Tensor]:
        # This is a utility function to parse the output of the layer back into a dictionary of moments by order.
        invariants = rearrange(invariants, 'b (ch o) h w -> b ch o h w', ch=self.in_channels)
        trivials, non_trivials = invariants[:, :, :self.trivial_idx, :, :], invariants[:, :, self.trivial_idx:, :, :]
        all_magnitudes, real_part = non_trivials[:, :, :self.non_trivials_count, :, :], non_trivials[:, :, self.non_trivials_count:, :, :]
        return dict(trivials=trivials, magnitudes=all_magnitudes, real=real_part)

class _FlexibleInvariantLayer(torch.nn.Module):
    def __init__(self,
                 orders: list[int],
                 in_channels: int,
                 groups: int, 
                 magnitude_func: str='sigmoid',
                 max_b_exponent: Optional[int]=2):
        super().__init__()
        for i in range(len(orders)-1):
            assert orders[i] <= orders[i+1], "Orders must be sorted in increasing order"
        # Parameters
        self.orders = orders
        self.in_channels = in_channels
        self.groups = groups
        self.max_b_exponent = max_b_exponent
        self.in_size = in_channels // groups
        self.trivial_idx = np.sum(np.array(self.orders) == 0)
        # Create a matrix of exponents for all combinations
        # Go thhrough all combination of exponents 
        n_non_trivial = len(self.orders) - self.trivial_idx
        exp_a = torch.zeros(n_non_trivial, n_non_trivial, dtype=torch.int32)
        exp_b = torch.zeros(n_non_trivial, n_non_trivial, dtype=torch.int32)

        non_trivial_orders = torch.tensor(self.orders[self.trivial_idx:], dtype=torch.int32)
        for idx_a, a in enumerate(self.orders[self.trivial_idx:]):
            for idx_b, b in enumerate(self.orders[self.trivial_idx:]):
                gcd = np.gcd(a, b)
                exp_a[idx_a, idx_b] = b // gcd
                exp_b[idx_a, idx_b] = -a // gcd

        tril_rows, tril_cols = torch.tril_indices(n_non_trivial, n_non_trivial, offset=-1)
        if self.max_b_exponent is not None:
            # Filter by original order on row-indexed `a` (before gcd reduction).
            keep = non_trivial_orders[tril_cols] <= self.max_b_exponent
            tril_rows = tril_rows[keep]
            tril_cols = tril_cols[keep]
        self.num_pairs = int(tril_rows.numel())
        self.register_buffer("exponents_a", exp_a)        
        self.register_buffer("exponents_b", exp_b)
        self.register_buffer("tril_rows", tril_rows)
        self.register_buffer("tril_cols", tril_cols)
        self.out_channels = (
            self.in_channels 
            * 
            (self.trivial_idx + n_non_trivial + self.num_pairs))
        # Magnitude func 
        if magnitude_func.lower() == "none":
            self.magnitude_func = torch.nn.Identity()
        elif magnitude_func.lower() == "sigmoid":
            self.magnitude_func = torch.sigmoid
        else:
            raise ValueError(f"magnitude_func '{magnitude_func}' not recognized")

    def forward(self, moments: torch.Tensor) -> torch.Tensor:
        trivial = moments[:, :, :self.trivial_idx, :, :]
        non_trivial = moments[:, :, self.trivial_idx:, :, :]
        
        # Non-trivial to complex 
        non_trivial = rearrange(non_trivial, 
                                'b ch (o c) h w -> b ch o c h w',
                                 o=non_trivial.shape[2]//2,
                                 c=2)
        # Norm
        magnitudes = torch.linalg.vector_norm(non_trivial, dim=-3)
        angles = SafeAtan2.apply(non_trivial[..., 1, :, :], non_trivial[..., 0, :, :], 1e-8)
        if self.num_pairs > 0:
            pair_exp_a = self.exponents_a[self.tril_rows, self.tril_cols].view(1, 1, -1, 1, 1)
            pair_exp_b = self.exponents_b[self.tril_rows, self.tril_cols].view(1, 1, -1, 1, 1)

            selected_angles_a = angles.index_select(2, self.tril_rows)
            selected_angles_b = angles.index_select(2, self.tril_cols)
            weighted_magnitudes_a = self.magnitude_func(magnitudes.index_select(2, self.tril_rows))
            weighted_magnitudes_b = self.magnitude_func(magnitudes.index_select(2, self.tril_cols))
            angles_a = selected_angles_a * pair_exp_a
            angles_b = selected_angles_b * pair_exp_b
            real_lower_triangle = _complex_mul_real_polar_parts(
                weighted_magnitudes_a,
                angles_a,
                weighted_magnitudes_b,
                angles_b,
            )
            stacked_invariants = torch.cat([trivial, magnitudes, real_lower_triangle], dim=2)
        else:
            stacked_invariants = torch.cat([trivial, magnitudes], dim=2)
        invariants = rearrange(stacked_invariants, 'b ch o h w -> b (ch o) h w') 

        return invariants

class LearnableCesa(torch.nn.Module): 
    # Flusser basis but basis learnable as in Cesa Escnn
    def __init__(self,
                  in_channels: int, 
                  out_channels: int, 
                  input_size: int,
                  padding: str='same',
                  max_order: int=4, 
                  groups: int = 1, 
                  kernel_size: int=15,
                  magnitude_func: str='sigmoid'):
        super().__init__()
        self.orders = flusser_basis_orders(max_order) 
        self.out_channels = out_channels
        self.input_channels = in_channels
        self.kernel_size = kernel_size
        # Construct moments
        self.moment_layer = MomentLayer(orders=self.orders,
                                        max_order=max_order,
                                        in_channels=in_channels,
                                        padding=padding,
                                        kernel_size=kernel_size, 
                                        groups=groups)
        # Construct invariants
        self.moment_types = self.moment_layer.out_type
        self.invariants_layer = InvariantsLayer(orders=self.orders,
                                                groups=groups,
                                                in_channels=in_channels)
        

        # TODO: Here should be some kind of normalization of either moments and invariants
        number_of_invariants = self.moment_types.size - 3*self.input_channels # Remove the norm part 
        self.conv1x1 = torch.nn.Conv2d(in_channels=number_of_invariants,
                                       out_channels=out_channels,
                                       kernel_size=1,
                                       bias=False)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moments = self.moment_layer(x)  # b, ch*o, h,
        invariants = self.invariants_layer(moments)
        # Separate trivials 
        out = self.conv1x1(invariants)
        return out
            
class LearnableCesaInvLayer(torch.nn.Module): 
    # Flusser basis but basis learnable as in Cesa Escnn
    def __init__(self,
                  in_channels: int, 
                  out_channels: int, 
                  input_size: int,
                  padding: str='same',
                  max_order: int=4, 
                  groups: int = 1, 
                  kernel_size: int=15,
                  magnitude_func: str='sigmoid'):
        super().__init__()
        self.orders = flusser_basis_orders(max_order) 
        self.out_channels = out_channels
        self.input_channels = in_channels
        self.kernel_size = kernel_size
        # Construct moments
        self.moment_layer = MomentLayer(orders=self.orders,
                                        max_order=max_order,
                                        in_channels=in_channels,
                                        padding=padding,
                                        kernel_size=kernel_size, 
                                        groups=groups)
        # Construct invariants
        self.moment_types = self.moment_layer.out_type
        self.invariants_layer = InvariantsLayer(orders=self.orders,
                                                groups=groups,
                                                in_channels=in_channels)
        
        self.out_channels = self.moment_types.size - 3*self.input_channels # Remove the norm part 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moments = self.moment_layer(x)  # b, ch*o, h,
        invariants = self.invariants_layer(moments)
        # Separate trivials 
        return invariants

class LearnableCesaMagRealLayer(torch.nn.Module): 
    # Flusser basis but basis learnable as in Cesa Escnn
    def __init__(self,
                  in_channels: int, 
                  out_channels: int, 
                  input_size: int,
                  padding: str='same',
                  max_order: int=4, 
                  groups: int = 1, 
                  kernel_size: int=15,
                  magnitude_func: str='sigmoid'):
        super().__init__()
        self.orders = flusser_basis_orders(max_order) 
        self.out_channels = out_channels
        self.input_channels = in_channels
        self.kernel_size = kernel_size
        # Construct moments
        self.moment_layer = MomentLayer(orders=self.orders,
                                        max_order=max_order,
                                        in_channels=in_channels,
                                        padding=padding,
                                        kernel_size=kernel_size, 
                                        groups=groups)
        # Construct invariants
        self.moment_types = self.moment_layer.out_type
        self.invariants_layer = InvariantLayerMagReal(orders=self.orders,
                                                      groups=groups,
                                                      in_channels=in_channels, 
                                                      norm_mag_func=magnitude_func)
        
        self.out_channels = self.invariants_layer.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moments = self.moment_layer(x)  # b, ch*o, h,
        invariants = self.invariants_layer(moments)
        # Separate trivials 
        return invariants

class LearnableCesaMag(torch.nn.Module):
    # Flusser basis but basis learnable as in Cesa Escnn
    def __init__(self,
                  in_channels: int,
                  out_channels: int,
                  input_size: int,
                  padding: str='same',
                  max_order: int=4,
                  groups: int = 1,
                  kernel_size: int=15,
                  magnitude_func: str='sigmoid'):
        super().__init__()
        self.orders = flusser_basis_orders(max_order)
        self.out_channels = out_channels
        self.input_channels = in_channels
        self.kernel_size = kernel_size
        self.moment_layer = MomentLayer(orders=self.orders,
                                        max_order=max_order,
                                        in_channels=in_channels,
                                        padding=padding,
                                        kernel_size=kernel_size,
                                        groups=groups)
        self.moment_types = self.moment_layer.out_type
        self.invariants_layer = InvariantLayerMag(orders=self.orders,
                                                  groups=groups,
                                                  in_channels=in_channels)

        self.out_channels = self.invariants_layer.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moments = self.moment_layer(x)
        invariants = self.invariants_layer(moments)
        return invariants

class LearnableFlexibleLayer(torch.nn.Module):
    # Flusser basis but basis learnable as in Cesa Escnn
    def __init__(self,
                  in_channels: int, 
                  out_channels: int, 
                  input_size: int,
                  padding: str='same',
                  max_order: int=4, 
                  groups: int = 1, 
                  kernel_size: int=15,
                  magnitude_func: str='none',
                  max_b_exponent: Optional[int]=None):
        super().__init__()
        self.orders = flusser_basis_orders(max_order) 
        self.out_channels = out_channels
        self.input_channels = in_channels
        self.kernel_size = kernel_size
        # Construct moments
        self.moment_layer = MomentLayer(orders=self.orders,
                                        max_order=max_order,
                                        in_channels=in_channels,
                                        padding=padding,
                                        kernel_size=kernel_size, 
                                        groups=groups)
        # Construct invariants
        self.moment_types = self.moment_layer.out_type
        self.invariants_layer = FlexibleInvariantLayer(orders=self.orders,
                                                       groups=groups,
                                                       in_channels=in_channels, 
                                                       magnitude_func=magnitude_func,
                                                       max_b_exponent=max_order)
        
        self.out_channels = self.invariants_layer.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moments = self.moment_layer(x)  # b, ch*o, h,
        invariants = self.invariants_layer(moments)
        x# Separate trivials 
        return invariants

class FixedMagRealLayer(torch.nn.Module): 
    # Fixed convolution
    def __init__(self,
                  in_channels: int, 
                  out_channels: int, 
                  input_size: int,
                  padding: str='same',
                  max_order: int=4, 
                  groups: int = 1, 
                  kernel_size: int=15,
                  magnitude_func: str='sigmoid'):
        super().__init__()
        self.basis_qp = flusser_basis(max_total_degree=max_order)
        self.basis_qp = sorted(self.basis_qp, key=lambda pq: pq[0] - pq[1])
        self.orders = torch.tensor([p-q for p, q in self.basis_qp]) 
        self.out_channels = out_channels
        self.input_channels = in_channels
        self.kernel_size = kernel_size
        # Construct moments
        self.moment_layer = FixedFlusserMomentLayer(orders=self.orders,
                                                    basis_qp=self.basis_qp,
                                                    in_channels=in_channels,
                                                    padding=padding,
                                                    kernel_size=kernel_size,
                                                    max_order=max_order,
                                                    groups=groups)
        
        # Construct invariants
        self.invariants_layer = InvariantLayerMagReal(orders=self.orders,
                                                      groups=groups,
                                                      in_channels=in_channels)
        
        self.out_channels = self.invariants_layer.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moments = self.moment_layer(x)  # b, ch*o, h,
        invariants = self.invariants_layer(moments)
        # Separate trivials 
        return invariants

class FixedFlexibleLayer(torch.nn.Module):
    def __init__(self,
                  in_channels: int, 
                  out_channels: int, 
                  input_size: int,
                  padding: str='same',
                  max_order: int=4, 
                  groups: int = 1, 
                  kernel_size: int=15,
                  magnitude_func: str='sigmoid',
                  max_b_exponent: Optional[int]=None):
        super().__init__()
        self.basis_qp = flusser_basis(max_total_degree=max_order)
        self.basis_qp = sorted(self.basis_qp, key=lambda pq: pq[0] - pq[1])
        self.orders = torch.tensor([p-q for p, q in self.basis_qp]) 
        self.out_channels = out_channels
        self.input_channels = in_channels
        self.kernel_size = kernel_size
        # Construct moments
        self.moment_layer = FixedFlusserMomentLayer(orders=self.orders,
                                                    basis_qp=self.basis_qp,
                                                    in_channels=in_channels,
                                                    padding=padding,
                                                    kernel_size=kernel_size,
                                                    max_order=max_order,
                                                    groups=groups)
        
        # Construct invariants
        self.invariants_layer = FlexibleInvariantLayer(orders=self.orders,
                                                      groups=groups,
                                                      in_channels=in_channels,
                                                      max_b_exponent=max_b_exponent)
        
        self.out_channels = self.invariants_layer.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moments = self.moment_layer(x)  # b, ch*o, h,
        invariants = self.invariants_layer(moments)
        # Separate trivials 
        return invariants

class LearnableCesaMagNormRealLayer(torch.nn.Module): 
    # Flusser basis but basis learnable as in Cesa Escnn
    def __init__(self,
                  in_channels: int, 
                  out_channels: int, 
                  input_size: int,
                  padding: str='same',
                  max_order: int=4, 
                  groups: int = 1, 
                  kernel_size: int=15,
                  magnitude_func: str='sigmoid'):
        super().__init__()
        self.orders = flusser_basis_orders(max_order) 
        self.out_channels = out_channels
        self.input_channels = in_channels
        self.kernel_size = kernel_size
        # Construct moments
        self.moment_layer = MomentLayer(orders=self.orders,
                                        max_order=max_order,
                                        in_channels=in_channels,
                                        padding=padding,
                                        kernel_size=kernel_size, 
                                        groups=groups)
        # Construct invariants
        self.moment_types = self.moment_layer.out_type
        self.invariants_layer = InvariantLayerMagNormReal(orders=self.orders,
                                                      groups=groups,
                                                      in_channels=in_channels)
        
        self.out_channels = self.invariants_layer.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moments = self.moment_layer(x)  # b, ch*o, h,
        invariants = self.invariants_layer(moments)
        # Separate trivials 
        return invariants


class LearnableCesaMagRealVarFunc(torch.nn.Module): 
    # Flusser basis but basis learnable as in Cesa Escnn
    def __init__(self,
                  in_channels: int, 
                  out_channels: int, 
                  input_size: int,
                  padding: str='same',
                  max_order: int=4, 
                  groups: int = 1, 
                  mag_func: str='sqrt_prod',
                  magnitude_only: bool=False,
                  norm_per_inv_type: str='batch',
                  kernel_size: int=15):
        super().__init__()
        self.orders = flusser_basis_orders(max_order) 
        self.out_channels = out_channels
        self.input_channels = in_channels
        self.kernel_size = kernel_size
        # Construct moments
        self.moment_layer = MomentLayer(orders=self.orders,
                                        max_order=max_order,
                                        in_channels=in_channels,
                                        padding=padding,
                                        kernel_size=kernel_size, 
                                        groups=groups)
        # Construct invariants
        self.moment_types = self.moment_layer.out_type
        self.invariants_layer = InvariantFuncMagReal(orders=self.orders,
                                                      groups=groups,
                                                      in_channels=in_channels, 
                                                      spatial_size=input_size,
                                                      norm_mag_func=mag_func,
                                                      norm_per_inv_type=norm_per_inv_type,
                                                      magnitude_only=magnitude_only)
        
        self.out_channels = self.invariants_layer.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moments = self.moment_layer(x)  # b, ch*o, h,
        invariants = self.invariants_layer(moments)
        # Separate trivials 
        return invariants