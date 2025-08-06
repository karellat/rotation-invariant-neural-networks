import torch 
from einops import rearrange
from loguru import logger

from hippy2d.complex_invariants_2d import get_complex_monomial
from hippy2d.utils import tukey_2d, get_default_complex, get_circular_mask

BASIS_P0 = 1
BASIS_Q0 = 0
FILTER_SIZE = 15
MAX_ORDER = 4

import torch

class SafeAtan2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x, eps=1e-12):
        ctx.save_for_backward(y, x)
        ctx.eps = eps
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, grad_output):
        y, x = ctx.saved_tensors
        denom = x*x + y*y + ctx.eps
        # d/dx = -y/(x^2+y^2+eps)
        # d/dy =  x/(x^2+y^2+eps)
        grad_x = -y / denom * grad_output
        grad_y =  x / denom * grad_output
        return grad_y, grad_x, None



class ComplexInvariantConv2D(torch.nn.Module):
    def __init__(self,
                 filter_size:int,
                 max_order:int, 
                 in_channels: int,
                 out_channels:int,
                 basis_p0:int = 1,
                 basis_q0:int = 0, 
                 circular_padding:str ="tukey",
                 conv_padding: str = "same", 
                 zero_order_scaling: bool = False,
                 eps=1e-8):
        super(ComplexInvariantConv2D, self).__init__()
        self.filter_size = filter_size
        self.max_order = max_order
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basis_p0 = basis_p0
        self.basis_q0 = basis_q0
        self.eps = eps
        self.zero_order_scaling = zero_order_scaling

        # Asserts 
        assert filter_size % 2 == 1, "Filter size must be odd"
        assert max_order >= 0, "Max order must be non-negative"
        assert in_channels > 0, "Number of input channels must be positive"
        assert out_channels > 0, "Number of output channels must be positive"
        assert filter_size >= 5, "Filters must be decent size for complex invariants"

        assert self.basis_p0 >= 0 and self.basis_q0 >= 0, "Basis indices must be non-negative integers"
        assert self.basis_p0 + self.basis_q0 <= max_order, "Basis indices must not exceed the maximum order"
        assert self.basis_p0 - self.basis_q0 == 1, "Basis indices must differ by 1 for the normalization term"

        # Prepare the fixed filters corresponding to the complex monomials
        filters = []
        ind = []
        # Add scaling term 
        if self.zero_order_scaling:
            filters.append(torch.complex(real=torch.ones(filter_size, filter_size),
                                         imag=torch.zeros(filter_size, filter_size)))
            ind.append((0, 0))  # Zero order term
        # Add the p0q0 term
        filters.append(get_complex_monomial(self.filter_size,
                                             self.basis_q0,
                                             self.basis_p0,
                                             dtype=torch.get_default_dtype()))
        ind.append((self.basis_q0, self.basis_p0))
        # Add the complex monomials up to the max order
        for p in range(0, self.max_order + 1):
            for q in range(0, min(self.max_order + 1-p, p + 1)):
                filters.append(get_complex_monomial(self.filter_size, p, q, dtype=torch.get_default_dtype()))
                ind.append((p, q))

        # NOTE: This part can be shared by all the layers, that can save memory 
        filters = rearrange(filters, 'n h w -> n 1 h w')
        Ch, _, _, _ = filters.shape
        self.complex_conv_groups = Ch
        filters = torch.cat(dim=0, tensors=[filters.real, filters.imag])
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
        self.padding = conv_padding 
        self.register_buffer('filters', filters)
        self.exponents = torch.tensor([p-q for (p,q) in ind], dtype=torch.int64) # Skip the normalization and scaling term
        if self.zero_order_scaling:
            self.exponents = self.exponents[2:] # Skip the zero order term and normalization
        else:
            self.exponents = self.exponents[1:] # Skip the normalization term
        self.exponents = torch.nn.Parameter(self.exponents[None, :, None, None], requires_grad=False) # Broadcasting dimension
        self.ind = torch.tensor(ind, dtype=torch.uint16)
        self.num_invariants = self.exponents.shape[1] # Number of invariants and skip the normalization and scaling term
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

        if self.zero_order_scaling:
            zero_order_moment = moments[:, 0:1, 0:1] # Take out the imaginary part, because it's zero anyway
            moments = moments[:, :, 1:] / torch.clamp(zero_order_moment, min=self.eps)  # Remove the zero order moment and safe normalize by zero order
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
        norm_angle = norm_angle * self.exponents
        normalization_factor_real = norm_magnitude[:, None] * torch.cos(norm_angle)
        normalization_factor_imag = norm_magnitude[:, None] * torch.sin(norm_angle)

        result = torch.cat(tensors=[
            moments[:, 0] * normalization_factor_real - moments[:, 1] * normalization_factor_imag,
            moments[:, 0] * normalization_factor_imag + moments[:, 1] * normalization_factor_real
        ], dim=1)  # Stack along the channel dimension
        if __debug__:
            assert result.dtype == torch.get_default_dtype(), f"Output dtype {result.dtype} does not match expected {torch.get_default_dtype()}"
            assert torch.all(~torch.isnan(result)), "Output contains NaN values"
        result = rearrange(result, '(b ch) n h w -> b (ch n) h w', ch=self.in_channels)
        features = self.conv1x1(result) # Convert back to real
        return features

# Create a block 
class ComplexBaseBlock(torch.nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int,
                 input_size:int,
                 basis_p0:int = BASIS_P0,
                 basis_q0:int = BASIS_Q0,
                 filter_size:int = FILTER_SIZE, 
                 max_order:int = MAX_ORDER, 
                 residual:bool = True, 
                 subsampling:bool = True, 
                 zero_order_scaling:bool = False,
                 conv_padding: str = "same"): 
        super(ComplexBaseBlock, self).__init__()
        self.conv = ComplexInvariantConv2D(filter_size=filter_size,
                                            max_order=max_order,
                                            in_channels=in_channels,
                                            out_channels=out_channels,
                                            zero_order_scaling=zero_order_scaling,
                                            basis_p0=basis_p0,
                                            basis_q0=basis_q0)
        self.norm = torch.nn.LayerNorm(normalized_shape=(out_channels, input_size, input_size),
                                       elementwise_affine=False,
                                       dtype=torch.get_default_dtype())
        self.activation = torch.nn.ReLU()
        self.residual = residual
        self.valid_padding = (self.conv.filter_size - 1) // 2
        self.padding = conv_padding
        self.input_size = input_size

        if in_channels != out_channels:
            self.residual_conv = torch.nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=1,
                                                 bias=False)
        else: 
            self.residual_conv = torch.nn.Identity()
        
        if subsampling: 
            self.subsampling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.subsampling = torch.nn.Identity()
        # Note: This can be done by torch.masked.MaskedTensor, but it is not supported for complex
        # it's possible to rewrite the whole block using own complex convolution implementation
        self.features_mask = torch.nn.Parameter(torch.from_numpy(tukey_2d(self.input_size, 0.5)).to(dtype=torch.get_default_dtype()), requires_grad=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complex invariant convolution block.
        :param x: Input tensor of shape (batch_size, in_channels, height, width)
        :return: Output tensor of shape (batch_size, out_channels, height', width')
        """

        if self.padding == "valid":
            # Use valid padding
            identity = x[..., 
                         self.valid_padding:-self.valid_padding,
                         self.valid_padding:-self.valid_padding]
        else:
            # Use same padding
            identity = x
        # TODO: Here should be a circular masking for the whole feature map
        x = x * self.features_mask
        x = self.conv(x)
        # Here we can use the Masked_tensor  instead zero masking
        # Apply batch normalization and activation
        x = self.norm(x)
        x = self.activation(x)
        x = self.subsampling(x)
        if self.residual:
            identity = self.subsampling(identity)
            # Add the residual connection
            x = x + self.residual_conv(identity)
        return x
