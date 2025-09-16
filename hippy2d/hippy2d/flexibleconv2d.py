# Flexible Complex Polynomials basis
import torch 
from einops import rearrange, repeat
import numpy as np

from hippy2d.complex_invariants_2d import get_complex_monomial
from hippy2d.utils import get_default_complex, tukey_2d, get_circular_mask, SafeAtan2

FILTER_SIZE = 15
MAX_ORDER = 4

# Generalized complex power function using De Moivre's theorem
@torch.jit.script
def complex_power_moivre(x: torch.Tensor,
                             exponents: torch.Tensor,
                             safe_magnitude_power: bool =False,
                             eps: float = 1e-8) -> torch.Tensor:
    """
    JIT-compatible complex power using De Moivre's theorem
    Args:
        x: Complex tensor with shape [..., 2] where last dim is [real, imag]
        exponents: Exponent tensor that broadcasts with x[..., 0]
        eps: Small value to avoid numerical issues
    Returns:
        Complex tensor with same shape as x
    """
    # Extract real and imaginary parts
    real_part = x[..., 0:1]  # [..., 1]
    imag_part = x[..., 1:2]  # [..., 1]
    
    # Compute magnitude and angle
    magnitude = torch.norm(x, dim=-1)  # [..., 1]
    angle = SafeAtan2.apply(imag_part, real_part,
                                eps)[..., 0]

    if safe_magnitude_power: 
        new_magnitude = torch.where(magnitude < eps, torch.tensor(eps, dtype=magnitude.dtype, device=magnitude.device), magnitude)
        new_magnitude = torch.pow(new_magnitude, exponents)  
    else: 
        new_magnitude = torch.pow(magnitude, exponents)  

    new_angle = angle * exponents
    # Convert back to rectangular form
    result_real = new_magnitude * torch.cos(new_angle)
    result_imag = new_magnitude * torch.sin(new_angle)
    
    return torch.stack([result_real, result_imag], dim=-1)


class FlexBaseBlock(torch.nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int,
                 input_size:int,
                 filter_size:int = FILTER_SIZE, 
                 max_order:int = MAX_ORDER, 
                 residual:bool = True, 
                 subsampling:bool = True, 
                 channels_masking: str = "tukey",
                 conv_padding: str = "same"):
        super(FlexBaseBlock, self).__init__()

        self.conv = FlexConv2d(input_shape=input_size,
                               in_channels=in_channels,
                              out_channels=out_channels,
                              filter_size=filter_size,
                              max_order=max_order, 
                              padding=conv_padding)

        if conv_padding == "same":
            conv_output_shape = input_size
        else:
            conv_output_shape = input_size + (2 * conv_padding) - filter_size + 1

        self.norm = torch.nn.LayerNorm(normalized_shape=[out_channels, conv_output_shape, conv_output_shape],
                                        eps=1e-5,
                                        elementwise_affine=False)
        self.activation = torch.nn.ELU()
        self.residual = residual
        self.padding = conv_padding
        assert (input_size - conv_output_shape) % 2 == 0, "Input size must be even for valid padding"
        self.identity_pad = (input_size - conv_output_shape) // 2 
        self.input_size = input_size

        if in_channels != out_channels:
            # TODO: Make it depth wise conv
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
        assert channels_masking in ["tukey", "none"], f"Unknown channels_masking: {channels_masking}"
        self.channels_masking = channels_masking
        if channels_masking == "tukey":
            self.features_mask = torch.nn.Parameter(torch.from_numpy(tukey_2d(self.input_size, 0.5)).to(dtype=torch.get_default_dtype()), requires_grad=False)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complex invariant convolution block.
        :param x: Input tensor of shape (batch_size, in_channels, height, width)
        :return: Output tensor of shape (batch_size, out_channels, height', width')
        """

        if self.padding == "same" or self.identity_pad == 0:
            identity = x
        else:
            identity = x[..., 
                         self.identity_pad: -self.identity_pad,
                         self.identity_pad: -self.identity_pad]
        # TODO: Here should be a circular masking for the whole feature map
        x = self.conv(x)
        # TODO: This must be tested properly 
        if self.channels_masking == "tukey":
            x = x * self.features_mask
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


class FlexConv2d(torch.nn.Module):
    def __init__(self, 
                 input_shape,
                 in_channels,
                 out_channels, 
                 max_order=4, 
                 filter_size=15,
                 gcd=True,
                 normalize_magnitude=False, 
                 masking_middles=False,
                 masking_borders=True, 
                 padding="same",
                 ):
        super().__init__()
        symmetric_polynomials = []
        non_symmetric_polynomials = []
        non_symmetric_exponents = []
        for p in range(0, max_order + 1):
            for q in range(0, min(max_order + 1-p, p + 1)):
                if p == q:
                    symmetric_polynomials.append((p, q))
                else:
                    non_symmetric_polynomials.append((p, q))
                    non_symmetric_exponents.append(p - q)
        non_symmetric_exponents = np.array(non_symmetric_exponents)
        exp_a = repeat(non_symmetric_exponents, 'n -> m n', m=len(non_symmetric_exponents))
        exp_b = repeat(non_symmetric_exponents, 'n -> n m', m=len(non_symmetric_exponents))

        filters = []
        types = []
        # First symmetric
        for (p, q) in symmetric_polynomials:
            filters.append(get_complex_monomial(filter_size,
                                                p, q, dtype=torch.get_default_dtype()))
            types.append(p-q)
        # Non-symmetric
        for (p, q) in non_symmetric_polynomials:
            filters.append(get_complex_monomial(filter_size,
                                                p, q, dtype=torch.get_default_dtype()))
            types.append(p-q)
        # Stack filters
        filters = torch.stack(filters, dim=0)[:, None]
        if normalize_magnitude: 
            filters /= filters.abs()

        if gcd:
            coef_gcd = np.gcd(exp_a, exp_b) 
            exp_a //= coef_gcd
            exp_b //= coef_gcd

        if masking_middles:
            for idx, type in enumerate(types):
                if type != 0: 
                    filters[idx, 0, filter_size//2, filter_size//2] = 0
                if type >= 3: 
                    filters[idx, 0, filter_size//2-1: filter_size//2+2, filter_size//2-1: filter_size//2+2] = 0
        if masking_borders:
            tukey_mask =  torch.from_numpy(
                tukey_2d(filter_size, alpha=0.5),
            ).to(dtype=torch.get_default_dtype())
            filters *= tukey_mask[None, None]
        
        # From complex to real
        filters = torch.cat(dim=0, tensors=[filters.real, filters.imag])

        # Parameters 
        self.in_channels = in_channels
        self.padding = padding

        # Fixed part
        self.register_buffer('_conj', torch.tensor([1.0, -1.0], dtype=torch.get_default_dtype()) )
        self.register_buffer('symmetric_polynomials', torch.tensor(symmetric_polynomials))
        self.register_buffer('non_symmetric_polynomials', torch.tensor(non_symmetric_polynomials))
        self.register_buffer('filters', filters, torch.get_default_dtype())
        # Assert none of exponents are zero
        assert np.all(exp_a != 0) and np.all(exp_b != 0), "There should be no zero exponents"

        self.register_buffer('exp_a', torch.tensor(exp_a, dtype=torch.get_default_dtype()))
        self.register_buffer('exp_b', torch.tensor(exp_b, dtype=torch.get_default_dtype()))
        self.register_buffer('types', torch.tensor(types, dtype=torch.uint8))

        # Learnable part
        self.norm = torch.nn.LayerNorm(normalized_shape=[in_channels*(len(symmetric_polynomials) + len(non_symmetric_polynomials)**2*2), input_shape, input_shape], 
                                       bias=False,
                                       elementwise_affine=False)
        self.conv1x1 = torch.nn.Conv2d(in_channels=in_channels*(len(symmetric_polynomials) + len(non_symmetric_polynomials)**2*2), out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # x shape (B, C, H, W)
        B, C, H, W = x.shape
        assert C == self.in_channels, f"Input channels {C} does not match layer in_channels {self.in_channels}" 
        # TODO: Change to depth wise convolution
        x = rearrange(x, 'b c h w -> (b c) 1 h w')
        x = torch.nn.functional.conv2d(x,
                                       self.filters,
                                       padding=self.padding)
        # Unpack complex 
        x = rearrange(x, 'b (co m) h w -> b m h w co', co=2).contiguous()
        symmetric = x[:, :len(self.symmetric_polynomials)]
        # Unpack the non-symmetric complex part
        # Remove the phase, because it should be zero for symmetric
        symmetric = torch.norm(symmetric, dim=-1)
        nonsymmetric = x[:, len(self.symmetric_polynomials):]

        a = complex_power_moivre(nonsymmetric[:, :, None],
                                 self.exp_a[..., None, None],
                                 safe_magnitude_power=False) # Note: There are not zero exponents
        # Make conjugate 
        b = complex_power_moivre(nonsymmetric[:, None, :] * self._conj,
                                 self.exp_b[..., None, None],
                                 safe_magnitude_power=False) # Note: There are not zero exponents
        # Make a complex multiplication between new_a and new_b
        nonsymmetric_real = (a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1])
        nonsymmetric_imag = (a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0])

        # Rearrange the moment x moment axis
        nonsymmetric_real = rearrange(nonsymmetric_real, 'b m1 m2 h w -> b (m1 m2) h w')
        nonsymmetric_imag = rearrange(nonsymmetric_imag, 'b m1 m2 h w -> b (m1 m2) h w')

        # Concatenate all features
        x = torch.cat([symmetric, nonsymmetric_real, nonsymmetric_imag], dim=1)
        x = rearrange(x, '(b cin) cout h w -> b (cin cout) h w', cin=C)
        x = self.norm(x)
        x = self.conv1x1(x)
        return x
   