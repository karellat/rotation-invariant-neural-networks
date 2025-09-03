# Flexible Complex Polynomials basis
import torch 
from einops import rearrange, repeat
import numpy as np

from hippy2d.complex_invariants_2d import get_complex_monomial
from hippy2d.utils import get_default_complex, tukey_2d

FILTER_SIZE = 15
MAX_ORDER = 3


class FlexInv2D(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels, 
                 filter_size=FILTER_SIZE,
                 max_order=MAX_ORDER,
                 padding="same"
                 ):
        super(FlexInv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.max_order = max_order
        self.padding = padding
        self.symmetric_polynomials = []
        self.non_symmetric_polynomials = []
        self.non_symmetric_exponents = []
        for p in range(0, MAX_ORDER + 1):
            for q in range(0, min(MAX_ORDER + 1-p, p + 1)):
                if p == q:
                    self.symmetric_polynomials.append((p, q))
                else:
                    self.non_symmetric_polynomials.append((p, q))
                    self.non_symmetric_exponents.append(p - q)
        
        self._symmetric_cnt = len(self.symmetric_polynomials)
        self.invariant_cnt = self._symmetric_cnt + len(self.non_symmetric_polynomials) ** 2
        self.non_symmetric_exponents = np.array(self.non_symmetric_exponents)
        coef_a = repeat(self.non_symmetric_exponents, 'n -> m n', m=len(self.non_symmetric_exponents))
        coef_b = -repeat(self.non_symmetric_exponents, 'n -> n m', m=len(self.non_symmetric_exponents))
        # Compute GCD
        coef = np.gcd(coef_a, coef_b)
        coef_a, coef_b = coef_a // coef, coef_b // coef
        coef_a = coef_a[..., None, None]
        coef_b = coef_b[..., None, None]
        self.register_buffer("coef_a", torch.tensor(coef_a))
        self.register_buffer("coef_b", torch.tensor(coef_b))
        # Generate all the polynomials 
        _filters = [] 
        # First symmetric
        for (p, q) in self.symmetric_polynomials:
            _filters.append(get_complex_monomial(self.filter_size,
                                                  p, q, 
                                                  dtype=torch.get_default_dtype()))
        # Non-symmetric
        for (p, q) in self.non_symmetric_polynomials:
            _filters.append(get_complex_monomial(self.filter_size,
                                                  p, q, dtype=torch.get_default_dtype()))
        _filters = torch.stack(_filters, dim=0)[:, None]
        self.register_buffer("filters", _filters)
        # Learnable layer 
        self.batch_norm = torch.nn.BatchNorm2d(num_features=self.invariant_cnt*2*self.in_channels,
                                               affine=False)

        self.conv1x1 = torch.nn.Conv2d(in_channels=self.invariant_cnt*2*self.in_channels,
                                       out_channels=self.out_channels,
                                       kernel_size=1)

    def forward(self, x):
        # Apply the filters to the input
        # TODO: Change padding
        x = rearrange(x, 'b c h w -> (b c) 1 h w', c=self.in_channels)
        x = x.to(dtype=get_default_complex())
        moments = torch.nn.functional.conv2d(x,
                                          self.filters,
                                          padding=self.padding)
        symmetric_invariants = moments[:, :self._symmetric_cnt]
        nonsymmetric_moments = moments[:, self._symmetric_cnt:]

        safe_nonsymmetric_moments = torch.where(nonsymmetric_moments.abs() > 1e-7, nonsymmetric_moments,
                                                                    torch.complex(torch.tensor(1e-7), torch.tensor(0.0)))
        # TODO: Negative coef_b can cause NaNs
        nonsymmetric_invariants = (
            (nonsymmetric_moments[:, :, None] ** self.coef_a)
            *
            (safe_nonsymmetric_moments[:, None, :] ** self.coef_b)
        )
        nonsymmetric_invariants = rearrange(nonsymmetric_invariants, 'c a b h w -> c (a b) h w')
        x = torch.cat(dim=1,
                  tensors=[symmetric_invariants, nonsymmetric_invariants])
        x = torch.view_as_real(x)
        x = rearrange(x, 'b c h w co -> b (c co) h w')
        x = rearrange(x, '(b cin) cout h w -> b (cin cout) h w', cin=self.in_channels)
        # TODO: Test this first in normal setting
        x = self.batch_norm(x)
        x = self.conv1x1(x)

        return x
    
class FlexBaseBlock(torch.nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int,
                 input_size:int,
                 filter_size:int = FILTER_SIZE, 
                 max_order:int = MAX_ORDER, 
                 residual:bool = True, 
                 subsampling:bool = True, 
                 channels_masking: str = "none",
                 conv_padding: str = "same"):
        super(FlexBaseBlock, self).__init__()

        self.conv = FlexInv2D(in_channels=in_channels,
                              out_channels=out_channels,
                              filter_size=filter_size,
                              max_order=max_order, 
                              padding=conv_padding)

        if conv_padding == "same":
            conv_output_shape = input_size
        else:
            conv_output_shape = input_size + (2 * conv_padding) - filter_size + 1
        
        self.norm = torch.nn.LayerNorm(normalized_shape=(out_channels, conv_output_shape, conv_output_shape),
                                       elementwise_affine=False,
                                       dtype=torch.get_default_dtype())
        self.activation = torch.nn.ELU()
        self.residual = residual
        self.padding = conv_padding
        assert (input_size - conv_output_shape) % 2 == 0, "Input size must be even for valid padding"
        self.identity_pad = (input_size - conv_output_shape) // 2 
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
        if self.channels_masking == "tukey":
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