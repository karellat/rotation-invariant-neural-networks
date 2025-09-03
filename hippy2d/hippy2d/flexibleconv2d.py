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
                 input_shape, 
                 filter_size=FILTER_SIZE,
                 max_order=MAX_ORDER,
                 padding="same", 
                 circular_padding="tukey"
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

        _filters = _filters * mask
        self.register_buffer("filters", _filters)
        # Learnable layer 
        self.norm = torch.nn.LayerNorm(normalized_shape=[self.invariant_cnt*2*self.in_channels, input_shape, input_shape],
                                       elementwise_affine=False)

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
        x = self.norm(x)
        x = self.conv1x1(x)

        return x