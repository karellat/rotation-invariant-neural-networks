
import torch 
import escnn
from hippy2d.learnable import complex_power_moivre, flusser_basis_orders
from hippy2d.utils import SafeAtan2
from einops import rearrange

from escnn import gspaces
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

def _complex_mul(x, y, complex_dim=3):
    xr, xi = torch.unbind(x, dim=complex_dim)  # real, imag
    yr, yi = torch.unbind(y, dim=complex_dim)
    real = xr * yr - xi * yi
    imag = xr * yi + xi * yr
    return torch.stack((real, imag), dim=complex_dim)


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
                 padding: str='same',
                 kernel_size: int=15):
        super().__init__()
        # Parameters
        self.orders = orders
        self.max_order = max_order
        self.in_size = 1
        self.groups = in_channels
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

class InvariantsLayer(torch.nn.Module):
    def __init__(self,
                 orders: list[int],
                 in_channels: int, 
                 padding: str='same'):
        super().__init__()
        # Parameters
        self.orders = orders
        self.trivial_idx = np.sum(np.array(self.orders) ==0)
        self.register_buffer("exponents", torch.tensor(self.orders[self.trivial_idx+1:], dtype=torch.int32)[:, None, None]) 

    def forward(self, moments: torch.Tensor) -> torch.Tensor:
        trivial = moments[:, :, :self.trivial_idx, :, :]
        non_trivial = moments[:, :, self.trivial_idx:, :, :]
        # Non-trivial to complex 
        non_trivial = rearrange(non_trivial, 'b ch (o c) h w -> b ch o c h w', o=non_trivial.shape[2]//2, c=2)
        norm = non_trivial[:, :, 0:1] 
        norm[..., 1, :, :] *= -1  # Conjugate 
        norm = _rotate_moments(norm, self.exponents, magnitude_func='copy')
        non_trivial = rearrange(_complex_mul(non_trivial[:, :, 1:], norm), 'b ch o c h w -> b ch (o c) h w')
        invariants = rearrange(torch.cat([trivial, non_trivial], dim=2), 'b ch o h w -> b (ch o) h w') 
        return invariants

class LearnableCesa(torch.nn.Module): 
    # Flusser basis but basis learnable as in Cesa Escnn
    def __init__(self,
                  in_channels: int, 
                  out_channels: int, 
                  input_size: int,
                  padding: str='same',
                  max_order: int=4, 
                  kernel_size: int=15,
                  ):
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
                                        kernel_size=kernel_size)
        # Construct invariants
        self.moment_types = self.moment_layer.out_type
        self.invariants_layer = InvariantsLayer(orders=self.orders,
                                            in_channels=in_channels)


        # TODO: Here should be some kind of normalization of either moments and invariants
        number_of_invariants = self.moment_types.size - 2*self.input_channels # Remove the norm part 
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
            

    