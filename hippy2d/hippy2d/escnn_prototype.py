
from typing import Optional
import torch 
import escnn
from hippy2d.learnable import complex_power_moivre, flusser_basis_orders
from hippy2d.utils import SafeAtan2
from einops import rearrange
from collections import defaultdict

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
        # TODO: We test group norm over different invariants
        

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
        # TODO: Add the norm guy magnitude 
        if not self.training:
            all_moments_magnitude = torch.linalg.vector_norm(non_trivial, dim=-3) < 1e-8
            vanished_moments = torch.sum(
            all_moments_magnitude[:, :, 0:1] 
                &
            ~(all_moments_magnitude[:, :, 1:]),
            dim=[0,1,3,4]) 
            total_non_zero = (~all_moments_magnitude).sum(dim=[0,1,3,4])

            print(f"Vanished moments (in ch:{self.in_channels})")
            print(np.array([-self.exponents[:,0,0].cpu().numpy(),
                  vanished_moments.cpu().numpy(),]))
            print("Non-zero moments total:" )
            print((vanished_moments / total_non_zero[1:]).cpu().numpy())
            print(total_non_zero.cpu().numpy())
        # Log the vanished moments 
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