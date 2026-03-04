
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
        self.register_buffer("exponents", -torch.tensor(self.orders[self.trivial_idx+1:], dtype=torch.int32)[:, None, None])
        # Calculate output channels for this layer
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
        
        magnitude = torch.sigmoid(norm_magnitude) 
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

class InvariantLayerMixedMagReal(torch.nn.Module):
    def __init__(self,
                 orders: list[int],
                 in_channels: int,
                 groups: int,
                 magnitude_func: str='sigmoid'):
        super().__init__()
        for i in range(len(orders)-1):
            assert orders[i] <= orders[i+1], "Orders must be sorted in increasing order"
        # Parameters
        self.orders = orders
        self.in_channels = in_channels
        self.groups = groups
        self.in_size = in_channels // groups
        self.trivial_idx = np.sum(np.array(self.orders) == 0)
        non_trivial_orders = torch.tensor(self.orders[self.trivial_idx:], dtype=torch.get_default_dtype())
        self.register_buffer("non_trivial_orders", non_trivial_orders[:, None, None])
        self.register_buffer("exponents", -torch.tensor(self.orders[self.trivial_idx+1:], dtype=torch.get_default_dtype())[:, None, None])
        self.magnitude_func = magnitude_func.lower()
        if self.magnitude_func not in ["softmax", "sigmoid"]:
            raise ValueError(f"magnitude_func '{magnitude_func}' not recognized")
        # Match InvariantLayerMagReal channel layout:
        # trivials + all magnitudes + real(normalized non-trivials excluding the reference)
        self.out_channels = (self.in_channels
                             *
                             (self.trivial_idx + 1
                              + 2*(len(self.orders) - self.trivial_idx - 1)))

    def forward(self, moments: torch.Tensor) -> torch.Tensor:
        trivial = moments[:, :, :self.trivial_idx, :, :]
        non_trivial = moments[:, :, self.trivial_idx:, :, :]

        # Non-trivial moments as complex tensor.
        non_trivial = rearrange(non_trivial,
                                'b ch (o c) h w -> b ch o c h w',
                                o=non_trivial.shape[2]//2,
                                c=2)
        all_magnitudes = torch.linalg.vector_norm(non_trivial, dim=-3)
        all_angles = SafeAtan2.apply(non_trivial[..., 1, :, :], non_trivial[..., 0, :, :], 1e-8)

        spin1_base = all_angles / self.non_trivial_orders
        # Resolve branch ambiguity by aligning to a robust anchor.
        # Prefer type-1 anchor when it has enough energy; otherwise use an all-types
        # iterative anchor initialized from the strongest available type.
        type1_mask = (self.non_trivial_orders == 1).to(all_angles.dtype)
        type1_weights = all_magnitudes * type1_mask
        type1_wsum = torch.clamp(type1_weights.sum(dim=2, keepdim=True), min=1e-8)
        ref1_cos = (type1_weights * torch.cos(all_angles)).sum(dim=2, keepdim=True) / type1_wsum
        ref1_sin = (type1_weights * torch.sin(all_angles)).sum(dim=2, keepdim=True) / type1_wsum
        ref1_angle = SafeAtan2.apply(ref1_sin, ref1_cos, 1e-8)

        magnitude_weights = torch.sigmoid(all_magnitudes)
        wsum = torch.clamp(magnitude_weights.sum(dim=2, keepdim=True), min=1e-8)
        max_idx = torch.argmax(all_magnitudes, dim=2, keepdim=True)
        ref_init = torch.gather(spin1_base, dim=2, index=max_idx)

        two_pi = 2.0 * torch.pi
        # Iteration 1: align to strong-type init, then average all aligned types.
        k0 = torch.round((ref_init - spin1_base) * self.non_trivial_orders / two_pi)
        spin1_aligned0 = spin1_base + two_pi * k0 / self.non_trivial_orders
        ref0_cos = (magnitude_weights * torch.cos(spin1_aligned0)).sum(dim=2, keepdim=True) / wsum
        ref0_sin = (magnitude_weights * torch.sin(spin1_aligned0)).sum(dim=2, keepdim=True) / wsum
        ref_fallback = SafeAtan2.apply(ref0_sin, ref0_cos, 1e-8)

        type1_energy = type1_weights.sum(dim=2, keepdim=True)
        all_energy = torch.clamp(all_magnitudes.sum(dim=2, keepdim=True), min=1e-8)
        use_type1 = type1_energy > (1e-3 * all_energy)
        ref_angle = torch.where(use_type1, ref1_angle, ref_fallback)

        # Final alignment against chosen robust reference.
        k = torch.round((ref_angle - spin1_base) * self.non_trivial_orders / two_pi)
        spin1_angles = spin1_base + two_pi * k / self.non_trivial_orders

        spin1_cos = (magnitude_weights * torch.cos(spin1_angles)).sum(dim=2, keepdim=True) / wsum
        spin1_sin = (magnitude_weights * torch.sin(spin1_angles)).sum(dim=2, keepdim=True) / wsum
        spin_angles = SafeAtan2.apply(spin1_sin, spin1_cos, 1e-8)
        mixed_magnitude = (magnitude_weights * all_magnitudes).sum(dim=2, keepdim=True) / wsum


        # Spin the mixed reference angle to each target type exactly as in MagReal.
        new_angle = spin_angles * self.exponents
        norm_real = mixed_magnitude * torch.cos(new_angle)
        norm_imag = mixed_magnitude * torch.sin(new_angle)
        mixed_normalizer = torch.stack([norm_real, norm_imag], dim=-3)

        # Keep only the real part of mixed invariants (exclude reference slot).
        mixed_real = _complex_mul_real(non_trivial[:, :, 1:], mixed_normalizer)
        invariants = rearrange(torch.cat([trivial, all_magnitudes, mixed_real], dim=2),
                               'b ch o h w -> b (ch o) h w')
        return invariants

class FlexibleInvariantLayer(torch.nn.Module):
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
        angles_a = angles[:, :, :, None] * self.exponents_a[:, :, None, None]
        angles_b = angles[:, :, None] * self.exponents_b[:, :, None, None]

        real_a = self.magnitude_func(magnitudes[:, :, :, None]) * torch.cos(angles_a)
        imag_a = self.magnitude_func(magnitudes[:, :, :, None]) * torch.sin(angles_a)
        real_b = self.magnitude_func(magnitudes[:, :, None]) * torch.cos(angles_b)
        imag_b = self.magnitude_func(magnitudes[:, :, None]) * torch.sin(angles_b)
        a = torch.stack([real_a, imag_a], dim=-3)
        b = torch.stack([real_b, imag_b], dim=-3)

        all_combinations = _complex_mul_real(a, b, complex_dim=-3)
        real_lower_triangle = all_combinations[:, :, self.tril_rows, self.tril_cols]
        invariants = rearrange(torch.cat([trivial, magnitudes, real_lower_triangle], dim=2),
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
                                                      in_channels=in_channels)
        
        self.out_channels = self.invariants_layer.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moments = self.moment_layer(x)  # b, ch*o, h,
        invariants = self.invariants_layer(moments)
        # Separate trivials 
        return invariants

class LearnableCesaMixedMagRealLayer(torch.nn.Module):
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
        self.invariants_layer = InvariantLayerMixedMagReal(orders=self.orders,
                                                           groups=groups,
                                                           in_channels=in_channels,
                                                           magnitude_func=magnitude_func)

        self.out_channels = self.invariants_layer.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moments = self.moment_layer(x)  # b, ch*o, h,
        invariants = self.invariants_layer(moments)
        # Separate trivials
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
                                                       max_b_exponent=max_b_exponent)
        
        self.out_channels = self.invariants_layer.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moments = self.moment_layer(x)  # b, ch*o, h,
        invariants = self.invariants_layer(moments)
        # Separate trivials 
        return invariants
