import torch 
from einops import rearrange
from collections.abc import Callable, Sequence

import escnn
from escnn import gspaces
from escnn.group import Representation
from escnn.nn.modules.basismanager import BlocksBasisExpansion
from escnn.nn.modules.conv.rd_convolution import get_grid_coords
from escnn.nn.modules.conv.r2convolution import compute_basis_params
from escnn.nn.modules.masking_module import build_mask as escnn_build_mask

from hippy2d.utils import SafeAtan2

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

def rotate_real_self_n(
    v: torch.Tensor,
    mag: torch.Tensor,
    o: torch.Tensor,
    magnitude_fn: Callable,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute normalized real-valued invariants via complex multiplication.

    Args:
        v: Non-trivial moments shaped ``[B, S, O_nt, 2, H, W]`` where axis ``2``
            stores ``(real, imag)``.
        mag: Magnitudes of ``v`` with shape ``[B, S, O_nt, H, W]``.
        o: Orders for moments excluding the first order-1 normalizer, shape
            ``[O_nt - 1]``.
        magnitude_fn: Callable ``f(mag1, mag2)`` producing a magnitude scaling
            term for normalizer magnitude ``mag1`` and target magnitudes ``mag2``.
        eps: Clamp minimum used to avoid division by zero.

    Returns:
        Tensor of phase-aligned real invariants with shape
        ``[B, S, O_nt - 1, H, W]``.
    """
    real, imag = v.unbind(dim=-3)

    n_mag = mag[:, :, 0:1]
    safe_n_mag = n_mag.clamp(min=eps)
    n_i, n_r = imag[:, :, 0:1] / safe_n_mag, real[:, :, 0:1] / safe_n_mag

    nt_mag = mag[:, :, 1:]
    nt_r, nt_i = real[:, :, 1:], imag[:, :, 1:]
    rot_angle = SafeAtan2.apply(n_i, n_r, eps) * -o[:, None, None]

    n_r = torch.cos(rot_angle)
    n_i = torch.sin(rot_angle)
    scaled_mag = magnitude_fn(mag1=n_mag, mag2=nt_mag) / nt_mag.clamp(min=eps)
    return scaled_mag * (n_r * nt_r - n_i * nt_i)

def _polar_relative_phase(
    v: torch.Tensor,
    mag: torch.Tensor,
    o: torch.Tensor,
    magnitude_fn: Callable,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return magnitude scaling and relative phases for target moments."""
    safe_mag = mag.clamp(min=eps)
    real, imag = v.unbind(dim=-3)
    angle = SafeAtan2.apply(imag / safe_mag, real / safe_mag, eps)

    n_angle = angle[:, :, 0:1] * -o[:, None, None]
    n_mag = mag[:, :, 0:1]
    nt_mag = mag[:, :, 1:]
    nt_angle = angle[:, :, 1:]
    scaled_mag = magnitude_fn(n_mag, nt_mag)

    return scaled_mag, nt_angle + n_angle

def rotate_polar_self_n(
    v: torch.Tensor,
    mag: torch.Tensor,
    o: torch.Tensor,
    magnitude_fn: Callable,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute one cosine projection of each relative invariant phase."""
    scaled_mag, relative_angle = _polar_relative_phase(
        v, mag, o, magnitude_fn, eps
    )

    return scaled_mag * torch.cos(relative_angle)

def rotate_polar_angle(
    v: torch.Tensor,
    mag: torch.Tensor,
    o: torch.Tensor,
    magnitude_fn: Callable,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute normalized real-valued invariants using polar angles.

    Args:
        v: Non-trivial moments shaped ``[B, S, O_nt, 2, H, W]``.
        mag: Magnitudes of ``v`` with shape ``[B, S, O_nt, H, W]``.
        o: Orders for moments excluding the first order-1 normalizer, shape
            ``[O_nt - 1]``.
        magnitude_fn: Callable ``f(mag1, mag2)`` for magnitude scaling.
        eps: Clamp minimum used for angle stabilization.

    Returns:
        Tensor of phase-aligned real invariants with shape
        ``[B, S, O_nt - 1, H, W]``.
    """
    scaled_mag, relative_angle = _polar_relative_phase(
        v, mag, o, magnitude_fn, eps
    )
    wrapped_angle = torch.remainder(relative_angle + torch.pi, 2 * torch.pi) - torch.pi

    return scaled_mag * wrapped_angle

def rotate_polar_components(
    v: torch.Tensor,
    mag: torch.Tensor,
    o: torch.Tensor,
    magnitude_fn: Callable,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Represent each relative phase by interleaved cosine/sine components.

    Returns:
        Tensor shaped ``[B, S, 2 * (O_nt - 1), H, W]`` with components ordered
        as ``[cos(phi_1), sin(phi_1), cos(phi_2), sin(phi_2), ...]``.
    """
    scaled_mag, relative_angle = _polar_relative_phase(
        v, mag, o, magnitude_fn, eps
    )
    components = torch.stack(
        (
            scaled_mag * torch.cos(relative_angle),
            scaled_mag * torch.sin(relative_angle),
        ),
        dim=3,
    )
    return components.flatten(start_dim=2, end_dim=3)

def rotate_complex_self_n(
    v: torch.Tensor,
    o: torch.Tensor,
    n: int = 7,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Raise normalized complex moments to a power (experimental helper).

    Args:
        v: Tensor shaped ``[..., 2, ...]`` where the complex axis is ``-3``.
        o: Unused in this implementation; kept for API compatibility.
        n: Complex exponent.
        eps: Clamp minimum for magnitude normalization.

    Returns:
        Tensor with same layout as ``v`` and complex axis at ``-3``.
    """
    real, imag = v.unbind(dim=-3)
    c = torch.complex(real, imag)
    r = c.abs()
    safe_r = r.clamp(min=eps)

    q = c / safe_r

    out = q.pow(n)
    return torch.stack((out.real, out.imag), dim=-3)

def shared_layer_norm(tuple_trivials_non_trivials, eps=1e-12): 
    """"
    Shift trivials and scale all fields together by shared variance
    trivials - [Batch Channels Orders H W]
    non_trivials - [Batch Channels Orders 2 H W]
    """
    trivials, non_trivials = tuple_trivials_non_trivials
    trivials_mean = trivials.mean(dim=[1, 2, 3, 4], keepdim=True)
    trivials_centered = trivials - trivials_mean
    

    trivial_sigma = torch.sqrt(
        torch.clamp(torch.mean(trivials_centered ** 2,  dim=[1, 2, 3, 4], keepdim=True), 
                    min=eps))[..., None]
    non_trivial_sigma = torch.sqrt(
        torch.clamp(torch.mean(non_trivials ** 2, dim=[1, 2, 3, 4, 5], keepdim=True), 
                    min=eps))
    shared_sigma = torch.sqrt(torch.clamp((trivial_sigma ** 2 + non_trivial_sigma ** 2)/2, min=eps))

    trivials_normed = trivials_centered / shared_sigma[...,0]
    non_trivials_normed = non_trivials / shared_sigma
    return trivials_normed, non_trivials_normed

@torch.compile
def nicks_magnitude(mag1, mag2):
    """Bounded magnitude coupling: ``(mag1 * mag2) / (mag1 * mag2 + 1)``."""
    return (mag1 * mag2) / (mag1 * mag2 + 1)

@torch.compile
def roxanas_magnitude(mag1, mag2):
    """Product magnitude coupling: ``mag1 * mag2``."""
    return mag1 * mag2

_PHASE_FUNCTIONS = {
    "real": rotate_real_self_n,
    "polar": rotate_polar_self_n,
    "angle": rotate_polar_angle,
    "circular": rotate_polar_components,
}

_PHASE_OUTPUT_MULTIPLIERS = {
    "real": 1,
    "polar": 1,
    "angle": 1,
    "circular": 2,
}

_MAGNITUDE_FUNCTIONS = {
    "nicks": nicks_magnitude,
    "roxanas": roxanas_magnitude,
}

class CompiledInvariantLayer(torch.nn.Module):
    """
    Invariant projection with runtime constants prepared in ``__init__``.

    Input moments shape: ``[B, S, O, H, W]`` where
    ``O = trivial_count + 2 * non_trivial_count``.

    The output contains trivial moments, all non-trivial magnitudes, and one or
    more phase features per non-normalizer moment. The ``"circular"`` phase
    mode emits two phase features; all other built-in modes emit one.
    """

    def __init__(
        self,
        orders: Sequence[int] | torch.Tensor,
        phase_function: str | Callable="real",
        magnitude_function: str | Callable="nicks",
        pre_norm_function: str | None = None,
#        after_norm_function: str | None = None,
        compile_functions: bool = True,
        compile_kwargs: dict | None = dict(fullgraph=True),
    ):
        """Initialize the invariant projection layer.

        Args:
            orders: Ordered moment frequencies. Must be 1D and include at least
                one ``0`` (trivial) and one non-trivial order. The first
                non-trivial order must be ``1`` to act as normalizer.
            phase_function: Either a key in
                ``{"real", "polar", "angle", "circular"}`` or a custom
                callable with signature ``fn(v, mag, o, magnitude_fn, eps=...)``.
            magnitude_function: Either a key in ``{"nicks", "roxanas"}`` or a
                custom callable with signature ``fn(mag1, mag2)``.
            pre_norm_function: Optional string specifying the prenormalization method.
            after_norm_function: Optional string specifying the afternormalization method.
            compile_functions: If ``True``, compiles the forward invariant kernel.
            compile_kwargs: Keyword args passed to ``torch.compile``.
        """
        super().__init__()

        if compile_kwargs is None:
            compile_kwargs = dict()

        orders = torch.as_tensor(orders, dtype=torch.int32)
        if orders.ndim != 1:
            raise ValueError("orders must be a 1D sequence")

        if int((orders == 0).sum().item()) == 0:
            raise ValueError("orders must contain at least one trivial order (0)")

        self.register_buffer("orders", orders)

        self.trivial_idx = int((orders == 0).sum().item())
        self.non_trivial_orders_count = int(orders.numel() - self.trivial_idx)

        if self.non_trivial_orders_count < 1:
            raise ValueError("Need at least one non-trivial order for normalization")
        
        if orders[self.trivial_idx] != 1: 
            raise ValueError("The first non-trivial order must be 1 for normalization")

        # Orders for targets only; exclude the first order-1 normalizer moment.
        self.register_buffer("non_trivial_orders", orders[self.trivial_idx + 1 :])

        self.phase_function, self.phase_output_multiplier = self._resolve_phase_function(
            phase_function
        )
        self.magnitude_function = self._resolve_magnitude_function(magnitude_function)
        self.layer_norm_function = self._resolve_norm_function(pre_norm_function)

        if compile_functions:
            self._compiled_forward = torch.compile(self._invariants_common, **compile_kwargs)
        else:
            self._compiled_forward = self._invariants_common
        
        self.out_channels = (
            self.trivial_idx
            + self.non_trivial_orders_count
            + self.phase_output_multiplier * (self.non_trivial_orders_count - 1)
        )

    @staticmethod
    def _resolve_phase_function(
        phase_function: str | Callable,
    ) -> tuple[Callable, int]:
        """Resolve a phase function and its output count per target moment."""
        if isinstance(phase_function, str):
            key = phase_function.lower()
            if key not in _PHASE_FUNCTIONS:
                raise ValueError(f"Unknown phase_function '{phase_function}'. Available: {list(_PHASE_FUNCTIONS)}")
            return _PHASE_FUNCTIONS[key], _PHASE_OUTPUT_MULTIPLIERS[key]
        return phase_function, 1

    @staticmethod 
    def _resolve_norm_function(norm_function: str | Callable) -> Callable | None: 
        """Resolve normalization function from string alias or return callable as-is."""
        if norm_function is None:
            return torch.nn.Identity()
        if isinstance(norm_function, str):
            key = norm_function.lower()
            if key == "layer_norm":
                return shared_layer_norm
            else:
                raise ValueError(f"Unknown norm_function '{norm_function}'. Available: ['shared_layer_norm']")
        return norm_function

    @staticmethod
    def _resolve_magnitude_function(magnitude_function: str | Callable) -> Callable:
        """Resolve magnitude function from string alias or return callable as-is."""
        if isinstance(magnitude_function, str):
            key = magnitude_function.lower()
            if key not in _MAGNITUDE_FUNCTIONS:
                raise ValueError(
                    f"Unknown magnitude_function '{magnitude_function}'. Available: {list(_MAGNITUDE_FUNCTIONS)}"
                )
            return _MAGNITUDE_FUNCTIONS[key]
        return magnitude_function
    
    def _invariants_common(self, moments: torch.Tensor) -> torch.Tensor:
        """Compute trivials, magnitudes, and phase-normalized invariants.

        Args:
            moments: Input tensor shaped ``[B, S, O, H, W]``.

        Returns:
            Tensor shaped ``[B, S * out_channels, H, W]``.
        """
        trivial = moments[:, :, : self.trivial_idx]

        non_trivial = rearrange(
            moments[:, :, self.trivial_idx :],
            "b s (o c) h w -> b s o c h w",
            o=self.non_trivial_orders_count,
            c=2,
        )
        tuple = (trivial, non_trivial, )

        trivial, non_trivial = self.layer_norm_function(tuple)
        magnitudes = torch.linalg.vector_norm(non_trivial, dim=-3)

        phase_invariants = self.phase_function(
            non_trivial,
            magnitudes,
            self.non_trivial_orders,
            self.magnitude_function,
        )

        invariants = torch.cat((trivial, magnitudes, phase_invariants), dim=2)
        invariants = rearrange(invariants, "b c i h w -> b (c i) h w")
        return invariants


    def forward(self, moments: torch.Tensor) -> torch.Tensor:
        """Forward pass over input moments."""
        return self._compiled_forward(moments)

class CompiledMomentO2Layer(torch.nn.Module): 
    def __init__(self,
                 max_order: int,
                 orders: list[int],
                 in_channels: int, 
                 padding: str='same',
                 kernel_size: int=11,
                 **kwargs):
        super().__init__()
        # Parameters
        self.orders = orders
        self.max_order = max_order
        self.in_size = in_channels
        self.padding = compute_padding(padding, kernel_size)
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        basis_filter, self.rings, self.sigma, _ = compute_basis_params(kernel_size) 
        gspace = gspaces.flipRot2dOnR2(N=-1)      
        self.in_type = escnn.nn.FieldType(gspace, self.in_size*[gspace.trivial_repr])
        irreps_per_input_channel = [gspace.irrep(1, order) if order != 0 else gspace.irrep(0, 0) for order in sorted(self.orders)]
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
                                            basis_filter=basis_filter,
                                            # ESCNN's global cache returns the same
                                            # module object across layers. Moving one
                                            # layer to CUDA would then move the basis
                                            # out from under later CPU constructors.
                                            recompute=True)

        # Learnable parameters
        self.weights = torch.nn.Parameter(torch.zeros(self.basisexpansion.dimension()), requires_grad=True)
        escnn.nn.init.generalized_he_init(self.weights.data, self.basisexpansion)

        # Caching filter for inference
        self.register_buffer("filter", self.expand_parameters())

    def expand_parameters(self):
        _filter = self.basisexpansion(self.weights)
        _filter = _filter.reshape(_filter.shape[0], _filter.shape[1], self.kernel_size, self.kernel_size)                      

        return _filter

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _filter = self.expand_parameters()
        moments = torch.nn.functional.conv2d(x,
                                         _filter,
                                          bias=None,
                                          stride=1,
                                          groups=1, 
                                          padding=self.padding) 
        moments = rearrange(moments, 'b (ch o) h w -> b ch o h w', ch=self.in_channels)
        # TODO: 
        return moments

class CompiledMomentLayer(torch.nn.Module): 
    def __init__(self,
                 max_order: int,
                 orders: list[int],
                 in_channels: int, 
                 padding: str='same',
                 kernel_size: int=11,
                 **kwargs):
        super().__init__()
        # Parameters
        self.orders = orders
        self.max_order = max_order
        self.in_size = in_channels
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
                                            basis_filter=basis_filter,
                                            # Keep the sampled basis owned by this
                                            # layer so device transfers cannot mutate
                                            # ESCNN's globally cached module.
                                            recompute=True)

        # Learnable parameters
        self.weights = torch.nn.Parameter(torch.zeros(self.basisexpansion.dimension()), requires_grad=True)
        escnn.nn.init.generalized_he_init(self.weights.data, self.basisexpansion)

        # Caching filter for inference
        self.register_buffer("filter", self.expand_parameters())

    def expand_parameters(self):
        _filter = self.basisexpansion(self.weights)
        _filter = _filter.reshape(_filter.shape[0], _filter.shape[1], self.kernel_size, self.kernel_size)                      

        return _filter

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _filter = self.expand_parameters()
        moments = torch.nn.functional.conv2d(x,
                                         _filter,
                                          bias=None,
                                          stride=1,
                                          groups=1, 
                                          padding=self.padding) 
        moments = rearrange(moments, 'b (ch o) h w -> b ch o h w', ch=self.in_channels)
        # TODO: 
        return moments

class CompiledReImLayer(torch.nn.Module):

    def __init__(
        self,
        orders: Sequence[int] | torch.Tensor,
        pre_norm_function: str | None = None,
#        after_norm_function: str | None = None,
        compile_functions: bool = False,
        compile_kwargs: dict | None = dict(fullgraph=True),
        eps=1e-7
    ):
        """Initialize the invariant projection layer.

        Args:
            orders: Ordered moment frequencies. Must be 1D and include at least
                one ``0`` (trivial) and one non-trivial order. The first
                non-trivial order must be ``1`` to act as normalizer.
            phase_function: Either a key in
                ``{"real", "polar", "angle", "circular"}`` or a custom
                callable with signature ``fn(v, mag, o, magnitude_fn, eps=...)``.
            magnitude_function: Either a key in ``{"nicks", "roxanas"}`` or a
                custom callable with signature ``fn(mag1, mag2)``.
            pre_norm_function: Optional string specifying the prenormalization method.
            after_norm_function: Optional string specifying the afternormalization method.
            compile_functions: If ``True``, compiles the forward invariant kernel.
            compile_kwargs: Keyword args passed to ``torch.compile``.
        """
        super().__init__()

        if compile_kwargs is None:
            compile_kwargs = dict()

        orders = torch.as_tensor(orders, dtype=torch.int32)
        if orders.ndim != 1:
            raise ValueError("orders must be a 1D sequence")

        if int((orders == 0).sum().item()) == 0:
            raise ValueError("orders must contain at least one trivial order (0)")

        self.register_buffer("orders", orders)

        self.trivial_idx = int((orders == 0).sum().item())
        self.non_trivial_orders_count = int(orders.numel() - self.trivial_idx)
        self.eps = eps 

        if self.non_trivial_orders_count < 1:
            raise ValueError("Need at least one non-trivial order for normalization")
        
        if orders[self.trivial_idx] != 1: 
            raise ValueError("The first non-trivial order must be 1 for normalization")

        # Orders for targets only; exclude the first order-1 normalizer moment.
        self.register_buffer("non_trivial_orders", orders[self.trivial_idx + 1 :])

        self.layer_norm_function = self._resolve_norm_function(pre_norm_function)

        if compile_functions:
            self._compiled_forward = torch.compile(self._flusser, **compile_kwargs)
        else:
            self._compiled_forward = self._flusser
        
        self.out_channels = (
            self.trivial_idx
            + 2 * (self.non_trivial_orders_count - 1)
        )

    
    @staticmethod 
    def _resolve_norm_function(norm_function: str | Callable) -> Callable | None: 
        """Resolve normalization function from string alias or return callable as-is."""
        if norm_function is None:
            return torch.nn.Identity()
        if isinstance(norm_function, str):
            key = norm_function.lower()
            if key == "layer_norm":
                return shared_layer_norm
            else:
                raise ValueError(f"Unknown norm_function '{norm_function}'. Available: ['shared_layer_norm']")
        return norm_function

    def _flusser(self, moments: torch.Tensor) -> torch.Tensor:
        """Compute trivials, magnitudes, and phase-normalized invariants.

        Args:
            moments: Input tensor shaped ``[B, S, O, H, W]``.

        Returns:
            Tensor shaped ``[B, S * out_channels, H, W]``.
        """
        trivial = moments[:, :, : self.trivial_idx]

        non_trivial = rearrange(
            moments[:, :, self.trivial_idx :],
            "b s (o c) h w -> b s o c h w",
            o=self.non_trivial_orders_count,
            c=2,
        )

        mag = torch.linalg.vector_norm(non_trivial, dim=-3)
        safe_mag = mag.clamp(min=self.eps)
        real, imag = non_trivial.unbind(dim=-3)
        angle = SafeAtan2.apply(imag / safe_mag, real / safe_mag, self.eps)
        n_angle = angle[:, :, 0:1] * -self.non_trivial_orders[:, None, None]
        nt_angle = angle[:, :, 1:]

        rel_angle = nt_angle + n_angle
        out_mag = mag[:, :, 0:1] * mag[:, :, 1:]
        real = out_mag * torch.cos(rel_angle)
        imag = out_mag * torch.sin(rel_angle)
        
        invariants = torch.cat((trivial, real, imag), dim=2)
        invariants = rearrange(invariants, "b c i h w -> b (c i) h w")
        return invariants

    def forward(self, moments: torch.Tensor) -> torch.Tensor:
        """Forward pass over input moments."""
        return self._compiled_forward(moments)