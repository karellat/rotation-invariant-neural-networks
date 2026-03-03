import torch
import escnn
from einops import rearrange, repeat
from collections import defaultdict

from hippy2d.utils import SafeAtan2

# Pure invariant layer
class Conv1x1EquivariantLayer(escnn.nn.EquivariantModule):
    def __init__(self, 
                 r2_act: escnn.gspaces.rot2dOnR2,
                 in_type: escnn.nn.FieldType,
                 out_type: escnn.nn.FieldType):
        super(Conv1x1EquivariantLayer, self).__init__()
        # Assert the gspaces
        assert isinstance(r2_act, escnn.gspaces.GSpace2D), "Must be 2D group"
        assert isinstance(r2_act.fibergroup, escnn.group.SO2), "Must be continous."
        assert in_type.gspace == r2_act, "in_type must have the same gspace as r2_act"
        assert out_type.gspace == r2_act, "out_type must have the same gspace as r2_act"

        self.in_type = in_type
        self.out_type = out_type
        
        self.conv = torch.nn.Conv2d()


class InvariantLayer(escnn.nn.EquivariantModule): 
    def __init__(self, 
                 r2_act: escnn.gspaces.rot2dOnR2,
                 in_type: escnn.nn.FieldType,
                 in_channels: int):
        super(InvariantLayer, self).__init__()
        # Assert the gspaces
        assert isinstance(r2_act, escnn.gspaces.GSpace2D), "Must be 2D group"
        assert isinstance(r2_act.fibergroup, escnn.group.SO2), "Must be continous."
        assert in_type.gspace == r2_act, "in_type must have the same gspace as r2_act"
        assert in_channels > 0, "in_channels must be positive"

        # Check norm_factor in the input type
        assert "norm_factor" in [r.name for r in in_type.representations], "Input type must contain norm_factor representation"

        self.in_type = in_type
        
        # To be init function 
        nfields = defaultdict(int)
        # indices of the channels corresponding to fields belonging to each group
        _indices = defaultdict(lambda: [])
        # whether each group of fields is contiguous or not
        _contiguous = {}


        position = 0
        types = []
        last_field = None
        output_size = 0 
        for _, r in enumerate(in_type.representations):
            if r.name == "irrep_0":
                name = "trivial"
                output_size += r.size
            elif r.name == "norm_factor":
                name = "norm_factor"
            else:
                name = "non_trivial"
                output_size += r.size

            if name != last_field:
                if not name in _contiguous:
                    _contiguous[name] = True
                else:
                    raise NotImplementedError("Non-contiguous fields are not supported yet.")

            last_field = name
            _indices[name] += list(range(position, position + r.size))
            if r.name != "norm_factor" and r.name != "irrep_0":
                types += r.id
            nfields[r.name] += 1
            position += r.size

        for name, _ in _contiguous.items():
            _indices[name] = [min(_indices[name]), max(_indices[name])+1]
        
        assert "trivial" in _indices or "non_trivial" in _indices, "Input type must contain at least trivial or non-trivial representations"
        assert "norm_factor" in _indices, "Input type must contain norm_factor representation"
        
        # Prepare exponents for rotation of norm_factors
        exponents = []
        for name, count in nfields.items():
            if name == "norm_factor" or name == "irrep_0":
                continue
            else: 
                assert count == in_channels, f"All fields of type {name} must have the same number of channels. Found {count} instead of {in_channels}."
                order = torch.tensor(-int(name.split('_')[1]))
                assert order < 0, "Rotating in oposite direction than moments."
                # Exponents Channels x Complex x H x W
                order = repeat(order, '-> c 1 1 1', c=in_channels)
                exponents.append(order)
        
        assert len(exponents) > 0, "Input type must contain at least one non-trivial representation"
        exponents = torch.stack(exponents, dim=0) # Orders x Channels x Complex x H x W

        self.register_buffer("trivial_indices", torch.tensor(_indices["trivial"]))
        self.register_buffer("non_trivial_indices", torch.tensor(_indices["non_trivial"]))
        self.register_buffer("norm_factor_indices", torch.tensor(_indices["norm_factor"]))
        self.register_buffer("conjugate_one", torch.tensor([1.0, -1.0]).view(1, 1, 2, 1, 1))
        self.register_buffer("in_channels", torch.tensor(in_channels))
        self.register_buffer("exponents", exponents)
        self.types = types
     
        self.out_type = escnn.nn.FieldType(r2_act, [r2_act.trivial_repr] * output_size)

    def forward(self, x: escnn.nn.GeometricTensor) -> escnn.nn.GeometricTensor:
        x  = x.tensor
        b, _, h, w = x.shape
        # Trivials are kept as is
        trivials = x[:, self.trivial_indices[0]:self.trivial_indices[1], :, :]

        # Norm Factor:  Batch x Orders x Channels x Complex x H x W
        norm_factor  = x[:, self.norm_factor_indices[0]:self.norm_factor_indices[1], :, :]
        norm_factor = norm_factor.view(b, 1, self.in_channels, 2, h, w)
        # Make a conjugate to match the rotation direction
        #norm_factor = norm_factor * self.conjugate_one
        # Rotate the moments
        norm_factor = self._rotate_norm_factor(norm_factor)
        
        # Non-Trivials: Batch x Orders x Channels x Complex x H x W
        non_trivials = x[:, self.non_trivial_indices[0]:self.non_trivial_indices[1], :, :]
        non_trivials = non_trivials.view(b, -1, self.in_channels, 2, h, w)

        invariants = self._complex_mul(non_trivials, norm_factor)
        invariants = invariants.view(b, -1, h, w)

        all_invariants = torch.cat([trivials, invariants], dim=1)
        return escnn.nn.GeometricTensor(all_invariants, self.out_type)


    def _rotate_norm_factor(self, 
                            norm_factor: torch.Tensor) -> torch.Tensor:

        ## Normed by magnitude
        magnitude = torch.linalg.vector_norm(norm_factor, dim=3, keepdim=True)
        # Note: Scale by sigmoid to prevent exploding
        # TODO: Try different functions
        magnitude = torch.sigmoid(magnitude)
        moment_real = norm_factor[:, :, :, 0:1, :, :]
        moment_imag = norm_factor[:, :, :, 1:2, :, :]
        angle = SafeAtan2.apply(moment_imag, moment_real)
        new_angle = angle * self.exponents
        new_angle.shape, magnitude.shape
        result_real = magnitude * torch.cos(new_angle)
        result_imag = magnitude * torch.sin(new_angle)
        # Prepare a copy version
        return torch.cat([result_real, result_imag], dim=3)
    
        # This does not improve accuracy but it stops growing of activations
        # TODO: but check the activations
        # Just rotate 
        #normed_factor = norm_factor / torch.clamp(torch.norm(norm_factor, dim=3, keepdim=True), min=1e-4)
        ## Calculate to keep zeros
        #magnitude = torch.norm(normed_factor, dim=3, keepdim=True)
        #moment_real = normed_factor[:, :, :, 0:1, :, :]
        #moment_imag = normed_factor[:, :, :, 1:2, :, :]
        #angle = SafeAtan2.apply(moment_imag, moment_real)
        #new_angle = angle * self.exponents
        #new_angle.shape, magnitude.shape
        #result_real = magnitude * torch.cos(new_angle)
        #result_imag = magnitude * torch.sin(new_angle)
        #return torch.cat([result_real, result_imag], dim=3)

    def _complex_mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xr = x[..., 0:1, :, :]
        xi = x[..., 1:2, :, :]
        yr = y[..., 0:1, :, :]
        yi = y[..., 1:2, :, :]
        real = xr * yr - xi * yi
        imag = xr * yi + xi * yr
        return torch.cat((real, imag), dim=3)
        

    def check_equivariance(self, atol: float = 1e-6) -> None:
        pass

    def evaluate_output_shape(self, input_shape):
        return super().evaluate_output_shape(input_shape)


class EscnnInvariantLayer(escnn.nn.EquivariantModule):
    """
    Convert mixed SO(2)/O(2) fields into pure type-0 fields.

    The first frequency-1 irrep in `in_type` is used as a normalizer.
    Output channels are:
    - all input trivial channels (copied),
    - magnitudes of all non-trivial 2D irreps,
    - real part of normalized non-trivial irreps (excluding the normalizer).
    """

    def __init__(self,
                 in_type: escnn.nn.FieldType,
                 return_normalizer_type1: bool = False,
                 equivariant_output: bool = None, 
                 ):
        super().__init__()

        assert isinstance(in_type.gspace, escnn.gspaces.GSpace2D), "Must be 2D group action"
        assert isinstance(in_type.gspace.fibergroup, (escnn.group.SO2, escnn.group.O2)), "Only SO(2) and O(2) are supported"

        self.in_type = in_type
        # Backward-compatible alias: `equivariant_output=True` means returning the type-1 normalizer too.
        if equivariant_output is not None:
            self.return_normalizer_type1 = bool(equivariant_output)
        else:
            self.return_normalizer_type1 = bool(return_normalizer_type1)

        trivial_indices = []
        complex_indices = []
        complex_reps = []
        frequencies = []
        normalizer_idx = None

        position = 0
        for rep in in_type.representations:
            if rep.size == 1:
                if rep.is_trivial():
                    trivial_indices.append(position)
                else:
                    raise ValueError(f"Unsupported non-trivial scalar representation: {rep.name}")
                position += 1
                continue

            if rep.size != 2:
                raise ValueError(f"Unsupported representation size {rep.size} for {rep.name}; only sizes 1 and 2 are supported")

            freq = self._frequency_from_rep(rep)
            if freq < 1:
                raise ValueError(f"Unsupported frequency {freq} for representation {rep.name}")

            complex_indices.append((position, position + 1))
            complex_reps.append(rep)
            frequencies.append(freq)
            if normalizer_idx is None and freq == 1:
                normalizer_idx = len(complex_indices) - 1

            position += 2

        if len(complex_indices) == 0:
            raise ValueError("Input type must contain at least one non-trivial size-2 representation")
        if normalizer_idx is None:
            raise ValueError("Input type must contain at least one frequency-1 representation")

        all_complex_idx = torch.tensor(complex_indices, dtype=torch.long)
        all_freq = torch.tensor(frequencies, dtype=torch.int32)
        all_ids = torch.arange(len(complex_indices), dtype=torch.long)
        other_ids = all_ids[all_ids != normalizer_idx]
        other_exp = (-all_freq[other_ids]).to(dtype=torch.get_default_dtype())

        self.register_buffer("trivial_indices", torch.tensor(trivial_indices, dtype=torch.long))
        self.register_buffer("complex_indices", all_complex_idx)
        self.register_buffer("other_ids", other_ids)
        self.register_buffer("other_exponents", other_exp)
        self.normalizer_idx = int(normalizer_idx)
        self.normalizer_rep = complex_reps[self.normalizer_idx]

        n_type0_channels = len(trivial_indices) + len(complex_indices) + len(other_ids)
        out_reprs = [in_type.gspace.trivial_repr] * n_type0_channels
        if self.return_normalizer_type1:
            out_reprs.append(self.normalizer_rep)
        self.out_type = escnn.nn.FieldType(in_type.gspace, out_reprs)

    @staticmethod
    def _frequency_from_rep(rep: escnn.group.Representation) -> int:
        rid = rep.id
        if isinstance(rid, tuple):
            freq = rid[-1]
        else:
            freq = rid
        return int(abs(freq))

    @staticmethod
    def _complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xr = x[:, :, 0]
        xi = x[:, :, 1]
        yr = y[:, :, 0]
        yi = y[:, :, 1]
        real = xr * yr - xi * yi
        return real
        # NOTE: We don't need the imaginary part for invariants, but we can keep it for debugging or future use.
        #imag = xr * yi + xi * yr
        #return torch.stack([real, imag], dim=2)

    def forward(self, x: escnn.nn.GeometricTensor) -> escnn.nn.GeometricTensor:
        assert x.type == self.in_type, "Input type mismatch"
        tensor = x.tensor
        b, _, h, w = tensor.shape

        if self.trivial_indices.numel() > 0:
            trivial = tensor.index_select(1, self.trivial_indices)
        else:
            trivial = tensor.new_zeros((b, 0, h, w))

        complex_flat_idx = self.complex_indices.reshape(-1)
        moments = rearrange(
            tensor.index_select(1, complex_flat_idx),
            "b (n c) h w -> b n c h w",
            c=2,
        )

        all_magnitudes = torch.linalg.vector_norm(moments, dim=2)

        normalizer = moments[:, self.normalizer_idx:self.normalizer_idx + 1]
        norm_magnitude = torch.linalg.vector_norm(normalizer, dim=2)
        magnitude = torch.sigmoid(norm_magnitude)
        angle = SafeAtan2.apply(normalizer[:, :, 1], normalizer[:, :, 0], 1e-8)

        if self.other_ids.numel() > 0:
            exponents = self.other_exponents.view(1, -1, 1, 1)
            new_angle = angle * exponents
            norm_real = magnitude * torch.cos(new_angle)
            norm_imag = magnitude * torch.sin(new_angle)
            rotated_normalizer = torch.stack([norm_real, norm_imag], dim=2)

            others = moments.index_select(1, self.other_ids)
            compensated_real = self._complex_mul(others, rotated_normalizer)
            out = torch.cat([trivial, compensated_real, all_magnitudes], dim=1)
        else:
            out = torch.cat([trivial, all_magnitudes], dim=1)

        if self.return_normalizer_type1:
            out = torch.cat([out, normalizer[:, 0]], dim=1)

        return escnn.nn.GeometricTensor(out, self.out_type)

    def evaluate_output_shape(self, input_shape):
        b, _, h, w = input_shape
        return b, self.out_type.size, h, w

    def check_equivariance(self, atol: float = 1e-6):
        return True
