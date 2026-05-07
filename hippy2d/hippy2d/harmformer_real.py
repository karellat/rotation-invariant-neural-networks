import torch
import numpy as np
from torch import nn
from scipy.linalg import dft
from einops import rearrange

from hippy2d.utils import get_circular_mask, tukey_2d
from hippy2d.harmformer import HConv2d, drop_path


def _as_real_pair(x: torch.Tensor) -> torch.Tensor:
    if x.is_complex():
        return torch.view_as_real(x)
    if x.shape[-1] == 2:
        return x
    return torch.stack((x, torch.zeros_like(x)), dim=-1)


class RealImg2H(nn.Module):
    def __init__(self, circular_mask=False, input_shape=None, alpha=0.5):
        super().__init__()
        self.circular_mask = circular_mask
        if circular_mask:
            if input_shape is None:
                raise ValueError("input_shape is required for circular_mask")
            self.mask = nn.Parameter(
                torch.from_numpy(tukey_2d(input_shape, alpha=alpha)[None, None, None, ..., None])
                .to(dtype=torch.get_default_dtype()),
                requires_grad=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_real_pair(x)
        if x.ndim != 5:
            raise ValueError("Expected [B, C, H, W] or [B, C, H, W, 2]")
        x = rearrange(x, "b (o c) h w q -> b o c h w q", o=1)
        if self.circular_mask:
            x = x * self.mask
        return x


class RealHConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        in_max_order: int,
        out_max_order: int,
        mask_shape: int,
        tukey_window: bool = False,
        tukey_alpha: float = 0.5,
        phase: bool = True,
        n_rings: int = 3,
        padding: int = 0,
        stride: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.phase = phase
        self.in_max_order = in_max_order
        self.out_max_order = out_max_order
        self.n_rings = n_rings
        self.padding = padding
        self.stride = stride
        self._max_order = max(in_max_order + 1, out_max_order + 1)

        self.ring_weights = nn.Parameter(
            HConv2d.init_weights(
                in_channels=in_channels,
                out_channels=out_channels,
                max_order=self._max_order,
                ring_count=n_rings,
            ),
            requires_grad=True,
        )
        if phase:
            self.phase_offset = nn.Parameter(
                HConv2d.init_phase(in_channels, out_channels, self._max_order).to(torch.get_default_dtype()),
                requires_grad=True,
            )

        self.tukey_window = tukey_window
        if tukey_window:
            mask = torch.from_numpy(tukey_2d(mask_shape, alpha=tukey_alpha)[None, None, None, ..., None])
        else:
            mask = ~get_circular_mask(mask_shape)[None, None, None, ..., None]
        self.mask = nn.Parameter(mask.to(dtype=torch.get_default_dtype()), requires_grad=False)

        weights2filters = HConv2d.init_weights2filters(self._max_order, kernel_size, n_rings)
        self.register_buffer("weights2filters_real", weights2filters.real.to(torch.get_default_dtype()))
        self.register_buffer("weights2filters_imag", weights2filters.imag.to(torch.get_default_dtype()))

    def get_filters(self) -> tuple[torch.Tensor, torch.Tensor]:
        real = torch.matmul(self.weights2filters_real, self.ring_weights)
        imag = torch.matmul(self.weights2filters_imag, self.ring_weights)
        real = rearrange(
            real,
            "o (h w) (ic oc) -> o oc ic h w",
            h=self.kernel_size,
            w=self.kernel_size,
            ic=self.in_channels,
            oc=self.out_channels,
        )
        imag = rearrange(
            imag,
            "o (h w) (ic oc) -> o oc ic h w",
            h=self.kernel_size,
            w=self.kernel_size,
            ic=self.in_channels,
            oc=self.out_channels,
        )
        if self.phase:
            cos = torch.cos(self.phase_offset)
            sin = torch.sin(self.phase_offset)
            real, imag = real * cos + imag * sin, imag * cos - real * sin
        return real, imag

    def _real_weight(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        weight_blocks = []
        for out_order in range(self.out_max_order + 1):
            in_blocks = []
            for in_order in range(self.in_max_order + 1):
                weight_order = out_order - in_order
                wr = real[abs(weight_order)]
                wi = imag[abs(weight_order)]
                if weight_order == 0:
                    wi = torch.zeros_like(wi)
                elif weight_order < 0:
                    wi = -wi
                block = torch.stack(
                    (
                        torch.stack((wr, -wi), dim=1),
                        torch.stack((wi, wr), dim=1),
                    ),
                    dim=1,
                )
                in_blocks.append(block)
            weight_blocks.append(torch.stack(in_blocks, dim=2))
        weight = torch.stack(weight_blocks, dim=0)
        return rearrange(weight, "oo oc oq io ic iq h w -> (oo oc oq) (io ic iq) h w")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_max_order + 1:
            raise ValueError(f"Expected {self.in_max_order + 1} orders, got {x.shape[1]}")
        x = rearrange(x, "b o c h w q -> b (o c q) h w")
        weight = self._real_weight(*self.get_filters())
        y = nn.functional.conv2d(x, weight, padding=self.padding, stride=self.stride)
        y = rearrange(y, "b (o c q) h w -> b o c h w q", o=self.out_max_order + 1, c=self.out_channels, q=2)
        if self.tukey_window:
            return y * self.mask
        return y.masked_fill(self.mask, 0)


class RealHNormAct(nn.Module):
    def __init__(self, act_fnc: str, channels: int, eps: float = 1e-8, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.batch_norm = nn.BatchNorm3d(channels, eps=eps, momentum=momentum, affine=affine)
        self.activation_fnc = getattr(nn.functional, act_fnc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.sqrt(torch.clamp(x[..., 0].square() + x[..., 1].square(), min=self.eps * self.eps))
        out = rearrange(magnitude, "b o c h w -> b c h w o")
        out = self.batch_norm(out)
        out = rearrange(out, "b c h w o -> b o c h w")
        out = self.activation_fnc(out)
        scale = self.activation_fnc(out) / magnitude
        return x * scale[..., None]


class RealHPooling(nn.Module):
    def __init__(self, number_of_ranks: int, kernel_size=(2, 2), pooling_type="avg", stride=(2, 2)):
        super().__init__()
        if pooling_type != "avg":
            raise NotImplementedError("RealHPooling currently supports avg pooling only")
        self.number_of_ranks = number_of_ranks
        self.pooling_layer = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b o c h w q -> b (o c q) h w")
        x = self.pooling_layer(x)
        return rearrange(x, "b (o c q) h w -> b o c h w q", o=self.number_of_ranks, q=2)


class RealDropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class RealOrderProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.real = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.imag = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = self.real(x[..., 0]) - self.imag(x[..., 1])
        imag = self.imag(x[..., 0]) + self.real(x[..., 1])
        return torch.stack((real, imag), dim=-1)


class RealHOut(nn.Module):
    def __init__(self, keep_order_dim=True, eps: float = 1e-8):
        super().__init__()
        self.keep_order_dim = keep_order_dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sqrt(torch.clamp(x[..., 0].square() + x[..., 1].square(), min=self.eps * self.eps))
        if self.keep_order_dim:
            return x
        return rearrange(x, "b o c h w -> b (o c) h w")
