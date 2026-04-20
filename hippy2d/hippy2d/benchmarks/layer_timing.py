from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

import escnn
from escnn import nn as enn

from hippy2d.escnn_prototype import LearnableCesaMagRealVarFunc, compute_padding, flusser_basis


def _sorted_flusser_basis(max_order: int) -> list[tuple[int, int]]:
    return sorted(flusser_basis(max_order), key=lambda pq: (pq[0] - pq[1], pq[0], pq[1]))


def flusser_orders(max_order: int) -> list[int]:
    return [p - q for p, q in _sorted_flusser_basis(max_order)]


def summarize_flusser_basis(max_order: int) -> dict[str, Any]:
    basis = _sorted_flusser_basis(max_order)
    records = []
    order_counter = Counter()
    rep_counter = Counter()

    for p, q in basis:
        order = p - q
        rep_type = "trivial" if order == 0 else "non_trivial"
        feature_dim = 1 if order == 0 else 2
        order_counter[order] += 1
        rep_counter[rep_type] += 1
        records.append(
            {
                "p": p,
                "q": q,
                "order": order,
                "type": rep_type,
                "feature_dim": feature_dim,
            }
        )

    return {
        "max_order": max_order,
        "basis_size": len(basis),
        "basis_functions": records,
        "counts_by_order": dict(sorted(order_counter.items())),
        "counts_by_type": dict(rep_counter),
        "total_feature_dim_per_channel": sum(item["feature_dim"] for item in records),
    }


def make_o2_gspace(max_order: int) -> escnn.gspaces.GSpace2D:
    return escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=max_order)


def make_o2_field_type(
    gspace: escnn.gspaces.GSpace2D,
    channels: int,
    orders: list[int],
) -> enn.FieldType:
    reps = []
    for _ in range(channels):
        for order in orders:
            reps.append(gspace.trivial_repr if order == 0 else gspace.irrep(order))
    return enn.FieldType(gspace, reps)


def create_o2_steerable_conv(
    orders: list[int],
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: str | int = "same",
    maximum_frequency: int | None = None,
) -> tuple[escnn.gspaces.GSpace2D, enn.FieldType, enn.FieldType, enn.R2Conv]:
    max_frequency = maximum_frequency or max(abs(order) for order in orders)
    gspace = make_o2_gspace(max_frequency)
    in_type = enn.FieldType(gspace, [gspace.trivial_repr] * in_channels)
    out_type = make_o2_field_type(gspace, out_channels, orders)
    layer = enn.R2Conv(
        in_type,
        out_type,
        kernel_size=kernel_size,
        padding=compute_padding(padding, kernel_size),
    )
    return gspace, in_type, out_type, layer


class O2SteerableConvStack(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        depth: int,
        orders: list[int],
        kernel_size: int,
        padding: str | int,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        max_frequency = max(abs(order) for order in orders)
        self.gspace = make_o2_gspace(max_frequency)
        self.input_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * in_channels)
        self.hidden_type = make_o2_field_type(self.gspace, hidden_channels, orders)
        self.layers = torch.nn.ModuleList()

        self.layers.append(
            enn.R2Conv(
                self.input_type,
                self.hidden_type,
                kernel_size=kernel_size,
                padding=compute_padding(padding, kernel_size),
            )
        )
        for _ in range(1, depth):
            self.layers.append(
                enn.R2Conv(
                    self.hidden_type,
                    self.hidden_type,
                    kernel_size=kernel_size,
                    padding=compute_padding(padding, kernel_size),
                )
            )

    def forward(self, x: torch.Tensor) -> enn.GeometricTensor:
        y = enn.GeometricTensor(x, self.input_type)
        for layer in self.layers:
            y = layer(y)
        return y


class InvariantStack(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        depth: int,
        spatial_size: int,
        max_order: int,
        kernel_size: int,
        padding: str | int,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.layers = torch.nn.ModuleList()
        current_channels = in_channels
        self.channel_progression = [current_channels]

        for _ in range(depth):
            layer = LearnableCesaMagRealVarFunc(
                in_channels=current_channels,
                out_channels=current_channels,
                input_size=spatial_size,
                padding=padding,
                max_order=max_order,
                kernel_size=kernel_size,
                mag_func="nick",
                norm_per_inv_type="none",
                output_features="all",
            )
            self.layers.append(layer)
            current_channels = layer.out_channels
            self.channel_progression.append(current_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_callable(fn, device: torch.device, iterations: int) -> float:
    _sync_if_needed(device)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    _sync_if_needed(device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def _prepare_o2_inputs(model: O2SteerableConvStack, x: torch.Tensor) -> list[enn.GeometricTensor]:
    prepared = []
    current = enn.GeometricTensor(x, model.input_type)
    for layer in model.layers:
        prepared.append(current)
        current = layer(current)
    return prepared


def _prepare_tensor_inputs(model: InvariantStack, x: torch.Tensor) -> list[torch.Tensor]:
    prepared = []
    current = x
    for layer in model.layers:
        prepared.append(current)
        current = layer(current)
    return prepared


@dataclass
class LayerTiming:
    layer_index: int
    input_shape: list[int]
    output_shape: list[int]
    time_ms: float


@dataclass
class PairTiming:
    first_layer_index: int
    second_layer_index: int
    input_shape: list[int]
    output_shape: list[int]
    time_ms: float


def benchmark_o2_stack(
    model: O2SteerableConvStack,
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        prepared = _prepare_o2_inputs(model, x)

        for _ in range(warmup):
            _ = model(x)

        layer_timings = []
        for idx, (layer, layer_input) in enumerate(zip(model.layers, prepared)):
            sample_output = layer(layer_input)
            elapsed = _time_callable(lambda: layer(layer_input), device, iterations)
            layer_timings.append(
                LayerTiming(
                    layer_index=idx,
                    input_shape=list(layer_input.tensor.shape),
                    output_shape=list(sample_output.tensor.shape),
                    time_ms=elapsed,
                )
            )

        pair_timings = []
        for idx in range(len(model.layers) - 1):
            first = model.layers[idx]
            second = model.layers[idx + 1]
            pair_input = prepared[idx]
            sample_output = second(first(pair_input))
            elapsed = _time_callable(lambda: second(first(pair_input)), device, iterations)
            pair_timings.append(
                PairTiming(
                    first_layer_index=idx,
                    second_layer_index=idx + 1,
                    input_shape=list(pair_input.tensor.shape),
                    output_shape=list(sample_output.tensor.shape),
                    time_ms=elapsed,
                )
            )

    return {
        "input_type_size": model.input_type.size,
        "hidden_type_size": model.hidden_type.size,
        "layer_timings": [asdict(item) for item in layer_timings],
        "pair_timings": [asdict(item) for item in pair_timings],
    }


def benchmark_invariant_stack(
    model: InvariantStack,
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        prepared = _prepare_tensor_inputs(model, x)

        for _ in range(warmup):
            _ = model(x)

        layer_timings = []
        for idx, (layer, layer_input) in enumerate(zip(model.layers, prepared)):
            sample_output = layer(layer_input)
            elapsed = _time_callable(lambda: layer(layer_input), device, iterations)
            layer_timings.append(
                LayerTiming(
                    layer_index=idx,
                    input_shape=list(layer_input.shape),
                    output_shape=list(sample_output.shape),
                    time_ms=elapsed,
                )
            )

        pair_timings = []
        for idx in range(len(model.layers) - 1):
            first = model.layers[idx]
            second = model.layers[idx + 1]
            pair_input = prepared[idx]
            sample_output = second(first(pair_input))
            elapsed = _time_callable(lambda: second(first(pair_input)), device, iterations)
            pair_timings.append(
                PairTiming(
                    first_layer_index=idx,
                    second_layer_index=idx + 1,
                    input_shape=list(pair_input.shape),
                    output_shape=list(sample_output.shape),
                    time_ms=elapsed,
                )
            )

    return {
        "channel_progression": model.channel_progression,
        "layer_timings": [asdict(item) for item in layer_timings],
        "pair_timings": [asdict(item) for item in pair_timings],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark O(2) steerable layers against invariant moment layers.")
    parser.add_argument("--max-order", type=int, default=1)
    parser.add_argument("--channels", type=int, default=32, help="Hidden channel count used for the O(2) stack.")
    parser.add_argument(
        "--spatial",
        nargs=3,
        type=int,
        metavar=("H", "W", "C"),
        default=[64, 64, 3],
        help="Input tensor spatial shape as H W C.",
    )
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--padding", type=str, default="same")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _format_ms(value: float) -> str:
    return f"{value:.3f} ms"


def _print_basis_summary(summary: dict[str, Any]) -> None:
    print("\n[Step 0] Flusser basis summary")
    print(f"max_order={summary['max_order']} basis_size={summary['basis_size']}")
    print(f"counts_by_order={summary['counts_by_order']}")
    print(f"counts_by_type={summary['counts_by_type']}")
    print(f"total_feature_dim_per_channel={summary['total_feature_dim_per_channel']}")
    print("basis_functions:")
    for item in summary["basis_functions"]:
        print(
            f"  (p={item['p']}, q={item['q']})"
            f" order={item['order']}"
            f" type={item['type']}"
            f" feature_dim={item['feature_dim']}"
        )


def _print_timing_table(name: str, result: dict[str, Any]) -> None:
    print(f"\n[{name}] layer timings")
    for item in result["layer_timings"]:
        print(
            f"  layer {item['layer_index']}: "
            f"{item['input_shape']} -> {item['output_shape']} "
            f"{_format_ms(item['time_ms'])}"
        )

    if result["pair_timings"]:
        print(f"\n[{name}] consecutive layer-pair timings")
        for item in result["pair_timings"]:
            print(
                f"  layers {item['first_layer_index']}->{item['second_layer_index']}: "
                f"{item['input_shape']} -> {item['output_shape']} "
                f"{_format_ms(item['time_ms'])}"
            )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64 if args.dtype == "float64" else torch.float32)

    height, width, input_channels = args.spatial
    if height != width:
        raise ValueError("This benchmark currently expects square inputs because LearnableCesaMagRealVarFunc takes a single input_size.")

    if args.depth < 1:
        raise ValueError("--depth must be at least 1")
    if args.kernel_size % 2 == 0 and args.padding == "same":
        raise ValueError("--kernel-size must be odd when --padding same is used")

    device = torch.device(args.device)
    orders = flusser_orders(args.max_order)
    basis_summary = summarize_flusser_basis(args.max_order)

    x = torch.randn(args.batch_size, input_channels, height, width, device=device)

    o2_stack = O2SteerableConvStack(
        in_channels=input_channels,
        hidden_channels=args.channels,
        depth=args.depth,
        orders=orders,
        kernel_size=args.kernel_size,
        padding=args.padding,
    ).to(device)

    invariant_stack = InvariantStack(
        in_channels=input_channels,
        depth=args.depth,
        spatial_size=height,
        max_order=args.max_order,
        kernel_size=args.kernel_size,
        padding=args.padding,
    ).to(device)

    o2_result = benchmark_o2_stack(o2_stack, x, device, args.warmup, args.iterations)
    invariant_result = benchmark_invariant_stack(invariant_stack, x, device, args.warmup, args.iterations)

    print(f"device={device.type} dtype={args.dtype} batch_size={args.batch_size}")
    print(f"spatial={height}x{width}x{input_channels} hidden_channels={args.channels} depth={args.depth}")
    print(f"kernel_size={args.kernel_size} padding={args.padding}")

    _print_basis_summary(basis_summary)

    print("\n[Step 1] O(2) representation")
    print(f"orders={orders}")
    print(f"input_type_size={o2_result['input_type_size']}")
    print(f"hidden_type_size={o2_result['hidden_type_size']}")

    print("\n[Step 2] Invariant stack")
    print("mag_func=nick norm_per_inv_type=none output_features=all")
    print(f"channel_progression={invariant_result['channel_progression']}")

    _print_timing_table("O(2) steerable", o2_result)
    _print_timing_table("Invariant", invariant_result)

    payload = {
        "config": {
            "max_order": args.max_order,
            "channels": args.channels,
            "spatial": [height, width, input_channels],
            "depth": args.depth,
            "kernel_size": args.kernel_size,
            "padding": args.padding,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "device": device.type,
            "dtype": args.dtype,
        },
        "basis_summary": basis_summary,
        "o2_steerable": o2_result,
        "invariant": invariant_result,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON results to {args.json_out}")


if __name__ == "__main__":
    main()
