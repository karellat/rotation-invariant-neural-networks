import time

import torch
import torch.profiler

from hippy2d.opt_inv_layers import CompiledInvariantLayer, CompiledMomentLayer


def check_grad_safety(fn, *inputs):
    inputs[0].requires_grad_(True)

    out = fn(*inputs)
    loss = out.sum()
    loss.backward()

    grads = [x.grad for x in inputs[:1]]

    for i, g in enumerate(grads):
        if g is None:
            print(f"Input {i}: no gradients")
        elif not torch.isfinite(g).all() or torch.isnan(g).any():
            print(f"Input {i}: gradients are not finite")

def moments() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    orders = [0, 1, 2, 3, 4, 5, 6]
    shape = [64, 16, 32, 32]
    z = torch.zeros(*shape, device=device)
    v = torch.randn(*shape, device=device)

    esnn_layer = CompiledMomentLayer(
        max_order=6,
        orders=orders,
        in_channels=16,
        kernel_size=11,
    ).to(device)


    for layer in [esnn_layer]:
        print(f"Checking gradients for {layer.__class__.__name__}")
        check_grad_safety(layer, z)
        check_grad_safety(layer, v)

    iters = 1000
    for layer in [esnn_layer]:
        for _ in range(20):
            _ = layer(v)

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(iters):
            _ = layer(v)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        print(layer.__class__.__name__, (t1 - t0) / iters)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(iters):
            _ = esnn_layer(v)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

def invariants() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    orders = [0, 1, 2, 3, 4, 5, 6]
    shape = [64, 16, 13, 32, 32]
    z = torch.zeros(*shape, device=device)
    v = torch.randn(*shape, device=device)

    real_layer = CompiledInvariantLayer(
        orders=orders,
        phase_function="real",
        magnitude_function="roxanas",
    ).to(device)

    polar_layer = CompiledInvariantLayer(
        orders=orders,
        phase_function="polar",
        magnitude_function="roxanas",
    ).to(device)

    for layer in [real_layer, polar_layer]:
        print(f"Checking gradients for {layer.__class__.__name__} ({layer.phase_function.__name__})")
        check_grad_safety(layer, z)
        check_grad_safety(layer, v)

    torch.testing.assert_close(
        real_layer(v),
        polar_layer(v),
        rtol=1e-5,
        atol=1e-2,
    )

    iters = 1000
    for layer in [real_layer, polar_layer]:
        for _ in range(20):
            _ = layer(v)

        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(iters):
            _ = layer(v)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        print(layer.phase_function.__name__, (t1 - t0) / iters)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(iters):
            _ = real_layer(v)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

if __name__ == "__main__":
    moments()
    #invariants()
