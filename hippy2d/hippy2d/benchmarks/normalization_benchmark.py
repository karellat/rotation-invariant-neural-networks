import time

import torch
import torch
import torch.profiler

def check_grad_safety(fn, *inputs):
    inputs[0].requires_grad_(True)

    out = fn(*inputs)
    loss = out.sum()
    loss.backward()

    grads = [x.grad for x in inputs[:1]]

    for i, g in enumerate(grads):
        if g is None:
            print(f"Input {i}: ❌ no gradients")
        elif not torch.isfinite(g).all() or torch.isnan(g).any():
            print(f"Input {i}: ❌ gradients are not finite")
        else:
            pass

@torch.compile
def rotate_real_self_n(
    v: torch.Tensor,
    o: torch.Tensor,
) -> torch.Tensor:
    mag = torch.linalg.vector_norm(v, dim=-3)

    safe_r = mag.clamp(min=1e-12) 
    real, imag = v.unbind(dim=-3)
    n_m = nicks_magnitude(mag[:, :, 0:1])
    nt_r, nt_i = real[:, :, 1:], imag[:, :, 1:]

    angle = torch.atan2(imag/safe_r, real/safe_r)
    rotated_angle = angle[:, :, 0:1, ...] * o[:, None, None]
    
    n_r = n_m * torch.cos(rotated_angle)
    n_i = n_m * torch.sin(rotated_angle)
    
    return n_r * nt_r - n_i * nt_i

@torch.compile
def rotate_polar_self_n(
    v: torch.Tensor,
    o: torch.Tensor,
) -> torch.Tensor:
    """Rotate each complex 2-vector by its own angle `n` additional times."""
    mag = torch.linalg.vector_norm(v, dim=-3)
    safe_mag = mag.clamp(min=1e-12) 
    # Project to mag/phase
    real, imag = (v).unbind(dim=-3)
    angle = torch.atan2(imag/safe_mag, real/safe_mag)
    # Normalizer and rotated phase
    n_mag = nicks_magnitude(mag[:, :, 0:1])
    n_angle = angle[:, :, 0:1] * o[:, None, None]
    # Non-trivials and
    nt_mag = mag[:, :, 1:]
    nt_angle = angle[:, :, 1:]
    
    return (n_mag * nt_mag) * torch.cos(nt_angle + n_angle)


@torch.compile
def nicks_magnitude(mag):
    return (mag * mag) / (mag * mag + 1)

@torch.compile
def rotate_complex_self_n(
    v: torch.Tensor,
    o: torch.Tensor,
    n: int = 7,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Reference implementation using complex multiplication."""
    real, imag = v.unbind(dim=-3)
    c = torch.complex(real, imag)
    r = c.abs()
    safe_r = r.clamp(min=eps)

    q = c / safe_r

    out = q.pow(n)
    return torch.stack((out.real, out.imag), dim=-3)


def main() -> None:
    fns =  [rotate_real_self_n, rotate_polar_self_n]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Check grad safety
    shape =  [256, 32, 8, 2, 64, 64]
    o = torch.tensor([1, 1, 2, 3, 4, 5, 6], dtype=torch.int, device=device)
    z = torch.zeros(*shape, device=device)
    v = torch.randn(*shape, device=device)
    iter = 5000
    for fn in fns:
        print(f"Checking gradients for {fn.__name__}...")
        check_grad_safety(fn, z, o)
        check_grad_safety(fn, v, o)
        print()

    torch.testing.assert_close(rotate_real_self_n(v, o), rotate_polar_self_n(v, o), rtol=1e-5, atol=1)


    for fn in fns:
        for _ in range(100):
            _ = fn(v, o)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iter):
            _ = fn(v, o)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        print(fn.__name__, (t1 - t0) / iter)


    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(iter):
            _ = rotate_real_self_n(v, o)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

if __name__ == "__main__":
    main()
