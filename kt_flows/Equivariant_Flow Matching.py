import torch
import torch.nn as nn
import torch.optim as optim
from Models import MLP, MLP1, MLP3
from distributions import sample_gmm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



device = "cpu"




import numpy as np
from complex_moments_2d import CostConfig, CostFunction
from utils import rotate_points_2d

# ------------------------------------------
# Invariant matching setup
# ------------------------------------------
cost_config = CostConfig(
    max_frequency=4,
    moment_module="FlusserMoments",
    invariant_module="FlexibleInvariants",
    alignment_module="KabschAlignment",
    centered_samples=True,
    norm_volume=True,
)

cost_fn = CostFunction(cost_config)

def flat_to_points(samples):
    return samples.reshape(samples.shape[0], 2, 3).transpose(1, 2)

def points_to_flat(points):
    return points.transpose(1, 2).reshape(points.shape[0], 6)

def center_samples(samples):
    points = flat_to_points(samples)
    points = points - points.mean(dim=1, keepdim=True)
    return points_to_flat(points)

def match_batch(source_batch, target_batch, cost_fn):
    # source_batch: [B, 6]
    # target_batch: [B, 6]

    source_pts = flat_to_points(source_batch)
    target_pts = flat_to_points(target_batch)

    _, matched_pts, g = cost_fn(noise=target_pts, samples=source_pts)

    matched_flat = points_to_flat(matched_pts)
    return matched_flat, g


def batchwise_noising(X0, cost_fn=None):
    B, d = X0.shape
    device = X0.device
    dtype = X0.dtype
    X0 = center_samples(X0)

    t = torch.rand(B, device=device, dtype=dtype).view(B, 1)
    X_noise = center_samples(torch.randn(B, d, device=device, dtype=dtype))

    if cost_fn is not None:
        X0_pts = flat_to_points(X0)
        X_noise_pts = flat_to_points(X_noise)

        _, matched_X0_pts, _ = cost_fn(noise=X_noise_pts, samples=X0_pts)
        X0_matched = points_to_flat(matched_X0_pts)
    else:
        X0_matched = X0

    Xt = (1.0 - t) * X0_matched + t * X_noise
    vel = X_noise - X0_matched

    return Xt, t, vel


def rotate_2x3_samples(samples):
    """Apply an independent uniformly random 2D rotation to each 2x3 sample."""
    if samples.ndim != 2 or samples.shape[1] != 6:
        raise ValueError(f"Expected samples with shape [N, 6], got {tuple(samples.shape)}")

    num_samples = samples.shape[0]
    points = samples.reshape(num_samples, 2, 3)
    angles = 2 * torch.pi * torch.rand(
        num_samples,
        device=samples.device,
        dtype=samples.dtype,
    )
    cos_theta = torch.cos(angles)
    sin_theta = torch.sin(angles)

    rotations = torch.stack(
        (
            cos_theta,
            -sin_theta,
            sin_theta,
            cos_theta,
        ),
        dim=1,
    ).reshape(num_samples, 2, 2)

    rotated_points = torch.bmm(rotations, points)
    return rotated_points.reshape(num_samples, 6)


# -----------------------
# Data
# -----------------------
d = 6
Ntrain = 1000 # Number of training samples
torch.manual_seed(10)
weights = torch.ones(5)
means = 4*torch.rand((weights.shape[0], d), device=device)-3
stds = torch.rand(weights.shape[0], device=device) * 0 + 0.05
Xdata = sample_gmm(Ntrain, weights, means, stds)
Xdata = rotate_2x3_samples(Xdata)
#Xdata = sample_two_moons(Ntrain,  noise = 0.01, device="cpu", dtype=torch.float32)
Xplt = Xdata.clone()
points = Xplt.reshape(-1, 2, 3)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for point_index, ax in enumerate(axes):
    ax.scatter(
        points[:, 0, point_index].detach().numpy(),
        points[:, 1, point_index].detach().numpy(),
        s=8,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Point {point_index + 1}")
plt.show(block=False)
plt.pause(0.1)
plt.close(fig)

# -----------------------
# Train
# -----------------------
lr = 0.0001
model = MLP3(input_dim=d + 1, hidden_dim=256, outputdim=d).to(device)
opt = optim.Adam(model.parameters(), lr=lr)

loader = DataLoader(Xdata, batch_size=128, shuffle=True, num_workers=0)

epochs = 1000
L = 1

loss_fn = nn.MSELoss()
for ep in range(epochs):
    model.train()
    running = 0.0
    ep_batch = 0
    for batch in loader:
        for nc in range(1):
            Xt, t, vel = batchwise_noising(batch, cost_fn)
            B = batch.shape[0]
            
            inputs = torch.cat([t.view(B, 1), Xt], dim=1).to(device)
            targets = vel.to(device)
    
            preds = model(inputs)
            loss = loss_fn(preds, targets)
    
            opt.zero_grad()
            loss.backward()
            opt.step()
    
            running += loss.item() * inputs.shape[0]
            ep_batch += inputs.shape[0]
    if (ep + 1) % 1000 == 0:
        print(f"Epoch {ep+1}/{epochs}  Loss: {running/ep_batch:.6f}", flush=True)
#%%
# # -----------------------
# # Sample (reverse integrate)
# # -----------------------
# Ntest = 1000
# dtype = torch.float32
# N_timegrid = 1000
# d = 2

# X_noise = torch.randn(Ntest, d, device=device, dtype=dtype)

# model = model.to(device).to(dtype)


# def revsample(X_noise, model, N_timegrid):
#     # minimal fixes:
#     #  - don't depend on global Ntest inside
#     #  - feed the *current* state into the model (Zt), not a stale copy (Z)
#     #  - disable autograd during sampling
#     model.eval()
#     dt = 1.0 / (L * N_timegrid)

#     Zt = X_noise.clone()
#     B = Zt.shape[0]

#     with torch.no_grad():
#         tim = torch.arange(1, 0, -dt, device=Zt.device, dtype=Zt.dtype)
#         for t in tim:
#           t0 = t.expand(B, 1)                 # current time (backward)
#           h  = dt

#     # k1 = f(t0, Zt)
#           v1 = model(torch.cat([t0, Zt], dim=1))

#     # midpoint state/time (since we're stepping backward: t_mid = t0 - h/2)
#           Z_mid = Zt - (h / 2.0) * v1
#           t_mid = t0 - (h / 2.0)

#     # k2 = f(t_mid, Z_mid)
#           v2 = model(torch.cat([t_mid, Z_mid], dim=1))

#     # RK2 update backward
#           Zt = Zt - h * v2

#     return Zt


# Z = revsample(X_noise, model, N_timegrid).clone()

#           plt.plot(Zt[:, 0].detach().numpy(), Z[:, 1].detach().numpy(), ".")
#%%

import torch
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# -----------------------
# Sample (reverse integrate) + GIF
# -----------------------
Ntest = 1000

dtype = torch.float32
N_timegrid = 1000


X_noise = center_samples(torch.randn(Ntest, d, device=device, dtype=dtype))
model = model.to(device).to(dtype)
model.eval()
model.requires_grad_(False)


def revsample_with_gif(X_noise, model, N_timegrid, gif_path="reverse_process.gif", frame_every=10):
    """
    Reverse-sample with RK2 and save a GIF of the sampled process.
    """
    dt = 1.0 / (L * N_timegrid)

    Zt = X_noise.clone()
    B = Zt.shape[0]

    os.makedirs("gif_frames_sample", exist_ok=True)
    frames = []
    frame_idx = 0

    with torch.no_grad():
        tim = torch.arange(1, 0, -dt, device=Zt.device, dtype=Zt.dtype)

        for t in tim:
            t0 = t.expand(B, 1)   # current time (backward)
            h = dt

            # k1 = f(t0, Zt)
            v1 = model(torch.cat([t0, Zt], dim=1))

            # midpoint state/time
            Z_mid = Zt - 0.5 * h * v1
            t_mid = t0 - 0.5 * h

            # k2 = f(t_mid, Z_mid)
            v2 = model(torch.cat([t_mid, Z_mid], dim=1))

            # RK2 update backward
            Zt = Zt - h * v2

            if frame_idx % frame_every == 0:
                fig, ax = plt.subplots(figsize=(5, 5))

                if "Xplt" in globals():
                    data_points = flat_to_points(center_samples(Xplt))
                    sample_points = flat_to_points(Zt)
                    for point_index in range(3):
                        ax.scatter(
                            data_points[:, point_index, 0].detach().cpu().numpy(),
                            data_points[:, point_index, 1].detach().cpu().numpy(),
                            s=8,
                            alpha=0.35,
                            label="data" if point_index == 0 else None,
                        )
                        ax.scatter(
                            sample_points[:, point_index, 0].detach().cpu().numpy(),
                            sample_points[:, point_index, 1].detach().cpu().numpy(),
                            s=8,
                            alpha=0.8,
                            label="samples" if point_index == 0 else None,
                        )

                ax.set_aspect("equal", adjustable="box")
                ax.autoscale()
                ax.set_title(f"L={L}")
                ax.legend(loc="upper right")

                fname = f"gif_frames_sample/frame_{frame_idx:05d}.png"
                plt.savefig(fname, bbox_inches="tight")
                plt.close(fig)
                frames.append(fname)

            frame_idx += 1

    with imageio.get_writer(gif_path, mode="I", duration=0.08 / L, loop=1) as writer:
        for fname in frames:
            writer.append_data(imageio.imread(fname))

    return Zt


with torch.no_grad():
    Z = revsample_with_gif(
        X_noise,
        model,
        N_timegrid,
        gif_path=f"reverse_process_usualFM_sample_2M_model0.gif",
        frame_every=10,
    ).clone()
    
#%%

    
    