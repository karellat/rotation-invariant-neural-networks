from __future__ import annotations

import torch 
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Optional
from einops import rearrange

# Utility functions for testing and visualization
def rotate_points_2d(points: torch.Tensor, angle: torch.Tensor | float) -> torch.Tensor:
    """Rotate points with shape [..., N, 2] by angle in radians around the origin."""
    if points.ndim < 2:
        raise ValueError(f"Expected points with shape [..., N, 2], got {tuple(points.shape)}")
    if points.shape[-1] != 2:
        raise ValueError(f"Expected last dimension to be 2, got {points.shape[-1]}")

    angle_tensor = torch.as_tensor(angle, dtype=points.dtype, device=points.device)
    cos_theta = torch.cos(angle_tensor)
    sin_theta = torch.sin(angle_tensor)

    rotation = torch.stack(
        (
            torch.stack((cos_theta, -sin_theta), dim=-1),
            torch.stack((sin_theta, cos_theta), dim=-1),
        ),
        dim=-2,
    )
    return torch.matmul(points, rotation.transpose(-1, -2))

def plot_points_2d(
    points: torch.Tensor,
    *,
    rotated_points: Optional[torch.Tensor] = None,
    title: str = "2D point cloud",
    ax: Optional[Any] = None,
) -> Any:
    """Plot the original points, and optionally a rotated copy, in 2D."""
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(f"Expected points with shape [N, 2], got {tuple(points.shape)}")


    points_np = points.detach().cpu()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(points_np[:, 0], points_np[:, 1], label="original", alpha=0.85)
    for start_idx in range(points_np.shape[0]):
        for end_idx in range(start_idx + 1, points_np.shape[0]):
            ax.plot(
                [points_np[start_idx, 0], points_np[end_idx, 0]],
                [points_np[start_idx, 1], points_np[end_idx, 1]],
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
            )

    if rotated_points is not None:
        if rotated_points.ndim != 2 or rotated_points.shape[-1] != 2:
            raise ValueError(
                f"Expected rotated_points with shape [N, 2], got {tuple(rotated_points.shape)}"
            )
        rotated_np = rotated_points.detach().cpu()
        ax.scatter(rotated_np[:, 0], rotated_np[:, 1], label="rotated", alpha=0.85)
        for start_idx in range(rotated_np.shape[0]):
            for end_idx in range(start_idx + 1, rotated_np.shape[0]):
                ax.plot(
                    [rotated_np[start_idx, 0], rotated_np[end_idx, 0]],
                    [rotated_np[start_idx, 1], rotated_np[end_idx, 1]],
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                )
        ax.legend()

    ax.axhline(0.0, color="0.8", linewidth=1.0)
    ax.axvline(0.0, color="0.8", linewidth=1.0)
    # TODO: Assert all the points are within the limits and adjust if necessary.
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax

def wrap_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))

def sample_points(batch_size,
                  n_points,
                  dimension,
                  centered=True,
                  dtype=None):
    # Sample [B, N, D] points from a standard normal distribution for testing.
    # B - batch size, N - number of points, D - dimension
    if dtype is None:
        dtype = torch.get_default_dtype()
    b, n, d = batch_size, n_points, dimension
    mean = np.zeros(d)
    cov = np.eye(d)

    samples = np.random.multivariate_normal(mean, cov, size=(b, n))
    samples = rearrange(torch.tensor(samples, dtype=dtype), 'b n d -> b n d')
    if centered:
        samples = samples - samples.mean(dim=-2, keepdim=True)
    return samples