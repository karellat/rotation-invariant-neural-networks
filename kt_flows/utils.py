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
    title: str = "2D point cloud",
    ax: Optional[Any] = None,
    color_map: Optional[str] = "tab10",
    marker='o',
    linestyle=':',
    connect_points=True,
    alpha=0.7,
) -> Any:
    """Plot the original points, and optionally a rotated copy, in 2D."""
    assert points.ndim == 3 and points.shape[-1] == 2, "Expected points with shape [B, N, 2]"
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap(color_map) if color_map else plt.get_cmap("tab10")
    for i in range(points.shape[0]):
        color = cmap(i % cmap.N)
        pts = points[i]
        ax.scatter(pts[..., 0], pts[..., 1], color=color, marker=marker)
        if connect_points:
            for j in range(pts.shape[0]):
                    ax.plot([pts[j, 0], pts[(j + 1) % pts.shape[0], 0]],
                            [pts[j, 1], pts[(j + 1) % pts.shape[0], 1]], 
                            color=color, alpha=alpha, linestyle=linestyle)
    ax.set_title(title)
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