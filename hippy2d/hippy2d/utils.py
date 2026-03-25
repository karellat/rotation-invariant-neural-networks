import os
import ast
import math
import click 
import torch
import colorsys
import numpy as np
from PIL import Image
import multiprocessing



import torch
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_lightning.utilities import rank_zero_only
from typing import Mapping

# Optimal number of workers
def get_optimal_workers():
    """Calculate optimal number of workers for DataLoader"""
    cpu_count = multiprocessing.cpu_count()
    
    # General rule: use 2-4 workers per GPU, or cpu_count for CPU-only
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        return min(cpu_count, gpu_count * 4)
    else:
        return min(cpu_count, 8)  # Cap at 8 for CPU-only training

def get_default_complex():
    """
    Default precision of complex numbers according to default dtype
    :return: returns torch.cfloat or torch.cdouble
    """
    assert (torch.get_default_dtype() is torch.float32) or (torch.get_default_dtype() is torch.float64)
    return torch.cfloat if torch.get_default_dtype() is torch.float32 else torch.cdouble

class ClickDictionaryType(click.ParamType):
    name = "dict"

    def convert(self, value, param, ctx):
        if isinstance(value, dict):
            return value
        try:
            return ast.literal_eval(value)
        except ValueError:
            self.fail(f"{value!r} is not a valid dictionary", param, ctx)

def tukey_2d(shape, alpha=0.5):
    """
    2D Tukey window based on https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.tukey.html
    :param shape: shape of the window
    :param alpha: alpha parameter of the Tukey window
    :return:
    """
    #assert shape % 2 == 0, "Only for even shapes"
    center = ((shape - 1) / 2), ((shape - 1) / 2)
    y, x = np.ogrid[:shape, :shape]
    width = int(np.floor(alpha*(shape-1)/2.0))
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    dist_from_center *= 2

    identity_mask = dist_from_center <= shape - width
    non_identity_mask = (dist_from_center > shape - width) & (dist_from_center <= shape)
    zero_mask = dist_from_center > shape
    tukey = -np.ones((shape, shape), dtype=np.float64)
    tukey[identity_mask] = 1.0
    tukey[zero_mask] = 0.0
    tukey[non_identity_mask] = 0.5 * (1 + np.cos(np.pi * (-2.0/alpha + 1 + 2.0*dist_from_center[non_identity_mask]/alpha/(shape-1))))
    return tukey

def radial_tukey_from_R(R, alpha=0.5, r_max=1.0, dtype=np.float32):
    R = np.asarray(R, dtype=dtype)
    W = np.zeros_like(R, dtype=dtype)

    if alpha <= 0:
        W[R <= r_max] = 1
        return W

    r0 = r_max * max(0.0, 1.0 - alpha)  # start of taper (plateau radius)

    # plateau (skip when alpha == 1 so we don't miss the center)
    if alpha < 1:
        W[R <= r0] = 1

    # cosine taper: from r0 to r_max
    mask = (R >= r0) & (R <= r_max)
    t = (R[mask] - r0) / (r_max * alpha)          # goes 0 -> 1 across the taper
    W[mask] = 0.5 * (1 + np.cos(np.pi * t))       # 1 -> 0 over the taper

    # outside r_max stays zero
    return W

def fixed_tukey(size:int, alpha): 
    size += 2
    assert (alpha >= 0.1) and (alpha <= 1.0), "Non-supported"
    y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    R = np.hypot(x, y)              # 0 at center, ~1.414 at corners
    W = radial_tukey_from_R(R, alpha=alpha, r_max=1.0)  # taper to 0 at radius 1
    return W[1:-1, 1:-1]

def get_testing_img(rgb: bool = False) -> Image: 
    # Los Alamos National Laboratory, Attribution, via Wikimedia Common
    if rgb:
        img = Image.open(os.path.join(os.path.dirname(__file__), "test_rgb.png"))
    else: 
        img = Image.open(os.path.join(os.path.dirname(__file__), "test_img.jpg"))
    # Get central crop 256x256
    img = img.crop((img.width // 2 - 128, img.height // 2 - 128, img.width // 2 + 128, img.height // 2 + 128))
    return img

def rasterize_point_configurations(
    configurations: torch.Tensor,
    grid_size: int | tuple[int, int],
    sigma: float,
    amplitude: float = 1.0,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Rasterize batched 2D point configurations onto a grid with Gaussian blobs.

    Parameters
    ----------
    configurations:
        Tensor of shape ``[N, A, 2]`` containing ``(x, y)`` point coordinates.
        Coordinates are interpreted in a Cartesian system centered at the grid
        origin, matching the convention used in ``make_blurred_atom_image``.
    grid_size:
        Either a single integer for a square grid or ``(height, width)``.
    sigma:
        Standard deviation of each Gaussian in pixel units.
    amplitude:
        Peak value of each Gaussian before summing.
    normalize:
        If ``True``, divide each Gaussian by ``2 * pi * sigma^2`` so every point
        contributes unit integral instead of unit peak.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``[N, H, W]`` containing the rasterized images.
    """
    if configurations.ndim != 3 or configurations.shape[-1] != 2:
        raise ValueError(
            f"configurations must have shape [N, A, 2], got {tuple(configurations.shape)}"
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    if isinstance(grid_size, int):
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        height = width = grid_size
    else:
        if len(grid_size) != 2:
            raise ValueError(f"grid_size must be an int or a pair, got {grid_size}")
        height, width = grid_size
        if height <= 0 or width <= 0:
            raise ValueError(f"grid dimensions must be positive, got {grid_size}")

    dtype = configurations.dtype
    device = configurations.device

    y_coords = torch.arange(height, dtype=dtype, device=device) - (height // 2)
    x_coords = torch.arange(width, dtype=dtype, device=device) - (width // 2)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    points_x = configurations[..., 0][:, :, None, None]
    points_y = configurations[..., 1][:, :, None, None]
    dist2 = (xx[None, None, :, :] - points_x) ** 2 + (yy[None, None, :, :] - points_y) ** 2

    images = amplitude * torch.exp(-dist2 / (2.0 * sigma ** 2))
    if normalize:
        images = images / (2.0 * math.pi * sigma ** 2)

    return images.sum(dim=1)

def get_circular_mask(shape: int, radius:int = None, dtype=torch.float64) -> torch.Tensor:
    """
    Create a circular mask of given shape and radius.
    
    Parameters:
    shape (int): Size of the square mask (shape x shape).
    radius (int): Radius of the circle.
    dtype (torch.dtype): Data type for the tensor.
    
    Returns:
    torch.Tensor: Circular mask tensor.
    """
    if radius is None:
        radius = (shape / 2)
    y = torch.linspace(-1, 1, shape, dtype=dtype)  # shape (H,)
    x = torch.linspace(-1, 1, shape, dtype=dtype)  # shape (W,)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # shape (H, W)
    
    mask = (xx**2 + yy**2 <= (radius / (shape / 2))**2).bool()
    return mask

def retrieve_elements_from_indices(tensor: torch.Tensor, indices: torch.Tensor):
    """
    Taken from https://github.com/wavefrontshaping/complexPyTorch/blob/2044cb077b3f139d59dff56abc378b1457de40d6/complexPyTorch/complexFunctions.py#L88
    Retrieve elements from tensor using indices
    :param tensor: input tensor
    :param indices: indices to be retrieved
    :return:
    """
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(
        dim=-1, index=indices.flatten(start_dim=-2)
    ).view_as(indices) 
    return output

class SafeAtan2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x, eps=1e-12):
        ctx.save_for_backward(y, x)
        ctx.eps = eps
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, grad_output):
        y, x = ctx.saved_tensors
        denom = x*x + y*y + ctx.eps
        # d/dx = -y/(x^2+y^2+eps)
        # d/dy =  x/(x^2+y^2+eps)
        grad_x = -y / denom * grad_output
        grad_y =  x / denom * grad_output
        return grad_y, grad_x, None


def complex_to_rgb(Z):
    mag = np.abs(Z)
    phase = np.angle(Z)
    norm_mag = mag / mag.max()
    
    hsv = np.zeros(Z.shape + (3,))
    hsv[..., 0] = (phase + np.pi) / (2*np.pi)   # hue
    hsv[..., 1] = 1.0                           # saturation
    hsv[..., 2] = norm_mag                      # value
    
    rgb = np.zeros_like(hsv)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            rgb[i,j] = colorsys.hsv_to_rgb(*hsv[i,j])
    return rgb
