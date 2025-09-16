import os
import ast
import click 
import torch
import numpy as np
from PIL import Image
import multiprocessing

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

def get_testing_img(rgb: bool = False) -> Image: 
    # Los Alamos National Laboratory, Attribution, via Wikimedia Common
    if rgb:
        img = Image.open(os.path.join(os.path.dirname(__file__), "test_rgb.png"))
    else: 
        img = Image.open(os.path.join(os.path.dirname(__file__), "test_img.jpg"))
    # Get central crop 256x256
    img = img.crop((img.width // 2 - 128, img.height // 2 - 128, img.width // 2 + 128, img.height // 2 + 128))
    return img

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
