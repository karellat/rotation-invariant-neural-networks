import torch
import numpy as np
from einops import rearrange
from hippy2d.flexibleconv2d import complex_power_moivre
from typing import List
from hippy2d.utils import get_default_complex
from scipy.linalg import dft

def init_radial_part(in_channels: int, out_channels: int, orders: List[int], ring_count: int):
    """
    Initializes the dict of weights

    Args:

        in_channels: input channels
        out_channels: output channels
        max_order: maximum rotation order from input and output
        ring_count (int): Number of rings for calculating the basis
    Returns:
        weights_dict: initialized dict of weights

    """
    weights = torch.zeros((len(orders), ring_count, in_channels, out_channels),
                            dtype=torch.get_default_dtype())
    for idx, _ in enumerate(orders):
        sh = [ring_count, in_channels, out_channels]
        # Initializes the weights using initialization method of He
        stddev = 0.4 * np.sqrt(2.0 / np.prod(sh[:3]))
        weights[idx] = torch.normal(torch.zeros(*sh), stddev)
    # Flat input & output channels
    return weights.view((len(orders), ring_count, in_channels * out_channels))

def get_angle_samples_count(kernel_size):
    return int(np.maximum(np.ceil(np.pi * kernel_size), 101))

def get_l2_neighbors(center, shape):
    lin = np.arange(shape) + 0.5
    jj, ii = np.meshgrid(lin, lin)
    ii = ii - center[1]
    jj = jj - center[0]
    return np.vstack((np.reshape(ii, -1), np.reshape(jj, -1)))

def get_interpolation_weights(fs, m, n_rings, angle_samples):
    """
    Used to construct the steerable filters using Radial basis functions.
    The filters are constructed on the patches of n_rings using Gaussian
    interpolation. (Code adapted from the tf code of Worrall et al., CVPR, 2017)

    Args:
        fs (int): filter size for the H-net convolutional layer
        m (int): max. rotation order for the steerable filters
        n_rings (int): No. of rings for the steerable filters
        angle_samples (int): No. of angle samples

    Returns:
        norm_weights (numpy): contains normalized weights for interpolation
        using the steerable filters
    """

    mid = int(np.floor(fs / 2))
    # Variance of Gaussian resampling
    std_gauss = (mid / n_rings) / 2
    # We define below radii up to n_rings-0.5 (as in Worrall et al, CVPR 2017)
    radii = np.linspace(m != 0, mid - 2 * std_gauss, n_rings)
    # We define pixel centers to be at positions 0.5
    center_pt = np.asarray([fs, fs]) / 2.

    # Extracting the set of angles to be sampled

    # Choosing the sampling locations for the rings
    lin = (2 * np.pi * np.arange(angle_samples)) / angle_samples
    ring_locations = np.vstack([-np.sin(lin), np.cos(lin)])

    # Create interpolation coefficient coordinates
    coords = get_l2_neighbors(center_pt, fs)

    # getting samples based on the chosen center_pt and the coords
    radii = radii[:, np.newaxis, np.newaxis, np.newaxis]
    ring_locations = ring_locations[np.newaxis, :, :, np.newaxis]
    diff = radii * ring_locations - coords[np.newaxis, :, np.newaxis, :]
    dist2 = np.sum(diff ** 2, axis=1)

    # Convert distances to weightings
    weights = np.exp(-0.5 * dist2 / (std_gauss ** 2))  # For bandwidth of 0.5

    # Normalizing the weights to calibrate the different steerable filters
    norm = np.sum(weights,
                    axis=2,
                    keepdims=True)
    assert np.all(norm != 0), "Normalizing by zero weights"
    return np.divide(weights,
                        norm,
                        where=(norm != 0)
                        )

def init_angular_part(kernel_size, orders, n_rings):
    N = get_angle_samples_count(kernel_size)
    weights2filter_sampler = []
    for order in orders:
        weights = get_interpolation_weights(kernel_size,
                                            m=order,
                                            n_rings=n_rings,
                                            angle_samples=N)
        low_pass_filter = np.dot(dft(N)[order, :], weights).T
        weights2filter_sampler.append(
                torch.from_numpy(low_pass_filter))
    return torch.stack(weights2filter_sampler)


class LearnableFlusser(torch.nn.Module):
    """ 
    Learnable Flusser layer as described in the notebook.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int, 
                 input_size: int = 64, # For compatible purposes
                 padding: str = "same", # "same" or "valid"
                 max_order: int=4,
                 ring_count: int=3,
                 kernel_size: int=15,
                 norm_factor_function="copy"): # copy when rotating only the phase
        super(LearnableFlusser, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_order = max_order
        self.ring_count = ring_count
        self.kernel_size = kernel_size
        self.norm_factor_function = norm_factor_function
        self.padding = padding
        self.complex_multiplier = 2

        orders = []
        # Centro-symmetric orders 
        symmetric_polynomials = 0
        for p in range(0, max_order//2 + 1):
            orders.append(0)
            symmetric_polynomials += 1
        non_symmetric_polynomials = 0
        # Non-symmetric orders starting with (0, 1)
        for p in range(0, max_order + 1):
            for q in range(0, min(max_order + 1-p, p + 1)):
                if p - q != 0:
                    orders.append(p - q)
                    non_symmetric_polynomials += 1
        self.symmetric_polynomials = symmetric_polynomials
        self.non_symmetric_polynomials = non_symmetric_polynomials
        self.register_buffer("orders", torch.tensor(orders, dtype=torch.int32))

        self.num_invariants = symmetric_polynomials + (non_symmetric_polynomials-1)*self.complex_multiplier + 1 # Minus one because of c_01 * c_10 is real
        # Init angular part that is fixed self.angular_part 
        # Note: This part can be also made learnable
        self.register_buffer("angular_part", init_angular_part(kernel_size, self.orders.tolist(), ring_count).to(get_default_complex()))
        # Init learnable radial part 
        self.weights = torch.nn.Parameter(init_radial_part(in_channels, out_channels, self.orders.tolist(), ring_count),
                                          requires_grad=True)
        # 1x1 real projection
        self.conv1x1 = torch.nn.Conv2d(in_channels=self.num_invariants * out_channels,
                                       out_channels=out_channels,
                                       kernel_size=1,
                                       dtype=torch.get_default_dtype())


    def forward(self, x):
        # Calculate the weights
        # TODO: Fix the complex dtype by replacing the dtype 
        x = x.to(get_default_complex())
        filters = rearrange(
            tensor=torch.matmul(self.angular_part, self.weights.to(get_default_complex())),
            pattern="o (h w) (ic oc) -> (o oc) ic h w",
            h=self.kernel_size,
            w=self.kernel_size,
            ic=self.in_channels,
            oc=self.out_channels)
        # Perform convolution 
        x = torch.conv2d(input=x,
                         weight=filters, 
                         padding=self.padding)
        x = rearrange(x, 'b (m out) h w -> b m out h w', m=len(self.orders))
        # symmetrics 
        symmetric = x[:, :self.symmetric_polynomials].real
        # moments 
        moments = x[:, self.symmetric_polynomials:]
        norm_factor = moments[:, 0:1].conj()
        # Compute invariants
        nonsymmetric = moments * torch.view_as_complex(
            complex_power_moivre(torch.view_as_real(norm_factor.resolve_conj()),
                                               self.orders[self.symmetric_polynomials:, None, None, None],
                                               magnitude_func=self.norm_factor_function)
        )
        diagonal = nonsymmetric[:, 0:1].real

        # Concatenate all parts to form the output
        x = torch.cat((symmetric, 
                       diagonal,
                       nonsymmetric[:, 1:].real,
                       nonsymmetric[:, 1:].imag), dim=1)
        x = rearrange(x, 'b m out h w -> b (m out) h w')
        x = self.conv1x1(x)
        # Return the output
        return x