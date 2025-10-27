import torch
import numpy as np
from typing import List
from einops import rearrange
from scipy.linalg import dft
import torch.nn.functional as F
from scipy.special import legendre

from hippy2d.flexibleconv2d import complex_power_moivre
from hippy2d.utils import get_default_complex, tukey_2d



# Complex monomials 
def phase_part(m, size=15): 
    """ Create a angular part with phase m. 
    Args:
        m (int): angular frequency 
        size (int): size of the filter 
    Returns:
        angular part as a 2D numpy array 
    """
    y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    angles = np.arctan2(y, x)
    angular_part = np.exp(1j * m * angles)
    return angular_part

def monomial_basis(r, size=15, masking="circ"):
    y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    R = np.hypot(x, y)
    # Define on circle of radius 1
    radial = R**r
    if masking == "circ":
        radial[R > 1] = 0
    elif masking == "tukey": 
        tukey_window = tukey_2d(size, alpha=0.5)
        radial = radial * tukey_window
    return radial

def legendre_basis(r, size=15, masking="circ"):
    """ Create a radial part with Legendre polynomial of degree r. 
    Args:
        r (int): radial degree 
        size (int): size of the filter 
    Returns:
        radial part as a 2D numpy array 
    """
    y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    R = np.hypot(x, y)
    P_r = legendre(r)
    radial = P_r(R)
    if masking == "circ":
        radial[R > 1] = 0
    elif masking == "tukey": 
        tukey_window = tukey_2d(size, alpha=0.5)
        radial = radial * tukey_window
    return radial

def legendre0_basis(r, size=15, masking="circ"):
    """
    Create a radial part using a polynomial basis that vanishes at 0:
    ψ_r(R) = R * P_r(R), where P_r is the Legendre polynomial of degree r.

    Args:
        r (int): radial degree
        size (int): size of the filter
    Returns:
        radial (2D numpy array): radial part of the basis
    """
    y, x = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    R = np.hypot(x, y)
    P_r = legendre(r)
    if r != 0:
        radial = R * P_r(R)   # multiply by R to ensure it vanishes at 0
    else:
        radial = P_r(R)
    if masking == "circ":
        radial[R > 1] = 0     # zero outside the unit disk
    elif masking == "tukey": 
        tukey_window = tukey_2d(size, alpha=0.5)
        radial = radial * tukey_window
    return radial


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
                     where=(norm != 0))

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

def flusser_basis_orders(max_order: int):
    orders = []
    for _ in range(0, max_order//2 + 1):
        orders.append(0)
    for p in range(0, max_order + 1):
        for q in range(0, min(max_order + 1-p, p + 1)):
            if p - q != 0:
                orders.append(p - q)
    return sorted(orders)

class LearnableFlusser(torch.nn.Module):
    """ 
    Learnable Flusser layer as described in the notebook.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int, 
                 input_size: int = 64, # For compatible purposes
                 padding: str = "same", # "same" or "valid"
                 ring_count: int=3,
                 kernel_size: int=15,
                 orders: List[int]=None, 
                 max_order: int=4, 
                 preserve_energy: bool = False, 
                 norm_factor_function="copy"): # copy when rotating only the phase
        super(LearnableFlusser, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ring_count = ring_count
        self.kernel_size = kernel_size
        self.norm_factor_function = norm_factor_function
        self.padding = padding
        self.complex_multiplier = 2

        if orders is None:
            orders = flusser_basis_orders(max_order)
        # Assert orders are sorted
        assert orders == sorted(orders), "Orders should be sorted"
        assert all([o <= max_order for o in orders]), "Orders should be less than max_order"
        # Centro-symmetric orders
        symmetric_polynomials = 0
        for order in orders:
            if order == 0:
                symmetric_polynomials += 1
            if order != 0:
                assert order == 1, "The very first after 0 should be 1, other exponents not implemented"
                break
        # Non-symmetric orders starting with (0, 1)
        self.symmetric_polynomials = symmetric_polynomials
        self.non_symmetric_polynomials = len(orders) - symmetric_polynomials
        self.register_buffer("orders", torch.tensor(orders, dtype=torch.int32))

        self.num_invariants = self.symmetric_polynomials + (self.non_symmetric_polynomials-1)*self.complex_multiplier + 1 # Minus one because of c_01 * c_10 is real
        self.preserve_energy = preserve_energy

        # Init angular part that is fixed self.angular_part 
        # Note: This part can be also made learnable
        self.register_buffer("angular_part", init_angular_part(kernel_size, self.orders.tolist(), ring_count).to(get_default_complex()))
        # Init learnable radial part 
        self.weights = torch.nn.Parameter(init_radial_part(in_channels, in_channels, self.orders.tolist(), ring_count),
                                          requires_grad=True)
        # 1x1 real projection
        self.conv1x1 = torch.nn.Conv2d(in_channels=self.num_invariants * in_channels,
                                       out_channels=out_channels,
                                       kernel_size=1)


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
            oc=self.in_channels)
        # Perform convolution 
        if self.preserve_energy:
            filters = filters / torch.sum(filters.abs(), dim=[-2, -1], keepdim=True)
        x = F.conv2d(input=x,
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


class VarLearnableFlusser(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int, 
                 input_size: int = 64, # For compatible purposes
                 padding: str = "same", # "same" or "valid"
                 kernel_size: int=15,
                 radial_order: int=3,
                 radial_basis:str = "monomial", 
                 phase_orders: List[int]=[0, 1, 2, 3], 
                 preserve_energy: bool = False, 
                 norm_factor_function="copy", 
                 radial_masking="circ"): # copy when rotating only the phase
        
        super(VarLearnableFlusser, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.preserve_energy = preserve_energy
        self.norm_factor_function = norm_factor_function
        self.radial_masking = radial_masking

        # Assert orders are sorted
        assert phase_orders == sorted(phase_orders), "Orders should be sorted"
        # Centro-symmetric orders
        symmetric_polynomials = 0
        for order in phase_orders:
            if order == 0:
                symmetric_polynomials += 1
            if order != 0:
                assert order == 1, "The very first after 0 should be 1, other exponents not implemented"
                break
        # Non-symmetric orders starting with (0, 1)
        self.symmetric_polynomials = symmetric_polynomials
        self.non_symmetric_polynomials = len(phase_orders) - symmetric_polynomials
        self.register_buffer("orders", torch.tensor(phase_orders, dtype=torch.int32))

        self.num_invariants = self.symmetric_polynomials + (self.non_symmetric_polynomials-1)*2 + 1 # Minus one because of c_01 * c_10 is real
        
        if radial_basis == "legendre0":
            _radial_func = legendre0_basis
        elif radial_basis == "legendre":
            _radial_func = legendre_basis
        elif radial_basis == "monomial":
            _radial_func = monomial_basis
        else: 
            raise ValueError(f"Unknown radial basis: {radial_basis}")

        # Prepare fixed bases
        radial_basis = np.array([_radial_func(r, size=kernel_size, masking=self.radial_masking) for r in range(radial_order)])[np.newaxis, np.newaxis, np.newaxis, ...]
        phase_basis = np.array([phase_part(m, size=kernel_size) for m in  phase_orders])[:, np.newaxis, np.newaxis, np.newaxis, ...]
        # Merge basis
        basis = torch.from_numpy(phase_basis * radial_basis).to(get_default_complex())
        # TODO: This should be different for different phase
        # Mask zeros
        basis[symmetric_polynomials:, :, :, :, kernel_size//2, kernel_size//2] = 0.0
        # TODO: This part can be shared 
        self.register_buffer("basis", basis)

        # Different weight for each [angular basis x out_channels x in_channels, radial basis]
        weights = torch.randn([len(phase_orders), out_channels, in_channels, radial_order, 1, 1], dtype=torch.float32)
        self.weights = torch.nn.Parameter(weights)
        # 1x1 real projection
        self.conv1x1 = torch.nn.Conv2d(in_channels=self.num_invariants * out_channels,
                                       out_channels=out_channels,
                                       kernel_size=1)
        
    def forward(self, x):
        # Calculate the weights
        # TODO: Fix the complex dtype by replacing the dtype 
        x = x.to(get_default_complex())
        filters = rearrange(torch.sum(self.basis * self.weights, dim=3), 
                            "m o i h w -> (m o) i h w")
        # Perform convolution 
        if self.preserve_energy:
            filters = filters / torch.sum(filters.abs(), dim=[-2, -1], keepdim=True)
            
        x = F.conv2d(input=x,
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