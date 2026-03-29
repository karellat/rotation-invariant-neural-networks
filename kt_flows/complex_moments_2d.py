from __future__ import annotations

import ot
import math
import torch
import numpy as np
from typing import Optional
from itertools import permutations

from utils import rotate_points_2d, wrap_angle


class CostConfig:
    def __init__(self, 
                 max_frequency: int, 
                 moment_module: str,
                 invariant_module: str,
                 alignment_module: str,
                 centered_samples: bool,
                 norm_volume: bool):
        # Static
        assert max_frequency >= 0, "max_frequency must be non-negative"
        self.max_frequency = max_frequency
        self.moment_module = moment_module
        self.invariant_module = invariant_module
        self.alignment_module = alignment_module
        self.norm_volume = norm_volume
        self.centered_samples = centered_samples
        # Dynamic
        self._orders = None
        self._num_invariants = None
        self._trivial_mask = None
        self._nontrivial_mask = None
        # Optional
        self._pq = None

    @property
    def num_invariants(self):
        if self._num_invariants is None:
            raise AttributeError("Config.num_invariants has not been initialized yet")
        return self._num_invariants

    @num_invariants.setter
    def num_invariants(self, value):
        self._num_invariants = value

    @property
    def orders(self):
        if self._orders is None:
            raise AttributeError("Config.orders has not been initialized yet")
        return self._orders

    @orders.setter
    def orders(self, value):
        self._orders = value

    @property
    def pq(self):
        return self._pq

    @pq.setter
    def pq(self, value):
        self._pq = value
    
    @property
    def trivial_mask(self):
        if self._trivial_mask is None:  
            self._trivial_mask = self.orders == 0
        return self._trivial_mask

    @property
    def nontrivial_mask(self):
        if self._nontrivial_mask is None:
            self._nontrivial_mask = ~self.trivial_mask
        return self._nontrivial_mask
    
    def moment_phase_shift(self, angle: torch.Tensor):
        assert 0 <= angle < 2 * math.pi, "Angle must be in [0, 2pi)"
        # For equivariance testing
        return torch.polar(torch.ones_like(self.orders, dtype=torch.get_default_dtype()), self.orders * angle)


def complex_moment_pq(max_frequency: int, *, device: Optional[torch.device] = None) -> torch.Tensor:
    """Return all (p, q) with p >= q >= 0 and p + q <= max_frequency."""
    if max_frequency < 0:
        raise ValueError("max_frequency must be non-negative")

    orders = [
        (p, q)
        for total_degree in range(max_frequency + 1)
        for q in range(total_degree // 2 + 1)
        for p in [total_degree - q]
    ]
    return torch.tensor(orders, dtype=torch.long, device=device)

class CostFunction(torch.nn.Module):
    def __init__(self,
                 config: CostConfig):
        super().__init__()
        # Dynamically get moment and invariant classes from config
        self.config = config

        moment_cls = globals().get(self.config.moment_module)
        if moment_cls is None:
            raise ValueError(f"Unknown moment module: {self.config.moment_module}")
        invariant_cls = globals().get(self.config.invariant_module)
        if invariant_cls is None:
            raise ValueError(f"Unknown invariant module: {self.config.invariant_module}")
        alignment_cls = globals().get(self.config.alignment_module)
        if alignment_cls is None:
            raise ValueError(f"Unknown alignment module: {self.config.alignment_module}")
        # Init moment layer
        self.moments = moment_cls(self.config)
        # Adjust config w.r.t. moment layer
        self.config.orders = self.moments.orders
        if self.moments.pq is not None:
            self.config.pq = self.moments.pq
        # Init invariant layer
        self.invariants = invariant_cls(self.config)
        # Adjust config w.r.t. invariant layer
        self.config.num_invariants = self.invariants.num_invariants
        self.phase_shift = self.config.moment_phase_shift
        # Init alignment layer
        self.alignment = alignment_cls(self.config)

    def forward(self, noise: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        m_noise = self.moments(noise)
        i_noise = self.invariants(m_noise)
        m_samples = self.moments(samples)
        i_samples = self.invariants(m_samples)
        # Distance
        L2_dist = torch.cdist(i_noise, i_samples, p=2)
        sol = ot.solve(L2_dist.cpu().detach().numpy())
        closest_idx = np.argmax(sol.plan, axis=1)
        paired_samples = samples[closest_idx]
        # Estimate the group alignment between noise and aligned samples
        R_sample, Perm_sample, t_sample = self.alignment(paired_samples, noise)
        g_for_noise = (R_sample, Perm_sample, t_sample)
        return L2_dist, paired_samples, g_for_noise

    def test_points(self, points: torch.Tensor, angle: torch.Tensor | float):
        rotated_points = rotate_points_2d(points, angle)
        moments = self.moments(points)
        rotated_moments = self.moments(rotated_points)
        # Test equivariance of moments
        expected_moments = moments * self.phase_shift(angle)
        assert torch.allclose(rotated_moments, expected_moments, atol=1e-6, rtol=1e-5), "Equivariance test failed"
        # Test invariance of invariants
        invariants = self.invariants(moments)
        rotated_invariants = self.invariants(rotated_moments)
        assert torch.allclose(invariants, rotated_invariants, atol=1e-6, rtol=1e-5), "Invariance test failed"
        
# Moment Layers 
class FlusserMoments(torch.nn.Module):
    def __init__(self,
                 config: CostConfig):
        super().__init__()
        self.max_frequency = config.max_frequency
        assert config.centered_samples, "Flusser moments require centered samples for translation invariance"
        pq = complex_moment_pq(self.max_frequency).to(dtype=torch.long)
        orders = pq[:, 0] - pq[:, 1]
        self.norm_volume = config.norm_volume
        if self.norm_volume: 
            if not torch.any((pq == torch.tensor([0, 0])).all(dim=1)): 
                raise ValueError("pq must include the zero-order moment (0, 0) for volume normalization")
            else:
                self.norm_idx = torch.where((pq == torch.tensor([0, 0])).all(dim=1))[0][0]
        # Register buffers 
        self.register_buffer("pq", pq)
        self.register_buffer("orders", orders.to(dtype=torch.long))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim < 2:
            raise ValueError(f"Expected points with shape [..., N, 2], got {tuple(points.shape)}")
        if points.shape[-1] != 2:
            raise ValueError(f"Expected last dimension to be 2, got {points.shape[-1]}")
        if not torch.is_floating_point(points):
            raise TypeError("points must have a real floating-point dtype")
        z = torch.complex(points[..., 0], points[..., 1])
        z = z.unsqueeze(-1)

        p = self.pq[:, 0].to(dtype=z.real.dtype)
        q = self.pq[:, 1].to(dtype=z.real.dtype)

        moments = (z.pow(p) * z.conj().pow(q)).sum(dim=-2)
        if self.norm_volume:
            moments = moments / torch.clamp(moments[..., self.norm_idx:self.norm_idx+1].abs(), min=1e-8)
        return moments

# Invariant layers
class FlusserInvariants(torch.nn.Module):
    def __init__(self, config: CostConfig):
        super().__init__()
        # Normalization guy 
        if config.pq is not None:
            self.norm_idx = torch.where((config.pq == torch.tensor([1, 0])).all(dim=1))[0][0]
        else: 
            self.norm_idx = torch.where((self.orders == 1).all(dim=0))[0][0]
        # Exponents 
        trivial_mask = config.trivial_mask
        nontrivial_mask = config.nontrivial_mask.clone()
        nontrivial_mask[self.norm_idx] = False
        exps = -config.orders[nontrivial_mask]
        self.register_buffer("orders", config.orders)
        self.register_buffer("trivial_mask", trivial_mask)
        self.register_buffer("nontrivial_mask", nontrivial_mask)
        self.register_buffer("exps", exps)
        # trivials + magnitudes + real parts 
        self.num_invariants = (trivial_mask.sum() + nontrivial_mask.sum() + (nontrivial_mask.sum())).item()

    def forward(self, moments: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(moments) and (moments.ndim != 2 or moments.shape[1] != 2):
            raise ValueError(f"Expected moments with a complex dtype or shape [M, 2], got {tuple(moments.shape)}")
        moments = moments if torch.is_complex(moments) else torch.view_as_complex(moments)
        if moments.shape[-1] != self.orders.shape[0]:
            raise ValueError(f"Expected moments with shape [M], got {tuple(moments.shape)}")
        # Remove nontrivial moments
        trivials = moments[..., self.trivial_mask].real
        nontrivials = moments[..., self.nontrivial_mask]
        norm = moments[..., self.norm_idx]

        nontrivials_mag = nontrivials.abs()
        # NOTE: This is not gradient safe
        angle = torch.atan2(norm.imag, norm.real) * self.exps
        _real = norm.abs() * torch.cos(angle)
        _imag = norm.abs() * torch.sin(angle)
        real_invariants = _real * nontrivials.real - _imag * nontrivials.imag
        # TODO: Magnitude of the normalizer? 
        invariants = torch.concat([trivials, nontrivials_mag, real_invariants], dim=-1)
        return invariants

class FlexibleInvariants(torch.nn.Module):
    def __init__(self, config: CostConfig):
        super().__init__()   
        trivial_mask = config.trivial_mask
        nontrivial_mask = config.nontrivial_mask
        nontrivial_orders = config.orders[nontrivial_mask]
        n_non_trivial = len(nontrivial_orders)
        exp_a = torch.zeros(n_non_trivial, n_non_trivial, dtype=torch.int32)
        exp_b = torch.zeros(n_non_trivial, n_non_trivial, dtype=torch.int32)

        for idx_a, a in enumerate(nontrivial_orders):
            for idx_b, b in enumerate(nontrivial_orders):
                gcd = np.gcd(a, b)
                exp_a[idx_a, idx_b] = b // gcd
                exp_b[idx_a, idx_b] = -a // gcd
        # Many things are symmetric within the flexible basis. 
        tril_rows, tril_cols = torch.tril_indices(n_non_trivial, n_non_trivial, offset=-1)
        # Select exponents 
        exp_a = exp_a[tril_rows, tril_cols]
        exp_b = exp_b[tril_rows, tril_cols]
        self.register_buffer("orders", config.orders)
        self.register_buffer("trivial_mask", trivial_mask)
        self.register_buffer("nontrivial_mask", nontrivial_mask)
        self.register_buffer("exp_a", exp_a)
        self.register_buffer("exp_b", exp_b)
        # tril_rows and tril_cols can be used to index into the nontrivial moments
        self.register_buffer("tril_rows", tril_rows)
        self.register_buffer("tril_cols", tril_cols)
        self.num_invariants = (trivial_mask.sum() + nontrivial_mask.sum() + len(exp_a)).item()
    
    def forward(self, moments: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(moments) and (moments.ndim != 2 or moments.shape[1] != 2):
            raise ValueError(f"Expected moments with a complex dtype or shape [M, 2], got {tuple(moments.shape)}")
        moments = moments if torch.is_complex(moments) else torch.view_as_complex(moments)
        if moments.shape[-1] != self.orders.shape[0]:
            raise ValueError(f"Expected moments with shape [M], got {tuple(moments.shape)}")
        trivials = moments[..., self.trivial_mask].real
        nontrivials = moments[..., self.nontrivial_mask]
        # NOTE: This might be gradient unsafe
        mag, angles = nontrivials.abs(), nontrivials.angle()
        mag_a = torch.index_select(mag, -1, self.tril_rows) 
        mag_b = torch.index_select(mag, -1, self.tril_cols) 
        # Angles
        angle_a = torch.index_select(angles, -1, self.tril_rows) * self.exp_a 
        angle_b = torch.index_select(angles, -1, self.tril_cols) * self.exp_b
        nontrivials_real = mag_a * mag_b * torch.cos(angle_a + angle_b)
        return torch.concat([trivials, mag, nontrivials_real], dim=-1)
        
# Group Estimating Layers 
class KabschAlignment(torch.nn.Module):
    def __init__(self, config: CostConfig):
        super().__init__()
        # No learnable parameters, just a utility layer for estimating rotation between two point clouds.
        self.centered = config.centered_samples 

    def _kabsch(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A, B: [Ba, Per, N, D]
        H = A.transpose(-2, -1) @ B
        U, _, Vh = torch.linalg.svd(H, full_matrices=False)
        V = Vh.transpose(-2, -1)
        R = V @ U.transpose(-2, -1)
        neg_mask = torch.det(R) < 0
        if neg_mask.any():
            V = V.clone()
            V[neg_mask, :, -1] *= -1
            R = V @ U.transpose(-2, -1)
        return R

    def forward(self,A: torch.Tensor, B: torch.Tensor):
        """
        See: https://en.wikipedia.org/wiki/Kabsch_algorithm
        NOTE: This is inpired by: https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
        2-D or 3-D registration with known correspondences.
        Registration occurs in the zero centered coordinate system, and then
        must be transported back.
            Args:
            -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
            -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
            Returns:
            -    R: optimal rotation
            -    P: optimal permutation
        """
        A_centroid = A.mean(dim=-2, keepdim=True)
        B_centroid = B.mean(dim=-2, keepdim=True)
        if not self.centered: 
            A = A - A_centroid
            B = B - B_centroid

        # Sample all the random permutations  [Per, N]
        perm_indices = torch.tensor(
            list(permutations(range(A.shape[-2]))),
            device=A.device,
            dtype=torch.long,
        )                                                 
        # Assume all permutation of A - [Ba, Per, N, D]
        A_perm = A[:, perm_indices, :]
        B_perm = B[:, None, :, :]        
        R = self._kabsch(A_perm, B_perm)       
        # Align all permuted A to B - [Ba, Per, N, D]
        A_aligned = (R @ A_perm.transpose(-2, -1)).transpose(-2, -1)
        L2 = torch.sqrt(((A_aligned - B_perm) ** 2).sum(dim=-1).mean(dim=-1))
        # Getting the minimum L2 across permutations for each batch element
        perm_idx = torch.argmin(L2, dim=-1)       # [Ba]
        batch_idx = torch.arange(R.shape[0], device=R.device)
        best_R_A = R[batch_idx, perm_idx]           # [Ba, D, D]
        best_perm = perm_indices[perm_idx]       
        best_perm_A = A[batch_idx[:, None], best_perm]
        if not self.centered: 
            best_t_A = A_centroid - R @ B_centroid.transpose(-2, -1)
        else:
            assert torch.all(A_centroid < 1e-6), "Expected centered samples to have zero centroid"
            assert torch.all(B_centroid < 1e-6), "Expected centered samples to have zero centroid"
            best_t_A = torch.zeros_like(A_centroid)
        return best_R_A, best_perm_A, best_t_A

# DEPRECATED: 
def _flusser_invariants(moments, pq): 
    # using the first type=1 as normalizer
    # check if there is a 1,0 and 0, 0 
    if not torch.is_complex(moments) and (moments.ndim != 2 or moments.shape[1] != 2):
        raise ValueError(f"Expected moments with a complex dtype or shape [M, 2], got {tuple(moments.shape)}")
    moments = moments if torch.is_complex(moments) else torch.view_as_complex(moments)
    if moments.shape[-1] != pq.shape[0]:
        raise ValueError(f"Expected moments with shape [M], got {tuple(moments.shape)}")
    # Note: This can be moved to init and pack this into class.
    vol_idx = np.where((pq == np.array([0,0])).all(axis=1))[0][0]
    norm_idx = np.where((pq == np.array([1,0])).all(axis=1))[0][0]
    orders = pq[:, 0] - pq[:, 1]
    # Normalize by the zero-order moment (intensity sum)
    moments = moments / torch.clamp(moments[..., vol_idx:vol_idx+1].abs(), min=1e-8)
    trivials = moments[..., orders == 0].abs()
    # Normalize by the first-order moment (centroid)
    c10 = moments[..., norm_idx]
    nontrivials_mask = orders != 0
    nontrivials_mask[norm_idx] = False  
    nontrivials = moments[..., nontrivials_mask]
    nontrivials_mag = nontrivials.abs()
    # Calculate magnitudes
    magnitude = c10.abs()
    # NOTE: This is not gradient safe
    angle = torch.atan2(c10.imag, c10.real) * -orders[nontrivials_mask]
    _real = magnitude * torch.cos(angle)
    _imag = magnitude * torch.sin(angle)
    real_invariants = _real * nontrivials.real - _imag * nontrivials.imag
    invariants = torch.concat([trivials, nontrivials_mag, real_invariants], dim=-1)
    return invariants

def _flexible_invariants(moments, pq):
    if not torch.is_complex(moments) and (moments.ndim != 2 or moments.shape[1] != 2):
        raise ValueError(f"Expected moments with a complex dtype or shape [M, 2], got {tuple(moments.shape)}")
    moments = moments if torch.is_complex(moments) else torch.view_as_complex(moments)
    if moments.shape[-1] != pq.shape[0]:
        raise ValueError(f"Expected moments with shape [M], got {tuple(moments.shape)}")
    # Normalize by number of particles (zero-order moment)
    vol_idx = np.where((pq == np.array([0,0])).all(axis=1))[0][0]
    moments = moments / torch.clamp(moments[..., vol_idx:vol_idx+1].abs(), min=1e-8)
    orders = pq[:, 0] - pq[:, 1]
    
    nontrivial_mask = orders != 0
    nontrivial_orders = orders[nontrivial_mask]
    nontrivials = moments[..., nontrivial_mask]
    n_non_trivial = len(nontrivial_orders)
    exp_a = torch.zeros(n_non_trivial, n_non_trivial, dtype=torch.int32)
    exp_b = torch.zeros(n_non_trivial, n_non_trivial, dtype=torch.int32)

    for idx_a, a in enumerate(nontrivial_orders):
        for idx_b, b in enumerate(nontrivial_orders):
            gcd = np.gcd(a, b)
            exp_a[idx_a, idx_b] = b // gcd
            exp_b[idx_a, idx_b] = -a // gcd
    # Many things are symmetric within the flexible basis. 
    tril_rows, tril_cols = torch.tril_indices(n_non_trivial, n_non_trivial, offset=-1)
    # Select exponents 
    exp_a = exp_a[tril_rows, tril_cols]
    exp_b = exp_b[tril_rows, tril_cols]
    # NOTE: This is not gradient safe 
    mag, angles = nontrivials = nontrivials.abs(), nontrivials.angle()
    mag_a = torch.index_select(mag, 1, tril_rows) 
    mag_b = torch.index_select(mag, 1, tril_cols) 
    # Angles
    angle_a = torch.index_select(angles, 1, tril_rows) * exp_a 
    angle_b = torch.index_select(angles, 1, tril_cols) * exp_b
    # atoms can be arranged in space, not just their count.
    real = mag_a * mag_b * torch.cos(angle_a + angle_b)
    return mag, real

def _compute_complex_moments_2d(
    points: torch.Tensor,
    max_frequency: int,
    *,
    pq: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute sparse 2D complex moments from points with shape [..., N, 2].

    Each moment is
        mu_{p,q} = sum_n z_n^p conj(z_n)^q,
    where z_n = x_n + i y_n.
    """
    if points.ndim < 2:
        raise ValueError(f"Expected points with shape [..., N, 2], got {tuple(points.shape)}")
    if points.shape[-1] != 2:
        raise ValueError(f"Expected last dimension to be 2, got {points.shape[-1]}")
    if not torch.is_floating_point(points):
        raise TypeError("points must have a real floating-point dtype")

    if pq is None:
        pq = complex_moment_pq(max_frequency, device=points.device)
    else:
        if pq.ndim != 2 or pq.shape[-1] != 2:
            raise ValueError(f"Expected orders with shape [M, 2], got {tuple(pq.shape)}")
        pq = pq.to(device=points.device, dtype=torch.long)

    z = torch.complex(points[..., 0], points[..., 1])
    z = z.unsqueeze(-1)

    p = pq[:, 0].to(dtype=z.real.dtype)
    q = pq[:, 1].to(dtype=z.real.dtype)

    moments = (z.pow(p) * z.conj().pow(q)).sum(dim=-2)
    return moments

def _estimate_rotation(m1, m2, pq, eps=1e-5):
    # GPTisch code, I think the mirroing is broken. We should use SVD instaed
    if not torch.is_complex(m1):
        m1 = torch.view_as_complex(m1)
    if not torch.is_complex(m2):
        m2 = torch.view_as_complex(m2)

    n = (pq[:, 0] - pq[:, 1]).to(m1.device, dtype=m1.real.dtype)
    mask = n != 0

    a = m1[..., mask]
    b = m2[..., mask]
    n = n[mask]

    # stable moments only
    w = torch.minimum(a.abs(), b.abs())
    stable = w > eps
    a = a[..., stable]
    b = b[..., stable]
    n = n[stable]
    w = w[..., stable]

    if a.shape[-1] == 0:
        raise ValueError("No stable nontrivial moments")

    dphi = torch.angle(a * b.conj())  # wrapped phase difference

    # initialize from order-1 if available, else strongest moment
    order1 = n == 1
    if order1.any():
        z = (w[..., order1] * torch.exp(1j * dphi[..., order1])).sum(dim=-1)
        theta0 = torch.angle(z)
    else:
        idx = w.argmax(dim=-1, keepdim=True)
        dphi0 = dphi.gather(-1, idx).squeeze(-1)
        n0 = n.gather(0, idx.squeeze(-1))
        theta0 = dphi0 / n0

    # unwrap each equation relative to theta0
    ell = torch.round((dphi - n * theta0.unsqueeze(-1)) / (2 * math.pi))
    rhs = dphi - 2 * math.pi * ell

    # weighted linear least squares:
    # minimize sum_k w_k * (n_k * theta - rhs_k)^2
    num = (w * n * rhs).sum(dim=-1)
    den = (w * n.square()).sum(dim=-1).clamp_min(eps)
    theta = num / den

    return wrap_angle(theta)
