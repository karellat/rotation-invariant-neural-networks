import numpy as np
import torch


def get_complex_monomial(shape: int, p: int, q: int, dtype=torch.float64) -> torch.Tensor:
    """
    Create a complex monomial centered at the origin.
    
    Formula: (x + iy)^p (x - iy)^q
    
    Parameters:
    shape (int): Size of the square grid (shape x shape)
    p (int): Power for (x + iy) term, must be >= q
    q (int): Power for (x - iy) term, must be >= 0
    
    Returns:
    torch.Tensor: Complex tensor of shape (shape, shape) containing the monomial
    """
    # Test p >= q, and both natural numbers
    assert p >= 0 and q >= 0, "p and q must be non-negative integers."
    assert isinstance(p, int), "p must be an integer."
    assert isinstance(q, int), "q must be an integer."
    assert isinstance(shape, int) and shape >= 0, "Shape must be a positive integer."
    
    # Create coordinate grid
    y = torch.linspace(-1, 1, shape, dtype=dtype)  # shape (H,)
    x = torch.linspace(-1, 1, shape, dtype=dtype)  # shape (W,)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # shape (H, W)

    # Create complex coordinates
    z = xx + 1j * yy
    z_conj = torch.conj(z)

    # Raise to powers
    z_p = z**p
    z_conj_q = z_conj**q
    
    # Combine to form the complex monomial
    complex_monomial = z_p * z_conj_q
    return complex_monomial


def get_complex_invariants(img, p0, q0, r):
    """
    Compute complex invariants for a given image.
    B = { b_pq ≡ c_pq · c_q₀p₀^{p−q} | p ≥ q ∧ p + q ≤ r }


    Parameters:
    img (torch.Tensor): Input image tensor.
    p0 (int): Power p for the reference monomial.
    q0 (int): Power q for the reference monomial.
    r (int): Maximum degree of invariants to compute.
    
    Returns:
    cm_indices (list): List of indices (p, q) for computed invariants.
    cm (list): List of computed complex invariants.
    """
    assert p0 >= 0 and q0 >= 0 and r >= 0, "p0, q0, and r must be non-negative integers."
    assert img.shape[-1] == img.shape[-2], "Image must be square (..., H, W) where H == W."
    size = img.shape[-1]
    cm_indices = []
    cm = []
    pi_q0p0 = get_complex_monomial(size, q0, p0)  
    c_q0p0 = torch.sum(img * pi_q0p0, axis=(-2, -1)) 
    for p in range(0, r+1):
        for q in range(0, p+1):
            if q + p > r:
                continue
            if p == p0 and q == q0:
                continue
            cm_indices.append((p, q))
            pi = get_complex_monomial(size, p, q)
            cpq = torch.sum(img * pi, axis=(-2, -1))
            bpq = cpq * c_q0p0 ** (p - q)
            cm.append(bpq)
    return cm_indices, torch.stack(cm, dim=0)