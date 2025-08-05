import pytest
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms

from hippy2d.complex_invariants_2d import get_complex_invariants 
from hippy2d.utils import get_testing_img

MAX_R = 20  # Maximum degree of invariants to compute

class TestComplexInvariants:
    """Test the complete complex invariants computation."""
    
    @pytest.fixture
    def test_image(self):
        img = get_testing_img()
        transform = transforms.ToTensor()
        img = transform(img).to(dtype=torch.float64)
        return img
    
    def test_get_complex_invariants_basic(self, test_image):
        """Test basic functionality of complex invariants computation."""
        img = test_image
        p0, q0, r = 1, 0, 5
        
        indices, invariants = get_complex_invariants(img, p0, q0, r)
        
        assert isinstance(indices, list)
        assert isinstance(invariants, torch.Tensor)
        assert len(indices) == invariants.shape[0]
        
        # Check that reference monomial (p0, q0) is excluded
        assert (p0, q0) not in indices
    
    def test_complex_invariants_rotation_invariance(self, test_image):
        """Test that complex invariants are rotation invariant."""
        img = test_image
        rot_img = torch.rot90(img, 1, [-2, -1])  # 90-degree rotation

        p0, q0, r = 1, 0, MAX_R

        # Compute invariants for both images
        indices_orig, inv_orig = get_complex_invariants(img, p0, q0, r)
        indices_rot, inv_rot = get_complex_invariants(rot_img, p0, q0, r)
        
        # Indices should be the same
        assert indices_orig == indices_rot
        
        # Invariants should be equal (within numerical tolerance)
        torch.testing.assert_close(inv_orig, inv_rot)
    
    def test_multiple_rotation_invariance(self, test_image):
        """Test invariance under multiple rotation angles."""
        img = test_image
        p0, q0, r = 1, 0, MAX_R
        
        # Get original invariants
        _, inv_orig = get_complex_invariants(img, p0, q0, r)
        
        # Test 90, 180, 270 degree rotations
        for k in [1, 2, 3]:
            rot_img = torch.rot90(img, k, [1, 2])
            _, inv_rot = get_complex_invariants(rot_img, p0, q0, r)
            torch.testing.assert_close(inv_orig, inv_rot)
    
    def test_different_reference_monomials(self, test_image):
        """Test that different reference monomials work correctly."""
        img = test_image
        rot_img = torch.rot90(img, 1, [-2, -1])  # 90-degree rotation
        r = 8
        
        reference_monomials = [ (2, 1), (3, 2), (4, 3)]
        for p0, q0 in reference_monomials: 
            # Compute invariants for both images
            indices_orig, inv_orig = get_complex_invariants(img, p0, q0, r)
            indices_rot, inv_rot = get_complex_invariants(rot_img, p0, q0, r)
            
            # Indices should be the same
            assert indices_orig == indices_rot
            
            # Invariants should be equal (within numerical tolerance)
            torch.testing.assert_close(inv_orig, inv_rot)



if __name__ == "__main__":
    pytest.main([__file__])