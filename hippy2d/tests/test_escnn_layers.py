import pytest
import torch
from einops import repeat
from warnings import warn
import torchvision.transforms.v2 as transforms

from hippy2d.utils import get_testing_img
from hippy2d.escnn_layers import InvariantLayer
import escnn

torch.set_default_dtype(torch.float32)

IMAGE_CHANNELS = 3  # RGB images
IMAGE_SIZE = 256


class TestInvariantLayer:
    """Test the ESCNN InvariantLayer."""
    
    @pytest.fixture
    def test_images(self):
        test_img = get_testing_img(rgb=True)
        test_img = transforms.ToTensor()(test_img).to(dtype=torch.get_default_dtype())
        rotated_img = torch.rot90(test_img, 1, [1, 2])
        x = repeat(test_img, f'c h w -> 1 c h w').to(dtype=torch.get_default_dtype())
        rot_input = repeat(rotated_img, f'c h w -> 1 c h w').to(dtype=torch.get_default_dtype())
        
        # Standardize the images
        mean = x.mean(dim=[0, 2, 3], keepdim=True)
        std = x.std(dim=[0, 2, 3], keepdim=True)
        x = (x - mean) / std
        rot_input = (rot_input - mean) / std
        
        return x, rot_input

    @pytest.fixture
    def test_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            warn("No GPU detected, using CPU.")
            return torch.device("cpu")
    
    @pytest.fixture
    def r2_act(self):
        """Create the R2 action group."""
        return escnn.gspaces.rot2dOnR2(N=-1, maximum_frequency=3)
    
    @pytest.fixture
    def conv_setup(self, r2_act):
        """Setup convolutional layer with proper field types."""
        in_type = escnn.nn.FieldType(r2_act, [r2_act.trivial_repr] * IMAGE_CHANNELS)
        out_channels = 4
        kernel_size = 7
        
        # Create irreps
        irreps = []
        for n, irr in enumerate(r2_act.fibergroup.irreps()):
            if not irr.is_trivial():
                irreps += [irr] * int(irr.size // irr.sum_of_squares_constituents)
        irreps = list(irreps)
        
        # Build output field type
        trivials = escnn.nn.FieldType(r2_act, [r2_act.trivial_repr] * out_channels)
        norm_factor = escnn.group.directsum([r2_act.irrep(1)] * out_channels, name="norm_factor")
        norm_factor = escnn.nn.FieldType(r2_act, [norm_factor])
        non_trivials = escnn.nn.FieldType(r2_act, sorted(irreps * out_channels, key=lambda r: r.id))
        conv_type = trivials + norm_factor + non_trivials
        
        # Create convolutional layer
        conv = escnn.nn.R2Conv(in_type,
                               conv_type,
                               kernel_size=kernel_size,
                               padding=2,
                               sigma=0.6,
                               initialize=True)
        conv.weights = torch.nn.Parameter(torch.ones_like(conv.weights) * 0.01)
        
        return conv, in_type, conv_type, out_channels
    
    @staticmethod
    def _test_90_module(module, conv, in_type, test_images, test_device):
        """Helper to test 90-degree rotation invariance."""
        input_rgb, rot_input_rgb = test_images
        input_rgb = input_rgb.to(test_device)
        rot_input_rgb = rot_input_rgb.to(test_device)
        
        # Define simple network
        def _simple_net(img):
            x = escnn.nn.GeometricTensor(img, in_type)
            x = conv(x)
            x = module(x)
            return x
        
        invariants = _simple_net(input_rgb)
        rotated_invariants = _simple_net(rot_input_rgb)
        
        # Assert close - the invariants should be rotated versions of each other
        torch.testing.assert_close(
            invariants.tensor, 
            torch.rot90(rotated_invariants.tensor, k=-1, dims=[-2, -1]),
            rtol=1e-4,
            atol=1e-4
        )
        
        all_channels = invariants.tensor.shape[1]
        zero_channels = 0
        
        # Check for zero channels
        for batch_idx in range(invariants.tensor.shape[0]):
            for channel in range(all_channels):
                gap = torch.mean(invariants.tensor[batch_idx, channel, :, :])
                if gap.abs() < 1e-5:
                    zero_channels += 1
        
        if zero_channels == all_channels:
            raise ValueError("All channels are zero, which is not expected.")
        elif zero_channels > 0:
            warn(f"Some channels ({zero_channels}/{all_channels}) are zero, which is not expected.")

    def test_invariant_layer_initialization(self, r2_act, conv_setup):
        """Test that InvariantLayer initializes correctly."""
        conv, in_type, conv_type, out_channels = conv_setup
        
        inv_layer = InvariantLayer(r2_act=r2_act, in_type=conv_type, in_channels=out_channels)
        
        assert inv_layer.in_type == conv_type
        assert inv_layer.out_type.gspace == r2_act
        assert len(inv_layer.out_type.representations) > 0
    
    def test_90_invariance(self, r2_act, conv_setup, test_images, test_device):
        """Test the 90-degree rotation invariance."""
        conv, in_type, conv_type, out_channels = conv_setup
        conv = conv.to(test_device)
        
        inv_layer = InvariantLayer(r2_act=r2_act, in_type=conv_type, in_channels=out_channels)
        inv_layer = inv_layer.to(test_device)
        
        self._test_90_module(inv_layer, conv, in_type, test_images, test_device)
    
    def test_forward_output_type(self, r2_act, conv_setup, test_images, test_device):
        """Test that forward pass returns correct output type."""
        conv, in_type, conv_type, out_channels = conv_setup
        conv = conv.to(test_device)
        
        inv_layer = InvariantLayer(r2_act=r2_act, in_type=conv_type, in_channels=out_channels)
        inv_layer = inv_layer.to(test_device)
        
        input_rgb, _ = test_images
        input_rgb = input_rgb.to(test_device)
        
        x = escnn.nn.GeometricTensor(input_rgb, in_type)
        x = conv(x)
        output = inv_layer(x)
        
        # Check output is GeometricTensor
        assert isinstance(output, escnn.nn.GeometricTensor)
        
        # Check output type is all trivial
        for rep in output.type.representations:
            assert rep.name == "irrep_0", "Output should only contain trivial representations"
    
    def test_gradient_flow(self, r2_act, conv_setup, test_images, test_device):
        """Test that gradients flow correctly through the layer."""
        conv, in_type, conv_type, out_channels = conv_setup
        conv = conv.to(test_device)
        
        inv_layer = InvariantLayer(r2_act=r2_act, in_type=conv_type, in_channels=out_channels)
        inv_layer = inv_layer.to(test_device)
        
        input_rgb, _ = test_images
        input_rgb = input_rgb.to(test_device)
        input_rgb.requires_grad = True
        
        with torch.autograd.detect_anomaly(True):
            x = escnn.nn.GeometricTensor(input_rgb, in_type)
            x = conv(x)
            output = inv_layer(x)
            loss = output.tensor.sum()
            loss.backward()
        
        # Check if the gradients are not NaN
        assert not torch.isnan(input_rgb.grad).any(), "Gradients contain NaN values."
        # Check if the gradients are finite
        assert torch.isfinite(input_rgb.grad).all(), "Gradients contain non-finite values."
        # Check if the gradients are not all zero
        assert not torch.all(input_rgb.grad == 0), "Gradients are all zero, which is unexpected."
    
    def test_different_channel_counts(self, r2_act, test_device):
        """Test InvariantLayer with different channel counts."""
        for out_channels in [2, 4, 8]:
            in_type = escnn.nn.FieldType(r2_act, [r2_act.trivial_repr] * IMAGE_CHANNELS)
            kernel_size = 7
            
            # Create irreps
            irreps = []
            for n, irr in enumerate(r2_act.fibergroup.irreps()):
                if not irr.is_trivial():
                    irreps += [irr] * int(irr.size // irr.sum_of_squares_constituents)
            
            # Build output field type
            trivials = escnn.nn.FieldType(r2_act, [r2_act.trivial_repr] * out_channels)
            norm_factor = escnn.group.directsum([r2_act.irrep(1)] * out_channels, name="norm_factor")
            norm_factor = escnn.nn.FieldType(r2_act, [norm_factor])
            non_trivials = escnn.nn.FieldType(r2_act, sorted(irreps * out_channels, key=lambda r: r.id))
            conv_type = trivials + norm_factor + non_trivials
            
            inv_layer = InvariantLayer(r2_act=r2_act, in_type=conv_type, in_channels=out_channels)
            inv_layer = inv_layer.to(test_device)
            
            assert inv_layer.in_channels == out_channels


if __name__ == "__main__":
    pytest.main([__file__])
