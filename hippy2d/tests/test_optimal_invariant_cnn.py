import pytest
import torch
from einops import repeat
from warnings import warn
import torchvision.transforms.v2 as transforms

from src.optimal_invariant_cnn import ComplexInvariantConv2D, ComplexBaseBlock, FILTER_SIZE, BASIS_P0, BASIS_Q0, MAX_ORDER
from src.models import PrototypeOptimalInvCNN
from src.utils import get_testing_img, get_default_complex

torch.set_default_dtype(torch.float64)
# NOTE: The tests will likely fail with float32 due to numerical precision issues. We should think of suitable normalization.

IMAGE_CHANNELS = 3  # RGB images
IMAGE_SIZE = 256

class TestComplexOptimalInvariants:
    """Test the optimal layers if invariant CNNs."""
    
    @pytest.fixture
    def test_images(self):
        test_img = get_testing_img(rgb=True)
        test_img = transforms.ToTensor()(test_img).to(dtype=torch.float64)
        rotated_img = torch.rot90(test_img, 1, [1, 2])
        x = repeat(test_img, f'c h w -> 1 c h w').to(dtype=torch.get_default_dtype())
        rot_input = repeat(rotated_img, f'c h w -> 1 c h w').to(dtype=torch.get_default_dtype())
        return x, rot_input


    @pytest.fixture
    def test_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def _test_90_module(module, test_images, test_device):
        input_rgb, rot_input_rgb = test_images
        input_rgb = input_rgb.to(test_device)
        rot_input_rgb = rot_input_rgb.to(test_device)

        c_channels = module(input_rgb)
        c_rot_channels = module(rot_input_rgb)

        all_channels = 0
        zero_channels = 0
        # Assert same features for the conv
        for batch_idx in range(c_channels.shape[0]):
            for channel in range(c_channels.shape[1]):
                gap = torch.mean(c_channels[batch_idx, channel, :, :])
                torch.testing.assert_close(
                    c_channels[batch_idx, channel, :, :],
                    torch.rot90(c_rot_channels[batch_idx, channel, :, :], k=-1, dims=(-2, -1))
                )
                all_channels += 1
                if gap < 1e-5:
                    zero_channels += 1
        if zero_channels == all_channels:
            raise ValueError("All channels are zero, which is not expected.")
        elif zero_channels > 0:
            warn(f"Some channels ({zero_channels}/{all_channels}) are zero, which is not expected.")
        else: 
            pass


    def test_90_layer(self, test_images, test_device):
        """Test the 90-degree rotation layer."""
        inv_conv = ComplexInvariantConv2D(filter_size=FILTER_SIZE,
                                          max_order=MAX_ORDER,
                                          in_channels=3,
                                          out_channels=12).to(test_device)
        # Forward pass through the complex invariant convolution layer
        self._test_90_module(inv_conv, test_images, test_device)

    def test_90_block(self, test_images, test_device):
        """Test the 90-degree rotation block."""
        inv_block = ComplexBaseBlock(in_channels=IMAGE_CHANNELS,
                                     input_size=IMAGE_SIZE,
                                     out_channels=12).to(test_device)
        # Forward pass through the complex invariant block
        self._test_90_module(inv_block, test_images, test_device)

    def test_90_network(self, test_images, test_device):
        """Test the 90-degree rotation network."""
        net = PrototypeOptimalInvCNN(in_channels=IMAGE_CHANNELS,
                                     input_size=IMAGE_SIZE,
                                     classification=False).to(test_device)

        self._test_90_module(net, test_images, test_device)

    def test_90_network_classification(self, test_images, test_device):
        """Test the 90-degree rotation network with classification."""
        net = PrototypeOptimalInvCNN(in_channels=IMAGE_CHANNELS,
                                     input_size=IMAGE_SIZE,
                                     classification=True).to(test_device)

        input_rgb, rot_input_rgb = test_images
        y = net(input_rgb.to(test_device))
        y_rot = net(rot_input_rgb.to(test_device))
        torch.testing.assert_close(
            y,
            y_rot
        )

if __name__ == "__main__":
    pytest.main([__file__])