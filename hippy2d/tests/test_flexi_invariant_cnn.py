import pytest
import torch
from einops import repeat
from warnings import warn
import torchvision.transforms.v2 as transforms

from hippy2d.flexibleconv2d import FlexConv2d, FILTER_SIZE, MAX_ORDER, FlexBaseBlock 
from hippy2d.models import FlexInvCNN
from hippy2d.utils import get_testing_img, get_default_complex
import time
import logging

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
        if torch.cuda.is_available():
            return torch.device("cuda")
        #elif torch.backends.mps.is_available():
        #    return torch.device("mps")
        else:
            warn("No GPU detected, using CPU.")
            # Fallback to CPU if no GPU is available
            return torch.device("cpu")
        #return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        inv_conv = FlexConv2d(input_shape=IMAGE_SIZE,
                             filter_size=FILTER_SIZE,
                             max_order=MAX_ORDER,
                             out_channels=12, 
                             in_channels=3,
                             gcd=True).to(test_device)
        # Forward pass through the complex invariant convolution layer
        self._test_90_module(inv_conv, test_images, test_device)

    def test_90_block(self, test_images, test_device):
        """Test the 90-degree rotation block."""
        inv_block = FlexBaseBlock(in_channels=IMAGE_CHANNELS,
                                   input_size=IMAGE_SIZE,
                                   out_channels=12).to(test_device)
        # Forward pass through the complex invariant block
        self._test_90_module(inv_block, test_images, test_device)

    def test_90_network(self, test_images, test_device):
        """Test the 90-degree rotation network."""
        net = FlexInvCNN(in_channels=IMAGE_CHANNELS,
                         input_size=IMAGE_SIZE,
                         classification=False).to(test_device)
        net.eval()
        self._test_90_module(net, test_images, test_device)

    def test_90_network_classification(self, test_images, test_device):
        """Test the 90-degree rotation network with classification."""
        net = FlexInvCNN(in_channels=IMAGE_CHANNELS,
                         input_size=IMAGE_SIZE,
                         classification=True).to(test_device)
        net.eval()

        input_rgb, rot_input_rgb = test_images
        y = net(input_rgb.to(test_device))
        y_rot = net(rot_input_rgb.to(test_device))
        torch.testing.assert_close(
            y,
            y_rot
        )

    def test_network_gradient_nan(self, test_images, test_device): 
        """Test if the gradient coming out of the network"""
        net = FlexInvCNN(in_channels=IMAGE_CHANNELS,
                         input_size=IMAGE_SIZE,
                         classification=True).to(test_device)
        with torch.autograd.detect_anomaly(True):
            input_rgb, _ = test_images
            input_rgbs = repeat(input_rgb, '1 c h w -> 4 c h w')
            input = input_rgbs.to(test_device)
            # track the gradients with dummy loss
            input.requires_grad = True
            y = net(input)
            loss = torch.nn.CrossEntropyLoss()(y, torch.tensor([0, 1, 2, 3], device=test_device))
            loss.backward()

        # Check if the gradients are NaN
        assert not torch.isnan(input.grad).any(), "Gradients contain NaN values."
        # Check if the gradients are finite
        assert torch.isfinite(input.grad).all(), "Gradients contain non-finite values."
        # Check if the gradients are not zero
        assert not torch.all(input.grad == 0), "Gradients are all zero, which is unexpected."
        
    def test_90_layer_speed(self, test_images, test_device):
        """Test the 90-degree rotation layer speed performance."""
        inv_conv = FlexConv2d(input_shape=IMAGE_SIZE,
                             filter_size=FILTER_SIZE,
                             max_order=MAX_ORDER,
                             out_channels=12, 
                             in_channels=3,
                             gcd=True).to(test_device)
        
        # Measure speed performance
        input_rgb, _ = test_images
        input_rgb = input_rgb.to(test_device)
        
        # Warmup
        for _ in range(3):
            _ = inv_conv(input_rgb)
        
        # Measure forward pass time
        torch.cuda.synchronize() if test_device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(10):
            _ = inv_conv(input_rgb)
        torch.cuda.synchronize() if test_device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        
        # Use pytest's built-in output capture or logging
        logging.error(f"FlexConv2d average forward pass time on {test_device}: {avg_time:.6f} seconds")
        
        # Or use pytest's live logging
        print(f"\nFlexConv2d Performance:")
        print(f"  Device: {test_device}")
        print(f"  Average forward pass time: {avg_time:.6f} seconds")
        print(f"  Throughput: {1/avg_time:.2f} inferences/second")
        
        # Forward pass through the complex invariant convolution layer
        self._test_90_module(inv_conv, test_images, test_device)

if __name__ == "__main__":
    pytest.main([__file__])