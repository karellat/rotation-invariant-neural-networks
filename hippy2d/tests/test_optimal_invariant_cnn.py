import pytest
import torch
from einops import repeat
from warnings import warn
import torchvision.transforms.v2 as transforms

from hippy2d.optimal_invariant_cnn import (
    ComplexInvariantConv2D,
    ComplexInvariantConv2DR, 
    KERNEL_SIZE,
    BASIS_P0,
    BASIS_Q0,
    N_RINGS,
    MAX_ORDER
)
from hippy2d.blocks import ResnetBlock
from hippy2d.models import PrototypeOptimalInvCNN
from hippy2d.utils import get_testing_img, get_default_complex

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
        for preserve_energy in [True, False]:
            for magnitude_normalization in ["one", "flussers", "copy"]:
                for invariant_norm in ["none", "rayleigh", "gauss"]:
                    if not preserve_energy and invariant_norm != "none":
                        # This combination does not make sense, skip
                        continue
                    inv_conv = ComplexInvariantConv2D(kernel_size=KERNEL_SIZE,
                                                    max_order=MAX_ORDER,
                                                    input_size=test_images[0].shape[-1],
                                                    magnitude_normalization=magnitude_normalization,    
                                                    preserve_energy=preserve_energy,
                                                    invariant_norm=invariant_norm,
                                                    in_channels=3,
                                                    out_channels=12).to(test_device)
                    # Forward pass through the complex invariant convolution layer
                    try:
                        self._test_90_module(inv_conv, test_images, test_device)
                    except Exception as e:
                        warn(f"Testing failed for {preserve_energy=}, {magnitude_normalization=}, {invariant_norm=}: {e}")
                        raise e

    def test_90_layer_normalized_moments(self, test_images, test_device):
        """Test the 90-degree rotation layer with normalized moments."""
        inv_conv = ComplexInvariantConv2D(kernel_size=KERNEL_SIZE,
                                          max_order=MAX_ORDER,
                                          input_size=test_images[0].shape[-1],
                                          in_channels=3,
                                          out_channels=12,
                                          polynomials_magnitude_normalization=True).to(test_device)
        # Forward pass through the complex invariant convolution layer
        self._test_90_module(inv_conv, test_images, test_device)

    def test_90_radial_layer(self, test_images, test_device):
        """Test the 90-degree rotation radial layer."""
        rc2_conv = ComplexInvariantConv2DR(kernel_size=KERNEL_SIZE,
                                           max_order=MAX_ORDER,
                                           n_rings=N_RINGS,
                                           input_size=test_images[0].shape[-1],
                                           in_channels=3,
                                           out_channels=12).to(test_device)
        # Forward pass through the complex invariant convolution layer
        self._test_90_module(rc2_conv, test_images, test_device)

    def test_90_block(self, test_images, test_device):
        """Test the 90-degree rotation block."""
        for norm in ['batch', 'layer']:
            inv_block = ResnetBlock(conv_layer=ComplexInvariantConv2D,
                                    conv_kwargs=dict(),
                                    in_channels=IMAGE_CHANNELS,
                                    input_size=IMAGE_SIZE,
                                    kernel_size=KERNEL_SIZE,
                                    out_channels=12,
                                    norm=norm).to(test_device)
            # Forward pass through the complex invariant block
            self._test_90_module(inv_block, test_images, test_device)

    def test_90_radial_block(self, test_images, test_device):
        """Test the 90-degree rotation radial block."""
        inv_block = ResnetBlock(conv_layer=ComplexInvariantConv2DR,
                                conv_kwargs=dict(),
                                in_channels=IMAGE_CHANNELS,
                                kernel_size=KERNEL_SIZE,
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
    
    def test_90_radial_network(self, test_images, test_device): 
        """Test the 90-degree rotation radial network."""
        net = PrototypeOptimalInvCNN(layer=ComplexInvariantConv2DR, 
                                     in_channels=IMAGE_CHANNELS,
                                     input_size=IMAGE_SIZE,
                                     classification=False).to(test_device)

        self._test_90_module(net, test_images, test_device)

    def test_90_network_classification(self, test_images, test_device):
        """Test the 90-degree rotation network with classification."""
        for masking in [True, False]: 
            net = PrototypeOptimalInvCNN(in_channels=IMAGE_CHANNELS,
                                        channels_masking=masking,
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
        for norm in ['batch', 'layer']:
            net = PrototypeOptimalInvCNN(in_channels=IMAGE_CHANNELS,
                                        input_size=IMAGE_SIZE,
                                        norm=norm,
                                        classification=True).to(test_device)
            with torch.autograd.detect_anomaly(True):
                input_rgb, _ = test_images
                input = repeat(input_rgb, "1 c h w -> b c h w", b=5).to(test_device)
                # track the gradients with dummy loss
                input.requires_grad = True
                y = net(input)
                loss = torch.mean(y)
                loss.backward()

            # Check if the gradients are NaN
            assert not torch.isnan(input.grad).any(), "Gradients contain NaN values."
            # Check if the gradients are finite
            assert torch.isfinite(input.grad).all(), "Gradients contain non-finite values."
            # Check if the gradients are not zero
            assert not torch.all(input.grad == 0), "Gradients are all zero, which is unexpected."
    
    def test_radial_network_gradient_nan(self, test_images, test_device): 
        """Test if the gradient coming out of the network"""
        net = PrototypeOptimalInvCNN(layer=ComplexInvariantConv2DR,
                                     in_channels=IMAGE_CHANNELS,
                                     input_size=IMAGE_SIZE,
                                     classification=True).to(test_device)
        with torch.autograd.detect_anomaly(True):
            input_rgb, _ = test_images
            input = repeat(input_rgb, "1 c h w -> b c h w", b=5).to(test_device)
            # track the gradients with dummy loss
            input.requires_grad = True
            y = net(input)
            loss = torch.mean(y)
            loss.backward()

        # Check if the gradients are NaN
        assert not torch.isnan(input.grad).any(), "Gradients contain NaN values."
        # Check if the gradients are finite
        assert torch.isfinite(input.grad).all(), "Gradients contain non-finite values."
        # Check if the gradients are not zero
        assert not torch.all(input.grad == 0), "Gradients are all zero, which is unexpected."
        

if __name__ == "__main__":
    pytest.main([__file__])