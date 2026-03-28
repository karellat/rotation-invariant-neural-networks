import pytest
import torch
from einops import repeat
from warnings import warn
import torchvision.transforms.v2 as transforms

from hippy2d.models import PrototypeOptimalInvCNN, PrototypeTiny, Resnet
from hippy2d.learnable import LearnableFlusser, VarLearnableFlusser
from hippy2d.blocks import ResnetBlock, TimmBasicBlock, MBConvBlock
from hippy2d.utils import get_testing_img, get_default_complex
from hippy2d.escnn_prototype import LearnableCesa, InvariantLayerMag
import time
import logging

from hippy2d.escnn_moment_invariants import (
    FixedFlusserMomentLayer,
    FlexibleInvariantLayer,
    MagnitudeInvariantLayer,
    MomentInvariantModule,
    flusser_basis,
)

torch.set_default_dtype(torch.float64)
# NOTE: The tests will likely fail with float32 due to numerical precision issues. We should think of suitable normalization.

IMAGE_CHANNELS = 3  # RGB images
IMAGE_SIZE = 256

class TestLearnable:
    """Test the optimal layers if invariant CNNs."""
    
    @pytest.fixture
    def test_images(self):
        test_img = get_testing_img(rgb=True)
        test_img = transforms.ToTensor()(test_img).to(dtype=torch.get_default_dtype())
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
        inv_conv = LearnableFlusser(out_channels=12, 
                                in_channels=3).to(test_device)
        # Forward pass through the complex invariant convolution layer
        self._test_90_module(inv_conv, test_images, test_device)
    

    def test_90_escnn_layer(self, test_images, test_device):
        """Test the 90-degree rotation layer."""
        # Change default to 32-bit for escnn layer
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        test_images = (test_images[0].to(torch.float32), test_images[1].to(torch.float32))
        # standardize images 
        mean = test_images[0].mean(dim=[0, 2, 3], keepdim=True)
        std = test_images[0].std(dim=[0, 2, 3], keepdim=True)
        test_images = ((test_images[0] - mean) / std, (test_images[1] - mean) / std)
        inv_conv = LearnableCesa(out_channels=12, 
                                 in_channels=3,
                                 input_size=None).to(test_device)
        # Forward pass through the complex invariant convolution layer
        self._test_90_module(inv_conv, test_images, test_device)
        torch.set_default_dtype(prev_dtype)

    def test_90_block(self, test_images, test_device):
        """Test the 90-degree rotation block."""
        inv_block = ResnetBlock(conv_layer="LearnableFlusser",
                                conv_kwargs=dict(),
                                in_channels=IMAGE_CHANNELS,
                                kernel_size=15,
                                input_size=IMAGE_SIZE,
                                out_channels=12).to(test_device)
        # Forward pass through the complex invariant block
        self._test_90_module(inv_block, test_images, test_device)

    def test_90_mbblock(self, test_images, test_device):
        """Test the 90-degree rotation block."""
        torch.set_default_dtype(torch.float32)
        test_images = (test_images[0].to(torch.float32), test_images[1].to(torch.float32))
        inv_block = MBConvBlock(input_size=IMAGE_SIZE,
                                in_channels=IMAGE_CHANNELS,
                                out_channels=12,
                                kernel_size=15, 
                                norm_layer="batch", 
                                conv_layer="LearnableCesaInvLayer",
                                conv_kwargs=dict(input_size=IMAGE_SIZE)).to(test_device)
        # Forward pass through the complex invariant block
        self._test_90_module(inv_block, test_images, test_device)

    def test_90_mbblock_magreal(self, test_images, test_device):
        """Test the 90-degree rotation block."""
        torch.set_default_dtype(torch.float32)
        test_images = (test_images[0].to(torch.float32), test_images[1].to(torch.float32))
        inv_block = MBConvBlock(input_size=IMAGE_SIZE,
                                in_channels=IMAGE_CHANNELS,
                                out_channels=12,
                                kernel_size=15, 
                                norm_layer="batch", 
                                conv_layer="LearnableCesaMagRealLayer",
                                conv_kwargs=dict(input_size=IMAGE_SIZE)).to(test_device)
        # Forward pass through the complex invariant block
        self._test_90_module(inv_block, test_images, test_device)

    def test_90_mbblock_magfuncreal(self, test_images, test_device):
        """Test the 90-degree rotation block."""
        torch.set_default_dtype(torch.float32)
        test_images = (test_images[0].to(torch.float32), test_images[1].to(torch.float32))
        inv_block = MBConvBlock(input_size=IMAGE_SIZE,
                                in_channels=IMAGE_CHANNELS,
                                out_channels=12,
                                kernel_size=15, 
                                norm1_layer="none", 
                                norm2_layer="batch", 
                                conv_layer="LearnableCesaMagRealVarFunc",
                                conv_kwargs=dict(input_size=IMAGE_SIZE)).to(test_device)
        # Forward pass through the complex invariant block
        self._test_90_module(inv_block, test_images, test_device)



    def test_90_mbblock_magnormreal(self, test_images, test_device):
        """Test the 90-degree rotation block."""
        torch.set_default_dtype(torch.float32)
        test_images = (test_images[0].to(torch.float32), test_images[1].to(torch.float32))
        inv_block = MBConvBlock(input_size=IMAGE_SIZE,
                                in_channels=IMAGE_CHANNELS,
                                out_channels=12,
                                kernel_size=15, 
                                norm_layer="batch", 
                                conv_layer="LearnableCesaMagNormRealLayer",
                                conv_kwargs=dict(input_size=IMAGE_SIZE)).to(test_device)
        # Forward pass through the complex invariant block
        self._test_90_module(inv_block, test_images, test_device)

    def test_90_mbblock_mag(self, test_images, test_device):
        """Test the 90-degree rotation block."""
        torch.set_default_dtype(torch.float32)
        test_images = (test_images[0].to(torch.float32), test_images[1].to(torch.float32))
        inv_block = MBConvBlock(input_size=IMAGE_SIZE,
                                in_channels=IMAGE_CHANNELS,
                                out_channels=12,
                                kernel_size=15,
                                norm_layer="batch",
                                conv_layer="LearnableCesaMag",
                                conv_kwargs=dict(input_size=IMAGE_SIZE)).to(test_device)
        self._test_90_module(inv_block, test_images, test_device)

    def test_90_fixed_flexiblebblock_magreal(self, test_images, test_device):
        """Test the 90-degree rotation block."""
        torch.set_default_dtype(torch.float32)
        test_images = (test_images[0].to(torch.float32), test_images[1].to(torch.float32))
        inv_block = MBConvBlock(input_size=IMAGE_SIZE,
                                in_channels=IMAGE_CHANNELS,
                                out_channels=12,
                                kernel_size=15, 
                                norm_layer="batch", 
                                conv_layer="FixedFlexibleLayer",
                                conv_kwargs=dict(input_size=IMAGE_SIZE)).to(test_device)
        # Forward pass through the complex invariant block
        self._test_90_module(inv_block, test_images, test_device)
    
    def test_90_fixedmbblock_magreal(self, test_images, test_device):
        """Test the 90-degree rotation block."""
        torch.set_default_dtype(torch.float32)
        test_images = (test_images[0].to(torch.float32), test_images[1].to(torch.float32))
        inv_block = MBConvBlock(input_size=IMAGE_SIZE,
                                in_channels=IMAGE_CHANNELS,
                                out_channels=12,
                                kernel_size=15, 
                                norm_layer="batch", 
                                conv_layer="FixedMagRealLayer",
                                conv_kwargs=dict(input_size=IMAGE_SIZE)).to(test_device)
        # Forward pass through the complex invariant block
        self._test_90_module(inv_block, test_images, test_device)

    def test_invariant_layer_mag_returns_trivials_and_magnitudes(self):
        orders = [0, 0, 1, 2]
        layer = InvariantLayerMag(orders=orders, in_channels=1, groups=1)

        moments = torch.tensor(
            [[[
                [[2.0]],
                [[3.0]],
                [[3.0]],
                [[4.0]],
                [[5.0]],
                [[12.0]],
            ]]]
        )

        y = layer(moments)

        expected = torch.tensor([[[[2.0]], [[3.0]], [[5.0]], [[13.0]]]])
        torch.testing.assert_close(y, expected)
        assert layer.out_channels == expected.shape[1]

    def test_composed_moment_invariant_module_supports_swappable_layers(self):
        basis_qp = flusser_basis(max_total_degree=2)
        orders = sorted([p - q for p, q in basis_qp])
        moment_layer = FixedFlusserMomentLayer(
            orders=orders,
            basis_qp=sorted(basis_qp, key=lambda pq: pq[0] - pq[1]),
            max_order=2,
            in_channels=2,
            groups=1,
            kernel_size=5,
        )
        invariant_layer = FlexibleInvariantLayer(
            orders=orders,
            in_channels=2,
            groups=1,
            magnitude_func="none",
            max_b_exponent=0,
        )
        module = MomentInvariantModule(moment_layer=moment_layer, invariant_layer=invariant_layer)

        x = torch.randn(1, 2, 16, 16)
        y = module(x)

        assert y.shape[0] == 1
        assert y.shape[1] == invariant_layer.out_channels
        assert module.out_channels == invariant_layer.out_channels

    def test_composed_moment_invariant_module_validates_configuration(self):
        moment_layer = FixedFlusserMomentLayer(
            orders=[0, 1],
            basis_qp=[(0, 0), (1, 0)],
            max_order=1,
            in_channels=1,
            groups=1,
            kernel_size=5,
        )
        invariant_layer = MagnitudeInvariantLayer(
            orders=[0, 2],
            in_channels=1,
            groups=1,
        )

        with pytest.raises(ValueError, match="same orders"):
            MomentInvariantModule(moment_layer=moment_layer, invariant_layer=invariant_layer)

    def test_moment_layer_parser_splits_trivial_and_complex_parts(self):
        moment_layer = FixedFlusserMomentLayer(
            orders=[0, 0, 1, 2],
            basis_qp=[(0, 0), (1, 1), (1, 0), (2, 0)],
            max_order=2,
            in_channels=1,
            groups=1,
            kernel_size=5,
        )
        moments = torch.tensor(
            [[[
                [[2.0]],
                [[3.0]],
                [[5.0]],
                [[7.0]],
                [[11.0]],
                [[13.0]],
            ]]]
        )

        parsed = moment_layer.parse_output(moments)

        assert moment_layer.num_features_per_input == 6
        assert moment_layer.num_features == 6
        assert parsed.trivial.shape == (1, 1, 2, 1, 1)
        assert parsed.non_trivial.shape == (1, 1, 4, 1, 1)
        assert parsed.non_trivial_complex.shape == (1, 1, 2, 2, 1, 1)
        torch.testing.assert_close(parsed.trivial[:, :, 0], torch.tensor([[[[2.0]]]]))
        torch.testing.assert_close(parsed.trivial[:, :, 1], torch.tensor([[[[3.0]]]]))
        torch.testing.assert_close(parsed.non_trivial_complex[:, :, 0, :, 0, 0], torch.tensor([[[5.0, 7.0]]]))
        torch.testing.assert_close(parsed.non_trivial_complex[:, :, 1, :, 0, 0], torch.tensor([[[11.0, 13.0]]]))

    def test_90_mbblock_flexible(self, test_images, test_device):
        """Test the 90-degree rotation block."""
        torch.set_default_dtype(torch.float32)
        test_images = (test_images[0].to(torch.float32), test_images[1].to(torch.float32))
        inv_block = MBConvBlock(input_size=IMAGE_SIZE,
                                in_channels=IMAGE_CHANNELS,
                                out_channels=12,
                                kernel_size=15, 
                                norm_layer="batch", 
                                conv_layer="LearnableFlexibleLayer",
                                conv_kwargs=dict(input_size=IMAGE_SIZE,
                                                 magnitude_func="none",
                                                 max_b_exponent=2)).to(test_device)
        # Forward pass through the complex invariant block
        self._test_90_module(inv_block, test_images, test_device)
    
    def test_90_timm_block(self, test_images, test_device):
        """Test the 90-degree rotation block."""
        inv_block = TimmBasicBlock(
                                input_size=IMAGE_SIZE,
                                in_channels=IMAGE_CHANNELS,
                                out_channels=12,
                                kernel_size=15, 
                                norm_layer="group", 
                                conv_layer="LearnableFlusser",
                                conv_kwargs=dict(input_size=IMAGE_SIZE)).to(test_device)
        # Forward pass through the complex invariant block
        self._test_90_module(inv_block, test_images, test_device)
        
    def test_90_network(self, test_images, test_device):
        """Test the 90-degree rotation network."""
        net = PrototypeOptimalInvCNN(
                         layer="LearnableFlusser",
                         in_channels=IMAGE_CHANNELS,
                         input_size=IMAGE_SIZE,
                         classification=False).to(test_device)
        net.eval()
        self._test_90_module(net, test_images, test_device)

    def test_90_resnet(self, test_images, test_device): 
        """Test the 90-degree rotation network with Resnet blocks."""
        net = Resnet(stem='single',
                     layer="LearnableFlusser",
                     default_layer_kwargs=dict(),
                     block_types=[TimmBasicBlock, TimmBasicBlock, TimmBasicBlock, TimmBasicBlock],
                     layers=[1, 1, 1, 1],
                     in_channels=IMAGE_CHANNELS,
                     input_size=IMAGE_SIZE,
                     classification=False).to(test_device)
        net.eval()
        self._test_90_module(net, test_images, test_device)

    def test_90_network_classification(self, test_images, test_device):
        """Test the 90-degree rotation network with classification."""
        net = PrototypeOptimalInvCNN(layer="LearnableFlusser",
                                     in_channels=IMAGE_CHANNELS,
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
        net = PrototypeOptimalInvCNN(layer="LearnableFlusser",
                         in_channels=IMAGE_CHANNELS,
                         kernel_size=15,
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
        inv_conv = LearnableFlusser(out_channels=12, 
                              in_channels=3).to(test_device)
        
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

    def test_tiny_network(self, test_images, test_device):
        """Test the 90-degree rotation network."""
        net = PrototypeTiny(
                         layer="LearnableFlusser",
                         in_channels=IMAGE_CHANNELS,
                         input_size=IMAGE_SIZE)
        net.eval()

        input_rgb, rot_input_rgb = test_images
        y = net(input_rgb.to(test_device))
        y_rot = net(rot_input_rgb.to(test_device))
        torch.testing.assert_close(
            y,
            y_rot
        )
    
    def test_90_var_layer(self, test_images, test_device):
        """Test the 90-degree rotation layer with variable radial parts."""
        for radial_basis in ["legendre0", "legendre", "monomial"]:
            inv_conv = VarLearnableFlusser(out_channels=12, 
                                    in_channels=3,
                                    radial_basis=radial_basis).to(test_device)
            # Forward pass through the complex invariant convolution layer
            self._test_90_module(inv_conv, test_images, test_device)


if __name__ == "__main__":
    pytest.main([__file__])
