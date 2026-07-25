
import pytest
import torch
from warnings import warn
from einops import repeat
import torchvision.transforms.v2 as transforms

from hippy2d.utils import get_testing_img
from hippy2d.opt_inv_layers import CompiledInvariantLayer, CompiledMomentLayer, CompiledMomentO2Layer
from hippy2d.escnn_prototype import MomentLayer

torch.set_default_dtype(torch.float32)


IMAGE_CHANNELS = 3  # RGB images
IMAGE_SIZE = 256
class TestCompiledInvariantLayer:
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
    def test_vparity_images(self):
        test_img = get_testing_img(rgb=True)
        test_img = transforms.ToTensor()(test_img).to(
            dtype=torch.get_default_dtype()
        )
        vertical_img = torch.flip(test_img, dims=[1])
        x = repeat(test_img, "c h w -> 1 c h w")
        vertical_input = repeat(
            vertical_img,
            "c h w -> 1 c h w",
        )
        # Standardize the images
        mean = x.mean(dim=[0, 2, 3], keepdim=True)
        std = x.std(dim=[0, 2, 3], keepdim=True)
        x = (x - mean) / std
        vertical_input = (vertical_input - mean) / std
        return x, vertical_input

    @pytest.fixture
    def test_hparity_images(self):
        test_img = get_testing_img(rgb=True)
        test_img = transforms.ToTensor()(test_img).to(
            dtype=torch.get_default_dtype()
        )
        horizontal_img = torch.flip(test_img, dims=[2])
        x = repeat(test_img, f'c h w -> 1 c h w').to(dtype=torch.get_default_dtype())
        horizontal_input = repeat(horizontal_img, f'c h w -> 1 c h w').to(dtype=torch.get_default_dtype())
        
        # Standardize the images
        mean = x.mean(dim=[0, 2, 3], keepdim=True)
        std = x.std(dim=[0, 2, 3], keepdim=True)
        x = (x - mean) / std
        horizontal_input = (horizontal_input - mean) / std
        
        return x, horizontal_input



    @pytest.fixture
    def test_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            warn("No GPU detected, using CPU.")
            return torch.device("cpu")

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
                    torch.rot90(c_rot_channels[batch_idx, channel, :, :], k=-1, dims=(-2, -1)),
                    rtol=1e-3,
                    atol=1e-1
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

    @staticmethod
    def _test_parity_module(module, test_images, test_device, parity_type="vertical", comparison_should_fail = False):
        input_rgb, vparity_input_rgb = test_images
        input_rgb = input_rgb.to(test_device)
        vparity_input_rgb = vparity_input_rgb.to(test_device)

        c_channels = module(input_rgb)
        c_vparity_channels = module(vparity_input_rgb)

        all_channels = 0
        zero_channels = 0
        # Assert same features for the conv
        for batch_idx in range(c_channels.shape[0]):
            for channel in range(c_channels.shape[1]):
                gap = torch.mean(c_channels[batch_idx, channel, :, :])
                if parity_type == "vertical": 
                    flip_dims = (-2,)
                elif parity_type == "horizontal":
                    flip_dims = (-1,)
                else:
                    raise ValueError(f"Invalid parity_type: {parity_type}. Must be 'vertical' or 'horizontal'.")
                torch.testing.assert_close(
                    c_channels[batch_idx, channel, :, :],
                    torch.flip(c_vparity_channels[batch_idx, channel, :, :], dims=flip_dims),
                    rtol=1e-3,
                    atol=1e-1
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
        orders = [0, 1, 2, 3]
        layers = torch.nn.Sequential(
            MomentLayer(max_order=4, orders=orders, in_channels=IMAGE_CHANNELS),
            CompiledInvariantLayer(orders=orders,
                                   pre_norm_function="layer_norm", 
                                   phase_function="polar",
                                   compile_functions=True,
                                   magnitude_function="nicks"),
        ).to(test_device)

        self._test_90_module(layers, test_images, test_device)

    def test_90_SO2_layer(self, test_images, test_device):
        """Test the 90-degree rotation layer."""
        orders = [0, 1, 2, 3]
        layers = torch.nn.Sequential(
            CompiledMomentLayer(max_order=4, orders=orders, in_channels=IMAGE_CHANNELS),
            CompiledInvariantLayer(orders=orders,
                                   pre_norm_function="layer_norm", 
                                   phase_function="polar",
                                   compile_functions=True,
                                   magnitude_function="nicks"),
        ).to(test_device)

        self._test_90_module(layers, test_images, test_device)

    def test_90_O2_layer(self, test_images, test_device):
        """Test the 90-degree rotation layer with O(2) symmetry."""
        orders = [0, 1, 2, 3]
        layers = torch.nn.Sequential(
            CompiledMomentO2Layer(max_order=4, orders=orders, in_channels=IMAGE_CHANNELS),
            CompiledInvariantLayer(orders=orders,
                                   pre_norm_function="layer_norm", 
                                   phase_function="polar",
                                   compile_functions=True,
                                   magnitude_function="nicks")
        ).to(test_device)

        self._test_90_module(layers, test_images, test_device)

    def test_parity_layer(self, test_vparity_images, test_hparity_images, test_device):
        """Test the vertical parity layer."""
        orders = [0, 1, 2, 3]
        layers = torch.nn.Sequential(
            CompiledMomentO2Layer(max_order=4, orders=orders, in_channels=IMAGE_CHANNELS),
            CompiledInvariantLayer(orders=orders,
                                   pre_norm_function="layer_norm", 
                                   phase_function="polar",
                                   compile_functions=True,
                                   magnitude_function="nicks")
        ).to(test_device)


        self._test_parity_module(layers, test_vparity_images, test_device, parity_type="vertical")
        self._test_parity_module(layers, test_hparity_images, test_device, parity_type="horizontal")
