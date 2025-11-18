from typing import Optional, Type
import torch
from torch import nn

from hippy2d import conv_factory
from hippy2d.utils import tukey_2d
from hippy2d.conv_factory import get_conv_layer


class ResnetBlock(torch.nn.Module):
    def __init__(self, 
                 conv_layer: str, 
                 conv_kwargs: dict,
                 kernel_size:int,
                 in_channels:int, 
                 out_channels:int,
                 input_size:int,
                 subsampling:bool = True,
                 residual:bool = True,
                 activation: str = "ELU",
                 norm: str = "layer", # "batch" or "layer"
                 conv_padding: str = "same",
                 channels_masking: str = "tukey" # "tukey" or "none"): 
    ):
        super(ResnetBlock, self).__init__()

        # Properties
        self.padding = conv_padding
        self.input_size = input_size
        self.residual = residual
        
        # Prepare convolutional layer
        
        assert 'in_channels' not in conv_kwargs, "in_channels already in conv_kwargs"
        assert 'out_channels' not in conv_kwargs, "out_channels already in conv_kwargs"
        assert 'kernel_size' not in conv_kwargs, "kernel_size already in conv_kwargs"
        assert 'padding' not in conv_kwargs, "padding already in conv_kwargs"

        conv_kwargs = conv_kwargs.copy()  # To avoid modifying the original dictionary

        conv_kwargs['in_channels'] = in_channels
        conv_kwargs['out_channels'] = out_channels
        conv_kwargs['input_size'] = input_size
        conv_kwargs['kernel_size'] = kernel_size
        conv_kwargs['padding'] = conv_padding
        
        self.conv = get_conv_layer(conv_layer, conv_kwargs)

        # Padding
        if conv_padding == "same":
            conv_output_shape = input_size
        else:
            conv_output_shape = input_size + (2 * conv_padding) - kernel_size + 1
        assert (input_size - conv_output_shape) % 2 == 0, "Input size must be even for valid padding"
        self.identity_pad = (input_size - conv_output_shape) // 2 
        
        # TODO: Here should be a group norm too
        # Normalization layer
        if norm == "batch":
            self.norm = torch.nn.BatchNorm2d(num_features=out_channels,
                                             affine=False,
                                             dtype=torch.get_default_dtype())
        elif norm == "layer":
            self.norm = torch.nn.LayerNorm(normalized_shape=(out_channels, conv_output_shape, conv_output_shape),
                                          elementwise_affine=False,
                                          dtype=torch.get_default_dtype())
        else: 
            raise ValueError(f"Unknown normalization type: {norm}. Use 'batch' or 'layer'.")


        # Activation function
        assert activation in dir(torch.nn), f"Unknown activation function: {activation}"
        self.activation = getattr(torch.nn, activation)()


        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = torch.nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=1,
                                                 bias=False)
        else: 
            self.residual_conv = torch.nn.Identity()
        
        # Subsampling layer
        if subsampling: 
            self.subsampling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.subsampling = torch.nn.Identity()
        

        # Masking layer
        # Note: This can be done by torch.masked.MaskedTensor, but it is not supported for complex
        # it's possible to rewrite the whole block using own complex convolution implementation
        assert channels_masking in ["tukey", "none"], f"Unknown channels_masking: {channels_masking}"
        self.channels_masking = channels_masking
        if channels_masking == "tukey":
            self.features_mask = torch.nn.Parameter(torch.from_numpy(tukey_2d(self.input_size, 0.5)).to(dtype=torch.get_default_dtype()), requires_grad=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complex invariant convolution block.
        :param x: Input tensor of shape (batch_size, in_channels, height, width)
        :return: Output tensor of shape (batch_size, out_channels, height', width')
        """

        if self.padding == "same" or self.identity_pad == 0:
            identity = x
        else:
            identity = x[..., 
                         self.identity_pad: -self.identity_pad,
                         self.identity_pad: -self.identity_pad]
        # TODO: Here should be a circular masking for the whole feature map
        if self.channels_masking == "tukey":
            x = x * self.features_mask
        # Radial Part
        x = self.conv(x)
        # Here we can use the Masked_tensor  instead zero masking
        # Apply batch normalization and activation
        x = self.norm(x)
        x = self.activation(x)
        x = self.subsampling(x)
        if self.residual:
            identity = self.subsampling(identity)
            # Add the residual connection
            x = x + self.residual_conv(identity)
        return x


def get_padding(kernel_size: int, stride: int, dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class TimmBasicBlock(torch.nn.Module): 
    def __init__(self, 
                 # Conv Settings 
                 input_size: int, 
                 in_channels: int, 
                 out_channels: int,
                 tukey_masking: bool = True,
                 conv_layer: Optional[torch.nn.Module] = "Conv2d",  
                 conv_kwargs: dict=dict(stride=1,
                                        padding=1,
                                        bias=False),
                 kernel_size:int=3,
                 # Layers Settings
                 act_layer: Type[nn.Module] = nn.ReLU, 
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 aa_layer: Optional[Type[nn.Module]] = nn.AvgPool2d,
                 drop_path: Optional[torch.nn.Module] = None,
                 drop_block:Optional[torch.nn.Module] = None, 
                 padding: str = "same"):
        super().__init__() 

        assert drop_path is None, "drop_path is not implemented yet"
        assert drop_block is None, "drop_block is not implemented yet"
        
        # Prepare convolutional layer
        
        assert 'in_channels' not in conv_kwargs, "in_channels already in conv_kwargs"
        assert 'out_channels' not in conv_kwargs, "out_channels already in conv_kwargs"
        assert 'kernel_size' not in conv_kwargs, "kernel_size already in conv_kwargs"
        assert 'padding' not in conv_kwargs, "padding already in conv_kwargs"

        # Padding
        if padding == "same":
            conv1_output_shape = input_size
            conv2_output_shape = input_size if aa_layer is None else input_size // 2
        else:
            conv1_output_shape = input_size + (2 * padding) - kernel_size + 1
            conv2_output_shape = conv1_output_shape + (2 * padding) - kernel_size + 1

        assert (input_size - conv1_output_shape) % 2 == 0, "Input size must be even for valid padding"

        conv_kwargs = conv_kwargs.copy()  # To avoid modifying the original dictionary

        conv_kwargs['in_channels'] = in_channels
        conv_kwargs['out_channels'] = out_channels
        conv_kwargs['input_size'] = input_size
        conv_kwargs['kernel_size'] = kernel_size
        conv_kwargs['padding'] = padding
        
        # 1. layer
        self.mask1 = None if not tukey_masking else torch.nn.Parameter(torch.from_numpy(tukey_2d(input_size, 0.5)).to(dtype=torch.get_default_dtype()), requires_grad=False)
        self.conv1 = conv_factory.get_conv_layer(conv_layer, conv_kwargs)
        self.norm1 = norm_layer(out_channels) if norm_layer is nn.BatchNorm2d else norm_layer((out_channels, conv1_output_shape, conv1_output_shape), elementwise_affine=False)
        self.drop_block = torch.nn.Identity() # TODO: implement drop_block
        self.act1 = act_layer(inplace=True)
        if aa_layer is not None:
            self.aa = aa_layer(kernel_size=2, stride=2) # Anti-aliasing and downscale # TODO: Try BlurPool2d here
        else:
            self.aa = torch.nn.Identity()

        conv_kwargs = conv_kwargs.copy()  # To avoid modifying the original dictionary
        conv_kwargs['in_channels'] = out_channels
        # 2. layer
        if tukey_masking:
            self.mask2 = torch.nn.Parameter(torch.from_numpy(tukey_2d(conv2_output_shape, 0.5)).to(dtype=torch.get_default_dtype()), requires_grad=False)
        else:
            self.mask2 = None

        self.conv2 = conv_factory.get_conv_layer(conv_layer, conv_kwargs)
        self.norm2 = norm_layer(out_channels) if norm_layer is nn.BatchNorm2d else norm_layer((out_channels, conv2_output_shape, conv2_output_shape), elementwise_affine=False)
        # self.drop_path  = torch.nn.Identity()
        self.act2 = act_layer(inplace=True) 

        # Residual Part 
        if in_channels != out_channels:
            self.identity= torch.nn.Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=1,
                                           #TODO: Add padding 
                                           bias=False)
        else: 
            self.identity = torch.nn.Identity()
        self.downsample = aa_layer(kernel_size=2, stride=2) if aa_layer is not None else torch.nn.Identity()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.identity(self.downsample(x))

        # First conv layer
        if self.mask1 is not None:
            x = x * self.mask1
        x = self.conv1(x) 
        x = self.norm1(x) # TODO: Mask should be here too
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x) # TODO: mask should be here too

        # Second conv layer 
        if self.mask2 is not None:
            x = x * self.mask2
        x = self.conv2(x)
        x = self.norm2(x) # TODO: Mask should be here too

        # TODO: Here is also squeze-and-excitation can be added
        # x = drop_path(x)
        x += shortcut
        x = self.act2(x)

        return x

