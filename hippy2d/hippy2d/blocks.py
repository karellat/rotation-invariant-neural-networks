import torch

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

