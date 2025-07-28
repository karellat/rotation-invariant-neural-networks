import torch 
from einops import rearrange

from src.complex_invariants_2d import get_complex_monomial
from src.utils import tukey_2d, get_default_complex, get_circular_mask

BASIS_P0 = 1
BASIS_Q0 = 0
FILTER_SIZE = 15
MAX_ORDER = 4

class ComplexInvariantConv2D(torch.nn.Module):
    def __init__(self,
                 filter_size:int,
                 max_order:int, 
                 in_channels: int,
                 out_channels:int,
                 basis_p0:int = 1,
                 basis_q0:int = 0, 
                 circular_padding:str ="tukey",
                 conv_padding: str = "same"):
        super(ComplexInvariantConv2D, self).__init__()
        self.filter_size = filter_size
        self.max_order = max_order
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basis_p0 = basis_p0
        self.basis_q0 = basis_q0

        # Asserts 
        assert filter_size % 2 == 1, "Filter size must be odd"
        assert max_order >= 0, "Max order must be non-negative"
        assert in_channels > 0, "Number of input channels must be positive"
        assert out_channels > 0, "Number of output channels must be positive"
        assert out_channels % 2 == 0, "Output channels must be even, we use complex dimension as 2xchannels"
        assert filter_size >= 5, "Filters must be decent size for complex invariants"

        # Group convolution requires out_channels to be divisible by in_channels
        assert out_channels % in_channels == 0, "Output channels must be divisible by input channels for group convolution"
        assert self.basis_p0 >= 0 and self.basis_q0 >= 0, "Basis indices must be non-negative integers"
        assert self.basis_p0 + self.basis_q0 <= max_order, "Basis indices must not exceed the maximum order"
        assert self.basis_p0 - self.basis_q0 == 1, "Basis indices must differ by 1 for the normalization term"

        # Prepare the fixed filters corresponding to the complex monomials
        filters = []
        ind = []
        # Add the normalizition term
        filters.append(get_complex_monomial(self.filter_size,
                                             self.basis_q0,
                                             self.basis_p0,
                                             dtype=torch.get_default_dtype()))
        ind.append((self.basis_q0, self.basis_p0))
        # Add the complex monomials up to the max order
        for p in range(0, self.max_order + 1):
            for q in range(0, min(self.max_order + 1-p, p + 1)):
                filters.append(get_complex_monomial(self.filter_size, p, q, dtype=torch.get_default_dtype()))
                ind.append((p, q))

        # NOTE: This part can be shared by all the layers, that can save memory 
        filters = rearrange(filters, 'n h w -> n 1 h w')

        # Radial padding
        if circular_padding == "tukey":
            # Use Tukey window for circular padding
            mask = torch.from_numpy(tukey_2d(self.filter_size, 0.5)).to(dtype=torch.get_default_dtype())
        elif circular_padding == "circular":
            # Use circular padding
            mask = get_circular_mask(self.filter_size, dtype=torch.get_default_dtype())
        elif circular_padding == "none":
            # No padding, just use the filters as they are
            mask = 1
        else:
            raise ValueError(f"Unknown circular padding type: {circular_padding}. Use 'tukey' or 'none'.")
        filters = filters * mask
        self.padding = conv_padding 
        self.filters = torch.nn.Parameter(filters,
                                          requires_grad=False) # Make it non-trainable parameter

        self.exponents = torch.tensor([p-q for (p,q) in ind], dtype=torch.int64)[1:] # Skip the normalization term
        self.exponents = self.exponents[None, :, None, None] # Broadcasting dimension
        self.ind = torch.tensor(ind, dtype=torch.uint16)
        self.num_invariants = len(self.ind) - 1 # Number of invariants and skip the normalization term
        # Conv1x1 that mixes invariants together
        self.conv1x1 = torch.nn.Conv2d(in_channels=self.num_invariants*self.in_channels,
                                       out_channels=self.out_channels//2,
                                       groups=self.in_channels,
                                       kernel_size=1, 
                                       dtype=get_default_complex())
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complex invariant convolution layer.
        :param x: Input tensor of shape (batch_size, in_channels, height, width)
        :return: Output tensor of shape (batch_size, out_channels, height', width')
        """
        # Check input shape
        assert x.dim() == 4, "Input must be a 4D tensor"
        assert x.shape[1] == self.in_channels, f"Input channels {x.shape[1]} do not match expected {self.in_channels}"
        
        
        # Act on batch and channels together
        x = rearrange(x, 'b c h w -> (b c) 1 h w', c=self.in_channels)
        # Apply the complex invariant convolution
        moments = torch.nn.functional.conv2d(x,
                                             weight=self.filters,
                                             padding=self.padding) # TODO: Solve padding

        normalization_moment = moments[:, 0:1]
        moments = moments[:, 1:]
        result = moments * (normalization_moment ** self.exponents)
        result = rearrange(result, '(b c) n h w -> b (c n) h w', c=self.in_channels)
        # TODO: This should be normalized properly 
        features = self.conv1x1(result) # Convert back to real
        # NOTE: We should find nicer way to do this, it should be just a view
        features = rearrange(torch.view_as_real(features), 'b c h w co -> b (c co) h w')
        return features

# Create a block 
class ComplexBaseBlock(torch.nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int,
                 input_size:int,
                 basis_p0:int = BASIS_P0,
                 basis_q0:int = BASIS_Q0,
                 filter_size:int = FILTER_SIZE, 
                 max_order:int = MAX_ORDER, 
                 residual:bool = True, 
                 subsampling:bool = True, 
                 conv_padding: str = "same"): 
        super(ComplexBaseBlock, self).__init__()
        self.conv = ComplexInvariantConv2D(filter_size=filter_size,
                                            max_order=max_order,
                                            in_channels=in_channels,
                                            out_channels=out_channels,
                                            basis_p0=basis_p0,
                                            basis_q0=basis_q0)
        self.batch_norm = torch.nn.BatchNorm2d(num_features=out_channels, affine=False)
        self.activation = torch.nn.ReLU(inplace=True)
        self.residual = residual
        self.valid_padding = (self.conv.filter_size - 1) // 2
        self.padding = conv_padding
        self.input_size = input_size

        if in_channels != out_channels:
            self.residual_conv = torch.nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=1,
                                                 bias=False)
        else: 
            self.residual_conv = torch.nn.Identity()
        
        if subsampling: 
            self.subsampling = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.subsampling = torch.nn.Identity()
        # Note: This can be done by torch.masked.MaskedTensor, but it is not supported for complex
        # it's possible to rewrite the whole block using own complex convolution implementation
        self.features_mask = torch.from_numpy(tukey_2d(self.input_size, 0.5))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the complex invariant convolution block.
        :param x: Input tensor of shape (batch_size, in_channels, height, width)
        :return: Output tensor of shape (batch_size, out_channels, height', width')
        """
        if self.padding == "valid":
            # Use valid padding
            identity = torch.view_as_real(x)[..., 
                                             self.valid_padding:-self.valid_padding,
                                             self.valid_padding:-self.valid_padding,
                                             0]
        else:
            # Use same padding

            identity = torch.view_as_real(x)[..., 
                                         0]  # Store input for residual connection
        # TODO: Here should be a circular masking for the whole feature map
        x = self.conv(x)
        # Here we can use the Masked_tensor  instead zero masking
        x *= self.features_mask
        # Apply batch normalization and activation
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.subsampling(x)
        if self.residual:
            identity = self.subsampling(identity)
            # Add the residual connection
            x += self.residual_conv(identity)
        return x