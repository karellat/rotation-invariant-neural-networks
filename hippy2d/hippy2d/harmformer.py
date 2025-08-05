import torch
import numpy as np
from loguru import logger
from torch import nn
from typing import Optional
from torch import masked_fill
from scipy.linalg import dft
from einops import rearrange
from collections import OrderedDict
from torch.nn.modules.module import T

from hippy2d.utils import get_default_complex, get_circular_mask, retrieve_elements_from_indices, tukey_2d


class ComplexImg2H(nn.Module):
    def __init__(self,
                 circular_mask=False,
                 input_shape: Optional[int] = None,
                 alpha: Optional[float] = 0.5):
        super(ComplexImg2H, self).__init__()
        self.circular_mask = circular_mask
        if circular_mask:
            assert input_shape is not None, "Expecting input shape for circular mask"
            self.mask = torch.nn.Parameter(
                torch.from_numpy(tukey_2d(input_shape, alpha=alpha)[None, None, None]).type(
                    torch.get_default_dtype()),
                requires_grad=False)

    def forward(self, x: torch.Tensor):
        # From [Batch Size, Channels, Height, Width
        # Expand Tensor to Hnet dimensions [Batch Size, Order, Channels, Height, Width]
        assert x.dtype is get_default_complex()
        assert x.ndim == 4, "Expecting [Batch Size, Channels, Height, Width]"
        assert x.shape[2] == x.shape[3], "Expecting square input"
        x = rearrange(x,
                      "b (o c) h w -> b o c h w", o=1)
        if self.circular_mask:
            return x * self.mask
        else:
            return x


class HConv2d(nn.Module):
    # TODO: Add reference to source codes
    @staticmethod
    def init_phase(inp_channel_count, out_channel_count, max_order):
        """
        Initializes phase dict with phase offsets

            out_channel_count (int): number of output channels
            in_max_order: maximum rotation order of input channels
            out_max_order: maximum rotation order of output channels
        Returns:
            phase_dict: initialized dict of phase offsets [
        """

        return torch.from_numpy(
            np.random.rand(max_order, out_channel_count, inp_channel_count, 1, 1) * 2. * np.pi
        )

    @staticmethod
    def init_weights(in_channels: int,
                     out_channels: int,
                     max_order: int,
                     ring_count: int):
        """
        Initializes the dict of weights

        Args:

            in_channels: input channels
            out_channels: output channels
            max_order: maximum rotation order from input and output
            ring_count (int): Number of rings for calculating the basis
        Returns:
            weights_dict: initialized dict of weights

        """
        weights = torch.zeros((max_order, ring_count, in_channels, out_channels),
                              dtype=torch.get_default_dtype())
        for order in range(-max_order, max_order):
            sh = [ring_count, in_channels, out_channels]
            # Initializes the weights using initialization method of He
            stddev = 0.4 * np.sqrt(2.0 / np.prod(sh[:3]))
            weights[order] = torch.normal(torch.zeros(*sh), stddev)
        # Flat input & output channels
        return weights.view((max_order, ring_count, in_channels * out_channels))

    @staticmethod
    def get_angle_samples_count(kernel_size):
        return int(np.maximum(np.ceil(np.pi * kernel_size), 101))

    @staticmethod
    def get_l2_neighbors(center, shape):
        lin = np.arange(shape) + 0.5
        jj, ii = np.meshgrid(lin, lin)
        ii = ii - center[1]
        jj = jj - center[0]
        return np.vstack((np.reshape(ii, -1), np.reshape(jj, -1)))

    @staticmethod
    def get_interpolation_weights(fs, m, n_rings, angle_samples):
        """
        Used to construct the steerable filters using Radial basis functions.
        The filters are constructed on the patches of n_rings using Gaussian
        interpolation. (Code adapted from the tf code of Worrall et al., CVPR, 2017)

        Args:
            fs (int): filter size for the H-net convolutional layer
            m (int): max. rotation order for the steerable filters
            n_rings (int): No. of rings for the steerable filters
            angle_samples (int): No. of angle samples

        Returns:
            norm_weights (numpy): contains normalized weights for interpolation
            using the steerable filters
        """

        mid = int(np.floor(fs / 2))
        # Variance of Gaussian resampling
        std_gauss = (mid / n_rings) / 2
        # We define below radii up to n_rings-0.5 (as in Worrall et al, CVPR 2017)
        radii = np.linspace(m != 0, mid - 2 * std_gauss, n_rings)
        # We define pixel centers to be at positions 0.5
        center_pt = np.asarray([fs, fs]) / 2.

        # Extracting the set of angles to be sampled

        # Choosing the sampling locations for the rings
        lin = (2 * np.pi * np.arange(angle_samples)) / angle_samples
        ring_locations = np.vstack([-np.sin(lin), np.cos(lin)])

        # Create interpolation coefficient coordinates
        coords = HConv2d.get_l2_neighbors(center_pt, fs)

        # getting samples based on the chosen center_pt and the coords
        radii = radii[:, np.newaxis, np.newaxis, np.newaxis]
        ring_locations = ring_locations[np.newaxis, :, :, np.newaxis]
        diff = radii * ring_locations - coords[np.newaxis, :, np.newaxis, :]
        dist2 = np.sum(diff ** 2, axis=1)

        # Convert distances to weightings
        weights = np.exp(-0.5 * dist2 / (std_gauss ** 2))  # For bandwidth of 0.5

        # Normalizing the weights to calibrate the different steerable filters
        norm = np.sum(weights,
                      axis=2,
                      keepdims=True)
        assert np.all(norm != 0), "Normalizing by zero weights"
        return np.divide(weights,
                         norm,
                         where=(norm != 0)
                         )

    @staticmethod
    def init_weights2filters(max_order, kernel_size, n_rings):
        # low pass filter
        # shape [Orders,  #rings, #Samples]

        N = HConv2d.get_angle_samples_count(kernel_size)
        weights2filter_sampler = []
        for m in range(max_order):
            # Get the basis matrices built from the steerable filters
            weights = HConv2d.get_interpolation_weights(kernel_size,
                                                        m=m,
                                                        n_rings=n_rings,
                                                        angle_samples=N)
            low_pass_filter = np.dot(dft(N)[m, :], weights).T
            weights2filter_sampler.append(
                torch.from_numpy(low_pass_filter).to(get_default_complex())
            )
        return torch.stack(weights2filter_sampler)

    @torch.jit.export
    def get_filters(self) -> torch.Tensor:
        """
        Calculates filters in the form of weight matrices through performing
        single-frequency DFT on every ring obtained from sampling in the polar
        domain.

        Args:

        Returns:
            W: Complex filters [Order, Out_channels, In_channels, Kernel, Kernel]
        """
        x = rearrange(torch.matmul(self.weights2filters, self.ring_weights.to(get_default_complex())),
                      pattern="o (h w) (ic oc) -> o oc ic h w",
                      h=self.kernel_size,
                      w=self.kernel_size,
                      ic=self.in_channels,
                      oc=self.out_channels)
        if self.phase:
            x *= torch.complex(real=torch.cos(self.phase_offset),
                               imag=-torch.sin(self.phase_offset))
        return x

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 in_max_order: int,
                 out_max_order: int,
                 mask_shape: int,
                 tukey_window: bool = False,
                 tukey_alpha: Optional[float] = 0.5,
                 phase: bool = True,
                 n_rings: int = 3,
                 stddev: float = 0.4,
                 padding: int = 0,
                 stride: int = 1):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param in_max_order:
        :param out_max_order:
        :param phase:
        :param n_rings:
        :param stddev:
        :return:
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.phase = phase
        self.in_max_order = in_max_order
        self.out_max_order = out_max_order
        self.stddev = stddev
        self.n_rings = n_rings
        self.shape = [kernel_size, kernel_size, self.in_channels, self.out_channels]
        self.padding = padding
        self.stride = stride
        self._max_order = max(in_max_order + 1, out_max_order + 1)
        # Se
        self.ring_weights = nn.Parameter(
            HConv2d.init_weights(in_channels=in_channels,
                                 out_channels=out_channels,
                                 max_order=self._max_order,
                                 ring_count=self.n_rings),
            requires_grad=True
        )
        if self.phase:
            # Phase Offset - [Orders, In_Channels, Out_Channel]
            self.phase_offset = nn.Parameter(HConv2d.init_phase(self.in_channels,
                                                                self.out_channels,
                                                                self._max_order),
                                             requires_grad=True)

        # Circular Masking
        # TODO: Improve circular mask, make gaussian etc..
        self.tukey_window = tukey_window
        if not tukey_window:
            self.mask = torch.nn.Parameter(
                ~get_circular_mask(mask_shape)[None, None, None],
                requires_grad=False)
        else:
            self.mask = torch.nn.Parameter(
                torch.from_numpy(tukey_2d(mask_shape, alpha=tukey_alpha)[None, None, None]).type(
                    torch.get_default_dtype()),
                requires_grad=False)

        # Weight2Filters
        self.weights2filters = torch.nn.Parameter(
            HConv2d.init_weights2filters(self._max_order, kernel_size, n_rings),
            requires_grad=False)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation function for the harmonic convolution operation

        Args:
            x (deep tensor): input feature tensor dimensions [Batch Size, Order, Channels, Height, Width]

        Returns:
            R (deep tensor): output feature tensor obtained from harmonic convolution

        """
        # Expecting shape [Batch, Order, Channels, H, W]
        assert x.shape[1] == (self.in_max_order + 1), f"Unexpected number of orders(at 2), got shape = {x.shape}"
        assert x.shape[2] == self.in_channels, f"Unexpected number of channels(at 1), got shape = {x.shape}"
        assert x.shape[4] == x.shape[3], f"Expected square image, got shape = {x.shape}"

        r = self.hconv(x=x, weight=self.get_filters())
        if self.tukey_window:
            r = r * self.mask
        else:
            r = masked_fill(r, self.mask, 0)
        return r

    @torch.jit.export
    def conv2d(self, x: torch.Tensor, weight: torch.Tensor):
        return nn.functional.conv2d(input=x,
                                    weight=weight,
                                    padding=self.padding,
                                    stride=self.stride)

    @torch.jit.export
    def hconv(self,
              x: torch.Tensor,
              weight: torch.Tensor):
        """

        Args:
            x: [Batch Size, Channels, Order, Height, Width]
            weight:

        Returns:

        """
        x = rearrange(x, "b o c h w -> b (o c) h w")

        # NOTE: Prepare filters before convolution
        # as in tests.legacy.h_conv
        weights_over_out_order = []
        for out_order in range(self.out_max_order + 1):
            weights_over_in_order = []
            for in_order in range(self.in_max_order + 1):
                weight_order = out_order - in_order
                weight_filters = weight[abs(weight_order)]
                # NOTE: This is the same as the legacy code
                # but according to definition in the paper
                # zero order should have imaginary part too
                if weight_order == 0:
                    weight_filters.imag *= 0
                elif weight_order < 0:
                    weight_filters = weight_filters.conj()
                else:
                    weight_filters = weight_filters
                # Make conjugate for negative orders and zero out imaginary part for zero orders
                weights_over_in_order.append(weight_filters)
                _tmp = torch.cat(dim=1, tensors=weights_over_in_order)
            weights_over_out_order.append(
                _tmp
            )
        _weights = torch.cat(dim=0, tensors=weights_over_out_order)
        return rearrange(self.conv2d(x=x, weight=_weights),
                         pattern="b (o c) h w -> b o c h w",
                         c=self.out_channels,
                         o=self.out_max_order + 1
                         )


class DepthwiseHConv2d(nn.Module):
    @staticmethod
    def init_phase(inp_channel_count, out_channel_count, max_order):
        """
            Initializes phase dict with phase offsets

                out_channel_count (int): number of output channels
                in_max_order: maximum rotation order of input channels
                out_max_order: maximum rotation order of output channels
            Returns:
                phase_dict: initialized dict of phase offsets [
            """

        return torch.from_numpy(
            np.random.rand(max_order, out_channel_count, inp_channel_count, 1, 1) * 2. * np.pi
        )

    @staticmethod
    def init_weights(in_channels: int,
                     out_channels: int,
                     max_order: int,
                     ring_count: int):
        """
            Initializes the dict of weights

            Args:

                in_channels: input channels
                out_channels: output channels
                max_order: maximum rotation order from input and output
                ring_count (int): Number of rings for calculating the basis
            Returns:
                weights_dict: initialized dict of weights

            """
        weights = torch.zeros((max_order, ring_count, in_channels, out_channels),
                              dtype=torch.get_default_dtype())
        for order in range(-max_order, max_order):
            sh = [ring_count, in_channels, out_channels]
            # Initializes the weights using initialization method of He
            stddev = 0.4 * np.sqrt(2.0 / np.prod(sh[:3]))
            weights[order] = torch.normal(torch.zeros(*sh), stddev)
        # Flat input & output channels
        return weights.view((max_order, ring_count, in_channels * out_channels))

    @staticmethod
    def get_angle_samples_count(kernel_size):
        return int(np.maximum(np.ceil(np.pi * kernel_size), 101))

    @staticmethod
    def get_l2_neighbors(center, shape):
        lin = np.arange(shape) + 0.5
        jj, ii = np.meshgrid(lin, lin)
        ii = ii - center[1]
        jj = jj - center[0]
        return np.vstack((np.reshape(ii, -1), np.reshape(jj, -1)))

    @staticmethod
    def get_interpolation_weights(fs, m, n_rings, angle_samples):
        """
            Used to construct the steerable filters using Radial basis functions.
            The filters are constructed on the patches of n_rings using Gaussian
            interpolation. (Code adapted from the tf code of Worrall et al., CVPR, 2017)

            Args:
                fs (int): filter size for the H-net convolutional layer
                m (int): max. rotation order for the steerable filters
                n_rings (int): No. of rings for the steerable filters
                angle_samples (int): No. of angle samples

            Returns:
                norm_weights (numpy): contains normalized weights for interpolation
                using the steerable filters
            """

        mid = int(np.floor(fs / 2))
        # Variance of Gaussian resampling
        std_gauss = (mid / n_rings) / 2
        # We define below radii up to n_rings-0.5 (as in Worrall et al, CVPR 2017)
        radii = np.linspace(m != 0, mid - 2 * std_gauss, n_rings)
        # We define pixel centers to be at positions 0.5
        center_pt = np.asarray([fs, fs]) / 2.

        # Extracting the set of angles to be sampled

        # Choosing the sampling locations for the rings
        lin = (2 * np.pi * np.arange(angle_samples)) / angle_samples
        ring_locations = np.vstack([-np.sin(lin), np.cos(lin)])

        # Create interpolation coefficient coordinates
        coords = HConv2d.get_l2_neighbors(center_pt, fs)

        # getting samples based on the chosen center_pt and the coords
        radii = radii[:, np.newaxis, np.newaxis, np.newaxis]
        ring_locations = ring_locations[np.newaxis, :, :, np.newaxis]
        diff = radii * ring_locations - coords[np.newaxis, :, np.newaxis, :]
        dist2 = np.sum(diff ** 2, axis=1)

        # Convert distances to weightings
        weights = np.exp(-0.5 * dist2 / (std_gauss ** 2))  # For bandwidth of 0.5

        # Normalizing the weights to calibrate the different steerable filters
        norm = np.sum(weights,
                      axis=2,
                      keepdims=True)
        assert np.all(norm != 0), "Normalizing by zero weights"
        return np.divide(weights,
                         norm,
                         where=(norm != 0)
                         )

    @staticmethod
    def init_weights2filters(max_order, kernel_size, n_rings):
        # low pass filter
        # shape [Orders,  #rings, #Samples]

        N = HConv2d.get_angle_samples_count(kernel_size)
        weights2filter_sampler = []
        for m in range(max_order):
            # Get the basis matrices built from the steerable filters
            weights = HConv2d.get_interpolation_weights(kernel_size,
                                                        m=m,
                                                        n_rings=n_rings,
                                                        angle_samples=N)
            low_pass_filter = np.dot(dft(N)[m, :], weights).T
            weights2filter_sampler.append(
                torch.from_numpy(low_pass_filter).to(get_default_complex())
            )
        return torch.stack(weights2filter_sampler)

    @torch.jit.export
    def get_filters(self) -> torch.Tensor:
        """
            Calculates filters in the form of weight matrices through performing
            single-frequency DFT on every ring obtained from sampling in the polar
            domain.

            Args:

            Returns:
                W: Complex filters [Order, Out_channels, In_channels, Kernel, Kernel]
            """
        x = rearrange(torch.matmul(self.weights2filters, self.ring_weights.to(get_default_complex())),
                      pattern="o (h w) (ic oc) -> o oc ic h w",
                      h=self.kernel_size,
                      w=self.kernel_size,
                      ic=self.in_channels // self.groups,
                      oc=self.out_channels)
        if self.phase:
            x *= torch.complex(real=torch.cos(self.phase_offset),
                               imag=-torch.sin(self.phase_offset))
        return x

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 in_max_order: int,
                 out_max_order: int,
                 mask_shape: int,
                 groups: int,
                 phase: bool = True,
                 n_rings: int = 3,
                 stddev: float = 0.4,
                 padding: int = 0,
                 stride: int = 1):
        """

            :param in_channels:
            :param out_channels:
            :param kernel_size:
            :param in_max_order:
            :param out_max_order:
            :param phase:
            :param n_rings:
            :param stddev:
            :return:
            """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.phase = phase
        self.groups = groups
        self.in_max_order = in_max_order
        self.out_max_order = out_max_order
        self.stddev = stddev
        self.n_rings = n_rings
        self.shape = [kernel_size, kernel_size, self.in_channels, self.out_channels]
        self.padding = padding
        self.stride = stride
        self._max_order = max(in_max_order + 1, out_max_order + 1)
        # Group num is correct
        assert in_channels % groups == 0, f"Number of input channels must be divisible by groups, got {in_channels} % {groups}"
        assert out_channels % groups == 0, f"Number of output channels must be divisible by groups, got {out_channels} % {groups}"
        # Se
        self.ring_weights = nn.Parameter(
            HConv2d.init_weights(in_channels=in_channels // groups,
                                 out_channels=out_channels,
                                 max_order=self._max_order,
                                 ring_count=self.n_rings),
            requires_grad=True
        )
        if self.phase:
            # Phase Offset - [Orders, In_Channels, Out_Channel]
            self.phase_offset = nn.Parameter(HConv2d.init_phase(self.in_channels // groups,
                                                                self.out_channels,
                                                                self._max_order),
                                             requires_grad=True)

        # Circular Masking
        # TODO: Improve circular mask, make gaussian etc..
        self.mask = torch.nn.Parameter(
            ~get_circular_mask(mask_shape)[None, None, None],
            requires_grad=False)

        # Weight2Filters
        self.weights2filters = torch.nn.Parameter(
            HConv2d.init_weights2filters(self._max_order, kernel_size, n_rings),
            requires_grad=False)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward propagation function for the harmonic convolution operation

            Args:
                x (deep tensor): input feature tensor dimensions [Batch Size, Order, Channels, Height, Width]

            Returns:
                R (deep tensor): output feature tensor obtained from harmonic convolution

            """
        # Expecting shape [Batch, Order, Channels, H, W]
        assert x.shape[1] == (self.in_max_order + 1), f"Unexpected number of orders(at 2), got shape = {x.shape}"
        assert x.shape[2] == self.in_channels, f"Unexpected number of channels(at 1), got shape = {x.shape}"
        assert x.shape[4] == x.shape[3], f"Expected square image, got shape = {x.shape}"

        r = self.hconv(x=x, weight=self.get_filters())
        return masked_fill(r, self.mask, 0)

    @torch.jit.export
    def conv2d(self, x: torch.Tensor, weight: torch.Tensor):
        return nn.functional.conv2d(input=x,
                                    weight=weight,
                                    groups=self.groups,
                                    padding=self.padding,
                                    stride=self.stride)

    @torch.jit.export
    def hconv(self,
              x: torch.Tensor,
              weight: torch.Tensor):
        """

            Args:
                x: [Batch Size, Channels, Order, Height, Width]
                weight:

            Returns:

            """
        x = rearrange(x, "b o c h w -> b (c o) h w")

        # NOTE: Prepare filters before convolution
        # as in tests.legacy.h_conv
        weights_over_out_order = []
        for out_order in range(self.out_max_order + 1):
            weights_over_in_order = []
            for in_order in range(self.in_max_order + 1):
                weight_order = out_order - in_order
                weight_filters = weight[abs(weight_order)]
                # NOTE: This is the same as the legacy code
                # but according to definition in the paper
                # zero order should have imaginary part too
                if weight_order == 0:
                    weight_filters.imag *= 0
                elif weight_order < 0:
                    weight_filters = weight_filters.conj()
                else:
                    weight_filters = weight_filters
                # Make conjugate for negative orders and zero out imaginary part for zero orders
                weights_over_in_order.append(weight_filters)
                _tmp = torch.stack(dim=2, tensors=weights_over_in_order)
            weights_over_out_order.append(
                _tmp
            )
        _weights = rearrange(torch.stack(dim=0, tensors=weights_over_out_order),
                             "out_o out_ch in_o in_ch kh kw -> (out_ch out_o) (in_ch in_o) kh kw")
        result = rearrange(self.conv2d(x=x, weight=_weights),
                           pattern="b (c o) h w -> b o c h w",
                           c=self.out_channels,
                           o=self.out_max_order + 1
                           )

        return result


class NegHConv2d(nn.Module):
    # TODO: Add reference to source codes
    @staticmethod
    def init_phase(inp_channel_count, out_channel_count, max_order):
        """
        Initializes phase dict with phase offsets

            out_channel_count (int): number of output channels
            in_max_order: maximum rotation order of input channels
            out_max_order: maximum rotation order of output channels
        Returns:
            phase_dict: initialized dict of phase offsets [
        """

        return torch.from_numpy(
            np.random.rand(2 * max_order - 1, out_channel_count, inp_channel_count, 1, 1) * 2. * np.pi
        )

    @staticmethod
    def init_weights(in_channels: int,
                     out_channels: int,
                     max_order: int,
                     ring_count: int):
        """
        Initializes the dict of weights

        Args:

            in_channels: input channels
            out_channels: output channels
            max_order: maximum rotation order from input and output
            ring_count (int): Number of rings for calculating the basis
        Returns:
            weights_dict: initialized dict of weights

        """
        weights = torch.zeros((2 * max_order - 1, ring_count, in_channels, out_channels),
                              dtype=torch.get_default_dtype())
        for order in range(-max_order + 1, max_order):
            sh = [ring_count, in_channels, out_channels]
            # Initializes the weights using initialization method of He
            stddev = 0.4 * np.sqrt(2.0 / np.prod(sh[:3]))
            weights[max_order + order - 1] = torch.normal(torch.zeros(*sh), stddev)
        # Flat input & output channels
        return weights.view((2 * max_order - 1, ring_count, in_channels * out_channels))

    @staticmethod
    def get_angle_samples_count(kernel_size):
        return int(np.maximum(np.ceil(np.pi * kernel_size), 101))

    @staticmethod
    def get_l2_neighbors(center, shape):
        lin = np.arange(shape) + 0.5
        jj, ii = np.meshgrid(lin, lin)
        ii = ii - center[1]
        jj = jj - center[0]
        return np.vstack((np.reshape(ii, -1), np.reshape(jj, -1)))

    @staticmethod
    def get_interpolation_weights(fs, m, n_rings, angle_samples):
        """
        Used to construct the steerable filters using Radial basis functions.
        The filters are constructed on the patches of n_rings using Gaussian
        interpolation. (Code adapted from the tf code of Worrall et al., CVPR, 2017)

        Args:
            fs (int): filter size for the H-net convolutional layer
            m (int): max. rotation order for the steerable filters
            n_rings (int): No. of rings for the steerable filters
            angle_samples (int): No. of angle samples

        Returns:
            norm_weights (numpy): contains normalized weights for interpolation
            using the steerable filters
        """

        mid = int(np.floor(fs / 2))
        # Variance of Gaussian resampling
        std_gauss = (mid / n_rings) / 2
        # We define below radii up to n_rings-0.5 (as in Worrall et al, CVPR 2017)
        radii = np.linspace(m != 0, mid - 2 * std_gauss, n_rings)
        # We define pixel centers to be at positions 0.5
        center_pt = np.asarray([fs, fs]) / 2.

        # Extracting the set of angles to be sampled

        # Choosing the sampling locations for the rings
        lin = (2 * np.pi * np.arange(angle_samples)) / angle_samples
        ring_locations = np.vstack([-np.sin(lin), np.cos(lin)])

        # Create interpolation coefficient coordinates
        coords = NegHConv2d.get_l2_neighbors(center_pt, fs)

        # getting samples based on the chosen center_pt and the coords
        radii = radii[:, np.newaxis, np.newaxis, np.newaxis]
        ring_locations = ring_locations[np.newaxis, :, :, np.newaxis]
        diff = radii * ring_locations - coords[np.newaxis, :, np.newaxis, :]
        dist2 = np.sum(diff ** 2, axis=1)

        # Convert distances to weightings
        weights = np.exp(-0.5 * dist2 / (std_gauss ** 2))  # For bandwidth of 0.5

        # Normalizing the weights to calibrate the different steerable filters
        norm = np.sum(weights,
                      axis=2,
                      keepdims=True)
        assert np.all(norm != 0), "Normalizing by zero weights"
        return np.divide(weights,
                         norm,
                         where=(norm != 0)
                         )

    @staticmethod
    def init_weights2filters(max_order, kernel_size, n_rings):
        # low pass filter
        # shape [Orders,  #rings, #Samples]

        N = NegHConv2d.get_angle_samples_count(kernel_size)
        weights2filter_sampler = []
        for m in range(-max_order + 1, max_order):
            # Get the basis matrices built from the steerable filters
            weights = NegHConv2d.get_interpolation_weights(kernel_size,
                                                           m=m,
                                                           n_rings=n_rings,
                                                           angle_samples=N)
            low_pass_filter = torch.from_numpy(
                np.dot(dft(N)[abs(m), :], weights).T
            ).to(get_default_complex())
            weights2filter_sampler.append(low_pass_filter.conj() if m < 0 else low_pass_filter)
        return torch.stack(weights2filter_sampler)

    @torch.jit.export
    def get_filters(self) -> torch.Tensor:
        """
        Calculates filters in the form of weight matrices through performing
        single-frequency DFT on every ring obtained from sampling in the polar
        domain.

        Args:

        Returns:
            W: Complex filters [Order, Out_channels, In_channels, Kernel, Kernel]
        """
        x = rearrange(torch.matmul(self.weights2filters, self.ring_weights.to(get_default_complex())),
                      pattern="o (h w) (ic oc) -> o oc ic h w",
                      h=self.kernel_size,
                      w=self.kernel_size,
                      ic=self.in_channels,
                      oc=self.out_channels)
        if self.phase:
            x *= torch.complex(real=torch.cos(self.phase_offset),
                               imag=-torch.sin(self.phase_offset))
        return x

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 in_max_order: int,
                 out_max_order: int,
                 mask_shape: int,
                 mask_type: str = "circular",
                 tukey_alpha: float = 0.4,
                 phase: bool = True,
                 n_rings: int = 3,
                 stddev: float = 0.4,
                 padding: int = 0,
                 stride: int = 1):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param in_max_order:
        :param out_max_order:
        :param phase:
        :param n_rings:
        :param stddev:
        :return:
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.phase = phase
        self.in_max_order = in_max_order
        self.out_max_order = out_max_order
        self.stddev = stddev
        self.n_rings = n_rings
        self.shape = [kernel_size, kernel_size, self.in_channels, self.out_channels]
        self.padding = padding
        self.stride = stride

        self._max_order = in_max_order + out_max_order + 1

        # Rings separated by orders
        self.ring_weights = nn.Parameter(
            NegHConv2d.init_weights(in_channels=in_channels,
                                    out_channels=out_channels,
                                    max_order=self._max_order,
                                    ring_count=self.n_rings),
            requires_grad=True
        )
        if self.phase:
            # Phase Offset - [Orders, In_Channels, Out_Channel]
            self.phase_offset = nn.Parameter(NegHConv2d.init_phase(self.in_channels,
                                                                   self.out_channels,
                                                                   self._max_order),
                                             requires_grad=True)
        # Circular Masking
        # TODO: Improve circular mask, make gaussian etc..
        self.mask_type = mask_type
        if mask_type == "circular":
            self.mask = torch.nn.Parameter(
                ~get_circular_mask(mask_shape)[None, None, None],
                requires_grad=False)
        elif mask_type == "tukey":
            self.mask = torch.nn.Parameter(
                torch.from_numpy(tukey_2d(mask_shape, alpha=tukey_alpha)[None, None, None]).type(
                    torch.get_default_dtype()),
                requires_grad=False)
        elif mask_type == "none":
            pass
        else:
            raise ValueError(f"Unknown mask type {mask_type}")

        # Weight2Filters
        self.weights2filters = torch.nn.Parameter(
            NegHConv2d.init_weights2filters(self._max_order, kernel_size, n_rings),
            requires_grad=False)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation function for the harmonic convolution operation

        Args:
            x (deep tensor): input feature tensor dimensions [Batch Size, Order, Channels, Height, Width]

        Returns:
            R (deep tensor): output feature tensor obtained from harmonic convolution

        """
        # Expecting shape [Batch, Order, Channels, H, W]
        assert len(x.shape) == 5, "Expecting [Batch Size, Order, Channels, Height, Width]"
        assert x.shape[1] == (2 * self.in_max_order + 1), f"Unexpected number of orders(at 1), got shape = {x.shape}"
        assert x.shape[
                   2] == self.in_channels, f"Unexpected number of channels(at 2), got shape = {x.shape}; expecting {self.in_channels}"
        assert x.shape[4] == x.shape[3], f"Expected square image, got shape = {x.shape}"

        r = self.hconv(x=x, weight=self.get_filters())
        if self.mask_type == "tukey":
            return r * self.mask
        elif self.mask_type == "circular":
            return masked_fill(r, self.mask, value=0)
        else:
            return r

    @torch.jit.export
    def conv2d(self, x: torch.Tensor, weight: torch.Tensor):
        return nn.functional.conv2d(input=x,
                                    weight=weight,
                                    padding=self.padding,
                                    stride=self.stride)

    @torch.jit.export
    def hconv(self,
              x: torch.Tensor,
              weight: torch.Tensor):
        """

        Args:
            x: [Batch Size, Channels, Order, Height, Width]
            weight:

        Returns:

        """
        x = rearrange(x, "b o c h w -> b (o c) h w")

        # NOTE: Prepare filters before convolution
        # as in tests.legacy.h_conv
        weights_over_out_order = []
        for out_order in range(-self.out_max_order, self.out_max_order + 1):
            weights_over_in_order = []
            for in_order in range(-self.in_max_order, self.in_max_order + 1):
                weight_order = out_order - in_order
                weight_filters = weight[self._max_order - 1 + weight_order]
                # NOTE: This is the same as the legacy code
                # but according to definition in the paper
                # zero order should have imaginary part too
                # Make conjugate for negative orders and zero out imaginary part for zero orders
                if weight_order == 0:
                    weight_filters.imag *= 0
                else:
                    pass
                weights_over_in_order.append(weight_filters)
                _tmp = torch.cat(dim=1, tensors=weights_over_in_order)
            weights_over_out_order.append(
                _tmp
            )
        _weights = torch.cat(dim=0, tensors=weights_over_out_order)
        return rearrange(self.conv2d(x=x, weight=_weights),
                         pattern="b (o c) h w -> b o c h w",
                         c=self.out_channels,
                         o=2 * self.out_max_order + 1
                         )


class HAct(nn.Module):
    """
    Non-linear activation function for the harmonic networks, operating on the complex domain.
    """

    def __init__(self,
                 number_of_ranks: int,
                 channels: int,
                 fnc="relu",
                 eps=1e-8,
                 has_weight=False,
                 has_bias=False):
        """

        :param number_of_ranks: # Input ranks number (dim 1)
        :param channels: Number of input channels (dim 2)
        :param fnc: Activation function operating on the magnitude
        :param eps: Epsilon for clamping the magnitude
        :param has_bias: Whether to use bias
        """
        super().__init__()
        assert hasattr(nn.functional, fnc), "Unknown activation function {fnc}"
        self.number_of_ranks = number_of_ranks
        self.channels = channels
        self.eps = eps
        self.has_bias = has_bias
        self.has_weight = has_weight
        self.activation_fnc = getattr(nn.functional, fnc)

        # Creating bias
        # Operating on magnitude [Batch Size, Order, Channels, Height, Width]
        if self.has_bias:
            self.bias = nn.Parameter(
                torch.zeros(dtype=torch.get_default_dtype(),
                            size=(1, number_of_ranks, channels, 1, 1)),
                requires_grad=True)
            nn.init.xavier_normal_(self.bias)
        # Creating weight multiplier before the activation
        if self.has_weight:
            self.weight = nn.Parameter(
                torch.ones(dtype=torch.get_default_dtype(),
                           size=(1, number_of_ranks, channels, 1, 1)),
                requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
        Activation on harmonic channels in complex domain
        :param x: channels (feature maps) [Batch Size, Order, Channels, Height, Width]
        :return:
        """
        assert x.dtype is get_default_complex()
        assert x.shape[1] == self.number_of_ranks, f"Unexpected number of orders(at 1), got shape = {x.shape}"
        magnitude = torch.clamp(x.abs(), min=self.eps)
        if self.has_weight:
            Rb = magnitude * self.weight
        else:
            Rb = magnitude

        if self.has_bias:
            Rb = Rb + self.bias
        else:
            Rb = Rb
            # NOTE: Because the zero division
        c = self.activation_fnc(Rb) / magnitude
        # NOTE: No need for masking if x is masked from HConv
        return c * x


class HPooling(nn.Module):
    def __init__(self,
                 number_of_ranks: int,
                 pooling_type: str = "avg",
                 kernel_size=(2, 2),
                 stride=(2, 2)):
        super().__init__()
        self.number_of_ranks = number_of_ranks
        self.pooling_type = pooling_type
        self.kernel_size = kernel_size
        self.stride = stride
        if pooling_type == "avg":
            self.pooling_layer = nn.AvgPool2d(kernel_size=kernel_size,
                                              stride=stride)
        elif pooling_type == "max":
            pass
        else:
            raise NotImplementedError(f"Unknown pooling type {pooling_type}")

    def forward(self, x: torch.Tensor):
        """
        Mean pooling on harmonic channels in the complex domain
        :param x: channels (feature maps) [Batch Size, Order, Channels, Height, Width]
        :return:
        """
        assert x.dtype is get_default_complex()
        assert (x.shape[-1] % 2 == 0) and (x.shape[-2] % 2 == 0)
        _x = rearrange(x, "b o c h w -> b (o c) h w")
        if self.pooling_type == "avg":
            real = rearrange(self.pooling_layer(_x.real),
                             "b (o c) h w -> b o c h w", o=self.number_of_ranks)
            imag = rearrange(self.pooling_layer(_x.imag),
                             "b (o c) h w -> b o c h w", o=self.number_of_ranks)
            return torch.complex(real=real, imag=imag)
        elif self.pooling_type == "max":
            _, indices = nn.functional.max_pool2d(input=_x.abs(),
                                                  kernel_size=self.kernel_size,
                                                  stride=self.stride,
                                                  return_indices=True)
            _x = retrieve_elements_from_indices(_x, indices)
            _x = rearrange(_x, "b (o c) h w -> b o c h w", o=self.number_of_ranks)
            return _x
        else:
            raise NotImplementedError(f"Unknown pooling type {self.pooling_type}")


class HUpSampling(nn.Module):
    def __init__(self,
                 number_of_ranks: int,
                 scale_factor: int = 2,
                 ):
        super().__init__()
        self.number_of_ranks = number_of_ranks
        self.scale_factor = scale_factor
        self.upsampling_layer = nn.Upsample(scale_factor=scale_factor,
                                            mode="bilinear")

    def forward(self, x: torch.Tensor):
        _x = rearrange(x, "b o c h w -> b (o c) h w")
        real = rearrange(self.upsampling_layer(_x.real),
                         "b (o c) h w -> b o c h w", o=self.number_of_ranks)
        imag = rearrange(self.upsampling_layer(_x.imag),
                         "b (o c) h w -> b o c h w", o=self.number_of_ranks)
        return torch.complex(real=real, imag=imag)


class HOut(nn.Module):
    """
    Output equivariant maps from the complex Harmonics in the real domain
    """

    def __init__(self,
                 keep_order_dim=True,
                 return_zero_order_phase=False):
        super().__init__()
        self.keep_order_dim = keep_order_dim
        assert ~return_zero_order_phase, "Not implemented return_zero_order_phase"

    def forward(self, x: torch.Tensor):
        """

        :param x: Complex Channels (feature maps) [Batch Size, Order, Channels, Height, Width]
        :return: Double Channels (feature maps)
        """
        if self.keep_order_dim:
            return x.abs()
        else:
            return rearrange(x.abs(), "b o c h w -> b (o c) h w")


# Harmonic Attention related modules

class HLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, number_of_ranks=3, bias=False):
        # Working independently on ranks
        assert bias is False, "Bias violates the equivariant property of the Harmonic Networks."
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.number_of_ranks = number_of_ranks
        # Normal linear layer??
        self.linear_layers = torch.nn.ModuleList([
            torch.nn.Linear(in_features=in_features,
                            out_features=out_features,
                            dtype=get_default_complex(),
                            bias=bias)
            for _ in range(number_of_ranks)
        ])

    def forward(self, x):
        assert x.dtype == get_default_complex(), f"Expected default complex dtype, got {x.dtype}"
        assert x.shape[1] == self.number_of_ranks, f"Expected {self.number_of_ranks} input ranks, got {x.shape[1]}"
        assert x.shape[-1] == self.in_features, f"Expected {self.in_features} input features, got {x.shape[-1]}"

        return torch.cat(dim=1, tensors=[
            self.linear_layers[i](x[:, i:i + 1])
            for i in range(self.number_of_ranks)])


class HSoftMax(torch.nn.Module):
    def __init__(self, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.activation_fnc = torch.nn.Softmax(dim=dim)

    def forward(self, x):
        magnitude = torch.clamp(x.abs(), min=self.eps)
        c = self.activation_fnc(magnitude) / magnitude
        return c * x


class HResidual(torch.nn.Module):
    def __init__(self,
                 block: torch.nn.Module,
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 number_of_ranks: Optional[int] = None,
                 upsampling: Optional[bool] = False):
        super().__init__()
        self.block = block
        self.upsampling = upsampling
        if upsampling:
            assert in_channels is not None, "Input channels must be provided for upsampling"
            assert out_channels is not None, "Output channels must be provided for upsampling"
            assert number_of_ranks is not None, "Number of ranks must be provided for upsampling"
            self.number_of_ranks = number_of_ranks
            self.upsampling_proj = torch.nn.ModuleList([
                torch.nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                                dtype=get_default_complex()) for _ in
                range(number_of_ranks)])

    def forward(self, x: torch.Tensor):
        if not self.upsampling:
            residual = x
        else:
            residual = torch.stack(dim=1, tensors=[
                self.upsampling_proj[order_idx](x[:, order_idx])
                for order_idx in range(self.number_of_ranks)
            ])
        x = self.block(x)
        return x + residual

    def train(self, mode: bool = True):
        super().train(mode=mode)
        self.block.train(mode=mode)


class H1x1Conv(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 number_of_ranks):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.number_of_ranks = number_of_ranks
        self.lin_proj = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=in_channels,
                                                             out_channels=out_channels,
                                                             kernel_size=1,
                                                             stride=1,
                                                             padding=0,
                                                             bias=False,
                                                             dtype=get_default_complex()) for _ in
                                             range(number_of_ranks)])

    def forward(self, x: torch.Tensor):
        x = torch.stack(dim=1, tensors=[
            self.lin_proj[order_idx](x[:, order_idx])
            for order_idx in range(self.number_of_ranks)
        ])
        return x


class MagnitudeResidual(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        logger.warning("This should be removed.")

    def forward(self, x, residual_magnitude):
        magnitude = torch.clamp(x.abs(), min=self.eps)
        norm = (magnitude + residual_magnitude) / torch.clamp(magnitude, min=self.eps)
        return norm * x


class FeatureMaps2Patches(torch.nn.Module):
    def __init__(self,
                 channels=4,
                 hidden_size=4,
                 number_of_ranks=3,
                 circular_masking=True,
                 shape: Optional[int] = None,
                 eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.circular_masking = circular_masking
        assert number_of_ranks == 3, "Number of ranks must be 3, (-1, 0, 1) for the Harmonic Convolution"
        assert (not circular_masking) or (shape is not None), "Shape must be provided for circular masking"
        if circular_masking:
            self.mask = torch.nn.Parameter(
                ~get_circular_mask(shape)[None, None, None],
                requires_grad=False)
        self.proj = HLinear(in_features=channels,
                            out_features=hidden_size,
                            number_of_ranks=number_of_ranks,
                            bias=False)

    def forward(self, x):
        assert x.ndim == 5, "Input must be 4D, (b, o, c,  h, w"
        assert x.shape[1] == 3, "Number of orders must be 3, (b, o, c, h, w)"
        # TODO: Masking by circle before projection
        # this should be more likely indexing by subset
        if self.circular_masking:
            x = masked_fill(x, self.mask, 0)
        x = rearrange(x, "b o c h w -> b o (h w) c")
        x = self.proj(x)
        return x


# Normalization layers for the Harmonic Networks
class HBatchNorm(nn.Module):
    def __init__(self,
                 channels: int,
                 eps=1e-8,
                 momentum=0.1,
                 affine=True):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.batch_norm = nn.BatchNorm3d(num_features=channels,
                                         eps=eps,
                                         momentum=momentum,
                                         affine=affine)

    def forward(self, x: torch.Tensor):
        """
        Batch normalization on harmonic channels in the complex domain
        :param x: channels (feature maps) [Batch Size, Order, Channels, Height, Width]
        :return:
        """
        assert x.dtype is get_default_complex()
        # To batch norm dimension
        magnitude = rearrange(torch.clamp(x.abs(), min=self.eps),
                              "b o c h w -> b c h w o")
        normalized_magnitude = self.batch_norm(magnitude)
        norm = rearrange(normalized_magnitude / torch.clamp(magnitude, min=1e-8),
                         "b c h w o -> b o c h w")
        # NOTE: No need for masking if x is masked from HConv
        return x * norm

    def __str__(self):
        return "HBatchNorm()"

    def train(self, mode: bool = True):
        super(HBatchNorm, self).train(mode=mode)
        self.batch_norm.train(mode=mode)


class HNorm(nn.Module):

    def __init__(self,
                 num_channels: int,
                 norm_type='layer',
                 eps=1e-8,
                 affine_alpha: bool = False,
                 feature_maps: str = "channels"):
        super().__init__()
        assert feature_maps in ["channels", "patches"], "Unknown feature_maps, expected 'channels' or 'patches'"
        assert norm_type == "layer", "Only layer normalization is supported"
        # Expected input shape
        self.eps = eps
        self.feature_maps = feature_maps
        self.norm_type = norm_type
        self.affine_alpha = affine_alpha
        if feature_maps == "channels":
            self.input_ndim = 5
            self._dims = (2, 3, 4)
            self.alpha = nn.Parameter(torch.ones(1, 1, num_channels, 1, 1), requires_grad=True)
        elif feature_maps == "patches":
            self.input_ndim = 4
            self._dims = (2, 3)
            self.alpha = nn.Parameter(torch.ones(1, 1, 1, num_channels), requires_grad=True)
        else:
            raise ValueError(f"Unknown feature_maps {feature_maps}")
            # TODO: Epsilon by the floating number precision

    def forward(self, x: torch.Tensor):
        """
        Batch normalization on harmonic channels in the complex domain
        :param x: channels (feature maps) [Batch Size, Order, Channels, Height, Width]
        :return:
        """
        assert x.ndim == self.input_ndim, f"Expected {self.input_ndim}D, got {x.ndim}D"
        if self.norm_type == "layer":
            mean = torch.mean(x,
                              dim=self._dims,
                              keepdim=True)
            std = torch.std(x,
                            dim=self._dims,
                            keepdim=True)

            if self.affine_alpha:
                return (x - mean) / torch.clamp(std, min=self.eps) * self.alpha
            else:
                return (x - mean) / torch.clamp(std, min=self.eps)

        elif self.norm_type == "batch":
            raise NotImplementedError("Batch normalization is not implemented yet")
        else:
            raise ValueError(f"Unknown norm_type {self.norm_type}")

    def __str__(self):
        return f"HNorm({self.norm_type})"

    def train(self, mode: bool = True):
        super(HNorm, self).train(mode=mode)


class HNormAct(nn.Module):
    """
    Batch normalization and activation function on magnitude
    """

    def __init__(self,
                 act_fnc: str,
                 channels: int,
                 eps: float = 1e-8,
                 momentum: float = 0.1,
                 input_type: str = "channels",
                 affine: bool = True):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.input_type = input_type
        if input_type == "channels":
            self.batch_norm = nn.BatchNorm3d(num_features=channels,
                                             eps=eps,
                                             momentum=momentum,
                                             affine=affine)
        elif input_type == "patches":
            self.batch_norm = nn.BatchNorm2d(num_features=channels,
                                             eps=eps,
                                             momentum=momentum,
                                             affine=affine)
        else:
            raise ValueError(f"Unknown input_type {input_type}, expected 'channels' or 'patches'")
        assert hasattr(nn.functional, act_fnc), "Unknown activation function {fnc}"
        self.activation_fnc = getattr(nn.functional, act_fnc)

    def forward(self, x: torch.Tensor):
        assert x.dtype is get_default_complex()
        # Operating on magnitude [Batch Size, Order, Channels, Height, Width]
        i_magnitude = torch.clamp(x.abs(), min=self.eps)
        # BatchNorm with affine
        # TODO: Write own BatchNorm with affine to avoid the rearrange
        # TODO: Clamp after??
        # NOTE:
        if self.input_type == "channels":
            o_magnitude = rearrange(i_magnitude, pattern="b o c h w -> b c h w o")
            o_magnitude = self.batch_norm(o_magnitude)
            o_magnitude = rearrange(o_magnitude, pattern="b c h w o -> b o c h w")
        elif self.input_type == "patches":
            o_magnitude = rearrange(i_magnitude, pattern="b o n c -> b c o n")
            o_magnitude = self.batch_norm(o_magnitude)
            o_magnitude = rearrange(o_magnitude, pattern="b c o n -> b o n c")
        else:
            raise ValueError(f"Unknown input_type {self.input_type}, expected 'channels' or 'patches'")
        # Activation
        o_magnitude = self.activation_fnc(o_magnitude)
        # Transform back to complex
        norm = self.activation_fnc(o_magnitude) / i_magnitude
        # NOTE: Zero magnitudes remain zeros
        return norm * x

    def train(self, mode: bool = True):
        super(HNormAct, self).train(mode=mode)
        self.batch_norm.train(mode=mode)


class VanillaBackbone(torch.nn.Module):
    def __init__(self,
                 channels: int,
                 number_of_ranks: int = 3,
                 in_channels: int = 1,
                 input_shape: int = 64,
                 mask_type: str = "tukey",
                 kernel_size: int = 15,
                 n_rings: int = 3,
                 eps=1e-6):
        super().__init__()
        assert number_of_ranks == 3, "Number of ranks must be 3, (-1, 0, 1) for the Harmonic Convolution"
        self.hidden_size = channels
        self.eps = eps
        self.to_feature_maps = ComplexImg2H(circular_mask=True,
                                            alpha=0.4,
                                            input_shape=input_shape)
        self.conv0 = NegHConv2d(in_channels=in_channels,
                                out_channels=channels,
                                kernel_size=kernel_size,
                                n_rings=n_rings,
                                in_max_order=0,
                                out_max_order=1,
                                mask_shape=input_shape,
                                mask_type=mask_type,
                                phase=True,
                                padding=(kernel_size - 1) // 2,
                                stride=1)
        self.hnormact0 = HNormAct(act_fnc="relu", channels=channels, affine=True)
        self.conv1 = NegHConv2d(in_channels=channels,
                                out_channels=channels,
                                kernel_size=kernel_size,
                                in_max_order=1,
                                out_max_order=1,
                                n_rings=n_rings,
                                mask_shape=input_shape,
                                mask_type=mask_type,
                                phase=True,
                                padding=(kernel_size - 1) // 2,
                                stride=1)
        self.hnormact1 = HNormAct(act_fnc="relu", channels=channels, affine=True)
        self.avg0 = HPooling(number_of_ranks=3,
                             kernel_size=(2, 2),
                             stride=(2, 2))

        self.conv2 = NegHConv2d(in_channels=channels,
                                out_channels=channels,
                                kernel_size=kernel_size,
                                in_max_order=1,
                                out_max_order=1,
                                n_rings=n_rings,
                                mask_shape=input_shape // 2,
                                mask_type=mask_type,
                                padding=(kernel_size - 1) // 2,
                                phase=True,
                                stride=1)
        self.hnormact2 = HNormAct(act_fnc="relu", channels=channels, affine=True)
        self.avg1 = HPooling(number_of_ranks=3,
                             kernel_size=(2, 2),
                             stride=(2, 2))

    def forward(self, x):
        x = self.to_feature_maps(x)
        x = self.conv0(x)
        x = self.hnormact0(x)
        identity = x
        x = self.conv1(x)
        x = self.hnormact1(x)
        x = x + identity
        x = self.avg0(x)
        identity = x
        x = self.conv2(x)
        x = self.hnormact2(x)
        x = x + identity
        x = self.avg1(x)
        return x


class HDropOut(torch.nn.Module):
    def __init__(self, p, eps: Optional[float] = None):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        if self.training and self.p > 0:
            # need to have the same dropout mask for real and imaginary part,
            # this not a clean solution!
            # NOTE: Taken from: https://github.com/wavefrontshaping/complexPyTorch/blob/2044cb077b3f139d59dff56abc378b1457de40d6/complexPyTorch/complexLayers.py#L38
            mask = torch.ones(*x.shape, dtype=torch.float32, device=x.device)
            mask = torch.nn.functional.dropout(mask, self.p, self.training)
            # TODO: Consider clipping the mask due to the Phase information
            if self.eps is not None:
                mask = torch.clamp(mask, min=self.eps)
            mask.type(x.dtype)

            return mask * x
        else:
            return x


class HDropOut2D(torch.nn.Module):
    def __init__(self, p, eps: Optional[float] = None):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        if self.training and self.p > 0:
            #NOTE: Taken from: https://github.com/wavefrontshaping/complexPyTorch/blob/2044cb077b3f139d59dff56abc378b1457de40d6/complexPyTorch/complexFunctions.py#L205C1-L211C22
            mask = torch.ones(*x.shape, dtype=torch.float32, device=x.device)
            mask = torch.nn.functional.dropout2d(mask, self.p, self.training)
            # TODO: Consider clipping the mask due to the Phase information
            if self.eps is not None:
                mask = torch.clamp(mask, min=self.eps)
            mask.type(x.dtype)

            return mask * x
        else:
            return x


# TODO: This is taken from timm
# NOTE: Make notice https://github.com/huggingface/pytorch-image-models/blob/70ccf00c95a2d78a166cca24ef6adbca46f47c2a/timm/layers/drop.py#L170
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = torch.empty(shape, dtype=torch.get_default_dtype(), device=x.device).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class StochasticDepthResidual(nn.Module):
    # TODO: Stochastic depth behaving as the residual
    # TODO: Testing
    def __init__(self,
                 module: torch.nn.Module,
                 survivor_p: float):
        super().__init__()
        # Module to be skipped
        self.module: torch.nn.Module = module
        self.m = torch.distributions.bernoulli.Bernoulli(probs=survivor_p)
        self.survivor_p = survivor_p

    def forward(self, x: torch.Tensor):
        if self.training:
            if (self.survivor_p == 1.0) or (self.m.sample() == 1):
                return self.module(x) + x
            else:
                return x
        else:
            return self.survivor_p * self.module(x) + x

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        self.module.train(mode=mode)
        return self


class Patches2Channels(nn.Module):
    def __init__(self,
                 spatial_shape: int,  # (H, W)
                 ):
        super().__init__()
        self.spatial_shape = spatial_shape

    def forward(self, x):
        assert x.ndim == 4, "Input must be 4D, (b, o, n, c)"
        x = rearrange(x,
                      "b o (h w) c -> b o c h w",
                      h=self.spatial_shape,
                      w=self.spatial_shape)
        return x


class Channels2Patches(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return rearrange(x, "b o c h w -> b o (h w) c")

class GAPMLP(nn.Module):
    def __init__(self,
                 in_channels: int = 16,
                 hidden_dim: int = 64,
                 masking_dim: int = 16,
                 drop: float = 0.0,
                 mask_type: str = 'circular',
                 tukey_alpha: float = 0.4,
                 num_classes: int = 10):
        super().__init__()
        self.ln1 = torch.nn.Linear(in_channels, hidden_dim)
        self.act = torch.nn.GELU()
        if drop > 0:
            self.drop = torch.nn.Dropout(p=drop)
        else:
            self.drop = lambda x: x
        self.ln2 = torch.nn.Linear(hidden_dim, num_classes)
        # TODO: Add tukey masking
        self.mask_type = mask_type
        if mask_type == 'circular':
            self.mask = torch.nn.Parameter(
                ~get_circular_mask(masking_dim)[None, None, None, ...],
                requires_grad=False)
        elif mask_type == 'tukey':
            self.mask = torch.nn.Parameter(
                torch.from_numpy(tukey_2d(masking_dim, alpha=tukey_alpha)[None, None, None]).type(
                    torch.get_default_dtype()),
                requires_grad=False)
        elif mask_type == 'none':
            pass
        else:
            raise ValueError(f'Unknown mask type: {mask_type}')

    def forward(self, x):
        # Masking
        if self.mask_type == 'tukey':
            x = x * self.mask
        elif self.mask_type == 'circular':
            x = masked_fill(x, self.mask, 0)
        else:  # No masking
            pass
        x = rearrange(x.mean(dim=(3, 4)),
                      'b o c -> b (c o)')
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.ln2(x)
        return x