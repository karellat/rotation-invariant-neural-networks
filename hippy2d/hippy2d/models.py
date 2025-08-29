import torch
from torch import nn
import lightning as L
from typing import List, Any, Dict, Optional
from loguru import logger
from hippy2d.harmformer import HConv2d, HNormAct, HOut, ComplexImg2H, DropPath, HPooling, GAPMLP
from hippy2d.optimal_invariant_cnn import ComplexBaseBlock
from hippy2d.e2sfcnn import ExpE2SFCNN

from hippy2d.utils import get_default_complex   
from torch.nn import functional as F

# Lightning wrapper
class InvNet(L.LightningModule):
    def __init__(self,
                 input_shape: List[int],
                 model: nn.Module,
                 optimizer_name: str = 'AdamW',
                 optimizer_hparams: Dict[str, Any] = dict(lr=1e-3),
                 lr_name: str = "MultiStepLR",  # None
                 label_smoothing: float = 0.0,
                 lr_hparams: Dict[str, Any] = dict(milestones=[3, 6, 9], gamma=0.1),
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.loss_fnc = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.input_shape = input_shape

    def configure_optimizers(self):
        # NOTE: Please see https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        if hasattr(torch.optim, self.hparams.optimizer_name):
            optimizer = getattr(torch.optim, self.hparams.optimizer_name)(self.parameters(),
                                                                          **self.hparams.optimizer_hparams)
        else:
            raise RuntimeError(f'Unknown optimizer: "{self.hparams.optimizer_name}"')

        if self.hparams.lr_name == "None":
            return optimizer
        elif hasattr(torch.optim.lr_scheduler, self.hparams.lr_name):
            scheduler = getattr(torch.optim.lr_scheduler, self.hparams.lr_name)(optimizer, **self.hparams.lr_hparams)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}
        else:
            raise RuntimeError(f'Unknown optimizer: "{self.hparams.optimizer_name}"')

    def shared_step(self, x, y):
        y_hat = self.model(x)
        loss = self.loss_fnc(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).to(torch.get_default_dtype()).mean()
        return y_hat, loss, acc

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss, acc = self.shared_step(x, y)

        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch if type(batch) is dict else {'val': batch}
        for k, v in batch.items():
            if len(v) == 1:
                x, y = v[0]
            else:
                x, y = v
            preds, loss, acc = self.shared_step(x, y)
            # NOTE: Return the validation loss with key 'val'
            # Other datasets are only for debugging purposes
            self.log(f'{k}_loss', loss, sync_dist=True)
            self.log(f'{k}_acc', acc, prog_bar=True, sync_dist=True)
            if k == 'val':
                res_preds = preds
                res_loss = loss
                res_acc = acc
        return {
            'loss': res_loss,
            'acc': res_acc,
            'preds': res_preds
        }

    def test_step(self, batch, batch_idx):
        batch = batch if type(batch) is dict else {'test': batch}
        for k, v in batch.items():
            x, y = v
            y_hat, loss, acc = self.shared_step(x, y)
            # NOTE: Return the validation loss with key 'val'
            # Other datasets are only for debugging purposes
            self.log(f'{k}_loss', loss, sync_dist=True)
            self.log(f'{k}_acc', acc, sync_dist=True)
            x_90, x_180, x_270 = torch.rot90(x, 1, [-2, -1]), torch.rot90(x, 2, [-2, -1]), torch.rot90(x, 3, [-2, -1])
            y_hat_90, _, _ = self.shared_step(x_90, y)
            y_hat_180, _, _ = self.shared_step(x_180, y)
            y_hat_270, _, _ = self.shared_step(x_270, y)

            y_hat_rd = torch.stack([y_hat_90, y_hat_180, y_hat_270], dim=1)
            norm = torch.norm(y_hat_rd - y_hat[:, None], dim=-1)
            sim = F.cosine_similarity(y_hat_rd, y_hat[:, None], dim=-1) 
            norm_max = norm.max()
            cos_min = sim.min()

            norm = norm.mean()
            cos_sim = sim.mean()

            self.log(f'{k}_rci_norm', norm, sync_dist=True)
            self.log(f'{k}_rci_sim', cos_sim, sync_dist=True)
            self.log(f'{k}_rci_norm_max', norm_max, sync_dist=True, reduce_fx="max")
            self.log(f'{k}_rci_sim_min', cos_min, sync_dist=True, reduce_fx="min")

# Model Zoo
class Resnet18(nn.Module): 
    def __init__(self,
                 in_channels: int,
                 num_classes: int):
        super().__init__()
        from torchvision.models import resnet18
        self.resnet = resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor): 
        return self.resnet(x)

class Blockv3(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 in_max_order: int,
                 out_max_order: int,
                 mask_shape: int,
                 residual: bool,
                 kernel_size: int,
                 n_rings: int,
                 drop_path: Optional[nn.Module] = None):
        super().__init__()
        self.hconv = HConv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             in_max_order=in_max_order,
                             out_max_order=out_max_order,
                             tukey_window=True,
                             tukey_alpha=0.4,
                             mask_shape=mask_shape,
                             kernel_size=kernel_size,
                             padding=(kernel_size - 1) // 2,
                             n_rings=n_rings)
        self.hnorm_act = HNormAct(act_fnc="relu",
                                  channels=out_channels,
                                  affine=True)
        self.drop_path = drop_path
        self.residual = residual
        if residual:
            assert in_max_order == out_max_order, "Residual connections require the same max order"
            self.upsampling = in_channels != out_channels
            self.number_orders = out_max_order + 1
            if self.upsampling:  # 1x1 conv for upsampling per Order
                self.proj = torch.nn.ModuleList([
                    torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                    dtype=get_default_complex()) for _ in
                    range(self.number_orders)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.hconv(x)
        x = self.hnorm_act(x)
        if self.residual:
            if self.upsampling:
                identity = torch.stack(dim=1, tensors=[
                    self.proj[order_idx](identity[:, order_idx])
                    for order_idx in range(self.number_orders)
                ])
            if self.drop_path is not None:
                x = self.drop_path(x)
            x = x + identity
        return x

    def __repr__(self):
        return f"ResBlockv-{self.number_orders}-{self.in_channels}->{self.out_channels}{'; Residual' if self.residual else ''}"

# H-NeXt taken from Harmformer implementation
class ResHNeXtv3(nn.Module):
    def __init__(self,
                 num_classes: int,
                 model_str: str = "8x3,MP,16x3,MP,32x3",
                 maximum_order=1,
                 in_channels=1,
                 activation_name="relu",
                 kernel_size: int = 15,
                 n_rings: int = 3,
                 input_size=64,
                 drop_path_rate: float = 0.0,
                 _return_phase_dim=False):

        super().__init__()
        model_str = model_str.upper().split(',')
        assert len(model_str) > 0, "No channels specified"

        self.maximum_order = maximum_order
        self.in_channels = in_channels
        img2input = ComplexImg2H(circular_mask=True,
                                 input_shape=input_size,
                                 alpha=0.4)
        self.activation_name = activation_name
        self.kernel_size = kernel_size
        self.n_rings = n_rings
        network_layers = []
        _last_max_order = 0
        _last_out_channels = in_channels
        _last_channel_size = input_size

        block_idx = 0
        for idx, layer_str in enumerate(model_str):
            if layer_str == "MP":
                assert idx != 0, "No avg pooling at the first layer"
                assert _last_channel_size % 2 == 0
                # Add residual connections for the pooling layers
                network_layers.append(HPooling(number_of_ranks=self.maximum_order + 1,
                                               kernel_size=(2, 2),
                                               pooling_type="avg",
                                               stride=(2, 2),
                                               ))
                _last_channel_size = int(_last_channel_size / 2)
            elif "X" in layer_str:
                channels, repeats = map(int, layer_str.split('X'))
                block_layers = []
                # Linear decay rule for drop path
                dropout_rate = drop_path_rate * block_idx
                logger.debug(f"Layer {block_idx} - {channels}x{repeats}, Drop {dropout_rate}")
                block_idx += 1
                for layer_idx in range(repeats):
                    # Add residual connections, skip first layer
                    if layer_idx == 0 and idx == 0:
                        residual = False
                        in_order = 0
                    else:
                        residual = True
                        in_order = self.maximum_order
                    block_layers.append(Blockv3(in_channels=_last_out_channels,
                                                out_channels=channels,
                                                in_max_order=in_order,
                                                out_max_order=self.maximum_order,
                                                mask_shape=_last_channel_size,
                                                residual=residual,
                                                kernel_size=self.kernel_size,
                                                n_rings=self.n_rings,
                                                drop_path=None if dropout_rate == 0 else DropPath(dropout_rate)))
                    _last_out_channels = channels
                network_layers.append(torch.nn.Sequential(*block_layers))
            else:
                raise ValueError(f"Unknown layer type {layer_str}")

        equivariant_stack = HOut(keep_order_dim=True, return_zero_order_phase=_return_phase_dim)
        self.hnext = nn.Sequential(img2input, *network_layers, equivariant_stack)
        self.classifier = GAPMLP(in_channels=2*_last_out_channels,
                                 masking_dim=_last_channel_size,
                                 num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        x = self.hnext(x)
        x = self.classifier(x)
        return x

# Optimal Convolution
class PrototypeOptimalInvCNN(torch.nn.Module): 
    def __init__(self, 
                 in_channels:int = 3,
                 input_size:int = 64,
                 num_classes:int = 10,
                 filter_size: int = 15,
                 n_blocks=3, 
                 m_layers=3,
                 init_channels=4,
                 max_order=4,
                 learnable_radial_basis: int = 0, 
                 zero_order_scaling:bool = False,
                 classification:bool = True, 
                 channels_masking:bool = True):
        super(PrototypeOptimalInvCNN, self).__init__()
        if type(filter_size) is int:
            filter_size = [filter_size] * n_blocks
        elif type(filter_size) is list:
            assert len(filter_size) == n_blocks, "If a list of filter sizes is provided, it must have length n_blocks"
        else:
            raise TypeError("filter_size must be an int or a list of ints")
        if type(init_channels) is int:
            channels = [init_channels * (2 ** i) for i in range(n_blocks + 1)]
        elif type(init_channels) is list:
            channels = init_channels
            assert len(channels) == n_blocks, "If a list of channels is provided, it must have length n_blocks + 1"
        else: 
            raise TypeError("init_channels must be an int or a list of ints")

        self.masking_channels = "tukey" if channels_masking else "none"
        self.in_channels = in_channels
        self.max_order = max_order
        out_channels = in_channels
        self.blocks = []
        for block_idx in range(n_blocks):
            block = []
            for _ in range(m_layers):
                out_channels = channels[block_idx]
                block.append(ComplexBaseBlock(in_channels=in_channels,
                                              out_channels=out_channels,
                                              max_order=self.max_order,
                                              filter_size=filter_size[block_idx],
                                              zero_order_scaling=zero_order_scaling,
                                              input_size=input_size,
                                              learnable_radial_basis=learnable_radial_basis,
                                              channels_masking=self.masking_channels,
                                              subsampling=False))
                in_channels = out_channels

            out_channels = channels[block_idx + 1] if block_idx + 1 < len(channels) else out_channels
            # Subsampling block at the end
            block.append(ComplexBaseBlock(in_channels=in_channels,
                                          out_channels=out_channels,
                                          filter_size=filter_size[block_idx],
                                          max_order=self.max_order,
                                          zero_order_scaling=zero_order_scaling,
                                          input_size=input_size,
                                          learnable_radial_basis=learnable_radial_basis,
                                          channels_masking=self.masking_channels,
                                          subsampling=True if block_idx < n_blocks - 1 else False))

            in_channels = out_channels
            input_size //= 2  # Reduce input size by half for the next block
            self.blocks.append(torch.nn.Sequential(*block))
        self.blocks = torch.nn.Sequential(*self.blocks)

        self.classification = classification
        if classification:
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.flat = torch.nn.Flatten()
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=out_channels, out_features=64),
                torch.nn.BatchNorm1d(num_features=64),
                torch.nn.ELU(),
                torch.nn.Linear(in_features=64, out_features=num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        if self.classification:
            x = self.pool(x)
            x = self.flat(x)
            x = self.classifier(x)
    
        return x

class PrototypeTiny(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 10, 
                 input_size: int = 28,
                 scale_channels: int = 1):
        super(PrototypeTiny, self).__init__()

        # 28 px
        self.layer_1 = ComplexBaseBlock(in_channels=in_channels,
                                        input_size=input_size,
                                        out_channels=16 * scale_channels, 
                                        max_order=3, 
                                        filter_size=7,
                                        conv_padding=1,
                                        channels_masking="none",
                                        subsampling=False)
        # 24 px 
        self.layer_2 = ComplexBaseBlock(in_channels=16 * scale_channels,
                                        input_size=input_size - 4,
                                        out_channels=32 * scale_channels,
                                        max_order=3,
                                        filter_size=5,
                                        conv_padding=2,
                                        channels_masking="none",
                                        subsampling=True)
                                        
        # 12 px 
        self.layer_3 = ComplexBaseBlock(in_channels=32 * scale_channels,
                                        input_size=(input_size - 4) // 2,
                                        out_channels=32 * scale_channels,
                                        max_order=3,
                                        filter_size=5,
                                        conv_padding=2,
                                        channels_masking="none",
                                        subsampling=False)

        self.layer_4 = ComplexBaseBlock(in_channels=32 * scale_channels,
                                        input_size=(input_size - 4) // 2,
                                        out_channels=32 * scale_channels,
                                        max_order=3,
                                        filter_size=5,
                                        conv_padding=2,
                                        channels_masking="none",
                                        subsampling=True)

        # 6 px
        self.layer_5 = ComplexBaseBlock(in_channels=32 * scale_channels,
                                        input_size=(input_size - 4) // 4,
                                        out_channels=48 * scale_channels,
                                        max_order=3,
                                        filter_size=5,
                                        conv_padding=2,
                                        channels_masking="none",
                                        subsampling=False)

        self.layer_6 = ComplexBaseBlock(in_channels=48 * scale_channels,
                                        input_size=(input_size - 4) // 4,
                                        out_channels=64 * scale_channels,
                                        max_order=3,
                                        filter_size=5,
                                        conv_padding=2,
                                        channels_masking="none",
                                        subsampling=False)

        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flat = torch.nn.Flatten()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=64 * scale_channels, out_features=64 * scale_channels),
            torch.nn.BatchNorm1d(num_features=64 * scale_channels),
            torch.nn.ELU(),
            torch.nn.Linear(in_features=64 * scale_channels, out_features=num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 28 px
        x = self.layer_1(x)
        # 24 px
        x = self.layer_2(x)
        # 12 px 
        x = self.layer_3(x)
        x = self.layer_4(x)
        # 6 px
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.classifier(x)
        return x

class RotMNISTE2CNN(ExpE2SFCNN):
    def __init__(self,
                 in_channels: int=1,
                 num_classes: int=10,
                 layer_type:str="gated_norm_shared",
                 restrict:int = 0,
                 N:int=-3,
                 fixparams:bool=True,
                 F:Optional[int]=None,
                 J:Optional[int]=None,
                 sigma:Optional[float]=None,
                 deltaorth:bool=False,
                 antialiasing:float=0.0,
                 sgsize:Optional[int]=None,
                 flip:bool=False
                 ):
        """
        RotMNIST E2CNN model for rotation-invariant classification.
        
        Args:
            n_inputs (int, optional): Number of input channels. Defaults to 1.
            n_outputs (int, optional): Number of output classes. Defaults to 10.
            layer_type (str, optional): Type of fiber for the EXP model. Defaults to "gated_norm_shared".
            restrict (int, optional): Layer where to restrict SFCNN from E(2) to SE(2). 
                Defaults to 0. Use -1 to disable restriction.
            N (int, optional): Size of cyclic group for GCNN and maximum frequency for HNET. 
                Defaults to -3.
            fixparams (bool, optional): Keep the number of parameters of the model fixed 
                by adjusting its topology. Defaults to True.
            F (Optional[int], optional): Frequency cut-off: maximum frequency at radius "r" 
                is "F*r". If None, no frequency cut-off is applied. Defaults to None.
            J (Optional[int], optional): Number of additional frequencies in the interwiners 
                of finite groups. If None, uses default value. Defaults to None.
            sigma (Optional[float], optional): Width of the rings building the bases 
                (std of the gaussian window). If None, uses default value. Defaults to None.
            deltaorth (bool, optional): Use delta orthogonal initialization in conv layers. 
                Defaults to False.
            antialiasing (float, optional): Std for the gaussian blur in the max-pool layer. 
                If zero, standard maxpooling is performed. Defaults to 0.0.
            sgsize (Optional[int], optional): Number of rotations in the subgroup to restrict 
                to in the EXP e2sfcnn models. If None, uses full group. Defaults to None.
            flip (bool, optional): Use also reflection equivariance in the EXP model. 
                Defaults to False.
        """
        super().__init__(in_channels, 
                         num_classes,
                         layer_type=layer_type,
                         restrict=restrict,
                         N=N,
                         fix_param=fixparams, fco=F, J=J, sigma=sigma,
                         deltaorth=deltaorth, antialias=antialiasing, sgsize=sgsize,
                         flip=flip)