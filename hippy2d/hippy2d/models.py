import torch
from torch import nn
import lightning as L
from typing import List, Any, Dict, Optional
from loguru import logger
from hippy2d.harmformer import HConv2d, HNormAct, HOut, ComplexImg2H, DropPath, HPooling, GAPMLP
from hippy2d.optimal_invariant_cnn import ComplexBaseBlock

from hippy2d.utils import get_default_complex   

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

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

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
            self.log(f'{k}_loss', loss)
            self.log(f'{k}_acc', acc, prog_bar=True)
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
            _, loss, acc = self.shared_step(x, y)
            # NOTE: Return the validation loss with key 'val'
            # Other datasets are only for debugging purposes
            self.log(f'{k}_loss', loss)
            self.log(f'{k}_acc', acc)

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
                 blocks = [3, 3, 3], 
                 channels= [4, 8, 16],
                 zero_order_scaling:bool = False,
                 classification:bool = True):
        super(PrototypeOptimalInvCNN, self).__init__()
        self.in_channels = in_channels
        out_channels = in_channels
        self.blocks = []
        for block_idx, num_blocks in enumerate(blocks):
            block = []
            for _ in range(num_blocks):
                out_channels = channels[block_idx]
                block.append(ComplexBaseBlock(in_channels=in_channels,
                                              out_channels=out_channels,
                                              zero_order_scaling=zero_order_scaling,
                                              input_size=input_size,
                                              subsampling=False))
                in_channels = out_channels

            out_channels = channels[block_idx + 1] if block_idx + 1 < len(channels) else out_channels
            # Subsampling block at the end
            block.append(ComplexBaseBlock(in_channels=in_channels,
                                          out_channels=out_channels,
                                          zero_order_scaling=zero_order_scaling,
                                          input_size=input_size,
                                          subsampling=True if block_idx < len(blocks) - 1 else False))

            in_channels = out_channels
            input_size //= 2  # Reduce input size by half for the next block
            self.blocks.append(torch.nn.Sequential(*block))
        self.blocks = torch.nn.Sequential(*self.blocks)

        self.classification = classification
        if classification:
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.flat = torch.nn.Flatten()
            self.classifier = torch.nn.Linear(in_features=out_channels, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        if self.classification:
            x = self.pool(x)
            x = self.flat(x)
            x = self.classifier(x)
    
        return x