import torch
from torch import nn
import lightning as L
from loguru import logger
from einops import rearrange
from torch.nn import functional as F
from typing import List, Any, Dict, Optional

from hippy2d import conv_factory
from hippy2d.blocks import ResnetBlock, TimmBasicBlock
from hippy2d.e2sfcnn import ExpE2SFCNN
from hippy2d.learnable import LearnableFlusser
from hippy2d.utils import get_default_complex   
from hippy2d.datasets import ROTATED_TEST_SET_KEY
from hippy2d.optimal_invariant_cnn import ComplexInvariantConv2D
from hippy2d.harmformer import HConv2d, HNormAct, HOut, ComplexImg2H, DropPath, HPooling, GAPMLP

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

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Handle sequential combined test loader with both regular and rotated test sets.
        
        With CombinedLoader sequential mode, batches come in order:
        - First all batches from 'test' loader (dataloader_idx=0)
        - Then all batches from 'rotated_test' loader (dataloader_idx=1)
        
        The dataloader_idx parameter indicates which loader in the sequence.
        """
        # Get dataloader name from the saved list in datamodule
        datamodule = self.trainer.datamodule
        if hasattr(self.trainer.datamodule, 'test_loaders_names'):
            dataloader_names = self.trainer.datamodule.test_loaders_names
            dataloader_name = dataloader_names[dataloader_idx]
        else:
            dataloader_name = 'test'
        
        x, y = batch
        
        if dataloader_name == 'test': 
            y_hat, loss, acc = self.shared_step(x, y)
            
            # Log basic metrics
            self.log(f'test_loss', loss, sync_dist=True, batch_size=datamodule.test_batch_size, add_dataloader_idx=False)
            self.log(f'test_acc', acc, sync_dist=True, batch_size=datamodule.test_batch_size, add_dataloader_idx=False)
            
            # Compute rotation consistency metrics (RCI) with N=4 rotations
            x_90, x_180, x_270 = torch.rot90(x, 1, [-2, -1]), torch.rot90(x, 2, [-2, -1]), torch.rot90(x, 3, [-2, -1])
            y_hat_90, _, _ = self.shared_step(x_90, y)
            y_hat_180, _, _ = self.shared_step(x_180, y)
            y_hat_270, _, _ = self.shared_step(x_270, y)

            # Stack rotated predictions
            y_hat_rd = torch.stack([y_hat_90, y_hat_180, y_hat_270], dim=1)
            
            # Compute consistency metrics
            norm = torch.norm(y_hat_rd - y_hat[:, None], dim=-1)
            sim = F.cosine_similarity(y_hat_rd, y_hat[:, None], dim=-1) 
            norm_max = norm.max()
            cos_min = sim.min()
            norm_mean = norm.mean()
            cos_sim_mean = sim.mean()

            # Log RCI metrics
            self.log(f'test_rci_norm_n4', norm_mean, sync_dist=True, batch_size=self.trainer.datamodule.test_batch_size, add_dataloader_idx=False)
            self.log(f'test_rci_sim_n4', cos_sim_mean, sync_dist=True, batch_size=self.trainer.datamodule.test_batch_size, add_dataloader_idx=False)
            self.log(f'test_rci_norm_max_n4', norm_max, sync_dist=True, reduce_fx="max", batch_size=self.trainer.datamodule.test_batch_size, add_dataloader_idx=False)
            self.log(f'test_rci_sim_min_n4', cos_min, sync_dist=True, reduce_fx="min", batch_size=self.trainer.datamodule.test_batch_size, add_dataloader_idx=False)
            
        elif dataloader_name == ROTATED_TEST_SET_KEY:
            # x is [B, n_angles, C, H, W] - batch of images, each with all rotations
            # y is [B, n_angles] - labels repeated for each rotation
            n_angles = self.trainer.datamodule.n_angles 
            batch_size = x.shape[0]
            
            # Reshape to process all rotations at once: [B*n_angles, C, H, W]
            x_flat = rearrange(x, 'b n c h w -> (b n) c h w')
            y_flat = rearrange(y, 'b n -> (b n)')
            
            # Get predictions for all rotations
            logits, _, _ = self.shared_step(x_flat, y_flat)
            y_hat = torch.argmax(logits, dim=1)
            correct = rearrange((y_hat == y_flat), '(b n) -> b n',
                                 b=batch_size,
                                 n=n_angles)

            # Reshape back: [B, n_angles, num_classes]
            logits = rearrange(logits, '(b n) c -> b n c',
                               b=batch_size,
                               n=n_angles)
            y_hat = rearrange(y_hat, '(b n) -> b n', 
                              b=batch_size,
                              n=n_angles)
            
            # First rotation (0 degrees) is the upright image
            upright_logits = logits[:, 0:1]
            rotated_logits = logits[:, 1:]
            
            # Compute consistency metrics between upright and rotated predictions
            norm = torch.norm(rotated_logits - upright_logits, dim=-1)
            sim = F.cosine_similarity(rotated_logits, upright_logits, dim=-1)
            
            # Upright predictions correctness: [B]
            upright_correct = correct[:, 0]
            rotated_correct = correct[:, 1:]

            # Rotation mismatch rate estimate
            rmie = (~rotated_correct).float().mean(dim=1)
            rmie_valid = rmie[upright_correct]
            if len(rmie_valid) > 0: 
                self.log(f'rmie', rmie_valid.mean(), sync_dist=True, add_dataloader_idx=False)
                # Invalid samples norm
                norm_valid = norm[(~rotated_correct) & upright_correct[:, None]]
                sim_valid =  sim[(~rotated_correct) & upright_correct[:, None]]
                if len(norm_valid) > 0:
                    self.log(f'mi_norm', norm_valid.mean(), sync_dist=True, add_dataloader_idx=False)
                    self.log(f'mi_sim', sim_valid.mean(), sync_dist=True, add_dataloader_idx=False)
            self.log(f'test_rci_norm_n{n_angles}', norm.mean(), sync_dist=True, add_dataloader_idx=False)
            self.log(f'test_rci_sim_n{n_angles}', sim.mean(), sync_dist=True, add_dataloader_idx=False)
            self.log(f'test_rci_norm_max_n{n_angles}', norm.max(), sync_dist=True, reduce_fx="max", add_dataloader_idx=False)
            self.log(f'test_rci_sim_min_n{n_angles}', sim.min(), sync_dist=True, reduce_fx="min", add_dataloader_idx=False)

        else:
            raise ValueError(f"Unexpected test dataset key: {dataloader_name}")

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

# Timm inspired Resnet
def make_blocks(layer, 
                layer_kwargs, 
                input_size,
                # Block settings
                block_types,
                kernels_size,
                channels, 
                layers, 
                norm,
                act,
                channels_masking):
    """
    Create ResNet-style stages with specified block configurations.
    
    Args:
        layer: Convolutional layer class/function to use (e.g., ComplexInvariantConv2D, LearnableFlusser)
        layer_kwargs: Dictionary of kwargs to pass to layer constructor
        input_size: Size of input feature maps at this stage
        block_types: List of block types for each stage (e.g., TimmBasicBlock)
        kernels_size: List of kernel sizes for each stage
        channels: List of output channels for each stage
        layers: List of number of blocks in each stage
        norm: Normalization layer class
        act: Activation layer class
        channels_masking: Whether to use Tukey masking on channels
        
    Returns:
        Tuple of (stage_modules, feature_info):
            - stage_modules: List of (name, nn.Sequential) tuples for each stage
            - feature_info: List of dicts with stage metadata
    """
    stages = []
    feature_info = []
    inplanes = channels[0]  # First stage output channels
    current_size = input_size
    
    for stage_idx in range(len(block_types)):
        stage_name = f'layer{stage_idx + 1}'
        block_fn = block_types[stage_idx]
        num_blocks = layers[stage_idx]
        out_channels = channels[stage_idx]
        kernel_size = kernels_size[stage_idx] if isinstance(kernels_size, list) else kernels_size
        
        blocks = []
        for block_idx in range(num_blocks):
            # Determine if this is the first block in the stage
            is_first_block = (block_idx == 0)
            
            # Apply downsampling (via aa_layer) only on first block of stages 2, 3, 4
            # Stage 1 (stage_idx=0) doesn't downsample
            use_downsample = (stage_idx > 0) and is_first_block
            aa_layer = nn.AvgPool2d if use_downsample else None
            
            # Build block kwargs
            block_kwargs = {
                'input_size': current_size,
                'in_channels': inplanes,
                'out_channels': out_channels,
                'kernel_size': kernel_size,
                'tukey_masking': channels_masking,
                'conv_layer': layer,
                'conv_kwargs': layer_kwargs.copy(),
                'act_layer': act,
                'norm_layer': norm,
                'aa_layer': aa_layer,
                'drop_path': None,  # Can be added later for stochastic depth
                'drop_block': None,
            }
            
            blocks.append(block_fn(**block_kwargs))
            
            # Update state for next block
            inplanes = out_channels
            if use_downsample:
                current_size = current_size // 2
        
        # Create stage as sequential module
        stages.append((stage_name, nn.Sequential(*blocks)))
        
        # Track feature info for this stage
        reduction = input_size // current_size
        feature_info.append({
            'num_chs': out_channels,
            'reduction': reduction,
            'module': stage_name
        })
    
    return stages, feature_info

class Resnet(torch.nn.Module): 
    def __init__(self, 
                 stem: str = "single", # single, deep, None 
                 # Layer settings
                 layer : str = "LearnableFlusser",
                 default_layer_kwargs: dict = dict(), 
                 # Shape informations
                 in_channels:int = 3,
                 input_size:int = 64,
                 stem_kernel_size: int = 15,
                 kernels_size: int = [11, 11, 11, 11],
                 block_types: list = [TimmBasicBlock, TimmBasicBlock, TimmBasicBlock, TimmBasicBlock],
                 layers: list = [1, 1, 1, 1], 
                 channels: list = [10, 10, 10, 10],
                 # Block settings
                 norm:str = "layer", 
                 activation: str = "ELU",
                 channels_masking:bool = True,
                 # Classifier settings
                 classification:bool = True, 
                 hidden_classifier_size:int = 64,
                 num_classes:int = 10,
                 drop_rate: float = 0.0): 
        super(Resnet, self).__init__()
        assert len(block_types) == len(layers) == len(channels), "block_types, layers and channels must have the same length"
        assert len(block_types) == 4, "Currently only 4 stages are supported"


        norm = nn.BatchNorm2d if norm == "batch" else nn.LayerNorm
        act = getattr(nn, activation)
        pool_layer = nn.AvgPool2d
        self.drop_rate = drop_rate
        self.num_classes = num_classes
    
        if stem == "single":
            logger.warning("Stem for ResNet expects masked inputs")
            inplanes = channels[0]
            stem_layer_kwargs = default_layer_kwargs.copy()
            stem_layer_kwargs['in_channels'] = in_channels
            stem_layer_kwargs['out_channels'] = inplanes
            stem_layer_kwargs['input_size'] = input_size
            stem_layer_kwargs['kernel_size'] = stem_kernel_size
            self.stem = torch.nn.Sequential(
                conv_factory.get_conv_layer(layer, stem_layer_kwargs),
                pool_layer(kernel_size=2, stride=2),
                norm(inplanes) if norm is nn.BatchNorm2d else norm((inplanes, input_size // 2, input_size // 2), elementwise_affine=False),
                act(inplace=True)
            )
            self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]
        else: 
            NotImplementedError("Only 'single' stem is implemented")

        stage_modules, stage_feature_info = make_blocks(
            layer=layer,
            layer_kwargs=default_layer_kwargs,
            input_size=input_size // 2,  # After stem pooling
            block_types=block_types,
            kernels_size=kernels_size,
            channels=channels,
            layers=layers,
            norm=norm,
            act=act,
            channels_masking=channels_masking,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_feature = self.head_hidden_size = channels[-1]
        self.classification = classification
        if classification:
            self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.flat = torch.nn.Flatten()
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.head_hidden_size, out_features=hidden_classifier_size),
                torch.nn.BatchNorm1d(num_features=hidden_classifier_size),
                act(inplace=True),
                torch.nn.Linear(in_features=hidden_classifier_size, out_features=num_classes)
            )
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.classification:
            x = self.global_pool(x)
            x = self.flat(x)
            x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = self.classifier(x)
        return x

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

# Optimal Convolution
class PrototypeOptimalInvCNN(torch.nn.Module): 
    def __init__(self, 
                 # Layer settings
                 layer : nn.Module = "ComplexInvariantConv2D",
                 layer_kwargs: dict = dict(), 
                 # Shape informations
                 in_channels:int = 3,
                 input_size:int = 64,
                 num_classes:int = 10,
                 kernel_size: int = 15,
                 n_blocks=3, 
                 m_layers=3,
                 init_channels=4,
                 # Block settings
                 norm:str = "layer",
                 activation: str = "ELU",
                 channels_masking:bool = True,
                 classification:bool = True): 
        super(PrototypeOptimalInvCNN, self).__init__()
        if type(kernel_size) is int:
            kernel_size = [kernel_size] * n_blocks
        elif type(kernel_size) is list:
            assert len(kernel_size) == n_blocks, "If a list of kernel sizes is provided, it must have length n_blocks"
        else:
            raise TypeError("kernel_size must be an int or a list of ints")
        if type(init_channels) is int:
            channels = [init_channels * (2 ** i) for i in range(n_blocks + 1)]
        elif type(init_channels) is list:
            channels = init_channels
            assert len(channels) == n_blocks, "If a list of channels is provided, it must have length n_blocks + 1"
        else: 
            raise TypeError("init_channels must be an int or a list of ints")

        self.masking_channels = "tukey" if channels_masking else "none"
        self.in_channels = in_channels
        out_channels = in_channels
        self.blocks = []
        for block_idx in range(n_blocks):
            block = []
            for _ in range(m_layers):
                out_channels = channels[block_idx]
                block.append(ResnetBlock(
                    conv_layer=layer,
                    conv_kwargs=layer_kwargs,
                    kernel_size=kernel_size[block_idx],
                    in_channels=in_channels,
                    out_channels=out_channels,
                    input_size=input_size,
                    subsampling=False,
                    activation=activation,
                    norm=norm,
                    channels_masking=self.masking_channels
                ))
                in_channels = out_channels

            out_channels = channels[block_idx + 1] if block_idx + 1 < len(channels) else out_channels
            # Subsampling block at the end
            block.append(ResnetBlock(
                conv_layer=layer,
                conv_kwargs=layer_kwargs,
                kernel_size=kernel_size[block_idx],
                in_channels=in_channels,
                out_channels=out_channels,
                input_size=input_size,
                subsampling=True if block_idx < n_blocks - 1 else False,
                activation=activation,
                norm=norm,
                channels_masking=self.masking_channels
            ))
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
                 # Layer settings
                 layer : nn.Module = "ComplexInvariantConv2D",
                 layer_kwargs: dict = dict(max_order=3), 
                 num_classes: int = 10, 
                 input_size: int = 28,
                 scale_channels: int = 1, 
                 kernel_size: List[int] = [7, 5, 5, 5, 5, 5]):
        super(PrototypeTiny, self).__init__()
        assert len(kernel_size) == 6, "kernel_size must be a list of 6 integers"

        # 28 px
        self.layer_1 = ResnetBlock(conv_layer=layer,
                                   conv_kwargs=layer_kwargs,
                                   in_channels=in_channels,
                                   input_size=input_size,
                                   out_channels=16 * scale_channels, 
                                   kernel_size=kernel_size[0],
                                   conv_padding="same",
                                   channels_masking="none",
                                   subsampling=False)
        # 24 px 
        self.layer_2 = ResnetBlock(conv_layer=layer,
                                   conv_kwargs=layer_kwargs,
                                   in_channels=16 * scale_channels,
                                   input_size=input_size,
                                   out_channels=32 * scale_channels,
                                   kernel_size=kernel_size[1],
                                   conv_padding="same",
                                   channels_masking="none",
                                   subsampling=True)

        # 12 px
        self.layer_3 = ResnetBlock(conv_layer=layer,
                                   conv_kwargs=layer_kwargs,
                                   in_channels=32 * scale_channels,
                                   input_size=input_size // 2,
                                   out_channels=32 * scale_channels,
                                   kernel_size=kernel_size[2],
                                   conv_padding="same",
                                   channels_masking="none",
                                   subsampling=False)

        self.layer_4 = ResnetBlock(conv_layer=layer,
                                   conv_kwargs=layer_kwargs,
                                   in_channels=32 * scale_channels,
                                   input_size=input_size // 2,
                                   out_channels=32 * scale_channels,
                                   kernel_size=kernel_size[3],
                                   conv_padding="same",
                                   channels_masking="none",
                                   subsampling=True)

        # 6 px
        self.layer_5 = ResnetBlock(conv_layer=layer,
                                   conv_kwargs=layer_kwargs,
                                   in_channels=32 * scale_channels,
                                   input_size=input_size // 4,
                                   out_channels=48 * scale_channels,
                                   kernel_size=kernel_size[4],
                                   conv_padding="same",
                                   channels_masking="none",
                                        subsampling=False)

        self.layer_6 = ResnetBlock(conv_layer=layer,
                                   conv_kwargs=layer_kwargs,
                                   in_channels=48 * scale_channels,
                                   input_size=input_size // 4,
                                   out_channels=64 * scale_channels,
                                   kernel_size=kernel_size[5],
                                   conv_padding="same",
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
                 input_size: int=28, # For compatibility, not used
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