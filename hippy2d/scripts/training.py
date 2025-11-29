from modulefinder import test
import os
import json
import click
from einops import rearrange
import torch
import wandb
import numpy as np
from git import Repo
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from torch.nn import functional as F
from lovely_tensors import monkey_patch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything, callbacks, Trainer
from lightning.pytorch.utilities.combined_loader import CombinedLoader
import matplotlib.pyplot as plt
import seaborn as sns

from hippy2d.learnable import LearnableFlusser
from hippy2d.trainer import get_trainer, EpochTimeLogger
from hippy2d.models import InvNet
from hippy2d.utils import ClickDictionaryType
from hippy2d.escnn_prototype import LearnableCesa, MomentLayer

# Work arround for automatic SLURM detection
from lightning.pytorch.plugins.environments import SLURMEnvironment
SLURMEnvironment.detect = lambda: False
WANDB_PATH = os.path.join(Path.home(), ".wandb_key.json")

def collect_activations(model, dataloader,device, max_batches=10, layer_type=LearnableCesa):
    """
    Collect activations from all LearnableCesa layers in the model.
    
    Args:
        model: The neural network model
        dataloader: DataLoader to get input data from
        max_batches: Maximum number of batches to process
        
    Returns:
        dict: Dictionary mapping layer names to activation arrays
    """
    model.eval()
    model = model.to(device)
    activations = {}
    hooks = []
    layer_names = []
    orders = {} 
    in_channels = {}

    
    # Register forward hooks for all LearnableCesa layers
    def get_activation_hook(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            if name not in orders and hasattr(module, 'orders'):
                orders[name] = module.orders
            if name not in in_channels and hasattr(module, 'in_channels'):
                in_channels[name] = module.in_channels
            # Store flattened activations
            activations[name].append(output.detach().cpu().numpy())
        return hook
    
    # Find and register hooks for all LearnableCesa layers
    for name, module in model.named_modules():
        if isinstance(module, layer_type):
            layer_names.append(name)
            hook = module.register_forward_hook(get_activation_hook(name))
            hooks.append(hook)
    
    logger.info(f"Found {len(layer_names)} {layer_type.__name__} layers: {layer_names}")
    
    # Pass data through the model
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            if isinstance(batch, dict):
                # Handle dictionary batch format
                for k, v in batch.items():
                    if len(v) == 1:
                        imgs, _ = v[0]
                    else:
                        imgs, _ = v
                    break  # Just use first dataset
            else:
                # Handle tuple batch format
                imgs, _ = batch
            
            imgs = imgs.to(device)
            _ = model(imgs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate all batches for each layer
    for name in activations:
        activations[name] = np.concatenate(activations[name])
    
    return activations, layer_names, orders, in_channels

def plot_activation_distributions(activations, layer_names, orders, in_channels, prefix="test"):
    """
    Create and log activation distribution plots to Wandb.
    
    Args:
        activations: Dictionary mapping layer names to activation arrays
        layer_names: List of layer names in order
        prefix: Prefix for wandb logging
    """
    # Create figure with subplots for each layer
    n_layers = len(layer_names)
    columns = 1 if len(orders) == 0 or len(in_channels) == 0 else len(np.unique(orders[layer_names[0]]))

    fig, axes = plt.subplots(n_layers, columns, figsize=(10*columns, 3 * n_layers))
    
    for idx, layer_name in enumerate(layer_names):
        acts = activations[layer_name]
        if layer_name in in_channels:
            in_ch = in_channels[layer_name] 
            ord = np.array(orders[layer_name])
            # Reshape using 
            trivial_idx = np.sum(np.array(orders['stem.0.moment_layer']) == 0)
            acts_trivial =  acts[:, :, :trivial_idx, :, :]
            acts_trivial = acts_trivial.reshape(-1)
            axes[idx][0].hist(acts_trivial, bins=50, density=True, alpha=0.7, color='C0', edgecolor='black')
            axes[idx][0].set_title(f"{layer_name}\n(mean={np.mean(acts_trivial):.3f}, var={np.var(acts_trivial):.3f})")
            axes[idx][0].set_xlabel("Trivial Activation values")
            axes[idx][0].set_ylabel("Density")
            axes[idx][0].grid(True, alpha=0.3)
            # Recomplex the rest 
            acts_non_trivial = rearrange(acts[:, :, trivial_idx:, :, :], 
                                         'b ch (o c) h w -> b ch o c h w', c=2)
            ord = ord[trivial_idx:]
            for col_idx, order in enumerate(np.unique(ord)):
                order_mask = ord == order
                acts_order = acts_non_trivial[:, :, order_mask, :, :, :]
                acts_order_magnitude = np.sqrt(acts_order[..., 0, :, :]**2 + acts_order[..., 1, :, :]**2).reshape(-1)
                axes[idx][col_idx+1].hist2d(acts_order[..., 0, :, :].reshape(-1), acts_order[..., 1, :, :].reshape(-1))
                axes[idx][col_idx+1].set_title(f"{layer_name} - Order {order}\n(mean={np.mean(acts_order_magnitude):.3f}, var={np.var(acts_order_magnitude):.3f})")
                axes[idx][col_idx+1].set_xlabel("Real part")
                axes[idx][col_idx+1].set_ylabel("Imaginary part")
        else: 
            # Plot histogram/density
            axes[idx].hist(acts.reshape(-1), bins=50, density=True, alpha=0.7, color='C0', edgecolor='black')
            axes[idx].set_title(f"{layer_name}\n(mean={np.mean(acts):.3f}, var={np.var(acts):.3f})")
            axes[idx].set_xlabel("Activation values")
            axes[idx].set_ylabel("Density")
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({f"{prefix}_activations": wandb.Image(fig)})
    plt.close(fig)

@click.command()
@click.option('-n', '--run_name', default='default', help='Name of the experiment.')
@click.option('--early_stopping', default=15, type=int, help='Early stopping patience.')
@click.option('--epochs', default=30, type=int, help='Number of training epochs.')
@click.option('--debug', is_flag=True, default=False, help='Run in debug mode with small datasets.')
@click.option('--dataset_name', default='ColorectalHistology', type=str, help='Name of the dataset.')
@click.option('--d_hparams', default=dict(batch_size=32), type=ClickDictionaryType(), help='Dataset hyperparameters.')
@click.option('--model_name', default="Resnet", type=str, help='Name of the model to use.')
@click.option('--m_param', default=dict(), type=ClickDictionaryType(), help='Model hyperparameters.')
@click.option('--optimizer_name', default='AdamW', type=str, help='Optimizer name.')
@click.option('--optimizer_hparams', default=dict(lr=1e-2), type=ClickDictionaryType(), help='Optimizer hyperparameters.')
@click.option('--lr_name', default='MultiStepLR', type=str, help='Learning rate scheduler name.')
@click.option('--lr_hparams', default=dict(milestones=[10, 50, 90], gamma=0.1), type=ClickDictionaryType(), help='Learning rate scheduler hyperparameters.')
@click.option('--label_smoothing', default=0.0, type=float, help='Label smoothing value.')
@click.option('--seed', default=42, type=int)
@click.option('--float_precision', default='float32', type=str, help='Float precision for training.')
def training_loop(run_name: str,
                  seed: int,
                  early_stopping: int,
                  epochs: int,
                  debug: bool,
                  dataset_name: str,
                  d_hparams: dict,
                  model_name: str,
                  m_param: dict,
                  optimizer_name: str,
                  optimizer_hparams: dict,
                  lr_name: str,
                  label_smoothing: float,
                  lr_hparams: dict,
                  float_precision: str):

    # Get git commit sha
    repo = Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    # Logger
    _log_dict = {
        'run_name': run_name,
        'early_stopping': early_stopping,
        'epochs': epochs,
        'debug': debug,
        'dataset_name': dataset_name,
        'd_hparams': d_hparams,
        'model_name': model_name,
        'm_param': m_param,
        'optimizer_name': optimizer_name,
        'optimizer_hparams': optimizer_hparams,
        'lr_name': lr_name,
        'lr_hparams': lr_hparams,
        'seed': seed,
        'sha' : sha
    }

    assert os.path.exists(WANDB_PATH), f"Wandb json not found at. {WANDB_PATH}"

    # Opening JSON file
    with open(WANDB_PATH, 'r') as f:
        config = json.load(f)
        logger.debug(f"Loading Wandb key from {WANDB_PATH}")
        os.environ["WANDB_API_KEY"] = config["WANDB_API_KEY"]
        os.environ["WANDB_HOST"] = config["WANDB_HOST"]

    wandb_logger = WandbLogger(
        name=run_name,
        project="hippy2d",
        log_model=True,
        entity="karella",
        config=_log_dict)
    
    # List available gpu
    if torch.cuda.is_available():
        accelerator = "gpu"
        for gpu_idx in range(torch.cuda.device_count()):
            logger.debug(
                f"GPU {gpu_idx} - {torch.cuda.get_device_name(gpu_idx)} ({torch.cuda.get_device_properties(gpu_idx).total_memory / 1e+6:.0f} MB)")
    elif torch.backends.mps.is_available():
        logger.debug(f"Using Apple Silicon GPU")
        accelerator = "mps"
    else:
        logger.warning("No GPU detected.")
        accelerator = "cpu"
    # Convert dictionary parameters
    if debug:
        #logger.setLevel("DEBUG")
        monkey_patch()
        logger.warning("Running in debug mode (small datasets, offline, etc).")
        # NOTE: Debug does not work running parallel workers
        d_hparams['num_workers'] = 1
        trainer_params = dict(
            limit_train_batches=0.25,
            limit_val_batches=1,
            limit_test_batches=1,
            detect_anomaly=True,
            deterministic="warn",
            devices=1,

        )
    else:
        trainer_params = dict(
            detect_anomaly=False,
            deterministic=False,
        )

    # Lightning callbacks
    checkpoint_callback = callbacks.ModelCheckpoint(monitor="val_loss",
                                                    mode="min",
                                                    save_weights_only=True)
    trainer_callbacks = [checkpoint_callback,
                         callbacks.ModelSummary(max_depth=-1),
                         callbacks.LearningRateMonitor(logging_interval='epoch'),
                         EpochTimeLogger()]
    if early_stopping > 0:
        trainer_callbacks.append(callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping))

    trainer, model, datamodule = get_trainer(seed=seed,
                          epochs=epochs,
                          dataset_name=dataset_name,
                          d_hparams=d_hparams,
                          model_name=model_name,
                          m_param=m_param,
                          label_smoothing=label_smoothing,
                          optimizer_name=optimizer_name,
                          optimizer_hparams=optimizer_hparams,
                          lr_name=lr_name,
                          lr_hparams=lr_hparams,
                          accelerator=accelerator,
                          trainer_params=trainer_params,
                          trainer_loggers=[wandb_logger],
                          trainer_callbacks=trainer_callbacks,
                          float_precision=float_precision)
    # Lightning trainer
    wandb_logger.watch(model,
                       log="all",
                       log_graph=True, 
                       log_freq=1000)
    # Force datamodule to prepare data
    datamodule.prepare_data()
    _wandb_out_status = 'aborted'
    wandb.init()
    wandb.define_metric('val_acc', summary='max')
    try:
        # TODO: add number of parameters
        # Training loop
        trainer.fit(model=model,
                    datamodule=datamodule)
    finally:
        if checkpoint_callback.best_model_path != '':
            # model
            best_model = InvNet(input_shape=datamodule.output_shape,
                                model=model.model,
                                optimizer_name=optimizer_name,
                                optimizer_hparams=optimizer_hparams,
                                lr_name=lr_name,
                                lr_hparams=lr_hparams)
            best_checkpoint = torch.load(checkpoint_callback.best_model_path, weights_only=False)
            _result = best_model.load_state_dict(best_checkpoint['state_dict'], strict=False )
            logger.debug(f"Best model loading from {checkpoint_callback.best_model_path} : {_result}")

            trainer.test(model=best_model,
                         datamodule=datamodule)
            _wandb_out_status = "success"
            
            # Check the model activations on test set for Moment layers
            logger.info("Collecting activations from  Moment layers...")
            try:
                # Get test dataloader
                test_loader = datamodule.test_dataloader()
                
                if type(test_loader) is CombinedLoader:
                    test_loader = test_loader.iterables['test']

                tracked_layers = [MomentLayer, LearnableCesa, LearnableFlusser]
                for layer_type in tracked_layers:
                    logger.info(f"Collecting activations from {layer_type.__name__} layers...")
                    activations, layer_names, orders, in_channels = collect_activations(
                        model=best_model.model,
                        dataloader=test_loader,
                        device=trainer.strategy.root_device,
                        max_batches=10,
                        layer_type=layer_type
                    )
                    
                    if len(layer_names) > 0:
                        plot_activation_distributions(activations, layer_names, orders, in_channels, prefix=f"{layer_type.__name__}")
                        logger.info(f"Successfully logged activations for {len(layer_names)} {layer_type.__name__} layers")
                    else:
                        logger.warning(f"No {layer_type.__name__} layers found in the model")
            except Exception as e:
                logger.error(f"Error collecting activations: {e}")
                import traceback
                traceback.print_exc()
            
        else:
            logger.warning("Best model not found.")
        wandb_logger.finalize(_wandb_out_status)

if __name__ == "__main__":
    training_loop()
