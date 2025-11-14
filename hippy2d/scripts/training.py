from modulefinder import test
import os
import json
import click
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

from hippy2d.trainer import get_trainer, EpochTimeLogger
from hippy2d.models import InvNet
from hippy2d.utils import ClickDictionaryType

# Work arround for automatic SLURM detection
from lightning.pytorch.plugins.environments import SLURMEnvironment
SLURMEnvironment.detect = lambda: False
WANDB_PATH = os.path.join(Path.home(), ".wandb_key.json")

@click.command()
@click.option('-n', '--run_name', default='default', help='Name of the experiment.')
@click.option('--early_stopping', default=15, type=int, help='Early stopping patience.')
@click.option('--epochs', default=30, type=int, help='Number of training epochs.')
@click.option('--debug', is_flag=True, default=False, help='Run in debug mode with small datasets.')
@click.option('--dataset_name', default='ColorectalHistology', type=str, help='Name of the dataset.')
@click.option('--d_hparams', default=dict(batch_size=32), type=ClickDictionaryType(), help='Dataset hyperparameters.')
@click.option('--model_name', default="PrototypeOptimalInvCNN", type=str, help='Name of the model to use.')
@click.option('--m_param', default=dict(in_channels=3, num_classes=8, input_size=150, norm='batch', init_channels=7, n_blocks=4, m_layers=3, kernel_size=11), type=ClickDictionaryType(), help='Model hyperparameters.')
@click.option('--optimizer_name', default='AdamW', type=str, help='Optimizer name.')
@click.option('--optimizer_hparams', default=dict(lr=1e-2), type=ClickDictionaryType(), help='Optimizer hyperparameters.')
@click.option('--lr_name', default='MultiStepLR', type=str, help='Learning rate scheduler name.')
@click.option('--lr_hparams', default=dict(milestones=[10, 50, 90], gamma=0.1), type=ClickDictionaryType(), help='Learning rate scheduler hyperparameters.')
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
            limit_train_batches=0.125,
            limit_val_batches=0.125,
            limit_test_batches=0.125,
            detect_anomaly=True,
            deterministic="warn",
        )
    else:
        trainer_params = dict(
            detect_anomaly=False,
            deterministic=False,
        )

    # Lightning callbacks
    checkpoint_callback = callbacks.ModelCheckpoint(monitor="val_acc",
                                                    mode="max",
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
                       log_graph=False)
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
        else:
            logger.warning("Best model not found.")
        wandb_logger.finalize(_wandb_out_status)

if __name__ == "__main__":
    training_loop()
