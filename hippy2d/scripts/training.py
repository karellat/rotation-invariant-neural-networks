import ssl 
import click
import torch
import lightning
from loguru import logger
from git import Repo
from lovely_tensors import monkey_patch
from lightning.pytorch import seed_everything, callbacks, Trainer

from src.factory import get_datamodule, get_model
from src.models import InvNet
from src.utils import ClickDictionaryType

# Work arround for automatic SLURM detection
from lightning.pytorch.plugins.environments import SLURMEnvironment
SLURMEnvironment.detect = lambda: False

@click.command()
@click.option('-n', '--run_name', default='default', help='Name of the experiment.')
@click.option('--early_stopping', default=15, type=int, help='Early stopping patience.')
@click.option('--epochs', default=200, type=int, help='Number of training epochs.')
@click.option('--debug', is_flag=True, default=False, help='Run in debug mode with small datasets.')
@click.option('--dataset_name', default='MnistRotTest', type=str, help='Name of the dataset.')
@click.option('--d_hparams', default=dict(batch_size=32, data_dir='data', pad=0, to_complex=False), type=ClickDictionaryType(), help='Dataset hyperparameters.')
@click.option('--model_name', default='Resnet18', type=str, help='Name of the model to use.')
@click.option('--m_param', default=dict(), type=ClickDictionaryType(), help='Model hyperparameters.')
@click.option('--optimizer_name', default='AdamW', type=str, help='Optimizer name.')
@click.option('--optimizer_hparams', default=dict(lr=1e-3), type=ClickDictionaryType(), help='Optimizer hyperparameters.')
@click.option('--lr_name', default='MultiStepLR', type=str, help='Learning rate scheduler name.')
@click.option('--lr_hparams', default=dict(milestones=[10, 50, 90], gamma=0.1), type=ClickDictionaryType(), help='Learning rate scheduler hyperparameters.')
@click.option('--seed', default=42, type=int)
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
                  lr_hparams: dict):
    # Set default float precision
    torch.set_default_dtype(torch.float32)
    seed_everything(seed, workers=True)
    # Fixing SSL certificate verification
    ssl._create_default_https_context = ssl._create_stdlib_context
    # Get git commit sha
    repo = Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    # Logger
    csv_logger = lightning.pytorch.loggers.CSVLogger('./logs/', name=run_name, version=sha)
    logger.add(csv_logger.log_dir + '/training.log', level='DEBUG', format="{time} {level} {message}")
    
    csv_logger.log_hyperparams({
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
        'seed': seed
    })

    # List available gpu
    if torch.cuda.is_available():
        for gpu_idx in range(torch.cuda.device_count()):
            logger.debug(
                f"GPU {gpu_idx} - {torch.cuda.get_device_name(gpu_idx)} ({torch.cuda.get_device_properties(gpu_idx).total_memory / 1e+6:.0f} MB)")
    elif torch.backends.mps.is_available():
        logger.debug(f"Using Apple Silicon GPU")
    else:
        logger.warning("No GPU detected.")
    # Convert dictionary parameters
    if debug:
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


    datamodule = get_datamodule(dataset_name, d_hparams)
    # Default model parameters 
    # TODO: This should be moved to the model factory
    if "in_channels" not in m_param:
        m_param["in_channels"] = datamodule.output_shape[1]
    if "num_classes" not in m_param:
        m_param["num_classes"] = datamodule.num_classes

    model = get_model(model_name, m_param)

    # Lighting model
    model = InvNet(input_shape=datamodule.output_shape,
                   model=model,
                   optimizer_name=optimizer_name,
                   optimizer_hparams=optimizer_hparams,
                   lr_name=lr_name,
                   lr_hparams=lr_hparams)

    # Lightning callbacks
    checkpoint_callback = callbacks.ModelCheckpoint(monitor="val_acc",
                                                    mode="max",
                                                    save_weights_only=True)
    trainer_callbacks = [checkpoint_callback,
                         callbacks.ModelSummary(max_depth=-1),
                         callbacks.LearningRateMonitor(logging_interval='epoch')]
    if early_stopping > 0:
        trainer_callbacks.append(callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping))
    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                      devices=-1 if torch.cuda.is_available() else "auto",
                      max_epochs=epochs,
                      **trainer_params,
                      enable_model_summary=False,
                      callbacks=trainer_callbacks,
                      logger=csv_logger,
                      )
    # Run training loop
    try:
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
        else:
            logger.warning("Best model not found.")


if __name__ == "__main__":
    training_loop()
