import ssl 
import torch 
from loguru import logger
from lightning.pytorch import seed_everything, Trainer
from lightning import Callback
from typing import List
from hippy2d.factory import get_datamodule, get_model
from hippy2d.models import InvNet

import time

class EpochTimeLogger(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        # Log using Lightning’s logger
        pl_module.log("train_epoch_time", epoch_time, prog_bar=True, logger=True)

def get_trainer(seed: int,
               epochs: int,
               dataset_name: str,
               d_hparams: dict,
               model_name: str,
               m_param: dict,
               optimizer_name: str,
               optimizer_hparams: dict,
               lr_name: str,
               lr_hparams: dict, 
               label_smoothing: float = 0.0,
               accelerator: str = 'auto',
               trainer_params: dict = {},
               trainer_loggers: List = [],
               trainer_callbacks: List = [],
               float_precision: str = 'float32'):
    
    torch.set_default_dtype(getattr(torch, float_precision))
    logger.debug(f"Default float precision to {torch.get_default_dtype()}")
    seed_everything(seed, workers=True)
    logger.debug(f"Setting random seed to {seed}")
    # Fixing SSL certificate verification
    ssl._create_default_https_context = ssl._create_stdlib_context

    datamodule = get_datamodule(dataset_name, d_hparams)
    if ("in_channels" in m_param) and (m_param["in_channels"] != "auto"):
        m_param["in_channels"] = datamodule.output_shape[1]
    if "num_classes" not in m_param:
        m_param["num_classes"] = datamodule.num_classes

    logger.debug(f"Using dataset: {dataset_name} with parameters: {d_hparams}")
    model = get_model(model_name, m_param)
    logger.debug(f"Using model: {model_name} with parameters: {m_param}")

    # Lighting model
    model = InvNet(input_shape=datamodule.output_shape[-1],
                   model=model,
                   optimizer_name=optimizer_name,
                   optimizer_hparams=optimizer_hparams,
                   label_smoothing=label_smoothing, 
                   lr_name=lr_name,
                   lr_hparams=lr_hparams)


    trainer = Trainer(accelerator=accelerator,
                      max_epochs=epochs,
                      **trainer_params,
                      callbacks=trainer_callbacks,
                      logger=trainer_loggers,
                      )
    return trainer, model, datamodule