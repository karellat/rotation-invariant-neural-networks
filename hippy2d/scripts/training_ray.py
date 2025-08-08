import ray 
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

from hippy2d.trainer import get_trainer
from ray.train.torch import TorchTrainer

def _config_test(config):
    assert "seed" in config, "Training function requires 'seed' in config."
    assert "epochs" in config, "Training function requires 'epochs' in config."
    assert "dataset_name" in config, "Training function requires 'dataset_name' in config."
    assert "dataset_hparams" in config, "Training function requires 'dataset_hparams' in config."
    assert "model_name" in config, "Training function requires 'model_name' in config."
    assert "m_param" in config, "Training function requires 'm_param' in config."
    assert "optimizer_name" in config, "Training function requires 'optimizer_name' in config."
    assert "optimizer_hparams" in config, "Training function requires 'optimizer_hparams' in config."
    assert "lr_name" in config, "Training function requires 'lr_name' in config."
    assert "lr_hparams" in config, "Training function requires 'lr_hparams' in config."

def train_func(config): 
    # Test the keys 
    _config_test(config)
    trainer, model, dm = get_trainer(seed=config["seed"],
                          epochs=config["epochs"],
                          dataset_name=config["dataset_name"],
                          d_hparams=config["dataset_hparams"],
                          model_name=config["model_name"],
                          m_param=config["m_param"],
                          optimizer_name=config["optimizer_name"],
                          optimizer_hparams=config["optimizer_hparams"],
                          lr_name=config["lr_name"],
                          lr_hparams=config["lr_hparams"],
                          accelerator="auto",
                          trainer_callbacks=[RayTrainReportCallback()],
                          trainer_params=dict(strategy=RayDDPStrategy(),
                                              plugins=[RayLightningEnvironment()],
                                              enable_progress_bar=False,
                          ))
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)   

search_space = {
    "seed": 42,
    "epochs": 100,
    "dataset_name": "MnistRotTest", 
    "model_name" : "PrototypeOptimalInvCNN",
    "m_param": {
        "in_channels" : 1,
        "input_size" : 64,
        "num_classes" : 10,
        "zero_order_scaling" : tune.choice([True, False]),
    },
    "optimizer_hparams": {
            "lr": tune.loguniform(1e-4, 1e-1),
    }, 
    "optimizer_name": "AdamW",
    "dataset_hparams": {
        "batch_size": tune.choice([32, 64, 128]),
        "data_dir" : "/vast/home/karella/rotation-invariant-neural-networks/hippy2d/data",
        "pad" : 0,
        "to_complex" : False
    },
    "lr_name": "MultiStepLR",
    "lr_hparams": {
        "milestones": [30, 80],
        "gamma": 0.1
    }
}
_config_test(search_space)

# The maximum training epochs
num_epochs = 5

# Number of samples from parameter space
num_samples = 10

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Tuner, TuneConfig, RunConfig

scaling_config = ray.train.ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 10, "GPU":1}
)

run_config = RunConfig(
    checkpoint_config=ray.tune.CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="val_acc",
        checkpoint_score_order="max",
    ),
)

ray_trainer =TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config,
)


scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

tuner = Tuner(
    ray_trainer,
    param_space=search_space,
    tune_config=tune.TuneConfig(
            metric="val_acc",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
    ),
)
results = tuner.fit()

