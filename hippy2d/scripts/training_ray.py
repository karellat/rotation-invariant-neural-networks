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
import click
import importlib.util
import sys
import os

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

def load_search_space(search_space_file):
    """Load search space configuration from a Python file."""
    if not os.path.exists(search_space_file):
        raise FileNotFoundError(f"Search space file not found: {search_space_file}")
    
    # Load the module from file
    spec = importlib.util.spec_from_file_location("search_space_module", search_space_file)
    search_space_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(search_space_module)
    
    # Get the search_space variable from the module
    if not hasattr(search_space_module, 'search_space'):
        raise AttributeError(f"The file {search_space_file} must contain a 'search_space' variable")
    
    return search_space_module.search_space

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

@click.command()
@click.option('--search_space', required=True, type=click.Path(exists=True),
              help='Path to Python file containing search_space configuration')
@click.option('--num_epochs', default=10, type=int,
              help='Maximum training epochs (default: 10)')
@click.option('--num_samples', default=20, type=int,
              help='Number of samples from parameter space (default: 20)')
def main(search_space, num_epochs, num_samples):
    """Ray Tune hyperparameter optimization for neural networks"""
    
    # Load search space from file
    search_space_config = load_search_space(search_space)
    _config_test(search_space_config)

    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune import Tuner, RunConfig

    scaling_config = ray.train.ScalingConfig(
            num_workers=1, use_gpu=False, resources_per_worker={"CPU": 10}
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

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=5, reduction_factor=2)

    tuner = Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space_config},
        tune_config=tune.TuneConfig(
                metric="val_acc",
                mode="max",
                num_samples=num_samples,
                scheduler=scheduler,
        ),
    )
    results = tuner.fit()
    results.get_best_result()

if __name__ == "__main__":
    main()

