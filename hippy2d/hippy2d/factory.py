from hippy2d import datasets, models
from typing import Dict


# Constructors from names and hparams

def _get_by_name(module, name: str, hparams: Dict):
    if hasattr(module, name):
        return getattr(module, name)(**hparams)
    else:
        raise RuntimeError(f'Unknown {module.__name__}: "{name}"')


def get_datamodule(dataset_name: str, d_hparams: Dict):
    return _get_by_name(datasets, dataset_name, d_hparams)


def get_model(model_name: str, b_hparams: Dict):
    return _get_by_name(models, model_name, b_hparams)


