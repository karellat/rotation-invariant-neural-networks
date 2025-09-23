from hippy2d import optimal_invariant_cnn, flexibleconv2d
from typing import Dict


def _get_by_name(module, name: str, hparams: Dict):
    if hasattr(module, name):
        return getattr(module, name)(**hparams)
    else:
        raise RuntimeError(f'Unknown {module.__name__}: "{name}"')


def get_conv_layer(conv_name: str, conv_hparams: Dict):
    if hasattr(optimal_invariant_cnn, conv_name):
        return _get_by_name(optimal_invariant_cnn, conv_name, conv_hparams)
    elif hasattr(flexibleconv2d, conv_name):
        return _get_by_name(flexibleconv2d, conv_name, conv_hparams)
    else:
        raise RuntimeError(f'Unknown conv layer: "{conv_name}"')