#!/usr/bin/python

# Export builder
from .data_loader_mnist_rot import build_mnist_rot_loader

# Make the builder function available as 'builder' for convenience
builder = build_mnist_rot_loader

# Define what gets exported when using "from mnist_rot import *"
__all__ = ['build_mnist_rot_loader', 'builder']
