# Project root directory
ROOT_DIR := $(shell pwd)
# Conda
ENV_NAME := hippy2d
REPOSITORY := 'https://github.com/karellat/rotation-invariant-neural-networks.git'
# Server arguments 
SERVER_DIR = '~/rotation-invariant-neural-networks/hippy2d'

# Export PYTHONPATH to project root
export PYTHONPATH := $(ROOT_DIR)

# Check if Conda env 'hippy2d' exists
CONDA_ENV_EXISTS := $(shell conda info --envs 2>/dev/null | awk '{print $$1}' | grep -x $(ENV_NAME) || echo "not_found")

# Repository setup
COMMIT=$(shell git rev-parse HEAD)
BRANCH=$(shell git rev-parse --abbrev-ref HEAD)

# Python executable depending on environment
ifeq ($(CONDA_ENV_EXISTS), $(ENV_NAME))
	PYTHON := $(shell conda run -n $(ENV_NAME) which python)
else
	PYTHON := $(ROOT_DIR)/.venv/bin/python
endif

# Target: train
.PHONY: train
train:
	@echo "Using Python: $(PYTHON)"
	@CUDA_VISIBLE_DEVICES=1 $(PYTHON) ./scripts/training.py $(ARGS)

# Run tests
.PHONY: test
test:
	@echo "Using Python: $(PYTHON)"
	@$(PYTHON) -m pytest --tb=short --disable-warnings -q $(ARGS) ./tests

# Connect to Darwin, allocate the GPU resource
.PHONY: connect
connect:
	@echo "Connecting to Darwin"
	@echo "Current commit: $(COMMIT) on branch: $(BRANCH)"
