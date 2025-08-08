# Project root directory
ROOT_DIR := $(shell pwd)
# Conda
ENV_NAME := hippy2d

# Export PYTHONPATH to project root
export PYTHONPATH := $(ROOT_DIR)

# Check if Conda env 'hippy2d' exists
CONDA_ENV_EXISTS := $(shell conda info --envs 2>/dev/null | awk '{print $$1}' | grep -x $(ENV_NAME) || echo "not_found")

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
	@CUDA_VISIBLE_DEVICES=0 $(PYTHON) ./scripts/training.py $(ARGS)

.PHONY: train_with_config
train_with_config:
	@echo "Using Python: $(PYTHON)"
	@CUDA_VISIBLE_DEVICES=0 $(PYTHON) ./scripts/training_with_config.py $(ARGS)

.PHONY: train_with_config
train_ray:
	@echo "Using Python: $(PYTHON)"
	@$(PYTHON) ./scripts/training_ray.py $(ARGS)

# Run tests
.PHONY: test
test:
	@echo "Using Python: $(PYTHON)"
	@$(PYTHON) -m pytest -q $(ARGS) ./tests
