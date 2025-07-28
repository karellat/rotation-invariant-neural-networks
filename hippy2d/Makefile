# Project root directory
ROOT_DIR := $(shell pwd)

# Export PYTHONPATH to project root
export PYTHONPATH := $(ROOT_DIR)

# Check if Conda env 'inv-test' exists
CONDA_ENV_EXISTS := $(shell conda info --envs 2>/dev/null | awk '{print $$1}' | grep -x 'inv-test')

# Python executable depending on environment
ifeq ($(CONDA_ENV_EXISTS),inv-test)
	PYTHON := $(shell conda run -n inv-test which python)
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