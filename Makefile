IMAGE_NAME = mamba-benchmarks
CONTAINER_NAME = mamba-benchmarks-container
TAG = latest

# Determine the current platform
CURRENT_ARCH = $(shell uname -m)

# Check if running on macOS
IS_MAC = $(shell uname -s | grep Darwin)

# Set the run platform based on CURRENT_ARCH and IS_MAC
ifeq ($(IS_MAC),Darwin)
  ifeq ($(CURRENT_ARCH),x86_64)
    PLATFORM = linux/arm64
  else ifeq ($(CURRENT_ARCH),arm64)
    PLATFORM = linux/arm64
  else
    PLATFORM = linux/amd64
  endif
else
  ifeq ($(CURRENT_ARCH),x86_64)
    PLATFORM = linux/amd64
  else ifeq ($(CURRENT_ARCH),aarch64)
    PLATFORM = linux/arm64
  else
    PLATFORM = linux/amd64
  endif
endif

.PHONY: setup-locally
setup-locally:
	pip install poetry
	poetry export -f requirements.txt --output requirements.txt --without-hashes --with dev
	poetry run pip install -r requirements.txt
	rm requirements.txt
	poetry run pre-commit install --install-hooks
	poetry shell

.PHONY: build
build:
	docker build --platform $(PLATFORM) -f .devcontainer/Dockerfile -t $(IMAGE_NAME):$(TAG) .

.PHONY: run
run:
	@if [ "$(IS_MAC)" = "Darwin" ]; then \
		docker run --platform $(PLATFORM) -it --name $(CONTAINER_NAME) $(IMAGE_NAME); \
	else \
		docker run --platform $(PLATFORM) --gpus all -it --name $(CONTAINER_NAME) $(IMAGE_NAME); \
	fi

.PHONY: stop
stop:
	docker stop $(CONTAINER_NAME) || true

.PHONY: rm
rm:
	docker rm $(CONTAINER_NAME) || true

.PHONY: clean
clean: stop rm
	docker rmi $(IMAGE_NAME):$(TAG)

.PHONY: help
help:
	@echo "Makefile commands:"
	@echo "  setup-locally  Install the required dependencies locally"
	@echo "  build          Build the Docker image"
	@echo "  run            Run the Docker container interactively"
	@echo "  stop           Stop the running Docker container"
	@echo "  rm             Remove the stopped Docker container"
	@echo "  clean          Clean up Docker images and containers"
