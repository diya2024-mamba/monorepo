# Monorepo

For the DIYA Mamba Team of 2024

## Prerequisites

- Anaconda/Mamba (conda)
- CUDA 12.1+

You also need to create a `.env` file, see `.env.example`

## Setup

If you use VSCode, you can use the provided devcontainer (`.devcontainer`).

To install locally using conda:

```bash
make setup-locally
conda activate diya2024-mamba
pre-commit install
```

To build a docker image and run the container:

```bash
make build
make run
```

You can see the list of available commands with:

```bash
make help
```

## Usage

Please refer to the README.md files in the [projects](projects) directory for usage examples.
