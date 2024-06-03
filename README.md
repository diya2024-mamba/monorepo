# benchmarks

Evaluate Mamba models using the [Eleuther AI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)

## Setup

If you use VSCode, you can use the provided devcontainer (`devcontainer.json`).

To install locally:

```bash
make setup-locally
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

Finetune a pretrained Mamba from HuggingFace:

```bash
cd src
python train.py
```
