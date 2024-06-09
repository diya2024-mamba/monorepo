# Monorepo

For the DIYA Mamba Team of 2024

## Prerequisites

- Anaconda/Mamba (conda)
- CUDA 12.1+

You also need to create a `.env` file, see `.env.example`

## Setup

If you use VSCode, you can use the provided devcontainer (`devcontainer.json`).

To install locally using conda:

```bash
make setup-locally
conda activate mamba-benchmarks
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

Finetune a pretrained Mamba from HuggingFace:

```bash
cd src/projects/llm
python train.py
```

Evaluate a pretrained Mamba using `lm-evaluation-harness`:

```bash
cd src/projects/llm
python eval.py --model mamba_ssm --model_args pretrained=state-spaces/mamba-130m --tasks hellaswag --device cuda --batch_size 32
```
