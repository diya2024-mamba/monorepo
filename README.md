# benchmarks

Evaluate Mamba models using the [Eleuther AI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)

## Setup

Install [poetry](https://python-poetry.org/docs/):

```bash
pip install --upgrade pip setuptools wheel packaging
pip install poetry
```

Install dependencies and pre-commit hooks:

```bash
bash ./scripts/setup.sh
```

## Usage

Finetune a pretrained Mamba from HuggingFace:
```bash
poetry run python train.py
```
