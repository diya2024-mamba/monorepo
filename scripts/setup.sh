#!/bin/bash

# install dependencies
poetry install --without mamba

# install mamba-ssm
poetry install --only mamba

# install pre-commit hooks
pre-commit install
