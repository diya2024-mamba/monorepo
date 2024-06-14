FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y git curl

RUN curl -L micro.mamba.pm/install.sh | bash -s
ENV MAMBA=/root/.local/bin/micromamba

WORKDIR /code

COPY environment.yml .
RUN $MAMBA create -n diya2024-mamba -f environment.yml
RUN $MAMBA run -n diya2024-mamba pip install \
	causal-conv1d==1.2.0.post2 \
	mamba-ssm==1.2.0.post1

COPY . .

RUN $MAMBA run -n diya2024-mamba pre-commit install --install-hooks

ENTRYPOINT [".devcontainer/entrypoint.sh"]
