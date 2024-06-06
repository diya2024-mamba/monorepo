FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y git

WORKDIR /code

RUN pip install \
	transformers==4.39.0 \
	lm-eval==0.4.2 \
	black==24.4.2 \
	pre-commit==3.7.1 \
	ruff==0.4.7 \
	causal-conv1d==1.2.0.post2 \
	mamba-ssm==1.2.0.post1

COPY . .

RUN pre-commit install --install-hooks

CMD ["/bin/bash"]
