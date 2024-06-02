FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y git

RUN pip install poetry==1.4.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /code

COPY . .

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes --dev \
	&& pip install --no-cache-dir -r requirements.txt \
	&& rm requirements.txt

RUN pre-commit install --install-hooks

CMD ["/bin/bash"]
