FROM python:3.12-slim

# Set up locale
RUN apt-get update && \
  apt-get install -y \
  locales \
  locales-all && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

WORKDIR /app

# Install dependencies
RUN pip install poetry==1.8.3
COPY pyproject.toml poetry.lock ./
RUN poetry install --without dev

COPY . .

EXPOSE 8080

CMD ["poetry", "run", "gunicorn", "--bind", "0.0.0.0:8080", "app:me"]
