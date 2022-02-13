FROM python:3.10.2-bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    fish \
    git \
    htop \
    liblapack-dev \
    libopenblas-base \
    libopenblas-dev \
    neovim \
    python3-dev \
    tmux \
    wget \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

# install poetry
ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 

RUN curl -sSL https://install.python-poetry.org | python3 -

RUN chmod +x ${POETRY_HOME}/bin/poetry

CMD ["/bin/bash"]
