FROM python:3.10-bullseye

ENV DEBIAN_FRONTEND noninteractive
COPY pyproject.toml poetry.loc[k] README.md /
RUN curl -sSL https://install.python-poetry.org | python - && \
    echo 'export PATH="/root/.local/bin:$PATH"' > ~/.bashrc && \
    export PATH="/root/.local/bin:$PATH"  && \
    poetry config virtualenvs.create false && \
    poetry self add poetry-bumpversion && \
    poetry install --no-root && \
    echo "/workspaces/pydantic-store/src/" > /usr/local/lib/python3.10/site-packages/pydantic_store.pth