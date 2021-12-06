FROM python:3.8 AS xaitk_base

# install poetry and configure to install to pip environment
RUN pip install poetry \
&& poetry config virtualenvs.create false

WORKDIR /xaitk

# install dependencies separately for caching optimization
COPY poetry.lock pyproject.toml /xaitk/
RUN poetry install --no-root

# install package
COPY . /xaitk/
RUN poetry install
