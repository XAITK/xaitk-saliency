FROM python:3.10 AS xaitk_base

# install poetry and configure to install to pip environment
ENV PATH=/root/.local/bin:${PATH}
RUN (curl -sSL 'https://install.python-poetry.org' | python3 -) \
 && poetry config virtualenvs.create false

WORKDIR /xaitk

# install dependencies separately for caching optimization
COPY poetry.lock pyproject.toml /xaitk/
RUN poetry install --no-root

# install package
COPY docs /xaitk/docs/
COPY scripts /xaitk/scripts/
COPY tests /xaitk/tests/
COPY xaitk_saliency /xaitk/xaitk_saliency
COPY .flake8 .mypy.ini CONTRIBUTING.md LICENSE.txt README.md /xaitk/
RUN poetry install
