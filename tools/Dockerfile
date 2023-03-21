FROM condaforge/miniforge3:latest

ARG TOX_VERSION=3.25.1

RUN apt-get update > /dev/null && \
    apt-get install \
        --no-install-recommends --yes \
        curl graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install tox==$TOX_VERSION
