FROM condaforge/miniforge3:latest

RUN  apt-get update \
  && apt-get install \
             --no-install-recommends \
             --yes \
             curl \
             graphviz \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN python -m pip install "tox<4"

RUN  curl -Os https://uploader.codecov.io/latest/linux/codecov  \
  && chmod +x codecov \
  && mv codecov /usr/bin
