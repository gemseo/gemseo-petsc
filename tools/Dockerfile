FROM registry.gitlab.com/gemseo/dev/gemseo/multi-python:latest

RUN  dnf -y install petsc \
  && dnf clean all \
  && rm -rf /var/cache/dnf
