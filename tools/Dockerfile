FROM registry.gitlab.com/gemseo/dev/gemseo/multi-python:8

RUN  dnf -y install petsc \
  && dnf clean all \
  && rm -rf /var/cache/dnf
