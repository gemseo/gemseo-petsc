# The image of the CI of the copier template:
# the PETSc 3.20 of Fedora 41, which the petsc4py wheels of the package registry
# of this project are built against, with the just of the gemseo test image
# and the same version of uv.
FROM registry.gitlab.com/gemseo/dev/gemseo-petsc/multi-python-petsc:latest

COPY --from=registry.gitlab.com/gemseo/dev/gemseo/multi-python:8 /usr/bin/just /usr/local/bin/
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /usr/local/bin/
