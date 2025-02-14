<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# gemseo-petsc

[![PyPI - License](https://img.shields.io/pypi/l/gemseo-petsc)](https://www.gnu.org/licenses/lgpl-3.0.en.html)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gemseo-petsc)](https://pypi.org/project/gemseo-petsc/)
[![PyPI](https://img.shields.io/pypi/v/gemseo-petsc)](https://pypi.org/project/gemseo-petsc/)
[![Codecov branch](https://img.shields.io/codecov/c/gitlab/gemseo:dev/gemseo-petsc/develop)](https://app.codecov.io/gl/gemseo:dev/gemseo-petsc)

## Overview

PETSc GEMSEO interface.

This plugin provides an interface to the PETSc linear solvers and Ordinary Differential Equations (ODE) solvers.
Linear solvers can be used for direct and adjoint linear system resolution in GEMSEO.
The ODE solver provides the computation of the adjoints with respect to the initial conditions of the ODE and with
respect to the design variables.

## Installation

**gemseo-petsc** relies on **petsc4py**, the Python bindings for
**PETSc**. **PETSc** and **petsc4py** are available on pypi, but no
wheel are available. Hence, depending on the initial situation, here are
our recommendations:

### Linux environment

#### Using Conda

**PETSc** and **petsc4py** are available on the conda-forge repository.
If you start from scratch of if you want to install the plugin in a
pre-existing conda environment, you can use the following command in
your current conda environment before installing gemseo-petsc:

```shell
conda install -c conda-forge petsc4py
```

#### Using pip

**PETSc** and **petsc4py** can be build from their sources by using pip.
To do so, use the following commands in your Python environment.

```shell
pip install petsc petsc4py
```

#### By building PETSc and petsc4py from sources

It is also possible to build **PETSc** and **petsc4py** from the PETSc
sources. To do so, please follow the information provided in the [PETSc
installation manual](https://petsc.org/release/install/), and do not
forget to enable the compilation of **petsc4py**.

Although it has not be tested, it is possible to build **PETSc** and
**petsc4py** under a Windows environment, and hence to have the
**gemseo-petsc** plugin working. A description of the procedure to build
these dependencies can be found
[here](https://openmdao.readthedocs.io/en/1.7.3/getting-started/mpi_windows.html)

## Bugs and questions

Please use the [gitlab issue tracker](https://gitlab.com/gemseo/dev/gemseo-petsc/-/issues)
to submit bugs or questions.

## Contributing

See the [contributing section of GEMSEO](https://gemseo.readthedocs.io/en/stable/software/developing.html#dev).

## Contributors

- François Gallard
- Jean-Christophe Giret
- Antoine Dechaume
