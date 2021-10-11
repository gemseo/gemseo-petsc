..
    Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

    This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
    International License. To view a copy of this license, visit
    http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
    Commons, PO Box 1866, Mountain View, CA 94042, USA.

PETSc GEMSEO interface
%%%%%%%%%%%%%%%%%%%%%%

This plugin provides an interface to the PETSc linear solvers.
They can be used for direct and adjoint linear system resolution in GEMSEO.

Installation
------------

**gemseo-petsc** relies on **petsc4py**, the Python bindings for **PETSc**.
**PETSc** and **petsc4py** are available on pypi,
but no wheel are available. Hence, depending on the initial situation, here are our recommendations:

Linux environment
=================

Using Conda
###########

**PETSc** and **petsc4py** are available on the conda-forge repository.
If you start from scratch of if you want to install the plugin in a pre-existing conda environment,
you can use the following command in your current conda environment before installing gemseo-petsc:

.. code-block::

    conda install -c conda-forge petsc4py

Using pip
#########

**PETSc** and **petsc4py** can be build from their sources by using pip.
To do so, use the following commands in your Python environment.

.. code-block::

    pip install petsc petsc4py


By building PETSc and petsc4py from sources
###########################################

It is also possible to build **PETSc** and **petsc4py** from the PETSc sources.
To do so,
please follow the information provided in the `PETSc installation manual <https://petsc.org/release/install/>`_,
and do not forget to enable the compilation of **petsc4py**.

Windows environment
===================

Although it has not be tested,
it is possible to build **PETSc** and **petsc4py** under a Windows environment,
and hence to have the **gemseo-petsc** plugin working.
A description of the procedure to build these dependencies can be found `here <https://openmdao.readthedocs.io/en/1.7.3/getting-started/mpi_windows.html>`_

Bugs/Questions
--------------

Bugs can be reported in the PETSc GEMSEO interface issue `tracker <http://forge-mdo.irt-aese.local/dev/gems/gemseo_petsc/-/issues>`_.

License
-------

The gemseo-petsc plugin is licensed under the `GNU Lesser General Public License v3 <https://www.gnu.org/licenses/lgpl-3.0.en.html.>`_

Contributors
------------

- François Gallard
- Jean-Christophe Giret
- Antoine Dechaume
