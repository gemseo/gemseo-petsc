PETSc GEMSEO interface
%%%%%%%%%%%%%%%%%%%%%%

This plugin provides an interface to the PETSc linear solvers.
They can be used for direct and adjoint linear system resolution in GEMSEO.

Installation
------------

**gemseo-petsc** relies on **petsc4py**, the Python bindings for **PETSc**.
**PETSc** and **petsc4py** are available on pypi,
but no wheel are available. Hence, depending on the initial situation, here are our recommendations:

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
