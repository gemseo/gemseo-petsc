# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A PETSC KSP linear solvers library wrapper."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

import petsc4py
from gemseo.algos.linear_solvers.base_linear_solver_library import (
    BaseLinearSolverLibrary,
)
from gemseo.algos.linear_solvers.base_linear_solver_library import (
    LinearSolverDescription,
)
from numpy import arange
from numpy import array
from numpy import ndarray
from scipy.sparse import csr_matrix
from scipy.sparse import find
from scipy.sparse.base import issparse

from gemseo_petsc.linear_solvers.settings.petsc_ksp_settings import BasePetscKSPSettings
from gemseo_petsc.linear_solvers.settings.petsc_ksp_settings import PreconditionerType

# Must be done before from petsc4py import PETSc, this loads the options from
# command args in the options database.
petsc4py.init(sys.argv)
from petsc4py import PETSc  # noqa: E402

if TYPE_CHECKING:
    from gemseo.algos.linear_solvers.linear_problem import LinearProblem

LOGGER = logging.getLogger(__name__)


@dataclass
class PetscKSPAlgorithmDescription(LinearSolverDescription):
    """The description of the PETSc KSP linear algebra library."""

    library_name: str = "PETSc KSP"
    """The library name."""

    lhs_must_be_linear_operator: bool = True
    """Whether the left-hand side matrix must be a linear operator."""


class PetscKSP(BaseLinearSolverLibrary):
    """Interface to PETSC KSP.

    For further information, please read
    [https://petsc4py.readthedocs.io/en/stable/manual/ksp/]
    and
    https://petsc.org/release/docs/manualpages/KSP/KSP.html#KSP].
    """

    PreconditionerType = PreconditionerType

    __SOLVER_LIST: tuple[str, ...] = (
        "RICHARDSON",
        "CHEBYSHEV",
        "CG",
        "GROPPCG",
        "PIPECG",
        "PIPECGRR",
        "PIPELCG",
        "PIPEPRCG",
        "PIPECG2",
        "CGNE",
        "NASH",
        "STCG",
        "GLTR",
        "FCG",
        "PIPEFCG",
        "GMRES",
        "PIPEFGMRES",
        "FGMRES",
        "LGMRES",
        "DGMRES",
        "PGMRES",
        "TCQMR",
        "BCGS",
        "IBCGS",
        "FBCGS",
        "FBCGSR",
        "BCGSL",
        "PIPEBCGS",
        "CGS",
        "TFQMR",
        "CR",
        "PIPECR",
        "LSQR",
        "PREONLY",
        "QCG",
        "BICG",
        "MINRES",
        "SYMMLQ",
        "LCD",
        "GCR",
        "PIPEGCR",
        "TSIRM",
        "CGLS",
        "FETIDP",
        "HPDDM",
    )
    """The available linear algebra solvers."""

    ALGORITHM_INFOS: ClassVar[dict[str, PetscKSPAlgorithmDescription]] = {
        f"PETSC_{solver_name}": PetscKSPAlgorithmDescription(
            algorithm_name=solver_name,
            internal_algorithm_name=solver_name.lower(),
            website=f"https://petsc.org/release/manualpages/KSP/KSP{solver_name}/",
            Settings=type(
                f"PETSC_{solver_name}_Settings",
                (BasePetscKSPSettings,),
                {"_TARGET_CLASS_NAME": f"PETSC_{solver_name}"},
            ),
        )
        for solver_name in __SOLVER_LIST
    }

    def _run(self, problem: LinearProblem, **settings: Any) -> ndarray:
        """Run the algorithm.

        Args:
            **options: The algorithm options.

        Returns:
            The solution of the problem.
        """
        rhs = problem.rhs
        if issparse(rhs):
            rhs = problem.rhs.toarray()

        # Initialize the KSP solver.
        # Create the options database
        options_cmd = settings["options_cmd"]
        if options_cmd is not None:
            petsc4py.init(options_cmd)
        else:
            petsc4py.init()
        ksp = PETSc.KSP().create()
        ksp.setType(self.ALGORITHM_INFOS[self._algo_name].internal_algorithm_name)
        ksp.setTolerances(
            settings["rtol"], settings["atol"], settings["dtol"], settings["maxiter"]
        )
        ksp.setConvergenceHistory()
        a_mat = _convert_ndarray_to_mat_or_vec(problem.lhs)
        ksp.setOperators(a_mat)
        prec_type = settings["preconditioner_type"]
        if prec_type is not None:
            pc = ksp.getPC()
            pc.setType(prec_type)
            pc.setUp()

        # Allow for solver choice to be set from command line with -ksp_type <solver>.
        # Recommended option: -ksp_type preonly -pc_type lu
        if settings["set_from_options"]:
            ksp.setFromOptions()

        ksp_pre_processor = settings["ksp_pre_processor"]
        if ksp_pre_processor is not None:
            ksp_pre_processor(ksp, settings)

        problem.residuals_history = []
        if settings["monitor_residuals"]:
            LOGGER.warning(
                "Petsc option monitor_residuals slows the process and"
                " should be used only for testing or convergence studies."
            )
            ksp.setMonitor(_Monitor(problem))

        b_mat = _convert_ndarray_to_mat_or_vec(problem.rhs)
        solution = b_mat.duplicate()
        if settings["view_config"]:
            ksp.view()
        ksp.solve(b_mat, solution)
        problem.solution = solution.getArray().copy()
        problem.convergence_info = ksp.reason
        return problem.solution


class _Monitor:
    """A functor to allow monitoring the residual history."""

    __problem: LinearProblem
    """The problem."""

    def __init__(self, problem: LinearProblem) -> None:
        self.__problem = problem

    def __call__(
        self,
        ksp: PETSc.KSP,
        its: int,
        rnorm: float,
    ) -> None:
        """Add the normed residual value to the problem residual history.

        This method is aimed to be passed to petsc4py as a reference.
        This is the reason why some of its arguments are not used.

        Args:
            ksp: The KSP PETSc solver.
            its: The current iteration.
            rnorm: The normed residual.
        """
        self.__problem.residuals_history.append(rnorm)


def _convert_ndarray_to_mat_or_vec(
    np_arr: ndarray,
) -> PETSc.Mat | PETSc.Vec:
    """Convert a Numpy array to a PETSc Mat or Vec.

    Args:
         np_arr: The input Numpy array.

    Returns:
        A PETSc Mat or Vec, depending on the input dimension.

    Raises:
        ValueError: If the dimension of the input vector is greater than 2.
    """
    n_dim = np_arr.ndim
    if n_dim > 2:
        msg = f"The dimension of the input array ({n_dim}) is not supported."
        raise ValueError(msg)

    if issparse(np_arr):
        if not isinstance(np_arr, csr_matrix):
            np_arr = np_arr.tocsr()
        if n_dim == 2 and np_arr.shape[1] > 1:
            petsc_arr = PETSc.Mat().createAIJ(
                size=np_arr.shape, csr=(np_arr.indptr, np_arr.indices, np_arr.data)
            )
            petsc_arr.assemble()
        else:
            petsc_arr = PETSc.Vec().createSeq(np_arr.shape[0])
            petsc_arr.setUp()
            inds, _, vals = find(np_arr)
            petsc_arr.setValues(inds, vals)
            petsc_arr.assemble()
    else:
        if n_dim == 1:
            a = array(np_arr, dtype=PETSc.ScalarType)
            petsc_arr = PETSc.Vec().createWithArray(a)
            petsc_arr.assemble()
        else:
            petsc_arr = PETSc.Mat().createDense(np_arr.shape)
            a_shape = np_arr.shape
            petsc_arr.setUp()
            petsc_arr.setValues(
                arange(a_shape[0], dtype="int32"),
                arange(a_shape[1], dtype="int32"),
                np_arr,
            )
            petsc_arr.assemble()
    return petsc_arr


# KSP example here
# https://fossies.org/linux/petsc/src/binding/petsc4py/demo/petsc-examples/ksp/ex2.py
