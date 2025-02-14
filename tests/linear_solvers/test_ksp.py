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
"""KSP algorithm library tests."""

from __future__ import annotations

import pickle
from itertools import product
from os.path import dirname
from os.path import join
from typing import TYPE_CHECKING
from typing import Any

import pytest
from gemseo import create_discipline
from gemseo import create_mda
from gemseo.algos.linear_solvers.factory import LinearSolverLibraryFactory
from gemseo.algos.linear_solvers.linear_problem import LinearProblem
from numpy import eye
from numpy import random
from scipy.sparse import coo_matrix
from scipy.sparse import load_npz

from gemseo_petsc.linear_solvers.petsc_ksp import PetscKSP
from gemseo_petsc.linear_solvers.petsc_ksp import _convert_ndarray_to_mat_or_vec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.core.discipline import MDODiscipline
    from petsc4py import PETSc


def test_algo_list():
    """Tests the algo list detection and the algo settings at library creation."""
    factory = LinearSolverLibraryFactory()
    for solver, infos in PetscKSP.ALGORITHM_INFOS.items():
        assert factory.is_available(solver)
        assert solver == infos.Settings._TARGET_CLASS_NAME


def test_basic():
    """Test the resolution of a random linear problem."""
    rng = random.default_rng(1)
    n = 3
    problem = LinearProblem(eye(n), rng.random(n))
    LinearSolverLibraryFactory().execute(
        problem,
        algo_name="PETSC_GMRES",
        maxiter=100_000,
        view_config=True,
        preconditioner_type=None,
    )
    assert problem.compute_residuals(True) < 1e-10


def test_basic_using_hook():
    """Test the resolution of a random linear problem."""

    def func(
        ksp: PETSc.KSP,
        options: Mapping[str, Any],
    ):
        """Set the options of the KSP with options."""
        ksp.setType("cg")

    rng = random.default_rng(1)
    n = 3
    problem = LinearProblem(eye(n), rng.random(n))
    LinearSolverLibraryFactory().execute(
        problem,
        algo_name="PETSC_GMRES",
        maxiter=100_000,
        view_config=True,
        preconditioner_type=None,
        ksp_pre_processor=func,
    )
    assert problem.compute_residuals(True) < 1e-10


def test_basic_with_options():
    """Test the resolution of a random linear problem."""
    rng = random.default_rng(1)
    n = 3
    problem = LinearProblem(eye(n), rng.random(n))
    petsc_options = {"ksp_type": "cg"}
    LinearSolverLibraryFactory().execute(
        problem,
        algo_name="PETSC_GMRES",
        maxiter=100_000,
        view_config=True,
        preconditioner_type=None,
        options_cmd=petsc_options,
    )
    assert problem.compute_residuals(True) < 1e-10


def test_basic_set_from_options():
    """Test the resolution with options set from command line.

    Note that, as we run the test from pytest, we cannot verify that the options are
    passed from the command line.
    """
    rng = random.default_rng(1)
    n = 3
    problem = LinearProblem(eye(n), rng.random(n))
    LinearSolverLibraryFactory().execute(
        problem,
        algo_name="PETSC_GMRES",
        maxiter=100_000,
        view_config=True,
        preconditioner_type=None,
        set_from_options=True,
    )
    assert problem.compute_residuals(True) < 1e-10


@pytest.mark.parametrize("seed", range(3))
def test_hard_conv(seed):
    """Test the resolution of a pseudo-random large linear problem."""
    rng = random.default_rng(seed)
    n = 300
    problem = LinearProblem(rng.random((n, n)), rng.random(n))
    LinearSolverLibraryFactory().execute(
        problem, algo_name="PETSC_GMRES", maxiter=100_000, view_config=True
    )
    assert problem.compute_residuals(True) < 1e-10


@pytest.mark.parametrize("solver_type", ["GMRES", "LGMRES", "FGMRES", "BCGS"])
@pytest.mark.parametrize("preconditioner_type", ["ilu", "jacobi"])
def test_options(solver_type, preconditioner_type):
    """Test the options to be passed to PETSc."""
    rng = random.default_rng(1)
    n = 3
    problem = LinearProblem(rng.random((n, n)), rng.random(n))
    LinearSolverLibraryFactory().execute(
        problem,
        algo_name=f"PETSC_{solver_type}",
        maxiter=100_000,
        preconditioner_type=preconditioner_type,
    )
    assert problem.compute_residuals(True) < 1e-10


def test_residuals_history():
    """Test that the residual history is correctly computed."""
    rng = random.default_rng(1)
    n = 3000
    problem = LinearProblem(rng.random((n, n)), rng.random(n))
    LinearSolverLibraryFactory().execute(
        problem,
        algo_name="PETSC_GMRES",
        maxiter=100_000,
        preconditioner_type="ilu",
        monitor_residuals=True,
    )
    assert len(problem.residuals_history) >= 2
    assert problem.compute_residuals(True) < 1e-10


def test_hard_pb1():
    """Test with a hard problem."""
    lhs = load_npz(join(dirname(__file__), "data", "a_mat.npz"))
    with open(join(dirname(__file__), "data", "b_vec.pkl"), "rb") as f:
        rhs = pickle.load(f)
    problem = LinearProblem(lhs, rhs)
    LinearSolverLibraryFactory().execute(
        problem,
        algo_name="PETSC_GMRES",
        rtol=1e-13,
        atol=1e-50,
        maxiter=100,
        preconditioner_type="ilu",
        monitor_residuals=False,
    )
    assert problem.compute_residuals(True) < 1e-3


@pytest.fixture
def sobieski_disciplines() -> list[MDODiscipline]:
    """Return the Sobieski disciplines.

    Returns:
         The Sobieski disciplines.
    """
    return create_discipline([
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiStructure",
        "SobieskiMission",
    ])


def test_mda_adjoint(sobieski_disciplines):
    """Test with an MDA with total derivatives computed with adjoint."""
    linear_solver_settings = {
        "maxiter": 100_000,
    }
    inner_mda_settings = {
        "linear_solver": "PETSC_GMRES",
        "linear_solver_settings": linear_solver_settings,
    }
    mda = create_mda(
        "MDAChain", sobieski_disciplines, inner_mda_settings=inner_mda_settings
    )
    assert mda.check_jacobian(threshold=1e-4)


def test_mda_newton(sobieski_disciplines):
    """Test a Newton MDA."""
    linear_solver_settings = {
        "maxiter": 100_000,
    }

    tolerance = 1e-13
    mda = create_mda(
        "MDANewtonRaphson",
        sobieski_disciplines[:3],
        tolerance=tolerance,
        newton_linear_solver_name="PETSC_GMRES",
        newton_linear_solver_settings=linear_solver_settings,
    )

    mda.execute()
    assert mda.residual_history[-1] <= tolerance
    assert mda.check_jacobian(threshold=1e-3)


def test_convert_ndarray_to_numpy():
    """Test that an exception is raised if the dimension of the ndarray > 2."""
    rng = random.default_rng(1)
    wrong_nd_array = rng.random((2, 2, 2))
    with pytest.raises(
        ValueError, match=r"The dimension of the input array \(\d*\) is not supported\."
    ):
        _convert_ndarray_to_mat_or_vec(wrong_nd_array)


def test_convert_ndarray_coo_to_mat():
    """Test that the conversion is correctly made from a sparse COO matrix to PETSc."""
    rng = random.default_rng(1)
    nd_array = rng.random((5, 5))
    coo_mat = coo_matrix(nd_array)
    petsc_mat = _convert_ndarray_to_mat_or_vec(coo_mat)

    # Somewhat inelegant but is there another way than comparing element by element?
    # Also, it only works if the test is running one only one MPI rank.
    for i, j in product(range(5), range(5)):
        assert nd_array[i, j] == petsc_mat[i, j]


def test_convert_1d_dense_ndarray_to_vec():
    """Test that the conversion is correctly made from a sparse COO matrix to PETSc."""
    rng = random.default_rng(1)
    nd_array = rng.random(5)
    petsc_vec = _convert_ndarray_to_mat_or_vec(nd_array)
    for i in range(5):
        assert nd_array[i] == petsc_vec[i]
