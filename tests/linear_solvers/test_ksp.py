# -*- coding: utf-8 -*-
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

from numpy import eye
from numpy import random
import pytest
import pickle
from os.path import join, dirname
from gemseo.algos.linear_solvers.linear_problem import LinearProblem
from gemseo.algos.linear_solvers.linear_solvers_factory import LinearSolversFactory

from gemseo.api import create_discipline, create_mda

from scipy.sparse import load_npz


def test_algo_list():
    """Tests the algo list detection at lib creation."""
    factory = LinearSolversFactory()
    assert factory.is_available("PetscKSPAlgos")
    assert factory.is_available("PETSC_KSP")


def test_basic():
    random.seed(1)
    n = 3
    problem = LinearProblem(eye(n), random.rand(n))
    LinearSolversFactory().execute(
        problem,
        "PETSC_KSP",
        max_iter=100000,
        view_config=True,
        preconditioner_type=None,
    )
    assert problem.residuals(True) < 1e-10


@pytest.mark.parametrize("seed", range(3))
def test_hard_conv(seed):
    random.seed(seed)
    n = 300
    problem = LinearProblem(random.rand(n, n), random.rand(n))
    LinearSolversFactory().execute(
        problem, "PETSC_KSP", max_iter=100000, view_config=True
    )
    assert problem.residuals(True) < 1e-10


@pytest.mark.parametrize("solver_type", ["gmres", "lgmres", "fgmres", "bcgs"])
@pytest.mark.parametrize("preconditioner_type", ["ilu", "jacobi"])
def test_options(solver_type, preconditioner_type):
    random.seed(1)
    n = 3
    problem = LinearProblem(random.rand(n, n), random.rand(n))
    LinearSolversFactory().execute(
        problem,
        "PETSC_KSP",
        solver_type=solver_type,
        max_iter=100000,
        preconditioner_type=preconditioner_type,
    )
    assert problem.residuals(True) < 1e-10


def test_residuals_history():
    random.seed(1)
    n = 3000
    problem = LinearProblem(random.rand(n, n), random.rand(n))
    LinearSolversFactory().execute(
        problem,
        "PETSC_KSP",
        max_iter=100000,
        preconditioner_type="ilu",
        monitor_residuals=True,
    )
    assert len(problem.residuals_history) >= 2
    assert problem.residuals(True) < 1e-10


def test_hard_pb1():
    lhs = load_npz(join(dirname(__file__), "data", "a_mat.npz"))
    rhs = pickle.load(open(join(dirname(__file__), "data", "b_vec.pkl"), "rb"))
    problem = LinearProblem(lhs, rhs)
    LinearSolversFactory().execute(
        problem,
        "PETSC_KSP",
        solver_type="gmres",
        tol=1e-13,
        atol=1e-50,
        max_iter=100,
        preconditioner_type="ilu",
        monitor_residuals=False,
    )
    assert problem.residuals(True) < 1e-3


def test_mda_adjoint():
    disciplines = create_discipline(
        [
            "SobieskiPropulsion",
            "SobieskiAerodynamics",
            "SobieskiStructure",
            "SobieskiMission",
        ]
    )
    linear_solver_options = {
        "solver_type": "gmres",
        "max_iter": 100000,
    }
    mda = create_mda(
        "MDAChain",
        disciplines,
        linear_solver="PETSC_KSP",
        linear_solver_options=linear_solver_options,
    )
    assert mda.check_jacobian(threshold=1e-4)


def test_mda_newton():
    disciplines = create_discipline(
        [
            "SobieskiPropulsion",
            "SobieskiAerodynamics",
            "SobieskiStructure",
            "SobieskiMission",
        ]
    )
    linear_solver_options = {
        "solver_type": "gmres",
        "max_iter": 100000,
    }

    tolerance = 1e-13
    mda = create_mda(
        "MDANewtonRaphson",
        disciplines,
        tolerance=tolerance,
        linear_solver="PETSC_KSP",
        linear_solver_options=linear_solver_options,
    )

    mda.execute()
    assert mda.residual_history[-1][0] <= tolerance
    assert mda.check_jacobian(threshold=1e-3)
