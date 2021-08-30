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

from gemseo.algos.linear_solvers.linear_problem import LinearProblem
from gemseo.algos.linear_solvers.linear_solvers_factory import LinearSolversFactory
from gemseo_petsc.linear_solvers.ksp_lib import PetscKSPAlgos

from gemseo.api import create_discipline,create_mda


def test_algo_list():
    """Tests the algo list detection at lib creation."""
    factory = LinearSolversFactory()
    assert factory.is_available("PetscKSPAlgos")
    assert factory.is_available("PETSC_KSP")
    
def test_basic():
    random.seed(1)
    n=3
    problem = LinearProblem(eye(n), random.rand(n))
    LinearSolversFactory().execute(
        problem,
        "PETSC_KSP",
        max_iter=100000,
        view_config=True,
        preconditioner_type=None
    )
    assert problem.residuals(True) < 1e-10
    
@pytest.mark.parametrize("seed", range(3))
def test_hard_conv(seed):
    random.seed(seed)
    n = 300
    problem = LinearProblem(random.rand(n, n), random.rand(n))
    LinearSolversFactory().execute(
        problem,
        "PETSC_KSP",
        max_iter=100000,
        view_config=True
    )
    assert problem.residuals(True) < 1e-10

__petsc=PetscKSPAlgos()
opt_grammar=__petsc.init_options_grammar("PETSC_KSP")
options=opt_grammar.schema.to_dict()
@pytest.mark.parametrize("solver_type", options["properties"]["solver_type"]["enum"])
@pytest.mark.parametrize("preconditioner_type", ['ilu','jacobi','sor'])
def test_options(solver_type, preconditioner_type):
    random.seed(1)
    n = 3
    problem = LinearProblem(random.rand(n, n), random.rand(n))
    LinearSolversFactory().execute(
        problem,
        "PETSC_KSP",
        solver_type=solver_type,
        max_iter=100000,
        preconditioner_type=preconditioner_type
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
        monitor_residuals=True
    )
    assert len(problem.residuals_history)>=2
    assert problem.residuals(True) < 1e-10
    
    
def test_mda_adjoint():
    disciplines=create_discipline(["SobieskiPropulsion","SobieskiAerodynamics",
                                   "SobieskiStructure","SobieskiMission"])
    linear_solver_options={"solver_type":"gmres",
        "max_iter":100000,
        "view_config":True}
    mda=create_mda("MDAChain",disciplines,linear_solver="PETSC_KSP",
                   linear_solver_options=linear_solver_options)
    mda.add_differentiated_inputs(["x_shared"])
    mda.add_differentiated_outputs(["y_4"])
    mda.linearize()
