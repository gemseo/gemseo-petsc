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
#        :author: Isabelle Santos
#        :author: Giulio Gargantini
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for the TS ODE solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gemseo.algos.ode.factory import ODESolverLibraryFactory
from gemseo.algos.ode.ode_problem import ODEProblem
from numpy import allclose
from numpy import arange
from numpy import array
from numpy import exp
from numpy import isclose
from numpy import ndarray
from numpy import sqrt
from numpy import zeros
from src.gemseo_petsc.problems.smooth_ode import SmoothODE

from gemseo_petsc.ode.ts_library import PetscOdeAlgo

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_algo_list():
    """Tests the algo list detection at lib creation."""
    factory = ODESolverLibraryFactory()
    assert factory.is_available("PETSC_ODE_RK")


def test_algo_list_full():
    """Tests the algo list detection and the algo settings at library creation."""
    factory = ODESolverLibraryFactory()
    for solver, infos in PetscOdeAlgo.ALGORITHM_INFOS.items():
        assert factory.is_available(solver)
        assert solver == infos.Settings._TARGET_CLASS_NAME


def test_solver_1d_problem_fixed_times():
    """Test the PETSc solver on a 1-D problem with a known analytical solution
    for a given set of times."""

    def _func(time: float, state: NDArray[float]) -> NDArray[float]:
        return state.copy()

    times = arange(0, 1, 0.1)
    problem = ODEProblem(
        _func,
        initial_state=array([1, 2]),
        times=times,
    )
    ODESolverLibraryFactory().execute(
        problem, algo_name="PETSC_ODE_RK", time_step=1e-3, maximum_steps=1000, atol=1e-8
    )
    assert problem.result.algorithm_has_converged
    analytic_solution = array([exp(times), 2 * exp(times)])
    assert allclose(problem.result.state_trajectories, analytic_solution, atol=1.0e-3)


def test_solver_1d_problem_algo_times():
    """Test the PETSc solver on a 1-D problem with a known analytical solution
    for a set of times chosen by the quadrature algorithm."""

    def _func(time: float, state: NDArray[float]) -> NDArray[float]:
        return state.copy()

    times = array((0, 1))
    problem = ODEProblem(
        _func, initial_state=array([1, 2]), times=times, solve_at_algorithm_times=True
    )
    ODESolverLibraryFactory().execute(
        problem, algo_name="PETSC_ODE_RK", time_step=1e-3, maximum_steps=1000, atol=1e-8
    )
    assert problem.result.algorithm_has_converged
    analytic_solution = array([
        exp(problem.result.times),
        2 * exp(problem.result.times),
    ])
    assert allclose(problem.result.state_trajectories, analytic_solution, atol=1.0e-3)


def test_solver_1d_problem_final_time():
    """Test the PETSc solver on a 1-D problem with a known analytical solution
    at final time."""

    def _func(time: float, state: NDArray[float]) -> NDArray[float]:
        return state.copy()

    def _jac(time: float, state: NDArray[float]) -> NDArray[float]:
        return array([[1.0, 0.0], [0.0, 1.0]])

    times = array((0, 1))
    problem = ODEProblem(
        _func, initial_state=array([1, 2]), times=times, jac_function_wrt_state=_jac
    )
    ODESolverLibraryFactory().execute(
        problem,
        algo_name="PETSC_ODE_BEULER",
        time_step=1e-3,
        maximum_steps=100000,
        atol=1e-8,
        use_jacobian=True,
    )
    assert problem.result.algorithm_has_converged
    analytic_solution = exp(problem.result.termination_time) * array([1, 2])
    assert allclose(problem.result.final_state, analytic_solution, atol=1.0e-2)


def test_vanderpol_problem():
    """Test the PETSc solver on a 2-D problem."""
    mu = 0.5

    st_dot = zeros(2)

    def _func(time: float, state: NDArray[float]) -> NDArray[float]:
        st_dot[0] = state[1]
        st_dot[1] = mu * ((1.0 - state[0] * state[0]) * state[1] - state[0])
        return st_dot

    initial_state = array([
        2.0,
        -2.0 / 3.0 + 10.0 / (81.0 * mu) - 292.0 / (2187.0 * mu * mu),
    ])
    time_vector = arange(0, 1, 0.1)
    time_step = 0.001
    max_steps = 1000

    problem = ODEProblem(
        _func,
        initial_state=initial_state,
        times=time_vector,
    )

    assert problem.result.n_func_evaluations == 0
    assert problem.result.n_jac_evaluations == 0
    assert problem.result.state_trajectories.size == 0

    ODESolverLibraryFactory().execute(
        problem,
        algo_name="PETSC_ODE_RK",
        time_step=time_step,
        maximum_steps=max_steps,
    )


def test_error_providing_jac():
    """Test the error message when no Jacobian function is provided while required."""

    def _func(time: float, state: NDArray[float]) -> NDArray[float]:
        return state.copy()

    problem = ODEProblem(
        _func,
        initial_state=array([1, 1]),
        times=arange(0, 1, 0.1),
    )

    with pytest.raises(
        ValueError, match=r"Jacobian of RHS function wrt state must be provided."
    ):
        ODESolverLibraryFactory().execute(
            problem,
            algo_name="PETSC_ODE_BEULER",
            time_step=1e-3,
            maximum_steps=1000,
            use_jacobian=True,
        )

    with pytest.raises(
        ValueError,
        match=r"'use_jacobian' setting is mandatory when 'compute_adjoint' is True",
    ):
        ODESolverLibraryFactory().execute(
            problem,
            algo_name="PETSC_ODE_BEULER",
            time_step=1e-3,
            maximum_steps=1000,
            compute_adjoint=True,
        )


def test_error_jacobian_shape():
    """Test the error message when the Jacobian has not the right shape."""

    def _func(time: float, state: NDArray[float]) -> NDArray[float]:
        return array([state[0], -state[1]])

    def _jac_1(time: float, state: NDArray[float]) -> NDArray[float]:
        return array([1, -1])

    def _jac_2(time: float, state: NDArray[float]) -> NDArray[float]:
        return array([[1, -1]])

    def _jac_3(time: float, state: NDArray[float]) -> NDArray[float]:
        return array([[1], [-1]])

    for _ii, _jac in enumerate([_jac_1, _jac_2, _jac_3]):
        problem = ODEProblem(
            _func,
            initial_state=array([1, 1]),
            times=arange(0, 1, 0.1),
            jac_function_wrt_state=_jac,
        )

        with pytest.raises(
            ValueError, match=r"The Jacobian should be a square matrix with shape"
        ):
            ODESolverLibraryFactory().execute(
                problem,
                algo_name="PETSC_ODE_BEULER",
                time_step=1e-3,
                maximum_steps=100000,
                atol=1e-8,
                use_jacobian=True,
            )


def test_non_convergence():
    """Test the error message when the quadrature algorithm does not converge."""

    def _func(time: float, state: NDArray[float]) -> NDArray[float]:
        state_1_dot = (1 - time**2) / time**2 * state[0] - state[1] / time
        return array([state[1], state_1_dot])

    problem = ODEProblem(
        _func,
        initial_state=array([1.0, 1.0]),
        times=arange(-1, 1, 0.1),
    )
    ODESolverLibraryFactory().execute(
        problem, algo_name="PETSC_ODE_RK", time_step=1e2, maximum_steps=10
    )

    assert problem.result.algorithm_has_converged is False


def test_one_termination_event():
    """Test the termination event."""

    def _func(time: float, state: NDArray[float]) -> NDArray[float]:
        return array([state[1], -10.0])

    def _termination(time: float, state: NDArray[float]) -> NDArray[float]:
        return state[0]

    problem = ODEProblem(
        _func,
        initial_state=array([10.0, 0.0]),
        times=arange(0, 5, 0.1),
        event_functions=(_termination,),
    )

    ODESolverLibraryFactory().execute(
        problem, algo_name="PETSC_ODE_RK", time_step=1e-3, maximum_steps=1000
    )

    assert isclose(problem.result.termination_time, sqrt(2.0))
    assert problem.result.terminal_event_index == 0


def test_multiple_termination_events():
    """Test the case when multiple termination events are present."""

    def _func(time: float, state: NDArray[float]) -> NDArray[float]:
        return array([state[1], -10.0])

    def _termination1(time: float, state: NDArray[float]) -> NDArray[float]:
        return state[0]

    def _termination2(time: float, state: NDArray[float]) -> NDArray[float]:
        return 20.0 - state[0]

    problem1 = ODEProblem(
        _func,
        initial_state=array([10.0, 0.0]),
        times=arange(0, 5, 0.1),
        event_functions=(_termination1, _termination2),
    )

    ODESolverLibraryFactory().execute(
        problem1, algo_name="PETSC_ODE_RK", time_step=1e-3, maximum_steps=1000
    )

    assert isclose(problem1.result.termination_time, sqrt(2.0))
    assert problem1.result.terminal_event_index == 0

    problem2 = ODEProblem(
        _func,
        initial_state=array([10.0, 0.0]),
        times=arange(0, 5, 0.1),
        event_functions=(_termination2, _termination1),
    )

    ODESolverLibraryFactory().execute(
        problem2, algo_name="PETSC_ODE_RK", time_step=1e-3, maximum_steps=1000
    )

    assert isclose(problem2.result.termination_time, sqrt(2.0))
    assert problem2.result.terminal_event_index == 1


def test_failure_snes():
    """Test the failure of the SNES solver."""

    def _func(time: float, state: NDArray[float]) -> NDArray[float]:
        return state.copy() ** 2

    times = array((0, 1))
    problem = ODEProblem(
        _func, initial_state=array([1]), times=times, solve_at_algorithm_times=True
    )
    ODESolverLibraryFactory().execute(
        problem,
        algo_name="PETSC_ODE_ALPHA",
        time_step=1e-3,
        maximum_steps=1000,
        atol=1e-8,
    )
    assert not problem.result.algorithm_has_converged


def test_check_final_state():
    def create_smooth_ode(initial_state=1.0):
        return SmoothODE(is_k_design_var=True, initial_state=initial_state)

    problem = create_smooth_ode(1.0)

    ODESolverLibraryFactory().execute(
        problem,
        algo_name="PETSC_ODE_RK",
        time_step=0.01,
        maximum_steps=1000,
        compute_adjoint=True,
        use_jacobian=True,
        atol=1e-5,
        rtol=1e-5,
        use_memory_checkpoints=True,
        max_disk_checkpoints=100,
        max_memory_checkpoints=100,
    )

    assert isinstance(problem.result.final_state, ndarray)
