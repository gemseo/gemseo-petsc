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
#                 Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import annotations

from typing import Any

import pytest
from gemseo.algos.ode.factory import ODESolverLibraryFactory

from gemseo_petsc.problems.smooth_ode import SmoothODE
from gemseo_petsc.problems.vanderpol import VanderPol


def solve_ode(ode, compute_adjoint=False, **options):
    ODESolverLibraryFactory().execute(
        ode,
        algo_name="PETSC_ODE_RK",
        time_step=0.00001,
        maximum_steps=1000000,
        compute_adjoint=compute_adjoint,
        use_jacobian=True,
        atol=1e-10,
        rtol=1e-10,
        **options,
    )


def get_fd_and_exact_jacobians(
    problem_factory, epsilon: float, desvar: float, **solve_options: Any
):
    """Computes exact and finite differences jacobian.

    Args:
        problem_factory: A method that creates the problem, and takes
            the design variable value as argument.
        epsilon: The finite differences step.
        desvar: The design variable value.
        solve_options: Options passed to the driver.

    Returns:
        The result of the reference run and the approximate Jacobian.
    """
    problem_ref = problem_factory(desvar)
    solve_ode(problem_ref, True, **solve_options)

    problem_pert_plus = problem_factory(desvar + epsilon)
    solve_ode(problem_pert_plus)
    problem_pert_min = problem_factory(desvar - epsilon)
    solve_ode(problem_pert_min)

    approx_jac = (
        problem_pert_plus.result.state_trajectories[:, -1]
        - problem_pert_min.result.state_trajectories[:, -1]
    ) / (2 * epsilon)
    return problem_ref.result, approx_jac


def test_vdp_with_adjoint():
    """Test the Van der Pol problem using the adjoint."""

    ref_result, approx_jac = get_fd_and_exact_jacobians(
        VanderPol, 1e-6, 0.5, use_memory_checkpoints=True
    )
    assert ref_result.algorithm_has_converged
    last_state = ref_result.state_trajectories[:, -1]
    assert last_state.shape == (2,)
    jacobian_desvars = ref_result.jac_wrt_desvar
    assert jacobian_desvars.shape == (2, 1)
    assert jacobian_desvars.flatten() == pytest.approx(approx_jac, 1e-4)


def test_smooth_adjoint_init_cond():
    """Test the SmoothODE problem adjoint of initial conditions."""

    def create_smooth_ode(initial_state=1.0):
        return SmoothODE(is_k_design_var=False, initial_state=initial_state)

    ref_result, approx_jac = get_fd_and_exact_jacobians(
        create_smooth_ode, epsilon=1e-6, desvar=1.0, use_memory_checkpoints=True
    )

    assert ref_result.algorithm_has_converged
    last_state = ref_result.state_trajectories[:, -1]
    assert last_state.shape == (1,)
    jacobian_init_state = ref_result.jac_wrt_initial_state
    assert jacobian_init_state.shape == (1, 1)
    assert jacobian_init_state.flatten() == pytest.approx(approx_jac, 1e-4)
    assert jacobian_init_state[0][0] == pytest.approx(4, 1e-3)


def test_smooth_ode_adjoint_desvars():
    """Test the SmoothODE problem adjoint of design variables."""

    def create_smooth_ode(k=1.0):
        return SmoothODE(is_k_design_var=True, k=k)

    ref_result, approx_jac = get_fd_and_exact_jacobians(
        create_smooth_ode, 1e-6, 1.0, max_disk_checkpoints=10, max_memory_checkpoints=10
    )

    assert ref_result.algorithm_has_converged
    jacobian_desvar = ref_result.jac_wrt_desvar
    assert jacobian_desvar.flatten() == pytest.approx(approx_jac, 1e-4)
