# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""# Compute the adjoint of an ODE

Let us consider an Initial Value Problem (IVP),
consisting of an Ordinary Differential Equation (ODE),
potentially depending on a set of design variables,
a time interval,
and a set of initial conditions for the state of the system.
"""

from gemseo.algos.ode.factory import ODESolverLibraryFactory
from numpy import linspace
from src.gemseo_petsc.problems.smooth_ode import SmoothODE

# %%
# Let us consider the following IVP:
# $$
#     \frac{dy(t)}{dt} = k t y^2
# $$
# where :math:`t` denotes the time, :math:`y` is the state variable,
# and :math:`k` is a design parameter.

# %%
# We define an initial state and a time interval for the IVP,
# as well as a design parameter :math:`k`.
# Then, we  define a SmoothODE problem.

init_state = 1.0
times = linspace(0.0, 0.5, 51)
k = 1.0

problem = SmoothODE(initial_state=init_state, times=times, k=k, is_k_design_var=False)

# %%
# In order to compute the sensitivity of the solution of the IVP at final time
# with respect to the initial conditions and the design parameter $k$,
# it is necessary to compute and solve the adjoint equation.
#
# The adjoint equation of an IVP is an ODE provided with a final condition,
# thus, it has to be solved backwards in time.

# The PETSc provides a time-stepping routine
# in order to save the intermediary values of the state
# and ease the solution of the adjoint equation.

ODESolverLibraryFactory().execute(
    problem,
    algo_name="PETSC_ODE_RK",
    time_step=0.001,
    maximum_steps=10000,
    compute_adjoint=True,
    use_jacobian=True,
    atol=1e-6,
    rtol=1e-6,
)

# %%
# The Jacobians of the solution at the final time with respect to the initial
# conditions and the design parameter $k$ are stored respectively in the attributes
# `jac_wrt_initial_state` and `jac_wrt_desvar` of `problem.result`.
#
# These Jacobians can be compared with the approximate Jacobians
# obtained by finite differences.
# In order to ease the definition and solution of the IVPs, we introduce the following
# function to create and solve instances of `SmoothODE` problems.


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


# %%
# We can start by evaluating the Jacobian with respect to the initial conditions,
# and compare it to the approximation by finite differences.
epsilon = 1e-6

problem_pert_plus = SmoothODE(initial_state=init_state + epsilon, times=times, k=k)
solve_ode(problem_pert_plus)
problem_pert_minus = SmoothODE(initial_state=init_state - epsilon, times=times, k=k)
solve_ode(problem_pert_minus)

approx_jac = (
    problem_pert_plus.result.state_trajectories[:, -1]
    - problem_pert_minus.result.state_trajectories[:, -1]
) / (2 * epsilon)
