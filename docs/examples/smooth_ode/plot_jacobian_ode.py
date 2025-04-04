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
"""# Compute the Jacobian of the solution of an ODE

Let us consider an Initial Value Problem (IVP),
consisting of an Ordinary Differential Equation (ODE),
potentially depending on a set of design variables,
a time interval,
and a set of initial conditions for the state of the system.

We are interested in computing the sensitivity of the solution
with respect to the initial conditions and eventual design variables
present in the expression of the ODE.
"""

from gemseo.algos.ode.factory import ODESolverLibraryFactory
from gemseo.algos.ode.ode_problem import ODEProblem
from gemseo.typing import RealArray
from numpy import atleast_1d
from numpy import linspace
from numpy import zeros

# %%
# Let us consider the same IVP presented in the example about the
# [Solution of an Initial Value Problem](../plot_smooth_ode):
#
# $$
#     \frac{dy(t)}{dt} = k t y^2
# $$
#
# where $t$ denotes the time, $y$ is the state variable,
# and $k$ is a design parameter.

init_state = 1.0
final_time = 0.5
times = linspace(0.0, final_time, 51)
k = 1.0

# %%
# The function defining the dynamics of the ODE is the following:


def rhs_func(t: float, y: RealArray, k: float) -> RealArray:
    st_dot = y.copy()
    st_dot[0] = k * t * y[0] ** 2
    return st_dot


# %%
# We provide the Jacobian of the dynamics with respect to the state
# and to the design variables.


def compute_jac_wrt_state(
    t: float,
    y: RealArray,
    k: float,
) -> RealArray:
    return k * 2 * t * y[0]


def compute_jac_wrt_desvar(
    t: float,
    y: RealArray,
    k: float,
) -> RealArray:
    return t * y[0] ** 2


# %%
# These functions are assembled into an
# [ODEProblem][gemseo.algos.ode.ode_problem.ODEProblem].


class SmoothODEProblem(ODEProblem):
    def __init__(self) -> None:  # noqa: D107
        self.__jac_wrt_state = zeros((1, 1))
        self.__k = k
        super().__init__(
            self.__compute_rhs_func,
            jac_function_wrt_state=self.__compute_jac_wrt_state,
            jac_function_wrt_desvar=self.__compute_jac_wrt_desvar,
            initial_state=atleast_1d(init_state),
            times=times,
        )

        self.__jac_wrt_desvar = zeros((1, 1))

    def __compute_rhs_func(self, time, state):
        return rhs_func(time, state, self.__k)

    def __compute_jac_wrt_state(self, time, state):
        self.__jac_wrt_state[0, 0] = compute_jac_wrt_state(time, state, self.__k)
        return self.__jac_wrt_state

    def __compute_jac_wrt_desvar(self, time, state):
        self.__jac_wrt_desvar[0, 0] = compute_jac_wrt_desvar(time, state, self.__k)
        return self.__jac_wrt_desvar


problem = SmoothODEProblem()

# %%
# By setting the parameter `compute_adjoint` to `True`
# in `ODESolverLibraryFactory().execute()`,
# the PETSc solver computes the sensitivity of the solution of the ODE with respect
# to the initial conditions of the IVP and to the design variables
# by solving a suitable adjoint problem.

# %%
# In order to solve the adjoint problem backwards in time,
# PETSc stores some intermediary values of the state under a checkpoint system.
# The checkpoints can be stored either on disk or on RAM.
# The storage of checkpoints on the RAM can be enabled by setting the optional parameter
# `use_memory_checkpoints` to `True`.
# The maximal number of checkpoints to be stored on disk or on the RAM
# can be controlled by the optional parameters `max_disk_checkpoints`
# and `max_memory_checkpoints`.

ODESolverLibraryFactory().execute(
    problem,
    algo_name="PETSC_ODE_RK",
    time_step=1e-2,
    maximum_steps=1000,
    rtol=1e-3,
    use_jacobian=True,
    compute_adjoint=True,
    use_memory_checkpoints=True,
)

# %%
# The Jacobian of the solution of the IVP with respect to the initial conditions
# and the design variables can be found respectively in the attributes
# `jac_wrt_initial_state` and `jac_wrt_desvar` of `problem.result`.


# %%
# The Jacobians computed by PETSc can be compared with their analytical counterparts.
# By knowing that the exact solution of the IVP is
# $y(t) = \frac{ 2 y_0}{2 - k t^2 y_0}$, the Jacobians of the solution with respect to
# $y_0$ and $k$ are:
#
# $$
# J_{y_0} = \left[\frac{4}{(2 - k t^2 y_0)^2}\right]
# $$
#
# $$
# J_{k} = \left[\frac{2 y_0^2 t^2}{(2 - k t^2 y_0)^2}\right]
# $$
#

analytical_jac_initial_state = 4.0 / (2 - k * final_time**2 * init_state) ** 2
analytical_jac_desvar = (
    2 * init_state**2 * final_time**2 / (2 - k * final_time**2 * init_state) ** 2
)

error_jac_initial_state = abs(
    analytical_jac_initial_state - problem.result.jac_wrt_initial_state[0, 0]
)

error_jac_desvar = abs(analytical_jac_desvar - problem.result.jac_wrt_desvar[0, 0])

print(
    f"Jacobian with respect to the initial state: \n"
    f"     absolute error = {error_jac_initial_state}, "
    f"     relative error = {error_jac_initial_state / analytical_jac_initial_state}"
)

print(
    f"Jacobian with respect to the design variable: \n"
    f"     absolute error = {error_jac_desvar}, "
    f"     relative error = {error_jac_desvar / analytical_jac_desvar}"
)
