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
"""# Solve an Initial Value Problem

Let us consider an Initial Value Problem (IVP),
consisting of an Ordinary Differential Equation (ODE),
potentially depending on a set of design variables,
a time interval,
and a set of initial conditions for the state of the system.
"""

from gemseo.algos.ode.factory import ODESolverLibraryFactory
from gemseo.algos.ode.ode_problem import ODEProblem
from gemseo.typing import RealArray
from matplotlib import pyplot as plt
from numpy import array
from numpy import atleast_1d
from numpy import linspace
from numpy import zeros
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

init_state = 1.0
times = linspace(0.0, 0.5, 51)
k = 1.0

# %%
# The function defining the dynamics of the ODE is the following.


def rhs_func(t: float, y: RealArray, k: float) -> RealArray:  # noqa:U100
    st_dot = y.copy()
    st_dot[0] = k * t * y[0] ** 2
    return st_dot


# %%
# We define the Jacobian of the dynamics with respect to the state
# and to the design variables.


def compute_jac_wrt_state(
    t: float,
    y: RealArray,
    k: float,
) -> RealArray:  # noqa:U100
    jac_wrt_state = k * 2 * t * y[0]
    return array([[jac_wrt_state]])


def compute_jac_wrt_desvar(
    t: float,
    y: RealArray,
    k: float,
) -> RealArray:  # noqa:U100
    jac_wrt_desvar = t * y[0] ** 2
    return array([[jac_wrt_desvar]])


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

    def __compute_rhs_func(self, time, state):  # noqa:U100
        st_dot = state.copy()
        st_dot[0] = self.__k * time * state[0] ** 2
        return st_dot

    def __compute_jac_wrt_state(self, time, state):  # noqa:U100
        self.__jac_wrt_state[0, 0] = self.__k * 2 * time * state[0]
        return self.__jac_wrt_state

    def __compute_jac_wrt_desvar(self, time, state):  # noqa:U100
        self.__jac_wrt_desvar[0, 0] = time * state[0] ** 2
        return self.__jac_wrt_desvar


problem = SmoothODEProblem()

# %%
# The IVP can be solved using the algorithms provided by `gemseo-petsc`.
# As an example, here the solution to the IVP is found using the backwards
# Euler algorithm.
#
# The list of all available algorithms is available at (?????)


ODESolverLibraryFactory().execute(
    problem,
    algo_name="PETSC_ODE_BEULER",
    time_step=1e-3,
    maximum_steps=1000,
    atol=1e-4,
    use_jacobian=True,
)

# %%
# The numerical solution can be compared with the analytical solution of the ODE.
# $$
#     y(t) = \frac{y_0}{1 - k t y_0}.
# $$

analytical_sol = init_state / (1.0 - k * times * init_state)
error = abs(analytical_sol - problem.result.state_trajectories[0])

plt.semilogy(times, error)
plt.title("Integration error")
plt.show()

# %%
# The `ODEProblem` describing the above ODE is available
# using the shortcut class `SmoothODE`

shortcut_problem = SmoothODE(
    initial_state=init_state, times=times, k=k, is_k_design_var=False
)

ODESolverLibraryFactory().execute(
    shortcut_problem,
    algo_name="PETSC_ODE_BEULER",
    time_step=1e-3,
    maximum_steps=1000,
    atol=1e-4,
    use_jacobian=True,
)
