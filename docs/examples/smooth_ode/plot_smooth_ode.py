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

# %%
# Let us consider the following IVP:
#
# $$
#     \frac{dy(t)}{dt} = k t y^2
# $$
#
# where $t$ denotes the time, $y$ is the state variable,
# and $k$ is a design parameter.

# %%
# We define an initial state and a time interval for the IVP,
# as well as a design parameter $k$.

init_state = 1.0
times = linspace(0.0, 0.5, 51)
k = 1.0

# %%
# The function defining the dynamics of the ODE is the following:


def rhs_func(t: float, y: RealArray, k: float) -> RealArray:
    st_dot = y.copy()
    st_dot[0] = k * t * y[0] ** 2
    return st_dot


# %%
# We define the Jacobian of the dynamics with respect to the state.


def compute_jac_wrt_state(
    t: float,
    y: RealArray,
    k: float,
) -> RealArray:
    jac_wrt_state = k * 2 * t * y[0]
    return array([[jac_wrt_state]])


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
            initial_state=atleast_1d(init_state),
            times=times,
        )

        self.__jac_wrt_desvar = zeros((1, 1))

    def __compute_rhs_func(self, time, state):
        return rhs_func(time, state, self.__k)

    def __compute_jac_wrt_state(self, time, state):
        self.__jac_wrt_state[0, 0] = compute_jac_wrt_state(time, state, self.__k)
        return self.__jac_wrt_state


problem = SmoothODEProblem()

# %%
# The IVP can be solved using the algorithms provided by `gemseo-petsc`.
# As an example, here the solution to the IVP is found using
# the Runge-Kutta algorithm.


ODESolverLibraryFactory().execute(
    problem,
    algo_name="PETSC_ODE_RK",
    time_step=1e-2,
    maximum_steps=1000,
    rtol=1e-3,
    use_jacobian=True,
)

# %%
# The numerical solution can be compared with the analytical solution of the ODE.
#
# $$
#     y(t) = \frac{ 2 y_0}{2 - k t^2 y_0}.
# $$
#

analytical_sol = 2.0 * init_state / (2.0 - k * times * times * init_state)
error = abs(analytical_sol - problem.result.state_trajectories[0])

plt.semilogy(times, error)
plt.title("Integration error")
plt.show()
