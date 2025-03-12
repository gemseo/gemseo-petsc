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
"""A Smooth ODE dy/dt=k.t.y²."""

from gemseo.algos.ode.ode_problem import ODEProblem
from gemseo.typing import RealArray
from numpy import atleast_1d
from numpy import linspace
from numpy import zeros


class SmoothODE(ODEProblem):
    """Smooth ODE dy/dt = k.t.y²."""

    __jac_wrt_state: RealArray
    """The Jacobian matrix with respect to the state."""

    __jac_wrt_desvar: RealArray
    """The Jacobian matrix with respect to the design variables."""

    __k: float
    """The parameter of the smooth ODE."""

    def __init__(
        self, initial_state: float = 1.0, k: float = 1.0, is_k_design_var: bool = False
    ) -> None:  # noqa: D107
        """Initialize state and ODE functions.

        Args:
            initial_state: The initial condition.
            k: A coefficient in the ODE.
            is_k_design_var: If True, k becomes a design variable
                otherwise, there is no dependency to k in the gradient.
        """
        self.__jac_wrt_state = zeros((1, 1))
        self.__k = k
        super().__init__(
            self.__compute_rhs_func,
            jac_function_wrt_state=self.__compute_jac_wrt_state,
            jac_function_wrt_desvar=self.__compute_jac_wrt_desvar
            if is_k_design_var
            else None,
            initial_state=atleast_1d(initial_state),
            times=linspace(0, 1.0, 100),
        )

        self.__jac_wrt_desvar = zeros((1, 1))

    def __compute_rhs_func(self, time: float, state: RealArray) -> RealArray:  # noqa:U100
        """The function defining the right-hand side of the ODE.

        Args:
            time: The current time.
            state: The state at current time.

        Returns:
            The time derivative of the state.
        """
        st_dot = state.copy()
        st_dot[0] = self.__k * time * state[0] ** 2
        return st_dot

    def __compute_jac_wrt_state(
        self,
        time: float,
        state: RealArray,
    ) -> RealArray:  # noqa:U100
        """The Jacobian of the right-hand side of the ODE with respect to the state.

        Args:
            time: The current time.
            state: The state at current time.

        Returns:
            The Jacobian with respect to the state.
        """
        self.__jac_wrt_state[0, 0] = self.__k * 2 * time * state[0]
        return self.__jac_wrt_state

    def __compute_jac_wrt_desvar(
        self,
        time: float,
        state: RealArray,
    ) -> RealArray:  # noqa:U100
        """The Jacobian of the right-hand side of the ODE with respect to ``k``.

        Args:
            time: The current time.
            state: The state at current time.

        Returns:
            The Jacobian with respect to the design variable ``k``.
        """
        self.__jac_wrt_desvar[0, 0] = time * state[0] ** 2
        return self.__jac_wrt_desvar
