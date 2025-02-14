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
"""Van der Pol ODE."""

from gemseo.algos.ode.ode_problem import ODEProblem
from gemseo.typing import RealArray
from numpy import array
from numpy import linspace
from numpy import zeros


class VanderPol(ODEProblem):
    """Van der Pol ODE."""

    __jac_wrt_state: RealArray
    """The Jacobian matrix with respect to the state."""

    __jac_wrt_desvar: RealArray
    """The Jacobian matrix with respect to the design variables."""

    __mu: float
    """The stiffness parameter of the ODE."""

    def __init__(
        self, mu: float = 0.5, final_time: float = 2.0, n_pts_time_vector: int = 1000
    ) -> None:  # noqa: D107
        """Initialize state and ODE functions.

        Args:
            mu: The mu coefficient in the ODE
            final_time: The final time of the ODE resolution.
            n_pts_time_vector: The number of points in the time_vector.
        """
        self.__jac_wrt_state = zeros((2, 2))
        self.__jac_wrt_desvar = zeros((2, 1))
        self.__mu = mu
        super().__init__(
            self.__func,
            initial_state=array([2.0, -2.0 / 3.0]),
            times=linspace(0.0, final_time, n_pts_time_vector),
            jac_function_wrt_state=self.__compute_jac_wrt_state,
            jac_function_wrt_desvar=self.__compute_jac_wrt_desvar,
        )

    def __func(self, time: float, state: RealArray) -> RealArray:  # noqa:U100
        """The function defining the right-hand side of the ODE.

        Args:
            time: The current time.
            state: The state at current time.

        Returns:
            The time derivative of the state.
        """
        state_dot = state.copy()
        state_dot[0] = state[1]
        state_dot[1] = self.__mu * ((1.0 - state[0] * state[0]) * state[1] - state[0])
        return state_dot

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
        self.__jac_wrt_state[0, 0] = 0
        self.__jac_wrt_state[0, 1] = 1.0
        self.__jac_wrt_state[1, 0] = -self.__mu * (2.0 * state[1] * state[0] + 1.0)
        self.__jac_wrt_state[1, 1] = self.__mu * (1.0 - state[0] * state[0])
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
        self.__jac_wrt_desvar[0, 0] = 0
        self.__jac_wrt_desvar[1, 0] = (1.0 - state[0] * state[0]) * state[1] - state[0]
        return self.__jac_wrt_desvar
