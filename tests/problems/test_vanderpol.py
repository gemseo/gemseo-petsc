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
from gemseo.core.function.array_function import ArrayFunction
from numpy import array

from gemseo_petsc.problems.vanderpol import VanderPol


def ode_func_mu(mu):
    problem = VanderPol(mu)
    return problem.rhs_function(0.0, array([0.0, 0.0]))


def ode_jac_mu(mu):
    problem = VanderPol(mu)
    return problem.jac_function_wrt_desvar(0.0, array([0.0, 0.0]))


def test_ode_jac_desvars():
    func = ArrayFunction(ode_func_mu, "jac_desvars", jac=ode_jac_mu)
    # TODO(bump-gemseo): cannot transform: the type of func could not be inferred; if it is an instance of MDOFunction, use gemseo.util.derivative.check.function.FunctionJacobianChecker instead  # noqa: E501
    func.check_grad(array([0.5]), step=1e-7, error_max=1e-5)


def test_ode_jac_state():
    problem = VanderPol()
    func = ArrayFunction(
        lambda x: problem.rhs_function(0.0, x),
        "jac_state",
        jac=lambda x: problem.jac_function_wrt_state(0.0, x),
    )
    # TODO(bump-gemseo): cannot transform: the type of func could not be inferred; if it is an instance of MDOFunction, use gemseo.util.derivative.check.function.FunctionJacobianChecker instead  # noqa: E501
    func.check_grad(array([0.5, 0.5]), step=1e-7, error_max=1e-5)
