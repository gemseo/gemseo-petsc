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
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from numpy import array

from gemseo_petsc.problems.vanderpol import VanderPol


def ode_func_mu(mu):
    problem = VanderPol(mu)
    return problem.rhs_function(0.0, array([0.0, 0.0]))


def ode_jac_mu(mu):
    problem = VanderPol(mu)
    return problem.jac_function_wrt_desvar(0.0, array([0.0, 0.0]))


def test_ode_jac_desvars():
    func = MDOFunction(ode_func_mu, "jac_desvars", jac=ode_jac_mu)
    func.check_grad(array([0.5]), step=1e-7, error_max=1e-5)


def test_ode_jac_state():
    problem = VanderPol()
    func = MDOFunction(
        lambda x: problem.rhs_function(0.0, x),
        "jac_state",
        jac=lambda x: problem.jac_function_wrt_state(0.0, x),
    )
    func.check_grad(array([0.5, 0.5]), step=1e-7, error_max=1e-5)
