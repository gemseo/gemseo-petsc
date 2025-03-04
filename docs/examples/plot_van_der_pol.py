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
potentially depending from a set of design variables,
a time interval,
and a set of initial conditions for the state of the system.
"""

from gemseo.typing import RealArray
from numpy import array

# %%
# Let us consider the following IVP:
# .. math::
#
#     \frac{dy(t)}{dt} = k t y^2
# where :math:`t` denotes the time, :math:`y` is the state variable,
# and :math:`k` is a design parameter.
#
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
