# -*- coding: utf-8 -*-
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
from __future__ import division, unicode_literals

import logging
from typing import Any, Dict, List, Tuple, Union

from numpy import ndarray, arange
from scipy.sparse.base import issparse
from scipy.sparse.linalg import LinearOperator, bicg, bicgstab, gmres, lgmres, qmr

from gemseo.algos.linear_solvers.linear_solver_lib import LinearSolverLib
import petsc4py
import sys

petsc4py.init(sys.argv)
from petsc4py import PETSc
from matplotlib import pylab


LOGGER = logging.getLogger(__name__)


class PetscKSPAlgos(LinearSolverLib):
    OPTIONS_MAP = {
        "max_iter": "maxits"
    }


    def __init__(self):  # type: (...) -> None # noqa: D107
        super(PetscKSPAlgos, self).__init__()
        self.lib_dict = {
            "PETSC_GMRES": self.get_default_properties("GMRES"),
        }

    def _get_options(
        self,
        max_iter=100000,  # type: int
        options_cmd=None,
        print_residuals=False,
        solver_type="gmres",
        rtol=1e-5,
        atol=1e-50,
        dtol = 1e5,
        preconditioner_type='ilu'
    ):  # type: (...) -> Dict
        """Checks the options and sets the default values.

        Args:
            max_iter: Maximum number of iterations.
            rtol: the relative convergence tolerance, relative decrease in the (possibly preconditioned) residual norm.
            abstol: the absolute convergence tolerance absolute size of the (possibly preconditioned) residual norm.
            dtol: the divergence tolerance, amount (possibly preconditioned) residual norm can increase.
        Returns:
            The options dict
        """
        return self._process_options(
            max_iter=max_iter,
            options_cmd=options_cmd,
            print_residuals=print_residuals,
            solver_type=solver_type,
            rtol=rtol,
            atol=atol,
            dtol=dtol,
            preconditioner_type=preconditioner_type
        )

    def _run(
        self, **options  # type: Any
    ):  # type: (...) -> ndarray
        """Runs the algorithm.

        Args:
            **options: The options for the algorithm.

        Returns:
            The solution of the problem.
        """
        rhs = self.problem.rhs
        lhs = self.problem.lhs
        if issparse(rhs):
            rhs = self.problem.rhs.toarray()

        # Initialize ksp solver.
        ksp = PETSc.KSP().create()
        ksp.setType(options["solver_type"])
        ksp.setTolerances(options["rtol"],options["atol"], options["dtol"], options["maxit"])

        ksp.view()
        ksp.setConvergenceHistory()
        a_mat = ndarray2petsc(self.problem.lhs)
        ksp.setOperators(a_mat)

        precond=options.get("preconditioner_type")
        if precond is not None:
            ksp.setPCType(precond)
            
        # Allow for solver choice to be set from command line with -ksp_type <solver>.
        # Recommended option: -ksp_type preonly -pc_type lu
        options_cmd=options.get("options_cmd")
        if options_cmd is not None:
            ksp.setFromOptions(options_cmd)
        print("Solving with:", ksp.getType())

        b_mat = ndarray2petsc(self.problem.rhs)
        solution = PETSc.Vec().createSeq(len(b_mat))
        sol = ksp.solve(b_mat, solution)
        self.problem.solution = solution.getArray()

        reason = ksp.getConvergedReason()
        print("converged reason", reason)
        return self.problem.solution


def ndarray2petsc(np_arr):
    np_shape = np_arr.shape
    if len(np_shape) == 1:
        petsc_arr = PETSc.Vec().create()
        petsc_arr.setSize(np_shape[0])
        petsc_arr.setType("aij")
        petsc_arr.setUp()
        petsc_arr.setValues(len(np_arr), np_arr)
        petsc_arr.assemble()
        return petsc_arr
    if len(np_shape) == 2:
        petsc_arr = PETSc.Mat().create()
        petsc_arr.setSizes(np_shape)
        petsc_arr.setType("aij")
        petsc_arr.setUp()

        petsc_arr.setValues(arange(np_shape[0]), arange(np_shape[1]), np_arr)
        petsc_arr.assemble()
        return petsc_arr
    raise ValueError("Unsupported dimension !")



# -ksp_type
# gmres
# -pc_type
# ilu
# -pc_factor_levels
# 10
# -pc_factor_fill
# 10
# -pc_factor_mat_solver_package
# spai
# -sub_pc_type
# ilu - sub_pc_factor_levels
# 0
# > -sub_pc_factor_fill
# 1.00 - sub_pc_factor_shift_nonzero
# > -sub_pc_factor_mat_ordering_type
# rcm -
# KSP = "-ksp_type gmres -ksp_max_it 800 -ksp_gmres_restart 800
# > -ksp_rtol
# 1.0e-12 - ksp_left_pc - ksp_gmres_modifiedgramschmidt
# > -ksp_gmres_cgs_refinement_type
# REFINE_NEVER - ksp_singmonitor
# > -ksp_compute_eigenvalues
# ": I'm forcing 800 krylovs without restart
# > to
# get
# the
# condition
# number
# of
# the
# pre - conditioned
# system( if I
#            > understand
# correctly
# the
# man
# page
# of - ksp_singmonitor)
# > *PC = "-pc_type asm -pc_asm_overlap 2": I
# 'm planning to run with more
# > than
# 1
# partition, but
# for the time being, only one partition is