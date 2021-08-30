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

from numpy import ndarray, arange, array, zeros_like
from scipy.sparse.base import issparse
from scipy.sparse.linalg import LinearOperator, bicg, bicgstab, gmres, lgmres, qmr

from gemseo.algos.linear_solvers.linear_solver_lib import LinearSolverLib
import petsc4py
import sys

# Must be done before from petsc4py import PETSc
petsc4py.init()
from petsc4py import PETSc
import petsc4py
from matplotlib import pylab


LOGGER = logging.getLogger(__name__)
comm = PETSc.COMM_WORLD
class PetscKSPAlgos(LinearSolverLib):
    OPTIONS_MAP = {
    }


    def __init__(self):  # type: (...) -> None # noqa: D107
        super(PetscKSPAlgos, self).__init__()
        self.lib_dict = {
            "PETSC_KSP": {
            self.RHS_MUST_BE_POSITIVE_DEFINITE: False,
            self.RHS_MUST_BE_SYMMETRIC: False,
            self.RHS_CAN_BE_LINEAR_OPERATOR: True,
            self.INTERNAL_NAME: "PETSC",
        }
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
        preconditioner_type='ilu',
        view_config=False
    ):  # type: (...) -> Dict
        """Checks the options and sets the default values.

        Args:
            max_iter: Maximum number of iterations.
            rtol: the relative convergence tolerance, relative decrease in the (possibly preconditioned) residual norm.
            abstol: the absolute convergence tolerance absolute size of the (possibly preconditioned) residual norm.
            dtol: the divergence tolerance, amount (possibly preconditioned) residual norm can increase.
            view_config: if True, calls ksp.view() to view the configuration of the solver before run
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
            preconditioner_type=preconditioner_type,
            view_config=view_config
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
        # Creates the options database
        options_cmd=options.get("options_cmd")
        if options_cmd is not None:
            petsc4py.init(options_cmd)
        else:
            petsc4py.init()
        ksp = PETSc.KSP().create()
        ksp.setType(options["solver_type"])
        ksp.setTolerances(options["rtol"],options["atol"], options["dtol"], options["max_iter"])
        ksp.setConvergenceHistory()
        a_mat = ndarray2petsc(self.problem.lhs)
        ksp.setOperators(a_mat)
        prec_type=options.get("preconditioner_type")
        if prec_type is not None:
            pc = ksp.getPC()
            pc.setType(prec_type)
            
            
        # Allow for solver choice to be set from command line with -ksp_type <solver>.
        # Recommended option: -ksp_type preonly -pc_type lu
        if  options_cmd is not None:
            ksp.setFromOptions()
        print("Solving with:", ksp.getType())

        b_mat = ndarray2petsc(self.problem.rhs)
        solution=b_mat.duplicate()
#         solution = PETSc.Vec(comm=comm).createSeq(len(b_mat))
#         solution.setUp()
#         solution.assemble()
        if options["view_config"]:
            ksp.view()
        ksp.solve(b_mat, solution)
        self.problem.solution = solution.getArray()

        reason = ksp.getConvergedReason()
        print("converged reason", reason)
        return self.problem.solution


def ndarray2petsc(np_arr):
    n_dim=np_arr.ndim
    if n_dim == 1:
        a = array(np_arr, dtype=PETSc.ScalarType)
        petsc_arr = PETSc.Vec().createWithArray(a)
        petsc_arr.assemble()
        return petsc_arr
    if n_dim == 2:
        #a = array(np_arr, dtype=PETSc.ScalarType)
        petsc_arr = PETSc.Mat().createDense(np_arr.shape)
        a_shape=np_arr.shape
        petsc_arr.setUp()
        petsc_arr.setValues(arange(a_shape[0],dtype="int32"),arange(a_shape[1],dtype="int32"),np_arr)
        petsc_arr.assemble()
        return petsc_arr
    raise ValueError("Unsupported dimension !")

# KSP example here
# https://fossies.org/linux/petsc/src/binding/petsc4py/demo/petsc-examples/ksp/ex2.py

#  dir(ksp()'ConvergedReason', 'NormType', 'appctx', 'atol', 'buildResidual', 'buildSolution', 'callConvergenceTest', 
#       'cancelMonitor', 'classid', 'comm', 'compose', 'computeEigenvalues', 'computeExtremeSingularValues', 
#       'converged', 'create', 'createPython', 'decRef', 'destroy', 'diverged', 'divtol', 'dm', 'fortran', 
#       'getAppCtx', 'getAttr', 'getClassId', 'getClassName', 'getComm', 'getComputeEigenvalues', 'getComputeSingularValues', 
#       'getConvergedReason', 'getConvergenceHistory', 'getConvergenceTest', 'getDM', 'getDict', 'getInitialGuessKnoll', 
#       'getInitialGuessNonzero', 'getIterationNumber', 'getMonitor', 'getName', 'getNormType', 'getOperators', 
#       'getOptionsPrefix', 'getPC', 'getPCSide', 'getPythonContext', 'getRefCount', 'getResidualNorm', 'getRhs', 
#       'getSolution', 'getTabLevel', 'getTolerances', 'getType', 'getWorkVecs', 'guess_knoll', 'guess_nonzero',
#        'handle', 'history', 'incRef', 'incrementTabLevel', 'iterating', 'its', 'klass', 'logConvergenceHistory',
#         'mat_op', 'mat_pc', 'max_it', 'monitor', 'name', 'norm', 'norm_type', 'pc', 'pc_side', 'prefix', 'query', 
#         'reason', 'refcount', 'reset', 'rtol', 'setAppCtx', 'setAttr', 'setComputeEigenvalues', 'setComputeOperators', 
#         'setComputeRHS', 'setComputeSingularValues', 'setConvergedReason', 'setConvergenceHistory', 'setConvergenceTest', 
#         'setDM', 'setDMActive', 'setFromOptions', 'setGMRESRestart', 'setInitialGuessKnoll', 'setInitialGuessNonzero', 
#         'setIterationNumber', 'setMonitor', 'setName', 'setNormType', 'setOperators', 'setOptionsPrefix', 'setPC', 'setPCSide', 
#         'setPythonContext', 'setPythonType', 'setResidualNorm', 'setTabLevel', 'setTolerances', 'setType', 
#         'setUp', 'setUpOnBlocks', 'setUseFischerGuess', 'solve', 'solveTranspose', 'stateIncrease', 'type', 
#         'vec_rhs', 'vec_sol', 'view', 'viewFromOptions


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
