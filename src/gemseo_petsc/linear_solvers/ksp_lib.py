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

"""A PETSC KSP linear solvers library wrapper"""

import logging
from typing import Any, Dict, List, Tuple, Union

from numpy import ndarray, arange, array, zeros_like
from scipy.sparse.base import issparse
from scipy.sparse.linalg import LinearOperator, bicg, bicgstab, gmres, lgmres, qmr
from scipy.sparse import csr_matrix, find
from gemseo.algos.linear_solvers.linear_solver_lib import LinearSolverLib
import petsc4py
import sys

# Must be done before from petsc4py import PETSc
petsc4py.init(sys.argv)
from petsc4py import PETSc
from matplotlib import pylab

LOGGER = logging.getLogger(__name__)

class PetscKSPAlgos(LinearSolverLib):
    """Interface to PETSC KSP
    
    For further information, please read 
    https://petsc4py.readthedocs.io/en/stable/manual/ksp/
    
    https://petsc.org/release/docs/manualpages/KSP/KSP.html#KSP
    
    """
    OPTIONS_MAP = { }

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
        solver_type='gmres',
        max_iter=100000,  # type: int
        tol=1e-5,
        atol=1e-50,
        dtol = 1e5,
        preconditioner_type='ilu',
        view_config=False,
        options_hook_func=None,
        set_from_options=False,
        monitor_residuals=False,
    ):  # type: (...) -> Dict
        """Checks the options and sets the default values.

        Args:
            max_iter: Maximum number of iterations.
            solver_type: The KSP solver type. 
                See https://petsc.org/release/docs/manualpages/KSP/KSPType.html#KSPType 
            tol: The relative convergence tolerance, relative decrease in the (possibly preconditioned) residual norm.
            abstol: The absolute convergence tolerance absolute size of the (possibly preconditioned) residual norm.
            dtol: The divergence tolerance, amount (possibly preconditioned) residual norm can increase.
            preconditioner_type: The name of the precondtioner, 
                see https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.PC.Type-class.html
            view_config: if True, calls ksp.view() to view the configuration of the solver before run
            options_hook_func: A callback functions that is called with (ksp problem, options dict) as arguments
                before calling ksp.solve(), use to allow the user an advanced configuration that is not
                supported by the current wrapper.
            set_from_options: if True, the options are set from sys.argv, a classical Petsc configuration mode.
            monitor_residuals: if True, stores the residuals during convergence in self.problem.
                WARNING: as said in Petsc documentation, "the routine is slow and should be used only for
                 testing or convergence studies, not for timing."
        Returns:
            The options dict
        """
        return self._process_options(
            max_iter=max_iter,
            solver_type=solver_type,
            monitor_residuals=monitor_residuals,
            tol=tol,
            atol=atol,
            dtol=dtol,
            preconditioner_type=preconditioner_type,
            view_config=view_config,
            set_from_options=set_from_options,
            options_hook_func=options_hook_func
        )
        
    def monitor(self, ksp, its, rnorm):
        self.problem.residuals_history.append(rnorm)

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
        ksp.setTolerances(options["tol"],options["atol"], options["dtol"], options["max_iter"])
        ksp.setConvergenceHistory()
        a_mat = ndarray2petsc(self.problem.lhs)
        ksp.setOperators(a_mat)
        prec_type=options.get("preconditioner_type")
        if prec_type is not None:
            pc = ksp.getPC()
            pc.setType(prec_type)
            pc.setUp()
            
        # Allow for solver choice to be set from command line with -ksp_type <solver>.
        # Recommended option: -ksp_type preonly -pc_type lu
        if  options["set_from_options"]  :
            ksp.setFromOptions()
            
        options_hook_func=options.get('options_hook_func')
        if options_hook_func is not None:
            options_hook_func(ksp, options)
        
        self.problem.residuals_history=[]
        if options["monitor_residuals"]:
            LOGGER.warning("Petsc option monitor_residuals slows the process and"
                           " should be used only for testing or convergence studies.")
            ksp.setMonitor(self.monitor)

        b_mat = ndarray2petsc(self.problem.rhs)
        solution=b_mat.duplicate()
        if options["view_config"]:
            ksp.view()
        ksp.solve(b_mat, solution)
        self.problem.solution = solution.getArray()
        self.problem.convergence_info=ksp.reason
        return self.problem.solution


def ndarray2petsc(np_arr):
    n_dim=np_arr.ndim
    if n_dim>2:
        raise ValueError("Unsupported dimension {}!".format(n_dim))
    
    if issparse(np_arr):
        if not isinstance(np_arr,csr_matrix):
            np_arr=np_arr.tocsr()
        if n_dim==2 and np_arr.shape[1]>1:
            petsc_arr = PETSc.Mat().createAIJ(size=np_arr.shape, 
                                       csr=(np_arr.indptr, np_arr.indices, np_arr.data))
            petsc_arr.assemble()
            return petsc_arr
        else:
            petsc_arr = PETSc.Vec().createSeq(np_arr.shape[0])
            petsc_arr.setUp()
            inds, _, vals=find(np_arr)
            petsc_arr.setValues(inds,vals)
            petsc_arr.assemble()
            return petsc_arr
             
    # Update because of flatten() called in previous line
    n_dim=np_arr.ndim
    if n_dim == 1:
        a = array(np_arr, dtype=PETSc.ScalarType)
        petsc_arr = PETSc.Vec().createWithArray(a)
        petsc_arr.assemble()
        return petsc_arr
    elif n_dim == 2:
        petsc_arr = PETSc.Mat().createDense(np_arr.shape)
        a_shape=np_arr.shape
        petsc_arr.setUp()
        petsc_arr.setValues(arange(a_shape[0],dtype="int32"),arange(a_shape[1],dtype="int32"),np_arr)
        petsc_arr.assemble()
        return petsc_arr
    else:
        raise ValueError("Unsupported dimension {}!".format(n_dim))

# KSP example here
# https://fossies.org/linux/petsc/src/binding/petsc4py/demo/petsc-examples/ksp/ex2.py
