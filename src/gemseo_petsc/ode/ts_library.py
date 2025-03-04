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
#        :author: Francois Gallard
#        :author: Giulio Gargantini
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A wrapper for the PETSc ODE solvers.

ODE stands for ordinary differential equation.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from gemseo.algos.ode.base_ode_solver_library import BaseODESolverLibrary
from gemseo.algos.ode.base_ode_solver_library import ODESolverDescription
from gemseo.algos.ode.base_ode_solver_settings import BaseODESolverSettings
from numpy import arange
from numpy import array
from numpy import extract
from numpy import interp
from numpy import vstack
from petsc4py import init

from gemseo_petsc.ode.settings.petsc_ts_settings import ODESolverType
from gemseo_petsc.ode.settings.petsc_ts_settings import PetscTSSettings
from gemseo_petsc.utils.conversion import convert_ndarray_to_mat_or_vec

init(sys.argv)
from petsc4py import PETSc  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.algos.ode.ode_problem import ODEProblem
    from gemseo.algos.ode.ode_result import ODEResult
    from gemseo.typing import RealArray

    FINAL_T_OPTION = PETSc.TS.TSExactFinalTimeOption

LOGGER = logging.getLogger(__name__)


@dataclass
class PetscTSAlgorithmDescription(ODESolverDescription):
    """The description of the PETSc TS ODE library."""

    Settings: type[BaseODESolverSettings] = PetscTSSettings
    """The settings validation model."""


class PetscOdeAlgo(BaseODESolverLibrary):
    """Interface to PETSC TS.

    For further information, please read https://petsc.org/release/docs/manual/ts/
    For a list of available methods, see
    https://petsc.org/release/overview/integrator_table/#integrator-table

    For the unsteady adjoint part:
    https://petsc4py.readthedocs.io/en/stable/manual/sensitivity_analysis/


    About checkpoints:
    https://petsc4py.readthedocs.io/en/stable/manual/sensitivity_analysis/#checkpointing
    """

    PETSC_ODE_PREFIX = "PETSC_ODE_"

    _TERMINATION_MESSAGES: Final[Mapping[int, str]] = {
        PETSc.TS.ConvergedReason.CONVERGED_EVENT: r"User requested termination"
        r" on event detection.",
        PETSc.TS.ConvergedReason.CONVERGED_ITERATING: r"This only occurs if "
        r"TSGetConvergedReason() is called "
        r"during the TSSolve().",
        PETSc.TS.ConvergedReason.CONVERGED_ITS: r"The maximum number of iterations "
        r"(time-steps) was reached"
        r" prior to the final time.",
        PETSc.TS.ConvergedReason.CONVERGED_TIME: r"The final time was reached.",
        PETSc.TS.ConvergedReason.CONVERGED_USER: r"User requested termination.",
        PETSc.TS.ConvergedReason.DIVERGED_NONLINEAR_SOLVE: r"Too many nonlinear solve "
        r"failures have occurred.",
        PETSc.TS.ConvergedReason.DIVERGED_STEP_REJECTED: r"Too many steps "
        r"were rejected.",
    }
    """The return messages of TS methods."""

    adjoint_wrt_state: list[PETSc.Vec]
    """The adjoint with respect to the state."""

    adjoint_wrt_desvar: list[PETSc.Vec]
    """The adjoint with respect to the design parameters."""

    _adjoints: list[PETSc.Vec]
    """Adjoint vectors to state and parameters."""

    _hist_iterations: list[int]
    """The indices of the performed iterations."""

    _problem: ODEProblem
    """The Initial Value Problem."""

    _hist_state: list[RealArray]
    """The history of the state through the performed iterations."""

    _hist_times: list[float]
    """The times of the performed iterations."""

    __n_events: int
    """The number of event functions considered in the Initial Value Problem."""

    __n_calls_jac: int
    """The number of calls of the Jacobian."""

    __rhs_indices: Sequence[int]
    """The indices corresponding to the state entries."""

    ALGORITHM_INFOS: ClassVar[dict[str, PetscTSAlgorithmDescription]] = {
        solver_name: PetscTSAlgorithmDescription(
            algorithm_name=solver_name,
            internal_algorithm_name=solver_name.lower(),
            website="https://petsc.org/release/manualpages/TS/",
            Settings=type(
                f"{solver_name}_Settings",
                (PetscTSSettings,),
                {"_TARGET_CLASS_NAME": f"{solver_name}"},
            ),
        )
        for solver_name in ODESolverType
    }

    def __init__(self, algo_name: str) -> None:  # noqa: D107
        """
        Args:
            algo_name: The name of the algorithm solving the ODE.
        """  # noqa: D205, D212, D415
        super().__init__(algo_name=algo_name)
        self.__rhs_indices = []
        # adjoint_wrt_state is adjoint for cost gradient wrt y at t0 (initial condition)
        # FG: this seems to be df(t=O)/d(y(t=0))
        # adjoint_wrt_desvar is adjoint for cost gradient wrt scalar parameter mu
        # FG: this seems to be df(t=O)/dx
        # Initialize them with cost function derivatives wrt u and mu (resp.) at t=tF
        self.adjoint_wrt_state = []
        self.adjoint_wrt_desvar = []
        # Save ODE history
        self._hist_iterations, self._hist_times, self._hist_state = [], [], []
        self.__n_calls_jac = 0

    def __monitor(
        self,
        ts: PETSc.TS,
        i: int,
        time: float,
        state: PETSc.Vec,
    ) -> None:
        """The monitoring function.

        Args:
            i: The current iteration.
            time: The current time.
            state: The current state.
        """
        self._hist_iterations.append(i)
        self._hist_times.append(time)
        self._hist_state.append(state.getArray().copy())

    def _rhs_function(
        self,
        ts: PETSc.TS,
        time: PETSc.Mat | PETSc.Vec,
        state: PETSc.Mat | PETSc.Vec,
        state_dot: PETSc.Mat | PETSc.Vec,
    ) -> None:
        """Evaluate the RHS of the problem, function to be passed to PETSc.

        Args:
            ts: The TS object.
            time: The time to evaluate the problem.
            state: The state of the problem.
            state_dot: The derivative of the state.
        """
        state_converted = state.getArray()
        val = self._problem.rhs_function(time, state_converted)
        state_dot.setValues(self.__rhs_indices, val)
        state_dot.assemble()

    def _jac_function_wrt_state(
        self,
        ts: PETSc.TS,
        time: PETSc.Mat | PETSc.Vec,
        state: PETSc.Mat | PETSc.Vec,
        jacobian_u: PETSc.Mat,
        preconditioner: PETSc.Mat,
    ) -> bool:
        """Jacobian of the problem, to be passed to PETSc.

        Args:
            ts: The TS object.
            time: The time to evaluate the problem.
            state: The state of the problem.
            jacobian_u: The Jacobian of the RHS function with respect to the state.
            preconditioner: The preconditioner for the Jacobian matrix.

        Returns:
            The successfulness of the operation.
        """
        state_converted = state.getArray()
        numpy_jac = self._problem.jac_function_wrt_state(time, state_converted)
        np_jac_shape = numpy_jac.shape

        if (
            len(np_jac_shape) < 2
            or np_jac_shape[0] != self._problem.initial_state.shape[0]
            or np_jac_shape[1] != self._problem.initial_state.shape[0]
        ):
            expected_size = self._problem.initial_state.shape[0]
            msg = f"""The Jacobian should be a square matrix with shape
            {expected_size} by {expected_size}. The provided Jacobian has shape
            {np_jac_shape} instead."""
            raise ValueError(msg)
        jacobian_u.setValues(self.__rhs_indices, self.__rhs_indices, numpy_jac)
        jacobian_u.assemble()

        self.__n_calls_jac += 1

        # TODO: Add the possibility for a preconditioner.
        return True

    def _jac_function_wrt_desvar(
        self,
        ts: PETSc.TS,
        time: PETSc.Mat | PETSc.Vec,
        state: PETSc.Mat | PETSc.Vec,
        jacobian_p: PETSc.Mat,
    ) -> bool:
        """Jacobian of the problem with respect to the design variables.

        Args:
            ts: The TS object.
            time: The time to evaluate the problem.
            state: The state of the problem.
            jacobian_p: The output for the Jacobian.

        Returns:
            The success.
        """
        state_converted = state.getArray()
        numpy_jac = self._problem.jac_function_wrt_desvar(time, state_converted)
        jacobian_p.setValues(self.__rhs_indices, self.__desvar_indices, numpy_jac)
        jacobian_p.assemble()
        return True

    def __indicator_events(
        self,
        ts: PETSc.TS,
        time: PETSc.Mat | PETSc.Vec,
        state: PETSc.Mat | PETSc.Vec,
        fvalue: list[float],
    ) -> None:
        """Evaluate the termination events.

        Args:
            ts: The TS object.
            time: The time to evaluate the problem.
            state: The state of the problem.
            fvalue: The output for the values of the event functions.
        """
        state_converted = state.getArray()
        for i in range(self.__n_events):
            fvalue[i] = self._problem.event_functions[i](time, state_converted)

    def __post_event(
        self,
        ts: PETSc.TS,
        events: RealArray,
        time: PETSc.Mat | PETSc.Vec,
        state: PETSc.Mat | PETSc.Vec,
        forward: PETSc.Bool,
    ) -> None:
        """The actions to perform as an event is triggered.

        Args:
            ts: The TS object.
            events: The current values of the event functions.
            time: The time to evaluate the problem.
            state: The state of the problem.
            forward: flag to indicate whether it is a forward or adjoint solve.
        """
        self._problem.result.terminal_event_index = events[0]

    def _run(self, problem: ODEProblem, **settings: Any) -> ODEResult:
        """Run the quadrature algorithm.

        Args:
            problem: The Initial Value Problem to be solved.
            settings: The settings of the algorithm.

        Returns:
            The solution of the Initial Value Problem.
        """
        settings_ = self._filter_settings(
            settings, model_to_exclude=BaseODESolverSettings
        )

        self._problem = problem
        self.__n_calls_jac = 0

        comm = PETSc.COMM_SELF
        time_stepper = PETSc.TS().create(comm=comm)
        time_stepper.setProblemType(time_stepper.ProblemType.NONLINEAR)
        method = (self.algo_name.replace(self.PETSC_ODE_PREFIX, "")).lower()
        time_stepper.setType(method)
        time_stepper.setTime(self._problem.time_interval[0])
        time_stepper.setTimeStep(settings_["time_step"])
        time_stepper.setMaxTime(self._problem.time_interval[-1])
        time_stepper.setMaxSteps(settings_["maximum_steps"])
        time_stepper.setTolerances(settings_["atol"], settings_["rtol"])
        time_stepper.setExactFinalTime(PETSc.TS.ExactFinalTime.MATCHSTEP)

        self.__set_time_stepper_checkpoint_options(settings_)

        time_stepper.setMonitor(self.__monitor)
        dim = self._problem.initial_state.shape[0]
        self.__rhs_indices = arange(dim, dtype="int32")

        initial_state = convert_ndarray_to_mat_or_vec(self._problem.initial_state)
        state_dot = initial_state.duplicate()
        time_stepper.setRHSFunction(self._rhs_function, state_dot)
        if self._problem.jac_function_wrt_state is None and settings_["use_jacobian"]:
            msg = (
                "Jacobian of RHS function wrt state must be provided"
                " when 'use_jacobian' setting is True"
            )
            raise ValueError(msg)

        if (
            self._problem.jac_function_wrt_state is not None
            and settings_["use_jacobian"]
        ):
            jac_mat = PETSc.Mat().createDense([dim, dim], comm=comm)
            jac_mat.setUp()
            time_stepper.setRHSJacobian(self._jac_function_wrt_state, jac_mat, jac_mat)
        elif settings_["compute_adjoint"]:
            msg = (
                "The 'use_jacobian' setting is mandatory "
                "when 'compute_adjoint' is True."
            )
            raise ValueError(msg)

        if settings_["compute_adjoint"]:
            self.__init_adjoint(time_stepper, comm)

        ode_method = time_stepper.getType()
        if ode_method == "rk":
            ode_method += "/" + time_stepper.getRKType()
        LOGGER.info("Petsc ODE method selected : %s", ode_method)

        if self._problem.event_functions:
            self.__n_events = len(self._problem.event_functions)
            time_stepper.setEventHandler(
                [0] * self.__n_events,
                [True] * self.__n_events,
                self.__indicator_events,
                postevent=self.__post_event,
            )
            time_stepper.setEventTolerances(
                settings_.get("events_tol", 1e-6),
                vtol=settings_.get("events_vtol", [1e-9] * self.__n_events),
            )

        # allow an unlimited number of failures (step will be rejected and retried)
        time_stepper.setMaxSNESFailures(-1)

        # TODO: handle options for KSP, SNES etc
        # snes = time_stepper.getSNES()  # Nonlinear solver
        # snes.setTolerances(max_it=10)  *
        # Stop nonlinear solve after 10 iterations (TS will retry with shorter step)
        # ksp = snes.getKSP()  # Linear solver
        # ksp.setType(ksp.Type.CG)  # Conjugate gradients
        # pc = ksp.getPC()  # Preconditioner
        # pc.setType(pc.Type.GAMG)

        # Apply run-time options, e.g. -ts_adapt_monitor
        time_stepper.setFromOptions()

        time_stepper.solve(initial_state)
        time_stepper.getDict()

        ########################################
        self._problem.result.algorithm_name = self._algo_name  # OK
        self._problem.result.algorithm_settings = settings_  # OK
        self._problem.result.termination_time = time_stepper.getTime()
        self._problem.result.algorithm_has_converged = (
            time_stepper.getConvergedReason() >= 0
            and time_stepper.getConvergedReason()
            != time_stepper.ConvergedReason.CONVERGED_ITS
        )  # OK
        self._problem.result.algorithm_termination_message = self._TERMINATION_MESSAGES[
            time_stepper.getConvergedReason()
        ]

        if self._problem.compute_trajectory:
            if self._problem.solve_at_algorithm_times:
                self._problem.result.times = array(self._hist_times)
                self._problem.result.state_trajectories = array(self._hist_state).T
            else:
                self._problem.result.times = extract(
                    condition=(
                        self._problem.evaluation_times
                        <= self._problem.result.termination_time
                    ),
                    arr=self._problem.evaluation_times,
                )
                states_res = array(self._hist_state)
                times_res = array(self._hist_times)
                states = [
                    interp(
                        self._problem.evaluation_times,
                        times_res,
                        states_res[:, i],
                    )
                    for i in range(dim)
                ]
                self._problem.result.state_trajectories = array(states)

        self._problem.result.n_func_evaluations = time_stepper.getStepNumber()  # OK
        self._problem.result.n_jac_evaluations = self.__n_calls_jac

        if not self._problem.result.algorithm_has_converged:
            LOGGER.warning(self._problem.result.algorithm_termination_message)  # OK

        self._problem.result.final_state = time_stepper.vec_sol.array

        LOGGER.info("TS steps : %s", time_stepper.getStepNumber())
        LOGGER.info("TS non linear iterations : %s", time_stepper.getSNESIterations())
        LOGGER.info("TS linear iterations : %s", time_stepper.getKSPIterations())
        if time_stepper.getStepRejections():
            LOGGER.warning(
                "TS rejected steps SNES : %s", time_stepper.getStepRejections()
            )
        if time_stepper.getSNESFailures():
            LOGGER.warning("SNES failed  : %s", time_stepper.getSNESFailures())

        if settings_["compute_adjoint"]:
            time_stepper.adjointSolve()

            if self.adjoint_wrt_desvar:
                adjoint_wrt_desvar = vstack([
                    adj.getArray() for adj in self.adjoint_wrt_desvar
                ])
                self._problem.result.jac_wrt_desvar = adjoint_wrt_desvar
            if self.adjoint_wrt_state:
                adjoint_wrt_state = vstack([
                    adj.getArray() for adj in self.adjoint_wrt_state
                ])
                self._problem.result.jac_wrt_initial_state = adjoint_wrt_state

        return self._problem.result

    @staticmethod
    def __set_time_stepper_checkpoint_options(settings_: dict[str, Any]) -> None:
        """Sets the checkpointing options in TS.

        Args:
            settings_: TS library settings.
        """
        if not settings_["compute_adjoint"]:
            return

        petsc_options = PETSc.Options()
        if settings_["use_memory_checkpoints"]:
            petsc_options.setValue("ts_trajectory_type", "memory")

        max_disk_checkpoints = settings_["max_disk_checkpoints"]
        if max_disk_checkpoints:
            petsc_options.setValue("ts_trajectory_max_cps_disk", max_disk_checkpoints)

        max_memory_checkpoints = settings_["max_memory_checkpoints"]
        if max_memory_checkpoints:
            petsc_options.setValue("ts_trajectory_max_cps_ram", max_disk_checkpoints)

    def __init_adjoint(self, time_stepper: PETSc.TS, comm: PETSc.Comm) -> None:
        """Initialize the objects storing the adjoints.

        Args:
            time_stepper: The TS object.
            comm: the communicator object.
        """
        time_stepper.setSaveTrajectory()
        dim = self._problem.initial_state.shape[0]
        # Create vectors for cost
        # https://petsc.org/release/manualpages/Sensitivity/TSSetCostGradients/
        # the entries in these vectors must be correctly initialized with the
        # values lambda_i = df/dy|final_time mu_i = df/dp|final_time
        self.adjoint_wrt_state = []
        self.adjoint_wrt_desvar = []
        # the entries in these vectors must be correctly initialized
        # with the values lambda_i = df/dy|final_time mu_i = df/dp|final_time
        #  We compute the adjoint of the state,
        #  we assume that the objective is the state
        for i in range(dim):
            a_u = PETSc.Vec().createSeq(dim)
            a_u.setValues(i, 1)
            a_u.assemble()
            self.adjoint_wrt_state.append(a_u)
        if self._problem.jac_function_wrt_desvar is not None:
            numpy_jac = self._problem.jac_function_wrt_desvar(
                self._problem.time_interval[0], self._problem.initial_state
            )
            n_desvar = numpy_jac.shape[1]
            self.__desvar_indices = arange(n_desvar, dtype="int32")

            jac_p = PETSc.Mat().createDense([dim, n_desvar], comm=comm)
            jac_p.setUp()
            time_stepper.setRHSJacobianP(self._jac_function_wrt_desvar, jac_p)

            # Create vectors for cost
            # adjoint_wrt_desvar is adjoint wrt parameters at final time
            # It is initialized with the partial derivative of the function
            # wt to the design variables. So here the state depends on the
            # function through the ODE but there is no additional dependency
            # When extending to genera functions, we will have to add this term
            # as the function may depend on X in addition to the state.
            for _ in range(dim):
                a_p = PETSc.Vec().createSeq(n_desvar)
                a_p.assemble()
                self.adjoint_wrt_desvar.append(a_p)

            time_stepper.setCostGradients(
                self.adjoint_wrt_state, self.adjoint_wrt_desvar
            )
        else:
            LOGGER.warning(
                "ODE problem solved in adjoint mode but has no jac_desvar,"
                " this means that the ODE function is assumed not to depend"
                " on the design variables, only the initial condition does."
            )
            time_stepper.setCostGradients(self.adjoint_wrt_state)
