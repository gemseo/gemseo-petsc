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
"""Settings for the PETSc ordinary differential equation solver. solvers."""

from __future__ import annotations

from gemseo.algos.ode.base_ode_solver_settings import BaseODESolverSettings
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt
from strenum import StrEnum


class ODESolverType(StrEnum):
    """The type of ODE Solver.

    See
    [https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.PC.Type-class.html].
    """

    EULER = "PETSC_ODE_EULER"
    SSP = "PETSC_ODE_SSP"
    RK = "PETSC_ODE_RK"
    BEULER = "PETSC_ODE_BEULER"
    CN = "PETSC_ODE_CN"
    THETA = "PETSC_ODE_THETA*"
    ALPHA = "PETSC_ODE_ALPHA"
    GL = "PETSC_ODE_GL"
    EIMEX = "PETSC_ODE_EIMEX"
    ARKIMEX = "PETSC_ODE_ERKIMEX"
    ROSW = "PETSC_ODE_ROSW"
    GLEE = "PETSC_ODE_GLEE"
    MPRK = "PETSC_ODE_MPRK"
    BASICSYMPLECTIC = "PETSC_ODE_BASICSYMPLECTIC"
    IRK = "PETSC_ODE_IRK"


class FinalTimeBehavior(StrEnum):
    """The behavior at the end of the resolution."""

    UNSPECIFIED = "TS_EXACTFINALTIME_UNSPECIFIED"
    STEP_OVER = "TS_EXACTFINALTIME_STEPOVER"
    INTERPOLATE = "TS_EXACTFINALTIME_INTERPOLATE"
    MATCH_STEP = "TS_EXACTFINALTIME_MATCHSTEP"


class PetscTSSettings(BaseODESolverSettings):
    """The base settings of the PETSc TS algorithms."""

    time_step: PositiveFloat = Field(
        description="""Value of the time step.""",
    )

    maximum_steps: PositiveInt = Field(
        description="""Total number of time steps to run""",
    )

    ode_solver_type: ODESolverType | None = Field(
        default=ODESolverType.RK,
        description="""The type of ODE solver.""",
    )

    max_disk_checkpoints: NonNegativeInt = Field(
        default=0,
        description="""Maximum number of checkpoints stored on the disk.
            When zero, no maximum is set.
            For more information on checkpoints, see:
            [https://petsc4py.readthedocs.io/en/stable/manual/sensitivity_analysis/#checkpointing]
            Maps the PETSC.TS ts_trajectory_max_cps_disk option.""",
    )

    max_memory_checkpoints: NonNegativeInt = Field(
        default=0,
        description="""Maximum number of checkpoints stored on in memory.
            When zero, no maximum is set.
            Maps the PETSC.TS ts_trajectory_max_cps_ram option.
            When both max_memory_checkpoints and max_disk_checkpoints are set,
            an automatic mix checkpointing (disk and RAM) scheme is setup.""",
    )

    rtol: NonNegativeFloat = Field(
        default=1e-6,
        description="""The relative convergence tolerance.

Set tolerance for local truncation error when using an adaptive controller. See:
[https://petsc.org/main/manualpages/TS/TSSetTolerances/]""",
    )

    atol: NonNegativeFloat = Field(
        default=1e-6,
        description="""The absolute convergence tolerance.

Set tolerance for local truncation error when using an adaptive controller. See:
[https://petsc.org/main/manualpages/TS/TSSetTolerances/]""",
    )

    use_memory_checkpoints: bool = Field(
        default=False,
        description="""Whether to store checkpoints in the RAM.
            By default the checkpoints are stored on the disk as binary files.
            Maps the PETSC.TS ts_trajectory_type memory option.""",
    )

    compute_adjoint: bool = Field(
        default=False,
        description="""Whether to use adjoint to compute the derivatives.""",
    )

    use_jacobian: bool = Field(
        default=False, description="""Whether to use the jacobian of the RHS."""
    )

    final_time_behaviour: FinalTimeBehavior | None = Field(
        default=FinalTimeBehavior.MATCH_STEP,
        description="""Behaviour to adopt near the final time.
            For more information, see
            https://petsc.org/release/manualpages/TS/TSExactFinalTimeOption/""",
    )
