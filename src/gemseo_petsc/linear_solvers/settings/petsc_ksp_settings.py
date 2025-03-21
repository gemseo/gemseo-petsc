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
"""Settings for the PETSc KSP linear solvers."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from typing import Annotated

from gemseo.algos.linear_solvers.base_linear_solver_settings import (
    BaseLinearSolverSettings,
)
from gemseo.typing import StrKeyMapping  # noqa: TC002
from pydantic import AliasChoices
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveInt
from pydantic import WithJsonSchema
from strenum import StrEnum


class PreconditionerType(StrEnum):
    """The type of the precondtioner.

    See
    [https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.PC.Type-class.html].
    """

    JACOBI = "jacobi"
    BJACOBI = "bjacobi"
    SOR = "sor"
    EISENSTAT = "eisenstat"
    ICC = "icc"
    ILU = "ilu"
    ASM = "asm"
    GASM = "gasm"
    GAMG = "gamg"
    BDDC = "bddc"
    KSP = "ksp"
    COMPOSITE = "composite"
    LU = "lu"
    CHOLESKY = "cholesky"
    NONE = "none"
    SHELL = "shell"


class BasePetscKSPSettings(BaseLinearSolverSettings):
    """The base settings of the PETSc KSP algorithms.

    `_TARGET_CLASS_NAME` will be overloaded for each algorithm.
    """

    _TARGET_CLASS_NAME = ""

    atol: NonNegativeFloat = Field(
        default=1e-50,
        description="""The absolute convergence tolerance.

Absolute tolerance of the (possibly preconditioned) residual norm.
Algorithm stops if norm(b - A @ x) <= max(rtol*norm(b), atol).""",
    )

    dtol: NonNegativeFloat = Field(
        default=1e5,
        description="""The divergence tolerance.

The amount the (possibly preconditioned) residual norm can increase.""",
    )

    ksp_pre_processor: Annotated[Callable, WithJsonSchema({})] | None = Field(
        default=None,
        description="""A callback function that is called before calling ksp.solve().

The function is called with (KSP problem, options dict) as arguments.
It allows the user to obtain an advanced configuration
that is not supported by the current wrapper.
If None, do not perform any call.""",
    )

    maxiter: PositiveInt = Field(
        default=100_000,
        validation_alias=AliasChoices("max_iter", "maxiter"),
        description="Maximum number of iterations.",
    )

    monitor_residuals: bool = Field(
        default=False,
        description="""Whether to store the residuals during convergence.

WARNING: as said in Petsc documentation,
 "the routine is slow and should be used only for testing or convergence studies,
 not for timing."
""",
    )

    options_cmd: StrKeyMapping | None = Field(
        default=None,
        description="""The options to pass to the PETSc KSP solver.

If None, use the default options.""",
    )

    preconditioner_type: PreconditionerType | None = Field(
        default=PreconditionerType.ILU,
        description="""The type of the precondtioner.

If None, no preconditioning is applied.
See [https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.PC.Type-class.html]""",
    )

    rtol: NonNegativeFloat = Field(
        default=1e-5,
        description="""The relative convergence tolerance.

Relative decrease in the (possibly preconditioned) residual norm.""",
    )

    set_from_options: bool = Field(
        default=False, description="""Whether the options are set from sys.argv."""
    )

    view_config: bool = Field(
        default=False,
        description="""Whether to view the configuration of the solver before run.

Configuration is viewed by calling ksp.view().""",
    )
