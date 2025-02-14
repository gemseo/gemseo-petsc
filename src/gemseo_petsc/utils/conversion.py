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
#        :author: Isabelle Santos
"""Conversion functions.

Utilities for converting objects between those used by SciPy and those used by PETSc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from petsc4py import PETSc
from scipy.sparse import csr_matrix
from scipy.sparse.base import issparse

if TYPE_CHECKING:
    from numpy.typing import NDArray


# def convert_vec_to_ndarray(vec_or_mat: PETSc.Mat | PETSc.Vec) -> NDArray[float]:
#     """Convert a PETSc Mat or Vec to a Numpy array.
#
#     Args:
#         vec_or_mat: A PETSc Mat or Vec.
#
#     Returns:
#         A Numpy array.
#     """
#     return vec_or_mat.getValues(range(vec_or_mat.getSize()))
#     # TODO: It does not work for matrices.


def convert_ndarray_to_mat_or_vec(
    np_arr: NDArray[float],
) -> PETSc.Mat | PETSc.Vec:
    """Convert a Numpy array to a PETSc Mat or Vec.

    Args:
         np_arr: The input Numpy array.

    Returns:
        A PETSc Mat or Vec, depending on the input dimension.

    Raises:
        ValueError: If the dimension of the input vector is greater than 2.
    """
    n_dim = np_arr.ndim
    if n_dim > 2:
        msg = f"The dimension of the input array ({n_dim}) is not supported."
        raise ValueError(msg)

    if issparse(np_arr):
        if not isinstance(np_arr, csr_matrix):
            np_arr = np_arr.tocsr()
        if n_dim == 2 and np_arr.shape[1] > 1:
            petsc_arr = PETSc.Mat().createAIJ(
                size=np_arr.shape, csr=(np_arr.indptr, np_arr.indices, np_arr.data)
            )
        else:
            petsc_arr = PETSc.Vec().createSeq(np_arr.shape[0])
            petsc_arr.setArray(np_arr.todense())
    else:
        if n_dim == 1:
            a = array(np_arr, dtype=PETSc.ScalarType)
            petsc_arr = PETSc.Vec().createWithArray(a)
        else:
            petsc_arr = PETSc.Mat().createDense(np_arr.shape, array=np_arr)
            # a_shape = np_arr.shape
            # petsc_arr.setUp()
            # petsc_arr.setValues(
            #     arange(a_shape[0], dtype="int32"),
            #     arange(a_shape[1], dtype="int32"),
            #     np_arr,
            # )
    petsc_arr.assemble()
    return petsc_arr
