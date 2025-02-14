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
#        :author: Giulio Gargantini
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for the conversion utility."""

from __future__ import annotations

import sys

import pytest
from numpy import array
from petsc4py import init
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix

from gemseo_petsc.utils.conversion import convert_ndarray_to_mat_or_vec

init(sys.argv)
from petsc4py import PETSc  # noqa: E402

# def test_convert_vec_to_ndarray():
#     """Tests the conversion from PETSc Vector to ndarray."""
#     ndarray1 = array([0.0, 1.0, 2.0])
#     array([[1.0, 0.0], [0.0, 1.0]])
#
#     vec1 = PETSc.Vec().createSeq(3)
#     vec1.setValues([0, 1, 2], ndarray1)
#     mat2d = PETSc.Mat().createDense([2, 2], array=array([1.0, 0.0, 0.0, 1.0]))
#     # mat2a = PETSc.Mat().createDiagonal([1., 0., 0., 1.])
#     mat2s = PETSc.Mat().createAIJ([2, 2])
#     mat2s.setValues([0, 1], [0, 1], [1.0, 0.0, 0.0, 1.0])
#
#     vec1.assemble()
#     mat2d.assemble()
#     # mat2a.assemble()
#     mat2s.assemble()
#
#     vec1_conv = convert_vec_to_ndarray(vec1)
#     # mat2d_conv = convert_vec_to_ndarray(mat2d)
#     # mat2a_conv = convert_vec_to_ndarray(mat2a)
#     # mat2s_conv = convert_vec_to_ndarray(mat2s)
#     assert allclose(vec1_conv, ndarray1, atol=1.0e-6)


def test_convert_ndarray_to_mat_or_vec():
    """Tests the conversion from ndarray to PETSc Vec or Mat."""
    ndarray1 = array([0.0, 1.0, 2.0])
    ndarray2 = array([[1.0, 0.0], [0.0, 1.0]])

    vec1 = PETSc.Vec().createSeq(3)
    vec1.setValues([0, 1, 2], ndarray1)
    mat2d = PETSc.Mat().createDense([2, 2], array=array([1.0, 0.0, 0.0, 1.0]))
    mat2s = PETSc.Mat().createAIJ([2, 2])
    mat2s.setValues([0, 1], [0, 1], [1.0, 0.0, 0.0, 1.0])

    vec1.assemble()
    mat2d.assemble()
    mat2s.assemble()

    vec1_conv = convert_ndarray_to_mat_or_vec(ndarray1)
    vec2_conv = convert_ndarray_to_mat_or_vec(ndarray2)

    assert vec1_conv.equal(vec1)
    assert vec2_conv.equal(mat2d)


def test_check_error_dimension():
    """Tests the error message when converting a ndarray with dimension > 2."""
    ndarray_3d = array([[[1.0, 0.0], [0.0, 1.0]]])
    with pytest.raises(
        ValueError, match=r"The dimension of the input array \(3\) is not supported\."
    ):
        convert_ndarray_to_mat_or_vec(ndarray_3d)


def test_convert_sparse_matrices():
    """Tests the conversion of sparse matrices."""
    # Let us consider the following sparse matrix
    #   1   0   0
    #   0   0   0
    #   0   1   3
    data = [1.0, 1.0, 3.0]
    row_ind_csc = [0, 2, 2]
    col_ind = [0, 1, 2]

    mat = PETSc.Mat().createAIJ([3, 3])
    mat.setValuesBlockedIJV([0, 1, 1, 3], col_ind, data)
    mat.assemble()

    csc_mat = csc_matrix((data, (row_ind_csc, col_ind)))
    mat_from_csc = convert_ndarray_to_mat_or_vec(csc_mat)
    assert mat_from_csc.equal(mat)


def test_convert_sparse_array():
    """Tests the conversion of sparse matrices."""
    # Let us consider the following sparse matrix
    #   1   0   0   2
    data = [1.0, 2.0]
    col_ind = [0, 3]

    vec = PETSc.Vec().createSeq(5)
    vec.setUp()
    vec.setValues(col_ind, data)
    vec.assemble()

    arr = csr_matrix((data, ([0, 3], [0, 0])), shape=(5, 1))
    vec_from_csr = convert_ndarray_to_mat_or_vec(arr)

    assert vec_from_csr.equal(vec)
