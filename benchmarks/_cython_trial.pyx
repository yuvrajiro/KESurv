# cython: boundscheck=False, wraparound=False, cdivision=True
"""Optional Cython trial for the similarity-vote hot loop.

Build only for comparison; the package uses NumPy/Numba backends.
"""

cimport cython
from libc.math cimport fabs


def agreement_cython(
    const float[:, :, ::1] train,
    const float[:, ::1] query,
    const long long[::1] indices,
    const double[::1] widths,
    double cutoff,
    int minimum,
):
    cdef Py_ssize_t n = indices.shape[0]
    cdef Py_ssize_t m = train.shape[0]
    cdef Py_ssize_t k, learner, point
    cdef double area
    cdef int votes
    cdef unsigned char[::1] selected = bytearray(n)
    for k in range(n):
        votes = 0
        for learner in range(m):
            area = 0.0
            for point in range(widths.shape[0]):
                area += fabs(train[learner, indices[k], point] - query[learner, point]) * widths[point]
                if area > cutoff + 1e-12:
                    break
            if area <= cutoff + 1e-12:
                votes += 1
            if votes >= minimum or votes + m - learner - 1 < minimum:
                break
        selected[k] = votes >= minimum
    return selected

