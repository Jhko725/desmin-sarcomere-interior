import scipy.sparse as scsparse


def remove_diagonal(coo_mat: scsparse.coo_matrix) -> scsparse.coo_matrix:
    """Remove the diagonal elements from a given COO sparse matrix.

    Unlike coo_mat.setdiag(0) followed by coo_mat.eliminate_zeros(), this method
    does not remove nondiagonal explicit zeros."""
    nondiag_inds = ~(coo_mat.row == coo_mat.col)
    return scsparse.coo_matrix(
        (
            coo_mat.data[nondiag_inds],
            (coo_mat.row[nondiag_inds], coo_mat.col[nondiag_inds]),
        )
    )
