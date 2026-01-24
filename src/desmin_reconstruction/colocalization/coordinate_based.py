import jax
import numpy as np
import scipy.spatial as scspatial
from jaxtyping import Float

from desmin_reconstruction.analysis.ripley.base import bin_sparse_distance_matrix
from desmin_reconstruction.utils.array import midpoints, powspace
from desmin_reconstruction.utils.math import n_sphere_volume
from desmin_reconstruction.utils.sparse import remove_diagonal
from desmin_reconstruction.utils.stats import pearsonr, spearmanr_batched


def coordinate_based_colocalization(
    points: Float[np.ndarray, "N dim"],
    points_other: Float[np.ndarray, "N_other dim"],
    r_max: float,
    bins: int,
) -> Float[np.ndarray, " N"]:
    """Given two sets of localization coordinates corresponding to two species to check
    colocalization for, compute the per-point CBC scores as described in [1].

    [1] S. Malkusch et al. Coordinate-based colocalization analysis of single-molecule
    localization microscopy data. Histochem. Cell. Biol. 137 (2012)."""

    dim = points.shape[1]
    if dim != points_other.shape[1]:
        raise ValueError(
            "Coordinates for the two species must have the same number of dimensions"
        )

    tree_self = scspatial.KDTree(points)
    tree_other = scspatial.KDTree(points_other)

    dist_mat_aa = tree_self.sparse_distance_matrix(
        tree_self, max_distance=r_max, output_type="coo_matrix"
    )
    dist_mat_ab = tree_self.sparse_distance_matrix(
        tree_other, max_distance=r_max, output_type="coo_matrix"
    )
    distance_bins: Float[np.ndarray, "bins+1"] = np.linspace(0, r_max, bins + 1)
    dist_weights: Float[np.ndarray, "bins"] = (r_max / midpoints(distance_bins)) ** dim

    counts_aa: Float[np.ndarray, "N bins"] = np.cumsum(
        bin_sparse_distance_matrix(dist_mat_aa, distance_bins), axis=-1
    )
    D_aa = (counts_aa / counts_aa[:, -1:]) * dist_weights

    counts_ab: Float[np.ndarray, "N bins"] = np.cumsum(
        bin_sparse_distance_matrix(dist_mat_ab, distance_bins), axis=-1
    )
    D_ab = (counts_ab / counts_ab[:, -1:]) * dist_weights

    S_a: Float[np.ndarray, " N"] = spearmanr_batched(D_aa, D_ab)

    min_dist_ab: Float[np.ndarray, " N"] = (
        dist_mat_ab.min(axis=1, explicit=True).toarray().squeeze()
    )
    C_a = S_a * np.exp(-min_dist_ab / r_max)
    return C_a


def coordinate_based_colocalization_liu(
    points: Float[np.ndarray, "N dim"],
    points_other: Float[np.ndarray, "N_other dim"],
    r_max: float,
    bins: int,
    sides: Float[np.ndarray, " dim"],
) -> Float[np.ndarray, " N"]:
    """Given two sets of localization coordinates corresponding to two species to check
    colocalization for, compute the per-point KCBC scores as described in [1].

    [1] X. Liu et al. KCBC- a correlation-based method for co-localization analysis of
    super-resolution microscopy images using bivariate Ripley's K functions. J. Appl.
    Stat. 51, 16 (2024)."""
    dim = points.shape[1]
    if dim != points_other.shape[1]:
        raise ValueError(
            "Coordinates for the two species must have the same number of dimensions"
        )

    tree_self = scspatial.KDTree(points)
    tree_other = scspatial.KDTree(points_other)

    dist_mat_aa = tree_self.sparse_distance_matrix(
        tree_self, max_distance=r_max, output_type="coo_matrix"
    )
    dist_mat_aa = remove_diagonal(dist_mat_aa)

    dist_mat_ab = tree_self.sparse_distance_matrix(
        tree_other, max_distance=r_max, output_type="coo_matrix"
    )

    distances: Float[np.ndarray, "bins+1"] = powspace(0, r_max, bins + 1, exponent=dim)

    vol = np.prod(sides)

    x_i, x_j = tree_self.data[dist_mat_aa.row], tree_self.data[dist_mat_aa.col]
    vol_intersection = np.prod(sides - np.abs(x_i - x_j), axis=1)
    weights = vol_intersection / vol
    G_aa_ = bin_sparse_distance_matrix(dist_mat_aa, distances, weights)
    # G_aa is the differenced per-point Ripley's K function
    G_aa = (vol / (len(points) - 1)) * G_aa_ / n_sphere_volume(dim, distances[1])

    xa_i, xb_j = tree_self.data[dist_mat_ab.row], tree_other.data[dist_mat_ab.col]
    vol_intersection = np.prod(sides - np.abs(xa_i - xb_j), axis=1)
    weights = vol_intersection / vol
    G_ab_ = bin_sparse_distance_matrix(dist_mat_ab, distances, weights)
    # G_aa is the differenced per-point bivariate Ripley's K function
    G_ab = (vol / len(points_other)) * G_ab_ / n_sphere_volume(dim, distances[1])

    KCBC = np.asarray(jax.vmap(pearsonr)(G_aa, G_ab))

    return KCBC
