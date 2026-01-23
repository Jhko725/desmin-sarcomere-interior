import numpy as np
import scipy.spatial as scspatial
from jaxtyping import Float

from .base import neighbor_distances_and_weights, ripley_K_per_point, ripley_L_from_K


def getis_franklin_L(
    points: Float[np.ndarray, "n_points dim"],
    r: Float[np.ndarray, ""],
    sides: Float[np.ndarray, " dim"],
    edge_correction: bool = True,
) -> Float[np.ndarray, " n_points"]:
    """Computes the univariate Getis and Franklin L function, which is a per-point
    variant of the Ripley's L function, that provides information on the local
    clusting strength."""
    tree = scspatial.KDTree(points)
    K_per_point = ripley_K_per_point(tree, r, sides, edge_correction)
    coeff = np.prod(sides) ** 2 / (points.shape[0] - 1)
    return ripley_L_from_K(coeff * K_per_point, points.shape[1])


def getis_franklin_L_bivariate(
    points: Float[np.ndarray, "n_points dim"],
    others: Float[np.ndarray, "n_others dim"],
    r: Float[np.ndarray, ""],
    sides: Float[np.ndarray, " dim"],
    edge_correction: bool = True,
) -> Float[np.ndarray, " n_points"]:
    """Computes the bivariate Getis and Franklin L function, which provides information
    on the local co-clustering properties of the two point clouds.

    This quantity is essentially a variant of the Ripley's L function, but with center
    point and the neighbors sampled from different point clouds."""
    points_tree = scspatial.KDTree(points)
    others_tree = scspatial.KDTree(others)
    in_range_inds: list[list[int]] = points_tree.query_ball_tree(others_tree, r)

    K_bivar = []
    for i, point in enumerate(points):
        neighbors = others[in_range_inds[i]]
        _, weights = neighbor_distances_and_weights(
            point, neighbors, sides, edge_correction
        )
        K_bivar.append(np.sum(weights))

    coeff = np.prod(sides) ** 2 / others.shape[0]
    return ripley_L_from_K(coeff * np.asarray(K_bivar), points.shape[1])
