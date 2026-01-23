import numpy as np
import scipy.spatial as scspatial
from jaxtyping import Float
from tqdm import tqdm

from desmin_reconstruction.utils.math import n_sphere_volume


def neighbor_distances_and_weights(
    point: Float[np.ndarray, " dim"],
    neighbors: Float[np.ndarray, "neighbors dim"],
    sides: Float[np.ndarray, " dim"],
    edge_correction: bool,
) -> tuple[Float[np.ndarray, "neighbors dim"], Float[np.ndarray, " neighbors"]]:
    """Given a center point and a set of points to compute its nearby neighbors to,
    return the displacement vectors from the point to all its neighbors with in given
    radius. Additionally, return the weight values to correct for edge effects.

    If edge_correction=False, the weights will be an array of ones. Otherwise, the
    weights will be computed using Miles-Lantuejoul-Stoyan-Hanisch translation-corrected
    estimator."""
    displacements = neighbors - point
    weights = np.ones_like(displacements, shape=(displacements.shape[0],))
    if edge_correction:
        vol_intersection: Float[np.ndarray, " neighbors"] = np.prod(
            sides - np.abs(displacements), axis=-1
        )
        weights = weights / vol_intersection
    return displacements, weights


def ripley_K_per_point(
    points_tree: scspatial.KDTree,
    r: Float[np.ndarray, ""],
    sides: Float[np.ndarray, " dim"],
    edge_correction: bool = True,
) -> Float[np.ndarray, " n_points"]:
    in_range_inds: list[list[int]] = points_tree.query_ball_tree(points_tree, r)
    K_i = []
    for i, point in enumerate(points_tree.data):
        in_range_inds[i].remove(i)  # Remove self counting
        neighbors = points_tree.data[in_range_inds[i]]
        _, weights = neighbor_distances_and_weights(
            point, neighbors, sides, edge_correction
        )
        K_i.append(np.sum(weights))
    return np.asarray(K_i)


def ripley_K(
    points: Float[np.ndarray, "n_points dim"],
    r: Float[np.ndarray, " radii"],
    sides: Float[np.ndarray, " dim"],
    edge_correction: bool = True,
) -> Float[np.ndarray, " radii"]:
    """Estimates the Ripley's K function from n-dimensional spatial data.

    If edge_correction is True, performs edge correction via the
    Miles-Lantuejoul-Stoyan-Hanisch translation-corrected estimator."""

    n_points, dim = points.shape
    if len(sides) != dim:
        raise ValueError(
            "The length of sides must correspond to the dimension of the points"
        )
    # Use KD tree to speed up neighbor search

    tree = scspatial.KDTree(points)
    K: Float[np.ndarray, " radii"] = np.asarray(
        [np.sum(ripley_K_per_point(tree, r_, sides, edge_correction)) for r_ in tqdm(r)]
    )

    # Inverse squared intensity
    lambda_sqr_inv = np.prod(sides) ** 2 / (n_points * (n_points - 1))

    return lambda_sqr_inv * K


def ripley_L_from_K(
    K_vals: Float[np.ndarray, " radii"], dim: int
) -> Float[np.ndarray, " radii"]:
    """Computes the Ripley's L function from values of the K function."""
    return (K_vals / n_sphere_volume(dim)) ** (1 / dim)


def ripley_H_from_L(
    L_vals: Float[np.ndarray, " radii"], r: Float[np.ndarray, " radii"]
) -> Float[np.ndarray, " radii"]:
    """Computes the Ripley's H function from values of the L function."""
    return L_vals - r
