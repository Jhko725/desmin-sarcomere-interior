import numpy as np
import scipy.spatial as scspatial
from jaxtyping import Float

from .base import neighbor_distances_and_weights, ripley_L_from_K


def cartesian_to_spherical(
    xyz: Float[np.ndarray, "N 3"],
) -> tuple[Float[np.ndarray, " N"], Float[np.ndarray, " N"], Float[np.ndarray, " N"]]:
    r = np.sqrt(np.sum(xyz * xyz, axis=-1))
    theta = np.atan2(xyz[:, 1], xyz[:, 0])
    phi = np.acos(xyz[:, 2] / r)
    return r, theta, phi


def angular_ripley_H(
    points: Float[np.ndarray, "N 2"],
    r: float = 1.0,
    num_bins: int = 100,
    sides: Float[np.ndarray, "2"] = np.array([4.0, 4.0]),
    edge_correction: bool = True,
):
    n_points = points.shape[0]
    sides = np.asarray(sides)
    vol = np.prod(sides)
    tree = scspatial.KDTree(points)
    in_range_inds = tree.query_ball_tree(tree, r)

    bins = np.linspace(-np.pi, np.pi, num_bins)
    k_angular = np.zeros_like(bins, shape=(len(bins) - 1,))

    for i, point in enumerate(points):
        in_range_inds[i].remove(i)  # Remove self counting
        neighbors = points[in_range_inds[i]]
        displacements, weights = neighbor_distances_and_weights(
            point, neighbors, sides, edge_correction
        )

        angles = np.atan2(displacements[:, 1], displacements[:, 0])

        k_angular = (
            k_angular + np.histogram(angles, bins, weights=weights, density=False)[0]
        )

    delta_angle = bins[1] - bins[0]
    lambda_sq_inv = (vol * vol) / (n_points * (n_points - 1))
    K_angular = (2 * np.pi / delta_angle) * lambda_sq_inv * k_angular
    H_angular = ripley_L_from_K(K_angular, points.shape[1]) - r

    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, H_angular


def angular_ripley_H_3d(
    points: Float[np.ndarray, "N 3"],
    r: float = 1.0,
    num_bins: tuple[int, int] = (100, 100),
    sides: Float[np.ndarray, "3"] = np.array([4.0, 4.0, 1.0]),
    edge_correction: bool = True,
):
    n_points = points.shape[0]
    sides = np.asarray(sides)
    volume = np.prod(sides)
    tree = scspatial.KDTree(points)
    in_range_inds: list[list[int]] = tree.query_ball_tree(tree, r)

    bins_theta: Float[np.ndarray, " N_theta+1"] = np.linspace(
        -np.pi, np.pi, num_bins[0]
    )
    bins_phi: Float[np.ndarray, " N_phi+1"] = np.linspace(0, np.pi, num_bins[1])
    k_angular: Float[np.ndarray, "N_theta N_phi"] = np.zeros_like(
        bins_theta, shape=(num_bins[0] - 1, num_bins[1] - 1)
    )

    for i, point in enumerate(points):
        in_range_inds[i].remove(i)  # Remove self counting
        neighbors = points[in_range_inds[i]]
        displacements, weights = neighbor_distances_and_weights(
            point, neighbors, sides, edge_correction
        )

        _, points_theta, points_phi = cartesian_to_spherical(displacements)

        k_angular = (
            k_angular
            + np.histogram2d(
                points_theta,
                points_phi,
                (bins_theta, bins_phi),
                weights=weights,
                density=False,
            )[0]
        )

    lambda_sq_inv = (volume * volume) / (n_points * (n_points - 1))
    phi: Float[np.ndarray, " N_phi"] = (bins_phi[1:] + bins_phi[:-1]) / 2
    d_theta, d_phi = bins_theta[1] - bins_theta[0], bins_phi[1] - bins_phi[0]
    d_Omega: Float[np.ndarray, " N_phi"] = np.sin(phi) * d_theta * d_phi

    K_angular = (4 * np.pi / d_Omega) * lambda_sq_inv * k_angular
    H_angular = ripley_L_from_K(K_angular, points.shape[1]) - r

    return H_angular, (bins_theta, bins_phi)
