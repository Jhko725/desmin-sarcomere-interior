import numpy as np
import scipy.spatial as scspatial
from jaxtyping import Float

from .base import ripley_K_per_point, ripley_L_from_K


def getis_franklin_L(
    points: Float[np.ndarray, "n_points dim"],
    r: Float[np.ndarray, ""],
    sides: Float[np.ndarray, " dim"],
    edge_correction: bool = True,
):
    """Computes the univariate Getis and Franklin L function, which is a per-point
    variant of the Ripley's L function, that provided information on the local
    clusting strength."""
    tree = scspatial.KDTree(points)
    K_per_point = ripley_K_per_point(tree, r, sides, edge_correction)
    coeff = np.prod(sides) ** 2 / (points.shape[0] - 1)
    return ripley_L_from_K(coeff * K_per_point, points.shape[1])
