import numpy as np
from jaxtyping import Array, Float


AbstractArray = Array | np.ndarray


def powspace(
    start: float, stop: float, num: int, exponent: int | float
) -> Float[np.ndarray, " num"]:
    """Return an array of values whose powers to the exponent are equispaced."""

    powers = np.linspace(start**exponent, stop**exponent, num)
    return powers ** (1 / exponent)


def midpoints(x: Float[AbstractArray, " N"]) -> Float[AbstractArray, " N-1"]:
    """Given an array of values, return an array of midpoints.

    Particularly useful when computing the centers of histogram bins given an array
    of bin edges.
    This function works with all ArrayLike types."""
    return (x[1:] + x[:-1]) / 2
