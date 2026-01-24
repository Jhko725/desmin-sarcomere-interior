import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats as scstats
from jaxtyping import Array, Bool, Float


def pearsonr(
    x: Float[Array, " N"], y: Float[Array, " N"], mask: Bool[Array, " N"] | None = None
) -> Float[Array, ""]:
    """
    A JAX function for computing the Pearson correlation coefficient (r)
     between data points x, and y.

    If mask is given, r value is computed between x[mask] and y[mask].
    """
    if mask is None:
        mask = jnp.ones_like(x, dtype=jnp.bool)

    def masked_mean(arr: Float[Array, " N"]) -> Float[Array, ""]:
        return jnp.sum(arr * mask) / jnp.sum(mask)

    x_, y_ = x - masked_mean(x), y - masked_mean(y)
    s_xx, s_yy = masked_mean(x_ * x_), masked_mean(y_ * y_)
    s_xy = masked_mean(x_ * y_)
    return s_xy / jnp.sqrt(s_xx * s_yy)


def spearmanr(x: Float[Array, " N"], y: Float[Array, " N"]) -> Float[Array, ""]:
    """A JAX function for computing the Spearman's rank correlation coefficient.

    This function is jittable and vmappable."""
    return pearsonr(
        jax.scipy.stats.rankdata(x),
        jax.scipy.stats.rankdata(y),
    )


def spearmanr_batched(
    x_batch: Float[np.ndarray, "B N"], y_batch: Float[np.ndarray, "B N"]
) -> Float[np.ndarray, " B"]:
    """A numpy-based function to compute the Spearman's rank corrlation coefficient
    over a batch of data points x_batch and y_batch.

    Note that one can achieve the same result via jax.vmap(spearmanr)(x_batch, y_batch).
    """
    return scstats.pearsonr(
        scstats.rankdata(x_batch, axis=1), scstats.rankdata(y_batch, axis=1), axis=1
    )[0]
