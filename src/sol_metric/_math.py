"""Shared random projections and empirical transport conventions.

First-level directions are uniform on the sphere; second-level directions are
raw Gaussian-process draws. Random draws are independent of device/block sizes.
"""

from __future__ import annotations

import numpy as np

QUANTILE_CONVENTION = "empirical_bin_average_v1"
GP_NORMALIZATION = "none"


def bin_average_plan(lengths, count):
    """Sparse exact overlaps of equal quantile bins with empirical atom masses.

    Indices/weights have shape (documents, bins, atoms_per_bin). All indices
    address actual tokens, including zero-weight entries, never +inf padding.
    Integer boundary arithmetic avoids ambiguity at empirical CDF jumps.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    if count < 1 or lengths.ndim != 1 or len(lengths) == 0 or np.any(lengths < 1):
        raise ValueError("positive bin count and nonempty documents required")
    t = lengths[:, None, None]
    bins = np.arange(count, dtype=np.int64)[None, :, None]
    width = (int(lengths.max()) + count - 1) // count + 1
    atoms = bins * t // count + np.arange(width, dtype=np.int64)
    mass = np.maximum(
        0, np.minimum((atoms + 1) * count, (bins + 1) * t) - np.maximum(atoms * count, bins * t)
    )
    return np.minimum(atoms, t - 1), mass / t


def directions(count, dimension, seed):
    values = np.random.default_rng(seed).standard_normal((count, dimension))
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def gp_directions(count, quantiles, lengthscale, seed):
    """Raw zero-mean-prior GP samples, without sample centering or RMS scaling.

    RBF covariance has unit diagonal before numerical Cholesky jitter. With
    ``lengthscale=None``, return independent standard Gaussian grid values.
    """
    values = np.random.default_rng(seed).standard_normal((count, quantiles))
    if lengthscale is not None:
        grid = (np.arange(quantiles) + 0.5) / quantiles
        covariance = np.exp(-0.5 * ((grid[:, None] - grid) / lengthscale) ** 2)
        for jitter in (1e-6, 1e-5, 1e-4):
            try:
                factor = np.linalg.cholesky(covariance + jitter * np.eye(quantiles))
                break
            except np.linalg.LinAlgError:
                continue
        else:
            raise np.linalg.LinAlgError("GP covariance factorization failed")
        values = values @ factor.T
    return values


def transport_grid(nx, ny):
    """Exact atom overlaps for sorted uniform empirical measures."""
    if nx == ny:
        return None
    edges = np.unique(np.concatenate((np.arange(nx + 1) / nx, np.arange(ny + 1) / ny)))
    weights = np.diff(edges)
    grid = edges[:-1] + weights / 2
    ix = np.clip(np.ceil(grid * nx).astype(np.int64) - 1, 0, nx - 1)
    iy = np.clip(np.ceil(grid * ny).astype(np.int64) - 1, 0, ny - 1)
    return ix, iy, weights
