"""Apple Silicon inference backend for HECTOR.

Provides TensorFlow-free implementations of the Apple Silicon inference path:
- UMAP / t-SNE dimensionality reduction (MLX Metal GPU)
- DualHeadVGAEEncoder (MLX Metal GPU)
- CellTypeEmbedder relational GAT (NumPy)
- HPLHead proxy-based classifier (numpy)
- Consensus scoring pipeline (numpy)
- GRIT label propagation (scipy.sparse)
- k-NN graph construction (MLX Metal GPU)

All public functions accept and return numpy arrays. MLX is imported lazily
inside GPU-touching functions so the module is importable on any platform ---
functions raise ImportError only when called without MLX installed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _l2_normalize(x, axis=-1, eps=1e-10):
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norms, eps)

def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def _load_weight_list(h5_group, prefix="weight"):
    weights = []
    i = 0
    while f"{prefix}_{i}" in h5_group:
        ds = h5_group[f"{prefix}_{i}"]
        weights.append(ds[()] if ds.shape == () else ds[:])
        i += 1
    return weights

def _mx_layer_norm(x, gamma, beta, eps=1e-6):
    import mlx.core as mx
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) * mx.rsqrt(var + eps) + beta

def _mx_leaky_relu(x, alpha=0.2):
    import mlx.core as mx
    return mx.where(x >= 0, x, alpha * x)


# =========================================================================== #
# UMAP engine (Metal GPU)
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Stage 1: a/b membership-curve parameters (pure SciPy)
# --------------------------------------------------------------------------- #
def find_ab_params(spread: float, min_dist: float) -> tuple[float, float]:
    """Fit the UMAP membership curve ``1/(1 + a*x^(2b))`` to the target ramp."""
    from scipy.optimize import curve_fit

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3.0, 300)
    yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))
    (a, b), _ = curve_fit(curve, xv, yv, p0=[1.0, 1.0], maxfev=10000)
    return float(a), float(b)


# --------------------------------------------------------------------------- #
# Stage 2: initialization (pure NumPy, deterministic)
# --------------------------------------------------------------------------- #
def pca_init(X_pca: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Initialize the 2-D layout from the first 2 PCs, per-axis scaled to
    [0, 10] (anisotropic) plus tiny seeded noise. Deterministic."""
    e = np.asarray(X_pca[:, :2], dtype=np.float32).copy()
    e = e - e.mean(axis=0)
    span = (e.max(axis=0) - e.min(axis=0)) + 1e-10
    e = 10.0 * (e - e.min(axis=0)) / span
    rng = np.random.RandomState(random_state)
    e = e + rng.normal(scale=1e-4, size=e.shape).astype(np.float32)
    return e.astype(np.float32)


# --------------------------------------------------------------------------- #
# Stage 3: exact k-nearest neighbors (Metal GPU, deterministic)
# --------------------------------------------------------------------------- #
def knn_exact(X: np.ndarray, k: int, chunk_size: int | None = None,
              metric: str = 'euclidean'):
    """Exact k-nearest neighbors via tiled pairwise distance on the Metal GPU.

    Excludes self. Deterministic (pure compute + argsort). The O(N²) operation is
    intended for inputs whose pairwise tiles fit the available memory budget.
    """
    import mlx.core as mx

    X = np.asarray(X, dtype=np.float32)
    if metric == 'cosine':
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-10)
    n = X.shape[0]
    Xm = mx.array(X)
    sumsq = mx.sum(Xm * Xm, axis=1)
    mx.eval(sumsq)
    if chunk_size is None:
        # Bound one [chunk x n] block by both the adaptive budget and Metal's
        # per-buffer limit; selection temporarily holds several block-sized arrays.
        block = _mlx_tile_bytes(2_000_000_000, _MLX_KNN_SIM_COPIES)
        chunk_size = min(n, max(1000, block // (n * 4)))

    knn_idx = np.empty((n, k), dtype=np.int32)
    knn_dist = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        Xc = Xm[start:end]
        D = mx.maximum(
            sumsq[start:end, None] + sumsq[None, :] - 2.0 * (Xc @ Xm.T), 0.0
        )
        ar = mx.arange(start, end)[:, None]
        al = mx.arange(n)[None, :]
        D = D + (ar == al).astype(mx.float32) * 1e30  # mask self
        # top-k via partition (O(n) per row) then sort only the k candidates,
        # instead of a full O(n log n) argsort of every row.
        part = mx.argpartition(D, kth=k, axis=1)[:, :k]
        dk = mx.take_along_axis(D, part, axis=1)
        srt = mx.argsort(dk, axis=1)
        order = mx.take_along_axis(part, srt, axis=1)
        dd = mx.take_along_axis(dk, srt, axis=1)
        mx.eval(order, dd)
        knn_idx[start:end] = np.array(order).astype(np.int32)
        knn_dist[start:end] = np.sqrt(np.maximum(np.array(dd), 0.0)).astype(np.float32)
    return knn_idx, knn_dist


def knn_pynndescent(X: np.ndarray, k: int, random_state: int = 42,
                    n_jobs: int = -1, metric: str = 'euclidean'):
    """Approximate k-nearest neighbors via pynndescent (the engine used by
    umap-learn). Excludes self. A fixed ``random_state`` controls the approximate
    search; reproducibility can also depend on the library version and execution
    environment. This is the default for the UMAP pipeline, while ``knn_exact``
    remains available when exact neighbors are required.
    """
    from pynndescent import NNDescent

    X = np.asarray(X, dtype=np.float32)
    index = NNDescent(
        X, n_neighbors=k + 1, metric=metric,
        random_state=random_state, n_jobs=n_jobs, verbose=False,
    )
    idx, dist = index.neighbor_graph  # column 0 is the point itself (dist 0)
    return idx[:, 1:].astype(np.int32), dist[:, 1:].astype(np.float32)


# --------------------------------------------------------------------------- #
# Stage 4: fuzzy simplicial set (vectorized NumPy/SciPy, deterministic)
# --------------------------------------------------------------------------- #
def fuzzy_simplicial_set(knn_indices, knn_dists, n: int, n_epochs: int):
    """Build the symmetric fuzzy simplicial set (UMAP graph) from kNN.

    Vectorized smooth-kNN (rho/sigma) + probabilistic symmetrization
    (``P = A + A^T - A*A^T``) + weak-edge pruning. Returns COO edges
    ``(rows, cols, weights)``.
    """
    from scipy.sparse import coo_matrix

    knn_indices = np.asarray(knn_indices)
    knn_dists = np.asarray(knn_dists, dtype=np.float64)
    k = knn_indices.shape[1]
    target = np.log2(k)

    # rho: distance to nearest non-zero neighbor (per row)
    masked = np.where(knn_dists > 0, knn_dists, np.inf)
    rho = masked.min(axis=1)
    rho[~np.isfinite(rho)] = 0.0

    # sigma: vectorized binary search so sum(exp(-(d-rho)/sigma)) == log2(k)
    dsh_tail = np.maximum(knn_dists - rho[:, None], 0.0)[:, 1:]  # skip nearest
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    mid = np.ones(n)
    for _ in range(64):
        psum = np.exp(-dsh_tail / mid[:, None]).sum(axis=1)
        too_high = psum > target
        hi = np.where(too_high, mid, hi)
        lo = np.where(~too_high, mid, lo)
        mid = np.where(
            too_high,
            (lo + hi) / 2.0,
            np.where(np.isinf(hi), mid * 2.0, (lo + hi) / 2.0),
        )
    sigma = np.maximum(mid, 1e-10)

    weights = np.exp(-np.maximum(knn_dists - rho[:, None], 0.0) / sigma[:, None])
    rows = np.repeat(np.arange(n), k)
    cols = knn_indices.ravel()
    vals = weights.ravel()

    A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    P = (A + A.T - A.multiply(A.T)).tocoo()

    thresh = P.data.max() / float(n_epochs)
    keep = P.data >= thresh
    return (
        P.row[keep].astype(np.int32),
        P.col[keep].astype(np.int32),
        P.data[keep].astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# Stage 5: deterministic SGD optimizer (Metal GPU) --- the reproducibility core
# --------------------------------------------------------------------------- #
def optimize_layout(rows, cols, weights, Y0, a, b, n, *,
                    n_epochs: int = 300, negative_sample_rate: int = 5,
                    learning_rate: float = 1.0, random_state: int = 42,
                    verbose: bool = False):
    """Deterministic UMAP SGD on the Metal GPU.

    Reproducibility is supported by drawing negatives from a seeded MLX RNG and
    avoiding atomic scatter-add. The ``negative_sample_rate`` contributions are
    pre-summed, folded into the head-node contribution, and accumulated with
    ``np.bincount``. Results can still depend on the MLX runtime and hardware.
    """
    import mlx.core as mx

    weights = np.asarray(weights, dtype=np.float32)
    max_w = float(weights.max())
    n_samples = n_epochs * (weights / max_w)
    eps = np.where(n_samples > 0, n_epochs / n_samples, -1.0)
    epn = eps.copy()
    active = []
    for ep in range(n_epochs):
        ac = np.where(epn <= ep)[0]
        if len(ac) > 0:
            epn[ac] += eps[ac]
            active.append(mx.array(ac.astype(np.int32)))
        else:
            active.append(None)

    rows_m = mx.array(np.asarray(rows, dtype=np.int32))
    cols_m = mx.array(np.asarray(cols, dtype=np.int32))
    a_m, b_m = mx.array(a), mx.array(b)
    mx.random.seed(random_state)
    Y = mx.array(np.asarray(Y0, dtype=np.float32))

    epoch_iter = range(n_epochs)
    if verbose:
        try:
            from tqdm import tqdm as _tqdm
            epoch_iter = _tqdm(epoch_iter, total=n_epochs,
                               desc="  UMAP", unit="epoch", ncols=80, mininterval=0)
        except ImportError:
            pass
    for ep in epoch_iter:
        am = active[ep]
        if am is None:
            continue
        ef = rows_m[am]
        et = cols_m[am]
        na = am.shape[0]
        alpha = learning_rate * (1.0 - ep / n_epochs)
        nti = mx.random.randint(0, n, (negative_sample_rate * na,))

        d = Y[ef] - Y[et]
        ds = mx.maximum(mx.sum(d * d, 1, keepdims=True), 1e-6)
        gc = -2.0 * a_m * b_m * mx.power(ds, b_m - 1.0) / (1.0 + a_m * mx.power(ds, b_m))
        pg = mx.clip(gc * d, -4.0, 4.0) * alpha

        yfrom = mx.tile(Y[ef], (negative_sample_rate, 1))
        nd = yfrom - Y[nti]
        nds = mx.maximum(mx.sum(nd * nd, 1, keepdims=True), 1e-6)
        ngc = 2.0 * b_m / ((0.001 + nds) * (1.0 + a_m * mx.power(nds, b_m)))
        ng = mx.clip(ngc * nd, -4.0, 4.0) * alpha
        negsum = mx.sum(ng.reshape(negative_sample_rate, na, 2), axis=0)
        ce = pg + negsum  # head-node gets positive + summed negative force

        mx.eval(ce, pg, ef, et)
        efn, etn = np.array(ef), np.array(et)
        cen, pn = np.array(ce), np.array(pg)
        idx = np.concatenate([efn, etn])
        gx = np.bincount(idx, np.concatenate([cen[:, 0], -pn[:, 0]]), minlength=n)
        gy = np.bincount(idx, np.concatenate([cen[:, 1], -pn[:, 1]]), minlength=n)
        Y = Y + mx.array(np.stack([gx, gy], 1).astype(np.float32))
        if (ep + 1) % 20 == 0:
            mx.eval(Y)

    mx.eval(Y)
    return np.array(Y, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Stage 6: orchestrator
# --------------------------------------------------------------------------- #
class MlxUMAP:
    """Deterministic UMAP on Apple Silicon.

    ``fit_transform`` accepts an HECTOR embedding of shape ``(n, 768)`` and
    reduces it to ``n_pcs`` dimensions before constructing the kNN graph and
    initialization.
    """

    def __init__(self, n_neighbors: int = 30, n_pcs: int = 50,
                 min_dist: float = 1.0, spread: float = 1.0,
                 n_epochs: int = 300, negative_sample_rate: int = 5,
                 learning_rate: float = 1.0, random_state: int = 42,
                 chunk_size: int | None = None, metric: str = 'cosine',
                 verbose: bool = False):
        self.n_neighbors = n_neighbors
        self.n_pcs = n_pcs
        self.min_dist = min_dist
        self.spread = spread
        self.n_epochs = n_epochs
        self.negative_sample_rate = negative_sample_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.chunk_size = chunk_size
        self.metric = metric
        self.verbose = verbose

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        if self.n_pcs and X.shape[1] > self.n_pcs:
            from sklearn.decomposition import PCA
            X_pca = PCA(
                n_components=self.n_pcs, svd_solver="randomized",
                random_state=self.random_state,
            ).fit_transform(X).astype(np.float32)
        else:
            X_pca = X
        knn_i, knn_d = knn_pynndescent(
            X_pca, self.n_neighbors, random_state=self.random_state,
            metric=self.metric,
        )
        rows, cols, w = fuzzy_simplicial_set(knn_i, knn_d, n, self.n_epochs)
        a, b = find_ab_params(self.spread, self.min_dist)
        Y0 = pca_init(X_pca, self.random_state)
        return optimize_layout(
            rows, cols, w, Y0, a, b, n,
            n_epochs=self.n_epochs,
            negative_sample_rate=self.negative_sample_rate,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            verbose=self.verbose,
        )


# =========================================================================== #
# t-SNE engine (Metal GPU)
# =========================================================================== #

def _conditional_P(knn_d2, perplexity, n_iter=100):
    """Per-row Gaussian P(j|i) from squared kNN distances; vectorized binary
    search on precision beta so each row's entropy matches log(perplexity)."""
    n, k = knn_d2.shape
    target = np.log(perplexity)
    beta = np.ones(n); lo = np.full(n, -np.inf); hi = np.full(n, np.inf)
    d2 = knn_d2.astype(np.float64)
    for _ in range(n_iter):
        Pw = np.exp(-d2 * beta[:, None])
        sumP = Pw.sum(axis=1) + 1e-12
        H = np.log(sumP) + beta * (d2 * Pw).sum(axis=1) / sumP
        too_high = (H - target) > 0.0
        lo = np.where(too_high, beta, lo)
        hi = np.where(~too_high, beta, hi)
        beta = np.where(too_high,
                        np.where(np.isinf(hi), beta * 2.0, (beta + hi) / 2.0),
                        np.where(np.isinf(lo), beta / 2.0, (beta + lo) / 2.0))
    Pw = np.exp(-d2 * beta[:, None])
    Pw /= (Pw.sum(axis=1, keepdims=True) + 1e-12)
    return Pw.astype(np.float64)


def build_symmetric_P(knn_idx, knn_d, perplexity, n, kmax_cap=256):
    """Symmetric normalized affinities as a padded per-node neighbour list
    (nbr_idx, P_pad, cap), capped to top-weight neighbours (vectorized)."""
    from scipy.sparse import coo_matrix
    knn_idx = np.asarray(knn_idx); k = knn_idx.shape[1]
    Pcond = _conditional_P(np.asarray(knn_d, dtype=np.float64) ** 2, perplexity)
    rows = np.repeat(np.arange(n), k)
    A = coo_matrix((Pcond.ravel(), (rows, knn_idx.ravel())), shape=(n, n)).tocsr()
    P = ((A + A.T) * (1.0 / (2.0 * n))).tocsr()
    indptr, indices, data = P.indptr, P.indices, P.data.astype(np.float32)
    degrees = np.diff(indptr); nnz = indices.shape[0]
    cap = int(min(degrees.max(), kmax_cap))
    row_id = np.repeat(np.arange(n), degrees)
    order = np.lexsort((-data, row_id))
    col_pos = np.arange(nnz) - np.repeat(indptr[:-1], degrees)
    keep = col_pos < cap
    o = order[keep]; rk = row_id[o]; ck = col_pos[keep]
    nbr_idx = np.tile(np.arange(n, dtype=np.int32)[:, None], (1, cap))
    P_pad = np.zeros((n, cap), dtype=np.float32)
    nbr_idx[rk, ck] = indices[o]; P_pad[rk, ck] = data[o]
    return nbr_idx, P_pad, cap


def tsne_pca(X, n_components, random_state=0):
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, svd_solver="randomized",
               random_state=random_state).fit_transform(X).astype(np.float32)


def tsne_pca_init(X50, random_state=0):
    """First 2 PCs scaled so std(PC1)=1e-4 (sklearn t-SNE init convention)."""
    Y = np.asarray(X50[:, :2], dtype=np.float64).copy()
    Y = Y / (Y[:, 0].std() + 1e-12) * 1e-4
    return Y.astype(np.float32)


def tsne_knn(X, k, random_state=0, approx_threshold=50000):
    """Use exact Metal kNN up to ``approx_threshold`` and pynndescent above it."""
    if X.shape[0] > approx_threshold:
        return knn_pynndescent(X, k, random_state=random_state)
    return knn_exact(X, k)


def tsne_repulsive_exact(Y, chunk_size):
    """Exact tiled repulsive force on Metal. Returns (F (n,2) unnormalized, Z)."""
    import mlx.core as mx
    n = Y.shape[0]; ysq = mx.sum(Y * Y, axis=1); Z = mx.array(0.0); parts = []
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n); Yc = Y[s:e]
        D2 = mx.maximum(ysq[s:e, None] + ysq[None, :] - 2.0 * (Yc @ Y.T), 0.0)
        W = 1.0 / (1.0 + D2)
        rid = mx.arange(s, e)[:, None]; cid = mx.arange(n)[None, :]
        W = W * (rid != cid).astype(mx.float32)
        Z = Z + mx.sum(W); W2 = W * W
        parts.append(mx.sum(W2, axis=1, keepdims=True) * Yc - W2 @ Y)
    return mx.concatenate(parts, axis=0), Z


def _lagrange_weights(s, n_interp):
    s = np.asarray(s, dtype=np.float64)
    W = np.ones((s.shape[0], n_interp), dtype=np.float64)
    for j in range(n_interp):
        for m in range(n_interp):
            if m != j:
                W[:, j] *= (s - m) / (j - m)
    return W


def _tsne_axis_setup(coord, lo, box_width, n_boxes, n_interp):
    h = box_width / n_interp
    rel = (coord - lo) / box_width
    box = np.clip(np.floor(rel).astype(np.int64), 0, n_boxes - 1)
    s = (coord - lo) / h - box * n_interp - 0.5
    return h, box * n_interp, _lagrange_weights(s, n_interp)


def tsne_repulsive_fft(Y_mx, n_interp=3, target_box_width=1.0,
                       min_boxes=50, max_boxes=680, return_grid_info=False):
    """GPU-resident FIt-SNE repulsive force. Returns (F (n,2) mx, Z float).
    Host: box assignment + deterministic np.bincount charge scatter. Metal:
    kernel build, FFT convolution, gather, assembly. The FFT grid is padded to
    the next power of two to bound the Metal FFT working set."""
    import mlx.core as mx
    Y = np.array(Y_mx, dtype=np.float64)
    n = Y.shape[0]
    lo = float(Y.min()); hi = float(Y.max())
    span = hi - lo; pad = span * 0.01 + 1e-6; lo -= pad; hi += pad; span = hi - lo
    nb = int(np.clip(np.ceil(span / target_box_width), min_boxes, max_boxes))
    bw = span / nb
    h, basex, Wx = _tsne_axis_setup(Y[:, 0], lo, bw, nb, n_interp)
    _, basey, Wy = _tsne_axis_setup(Y[:, 1], lo, bw, nb, n_interp)
    nt = nb * n_interp
    L = 1
    while L < 2 * nt:
        L *= 2
    ii, jj = np.meshgrid(np.arange(n_interp), np.arange(n_interp), indexing="ij")
    ii = ii.ravel(); jj = jj.ravel()
    gx = basex[:, None] + ii[None, :]
    gy = basey[:, None] + jj[None, :]
    w = Wx[:, ii] * Wy[:, jj]
    flat = (gx * nt + gy).ravel()
    charges = np.stack([np.ones(n), Y[:, 0], Y[:, 1], np.ones(n)], axis=1)
    ml = nt * nt
    grids = np.empty((4, ml), dtype=np.float32)
    for c in range(4):
        grids[c] = np.bincount(flat, weights=(w * charges[:, c:c + 1]).ravel(),
                               minlength=ml).astype(np.float32)
    g = mx.array(grids.reshape(4, nt, nt))
    idx = mx.arange(L)
    off = ((idx + nt) % L) - nt
    ax = off.astype(mx.float32) * h
    AX = ax[:, None] ** 2 + ax[None, :] ** 2
    r1 = 1.0 / (1.0 + AX)
    maskf = (idx != nt).astype(mx.float32)
    M2 = maskf[:, None] * maskf[None, :]
    K0 = mx.fft.fft2(r1 * M2)
    K1 = mx.fft.fft2(r1 * r1 * M2)
    paddec = mx.zeros((4, L, L)); paddec[:, :nt, :nt] = g
    Fhat = mx.fft.fft2(paddec, axes=(1, 2))
    Kst = mx.stack([K1, K1, K1, K0], axis=0)
    pot = mx.fft.ifft2(Fhat * Kst, axes=(1, 2)).real[:, :nt, :nt].reshape(4, -1)
    flat_mx = mx.array(flat.astype(np.int32))
    w_mx = mx.array(w.astype(np.float32))
    P = mx.take(pot, flat_mx, axis=1).reshape(4, n, n_interp * n_interp)
    vals = mx.sum(P * w_mx[None], axis=2)
    A, Bx, By, Zpot = vals[0], vals[1], vals[2], vals[3]
    Frep = mx.stack([Y_mx[:, 0] * A - Bx, Y_mx[:, 1] * A - By], axis=1)
    Z = mx.sum(Zpot) - n
    mx.eval(Frep, Z)
    if return_grid_info:
        return Frep, float(Z), dict(n_total=nt)
    return Frep, float(Z)


def tsne_attractive_force(Y, nbr, Pw, exag, chunk=None):
    """Sparse attractive force on MLX (gather+reduce, no scatter). chunk tiles
    rows so peak memory is O(chunk*kmax); identical math either way."""
    import mlx.core as mx
    n = Y.shape[0]
    if chunk is None or chunk >= n:
        diff = Y[:, None, :] - Y[nbr]; d2 = mx.sum(diff * diff, axis=2)
        coef = (Pw * exag) / (1.0 + d2)
        return mx.sum(coef[:, :, None] * diff, axis=1)
    parts = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        diff = Y[s:e, None, :] - Y[nbr[s:e]]; d2 = mx.sum(diff * diff, axis=2)
        coef = (Pw[s:e] * exag) / (1.0 + d2)
        parts.append(mx.sum(coef[:, :, None] * diff, axis=1))
    return mx.concatenate(parts, axis=0)


def tsne_estimate_floor_gb(n, n_input_dims, kmax):
    """Estimate the host-and-device working set in GB.

    The estimate includes the input, PCA representation, kNN graph, padded
    probabilities, and a safety margin. Per-iteration attractive-force
    temporaries are excluded because tiling bounds them.
    """
    b = (n * n_input_dims * 4 + n * 50 * 4 + n * 90 * 8 + n * kmax * 8)
    return b / 1e9 * 1.6


# --------------------------------------------------------------------------- #
# Unified MLX/Metal memory budget                                             #
# --------------------------------------------------------------------------- #
# MLX operations use mlx_gpu_budget() to account for current unified-memory
# pressure instead of relying on a static allocation ceiling.

# Fraction of total unified memory reserved for the host process and operating
# system. The environment variable allows deployment-specific adjustment.
_MLX_HEADROOM_FRACTION = float(
    os.environ.get("HECTOR_MLX_HEADROOM_FRACTION",
                   os.environ.get("HECTOR_METAL_HEADROOM_FRACTION", "0.10"))
)

# GRIT k-NN sizing accounts for simultaneously live similarity and selection
# buffers. Candidate tiles also respect Metal's per-buffer limit and bound the
# host-side dense tile. Both values are configurable through environment variables.
_MLX_KNN_SIM_COPIES = int(os.environ.get("HECTOR_MLX_KNN_COPIES", "8"))
_MLX_KNN_CAND_TILE = int(os.environ.get("HECTOR_MLX_KNN_CAND_TILE", "8192"))


def mlx_device_limits() -> Optional[Tuple[int, int, int]]:
    """``(working_set, max_buffer, total)`` bytes from ``mx.device_info()``.

    Returns ``None`` when MLX/Metal is unavailable so callers can use their
    fallback budgets.

    * ``working_set`` -- ``max_recommended_working_set_size``; Metal's soft cap
      on one process's GPU working set (exceeding it risks the OOM abort).
    * ``max_buffer`` -- ``max_buffer_length``; the HARD cap on any single buffer.
    * ``total`` -- ``memory_size``; total unified RAM.
    """
    try:
        import mlx.core as mx
        di = mx.device_info()
        return (
            int(di["max_recommended_working_set_size"]),
            int(di["max_buffer_length"]),
            int(di["memory_size"]),
        )
    except Exception:
        return None


def mlx_device_name() -> Optional[str]:
    """Marketing name of the Metal GPU, e.g. ``"Apple M1 Max"``.

    Reads ``device_name`` from the same ``mx.device_info()`` dict as
    :func:`mlx_device_limits`. Returns ``None`` off-Metal, or when the field is
    missing, so callers can fall back to a generic label. Cosmetic only -- it
    feeds the model-load log line that tells users which hardware will run the
    prediction.
    """
    try:
        import mlx.core as mx
        return mx.device_info().get("device_name") or None
    except Exception:
        return None


def mlx_max_alloc(safety: float = 0.90) -> Optional[int]:
    """Cap in bytes on any SINGLE ``mx.array`` == ``safety * max_buffer_length``.

    Metal refuses a single buffer larger than ``max_buffer_length``, so callers
    shrink a tile when its largest array would exceed this cap even if the total
    memory budget would fit. Returns ``None`` off-Metal.
    """
    lim = mlx_device_limits()
    if lim is None:
        return None
    return int(lim[1] * safety)


def _mlx_budget_bytes(
    working_set, total, active, cached, free_now, op_host_bytes, headroom_frac,
) -> int:
    """Compute the allocation budget used by :func:`mlx_gpu_budget`.

    Two hard limits; the tighter one wins:

    * **Metal working set** -- MLX-resident memory (``active + cached``) must stay
      under ``working_set``, so new-allocation room is
      ``working_set - active - cached``.
    * **Physical unified RAM** -- the allocation needs real pages. What is
      available is the macOS-free reading plus MLX's own reusable cache, minus a
      headroom the host + OS keep: ``free_now + cached - total*headroom_frac``.

    ``free_now`` (Mach ``host_statistics64``) already nets out every resident
    consumer -- the model, the anchor bank, other apps -- so this covers them
    without enumeration. When ``free_now`` is ``None`` (no Mach reading) only the
    Metal-side room is known; halve it for safety. ``op_host_bytes`` is the op's
    own newly allocated host arrays, not yet reflected in ``free_now``. The
    result is floored at zero.
    """
    metal_room = working_set - active - cached
    if free_now is None:
        return max(int(metal_room * 0.5) - op_host_bytes, 0)
    phys_room = (free_now + cached) - int(total * headroom_frac)
    return max(int(min(metal_room, phys_room)) - op_host_bytes, 0)


def mlx_gpu_budget(op_host_bytes: int = 0,
                   headroom_frac: Optional[float] = None) -> Optional[int]:
    """Estimate a conservative unified-memory budget for the next MLX operation.

    Reads the Metal limits, MLX's current ``active``/``cache`` footprint, and the
    live macOS-free reading ONCE (call at op entry, before allocating), then
    returns ``min(Metal-working-set room, physical-RAM room) - op_host_bytes``
    via :func:`_mlx_budget_bytes`. The result is an operational estimate because
    allocator reclamation and the system's free-memory reading can lag.

    Returns ``0`` when nothing fits (the caller should raise ``MemoryError`` /
    skip the op rather than let Metal abort the process), or ``None`` off-Metal.
    """
    lim = mlx_device_limits()
    if lim is None:
        return None
    working_set, _max_buffer, total = lim
    if headroom_frac is None:
        headroom_frac = _MLX_HEADROOM_FRACTION
    import mlx.core as mx
    active = int(mx.get_active_memory())
    cached = int(mx.get_cache_memory())
    from .predictor_support import _get_macos_available_bytes
    free_now = _get_macos_available_bytes()
    if free_now is not None:
        free_now = int(free_now)
    return _mlx_budget_bytes(
        working_set, total, active, cached, free_now, op_host_bytes, headroom_frac,
    )


def _mlx_tile_bytes(old_cap: int, copies: int) -> int:
    """Return the byte target for one transient tiled GPU block.

    The target does not exceed the per-buffer allocation limit and is reduced
    when the current adaptive budget cannot hold ``copies`` simultaneous
    blocks. Off Metal, the supplied base cap is returned unchanged.
    """
    block = old_cap
    mb = mlx_max_alloc()
    if mb is not None:
        block = min(block, mb)
    budget = mlx_gpu_budget()
    if budget is not None and budget < copies * block:
        block = int(budget // max(copies, 1))
    return max(int(block), 1)


def tsne_check_memory(n, n_input_dims, kmax, budget_gb=None):
    """Raise ``MemoryError`` when estimated demand exceeds the memory budget."""
    if budget_gb is None:
        b = mlx_gpu_budget()                     # free-aware unified budget (bytes)
        budget_gb = (b / 1e9) if b is not None else 8.0
    need = tsne_estimate_floor_gb(n, n_input_dims, kmax)
    if need > budget_gb:
        max_n = int(n * budget_gb / need)
        raise MemoryError(
            f"t-SNE on {n:,} cells needs ~{need:.1f} GB but only ~{budget_gb:.1f} GB "
            f"is available. Downsample to ~{max_n:,} cells or use a larger machine.")


class MlxTSNE:
    """Deterministic t-SNE on Apple Silicon (MLX). Auto-switches exact/FFT
    repulsive by N; tiles the attractive force; checks estimated memory demand.
    Companion to MlxUMAP. fit_transform(X) -> (n,2) float32."""

    def __init__(self, perplexity=30.0, n_pcs=50, n_iter=1000,
                 exaggeration_iter=250, early_exaggeration=12.0,
                 learning_rate="auto", random_state=0,
                 method="auto", method_threshold=7000, kmax_cap=256,
                 n_interp=3, target_box_width=1.0,
                 attr_chunk=None, memory_budget_gb=None, verbose=True):
        self.perplexity = perplexity; self.n_pcs = n_pcs; self.n_iter = n_iter
        self.exaggeration_iter = exaggeration_iter
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate; self.random_state = random_state
        self.method = method; self.method_threshold = method_threshold
        self.kmax_cap = kmax_cap; self.n_interp = n_interp
        self.target_box_width = target_box_width; self.attr_chunk = attr_chunk
        self.memory_budget_gb = memory_budget_gb; self.verbose = verbose

    def fit_transform(self, X):
        import mlx.core as mx
        X = np.asarray(X, dtype=np.float32); n = X.shape[0]
        tsne_check_memory(n, X.shape[1], self.kmax_cap, self.memory_budget_gb)
        X50 = (tsne_pca(X, self.n_pcs, self.random_state)
               if (self.n_pcs and X.shape[1] > self.n_pcs) else X)
        k = int(min(3 * self.perplexity, n - 1))
        knn_idx, knn_d = tsne_knn(X50, k, self.random_state)
        nbr_idx, P_pad, kmax = build_symmetric_P(knn_idx, knn_d, self.perplexity, n, self.kmax_cap)
        method = self.method
        if method == "auto":
            method = "exact" if n <= self.method_threshold else "fft"
        chunk = self.attr_chunk or max(2000, int(5e7 / max(kmax, 1)))
        ex_chunk = min(n, max(1000, _mlx_tile_bytes(1_500_000_000, 4) // (n * 4)))
        if self.verbose:
            print(f"  t-SNE: n={n} method={method} kmax={kmax} attr_chunk={chunk}")
        Y = mx.array(tsne_pca_init(X50, self.random_state))
        nbr = mx.array(nbr_idx); Pw = mx.array(P_pad)
        lr = self.learning_rate
        if lr == "auto":
            lr = max(n / self.early_exaggeration / 4.0, 50.0)
        update = mx.zeros((n, 2)); gains = mx.ones((n, 2)); min_gain = 0.01
        iter_range = range(self.n_iter)
        if self.verbose:
            try:
                from tqdm import tqdm as _tqdm
                iter_range = _tqdm(iter_range, total=self.n_iter,
                                   desc="  t-SNE", unit="iter", ncols=80, mininterval=0)
            except ImportError:
                pass
        for it in iter_range:
            momentum = 0.5 if it < 250 else 0.8
            exag = self.early_exaggeration if it < self.exaggeration_iter else 1.0
            F_attr = tsne_attractive_force(Y, nbr, Pw, exag, chunk)
            if method == "exact":
                Frep, Z = tsne_repulsive_exact(Y, ex_chunk)
            else:
                Frep, Z = tsne_repulsive_fft(Y, n_interp=self.n_interp,
                                             target_box_width=self.target_box_width)
            grad = 4.0 * (F_attr - Frep / Z)
            inc = (update * grad) < 0.0
            gains = mx.maximum(mx.where(inc, gains + 0.2, gains * 0.8), min_gain)
            update = momentum * update - lr * gains * grad
            Y = Y + update
            mx.eval(Y, update, gains)
        mx.eval(Y)
        return np.array(Y, dtype=np.float32)


# =========================================================================== #
# Encoder --- DualHeadVGAEEncoder (Metal GPU)
# =========================================================================== #

# --------------------------------------------------------------------------- #
# EdgeIndexGAT --- additive attention (Velickovic 2017)
# --------------------------------------------------------------------------- #

class MlxEdgeIndexGAT:
    """Graph Attention layer with additive attention and edge-index input."""

    def __init__(
        self,
        kernel,           # [F_in, units * num_heads]
        attn_kernel_self,  # [num_heads, units]
        attn_kernel_neigh, # [num_heads, units]
        bias,              # [output_dim] or None
        num_heads: int,
        units: int,
        concat_heads: bool = False,
        activation: str = "relu",
    ):
        self.kernel = kernel
        self.attn_kernel_self = attn_kernel_self
        self.attn_kernel_neigh = attn_kernel_neigh
        self.bias = bias
        self.num_heads = num_heads
        self.units = units
        self.concat_heads = concat_heads
        self.activation = activation

    def __call__(
        self, node_features, edge_index,
        batch_size: int,
    ):
        import mlx.core as mx
        N = node_features.shape[0]

        # [N, F_in] @ [F_in, units*heads] -> [N, units*heads] -> [N, heads, units]
        ft = (node_features @ self.kernel).reshape(N, self.num_heads, self.units)

        edge_sources = edge_index[0]  # [E]
        edge_targets = edge_index[1]  # [E]

        feat_src = ft[edge_sources]   # [E, heads, units]
        feat_tgt = ft[edge_targets]   # [E, heads, units]

        # Additive attention logits: sum(src * a_self, -1) + sum(tgt * a_neigh, -1)
        attn_src = mx.sum(feat_src * self.attn_kernel_self, axis=-1)  # [E, heads]
        attn_tgt = mx.sum(feat_tgt * self.attn_kernel_neigh, axis=-1)
        attn_logits = _mx_leaky_relu(attn_src + attn_tgt)

        num_edges = edge_index.shape[1]
        k = num_edges // batch_size

        # [batch_size, k, heads]
        attn_logits = attn_logits.reshape(batch_size, k, self.num_heads)
        attn_coeffs = mx.softmax(attn_logits, axis=1)

        # [batch_size, k, heads, units]
        feat_tgt = feat_tgt.reshape(batch_size, k, self.num_heads, self.units)
        weighted = mx.expand_dims(attn_coeffs, axis=-1) * feat_tgt
        aggregated = mx.sum(weighted, axis=1)  # [batch_size, heads, units]

        if self.concat_heads:
            output = aggregated.reshape(batch_size, self.num_heads * self.units)
        else:
            output = mx.mean(aggregated, axis=1)  # [batch_size, units]

        if self.bias is not None:
            output = output + self.bias

        if self.activation == "relu":
            output = mx.maximum(output, 0.0)

        return output


# --------------------------------------------------------------------------- #
# EdgeIndexTransformer --- dot-product attention (pre-norm residual)
# --------------------------------------------------------------------------- #

class MlxEdgeIndexTransformer:
    """Graph Transformer layer with dot-product multi-head attention."""

    def __init__(
        self,
        ln_gamma,    # [units]
        ln_beta,     # [units]
        q_kernel,    # [F_in, num_heads, key_dim]
        q_bias,      # [num_heads, key_dim]
        k_kernel,    # [F_in, num_heads, key_dim]
        k_bias,      # [num_heads, key_dim]
        v_kernel,    # [F_in, num_heads, key_dim]
        v_bias,      # [num_heads, key_dim]
        o_kernel,    # [num_heads, key_dim, F_out]
        o_bias,      # [F_out]
        bias,        # [units]
        units: int,
        num_heads: int,
        activation: str = "relu",
    ):
        self.ln_gamma = ln_gamma
        self.ln_beta = ln_beta
        self.q_kernel = q_kernel
        self.q_bias = q_bias
        self.k_kernel = k_kernel
        self.k_bias = k_bias
        self.v_kernel = v_kernel
        self.v_bias = v_bias
        self.o_kernel = o_kernel
        self.o_bias = o_bias
        self.bias = bias
        self.units = units
        self.num_heads = num_heads
        self.key_dim = units // num_heads
        self.activation = activation
        self._scale = self.key_dim ** -0.5

    def __call__(
        self, node_features, edge_index,
        batch_size: int,
    ):
        import mlx.core as mx
        # 1. Pre-norm all nodes
        nf_norm = _mx_layer_norm(node_features, self.ln_gamma, self.ln_beta)

        # 2. Query = first batch_size nodes (normalized)
        h_query_norm = nf_norm[:batch_size]          # [B, F]
        h_query = node_features[:batch_size]          # [B, F] --- for residual

        # 3. Gather normalized neighbor features
        num_edges = edge_index.shape[1]
        k = num_edges // batch_size
        neighbor_indices = edge_index[1]              # [E]
        h_neighbors = nf_norm[neighbor_indices].reshape(batch_size, k, -1)  # [B, k, F]

        # 4. Multi-head attention
        # Q: [B, 1, F] -> [B, 1, H, D] via einsum "bsf,fhd->bshd"
        q = mx.expand_dims(h_query_norm, 1)           # [B, 1, F]
        Q = mx.einsum("bsf,fhd->bshd", q, self.q_kernel) + self.q_bias  # [B, 1, H, D]

        # K, V: [B, k, F] -> [B, k, H, D]
        K = mx.einsum("bkf,fhd->bkhd", h_neighbors, self.k_kernel) + self.k_bias
        V = mx.einsum("bkf,fhd->bkhd", h_neighbors, self.v_kernel) + self.v_bias

        # Scaled dot-product: [B, H, 1, D] @ [B, H, D, k] -> [B, H, 1, k]
        Q_t = mx.transpose(Q, axes=(0, 2, 1, 3))     # [B, H, 1, D]
        K_t = mx.transpose(K, axes=(0, 2, 3, 1))     # [B, H, D, k]
        attn = (Q_t @ K_t) * self._scale              # [B, H, 1, k]
        attn = mx.softmax(attn, axis=-1)

        V_t = mx.transpose(V, axes=(0, 2, 1, 3))     # [B, H, k, D]
        out = attn @ V_t                               # [B, H, 1, D]

        # Reshape: [B, H, 1, D] -> [B, 1, H, D]
        out = mx.transpose(out, axes=(0, 2, 1, 3))
        # Output projection: [B, 1, H, D] -> [B, 1, F]
        attn_out = mx.einsum("bshd,hdf->bsf", out, self.o_kernel) + self.o_bias
        attn_out = attn_out.squeeze(axis=1)            # [B, F]

        # 5. Residual connection
        output = h_query + attn_out

        # 6. Bias + activation
        if self.bias is not None:
            output = output + self.bias

        if self.activation == "relu":
            output = mx.maximum(output, 0.0)

        return output


# --------------------------------------------------------------------------- #
# DualHeadVGAEEncoder --- full forward pass
# --------------------------------------------------------------------------- #

class MlxDualHeadVGAEEncoder:
    """MLX inference-only encoder matching TF DualHeadVGAEEncoder."""

    def __init__(
        self,
        hidden_layers: list,
        attention_layer,
        class_mu_kernel,
        class_mu_bias,
        class_logvar_kernel,
        class_logvar_bias,
    ):
        self.hidden_layers = hidden_layers
        self.attention_layer = attention_layer
        self.class_mu_kernel = class_mu_kernel
        self.class_mu_bias = class_mu_bias
        self.class_logvar_kernel = class_logvar_kernel
        self.class_logvar_bias = class_logvar_bias
        self.latent_dim = int(class_mu_kernel.shape[1])

    def encode_hidden(self, gene_values):
        """Per-node hidden-layer stack: Dense(ReLU) -> LayerNorm (dropout is a
        no-op at inference). Each row is transformed independently, so

            encode_hidden([batch; anchors]) == [encode_hidden(batch); encode_hidden(anchors)]

        which lets callers compute the anchors' hidden states once and reuse them
        every batch instead of re-running the hidden layers on the anchors.
        """
        import mlx.core as mx
        h = gene_values
        for kernel, bias, ln_gamma, ln_beta in self.hidden_layers:
            h = h @ kernel + bias
            h = mx.maximum(h, 0.0)  # ReLU
            h = _mx_layer_norm(h, ln_gamma, ln_beta)
        return h

    def encode_from_hidden(self, h, edge_index, batch_size: int) -> tuple:
        """Graph attention + classification head, given hidden node features
        ``h`` of shape ``[batch + n_anchors, units]``. Returns (mu, log_var)."""
        import mlx.core as mx
        h_out = self.attention_layer(h, edge_index, batch_size)
        mu_class = h_out @ self.class_mu_kernel + self.class_mu_bias
        log_var_class = h_out @ self.class_logvar_kernel + self.class_logvar_bias
        log_var_class = mx.clip(log_var_class, -10.0, 10.0)
        return mu_class, log_var_class

    def __call__(
        self,
        gene_values,
        edge_index,
        batch_size: int,
    ) -> tuple:
        """Forward pass. Returns (mu_class, log_var_class)."""
        return self.encode_from_hidden(
            self.encode_hidden(gene_values), edge_index, batch_size,
        )


# --------------------------------------------------------------------------- #
# Weight loader
# --------------------------------------------------------------------------- #

def load_mlx_encoder_from_h5(
    path: str | Path,
) -> tuple[MlxDualHeadVGAEEncoder, dict]:
    """Load an MLX encoder from a HECTOR HDF5 checkpoint.

    Returns (encoder, config_dict). The encoder is ready to call.
    Only loads the shared encoder + classification head weights.
    """
    import mlx.core as mx
    import h5py

    path = Path(path).expanduser()

    with h5py.File(path, "r") as f:
        config = json.loads(f["config"].attrs["model_config"])

        enc_grp = f["weights/vgae_encoder"]
        is_dual_head = enc_grp.attrs.get("is_dual_head", False)

        if is_dual_head and "shared_encoder" in enc_grp:
            sw = _load_weight_list(enc_grp["shared_encoder"])
            cw = _load_weight_list(enc_grp["class_head"])
        else:
            all_w = _load_weight_list(enc_grp)
            raise NotImplementedError(
                "Single-head encoder loading not yet supported"
            )

    hidden_dims = config.get("hidden_dims") or []
    graph_attention_type = config.get("graph_attention_type", "transformer")
    num_attention_heads = config.get("num_attention_heads", 4)
    shared_dim = hidden_dims[-1] if hidden_dims else config.get("shared_dim", 1024)
    activation = config.get("activation", "relu")

    # --- Map shared weights to layers ---
    idx = 0
    hidden_layers = []
    for _ in hidden_dims:
        kernel = mx.array(sw[idx])       # Dense kernel
        bias = mx.array(sw[idx + 1])     # Dense bias
        ln_gamma = mx.array(sw[idx + 2]) # LayerNorm gamma
        ln_beta = mx.array(sw[idx + 3])  # LayerNorm beta
        hidden_layers.append((kernel, bias, ln_gamma, ln_beta))
        idx += 4

    # --- Attention layer ---
    if graph_attention_type == "transformer":
        effective_num_heads = num_attention_heads
        while shared_dim % effective_num_heads != 0 and effective_num_heads > 1:
            effective_num_heads -= 1

        # Transformer weights: bias(1) + LN(2) + MHA_Q(2) + MHA_K(2) + MHA_V(2) + MHA_O(2) = 11
        t_bias = mx.array(sw[idx])           # idx+0: transformer output bias
        t_ln_gamma = mx.array(sw[idx + 1])   # idx+1: pre-norm LN gamma
        t_ln_beta = mx.array(sw[idx + 2])    # idx+2: pre-norm LN beta
        q_kernel = mx.array(sw[idx + 3])     # idx+3: Q kernel [F, H, D]
        q_bias = mx.array(sw[idx + 4])       # idx+4: Q bias [H, D]
        k_kernel = mx.array(sw[idx + 5])     # idx+5: K kernel [F, H, D]
        k_bias = mx.array(sw[idx + 6])       # idx+6: K bias [H, D]
        v_kernel = mx.array(sw[idx + 7])     # idx+7: V kernel [F, H, D]
        v_bias = mx.array(sw[idx + 8])       # idx+8: V bias [H, D]
        o_kernel = mx.array(sw[idx + 9])     # idx+9: output kernel [H, D, F]
        o_bias = mx.array(sw[idx + 10])      # idx+10: output bias [F]
        idx += 11

        attention_layer = MlxEdgeIndexTransformer(
            ln_gamma=t_ln_gamma, ln_beta=t_ln_beta,
            q_kernel=q_kernel, q_bias=q_bias,
            k_kernel=k_kernel, k_bias=k_bias,
            v_kernel=v_kernel, v_bias=v_bias,
            o_kernel=o_kernel, o_bias=o_bias,
            bias=t_bias,
            units=shared_dim,
            num_heads=effective_num_heads,
            activation=activation,
        )

    elif graph_attention_type == "gat":
        # GAT weights: kernel(1) + attn_self(1) + attn_neigh(1) + bias(1) = 4
        gat_kernel = mx.array(sw[idx])
        gat_attn_self = mx.array(sw[idx + 1])
        gat_attn_neigh = mx.array(sw[idx + 2])
        gat_bias = mx.array(sw[idx + 3]) if (idx + 3) < len(sw) else None
        idx += 4 if gat_bias is not None else 3

        units_per_head = shared_dim // num_attention_heads

        attention_layer = MlxEdgeIndexGAT(
            kernel=gat_kernel,
            attn_kernel_self=gat_attn_self,
            attn_kernel_neigh=gat_attn_neigh,
            bias=gat_bias,
            num_heads=num_attention_heads,
            units=units_per_head,
            concat_heads=False,
            activation=activation,
        )
    else:
        raise ValueError(f"Unknown graph_attention_type: {graph_attention_type}")

    assert idx == len(sw), (
        f"Weight count mismatch: consumed {idx}, have {len(sw)} shared weights"
    )

    # --- Classification head ---
    class_mu_kernel = mx.array(cw[0])
    class_mu_bias = mx.array(cw[1])
    class_logvar_kernel = mx.array(cw[2])
    class_logvar_bias = mx.array(cw[3])

    encoder = MlxDualHeadVGAEEncoder(
        hidden_layers=hidden_layers,
        attention_layer=attention_layer,
        class_mu_kernel=class_mu_kernel,
        class_mu_bias=class_mu_bias,
        class_logvar_kernel=class_logvar_kernel,
        class_logvar_bias=class_logvar_bias,
    )

    return encoder, config


# =========================================================================== #
# k-NN graph construction (Metal GPU)
# =========================================================================== #

def build_knn_graph_cells_mlx(
    X_normalized,
    k: int,
    query_indices: Optional[np.ndarray] = None,
    chunk_size: Optional[int] = None,
    cand_chunk: Optional[int] = None,
    silent: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cell-to-cell cosine kNN using MLX Metal GPU, streamed from sparse.

    The expression matrix is kept **sparse** on the
    host and only small query/candidate tiles are densified and pushed to the
    Metal device. The full dense matrix is never materialized -- neither on the
    host nor on the GPU.

    Each query remains in the candidate pool, producing the implicit self-edge
    used by the GRIT attention contract. No additional explicit self-edge is
    appended.

    Each dense tile is independently L2-normalized with ``np.linalg.norm``.
    The candidate set is
    streamed in tiles (width ``_MLX_KNN_CAND_TILE``) and a running top-k is merged
    across them. Floating-point accumulation and near-tie ordering can vary from
    a single-matmul implementation.

    Args:
        X_normalized: [n_cells, n_genes] gene expression as a scipy.sparse CSR
            matrix (preferred) or a dense ndarray. L2-normalized internally for
            cosine similarity. Not mutated.
        k: Number of neighbours per query.
        query_indices: Optional int array of row indices into X to query.
            When provided only these rows are queries (all rows are
            candidates).  When None all rows are queries.
        chunk_size: Query tile size (auto-sized from ``mlx_gpu_budget`` if None).
        cand_chunk: Candidate tile size (auto-sized if None: capped at
            ``_MLX_KNN_CAND_TILE`` to bound the similarity buffer + host densify).
        silent: Suppress the progress bar.

    Returns:
        edge_index: [2, num_edges] int32 numpy array.
        edge_weights: [num_edges] float32 numpy array (all ones).
    """
    import mlx.core as mx

    # Keep the matrix sparse on the host (CSR); densify only per tile below.
    if sp.issparse(X_normalized):
        Xcsr = X_normalized.tocsr()
        if Xcsr.dtype != np.float32:
            Xcsr = Xcsr.astype(np.float32)
    else:
        Xcsr = sp.csr_matrix(np.asarray(X_normalized, dtype=np.float32))
    n_total, n_features = Xcsr.shape

    k = min(k, n_total - 1)

    # Determine query rows.
    if query_indices is not None:
        query_row_indices = np.asarray(query_indices, dtype=np.int32)
    else:
        query_row_indices = np.arange(n_total, dtype=np.int32)
    n_queries = len(query_row_indices)

    if n_queries == 0 or k <= 0:
        return (
            np.zeros((2, 0), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    def _to_mx_normalized(dense_blk):
        """L2-normalize a dense tile and transfer it to the Metal device.

        Normalization is performed independently per row using a minimum norm
        of ``1e-10``.
        """
        norms = np.linalg.norm(dense_blk, axis=1, keepdims=True)
        dense_blk = dense_blk / np.maximum(norms, 1e-10)
        return mx.array(dense_blk.astype(np.float32, copy=False))

    # Size tiles from the current Metal working set, MLX footprint, and available
    # unified memory. The resident CSR and other allocations are already reflected
    # in those readings.
    # The transient DEVICE peak per inner step is
    #     ALPHA*(qt*cc*4)  +  (qt+cc)*row_bytes
    # (MLX keeps ~7 live copies of the [qt x cc] cosine block; +one dense query
    # tile +one dense candidate tile). Cap the candidate tile cc so one similarity
    # buffer stays under max_buffer_length and the host .toarray() copy stays
    # small; then solve qt from the budget. If no minimal tile fits, raise
    # MemoryError so the caller can skip this refinement.
    row_bytes = n_features * 4
    MIN_TILE = 512
    ALPHA = _MLX_KNN_SIM_COPIES
    max_buffer = mlx_max_alloc() or (12 * 1024 ** 3)

    if cand_chunk is None:
        cand_chunk = min(n_total, _MLX_KNN_CAND_TILE)
    cand_chunk = max(1, min(int(cand_chunk), n_total))

    if chunk_size is None:
        # This call's own new host arrays: the edge list (index + weights) and the
        # resident dense candidate tile. free_now already covers the resident CSR.
        op_host = n_queries * k * 12 + cand_chunk * row_bytes
        budget = mlx_gpu_budget(op_host_bytes=op_host)
        if budget is None:                      # off-Metal safety net (shouldn't hit)
            budget = 4 * 1024 ** 3

        def _fit_qt(cc):
            per_q = ALPHA * cc * 4 + row_bytes           # sim copies + dense query row
            avail = budget - cc * row_bytes              # minus resident dense cand tile
            qt = int(avail / per_q) if avail > 0 else 0
            qt = min(qt, int(max_buffer / max(cc * 4, 1)))   # one sim buffer < max_buffer
            return qt

        chunk_size = _fit_qt(cand_chunk)
        # Under tight memory, shrink the candidate tile too before giving up.
        while chunk_size < MIN_TILE and cand_chunk > MIN_TILE:
            cand_chunk = max(MIN_TILE, cand_chunk // 2)
            chunk_size = _fit_qt(cand_chunk)
        if chunk_size < 1:
            raise MemoryError(
                f"GRIT k-NN on {n_total:,} cells x {n_features:,} genes cannot fit "
                f"even a minimal tile in ~{budget / 1e9:.1f} GB of free unified "
                f"memory. Downsample the dataset, free memory, or use a larger machine."
            )

    chunk_size = max(1, min(int(chunk_size), n_queries))
    cand_chunk = max(1, min(int(cand_chunk), n_total))

    all_sources = []
    all_targets = []

    # Advance progress over query-tile by candidate-tile work units.
    n_query_tiles = (n_queries + chunk_size - 1) // chunk_size
    n_cand_tiles = (n_total + cand_chunk - 1) // cand_chunk
    pbar = None
    if not silent:
        from tqdm import tqdm as _tqdm

        pbar = _tqdm(
            total=n_query_tiles * n_cand_tiles,
            desc="  kNN graph",
            unit="tile",
            ncols=80,
        )

    try:
        for q_start in range(0, n_queries, chunk_size):
            q_end = min(q_start + chunk_size, n_queries)
            q_idx = query_row_indices[q_start:q_end]
            tile_size = q_end - q_start

            # Query tile: densify + normalize only these rows on the fly.
            Q = _to_mx_normalized(Xcsr[q_idx].toarray())

            # Running top-k buffer.
            topk_vals = mx.full((tile_size, k), -1e30)
            topk_idxs = mx.zeros((tile_size, k), dtype=mx.int32)

            for c_start in range(0, n_total, cand_chunk):
                c_end = min(c_start + cand_chunk, n_total)
                # Candidate tile: densify + normalize only these rows on the fly.
                C = _to_mx_normalized(Xcsr[c_start:c_end].toarray())  # [cand_tile, features]

                # Cosine similarity block: [tile_size, cand_tile].
                sim = Q @ C.T

                # Deterministic tie-break: small negative bias by global index
                # (lower index wins ties, matches TF path).
                bias = mx.array(
                    -1e-7 * np.arange(c_start, c_end, dtype=np.float32)
                    / max(n_total, 1),
                )
                sim = sim + bias[None, :]

                # Top-k within this candidate tile.
                cand_k = min(k, c_end - c_start)
                # Use argpartition + sort for top-k.
                neg_sim = -sim  # negate so partition gives smallest (= largest sim)
                part_idx = mx.argpartition(neg_sim, kth=cand_k, axis=1)[:, :cand_k]
                part_vals = mx.take_along_axis(sim, part_idx, axis=1)
                srt = mx.argsort(-part_vals, axis=1)  # descending by sim
                tile_topk_local = mx.take_along_axis(part_idx, srt, axis=1)
                tile_topk_vals = mx.take_along_axis(part_vals, srt, axis=1)

                # Map to global candidate indices.
                tile_topk_global = tile_topk_local + c_start

                # Pad if cand_k < k.
                if cand_k < k:
                    pad_w = k - cand_k
                    tile_topk_vals = mx.concatenate(
                        [tile_topk_vals, mx.full((tile_size, pad_w), -1e30)],
                        axis=1,
                    )
                    tile_topk_global = mx.concatenate(
                        [tile_topk_global, mx.zeros((tile_size, pad_w), dtype=mx.int32)],
                        axis=1,
                    )

                # Merge with running buffer: concat along axis=1, re-select top-k.
                combined_vals = mx.concatenate([topk_vals, tile_topk_vals], axis=1)
                combined_idxs = mx.concatenate([topk_idxs, tile_topk_global], axis=1)

                # Re-select top-k from combined (2k columns).
                neg_cv = -combined_vals
                re_part = mx.argpartition(neg_cv, kth=k, axis=1)[:, :k]
                re_vals = mx.take_along_axis(combined_vals, re_part, axis=1)
                re_srt = mx.argsort(-re_vals, axis=1)
                re_order = mx.take_along_axis(re_part, re_srt, axis=1)

                topk_vals = mx.take_along_axis(combined_vals, re_order, axis=1)
                topk_idxs = mx.take_along_axis(combined_idxs, re_order, axis=1)

                mx.eval(topk_vals, topk_idxs)

                if pbar is not None:
                    pbar.update(1)

            # Emit edges for this query tile.
            topk_idxs_np = np.array(topk_idxs, dtype=np.int32)
            sources = np.repeat(q_idx, k)
            targets = topk_idxs_np.ravel()

            all_sources.append(sources)
            all_targets.append(targets)
    finally:
        if pbar is not None:
            pbar.close()

    edge_index = np.stack(
        [np.concatenate(all_sources), np.concatenate(all_targets)],
        axis=0,
    ).astype(np.int32)
    edge_weights = np.ones(edge_index.shape[1], dtype=np.float32)

    return edge_index, edge_weights


def build_knn_graph_mlx(batch_X, anchor_features, k=30,
                        anchor_raw_mx=None, anchor_norms=None):
    """Build k-NN graph via cosine similarity on MLX Metal GPU.
    Anchors-only candidates. Returns (edge_index [2, n_edges], edge_weights [n_edges]).

    Fold path (``anchor_raw_mx`` + ``anchor_norms`` given): the caller keeps ONE
    resident raw anchor matrix on the GPU plus a per-anchor L2-norm vector, and
    the cosine normalization is folded into each similarity tile
    (``(bx_norm @ raw[s:e].T) / norms[s:e]``) instead of materializing a separate
    normalized anchor matrix. Fallback (neither given): upload + normalize the
    anchors internally.
    """
    import mlx.core as mx
    B = batch_X.shape[0]
    bx = mx.array(np.asarray(batch_X, dtype=np.float32))
    bx_norm = bx / (mx.linalg.norm(bx, axis=1, keepdims=True) + 1e-10)
    _fold = anchor_raw_mx is not None
    if _fold:
        ax_cand = anchor_raw_mx                       # resident raw anchors
    else:
        ax = mx.array(np.asarray(anchor_features, dtype=np.float32))
        ax_cand = ax / (mx.linalg.norm(ax, axis=1, keepdims=True) + 1e-10)
    n_anchors = anchor_features.shape[0]
    chunk = min(n_anchors, max(1000, _mlx_tile_bytes(2_000_000_000, 3) // (B * 4)))
    topk_idx = np.empty((B, k), dtype=np.int32)
    topk_val = np.full((B, k), -1.0, dtype=np.float32)
    for start in range(0, n_anchors, chunk):
        end = min(start + chunk, n_anchors)
        if _fold:
            sim = (bx_norm @ ax_cand[start:end].T) / anchor_norms[start:end].T
        else:
            sim = bx_norm @ ax_cand[start:end].T
        mx.eval(sim)
        sim_np = np.array(sim)
        combined = np.concatenate([topk_val, sim_np], axis=1)
        combined_idx = np.concatenate([
            topk_idx,
            np.broadcast_to(np.arange(start, end, dtype=np.int32)[None, :], (B, end - start))
        ], axis=1)
        part = np.argpartition(-combined, k, axis=1)[:, :k]
        topk_val = np.take_along_axis(combined, part, axis=1)
        topk_idx = np.take_along_axis(combined_idx, part, axis=1)
    sort_order = np.argsort(-topk_val, axis=1)
    topk_idx = np.take_along_axis(topk_idx, sort_order, axis=1)
    topk_val = np.take_along_axis(topk_val, sort_order, axis=1)
    # Build edge_index with self-loops, matching TF convention
    sources_list = []
    targets_list = []
    for i in range(B):
        row_src = np.full(k + 1, i, dtype=np.int32)
        row_tgt = np.empty(k + 1, dtype=np.int32)
        row_tgt[:k] = topk_idx[i] + B
        row_tgt[k] = i  # self-loop
        sources_list.append(row_src)
        targets_list.append(row_tgt)
    sources = np.concatenate(sources_list)
    targets = np.concatenate(targets_list)
    edge_index = np.stack([sources, targets], axis=0)
    edge_weights = np.ones(edge_index.shape[1], dtype=np.float32)
    return edge_index, edge_weights


# =========================================================================== #
# CellTypeEmbedder --- relational GAT (numpy, ~1400 nodes)
# =========================================================================== #

# --------------------------------------------------------------------------- #
# GATv2 single-head attention (pure numpy --- graph is tiny)
# --------------------------------------------------------------------------- #

class MlxGraphAttention:
    """GATv2 single-head attention, mirrors TF GraphAttention.call().

    All computation is done in numpy/float32 (graph has ~1400 nodes, ~8000
    edges --- performance is irrelevant, correctness is everything).
    """

    def __init__(
        self,
        units: int,
        residual_weight: float = 0.7,
        use_residual: bool = True,
    ):
        self.units = units
        self.residual_weight = residual_weight
        self.use_residual = use_residual

        # Weights to be loaded
        self.attn_kernel: np.ndarray | None = None        # (units, 1)
        self.structural_weight: np.ndarray | None = None   # (1,)
        self.node_transform_kernel: np.ndarray | None = None  # (in_dim, units)
        self.residual_proj_kernel: np.ndarray | None = None   # (in_dim, units)

    def __call__(
        self,
        node_states: np.ndarray,   # [N, in_dim]
        edges: np.ndarray,         # [E, 2]  int
        edge_weights: np.ndarray,  # [E, 1]  float
    ) -> np.ndarray:
        """Forward pass --- returns [N, units]."""
        N = node_states.shape[0]
        original = node_states

        # 1. Linear transform  (no bias)
        h = node_states @ self.node_transform_kernel              # [N, units]

        # 2. Gather src / dst
        src_idx = edges[:, 0].astype(np.intp)
        dst_idx = edges[:, 1].astype(np.intp)
        feat_src = h[src_idx]   # [E, units]
        feat_dst = h[dst_idx]   # [E, units]

        # 3. GATv2: LeakyReLU(h_src + h_dst) then dot with attn_kernel
        summed = feat_src + feat_dst
        activated = np.where(summed > 0, summed, 0.2 * summed)  # LeakyReLU(0.2)
        logits = activated @ self.attn_kernel                    # [E, 1]

        # 4. Structural bias
        ew = edge_weights
        if ew.ndim == 1:
            ew = ew[:, None]
        logits = logits + ew * self.structural_weight            # [E, 1]

        # 5. Clip
        logits = np.clip(logits, -20.0, 20.0)

        # 6. Softmax per source node (segment softmax)
        scores = np.exp(logits)                                  # [E, 1]
        scores = np.clip(scores, 1e-10, 1e10)

        # segment sum over source indices
        seg_sum = np.zeros((N, 1), dtype=np.float32)
        np.add.at(seg_sum, src_idx, scores)
        denom = seg_sum[src_idx] + 1e-6                          # [E, 1]
        alpha = scores / denom                                   # [E, 1]

        # 7. Aggregate: weighted sum of dst features -> src
        neighbor_feats = h[dst_idx]                              # [E, units]
        weighted = neighbor_feats * alpha                        # [E, units]
        agg = np.zeros((N, self.units), dtype=np.float32)
        np.add.at(agg, src_idx, weighted)

        # NOTE: activation is NOT applied here --- it is applied after merge
        # in MultiHeadGraphAttention (matching TF where activation=None per head)

        # 8. Residual
        if self.use_residual:
            res = original @ self.residual_proj_kernel            # [N, units]
            out = (1.0 - self.residual_weight) * agg + self.residual_weight * res
        else:
            out = agg

        return out


# --------------------------------------------------------------------------- #
# Multi-head GAT + LayerNorm
# --------------------------------------------------------------------------- #

class MlxMultiHeadGraphAttention:
    """Multi-head GATv2 (four heads by default) with merge and normalization."""

    def __init__(self, units_per_head: int, num_heads: int = 4):
        self.units_per_head = units_per_head
        self.num_heads = num_heads
        self.heads: list[MlxGraphAttention] = [
            MlxGraphAttention(units_per_head) for _ in range(num_heads)
        ]
        # LayerNorm params
        self.ln_gamma: np.ndarray | None = None  # (out_dim,)
        self.ln_beta: np.ndarray | None = None   # (out_dim,)

    def __call__(
        self,
        node_states: np.ndarray,
        edges: np.ndarray,
        edge_weights: np.ndarray,
    ) -> np.ndarray:
        original = node_states

        # Per-head forward
        head_outs = [
            head(node_states, edges, edge_weights) for head in self.heads
        ]
        # Concat  [N, units_per_head * num_heads]
        merged = np.concatenate(head_outs, axis=-1)

        # Activation (relu) --- applied after merge, matching TF
        merged = np.maximum(merged, 0.0)

        # Skip connection (only when shapes match)
        if merged.shape[-1] == original.shape[-1]:
            merged = merged + original

        # LayerNorm
        mean = merged.mean(axis=-1, keepdims=True)
        var = merged.var(axis=-1, keepdims=True)
        merged = (merged - mean) / np.sqrt(var + 1e-5)
        merged = merged * self.ln_gamma + self.ln_beta

        return merged


# --------------------------------------------------------------------------- #
# Full CellTypeEmbedder
# --------------------------------------------------------------------------- #

class MlxCellTypeEmbedder:
    """Checkpoint-compatible relational GAT cell-type embedder.

    Produces L2-normalized prototypes with the checkpoint's embedding dimension.
    """

    def __init__(self):
        # Graph data (loaded from h5)
        self.edges_up: np.ndarray | None = None
        self.edge_weights_up: np.ndarray | None = None
        self.edges_down: np.ndarray | None = None
        self.edge_weights_down: np.ndarray | None = None
        self.semantic_emb: np.ndarray | None = None
        self.level_ids: np.ndarray | None = None

        self.full_edges_up: np.ndarray | None = None
        self.full_edge_weights_up: np.ndarray | None = None
        self.full_edges_down: np.ndarray | None = None
        self.full_edge_weights_down: np.ndarray | None = None
        self.full_semantic_emb: np.ndarray | None = None
        self.full_level_ids: np.ndarray | None = None

        # GAT layers (1 layer each direction, shared for full ontology)
        self.gat_up = MlxMultiHeadGraphAttention(192, 4)
        self.gat_down = MlxMultiHeadGraphAttention(192, 4)

        # Fusion gate: Dense(1536 -> 1, sigmoid)
        self.fusion_kernel: np.ndarray | None = None   # (1536, 1)
        self.fusion_bias: np.ndarray | None = None     # (1,)

        # Level embedding
        self.use_level_embeddings: bool = True
        self.level_emb_table: np.ndarray | None = None  # (12, 32)

        # Config
        self.structural_dim: int = 768

    def get_gat_prototypes(
        self,
        for_full_ontology: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Compute GAT-derived class prototypes.

        Returns:
            [num_classes, 768] float32 array (L2-normalized if normalize=True).
        """
        if for_full_ontology:
            edges_up = self.full_edges_up
            ew_up = self.full_edge_weights_up
            edges_down = self.full_edges_down
            ew_down = self.full_edge_weights_down
            sem = self.full_semantic_emb
            level_ids = self.full_level_ids
        else:
            edges_up = self.edges_up
            ew_up = self.edge_weights_up
            edges_down = self.edges_down
            ew_down = self.edge_weights_down
            sem = self.semantic_emb
            level_ids = self.level_ids

        # Start with semantic embeddings
        h = sem.copy()  # [N, 768]

        # Concatenate level embeddings
        if self.use_level_embeddings and level_ids is not None:
            lev = self.level_emb_table[level_ids.astype(np.intp)]  # [N, 32]
            h = np.concatenate([h, lev], axis=-1)                  # [N, 800]

        # Relational GAT pass (1 layer)
        h_up = self.gat_up(h, edges_up, ew_up)        # [N, 768]
        h_down = self.gat_down(h, edges_down, ew_down) # [N, 768]

        # Gated fusion
        concat_ud = np.concatenate([h_up, h_down], axis=-1)  # [N, 1536]
        gate = 1.0 / (1.0 + np.exp(-(concat_ud @ self.fusion_kernel + self.fusion_bias)))  # sigmoid
        h_fused = gate * h_up + (1.0 - gate) * h_down

        # Skip connection (input 800-dim vs fused 768-dim -> shapes don't match -> skip)
        if h.shape[-1] == h_fused.shape[-1]:
            h = h_fused + h
        else:
            h = h_fused

        # L2 normalize
        if normalize:
            norms = np.linalg.norm(h, axis=-1, keepdims=True)
            h = h / np.maximum(norms, 1e-8)

        return h


# --------------------------------------------------------------------------- #
# CellTypeEmbedder weight loader
# --------------------------------------------------------------------------- #

def load_celltype_embedder_mlx(h5_path: str) -> MlxCellTypeEmbedder:
    """Load CellTypeEmbedder weights from HECTOR h5 checkpoint.

    Weight mapping (from TF weight dump):
        0-11:  non-trainable graph data
        12-27: gat_layers_up[0]  --- 4 heads x (attn_kernel, structural_weight,
               node_transform kernel, residual_proj kernel)
        28-29: gat_layers_up[0] LayerNorm (gamma, beta)
        30-45: gat_layers_down[0] --- same layout as up
        46-47: gat_layers_down[0] LayerNorm (gamma, beta)
        48-49: fusion_gate Dense (kernel, bias)
        50:    level_embedding table
    """
    import h5py

    emb = MlxCellTypeEmbedder()

    with h5py.File(h5_path, "r") as f:
        g = f["weights/celltype_gat"]

        # ---------- graph data (weights 0-11) ----------
        emb.edges_up            = g["weight_0"][:].astype(np.int64)
        emb.edge_weights_up     = g["weight_1"][:].astype(np.float32)
        emb.edges_down          = g["weight_2"][:].astype(np.int64)
        emb.edge_weights_down   = g["weight_3"][:].astype(np.float32)
        emb.semantic_emb        = g["weight_4"][:].astype(np.float32)
        emb.level_ids           = g["weight_5"][:].astype(np.int32)
        emb.full_level_ids      = g["weight_6"][:].astype(np.int32)
        emb.full_edges_up       = g["weight_7"][:].astype(np.int64)
        emb.full_edge_weights_up = g["weight_8"][:].astype(np.float32)
        emb.full_edges_down     = g["weight_9"][:].astype(np.int64)
        emb.full_edge_weights_down = g["weight_10"][:].astype(np.float32)
        emb.full_semantic_emb   = g["weight_11"][:].astype(np.float32)

        # ---------- GAT UP heads (weights 12-27) ----------
        # Each head: attn_kernel, structural_weight, node_transform_kernel, residual_proj_kernel
        for head_idx in range(4):
            base = 12 + head_idx * 4
            head = emb.gat_up.heads[head_idx]
            head.attn_kernel           = g[f"weight_{base}"][:].astype(np.float32)
            head.structural_weight     = g[f"weight_{base+1}"][:].astype(np.float32)
            head.node_transform_kernel = g[f"weight_{base+2}"][:].astype(np.float32)
            head.residual_proj_kernel  = g[f"weight_{base+3}"][:].astype(np.float32)

        # UP LayerNorm (weights 28-29)
        emb.gat_up.ln_gamma = g["weight_28"][:].astype(np.float32)
        emb.gat_up.ln_beta  = g["weight_29"][:].astype(np.float32)

        # ---------- GAT DOWN heads (weights 30-45) ----------
        for head_idx in range(4):
            base = 30 + head_idx * 4
            head = emb.gat_down.heads[head_idx]
            head.attn_kernel           = g[f"weight_{base}"][:].astype(np.float32)
            head.structural_weight     = g[f"weight_{base+1}"][:].astype(np.float32)
            head.node_transform_kernel = g[f"weight_{base+2}"][:].astype(np.float32)
            head.residual_proj_kernel  = g[f"weight_{base+3}"][:].astype(np.float32)

        # DOWN LayerNorm (weights 46-47)
        emb.gat_down.ln_gamma = g["weight_46"][:].astype(np.float32)
        emb.gat_down.ln_beta  = g["weight_47"][:].astype(np.float32)

        # ---------- Fusion gate (weights 48-49) ----------
        emb.fusion_kernel = g["weight_48"][:].astype(np.float32)
        emb.fusion_bias   = g["weight_49"][:].astype(np.float32)

        # ---------- Level embedding table (weight 50) ----------
        emb.level_emb_table = g["weight_50"][:].astype(np.float32)

    return emb


# =========================================================================== #
# HPLHead + consensus scoring (numpy)
# =========================================================================== #

from dataclasses import dataclass, field

PROFILE_TOPK = 15  # Number of top classes for profile masking (matches TF)


class MlxHPLHead:
    """Numpy/MLX port of HPLHead inference (labels=None path only).

    The forward pass is:
        augmented = node_parts (+ level_proj if use_level_embeddings)
        proxies = ancestry_matrix @ augmented
        emb_norm = l2_normalize(embeddings)
        prx_norm = l2_normalize(proxies)
        logits = emb_norm @ prx_norm.T * scale
    """

    def __init__(
        self,
        node_parts: np.ndarray,  # [num_total_nodes, embedding_dim]
        ancestry_matrix: np.ndarray,  # [num_seen_classes, num_total_nodes]
        scale: float,
        *,
        node_level_ids: Optional[np.ndarray] = None,  # [num_total_nodes]
        level_embeddings: Optional[np.ndarray] = None,  # [max_level+1, level_dim]
        level_projection: Optional[np.ndarray] = None,  # [level_dim, embedding_dim]
    ):
        self.node_parts = node_parts.astype(np.float32)
        self.ancestry_matrix = ancestry_matrix.astype(np.float32)
        self.scale = float(scale)
        self.num_seen_classes = ancestry_matrix.shape[0]
        self.num_total_nodes = ancestry_matrix.shape[1]
        self.embedding_dim = node_parts.shape[1]

        # Level embeddings
        self.use_level_embeddings = (
            node_level_ids is not None
            and level_embeddings is not None
            and level_projection is not None
        )
        if self.use_level_embeddings:
            self.node_level_ids = node_level_ids.astype(np.int32)
            self.level_embeddings = level_embeddings.astype(np.float32)
            self.level_projection = level_projection.astype(np.float32)
        else:
            self.node_level_ids = None
            self.level_embeddings = None
            self.level_projection = None

        # Pre-compute proxies (they are static at inference)
        self._proxies_norm = None  # lazily computed

    @property
    def proxies_norm(self) -> np.ndarray:
        """Normalized proxy vectors [num_seen_classes, embedding_dim]."""
        if self._proxies_norm is None:
            self._proxies_norm = self._compute_proxies_norm()
        return self._proxies_norm

    def _compute_proxies_norm(self) -> np.ndarray:
        """Compute L2-normalized proxy vectors from node_parts + level embeddings."""
        augmented = self.node_parts.copy()

        if self.use_level_embeddings:
            # level_emb = level_embeddings[node_level_ids]  -> [N, level_dim]
            level_emb = self.level_embeddings[self.node_level_ids]
            # level_proj = level_emb @ level_projection     -> [N, embedding_dim]
            level_proj = level_emb @ self.level_projection
            augmented = augmented + level_proj

        # proxies = ancestry_matrix @ augmented -> [num_seen, embedding_dim]
        proxies = self.ancestry_matrix @ augmented

        # L2 normalize
        norms = np.linalg.norm(proxies, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return (proxies / norms).astype(np.float32)

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        """Forward pass: compute scaled cosine similarity logits.

        Args:
            embeddings: [batch_size, embedding_dim] cell embeddings (mu).

        Returns:
            logits: [batch_size, num_seen_classes] scaled cosine similarities.
        """
        emb = np.asarray(embeddings, dtype=np.float32)
        # L2 normalize embeddings
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        emb_norm = emb / norms

        # Cosine similarity: [batch, dim] @ [dim, num_seen] -> [batch, num_seen]
        cos_theta = emb_norm @ self.proxies_norm.T

        return cos_theta * self.scale


def load_hpl_head_mlx(h5_path: str) -> Optional[MlxHPLHead]:
    """Load HPLHead weights from an HDF5 checkpoint.

    Args:
        h5_path: Path to the HECTOR .h5 checkpoint file.

    Returns:
        MlxHPLHead instance, or None if no HPLHead is in the checkpoint.
    """
    import os
    import h5py

    h5_path = os.path.expanduser(h5_path)

    with h5py.File(h5_path, "r") as f:
        if "weights/aux_classifier" not in f:
            return None

        # Load config
        config_str = f["config"].attrs.get("model_config")
        if config_str is None:
            return None
        config = json.loads(config_str)
        aux_config = config.get("aux_classifier_config")
        if aux_config is None:
            return None

        scale = aux_config.get("scale", 20.0)

        # Load ancestry matrix
        if "config/ancestry_matrix" in f:
            ancestry_matrix = f["config/ancestry_matrix"][:].astype(np.float32)
        else:
            return None

        # Load weights from weights/aux_classifier
        grp = f["weights/aux_classifier"]

        def load_weight(idx: int) -> np.ndarray:
            key = f"weight_{idx}"
            ds = grp[key]
            if ds.shape == ():
                return np.array(ds[()], dtype=np.float32)
            return ds[:].astype(np.float32)

        # Checkpoint weight layout:
        #   weight_0: node_parts [num_total_nodes, embedding_dim]
        #   weight_1: hpl_scale scalar
        #   weight_2: hpl_margin scalar (unused at inference)
        #   weight_3: node_level_ids [num_total_nodes] int32
        #   weight_4: level embeddings [num_levels, level_dim]
        #   weight_5: level projection kernel [level_dim, embedding_dim]
        node_parts = load_weight(0)  # [num_total_nodes, embedding_dim]
        loaded_scale = float(load_weight(1))  # override config scale with trained value

        # Check for level embeddings
        node_level_ids = None
        level_embeddings_arr = None
        level_projection_arr = None

        num_weights = len(grp.keys())
        if num_weights >= 6:
            # Has level embeddings
            node_level_ids = grp["weight_3"][:].astype(np.int32)
            level_embeddings_arr = load_weight(4)  # [max_level+1, level_dim]
            level_projection_arr = load_weight(5)  # [level_dim, embedding_dim]

        return MlxHPLHead(
            node_parts=node_parts,
            ancestry_matrix=ancestry_matrix,
            scale=loaded_scale,
            node_level_ids=node_level_ids,
            level_embeddings=level_embeddings_arr,
            level_projection=level_projection_arr,
        )


@dataclass
class ProcrustesCfg:
    """Procrustes alignment configuration loaded from checkpoint."""

    is_aligned: bool = False
    rotation_matrix: Optional[np.ndarray] = None  # [dim, dim]
    gat_mean: Optional[np.ndarray] = None  # [1, dim]
    hpl_mean: Optional[np.ndarray] = None  # [1, dim]
    use_centering: bool = False


def load_procrustes_cfg(h5_path: str) -> ProcrustesCfg:
    """Load Procrustes alignment data from checkpoint."""
    import os
    import h5py

    h5_path = os.path.expanduser(h5_path)
    cfg = ProcrustesCfg()

    with h5py.File(h5_path, "r") as f:
        config_str = f["config"].attrs.get("model_config")
        if config_str:
            model_config = json.loads(config_str)
        else:
            model_config = {}

        if "procrustes/rotation_matrix" not in f:
            return cfg

        cfg.rotation_matrix = f["procrustes/rotation_matrix"][:].astype(np.float32)
        cfg.is_aligned = True

        if "procrustes" in f:
            grp = f["procrustes"]
            use_centering_attr = grp.attrs.get("use_centering", False)
            has_means = "hpl_mean" in grp and "gat_mean" in grp

            if use_centering_attr or has_means:
                if has_means:
                    cfg.hpl_mean = grp["hpl_mean"][:].astype(np.float32)
                    cfg.gat_mean = grp["gat_mean"][:].astype(np.float32)
                    cfg.use_centering = True

    return cfg


@dataclass
class OntologyCfg:
    """Ontology metadata loaded from checkpoint."""

    seen_classes: List[str] = field(default_factory=list)
    full_classes: List[str] = field(default_factory=list)
    full_ontology_adj: Optional[np.ndarray] = None  # [N_all, N_all]
    seen_indices: Optional[np.ndarray] = None  # [N_seen] -> position in full


def load_ontology_cfg(h5_path: str) -> OntologyCfg:
    """Load ontology class lists and adjacency from checkpoint."""
    import os
    import h5py

    h5_path = os.path.expanduser(h5_path)
    cfg = OntologyCfg()

    with h5py.File(h5_path, "r") as f:
        if "ontology/classes" in f:
            raw = f["ontology/classes"][:]
            cfg.seen_classes = [
                c.decode() if isinstance(c, bytes) else str(c) for c in raw
            ]
        if "ontology/full_classes" in f:
            raw = f["ontology/full_classes"][:]
            cfg.full_classes = [
                c.decode() if isinstance(c, bytes) else str(c) for c in raw
            ]
        if "ontology/full_ontology_adj" in f:
            cfg.full_ontology_adj = f["ontology/full_ontology_adj"][:].astype(
                np.float32
            )

    # Compute seen_indices: mapping seen class -> index in full class list
    if cfg.seen_classes and cfg.full_classes:
        full_map = {cls: idx for idx, cls in enumerate(cfg.full_classes)}
        cfg.seen_indices = np.array(
            [full_map.get(cls, 0) for cls in cfg.seen_classes], dtype=np.int32
        )

    return cfg


# --------------------------------------------------------------------------- #
# PPR (Personalized PageRank) computation --- numpy port
# --------------------------------------------------------------------------- #

def compute_ppr_matrix(
    adj: np.ndarray,
    alpha: float = 0.15,
) -> np.ndarray:
    """Compute symmetric PPR matrix (baseline mode).

    Args:
        adj: Adjacency matrix [N, N].
        alpha: Restart probability.

    Returns:
        PPR matrix [N, N], float32, row-normalized.
    """
    n = adj.shape[0]
    identity = np.eye(n, dtype=np.float64)
    adj_sym = np.maximum(adj, adj.T).astype(np.float64)
    adj_sym = adj_sym + identity

    degrees = np.sum(adj_sym, axis=1)
    degrees = np.maximum(degrees, 1e-8)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    norm_adj = d_inv_sqrt @ adj_sym @ d_inv_sqrt

    matrix_to_invert = identity - (1 - alpha) * norm_adj
    try:
        inv_matrix = np.linalg.inv(matrix_to_invert)
        ppr = alpha * inv_matrix
    except np.linalg.LinAlgError:
        return np.eye(n, dtype=np.float32)

    row_sums = ppr.sum(axis=1, keepdims=True)
    ppr = ppr / np.maximum(row_sums, 1e-8)
    return ppr.astype(np.float32)


def compute_asymmetric_ppr(
    adj: np.ndarray,
    alpha: float = 0.15,
    forward_weight: float = 0.6,
) -> np.ndarray:
    """Compute asymmetric (directional) PPR by blending forward and backward PPR.

    Args:
        adj: Adjacency matrix [N, N].
        alpha: Restart probability.
        forward_weight: Blending weight for forward PPR in [0.0, 1.0].

    Returns:
        Asymmetric PPR matrix [N, N], float32, row-normalized.
    """
    n = adj.shape[0]
    identity = np.eye(n, dtype=np.float64)

    def _ppr_from_adj(a: np.ndarray) -> np.ndarray:
        a_loop = a.astype(np.float64) + identity
        degrees = np.sum(a_loop, axis=1)
        degrees = np.maximum(degrees, 1e-8)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        norm_adj = d_inv_sqrt @ a_loop @ d_inv_sqrt
        matrix_to_invert = identity - (1 - alpha) * norm_adj
        try:
            ppr = alpha * np.linalg.inv(matrix_to_invert)
        except np.linalg.LinAlgError:
            return identity.astype(np.float32)
        row_sums = ppr.sum(axis=1, keepdims=True)
        ppr = ppr / np.maximum(row_sums, 1e-8)
        return ppr.astype(np.float32)

    ppr_forward = _ppr_from_adj(adj)
    ppr_backward = _ppr_from_adj(adj.T)

    ppr = forward_weight * ppr_forward + (1 - forward_weight) * ppr_backward

    # Ensure diagonal dominance
    for i in range(n):
        row_max = np.max(ppr[i])
        if ppr[i, i] < row_max:
            ppr[i, i] = row_max + 1e-6

    row_sums = ppr.sum(axis=1, keepdims=True)
    ppr = ppr / np.maximum(row_sums, 1e-8)
    return ppr.astype(np.float32)


# --------------------------------------------------------------------------- #
# Scoring pipeline --- numpy port
# --------------------------------------------------------------------------- #

def _topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of row-wise top-K columns.

    Matches _topk_mask_tf: at exact ties on the boundary this may include
    slightly more than K classes.
    """
    # Get the k-th largest value per row
    # np.partition is O(n) instead of O(n log n) for np.sort
    if k >= scores.shape[1]:
        return np.ones_like(scores, dtype=bool)
    kth_values = np.partition(scores, -k, axis=1)[:, -k : -k + 1]
    return scores >= kth_values


def _expand_hpl_probs(
    hpl_probs: np.ndarray,
    seen_indices: np.ndarray,
    num_total_classes: int,
) -> np.ndarray:
    """Expand HPL probabilities from seen classes to full ontology.

    Args:
        hpl_probs: [batch, num_seen] softmax probabilities.
        seen_indices: [num_seen] indices into the full class list.
        num_total_classes: Total number of classes in full ontology.

    Returns:
        expanded: [batch, num_total] with seen-class probs placed, rest 1e-10.
    """
    batch_size = hpl_probs.shape[0]
    expanded = np.full(
        (batch_size, num_total_classes), 1e-10, dtype=np.float32
    )
    expanded[:, seen_indices] = hpl_probs
    return expanded


def _compute_profile_matching_scores(
    z_latent: np.ndarray,
    prototypes_seen: np.ndarray,
    prior_matrix_full: np.ndarray,
    seen_indices: np.ndarray,
) -> np.ndarray:
    """Compute profile-based voting scores using Pearson correlation.

    Args:
        z_latent: [batch, dim] cell embeddings.
        prototypes_seen: [N_seen, dim] HPL prototypes (raw, not normalized).
        prior_matrix_full: [N_all, N_all] PPR matrix.
        seen_indices: [N_seen] mapping from seen class to full class index.

    Returns:
        profile_scores: [batch, N_all] correlation scores clipped to [0, 1].
    """
    z_norm = _l2_normalize(z_latent, axis=1)
    protos_norm = _l2_normalize(prototypes_seen, axis=1)

    # Observed profile: cosine sim of each cell to all seen prototypes
    sim_obs = z_norm @ protos_norm.T  # [batch, N_seen]

    # Theoretical profiles: PPR rows for all classes, columns for seen classes
    theoretical_profiles = prior_matrix_full[:, seen_indices]  # [N_all, N_seen]

    # Center and normalize (Pearson correlation = cosine of centered vectors)
    obs_mean = np.mean(sim_obs, axis=1, keepdims=True)
    obs_centered = sim_obs - obs_mean
    obs_normalized = _l2_normalize(obs_centered, axis=1)

    theo_mean = np.mean(theoretical_profiles, axis=1, keepdims=True)
    theo_centered = theoretical_profiles - theo_mean
    theo_normalized = _l2_normalize(theo_centered, axis=1)

    # [batch, N_seen] @ [N_all, N_seen].T -> [batch, N_all]
    profile_scores = obs_normalized @ theo_normalized.T

    return np.maximum(profile_scores, 0.0).astype(np.float32)


def _get_neighborhood_mask(
    top_seen_indices: np.ndarray,
    seen_indices: np.ndarray,
    full_ontology_adj: np.ndarray,
) -> np.ndarray:
    """Compute a one-hop boolean neighborhood mask for expert predictions.

    Args:
        top_seen_indices: [batch, k] top expert prediction indices (seen-class space).
        seen_indices: [N_seen] mapping from seen class to full class index.
        full_ontology_adj: [N_all, N_all] full ontology adjacency matrix.

    Returns:
        neighborhood_mask: [batch, N_all] boolean mask.
    """
    batch_size, k = top_seen_indices.shape
    n_all = full_ontology_adj.shape[0]

    # Map seen indices -> full indices
    flat_seen = top_seen_indices.ravel()
    flat_full = seen_indices[flat_seen]

    # Symmetric adjacency + self-loops
    A_sym = np.maximum(full_ontology_adj, full_ontology_adj.T)
    A_sym = A_sym + np.eye(n_all, dtype=np.float32)

    # One-hot seeds: [batch*k, N_all]
    seeds = np.zeros((batch_size * k, n_all), dtype=np.float32)
    seeds[np.arange(batch_size * k), flat_full] = 1.0

    # 1-hop propagation: [batch*k, N_all] @ [N_all, N_all] -> [batch*k, N_all]
    propagated = seeds @ A_sym

    # Reshape to [batch, k, N_all] and union over k
    propagated = propagated.reshape(batch_size, k, n_all)
    neighborhood_mask = np.max(propagated, axis=1) > 0.001

    return neighborhood_mask


def score_embeddings_np(
    mu: np.ndarray,
    gat_prototypes_full: np.ndarray,
    hpl_head: Optional[MlxHPLHead],
    *,
    procrustes: Optional[ProcrustesCfg] = None,
    ontology: Optional[OntologyCfg] = None,
    prior_matrix_full: Optional[np.ndarray] = None,
    use_full_ontology: bool = True,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Compute consensus scores from embeddings and ontology information.

    Args:
        mu: [batch, dim] cell embeddings.
        gat_prototypes_full: [N_all, dim] GAT-derived full-ontology prototypes.
        hpl_head: MlxHPLHead instance (or None for GAT-only).
        procrustes: ProcrustesCfg for aligning GAT to HPL space.
        ontology: OntologyCfg with class lists and adjacency.
        prior_matrix_full: [N_all, N_all] PPR matrix for profile matching.
            If None and ontology.full_ontology_adj is available, it will be
            computed on the fly (asymmetric PPR, alpha=0.15, fw=0.6).
        use_full_ontology: Whether to return scores for full ontology (True)
            or restrict to seen classes only.
        top_k: Number of top predictions to return.

    Returns:
        Dict with:
            top_indices: [batch, top_k] int32
            top_scores: [batch, top_k] float64
            score_generalist: [batch, N] float32
            score_expert: [batch, N] float32
            score_profile: [batch, N] float32
            expert_confidence: [batch] float32
            base_final_scores: [batch, N] float32
            filtered_scores: [batch, N] float32
    """
    mu = np.asarray(mu, dtype=np.float32)
    gat_prototypes_full = np.asarray(gat_prototypes_full, dtype=np.float32)
    batch_size = mu.shape[0]

    if procrustes is None:
        procrustes = ProcrustesCfg()
    if ontology is None:
        ontology = OntologyCfg()

    n_all = gat_prototypes_full.shape[0]

    # --- Step 1: Generalist scores (GAT cosine similarity) ---
    z_norm = _l2_normalize(mu, axis=1)
    gat_emb_norm = _l2_normalize(gat_prototypes_full, axis=1)

    if procrustes.is_aligned and procrustes.rotation_matrix is not None:
        R = procrustes.rotation_matrix
        if procrustes.use_centering and procrustes.hpl_mean is not None:
            gat_aligned = (gat_prototypes_full - procrustes.gat_mean) @ R + procrustes.hpl_mean
        else:
            gat_aligned = gat_prototypes_full @ R
        gat_aligned_norm = _l2_normalize(gat_aligned, axis=1)
        score_generalist = z_norm @ gat_aligned_norm.T
    else:
        score_generalist = z_norm @ gat_emb_norm.T

    score_generalist = np.clip(score_generalist, 0.0, 1.0).astype(np.float32)

    # --- Step 2: Expert scores (HPL head) ---
    if hpl_head is not None and ontology.seen_indices is not None:
        hpl_logits = hpl_head(mu)  # [batch, N_seen]
        score_expert_seen = _softmax(hpl_logits, axis=-1)  # [batch, N_seen]
        score_expert = _expand_hpl_probs(
            score_expert_seen, ontology.seen_indices, n_all
        )
    else:
        score_expert_seen = None
        score_expert = np.zeros_like(score_generalist)

    # --- Step 3: Profile matching scores ---
    if hpl_head is not None and ontology.seen_indices is not None:
        # Compute raw (unnormalized) seen prototypes.
        # Profile matching and expert confidence use ancestry-derived prototypes
        # without level embeddings. Classification logits use the level-augmented
        # path inside HPLHead.
        protos_seen = hpl_head.ancestry_matrix @ hpl_head.node_parts

        # Compute PPR if not provided
        if prior_matrix_full is None:
            if ontology.full_ontology_adj is not None:
                prior_matrix_full = compute_asymmetric_ppr(
                    ontology.full_ontology_adj, alpha=0.15, forward_weight=0.6
                )
            else:
                prior_matrix_full = np.eye(n_all, dtype=np.float32)

        score_profile = _compute_profile_matching_scores(
            mu, protos_seen, prior_matrix_full, ontology.seen_indices
        )
    else:
        protos_seen = None
        score_profile = np.zeros_like(score_generalist)

    # --- Step 4: Expert confidence ---
    if hpl_head is not None and score_expert_seen is not None:
        # Top-2 expert predictions (seen class space)
        top2_idx = np.argsort(score_expert_seen, axis=1)[:, -2:][:, ::-1]
        p1_idx = top2_idx[:, 0]
        p2_idx = top2_idx[:, 1]

        # Cosine similarity with top-2 prototypes
        protos_norm = _l2_normalize(protos_seen, axis=1)
        p1_vec = protos_norm[p1_idx]  # [batch, dim]
        p2_vec = protos_norm[p2_idx]  # [batch, dim]

        sim_p1 = np.sum(z_norm * p1_vec, axis=1)
        sim_p2 = np.sum(z_norm * p2_vec, axis=1)

        expert_confidence = np.clip(sim_p1 - sim_p2, 0.0, 1.0)  # [batch]

        # Expert safety mask: one-hot at top-1 expert prediction (in full space)
        full_p1_idx = ontology.seen_indices[p1_idx]
        mask_expert_safety = np.zeros((batch_size, n_all), dtype=bool)
        mask_expert_safety[np.arange(batch_size), full_p1_idx] = True
    else:
        expert_confidence = np.zeros(batch_size, dtype=np.float32)
        mask_expert_safety = np.zeros((batch_size, n_all), dtype=bool)

    # --- Step 5: Restrict to seen classes if requested ---
    if not use_full_ontology and ontology.seen_indices is not None:
        si = ontology.seen_indices
        score_generalist = score_generalist[:, si]
        score_expert = score_expert[:, si]
        score_profile = score_profile[:, si]
        mask_expert_safety = mask_expert_safety[:, si]
        n_out = len(si)
    else:
        n_out = n_all

    # --- Step 6: Consensus weighting ---
    ec = expert_confidence[:, np.newaxis]  # [batch, 1]
    w_exp = 0.8 * ec
    w_gen = 0.1 * ec + 0.6 * (1.0 - ec)
    w_prof = 0.1 * ec + 0.4 * (1.0 - ec)

    base_final_scores = (
        w_exp * score_expert + w_gen * score_generalist + w_prof * score_profile
    ).astype(np.float32)

    # --- Step 7: Masking ---
    # Generalist mask
    gen_max = np.max(score_generalist, axis=1, keepdims=True)
    mask_gen = score_generalist >= (0.85 * gen_max)

    # Profile top-K mask
    mask_prof_topk = _topk_mask(score_profile, k=PROFILE_TOPK)

    # Expert neighborhood mask
    if hpl_head is not None and score_expert_seen is not None and ontology.full_ontology_adj is not None:
        top3_idx = np.argsort(score_expert_seen, axis=1)[:, -3:][:, ::-1]
        mask_expert_neighborhood = _get_neighborhood_mask(
            top3_idx, ontology.seen_indices, ontology.full_ontology_adj
        )
        if not use_full_ontology and ontology.seen_indices is not None:
            mask_expert_neighborhood = mask_expert_neighborhood[:, ontology.seen_indices]
    else:
        mask_expert_neighborhood = np.ones_like(mask_gen)

    # Union mask
    mask_final = mask_gen | mask_prof_topk | mask_expert_neighborhood | mask_expert_safety
    filtered_scores = base_final_scores * mask_final.astype(np.float32)

    # --- Step 8: Top-K ---
    top_k_clamped = min(top_k, n_out)
    top_indices = np.argsort(filtered_scores, axis=1)[:, -top_k_clamped:][:, ::-1]
    top_scores = np.take_along_axis(filtered_scores, top_indices, axis=1)

    return {
        "top_indices": top_indices.astype(np.int32),
        "top_scores": top_scores.astype(np.float64),
        "score_generalist": score_generalist,
        "score_expert": score_expert,
        "score_profile": score_profile,
        "expert_confidence": expert_confidence,
        "base_final_scores": base_final_scores,
        "filtered_scores": filtered_scores,
    }


# =========================================================================== #
# GRIT label propagation (scipy.sparse)
# =========================================================================== #

def build_normalized_adjacency_scipy(
    edge_index: np.ndarray,
    edge_weights: np.ndarray,
    num_nodes: int,
) -> sp.csr_matrix:
    """Build a max-symmetrized, self-looped, degree-normalized adjacency.

    The transformation is ``D^{-1/2} (max(A, A.T) + I) D^{-1/2}``.

    The result remains a SciPy CSR matrix.

    Args:
        edge_index: [2, num_edges] int array -- (row, col) pairs.
        edge_weights: [num_edges] float array.
        num_nodes: Total graph size (adjacency is num_nodes x num_nodes).

    Returns:
        Normalized adjacency as scipy CSR matrix (float32).
    """
    rows = edge_index[0]
    cols = edge_index[1]
    adj = sp.coo_matrix(
        (edge_weights.astype(np.float32), (rows, cols)),
        shape=(num_nodes, num_nodes),
    )

    # Make symmetric (max of A[i,j] and A[j,i] for each pair).
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Self-loops.
    adj_hat = adj + sp.eye(num_nodes, dtype=np.float32)

    # Degree normalization: D^{-1/2}, leaving isolated nodes at 0.
    # ``out=`` initializes entries skipped by ``where`` to zero. Self-loops make
    # positive row sums the normal case; the mask also guards malformed inputs.
    row_sum = np.asarray(adj_hat.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(row_sum, -0.5, out=np.zeros_like(row_sum),
                          where=row_sum > 0)
    d_mat = sp.diags(d_inv_sqrt.astype(np.float32))

    # A_norm = D^{-1/2} * A_hat * D^{-1/2}
    return (d_mat @ adj_hat @ d_mat).tocsr().astype(np.float32)


def _get_cpu_class_chunk_size(
    n_rows: int,
    n_classes: int,
    *,
    working_buffers: int = 3,
) -> int:
    """Estimate a memory-bounded class chunk for CPU-side score processing."""
    try:
        import psutil
        available_memory = int(psutil.virtual_memory().available)
    except (ImportError, AttributeError):
        available_memory = 2 * 1024 ** 3

    if n_rows <= 0:
        return max(1, int(n_classes))

    bytes_per_class = int(n_rows) * 4 * max(1, working_buffers)
    chunk_size = int(available_memory) // max(1, bytes_per_class)
    chunk_size = max(1, min(chunk_size, int(n_classes)))
    return chunk_size


def apply_grit_refinement_scipy(
    initial_scores: np.ndarray,
    adj_csr: sp.csr_matrix,
    num_iterations: int = 3,
    alpha: float = 0.2,
    *,
    score_mode: str = "log_scores",
) -> np.ndarray:
    """Label propagation using scipy sparse matmul.

    Z_{k+1} = (1-alpha) * P_0 + alpha * A_norm @ Z_k

    Handles class-chunked processing and two score modes exactly like
    ``predictor_support.apply_grit_refinement``.

    Args:
        initial_scores: [n_cells, n_classes] float32 logits or positive scores.
        adj_csr: Normalized adjacency (scipy CSR).
        num_iterations: Propagation iterations.
        alpha: Mixing weight (higher = more neighbour influence).
        score_mode: ``"log_scores"`` (softmax first) or ``"positive_scores"``
                    (additive-eps normalization).

    Returns:
        refined_probs: [n_cells, n_classes] float32 refined probabilities.
    """
    initial_scores = np.asarray(initial_scores, dtype=np.float32)
    n_cells, n_classes = initial_scores.shape
    class_chunk_size = _get_cpu_class_chunk_size(
        n_cells, n_classes, working_buffers=3,
    )

    if score_mode not in {"log_scores", "positive_scores"}:
        raise ValueError(
            f"score_mode must be 'log_scores' or 'positive_scores', "
            f"got {score_mode!r}"
        )

    # --- Softmax / normalization into P_initial ---
    p_initial = np.empty_like(initial_scores)
    z_current = np.empty_like(initial_scores)

    if score_mode == "log_scores":
        # Chunked numerically-stable softmax.
        row_max = np.full(n_cells, -np.inf, dtype=np.float32)
        for start in range(0, n_classes, class_chunk_size):
            end = min(start + class_chunk_size, n_classes)
            block = initial_scores[:, start:end]
            row_max = np.maximum(row_max, np.max(block, axis=1))

        denom = np.zeros(n_cells, dtype=np.float64)
        for start in range(0, n_classes, class_chunk_size):
            end = min(start + class_chunk_size, n_classes)
            probs = np.exp(
                initial_scores[:, start:end] - row_max[:, None]
            ).astype(np.float32, copy=False)
            denom += np.sum(probs, axis=1, dtype=np.float64)

        for start in range(0, n_classes, class_chunk_size):
            end = min(start + class_chunk_size, n_classes)
            probs = np.exp(
                initial_scores[:, start:end] - row_max[:, None]
            ).astype(np.float32, copy=False)
            probs /= denom[:, None]
            p_initial[:, start:end] = probs
            z_current[:, start:end] = probs
    else:
        # positive_scores: (score + eps) / sum.
        eps = 1e-15
        denom = np.full(n_cells, eps * n_classes, dtype=np.float64)
        for start in range(0, n_classes, class_chunk_size):
            end = min(start + class_chunk_size, n_classes)
            denom += np.sum(
                initial_scores[:, start:end], axis=1, dtype=np.float64,
            )
        for start in range(0, n_classes, class_chunk_size):
            end = min(start + class_chunk_size, n_classes)
            probs = (
                (initial_scores[:, start:end] + eps) / denom[:, None]
            ).astype(np.float32, copy=False)
            p_initial[:, start:end] = probs
            z_current[:, start:end] = probs

    # --- Propagation loop ---
    z_next = np.empty_like(z_current)

    for _it in range(num_iterations):
        row_sums = np.zeros(n_cells, dtype=np.float64)
        for start in range(0, n_classes, class_chunk_size):
            end = min(start + class_chunk_size, n_classes)
            current_block = z_current[:, start:end]
            # scipy CSR @ dense numpy --- the key replacement for tf.sparse
            chunk_prop = (adj_csr @ current_block).astype(np.float32)
            mixed = (
                (1.0 - alpha) * p_initial[:, start:end]
                + alpha * chunk_prop
            )
            z_next[:, start:end] = mixed
            row_sums += np.sum(mixed, axis=1, dtype=np.float64)

        # Row normalize.
        row_sums += 1e-10
        for start in range(0, n_classes, class_chunk_size):
            end = min(start + class_chunk_size, n_classes)
            z_next[:, start:end] /= row_sums[:, None].astype(np.float32)

        z_current, z_next = z_next, z_current

    return z_current


# =========================================================================== #
# Batch orchestrator
# =========================================================================== #

def run_encoder_mlx(encoder, X_normalized, anchor_features,
                    k_neighbors=30, batch_size=2040, verbose=True):
    """Run full batched encoding with MLX encoder and MLX kNN.
    Returns (mu_class [n_cells, latent_dim], variance [n_cells]).
    """
    import mlx.core as mx
    import scipy.sparse as sp_mod

    n_cells = X_normalized.shape[0]
    n_genes = X_normalized.shape[1]
    n_anchors = anchor_features.shape[0]

    mu_out = None
    var_out = np.empty(n_cells, dtype=np.float32)
    num_batches = (n_cells + batch_size - 1) // batch_size

    _is_sparse = sp_mod.issparse(X_normalized)
    if _is_sparse:
        X_normalized = X_normalized.tocsr()
        if X_normalized.dtype != np.float32:
            X_normalized = X_normalized.astype(np.float32, copy=False)
    else:
        X_normalized = np.asarray(X_normalized, dtype=np.float32)

    # The anchors are constant across batches. Keep ONE resident raw copy on the
    # GPU and derive everything else from it once, instead of re-uploading and
    # re-processing the anchors every batch:
    #   * anchor_norms — per-anchor L2 norm, for the folded cosine kNN.
    #   * h_anchors    — the anchors' hidden representation. The hidden layers are
    #                    per-node, so this equals the anchor rows of a full
    #                    hidden([batch; anchors]) pass, but is computed just once.
    anchors_mx = mx.array(np.asarray(anchor_features, dtype=np.float32))
    anchor_norms = mx.linalg.norm(anchors_mx, axis=1, keepdims=True) + 1e-10
    h_anchors = encoder.encode_hidden(anchors_mx)
    mx.eval(anchor_norms, h_anchors)

    _dense_buf = np.empty((batch_size, n_genes), dtype=np.float32) if _is_sparse else None

    batch_iterator = range(0, n_cells, batch_size)
    if verbose:
        try:
            from tqdm import tqdm
            batch_iterator = tqdm(
                batch_iterator, total=num_batches,
                desc="  Encoding", unit="batch", ncols=80,
            )
        except ImportError:
            pass

    for i in batch_iterator:
        batch_end = min(i + batch_size, n_cells)
        actual = batch_end - i

        if _is_sparse:
            X_normalized[i:batch_end].toarray(out=_dense_buf[:actual])
            batch_X = _dense_buf[:actual]
        else:
            batch_X = X_normalized[i:batch_end]

        edge_index, _ = build_knn_graph_mlx(
            batch_X, anchor_features, k=k_neighbors,
            anchor_raw_mx=anchors_mx, anchor_norms=anchor_norms,
        )

        # Forward: run the hidden layers on the batch only, then reuse the
        # precomputed anchor hidden states. h_full matches what a full
        # hidden([batch; anchors]) pass would produce, with no per-batch anchor
        # copy or recompute.
        h_batch = encoder.encode_hidden(mx.array(np.asarray(batch_X, dtype=np.float32)))
        h_full = mx.concatenate([h_batch, h_anchors], axis=0)
        mu, lv = encoder.encode_from_hidden(
            h_full, mx.array(edge_index), batch_size=actual,
        )
        mx.eval(mu, lv)
        mu_np, lv_np = np.array(mu), np.array(lv)

        if mu_out is None:
            mu_out = np.empty((n_cells, mu_np.shape[1]), dtype=np.float32)
        mu_out[i:batch_end] = mu_np
        var_out[i:batch_end] = np.mean(
            np.exp(np.clip(lv_np, -20, 20)), axis=1,
        )

    if mu_out is None:
        mu_out = np.empty((0, 0), dtype=np.float32)
    return mu_out, var_out


def measure_encoder_floor_mlx(encoder, anchor_features) -> "int | None":
    """Measure the encoder's fixed memory FLOOR in bytes on Metal.

    The floor is the one-time ``encode_hidden(anchors)`` working set — the
    resident anchor matrix + the hidden-stack transient. It does NOT scale with
    batch size. It is measured with ``mx.get_peak_memory()`` for the current
    model, hardware, and MLX version.

    Returns int bytes, or None if MLX memory introspection is unavailable.
    """
    try:
        import mlx.core as mx
    except ImportError:
        return None
    try:
        mx.clear_cache()
        mx.reset_peak_memory()
        anchors_mx = mx.array(np.asarray(anchor_features, dtype=np.float32))
        anchor_norms = mx.linalg.norm(anchors_mx, axis=1, keepdims=True) + 1e-10
        h_anchors = encoder.encode_hidden(anchors_mx)
        mx.eval(anchors_mx, anchor_norms, h_anchors)
        floor = int(mx.get_peak_memory())
        del anchors_mx, anchor_norms, h_anchors
        mx.clear_cache()
        return floor if floor > 0 else None
    except Exception:
        return None
