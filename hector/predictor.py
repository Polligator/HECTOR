#!/usr/bin/env python3
"""Prediction and cell-evaluation interfaces for HECTOR models.

Key Features:
1. Loads model checkpoint and extracts gene order
2. Accepts AnnData objects as input
3. Automatically reorders genes to match training data
4. Fills missing genes with zeros
5. Applies same normalization as training
6. Returns predictions with confidence scores
7. Supports Procrustes-aligned checkpoints for zero-shot prediction

"""
from __future__ import annotations

import hashlib
import math
import numpy as np
import os
from pathlib import Path
import random
import weakref
import pandas as pd
from . import _HECTOR_USE_MLX as _USE_MLX
if not _USE_MLX:
    import tensorflow as tf
    from tensorflow.keras import layers
else:
    tf = None
    layers = None
import h5py
import re
import traceback
import warnings, random
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass
import anndata
from . import get_model_path
from . import predictor_support as _predictor_support
from .predictor_support import (
    Config,
    _ScoreMatrixBuffer,
    reorder_and_fill_genes,
    stable_environments,
)
if not _USE_MLX:
    from .predictor_support import (
        CellTypeEmbedder,
        DualHeadVGAEEncoder,
        HPLHead,
        VGAEDecoder,
        apply_grit_refinement,
        build_normalized_adjacency,
    )
from .trajectory_support import (
    _DEFAULT_MAX_COMPONENT_STD,
    _DEFAULT_MIN_COMPONENT_MEAN,
    _DEFAULT_MIN_COMPONENT_WEIGHT,
)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable





    



# ============================================================================
# Configuration
# ============================================================================

@dataclass
class InferenceConfig:
    """Configuration for production inference."""
    # Prediction parameters
    top_k: int = 1
    use_full_ontology: bool = True  # Use zero-shot mode
    
    # GRIT refinement parameters
    use_grit: bool = True
    grit_iterations: int = 3
    grit_alpha: float = 0.2
    
    # GRIT refinement targeting
    # Entropy is used to identify lower end of the prediction distribution (flat probability distributions).
    # If grit_entropy_threshold is None (default), the threshold is auto-calculated:
    #   - If grit_refinement_percentile is also None: Uses Multi-Otsu to find natural boundary
    #   - If grit_refinement_percentile is set: Uses that percentile (e.g., 10.0 = top 10%)
    grit_entropy_threshold: Optional[float] = None  # None = Auto, Float = Manual threshold
    grit_refinement_percentile: Optional[float] = None  # None = Multi-Otsu auto, Float = Manual percentile
    
    # Graph construction
    k_neighbors: int = 30
    
    # Batch processing
    batch_size: int = None

    # Enhanced PPR toggle
    use_asymmetric_ppr: bool = True

    # Variance-based quality filtering
    filter_low_quality: bool = False  # Enable variance-based quality filtering
    variance_threshold: Optional[float] = None  # Manual threshold override; None = auto (Kneedle)

    # Asymmetric PPR parameters
    forward_weight: float = 0.6   # Weight for forward (child→parent) PPR, range [0.0, 1.0]
    ppr_alpha: float = None       # PPR restart probability. None = use checkpoint value.

    def __post_init__(self):
        """Validate weight parameters are within valid ranges."""
        if not (0.0 <= self.forward_weight <= 1.0):
            raise ValueError(
                f"forward_weight must be in [0.0, 1.0], got {self.forward_weight}"
            )
        if self.ppr_alpha is not None and not (0.0 < self.ppr_alpha < 1.0):
            raise ValueError(
                f"ppr_alpha must be in (0.0, 1.0), got {self.ppr_alpha}"
            )


_ANNOTATION_MANIFEST_KEY = "hector_annotation"
_ANNOTATION_MANIFEST_VERSION = 1
_PREDICTION_FINGERPRINT_VERSION = 2
_PREDICTION_METADATA_ATTR_KEY = "hector_prediction_metadata"
_EMBEDDING_MANIFEST_KEY = "hector_embedding_manifest"
_EMBEDDING_MANIFEST_VERSION = 1
_PREDICTION_CORE_MANIFEST_KEY = "hector_prediction_core_manifest"
_PREDICTION_CORE_MANIFEST_VERSION = 1
_ATYPICAL_MANIFEST_KEY = "hector_atypical_manifest"
_ATYPICAL_MANIFEST_VERSION = 9
_ATYPICAL_FINGERPRINT_VERSION = 18


def _sanitize_for_h5ad_uns(obj):
    """Recursively convert ``obj`` to a structure anndata can write to h5ad.

    anndata's h5py writer uses dict keys as HDF5 group names, so every key
    must be a string -- a float/int key raises
    ``AttributeError: 'float' object has no attribute 'split'``.
    It also serializes lists by wrapping with ``np.array``, which fails on
    a list-of-dicts with
    ``TypeError: Can't implicitly convert non-string objects to strings``.

    This helper normalises three shapes that recur in HECTOR's ``uns``
    manifests so they round-trip through ``adata.write_h5ad`` / ``read_h5ad``:

    * dict keys -> ``str(k)``;
    * list-of-dicts with a shared key set -> dict-of-lists (which anndata
      serializes via ``write_mapping``);
    * numpy scalars (``np.int64``, ``np.float64``, ``np.bool_``) -> Python
      scalars.

    Numpy arrays, anndata DataFrames, and other anndata-friendly types pass
    through untouched.
    """
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_h5ad_uns(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        items = list(obj)
        if items and all(isinstance(e, dict) for e in items):
            key_sets = {tuple(sorted(e.keys())) for e in items}
            if len(key_sets) == 1:
                cols = sorted(items[0].keys())
                return {
                    str(c): [_sanitize_for_h5ad_uns(e[c]) for e in items]
                    for c in cols
                }
        return [_sanitize_for_h5ad_uns(e) for e in items]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _list_of_dicts_from_uns(value: Any) -> List[Dict[str, Any]]:
    """Reverse :func:`_sanitize_for_h5ad_uns` for a known list-of-dicts field.

    Accepts the three shapes that may appear after a sanitise / write /
    read round-trip:

    * a plain list-of-dicts (sanitiser left an empty or non-uniform list
      alone, or the value was never sanitised);
    * a dict-of-lists (sanitiser repacked a uniform list-of-dicts);
    * a dict-of-arrays (h5ad read-back of the dict-of-lists form).

    Returns ``[]`` for ``None`` or empty inputs.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [dict(x) if isinstance(x, dict) else x for x in value]
    if isinstance(value, dict):
        if not value:
            return []
        cols = list(value.keys())
        first = value[cols[0]]
        try:
            n = len(first)
        except TypeError:
            return []
        return [{k: value[k][i] for k in cols} for i in range(n)]
    return list(value)


def _restore_atypical_manifest_shapes(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Restore list-of-dicts shapes within a sanitised atypical manifest.

    Fields that were list-of-dicts pre-sanitise are restored so consumers
    such as :meth:`HECTOR._build_result_from_adata` (which calls
    ``list(...)`` and ``pd.DataFrame(...)`` on them) see the original
    shape regardless of whether the manifest was just written this
    session or loaded from disk via ``read_h5ad``.
    """
    if "community_stats" in manifest:
        manifest["community_stats"] = _list_of_dicts_from_uns(
            manifest["community_stats"]
        )
    if "g5_per_depth_summary" in manifest:
        manifest["g5_per_depth_summary"] = _list_of_dicts_from_uns(
            manifest["g5_per_depth_summary"]
        )
    bgmm_info = manifest.get("bgmm_info")
    if isinstance(bgmm_info, dict) and "component_report" in bgmm_info:
        bgmm_info = dict(bgmm_info)
        bgmm_info["component_report"] = _list_of_dicts_from_uns(
            bgmm_info["component_report"]
        )
        manifest["bgmm_info"] = bgmm_info
    return manifest


def _canonicalise_for_fingerprint(obj):
    """Normalise a fingerprint-like value for equality comparison.

    A fingerprint dict built in-memory contains Python lists/scalars, but
    after an h5ad write/read round-trip the same fields come back as
    numpy arrays / scalars (and dict keys become strings).  A bare
    ``dict != dict`` then triggers element-wise comparison on the
    embedded arrays and raises
    ``ValueError: The truth value of an array with more than one element
    is ambiguous``.

    This walker converts ``np.ndarray`` -> ``list``, ``np.generic`` ->
    Python scalar, and stringifies dict keys, so the two sides can be
    compared with ordinary ``==``.
    """
    if isinstance(obj, dict):
        return {str(k): _canonicalise_for_fingerprint(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [_canonicalise_for_fingerprint(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_canonicalise_for_fingerprint(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _fingerprints_equal(a, b) -> bool:
    """Return True iff two fingerprints are equal after canonicalisation.

    Safe to call when one side is the freshly built in-memory fingerprint
    (Python lists / scalars) and the other was reloaded from
    ``adata.uns`` after an h5ad round-trip (numpy arrays / scalars,
    stringified keys).
    """
    return _canonicalise_for_fingerprint(a) == _canonicalise_for_fingerprint(b)


# Fixed BGMM and community-gate parameters used by ``evaluate_cells``.
# They define component selection, multiple-testing control, and the bounded
# fitting subsample used by the current atypical-cell pipeline.
_ATYPICAL_FDR_ALPHA       = 0.05
_BGMM_N_COMPONENTS        = 10
_BGMM_MIN_WEIGHT          = 0.01
_BGMM_MAX_STD             = 0.20
_BGMM_TRAIN_SUBSAMPLE     = 10000
_ATYPICAL_OBS_COLUMNS: Tuple[str, ...] = (
    "abnormality",
    "bgmm_component",
    "is_aberrant",
    "snn_community",
    "snn_community_size",
    "community_fraction_aberrant",
    "community_hypergeom_q",
    "is_suspicious_community",
    "is_atypical",
    # Full 4-D abnormality feature stack (matches ``_ATYPICAL_FEATURE_NAMES``):
    # adherence, jsd, predicted_node_similarity,
    # predicted_class_similarity_margin.
    "adherence_score",
    "jsd",
    "predicted_node_similarity",
    "predicted_class_similarity_margin",
    # Pre-blend adherence components that feed the final ``adherence_score``
    # column when ``class_relative_adherence`` is enabled.
    # ``original_adherence_score`` is the raw adherence; the final
    # ``adherence_score`` equals it when class-relative adherence is
    # disabled or only a single class is predicted.
    # ``class_relative_adherence_score`` is the class-z sigmoid-rescaled
    # value before alignment+blending; NaN when class-relative adherence
    # is not applied.
    "original_adherence_score",
    "class_relative_adherence_score",
    # Auxiliary diagnostic feature computed alongside the 4-D abnormality
    # stack but NOT folded into ``abnormality`` itself. Exposed for user-side
    # exploration of the neighbourhood-of-confusion signal.
    "jsd_knn_std",
)
@dataclass(frozen=True)
class EvaluateCellsResult:
    """Result object returned by :meth:`HECTOR.evaluate_cells`.

    Per-cell fields live in ``cells`` as a DataFrame (one row per cell,
    index matches ``adata.obs_names``).  Per-community fields live in
    ``communities``.  Small metadata is grouped into ``summary`` /
    ``bgmm_info`` / ``snn_info``.

    The per-cell DataFrame is a convenience view over ``adata.obs`` — the
    authoritative store — so mutating the DataFrame does not affect the
    AnnData (they share values, not memory, because ``pd.DataFrame`` copies
    on construction from a dict of Series).
    """
    cells: pd.DataFrame
    communities: pd.DataFrame
    summary: Dict[str, Any]
    bgmm_info: Dict[str, Any]
    snn_info: Dict[str, Any]



# Number of profile-head candidates admitted to the consensus union mask.
PROFILE_TOPK = 15


def _topk_mask_tf(score: tf.Tensor, k: int) -> tf.Tensor:
    """Boolean ``[batch, n_classes]`` mask of the row-wise top-K columns.

    Graph-mode safe.  Each row is True at columns whose score reaches the
    k-th largest value in that row.  At exact ties on the boundary this
    may include slightly more than K classes — acceptable because the mask
    feeds a generous ``logical_or`` union in the consensus scorer.
    """

    top_values, _ = tf.nn.top_k(score, k=k)  # [batch, k]
    threshold = top_values[:, -1:]            # [batch, 1]
    return score >= threshold


def _knn_topk_cosine_full(
    query_norm: tf.Tensor, candidates_norm: tf.Tensor, k: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Single-shot cosine top-k over the FULL candidate set (no candidate tiling).

    Computes the dense ``[n_queries, n_candidates]`` cosine-similarity matrix in
    one matmul and selects the top-k candidates per query. Inputs must already be
    L2-normalized. Applies the same deterministic tie-break as the tiled path — a
    tiny bias that decreases with the global candidate index — so identical
    similarities resolve to the lower candidate index.

    This is the fast path used by :meth:`HECTOR._build_knn_graph_on_expression`
    when the full similarity matrix fits the GPU memory budget; the tiled loop
    remains the fallback for the constrained case.

    Args:
        query_norm: L2-normalized query rows ``[n_queries, n_features]``.
        candidates_norm: L2-normalized candidate rows ``[n_candidates, n_features]``.
        k: Number of neighbors per query.

    Returns:
        ``(values, indices)`` — top-k similarity values and candidate-relative
        indices, each ``[n_queries, k]`` (indices are ``int32``).
    """
    n_candidates = int(candidates_norm.shape[0])
    sim = tf.matmul(query_norm, candidates_norm, transpose_b=True)
    global_idx_bias = -1e-7 * tf.cast(tf.range(n_candidates), tf.float32) / float(max(n_candidates, 1))
    sim = sim + global_idx_bias[tf.newaxis, :]
    return tf.math.top_k(sim, k=k)


def _encoder_knn_cosine_cpu_topk(query_rows, candidates, k):
    """Exact cosine top-k over a separate candidate pool, per query row (CPU).

    The encoder OOM-fallback neighbor core. Unlike :func:`_grit_knn_cosine_cpu`
    (queries are indices INTO the candidate matrix, with an implicit self-edge),
    the encoder has a DISTINCT candidate pool — the anchor/memory-bank rows — so
    queries and candidates are separate matrices and the returned indices are
    candidate-relative (``0..n_candidates-1``). The caller shifts them into the
    combined ``[X, anchors]`` layout and appends the explicit ``(q, q)`` self-edge.

    Both sides are L2-normalized so the dense matmul is exact cosine similarity
    — the same metric as the GPU path. Applies the same ``-1e-7*idx/n_candidates``
    tie-break as the GPU/GRIT paths, so identical cosine similarities resolve to
    the lower
    candidate index.

    Args:
        query_rows: ``[n_queries, n_features]`` query expression (dense ndarray).
        candidates: ``[n_candidates, n_features]`` candidate (anchor) expression.
        k: Neighbors per query; clamped to ``n_candidates``.

    Returns:
        ``[n_queries, k']`` int64 candidate-relative neighbor indices, where
        ``k' = min(k, n_candidates)``. The order within a row is unspecified —
        the downstream graph aggregation (``unsorted_segment_sum``) is
        order-invariant; only the neighbor SET is contractual.
    """
    q = np.asarray(query_rows, dtype=np.float32)
    c = np.asarray(candidates, dtype=np.float32)
    n_candidates = int(c.shape[0])
    k = min(int(k), n_candidates)
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    cn = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    sim = qn @ cn.T
    sim += (
        -1e-7 * np.arange(n_candidates, dtype=np.float32) / float(max(n_candidates, 1))
    )[None, :]
    part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    return part.astype(np.int64)


def _grit_knn_cosine_cpu(
    X,
    query_indices,
    k,
    cand_tile=8192,
    query_tile=8192,
    silent=False,
):
    """Exact cosine k-NN for the GRIT graph on CPU (no GPU / no MLX).

    The dedicated CPU hardware path — NOT a GPU fallback. Streams candidate
    tiles against a resident query tile via tiled BLAS matmul, maintaining a
    per-query running top-k. The expression matrix is L2-normalized **once**
    (rows of the sparse CSR, into a fresh copy) and that normalized matrix is
    reused for both the query gather and every candidate tile, so no tile is
    ever re-normalized. Uses the same ``-1e-7 * global_idx / n_cells`` tie-break
    as the GPU single-shot/tiled paths, so identical cosine similarities resolve
    to the lower global candidate index. The query's own row stays in the
    candidate pool (implicit self-edge, cosine 1.0), matching the GRIT contract
    (``anchor_features is None``: candidates are ``X`` with no explicit
    self-loop).

    Exact only: no approximate ANN. At millions of cells this is inherently a
    long brute-force computation; a tqdm bar over query tiles reports ETA.

    Args:
        X: ``[n_cells, n_features]`` normalized gene expression (scipy.sparse
            CSR or dense ndarray). Not mutated.
        query_indices: 1-D array of query row indices, or ``None`` for all rows.
        k: Neighbors per query (clamped to ``n_cells - 1``).
        cand_tile: Candidate rows densified + matmul'd per inner step.
        query_tile: Query rows held resident per outer step.
        silent: Suppress the progress bar.

    Returns:
        ``(edge_index, edge_weights)`` numpy arrays — ``edge_index`` is
        ``[2, n_q * k]`` int32 (row 0 = global query index, row 1 = global
        candidate index); ``edge_weights`` is ``[n_q * k]`` float32 ones.
    """
    import scipy.sparse as _sp

    n_cells = int(X.shape[0])
    k = min(int(k), max(n_cells - 1, 1))
    if query_indices is None:
        q_idx = np.arange(n_cells, dtype=np.int64)
    else:
        q_idx = np.asarray(query_indices, dtype=np.int64)
    n_q = int(len(q_idx))
    if n_q == 0:
        return (
            np.zeros((2, 0), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    # --- Normalize ONCE. Reused for both query gather and candidate stream. ---
    if _sp.issparse(X):
        Xcsr = X.tocsr().astype(np.float32, copy=False)
        sq = Xcsr.multiply(Xcsr).sum(axis=1)          # [n_cells, 1] matrix
        row_norm = np.sqrt(np.asarray(sq).ravel())
        row_norm[row_norm == 0.0] = 1.0
        # Scale each row by 1/norm (broadcast a column vector); stays sparse.
        Xn = Xcsr.multiply((1.0 / row_norm)[:, None]).tocsr().astype(np.float32)
    else:
        Xn = np.asarray(X, dtype=np.float32)
        row_norm = np.linalg.norm(Xn, axis=1, keepdims=True)
        row_norm[row_norm == 0.0] = 1.0
        Xn = Xn / row_norm

    def _dense(block):
        return block.toarray() if _sp.issparse(block) else np.asarray(
            block, dtype=np.float32
        )

    edge_src = np.empty(n_q * k, dtype=np.int32)
    edge_dst = np.empty(n_q * k, dtype=np.int32)

    q_iter = range(0, n_q, query_tile)
    if not silent and n_q > query_tile:
        from tqdm import tqdm as _tqdm

        q_iter = _tqdm(
            q_iter,
            total=(n_q + query_tile - 1) // query_tile,
            desc="  kNN graph (CPU)",
            unit="tile",
            ncols=80,
        )

    for q0 in q_iter:
        q1 = min(q0 + query_tile, n_q)
        rows = q_idx[q0:q1]
        qn = _dense(Xn[rows])
        best_val = np.full((q1 - q0, k), -np.inf, dtype=np.float32)
        best_idx = np.zeros((q1 - q0, k), dtype=np.int64)
        for c0 in range(0, n_cells, cand_tile):
            c1 = min(c0 + cand_tile, n_cells)
            cn = _dense(Xn[c0:c1])
            sim = qn @ cn.T
            bias = -1e-7 * np.arange(c0, c1, dtype=np.float32) / float(
                max(n_cells, 1)
            )
            sim += bias[None, :]
            ck = min(k, c1 - c0)
            part = np.argpartition(-sim, ck - 1, axis=1)[:, :ck]
            part_val = np.take_along_axis(sim, part, axis=1)
            part_idx = part.astype(np.int64) + c0
            all_val = np.concatenate([best_val, part_val], axis=1)
            all_idx = np.concatenate([best_idx, part_idx], axis=1)
            sel = np.argpartition(-all_val, k - 1, axis=1)[:, :k]
            best_val = np.take_along_axis(all_val, sel, axis=1)
            best_idx = np.take_along_axis(all_idx, sel, axis=1)
        for r in range(q1 - q0):
            base = (q0 + r) * k
            edge_src[base:base + k] = rows[r]
            edge_dst[base:base + k] = best_idx[r]

    edge_index = np.stack([edge_src, edge_dst], axis=0)
    edge_weights = np.ones(edge_index.shape[1], dtype=np.float32)
    return edge_index, edge_weights


def _grit_residency_plan(
    avail_bytes,
    n_q,
    n_features,
    k,
    cand_tile=8192,
    query_tile=8192,
    safety=0.85,
):
    """Decide the resident query super-tile and tile sizes from measured VRAM.

    The GRIT reorder densifies each library tile once **per resident query
    super-tile**. So the number of times the library re-streams is
    ``S = ceil(n_q / query_super)``, where ``query_super`` is how many query
    rows fit resident in VRAM alongside the transient working set (one densified
    library tile + one similarity block + the running top-k buffers). When the
    whole query set fits, ``S == 1`` and the library is densified exactly once.

This sizing calculation performs no GPU calls. The caller supplies a measured,
arena-aware ``avail_bytes`` value and must respect the returned tile bounds.
An OOM guard at the call site halves the bounds and retries if the measurement
was optimistic.

    Args:
        avail_bytes: Measured free VRAM in bytes (arena-aware).
        n_q: Number of query rows.
        n_features: Feature dimension.
        k: Neighbors per query.
        cand_tile: Desired library tile size (rows densified per step).
        query_tile: Desired inner query sub-tile size (bounds the sim block).
        safety: Fraction of ``avail_bytes`` we allow ourselves to use.

    Returns:
        ``{"query_super", "S", "cand_tile", "query_tile"}``.
    """
    budget = max(int(avail_bytes * safety), 1)
    ct = max(1, min(int(cand_tile), 1 << 20))
    qt = max(1, min(int(query_tile), int(n_q)))

    def _transient(ct_, qt_):
        return (
            ct_ * n_features * 4          # one densified library tile (fp32)
            + qt_ * ct_ * 4               # one similarity block
            + n_q * k * 8                 # running top-k (vals f32 + idx i32)
        )

    # Shrink tiles until the transient working set alone fits the budget.
    while _transient(ct, qt) > budget and (ct > 256 or qt > 256):
        if ct >= qt and ct > 256:
            ct = max(256, ct // 2)
        elif qt > 256:
            qt = max(256, qt // 2)
        else:
            break

    resident_budget = max(budget - _transient(ct, qt), n_features * 4)
    query_super = int(resident_budget // (n_features * 4))
    query_super = max(qt, min(query_super, int(n_q)))
    S = -(-int(n_q) // query_super)       # ceil division
    return {"query_super": query_super, "S": S, "cand_tile": ct, "query_tile": qt}


def _grit_knn_gpu_reordered(
    X,
    query_indices,
    k,
    available_bytes_fn,
    cand_tile=8192,
    query_tile=8192,
    silent=False,
):
    """Reordered GPU cosine k-NN for the GRIT graph (sparse-stream + GPU densify).

    Library tiles are the OUTER loop: each is streamed to the GPU as a compact
    CSR slice and densified there ONCE via ``tf.scatter_nd`` (the L2-normalized
    library is built once on CPU; ``tf.sparse.sparse_dense_matmul`` is avoided —
    it raises ``n_q*nnz > 2^31`` on GPU). A resident query super-tile (sized from
    a measured, arena-aware VRAM probe via :func:`_grit_residency_plan`) is
    sliced into inner query sub-tiles to bound the similarity block; a per-query
    running top-k is merged tile by tile. Same ``-1e-7*idx/n_cells`` tie-break
    and implicit self-edge as :func:`_grit_knn_cosine_cpu`, so the neighbor SET
    matches the exact core.

    Args:
        X: ``[n_cells, n_features]`` normalized expression (scipy.sparse CSR or
            dense ndarray). Not mutated.
        query_indices: 1-D query row indices, or ``None`` for all rows.
        k: Neighbors per query.
        available_bytes_fn: Zero-arg callable returning measured free VRAM bytes.
        cand_tile: Desired library tile size.
        query_tile: Desired inner query sub-tile size.
        silent: Suppress the tqdm progress bar.

    Returns:
        ``(edge_index, edge_weights)`` numpy arrays — ``edge_index`` ``[2,n_q*k]``
        int32 (row 0 query, row 1 candidate), ``edge_weights`` ``[n_q*k]`` ones.
    """
    import scipy.sparse as _sp

    n_cells = int(X.shape[0])
    n_features = int(X.shape[1])
    k = min(int(k), max(n_cells - 1, 1))
    if query_indices is None:
        q_idx = np.arange(n_cells, dtype=np.int64)
    else:
        q_idx = np.asarray(query_indices, dtype=np.int64)
    n_q = int(len(q_idx))
    if n_q == 0:
        return (
            np.zeros((2, 0), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    # --- Normalize the library ONCE (sparse-friendly), reused for q + cand. ---
    if _sp.issparse(X):
        Xcsr = X.tocsr().astype(np.float32, copy=False)
        rn = np.sqrt(np.asarray(Xcsr.multiply(Xcsr).sum(axis=1)).ravel())
        rn[rn == 0.0] = 1.0
        Xn = Xcsr.multiply((1.0 / rn)[:, None]).tocsr().astype(np.float32)
    else:
        Xd = np.asarray(X, dtype=np.float32)
        rn = np.linalg.norm(Xd, axis=1, keepdims=True)
        rn[rn == 0.0] = 1.0
        Xn = _sp.csr_matrix(Xd / rn)

    # Decide residency and tile sizes from the current VRAM estimate.
    avail = available_bytes_fn() or (4 << 30)
    plan = _grit_residency_plan(
        avail, n_q, n_features, k, cand_tile=cand_tile, query_tile=query_tile
    )
    QSUP, QT, CT = plan["query_super"], plan["query_tile"], plan["cand_tile"]

    def _densify_tile_gpu(c0, c1):
        """CSR slice -> GPU scatter_nd dense [c1-c0, n_features]."""
        tile = Xn[c0:c1]
        rows = np.repeat(
            np.arange(c1 - c0, dtype=np.int32), np.diff(tile.indptr)
        )
        idx = tf.stack(
            [tf.constant(rows), tf.constant(tile.indices.astype(np.int32))],
            axis=1,
        )
        return tf.scatter_nd(
            idx, tf.constant(tile.data.astype(np.float32)), [c1 - c0, n_features]
        )

    edge_src = np.empty(n_q * k, dtype=np.int32)
    edge_dst = np.empty(n_q * k, dtype=np.int32)

    # Advance progress over library-tile work units for every resident query block.
    n_lib_tiles = (n_cells + CT - 1) // CT
    n_super = (n_q + QSUP - 1) // QSUP
    pbar = None
    if not silent:
        from tqdm import tqdm as _tqdm

        pbar = _tqdm(
            total=n_super * n_lib_tiles,
            desc="  kNN graph",
            unit="tile",
            ncols=80,
        )

    # Similarity-matmul precision follows the package-level TF32 setting. TF32 is
    # disabled by default for cross-backend fp32 consistency and can be enabled
    # globally with HECTOR_TF32=1.
    try:
        for s0 in range(0, n_q, QSUP):
            s1 = min(s0 + QSUP, n_q)
            srows = q_idx[s0:s1]
            # Resident query super-tile (normalized rows gathered to dense).
            Qres = tf.identity(
                tf.constant(Xn[srows].toarray(), dtype=tf.float32)
            )
            nqs = s1 - s0
            # Per-(inner query tile) running top-k buffers, kept across tiles.
            qstarts = list(range(0, nqs, QT))
            TV = [tf.fill([min(j + QT, nqs) - j, k], -np.inf) for j in qstarts]
            TI = [tf.zeros([min(j + QT, nqs) - j, k], dtype=tf.int32) for j in qstarts]

            for c0 in range(0, n_cells, CT):
                c1 = min(c0 + CT, n_cells)
                cand = _densify_tile_gpu(c0, c1)            # GPU densify ONCE
                bias = (-1e-7 * tf.cast(tf.range(c0, c1), tf.float32)
                        / float(max(n_cells, 1)))[tf.newaxis, :]
                ck = min(k, c1 - c0)
                for j, qj in enumerate(qstarts):
                    qj1 = min(qj + QT, nqs)
                    sim = tf.matmul(Qres[qj:qj1], cand, transpose_b=True) + bias
                    tvk, tlk = tf.math.top_k(sim, k=ck)
                    tik = tlk + c0
                    if ck < k:
                        pad = k - ck
                        tvk = tf.concat(
                            [tvk, tf.fill([qj1 - qj, pad], -np.inf)], axis=1
                        )
                        tik = tf.concat(
                            [tik, tf.zeros([qj1 - qj, pad], dtype=tf.int32)], axis=1
                        )
                    cv = tf.concat([TV[j], tvk], axis=1)
                    ci = tf.concat([TI[j], tik], axis=1)
                    _, sel = tf.math.top_k(cv, k=k)
                    b = tf.repeat(tf.range(qj1 - qj)[:, None], k, axis=1)
                    g = tf.stack([b, sel], axis=2)
                    TV[j] = tf.gather_nd(cv, g)
                    TI[j] = tf.gather_nd(ci, g)
                if pbar is not None:
                    pbar.update(1)

            # Emit edges for this super-tile.
            best = np.concatenate([t.numpy() for t in TI], axis=0)  # [nqs, k]
            for r in range(nqs):
                base = (s0 + r) * k
                edge_src[base:base + k] = srows[r]
                edge_dst[base:base + k] = best[r]
    finally:
        if pbar is not None:
            pbar.close()

    edge_index = np.stack([edge_src, edge_dst], axis=0)
    edge_weights = np.ones(edge_index.shape[1], dtype=np.float32)
    return edge_index, edge_weights


@dataclass
class _PredictionSettings:
    """Resolved runtime settings for a single prediction call."""

    top_k: int
    use_full_ontology: bool
    use_grit: bool
    grit_iterations: int
    grit_alpha: float
    grit_entropy_threshold: Optional[float]
    grit_refinement_percentile: Optional[float]
    filter_low_quality: bool
    variance_threshold: Optional[float]
    use_asymmetric_ppr: bool
    forward_weight: float
    ppr_alpha: Optional[float]

    def cache_signature(self) -> Dict[str, Any]:
        """Return the prediction-fingerprint subset of runtime settings."""
        return {
            "version": _PREDICTION_FINGERPRINT_VERSION,
            "use_full_ontology": self.use_full_ontology,
            "use_grit": self.use_grit,
            "grit_iterations": self.grit_iterations,
            "grit_alpha": self.grit_alpha,
            "grit_entropy_threshold": self.grit_entropy_threshold,
            "grit_refinement_percentile": self.grit_refinement_percentile,
            "filter_low_quality": self.filter_low_quality,
            "variance_threshold": self.variance_threshold,
            "use_asymmetric_ppr": self.use_asymmetric_ppr,
            "forward_weight": self.forward_weight,
            "ppr_alpha": self.ppr_alpha,
        }


@dataclass
class _PredictionCoreResult:
    """Shared raw prediction outputs used by public inference entry points."""
    top_indices: np.ndarray
    top_ids: np.ndarray
    top_scores: np.ndarray
    prediction_ids: np.ndarray
    prediction_scores: np.ndarray
    cell_variance_array: np.ndarray
    low_quality_mask: Optional[np.ndarray]
    score_matrix_array: Optional[np.ndarray]
    pre_grit_score_matrix_array: Optional[np.ndarray] = None
    transient_cleanup: Optional[Callable[[], None]] = None

    def close_transient_arrays(self) -> None:
        if self.transient_cleanup is None:
            return
        cleanup = self.transient_cleanup
        self.transient_cleanup = None
        cleanup()


@dataclass
class _PredictionSession:
    """Resolved top-1 prediction state shared across workflows."""

    cell_vectors: np.ndarray
    settings: _PredictionSettings
    core_result: _PredictionCoreResult
    prediction_fingerprint: Dict[str, Any]
    source: str
    captured_pre_grit_scores: bool
    has_public_annotation: bool = False



# ============================================================================
# UMAP backend selection
# ============================================================================

def _select_umap_backend() -> str:
    """Pick the UMAP engine by available hardware.

    MLX (Apple Silicon + ``mlx`` importable) -> cuML (NVIDIA + importable)
    -> scanpy (CPU). Imports are probed lazily and guarded so this is safe on
    any platform.
    """
    import sys

    if sys.platform == 'darwin':
        try:
            import mlx.core  # noqa: F401
            return 'mlx'
        except Exception:
            pass
    try:
        import cuml  # noqa: F401
        return 'cuml'
    except Exception:
        pass
    return 'scanpy'


def _select_encoder_backend() -> str:
    """Select encoder backend: 'mlx' on Apple Silicon, else 'tf'."""
    import sys
    if sys.platform == 'darwin':
        try:
            import mlx.core  # noqa: F401
            return 'mlx'
        except Exception:
            pass
    return 'tf'


# ============================================================================
# Preflight peak-memory guard
# ----------------------------------------------------------------------------
# Combines live model and AnnData dimensions with system memory, then calls the
# dependency-light estimator and warns when projected demand exceeds availability.
# ============================================================================

_GB = 1024 ** 3


def _count_reordered_nnz(selected_matrix, selected_gene_ids, model_gene_ids) -> int:
    """Exact count of stored non-zeros in the columns matching the model's genes.

    Reuses HECTOR's ``_match_gene_columns`` (the same matcher
    ``reorder_and_fill_genes`` uses) to find the kept source columns, then counts
    via a CHUNKED membership sum over the column-index array -- bounded transient
    regardless of index dtype (``np.bincount`` would upcast int32 indices, a full
    ``nnz*8`` copy). Returns ``m.nnz`` (conservative) if IDs cannot be located.
    """
    import scipy.sparse as sp
    from .predictor_support import _match_gene_columns

    if not sp.issparse(selected_matrix):
        return int(np.count_nonzero(np.asarray(selected_matrix)))
    m = selected_matrix if selected_matrix.format == "csr" else selected_matrix.tocsr()
    if selected_gene_ids is None:
        return int(m.nnz)
    source_indices, _tgt, n_found, _n_missing = _match_gene_columns(
        selected_gene_ids, model_gene_ids
    )
    if n_found == 0:
        return 0
    kept_mask = np.zeros(int(m.shape[1]), dtype=bool)
    kept_mask[source_indices] = True
    idx = m.indices
    total = 0
    chunk = 8_000_000
    for start in range(0, idx.size, chunk):
        total += int(kept_mask[idx[start:start + chunk]].sum())
    return total


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment; fall back on missing/unparseable."""
    import os
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _has_cuda_gpu() -> bool:
    """Return whether TensorFlow can see a CUDA GPU."""
    try:
        import tensorflow as tf
        return bool(tf.config.list_physical_devices("GPU"))
    except Exception:
        return False


def _describe_tf_device() -> str:
    """Human-readable name of the hardware the TF path will actually use.

    Returns the GPU's marketing name when TensorFlow can see a CUDA device
    (falling back to a bare ``"GPU"`` when the name is unavailable), otherwise
    ``"CPU - no GPU detected"``. Cosmetic only: it feeds the model-load log line
    so users can confirm that a ``[cuda12]`` install really engaged the card,
    which ``nvidia-smi`` cannot tell them. Never raises.
    """
    if not _has_cuda_gpu():
        return "CPU (no GPU detected)"
    try:
        import tensorflow as tf
        device = tf.config.list_physical_devices("GPU")[0]
        name = tf.config.experimental.get_device_details(device).get("device_name")
        return f"GPU ({name})" if name else "GPU"
    except Exception:
        return "GPU"


def _refine_backend(backend: str) -> str:
    """Refine the 'tf' backend to 'cuda' vs 'cpu'; pass 'mlx' through."""
    if backend == "tf":
        return "cuda" if _has_cuda_gpu() else "cpu"
    return backend


def _read_memory(backend: str):
    """(resident, available) host RAM bytes at call time. BACKEND SEAM: MLX reads
    the macOS 'available' figure; host RAM elsewhere via psutil."""
    import psutil
    resident = int(psutil.Process().memory_info().rss)
    available = int(psutil.virtual_memory().available)
    if backend == "mlx":
        from . import predictor_support as ps
        macos_avail = ps._get_macos_available_bytes()
        if macos_avail is not None:
            available = int(macos_avail)
    return resident, available


def _encoder_terms(model, backend, n_genes, n_anchors):
    """(fixed_floor, per_cell, latent_dim) for the encoder. BACKEND SEAM: MLX
    uses the Metal-calibrated estimator; the TF/CPU path uses the generic one."""
    from . import predictor_support as ps
    if backend == "mlx":
        cfg = getattr(model, "_mlx_config", None) or {}
        hidden = cfg.get("hidden_dims") or []
        shared_dim = hidden[-1] if hidden else cfg.get("shared_dim", 1024)
        latent_dim = int(getattr(getattr(model, "_mlx_encoder", None), "latent_dim", 768))
        floor, per_cell = ps._estimate_encoder_memory_terms_metal(
            n_genes=n_genes, n_anchors=n_anchors, shared_dim=shared_dim,
        )
    else:
        enc = getattr(model, "vgae_encoder", None)
        if enc is None:
            raise ValueError(
                "backend is not 'mlx' but model.vgae_encoder is None -- this "
                "model was loaded on the MLX path. Use backend='mlx'."
            )
        latent_dim = int(enc.latent_dim)
        floor, per_cell = ps._estimate_encoder_memory_terms(
            n_genes=n_genes, shared_dim=enc.shared_dim,
            k_neighbors=model.config.k_neighbors,
            graph_attention_type=enc.graph_attention_type,
            latent_dim=latent_dim, n_anchors=n_anchors,
        )
    return int(floor), int(per_cell), int(latent_dim)


def _embeddings_cached_via_model(model, adata) -> bool:
    """True iff HECTOR would skip the encoder for this adata. Delegates to the
    model's own ``_embeddings_cache_valid`` (shared with compute_cell_embeddings);
    conservative -- any missing method or error means 'not cached'."""
    fn = getattr(model, "_embeddings_cache_valid", None)
    if fn is None:
        return False
    try:
        return bool(fn(adata))
    except Exception:
        return False


def run_preflight(model, adata, *, use_grit=None, use_full_ontology=None,
                  batch_size=None, backend=None, safety_fraction=None,
                  projection_margin=None, verbose=True):
    """Project peak memory for ``model.predict(adata)`` and (optionally) print a
    one-line warning. Pure-advisory: reads only, never mutates ``adata``.

    Returns a ``predictor_support.PreflightReport``.
    """
    from . import predictor_support as ps

    cfg = model.config
    use_grit = cfg.use_grit if use_grit is None else use_grit
    use_full_ontology = cfg.use_full_ontology if use_full_ontology is None else use_full_ontology
    batch_size = batch_size or getattr(cfg, "batch_size", None) or 2040
    backend = _refine_backend(backend or _select_encoder_backend())
    if projection_margin is None:
        projection_margin = _env_float("HECTOR_PREFLIGHT_MARGIN", 1.3)
    projection_margin = max(1.0, float(projection_margin))
    if safety_fraction is None:
        safety_fraction = _env_float("HECTOR_PREFLIGHT_SAFETY_FRACTION", 0.85)

    selected = ps.resolve_expression_source(adata, allow_log1p=True, return_candidates=True)
    selected_name = selected["name"]
    selected_matrix = selected["matrix"]
    selected_var = selected["var"]
    unused = sorted(
        [(c["name"], c["nbytes"], f"HECTOR reads raw counts from {selected_name}")
         for c in selected["candidates"] if c["name"] != selected_name],
        key=lambda t: -t[1],
    )

    n_cells = int(adata.n_obs)
    model_gene_ids = list(model.gene_ids)
    n_genes = int(model.num_genes)
    n_anchors = int(model.anchor_features.shape[0]) if model.anchor_features is not None else 0
    n_active_classes = (len(model.celltype_gat.full_classes) if use_full_ontology
                        else len(model.celltype_gat.classes))

    floor, per_cell, latent_dim = _encoder_terms(model, backend, n_genes, n_anchors)
    selected_gene_ids = ps._find_ensembl_id_array_in_var(selected_var)
    nnz_reordered = _count_reordered_nnz(selected_matrix, selected_gene_ids, model_gene_ids)
    embeddings_cached = _embeddings_cached_via_model(model, adata)
    resident, available = _read_memory(backend)

    if backend == "cuda":
        # anchor .numpy() backup (n_anchors x n_genes x 4) + ~4 GB TF/CUDA host
        # staging & glibc arena; GRIT holds ~3 host copies of the score matrix.
        fixed_host_overhead = n_anchors * n_genes * 4 + 4 * _GB
        grit_score_multiplier = 3
    else:
        fixed_host_overhead = 0
        grit_score_multiplier = 0

    report = ps.estimate_prediction_memory(
        n_cells=n_cells, nnz_reordered=nnz_reordered, latent_dim=latent_dim,
        n_active_classes=n_active_classes, encoder_fixed_floor=floor,
        encoder_per_cell=per_cell, batch_size=batch_size,
        k_neighbors=cfg.k_neighbors, use_grit=use_grit,
        embeddings_cached=embeddings_cached, backend=backend,
        available_bytes=available, resident_bytes=resident,
        unused_copy_sizes=unused, safety_fraction=safety_fraction,
        projection_margin=projection_margin,
        fixed_host_overhead=int(fixed_host_overhead),
        grit_score_multiplier=grit_score_multiplier,
    )
    if verbose and report.message:
        print(report.message)
    return report


def _select_tsne_backend() -> str:
    """Pick the t-SNE engine by hardware: MLX (Apple Silicon) -> cuML (NVIDIA)
    -> scanpy (CPU). Mirrors _select_umap_backend; imports are probed lazily."""
    import sys

    if sys.platform == 'darwin':
        try:
            import mlx.core  # noqa: F401
            return 'mlx'
        except Exception:
            pass
    try:
        import cuml  # noqa: F401
        return 'cuml'
    except Exception:
        pass
    return 'scanpy'


# Per-backend layout defaults. Shared parameters such as ``n_neighbors``,
# ``n_pcs``, and ``random_state`` are handled by ``reduce_dimensions()``.
_UMAP_BACKEND_DEFAULTS = {
    'mlx': {
        'min_dist': 1.0, 'spread': 1.0, 'n_epochs': 300,
        'init': None, 'negative_sample_rate': 5,
    },
    'cuml': {
        # A spectral request uses a deterministic PCA-2D initialization array on
        # cuML versions that accept array initialization; other versions use the
        # Scanpy path.
        'min_dist': 0.5, 'spread': 1.0, 'n_epochs': 200,
        'init': 'spectral', 'negative_sample_rate': 5,
    },
    'scanpy': {
        'min_dist': 0.5, 'spread': 1.0, 'n_epochs': 200,
        'init': 'spectral', 'negative_sample_rate': 5,
    },
}


def _cuml_version():
    """Return the installed cuML ``(major, minor)`` version, or ``None`` if cuML
    is not importable.

    Used to gate GPU-only features. cuML's UMAP gained array-like ``init``
    support around release 26.04 (documented in 26.06); older cuML only accepts
    the string inits ``'spectral'``/``'random'`` and raises on an array.
    """
    try:
        import cuml
        parts = str(cuml.__version__).split('.')
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return None


def _pca2d_init(rep, random_state):
    """Deterministic 2-D PCA initialization for UMAP (umap-learn style).

    Seeds the low-dim layout with the top-2 principal components of ``rep``,
    rescaled to max-abs 10 with tiny jitter (mirrors umap-learn's handling of a
    user/spectral init). Gives the clean global structure of a spectral init
    without cuML's spectral-eigendecomposition spikes, and is fully
    reproducible. Robust to degenerate shapes (few samples / features).
    """
    from sklearn.decomposition import PCA

    seed = int(random_state) if random_state is not None else 0
    rep = np.asarray(rep, dtype=np.float32)
    n_samples, n_features = rep.shape
    n_comp = min(2, n_features, max(1, n_samples - 1))
    pcs = (
        PCA(n_components=n_comp, svd_solver='randomized',
            random_state=seed).fit_transform(rep)
        if n_comp >= 1 else np.zeros((n_samples, 0), dtype=np.float32)
    )
    if pcs.shape[1] < 2:  # pad to exactly 2 columns for degenerate inputs
        pcs = np.hstack(
            [pcs, np.zeros((n_samples, 2 - pcs.shape[1]), dtype=np.float32)]
        )
    pcs = np.asarray(pcs, dtype=np.float32)
    scale = float(np.abs(pcs).max())
    if scale > 0.0:
        pcs = 10.0 * pcs / scale
    pcs = pcs + np.random.default_rng(seed).normal(scale=1e-4, size=pcs.shape)
    return np.ascontiguousarray(pcs, dtype=np.float32)


# ============================================================================
# Main Predictor Class
# ============================================================================

class HECTOR:
    """
    Predictor for HECTOR models.
    
    This class handles all aspects of prediction:
    1. Loading the model and extracting gene order
    2. Preprocessing new data (gene reordering, normalization)
    3. Building k-NN graphs
    4. Making predictions using gated inference
    5. Formatting results
    
    Example:
        predictor = HECTOR("human")
        predictions = predictor.predict(adata, top_k=5)
        predictor.write_predictions(adata, predictions)
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config: Optional[InferenceConfig] = None,
        verbose: bool = True,
        auto_download: bool = True,
        **config_kwargs
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model checkpoint (.h5 file) or a registered
                        species key such as "human"
            config: InferenceConfig object (uses defaults if None)
            verbose: Whether to print progress messages
            auto_download: Whether to download a missing registered model automatically
            **config_kwargs: Additional config parameters to override defaults
                           (e.g., top_k=10, batch_size=256, use_grit=True)
                           
        Example:
            # Using default config
            predictor = HECTOR("model.h5")
            
            # Overriding specific parameters
            predictor = HECTOR("model.h5", top_k=10, batch_size=256)
            
            # Using custom config object
            config = InferenceConfig(top_k=10, batch_size=256)
            predictor = HECTOR("model.h5", config=config)
        """
        self.requested_model = str(model_path)
        resolved_model_path = get_model_path(model_path, auto_download=auto_download)
        self.model_path = str(resolved_model_path)
        self.auto_download = auto_download

        # Call stable_environments to set deterministic behavior and seeds
        stable_environments()
        
        # Create config from kwargs if not provided
        if config is None:
            self.config = InferenceConfig(**config_kwargs)
        else:
            # If config is provided, kwargs are ignored
            if config_kwargs:
                warnings.warn(
                    "Both 'config' and config kwargs provided. "
                    "Config kwargs will be ignored. Use either config object OR kwargs, not both."
                )
            self.config = config
        
        self.verbose = verbose
        
        # Model components (loaded in _load_model)
        self.vgae_encoder = None
        self.vgae_decoder = None
        self.celltype_gat = None
        self.aux_classifier = None  # HPL head
        self._mlx_encoder = None  # Lazy-loaded MLX encoder (Apple Silicon)
        self._mlx_config = None   # Config dict from load_mlx_encoder_from_h5
        self._mlx_floor_bytes = None  # Cached measured encode_hidden(anchors) peak
        self._mlx_celltype_embedder = None
        self._mlx_hpl_head = None
        self._mlx_procrustes = None
        self._mlx_ontology = None

        # Model metadata
        self.gene_ids = None  # Required gene IDs in correct order
        self.num_genes = None
        self.normalization_config = None
        self.checkpoint_version = None
        
        # Load model
        self._load_model()
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _check_cache_valid(self, adata, key: str, location: str = 'obsm', expected_rows: Optional[int] = None) -> bool:
        """Check if a cache signal exists and has the correct row count.

        Args:
            adata: AnnData object to inspect.
            key: Cache key to look up (e.g. ``'X_hector'`` or
                ``'hector_cell_variance'``).
            location: ``'obsm'`` to check ``adata.obsm``, ``'obs'`` to check
                ``adata.obs`` columns.
            expected_rows: Expected number of rows.  Defaults to
                ``adata.shape[0]`` when *None*.

        Returns:
            ``True`` if the key is present and its row count matches
            *expected_rows*; ``False`` otherwise.
        """
        if expected_rows is None:
            expected_rows = adata.shape[0]

        if location == 'obsm':
            return key in adata.obsm and adata.obsm[key].shape[0] == expected_rows
        elif location == 'obs':
            return key in adata.obs.columns and len(adata.obs[key]) == expected_rows
        return False

    def _parse_ontology_id(self, prediction_str: str) -> Optional[str]:
        """Parse an ontology ID from any label format.

        Handles three formats produced by :meth:`_format_prediction`:

        * **combined** – ``"monocyte (CL:0000128)"`` → ``"CL:0000128"``
        * **plain ID** – ``"CL:0000128"`` → ``"CL:0000128"``
        * **name-only** – ``"monocyte"`` → reverse-lookup via
          :attr:`id_to_name_map`

        Args:
            prediction_str: A prediction label string in any of the
                supported formats.

        Returns:
            The raw ontology ID (e.g. ``"CL:0000128"``), or ``None`` if
            the string cannot be mapped to a known ID.
        """
        # 1. Combined format: "name (ID)"
        m = re.search(r'\(([A-Z]+:\d+)\)$', prediction_str)
        if m:
            return m.group(1)

        # 2. Plain ID format: "CL:0000128", "UBERON:0001234", etc.
        if re.match(r'^[A-Z]+:\d+$', prediction_str):
            return prediction_str

        # 3. Name-only format: reverse lookup (cached)
        rmap = getattr(self, '_name_to_id_cache', None)
        if rmap is None:
            rmap = {name: ont_id for ont_id, name in self.id_to_name_map.items()}
            self._name_to_id_cache = rmap
        return rmap.get(prediction_str)

    def _resolve_class_labels(self, labels, valid_ids):
        """Resolve cell-type labels to model class IDs, accepting class IDs,
        readable names, or the combined ``"name (ID)"`` format.

        Reuses :meth:`_parse_ontology_id` for the per-label ID/name parsing, then
        keeps only results that are in ``valid_ids`` (the active class set).
        Labels already equal to a valid class ID pass through directly, so the
        ``_parse_ontology_id`` name branch (which needs ``id_to_name_map``) is
        skipped when unneeded. Unresolvable / out-of-set labels map to None.

        Args:
            labels: iterable of label strings.
            valid_ids: iterable of acceptable class IDs.

        Returns:
            list aligned with ``labels`` of class-ID strings or None.
        """
        valid = set(valid_ids)
        out = []
        for raw in labels:
            s = str(raw)
            if s in valid:
                out.append(s)
                continue
            oid = self._parse_ontology_id(s)
            out.append(oid if oid in valid else None)
        return out

    def _ensure_cache_writable(self, adata: anndata.AnnData) -> None:
        """Materialize AnnData views before writing cache signals.

        Creates a lightweight AnnData that shares the large X/raw/layers
        arrays without copying them and deep-copies only the smaller metadata
        containers (``obs``, ``var``, ``obsm``, and ``uns``).
        """
        if not getattr(adata, "is_view", False):
            return
        if hasattr(adata, "_init_as_actual") and hasattr(adata, "X"):
            import anndata as _ad
            lightweight = _ad.AnnData(
                X=adata.X,
                obs=adata.obs.copy(),
                var=adata.var.copy(),
                obsm=dict(adata.obsm),
                uns=dict(adata.uns),
                layers=dict(adata.layers) if adata.layers else {},
            )
            if adata.raw is not None:
                lightweight.raw = _ad.AnnData(
                    X=adata.raw.X, var=adata.raw.var,
                )
            adata._init_as_actual(lightweight)
        else:
            # _FakeAnnData or non-standard AnnData without _init_as_actual
            adata.obs = adata.obs.copy()
            if hasattr(adata, "obsm"):
                adata.obsm = dict(adata.obsm)
            if hasattr(adata, "uns"):
                adata.uns = dict(adata.uns)
            adata.is_view = False

    def _get_model_identity(self) -> Optional[str]:
        """Return the model identifier used for reusable cache manifests."""

        return getattr(self, "requested_model", None)

    def _fingerprint_cell_order(self, adata: anndata.AnnData) -> str:
        """Return a stable fingerprint of AnnData cell names and order."""

        digest = hashlib.sha256()
        for obs_name in adata.obs_names.tolist():
            encoded = str(obs_name).encode("utf-8", errors="surrogatepass")
            digest.update(
                len(encoded).to_bytes(8, byteorder="little", signed=False)
            )
            digest.update(encoded)
        return digest.hexdigest()

    def _build_embedding_manifest(
        self,
        adata: anndata.AnnData,
        source_slot: str,
    ) -> Dict[str, Any]:
        """Build the reusable embedding manifest stored in ``adata.uns``."""

        return {
            "version": _EMBEDDING_MANIFEST_VERSION,
            "model_identity": self._get_model_identity(),
            "checkpoint_version": getattr(self, "checkpoint_version", None),
            "source_slot": source_slot,
            "cell_order_fingerprint": self._fingerprint_cell_order(adata),
            "n_cells": int(adata.shape[0]),
        }

    def _get_embedding_manifest(
        self,
        adata: anndata.AnnData,
    ) -> Optional[Dict[str, Any]]:
        """Return a copy of the cached embedding manifest if present."""

        manifest = adata.uns.get(_EMBEDDING_MANIFEST_KEY)
        if isinstance(manifest, dict):
            return dict(manifest)
        return None

    def _write_embedding_manifest(
        self,
        adata: anndata.AnnData,
        manifest: Dict[str, Any],
    ) -> None:
        """Persist the embedding manifest in ``adata.uns``."""

        self._ensure_cache_writable(adata)
        adata.uns[_EMBEDDING_MANIFEST_KEY] = _sanitize_for_h5ad_uns(dict(manifest))

    def _embedding_manifest_matches(
        self,
        adata: anndata.AnnData,
        manifest: Optional[Dict[str, Any]],
        *,
        source_slot: Optional[str] = None,
    ) -> bool:
        """Validate an embedding manifest against the current AnnData state."""

        if not isinstance(manifest, dict):
            return False
        if manifest.get("version") != _EMBEDDING_MANIFEST_VERSION:
            return False
        if not isinstance(source_slot, str):
            return False

        expected_manifest = self._build_embedding_manifest(
            adata,
            source_slot=source_slot,
        )
        for key in (
            "model_identity",
            "checkpoint_version",
            "source_slot",
            "cell_order_fingerprint",
            "n_cells",
        ):
            # ``_fingerprints_equal`` so this stays correct if any of these
            # fields ever holds a list / numpy array (after an h5ad round
            # trip a Python list comes back as ``np.ndarray``).
            if not _fingerprints_equal(
                manifest.get(key), expected_manifest[key]
            ):
                return False
        return True

    def _resolve_prediction_settings(
        self,
        top_k: Optional[int] = None,
        use_full_ontology: Optional[bool] = None,
        use_grit: Optional[bool] = None,
        grit_iterations: Optional[int] = None,
        grit_alpha: Optional[float] = None,
        grit_entropy_threshold: Optional[float] = None,
        grit_refinement_percentile: Optional[float] = None,
        use_asymmetric_ppr: Optional[bool] = None,
        forward_weight: Optional[float] = None,
        ppr_alpha: Optional[float] = None,
        filter_low_quality: Optional[bool] = None,
        variance_threshold: Optional[float] = None,
    ) -> _PredictionSettings:
        """Merge runtime overrides with config defaults."""

        top_k = top_k if top_k is not None else self.config.top_k
        use_full_ontology = (
            use_full_ontology
            if use_full_ontology is not None
            else self.config.use_full_ontology
        )
        use_grit = use_grit if use_grit is not None else self.config.use_grit
        grit_iterations = (
            grit_iterations
            if grit_iterations is not None
            else self.config.grit_iterations
        )
        grit_alpha = (
            grit_alpha if grit_alpha is not None else self.config.grit_alpha
        )
        grit_entropy_threshold = (
            grit_entropy_threshold
            if grit_entropy_threshold is not None
            else self.config.grit_entropy_threshold
        )
        grit_refinement_percentile = (
            grit_refinement_percentile
            if grit_refinement_percentile is not None
            else self.config.grit_refinement_percentile
        )

        if use_asymmetric_ppr is not None:
            self.config.use_asymmetric_ppr = use_asymmetric_ppr
        if forward_weight is not None:
            self.config.forward_weight = forward_weight
        if ppr_alpha is not None:
            self.config.ppr_alpha = ppr_alpha

        filter_low_quality = (
            filter_low_quality
            if filter_low_quality is not None
            else self.config.filter_low_quality
        )
        variance_threshold = (
            variance_threshold
            if variance_threshold is not None
            else self.config.variance_threshold
        )

        return _PredictionSettings(
            top_k=top_k,
            use_full_ontology=use_full_ontology,
            use_grit=use_grit,
            grit_iterations=grit_iterations,
            grit_alpha=grit_alpha,
            grit_entropy_threshold=grit_entropy_threshold,
            grit_refinement_percentile=grit_refinement_percentile,
            filter_low_quality=filter_low_quality,
            variance_threshold=variance_threshold,
            use_asymmetric_ppr=self.config.use_asymmetric_ppr,
            forward_weight=self.config.forward_weight,
            ppr_alpha=self.config.ppr_alpha,
        )

    def _build_prediction_fingerprint(
        self,
        settings: _PredictionSettings,
    ) -> Dict[str, Any]:
        """Build a public fingerprint for raw top-1 prediction settings."""

        fingerprint = settings.cache_signature()
        fingerprint["model_identity"] = self._get_model_identity()
        return fingerprint

    def _extract_prediction_metadata(
        self,
        results: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Read prediction metadata attached to a DataFrame result."""

        metadata = results.attrs.get(_PREDICTION_METADATA_ATTR_KEY)
        if isinstance(metadata, dict):
            return dict(metadata)
        return {}

    def _infer_label_format(self, values: pd.Series) -> Optional[str]:
        """Infer a likely label format from prediction strings."""

        sample = []
        for raw_value in values.astype(object).tolist():
            if pd.isna(raw_value):
                continue
            text = str(raw_value)
            if text:
                sample.append(text)
            if len(sample) >= 20:
                break

        if not sample:
            return None
        if all(re.match(r"^[A-Z]+:\d+$", value) for value in sample):
            return "id"
        if all(re.search(r"\([A-Z]+:\d+\)$", value) for value in sample):
            return "both"
        return "name"

    def _build_annotation_manifest(
        self,
        prediction_col: str,
        score_col: Optional[str],
        cell_variance_col: Optional[str],
        label_format: Optional[str],
        prediction_fingerprint: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the public annotation manifest stored in ``adata.uns``."""

        return {
            "version": _ANNOTATION_MANIFEST_VERSION,
            "prediction_col": prediction_col,
            "score_col": score_col,
            "cell_variance_col": cell_variance_col,
            "label_format": label_format,
            "prediction_fingerprint": (
                dict(prediction_fingerprint)
                if isinstance(prediction_fingerprint, dict)
                else None
            ),
        }

    def _get_annotation_manifest(
        self,
        adata: anndata.AnnData,
    ) -> Optional[Dict[str, Any]]:
        """Return a copy of the public annotation manifest if present."""

        manifest = adata.uns.get(_ANNOTATION_MANIFEST_KEY)
        if not isinstance(manifest, dict):
            manifest = adata.uns.get("hector_annotation_manifest")
        if isinstance(manifest, dict):
            return dict(manifest)
        return None

    def _write_annotation_manifest(
        self,
        adata: anndata.AnnData,
        manifest: Dict[str, Any],
    ) -> None:
        """Persist a public annotation manifest into ``adata.uns``."""

        self._ensure_cache_writable(adata)
        sanitised = _sanitize_for_h5ad_uns(dict(manifest))
        adata.uns[_ANNOTATION_MANIFEST_KEY] = sanitised
        adata.uns["hector_annotation_manifest"] = dict(sanitised)

    def _backfill_annotation_manifest_if_possible(
        self,
        adata: anndata.AnnData,
        annotation_ref: Optional[Dict[str, Any]],
        prediction_fingerprint: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a manifest when fallback annotation columns are usable."""

        if annotation_ref is None:
            return
        if annotation_ref.get("_source") != "fallback":
            return

        prediction_col = annotation_ref.get("prediction_col")
        score_col = annotation_ref.get("score_col")
        if (
            not isinstance(prediction_col, str)
            or prediction_col not in adata.obs.columns
            or not isinstance(score_col, str)
            or score_col not in adata.obs.columns
        ):
            return

        manifest = self._build_annotation_manifest(
            prediction_col=prediction_col,
            score_col=score_col,
            cell_variance_col=annotation_ref.get("cell_variance_col"),
            label_format=annotation_ref.get("label_format"),
            prediction_fingerprint=prediction_fingerprint,
        )
        self._write_annotation_manifest(adata, manifest)

    def _attach_prediction_metadata(
        self,
        results: pd.DataFrame,
        prediction_fingerprint: Dict[str, Any],
        label_format: str,
    ) -> None:
        """Attach reusable prediction metadata to a prediction DataFrame."""

        results.attrs[_PREDICTION_METADATA_ATTR_KEY] = {
            "version": _ANNOTATION_MANIFEST_VERSION,
            "prediction_fingerprint": dict(prediction_fingerprint),
            "label_format": label_format,
        }

    def _resolve_annotation_reference(
        self,
        adata: anndata.AnnData,
        require_score: bool = True,
        allow_fallback: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Resolve public annotation columns from a manifest or fallback names."""

        manifest = self._get_annotation_manifest(adata)
        if manifest is not None:
            prediction_col = manifest.get("prediction_col")
            score_col = manifest.get("score_col")
            cell_variance_col = manifest.get("cell_variance_col")

            if (
                isinstance(prediction_col, str)
                and prediction_col in adata.obs.columns
                and (
                    not require_score
                    or (
                        isinstance(score_col, str)
                        and score_col in adata.obs.columns
                    )
                )
            ):
                if not isinstance(cell_variance_col, str) or cell_variance_col not in adata.obs.columns:
                    cell_variance_col = None
                return {
                    **manifest,
                    "prediction_col": prediction_col,
                    "score_col": score_col if isinstance(score_col, str) else None,
                    "cell_variance_col": cell_variance_col,
                    "_source": "manifest",
                }

        if not allow_fallback or "hector_prediction" not in adata.obs.columns:
            return None

        score_col = "hector_prediction_confidence" if "hector_prediction_confidence" in adata.obs.columns else None
        if require_score and score_col is None:
            return None

        cell_variance_col = (
            "hector_cell_variance"
            if "hector_cell_variance" in adata.obs.columns
            else None
        )
        return {
            **self._build_annotation_manifest(
                prediction_col="hector_prediction",
                score_col=score_col,
                cell_variance_col=cell_variance_col,
                label_format=self._infer_label_format(adata.obs["hector_prediction"]),
                prediction_fingerprint=None,
            ),
            "_source": "fallback",
        }

    def _parse_prediction_values_to_raw_ids(
        self,
        values: pd.Series,
    ) -> Optional[np.ndarray]:
        """Convert annotated prediction values back into canonical ontology IDs."""

        raw_ids: List[str] = []
        for raw_value in values.astype(object).tolist():
            if pd.isna(raw_value):
                return None
            ontology_id = self._parse_ontology_id(str(raw_value))
            if ontology_id is None:
                return None
            raw_ids.append(ontology_id)
        return np.asarray(raw_ids, dtype=object)

    def _annotation_reference_matches(
        self,
        adata: anndata.AnnData,
        annotation_ref: Optional[Dict[str, Any]],
        expected_fingerprint: Optional[Dict[str, Any]] = None,
        require_score: bool = True,
    ) -> bool:
        """Validate a public annotation reference against current expectations."""

        if annotation_ref is None:
            return False

        prediction_col = annotation_ref.get("prediction_col")
        if (
            not isinstance(prediction_col, str)
            or prediction_col not in adata.obs.columns
            or len(adata.obs[prediction_col]) != adata.shape[0]
        ):
            return False

        score_col = annotation_ref.get("score_col")
        if require_score:
            if (
                not isinstance(score_col, str)
                or score_col not in adata.obs.columns
                or len(adata.obs[score_col]) != adata.shape[0]
            ):
                return False

        raw_ids = self._parse_prediction_values_to_raw_ids(adata.obs[prediction_col])
        if raw_ids is None:
            return False

        if expected_fingerprint is not None:
            stored_fingerprint = annotation_ref.get("prediction_fingerprint")
            if not _fingerprints_equal(stored_fingerprint, expected_fingerprint):
                return False

        return True

    def _annotation_reference_to_core_result(
        self,
        adata: anndata.AnnData,
        annotation_ref: Dict[str, Any],
        settings: _PredictionSettings,
        cell_variance_array: np.ndarray,
    ) -> Optional[_PredictionCoreResult]:
        """Rebuild a top-1 prediction result from public annotation columns."""

        prediction_col = annotation_ref.get("prediction_col")
        score_col = annotation_ref.get("score_col")
        if (
            not isinstance(prediction_col, str)
            or not isinstance(score_col, str)
            or prediction_col not in adata.obs.columns
            or score_col not in adata.obs.columns
        ):
            return None

        raw_ids = self._parse_prediction_values_to_raw_ids(adata.obs[prediction_col])
        if raw_ids is None:
            return None

        active_class_ids = self._get_active_class_ids(settings.use_full_ontology)
        name_to_idx = {
            ontology_id: idx for idx, ontology_id in enumerate(active_class_ids)
        }
        top_indices = np.array(
            [[name_to_idx.get(prediction_id, -1)] for prediction_id in raw_ids],
            dtype=np.int64,
        )
        if np.any(top_indices < 0):
            return None

        prediction_scores = adata.obs[score_col].astype(float).values
        top_scores = prediction_scores.reshape(-1, 1)
        low_quality_mask = self._compute_low_quality_mask(
            cell_variance_array,
            settings,
        )

        return _PredictionCoreResult(
            top_indices=top_indices,
            top_ids=raw_ids.reshape(-1, 1),
            top_scores=top_scores,
            prediction_ids=raw_ids,
            prediction_scores=prediction_scores,
            cell_variance_array=cell_variance_array,
            low_quality_mask=low_quality_mask,
            score_matrix_array=None,
        )

    def _build_prediction_core_manifest(
        self,
        adata: anndata.AnnData,
        *,
        prediction_fingerprint: Dict[str, Any],
        core_result: _PredictionCoreResult,
    ) -> Dict[str, Any]:
        """Build the reusable private top-1 prediction core manifest."""

        low_quality_mask = core_result.low_quality_mask
        return {
            "version": _PREDICTION_CORE_MANIFEST_VERSION,
            "prediction_fingerprint": dict(prediction_fingerprint),
            "cell_order_fingerprint": self._fingerprint_cell_order(adata),
            "n_cells": int(adata.shape[0]),
            "top_indices": np.asarray(
                core_result.top_indices, dtype=np.int64
            ).copy(),
            "prediction_scores": np.asarray(
                core_result.prediction_scores, dtype=np.float64
            ).copy(),
            "cell_variance_array": np.asarray(
                core_result.cell_variance_array, dtype=np.float64
            ).copy(),
            "low_quality_mask": (
                None
                if low_quality_mask is None
                else np.asarray(low_quality_mask, dtype=bool).copy()
            ),
        }

    def _get_prediction_core_manifest(
        self,
        adata: anndata.AnnData,
    ) -> Optional[Dict[str, Any]]:
        """Return a copy of the cached private prediction core manifest."""

        manifest = adata.uns.get(_PREDICTION_CORE_MANIFEST_KEY)
        if isinstance(manifest, dict):
            return dict(manifest)
        return None

    def _write_prediction_core_manifest(
        self,
        adata: anndata.AnnData,
        manifest: Dict[str, Any],
    ) -> None:
        """Persist the private top-1 prediction core manifest."""

        self._ensure_cache_writable(adata)
        adata.uns[_PREDICTION_CORE_MANIFEST_KEY] = _sanitize_for_h5ad_uns(
            dict(manifest)
        )

    def _prediction_core_manifest_matches(
        self,
        adata: anndata.AnnData,
        manifest: Optional[Dict[str, Any]],
        *,
        expected_prediction_fingerprint: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Validate a private prediction core manifest for the current AnnData."""

        if not isinstance(manifest, dict):
            return False
        if manifest.get("version") != _PREDICTION_CORE_MANIFEST_VERSION:
            return False
        if expected_prediction_fingerprint is not None and not _fingerprints_equal(
            manifest.get("prediction_fingerprint"),
            expected_prediction_fingerprint,
        ):
            return False
        if manifest.get("n_cells") != int(adata.shape[0]):
            return False
        if (
            manifest.get("cell_order_fingerprint")
            != self._fingerprint_cell_order(adata)
        ):
            return False

        top_indices = np.asarray(manifest.get("top_indices"))
        prediction_scores = np.asarray(manifest.get("prediction_scores"))
        cell_variance_array = np.asarray(manifest.get("cell_variance_array"))
        if top_indices.shape != (adata.shape[0], 1):
            return False
        if prediction_scores.shape != (adata.shape[0],):
            return False
        if cell_variance_array.shape != (adata.shape[0],):
            return False

        low_quality_mask = manifest.get("low_quality_mask")
        if low_quality_mask is not None and np.asarray(low_quality_mask).shape != (
            adata.shape[0],
        ):
            return False
        return True

    def _prediction_core_manifest_to_core_result(
        self,
        manifest: Dict[str, Any],
        settings: _PredictionSettings,
    ) -> Optional[_PredictionCoreResult]:
        """Rebuild a top-1 prediction result from the private core manifest."""

        top_indices = np.asarray(manifest.get("top_indices"), dtype=np.int64)
        prediction_scores = np.asarray(
            manifest.get("prediction_scores"), dtype=np.float64
        )
        cell_variance_array = np.asarray(
            manifest.get("cell_variance_array"), dtype=np.float64
        )
        if top_indices.ndim != 2 or top_indices.shape[1] != 1:
            return None
        if prediction_scores.ndim != 1 or len(prediction_scores) != top_indices.shape[0]:
            return None

        active_class_ids = np.asarray(
            self._get_active_class_ids(settings.use_full_ontology),
            dtype=object,
        )
        if top_indices.size == 0:
            top_ids = np.empty((0, 1), dtype=object)
        else:
            flat_indices = top_indices[:, 0]
            if np.any(flat_indices < 0) or np.any(flat_indices >= len(active_class_ids)):
                return None
            top_ids = active_class_ids[flat_indices].reshape(-1, 1)

        low_quality_mask = manifest.get("low_quality_mask")
        if low_quality_mask is None:
            low_quality_mask = self._compute_low_quality_mask(
                cell_variance_array,
                settings,
            )
        else:
            low_quality_mask = np.asarray(low_quality_mask, dtype=bool)

        top_scores = prediction_scores.reshape(-1, 1)
        return _PredictionCoreResult(
            top_indices=top_indices,
            top_ids=top_ids,
            top_scores=top_scores,
            prediction_ids=top_ids[:, 0] if top_ids.size > 0 else np.array([], dtype=object),
            prediction_scores=prediction_scores,
            cell_variance_array=cell_variance_array,
            low_quality_mask=low_quality_mask,
            score_matrix_array=None,
        )

    def _cache_prediction_core_result(
        self,
        adata: anndata.AnnData,
        *,
        prediction_fingerprint: Dict[str, Any],
        core_result: _PredictionCoreResult,
    ) -> None:
        """Persist the reusable private top-1 prediction core state."""

        manifest = self._build_prediction_core_manifest(
            adata,
            prediction_fingerprint=prediction_fingerprint,
            core_result=core_result,
        )
        self._write_prediction_core_manifest(adata, manifest)

    # -------------------------------------------------------------------- #
    # Atypical manifest (evaluate_cells) — cache namespace.                 #
    # -------------------------------------------------------------------- #

    def _build_atypical_fingerprint(
        self,
        *,
        snn_n_neighbors: int,
        snn_min_shared_fraction: float,
        backend: str,
        min_community_size: Optional[int],
        fdr_alpha: float,
        effect_floor: float,
        temperature: float,
        top_k_neighbors: int,
        min_cells_number: int,
        class_relative_adherence: bool,
        class_relative_blend_weight: float,
        bgmm_n_components: int,
        bgmm_min_weight: float,
        bgmm_max_std: float,
        bgmm_train_subsample: int,
        bgmm_threshold: float,
        random_seed: int,
        pynndescent_n_jobs: Optional[int],
        ontology_prune_lineage_levels: Optional[int],
        ontology_prune_soft_power: float,
        g1_resolution_range: Tuple[float, float],
        g1_resolution_step: float,
        g5_d_cuts: Tuple[float, ...],
        g5_min_lca_level: int,
    ) -> Dict[str, Any]:
        """Fingerprint for the canonical routed atypical-cell manifest.

        Locks the upstream 4-D feature stack and the LCA-routing knobs so
        cached results are reused only when the routed evaluation contract is
        unchanged.
        """
        return {
            "version": _ATYPICAL_FINGERPRINT_VERSION,
            "method": "lca_routed_g8pp",
            "gate_kind": "lca_routed",
            "snn_n_neighbors": int(snn_n_neighbors),
            "snn_min_shared_fraction": float(snn_min_shared_fraction),
            "backend": str(backend),
            "min_community_size": (
                None if min_community_size is None else int(min_community_size)
            ),
            "fdr_alpha": float(fdr_alpha),
            "effect_floor": float(effect_floor),
            "temperature": float(temperature),
            "top_k_neighbors": int(top_k_neighbors),
            "min_cells_number": int(min_cells_number),
            "class_relative_adherence": bool(class_relative_adherence),
            "class_relative_blend_weight": float(class_relative_blend_weight),
            "bgmm_n_components": int(bgmm_n_components),
            "bgmm_min_weight": float(bgmm_min_weight),
            "bgmm_max_std": float(bgmm_max_std),
            "bgmm_train_subsample": int(bgmm_train_subsample),
            "bgmm_threshold": float(bgmm_threshold),
            "feature_names": list(_ATYPICAL_FEATURE_NAMES),
            "abnormality_formula": "v7_unweighted_mean_4_bad_scores",
            "random_seed": int(random_seed),
            "pynndescent_n_jobs": (
                None if pynndescent_n_jobs is None else int(pynndescent_n_jobs)
            ),
            "ontology_prune_lineage_levels": (
                None if ontology_prune_lineage_levels is None
                else int(ontology_prune_lineage_levels)
            ),
            "ontology_prune_soft_power": float(ontology_prune_soft_power),
            "g1_resolution_range": [
                float(g1_resolution_range[0]),
                float(g1_resolution_range[1]),
            ],
            "g1_resolution_step": float(g1_resolution_step),
            "g5_d_cuts": [float(d) for d in g5_d_cuts],
            "g5_min_lca_level": int(g5_min_lca_level),
        }

    def _build_atypical_manifest(
        self,
        *,
        prediction_fingerprint: Dict[str, Any],
        atypical_fingerprint: Dict[str, Any],
        community_stats: List[Dict[str, Any]],
        bgmm_info: Dict[str, Any],
        snn_info: Dict[str, Any],
        halted: bool,
        bgmm_halted: bool,
        n_suspicious_communities: int,
        fdr_alpha: float,
        effect_floor: float,
        min_community_size: int,
        bgmm_threshold: float,
        route: str,
        lca: List[str],
        lca_is_root_only: bool,
        g1_resolution: float,
        g1_ari_scores: Dict[float, float],
        g5_per_depth_summary: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the canonical routed atypical-detection manifest.

        Per-cell arrays live in ``adata.obs``. The manifest carries cache
        fingerprints, community stats from G1's representative partition, and
        the routed diagnostics needed to interpret G1/G5 dispatch.
        """
        return {
            "version": _ATYPICAL_MANIFEST_VERSION,
            "gate_kind": "lca_routed",
            "columns": {name: name for name in _ATYPICAL_OBS_COLUMNS},
            "prediction_fingerprint": dict(prediction_fingerprint),
            "atypical_fingerprint": dict(atypical_fingerprint),
            "community_stats": [dict(s) for s in community_stats],
            "bgmm_info": dict(bgmm_info),
            "snn_info": dict(snn_info),
            "halted": bool(halted),
            "bgmm_halted": bool(bgmm_halted),
            "n_suspicious_communities": int(n_suspicious_communities),
            "fdr_alpha": float(fdr_alpha),
            "effect_floor": float(effect_floor),
            "min_community_size": int(min_community_size),
            "bgmm_threshold": float(bgmm_threshold),
            "route": str(route),
            "lca": [str(n) for n in lca],
            "lca_is_root_only": bool(lca_is_root_only),
            "g1_resolution": float(g1_resolution),
            "g1_ari_scores": {float(k): float(v) for k, v in g1_ari_scores.items()},
            "g5_per_depth_summary": [dict(d) for d in g5_per_depth_summary],
        }

    def _get_atypical_manifest(
        self, adata: anndata.AnnData
    ) -> Optional[Dict[str, Any]]:
        manifest = adata.uns.get(_ATYPICAL_MANIFEST_KEY)
        if isinstance(manifest, dict):
            # Reverse the h5ad sanitisation applied at write time so
            # downstream consumers (``pd.DataFrame(community_stats)``,
            # ``list(g5_per_depth_summary)``, ``for r in component_report``)
            # see the original list-of-dicts shape whether the manifest
            # was just written this session or loaded from disk.
            return _restore_atypical_manifest_shapes(dict(manifest))
        return None

    def _write_atypical_manifest(
        self,
        adata: anndata.AnnData,
        manifest: Dict[str, Any],
    ) -> None:
        self._ensure_cache_writable(adata)
        # Sanitize so adata.write_h5ad() succeeds: stringify dict keys
        # (g1_ari_scores has float keys, lineage_distance_histogram has
        # int keys), collapse list-of-dicts to dict-of-lists
        # (community_stats, component_report, g5_per_depth_summary).
        adata.uns[_ATYPICAL_MANIFEST_KEY] = _sanitize_for_h5ad_uns(dict(manifest))
        adata.uns.pop("hector_lca_routed_manifest", None)

    def _atypical_manifest_matches(
        self,
        adata: anndata.AnnData,
        manifest: Optional[Dict[str, Any]],
        *,
        expected_prediction_fingerprint: Optional[Dict[str, Any]] = None,
        expected_atypical_fingerprint: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not isinstance(manifest, dict):
            return False
        if manifest.get("version") != _ATYPICAL_MANIFEST_VERSION:
            return False
        if expected_prediction_fingerprint is not None and not _fingerprints_equal(
            manifest.get("prediction_fingerprint"),
            expected_prediction_fingerprint,
        ):
            return False
        if expected_atypical_fingerprint is not None and not _fingerprints_equal(
            manifest.get("atypical_fingerprint"),
            expected_atypical_fingerprint,
        ):
            return False

        n_cells = adata.shape[0]
        for col_name in _ATYPICAL_OBS_COLUMNS:
            if col_name not in adata.obs.columns:
                return False
            if len(adata.obs[col_name]) != n_cells:
                return False
        return True

    def _resolve_atypical_manifest(
        self,
        adata: anndata.AnnData,
        *,
        expected_prediction_fingerprint: Optional[Dict[str, Any]] = None,
        expected_atypical_fingerprint: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        manifest = self._get_atypical_manifest(adata)
        if not self._atypical_manifest_matches(
            adata,
            manifest,
            expected_prediction_fingerprint=expected_prediction_fingerprint,
            expected_atypical_fingerprint=expected_atypical_fingerprint,
        ):
            return None
        return manifest

    def _build_result_from_adata(
        self,
        adata: anndata.AnnData,
        manifest: Dict[str, Any],
    ) -> EvaluateCellsResult:
        """Assemble an :class:`EvaluateCellsResult` from obs + manifest."""
        cells = pd.DataFrame(
            {
                name: adata.obs[name].to_numpy()
                for name in _ATYPICAL_OBS_COLUMNS
            },
            index=adata.obs_names,
        )
        communities = pd.DataFrame(manifest.get("community_stats", []))
        snn_info = dict(manifest.get("snn_info", {}))
        summary = {
            "halted": bool(manifest.get("halted", False)),
            "bgmm_halted": bool(manifest.get("bgmm_halted", False)),
            "n_suspicious_communities": int(
                manifest.get("n_suspicious_communities", 0)
            ),
            "fdr_alpha": float(manifest.get("fdr_alpha", 0.0)),
            "effect_floor": float(manifest.get("effect_floor", 0.0)),
            "min_community_size": int(manifest.get("min_community_size", 0)),
            "bgmm_threshold": float(manifest.get("bgmm_threshold", 0.0)),
            "backend_used": snn_info.get("backend_used"),
            "gate_kind": str(manifest.get("gate_kind", "lca_routed")),
            "route": str(manifest.get("route", "")),
            "lca": list(manifest.get("lca", [])),
            "lca_is_root_only": bool(manifest.get("lca_is_root_only", False)),
            "g1_resolution": float(manifest.get("g1_resolution", float("nan"))),
            "g5_per_depth_summary": list(manifest.get("g5_per_depth_summary", [])),
        }
        return EvaluateCellsResult(
            cells=cells,
            communities=communities,
            summary=summary,
            bgmm_info=dict(manifest.get("bgmm_info", {})),
            snn_info=snn_info,
        )

    def _format_prediction_results_from_core_result(
        self,
        adata: anndata.AnnData,
        *,
        core_result: _PredictionCoreResult,
        settings: _PredictionSettings,
        prediction_fingerprint: Dict[str, Any],
        label_format: str,
        cell_id_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Format a shared top-1 core result into the public prediction table."""

        if cell_id_column is not None:
            cell_ids = adata.obs[cell_id_column].tolist()
        else:
            cell_ids = adata.obs_names.tolist()

        results = {'cell_id': cell_ids}
        for k in range(settings.top_k):
            formatted_predictions = [
                self._format_prediction(top_ids[k], label_format)
                for top_ids in core_result.top_ids
            ]
            results[f'top_{k+1}_prediction'] = formatted_predictions
            results[f'top_{k+1}_score'] = core_result.top_scores[:, k]

        results['hector_cell_variance'] = core_result.cell_variance_array
        if core_result.low_quality_mask is not None:
            results['is_low_quality'] = core_result.low_quality_mask

        results_df = pd.DataFrame(results)
        results_df.index = pd.Index(cell_ids, name='cell_id')
        results_df.drop(columns='cell_id', inplace=True)
        self._attach_prediction_metadata(
            results_df,
            prediction_fingerprint=prediction_fingerprint,
            label_format=label_format,
        )
        return results_df

    def _publish_public_annotation_from_session(
        self,
        adata: anndata.AnnData,
        prediction_session: _PredictionSession,
        *,
        prefix: str = 'hector_',
        label_format: str = 'id',
    ) -> Dict[str, Any]:
        """Publish reusable public annotation columns from a resolved session."""

        annotation_ref = self._resolve_annotation_reference(
            adata,
            require_score=True,
            allow_fallback=True,
        )
        # Fast path: cached annotation matches both fingerprint AND requested
        # label_format. label_format is not in the fingerprint (it's purely
        # cosmetic), so we check it explicitly here.
        if self._annotation_reference_matches(
            adata,
            annotation_ref,
            expected_fingerprint=prediction_session.prediction_fingerprint,
            require_score=True,
        ) and annotation_ref.get("label_format") == label_format:
            self._backfill_annotation_manifest_if_possible(
                adata,
                annotation_ref,
                prediction_fingerprint=prediction_session.prediction_fingerprint,
            )
            return self._resolve_annotation_reference(
                adata,
                require_score=True,
                allow_fallback=False,
            ) or annotation_ref

        # A cache miss or different label format re-emits columns through the
        # formatter using the session's resolved core result, without repeating
        # inference solely to change label presentation.
        results = self._format_prediction_results_from_core_result(
            adata,
            core_result=prediction_session.core_result,
            settings=prediction_session.settings,
            prediction_fingerprint=prediction_session.prediction_fingerprint,
            label_format=label_format,
        )
        self.write_predictions(
            adata,
            results,
            prefix=prefix,
            overwrite=True,
            rare_rollup=False,
        )
        annotation_ref = self._resolve_annotation_reference(
            adata,
            require_score=True,
            allow_fallback=False,
        )
        if not self._annotation_reference_matches(
            adata,
            annotation_ref,
            expected_fingerprint=prediction_session.prediction_fingerprint,
            require_score=True,
        ):
            raise RuntimeError(
                "Failed to publish reusable public annotation from the "
                "resolved prediction session."
            )
        return annotation_ref

    def _resolve_prediction_session(
        self,
        adata: anndata.AnnData,
        use_full_ontology: Optional[bool] = None,
        use_grit: Optional[bool] = None,
        grit_iterations: Optional[int] = None,
        grit_alpha: Optional[float] = None,
        grit_entropy_threshold: Optional[float] = None,
        grit_refinement_percentile: Optional[float] = None,
        use_asymmetric_ppr: Optional[bool] = None,
        forward_weight: Optional[float] = None,
        ppr_alpha: Optional[float] = None,
        filter_low_quality: Optional[bool] = None,
        variance_threshold: Optional[float] = None,
        force_recompute: bool = False,
        require_public_annotation: bool = False,
        require_pre_grit_scores: bool = False,
        allow_cached_prediction_reuse: bool = True,
    ) -> _PredictionSession:
        """Resolve embeddings plus reusable top-1 prediction state for one workflow."""

        settings = self._resolve_prediction_settings(
            top_k=1,
            use_full_ontology=use_full_ontology,
            use_grit=use_grit,
            grit_iterations=grit_iterations,
            grit_alpha=grit_alpha,
            grit_entropy_threshold=grit_entropy_threshold,
            grit_refinement_percentile=grit_refinement_percentile,
            use_asymmetric_ppr=use_asymmetric_ppr,
            forward_weight=forward_weight,
            ppr_alpha=ppr_alpha,
            filter_low_quality=filter_low_quality,
            variance_threshold=variance_threshold,
        )
        cell_vectors, cell_variance_array, prepared_expression = self._ensure_embeddings(
            adata,
            force_recompute=force_recompute,
        )
        prediction_fingerprint = self._build_prediction_fingerprint(settings)
        has_public_annotation = False
        annotation_ref = None

        if not force_recompute:
            annotation_ref = self._resolve_annotation_reference(
                adata,
                require_score=True,
                allow_fallback=True,
            )
            has_public_annotation = self._annotation_reference_matches(
                adata,
                annotation_ref,
                expected_fingerprint=prediction_fingerprint,
                require_score=True,
            )
            if has_public_annotation:
                self._backfill_annotation_manifest_if_possible(
                    adata,
                    annotation_ref,
                    prediction_fingerprint=prediction_fingerprint,
                )

        if (
            not force_recompute
            and allow_cached_prediction_reuse
            and not require_pre_grit_scores
        ):
            core_manifest = self._get_prediction_core_manifest(adata)
            if self._prediction_core_manifest_matches(
                adata,
                core_manifest,
                expected_prediction_fingerprint=prediction_fingerprint,
            ):
                core_result = self._prediction_core_manifest_to_core_result(
                    core_manifest,
                    settings,
                )
                if core_result is not None:
                    self._log(
                        "[INFO] Reusing cached predictions from "
                        "adata.uns['hector_prediction_core_manifest'] "
                        "(same cells, same settings). Pass "
                        "force_recompute=True to recompute."
                    )
                    return _PredictionSession(
                        cell_vectors=cell_vectors,
                        settings=settings,
                        core_result=core_result,
                        prediction_fingerprint=prediction_fingerprint,
                        source="core_manifest",
                        captured_pre_grit_scores=False,
                        has_public_annotation=has_public_annotation,
                    )

            if has_public_annotation:
                core_result = self._annotation_reference_to_core_result(
                    adata,
                    annotation_ref,
                    settings,
                    cell_variance_array,
                )
                if core_result is not None:
                    self._cache_prediction_core_result(
                        adata,
                        prediction_fingerprint=prediction_fingerprint,
                        core_result=core_result,
                    )
                    self._log(
                        "[INFO] Reusing public prediction annotation from "
                        "adata.obs/adata.uns..."
                    )
                    return _PredictionSession(
                        cell_vectors=cell_vectors,
                        settings=settings,
                        core_result=core_result,
                        prediction_fingerprint=prediction_fingerprint,
                        source="annotation",
                        captured_pre_grit_scores=False,
                        has_public_annotation=True,
                    )

        core_result = self._run_prediction_from_embeddings(
            adata,
            cell_vectors=cell_vectors,
            cell_variance_array=cell_variance_array,
            settings=settings,
            prepared_expression=prepared_expression,
            export_score_matrix=False,
            capture_pre_grit_score_matrix=require_pre_grit_scores,
            force_recompute=force_recompute,
        )
        self._cache_prediction_core_result(
            adata,
            prediction_fingerprint=prediction_fingerprint,
            core_result=core_result,
        )
        return _PredictionSession(
            cell_vectors=cell_vectors,
            settings=settings,
            core_result=core_result,
            prediction_fingerprint=prediction_fingerprint,
            source="fresh",
            captured_pre_grit_scores=require_pre_grit_scores,
            has_public_annotation=has_public_annotation,
        )

    def _ensure_embeddings(
        self,
        adata: anndata.AnnData,
        force_recompute: bool = False,
    ):
        """Guarantee cached embeddings and scalar variance for *adata*.

        Returns the prepared expression matrix when embeddings were freshly
        computed in this call so downstream GRIT refinement can reuse it
        instead of re-running gene matching and normalization.
        """

        prepared_expression = self.compute_cell_embeddings(
            adata,
            force_recompute=force_recompute,
            _return_prepared_expression=True,
        )
        return (
            adata.obsm["X_hector"],
            adata.obs["hector_cell_variance"].values,
            prepared_expression,
        )

    def _ensure_top1_predictions(
        self,
        adata: anndata.AnnData,
        use_full_ontology: Optional[bool] = None,
        use_grit: Optional[bool] = None,
        grit_iterations: Optional[int] = None,
        grit_alpha: Optional[float] = None,
        grit_entropy_threshold: Optional[float] = None,
        grit_refinement_percentile: Optional[float] = None,
        use_asymmetric_ppr: Optional[bool] = None,
        forward_weight: Optional[float] = None,
        ppr_alpha: Optional[float] = None,
        filter_low_quality: Optional[bool] = None,
        variance_threshold: Optional[float] = None,
        force_recompute: bool = False,
        allow_annotation_reuse: bool = True,
        capture_pre_grit_score_matrix: bool = False,
    ) -> Tuple[np.ndarray, _PredictionSettings, _PredictionCoreResult]:
        """Guarantee cached embeddings and canonical raw top-1 predictions."""
        session = self._resolve_prediction_session(
            adata,
            use_full_ontology=use_full_ontology,
            use_grit=use_grit,
            grit_iterations=grit_iterations,
            grit_alpha=grit_alpha,
            grit_entropy_threshold=grit_entropy_threshold,
            grit_refinement_percentile=grit_refinement_percentile,
            use_asymmetric_ppr=use_asymmetric_ppr,
            forward_weight=forward_weight,
            ppr_alpha=ppr_alpha,
            filter_low_quality=filter_low_quality,
            variance_threshold=variance_threshold,
            force_recompute=force_recompute,
            require_pre_grit_scores=capture_pre_grit_score_matrix,
            allow_cached_prediction_reuse=allow_annotation_reuse,
        )
        return session.cell_vectors, session.settings, session.core_result

    def _ensure_public_annotation(
        self,
        adata: anndata.AnnData,
        use_full_ontology: Optional[bool] = None,
        use_grit: Optional[bool] = None,
        grit_iterations: Optional[int] = None,
        grit_alpha: Optional[float] = None,
        grit_entropy_threshold: Optional[float] = None,
        grit_refinement_percentile: Optional[float] = None,
        use_asymmetric_ppr: Optional[bool] = None,
        forward_weight: Optional[float] = None,
        ppr_alpha: Optional[float] = None,
        filter_low_quality: Optional[bool] = None,
        variance_threshold: Optional[float] = None,
        force_recompute: bool = False,
        prefix: str = 'hector_',
        label_format: str = 'id',
    ) -> Dict[str, Any]:
        """Ensure reusable public annotation exists for downstream workflows."""
        valid_formats = ('id', 'name', 'both')
        if label_format not in valid_formats:
            raise ValueError(
                f"Invalid label_format '{label_format}'. Must be one of: {list(valid_formats)}"
            )
        session = self._resolve_prediction_session(
            adata,
            use_full_ontology=use_full_ontology,
            use_grit=use_grit,
            grit_iterations=grit_iterations,
            grit_alpha=grit_alpha,
            grit_entropy_threshold=grit_entropy_threshold,
            grit_refinement_percentile=grit_refinement_percentile,
            use_asymmetric_ppr=use_asymmetric_ppr,
            forward_weight=forward_weight,
            ppr_alpha=ppr_alpha,
            filter_low_quality=filter_low_quality,
            variance_threshold=variance_threshold,
            force_recompute=force_recompute,
            require_public_annotation=True,
            require_pre_grit_scores=False,
            allow_cached_prediction_reuse=True,
        )
        try:
            return self._publish_public_annotation_from_session(
                adata,
                session,
                prefix=prefix,
                label_format=label_format,
            )
        finally:
            session.core_result.close_transient_arrays()

    def _load_model_mlx(self):
        """Load model from h5 checkpoint using only h5py/numpy + backend_mlx.

        This is the TF-free counterpart of :meth:`_load_model`. It populates
        every ``self.*`` attribute that the downstream prediction pipeline
        requires, using the MLX/numpy loaders in :mod:`backend_mlx` for the
        neural-network components and raw h5py reads for metadata.

        After this method returns the predictor is fully functional on the
        MLX backend — no TensorFlow objects exist.
        """
        from types import SimpleNamespace
        import json
        from .backend_mlx import (
            load_mlx_encoder_from_h5,
            load_celltype_embedder_mlx,
            load_hpl_head_mlx,
            load_procrustes_cfg,
            load_ontology_cfg,
        )

        # ------------------------------------------------------------------ #
        # 1. Checkpoint metadata (gene order, normalization, version)
        # ------------------------------------------------------------------ #
        checkpoint_data = self._load_checkpoint_metadata()
        self.checkpoint_version = checkpoint_data.get("checkpoint_version")
        self._extract_gene_order(checkpoint_data)
        self._extract_normalization_config(checkpoint_data)

        # ------------------------------------------------------------------ #
        # 2. MLX encoder
        # ------------------------------------------------------------------ #
        self._mlx_encoder, self._mlx_config = load_mlx_encoder_from_h5(
            self.model_path
        )
        self._log("  ☑️  MLX encoder loaded")

        # ------------------------------------------------------------------ #
        # 3. CellTypeEmbedder (numpy GAT prototypes)
        # ------------------------------------------------------------------ #
        self._mlx_celltype_embedder = load_celltype_embedder_mlx(
            self.model_path
        )
        self._log("  ☑️  MLX CellTypeEmbedder loaded")

        # ------------------------------------------------------------------ #
        # 4. HPLHead (numpy proxy scorer)
        # ------------------------------------------------------------------ #
        self._mlx_hpl_head = load_hpl_head_mlx(self.model_path)
        if self._mlx_hpl_head is not None:
            self._log("  ☑️  MLX HPLHead loaded")
        else:
            self._log("  ⚠️  No HPL head in checkpoint (GAT-only mode)")

        # ------------------------------------------------------------------ #
        # 5. Procrustes alignment
        # ------------------------------------------------------------------ #
        self._mlx_procrustes = load_procrustes_cfg(self.model_path)
        self.is_procrustes_aligned = self._mlx_procrustes.is_aligned
        self.procrustes_rotation_matrix = self._mlx_procrustes.rotation_matrix
        self.procrustes_use_centering = self._mlx_procrustes.use_centering
        self.procrustes_gat_mean = self._mlx_procrustes.gat_mean
        self.procrustes_hpl_mean = self._mlx_procrustes.hpl_mean
        self.alignment_quality = checkpoint_data.get("alignment_quality")
        self.alignment_cosine_sim = checkpoint_data.get("alignment_cosine_sim")

        # ------------------------------------------------------------------ #
        # 6. Ontology (class lists, adjacency, name map)
        # ------------------------------------------------------------------ #
        self._mlx_ontology = load_ontology_cfg(self.model_path)

        with h5py.File(self.model_path, "r") as f:
            ontology_grp = f["ontology"]

            def _load_string_list(grp, key):
                if key in grp:
                    return [
                        s.decode("utf-8") if isinstance(s, bytes) else str(s)
                        for s in grp[key][:]
                    ]
                return []

            seen_classes = _load_string_list(ontology_grp, "classes")
            full_classes = _load_string_list(ontology_grp, "full_classes")
            full_class_names = _load_string_list(ontology_grp, "full_class_names")

            ontology_adj = (
                ontology_grp["ontology_adj"][:].astype(np.float32)
                if "ontology_adj" in ontology_grp
                else None
            )
            full_ontology_adj = (
                ontology_grp["full_ontology_adj"][:].astype(np.float32)
                if "full_ontology_adj" in ontology_grp
                else None
            )
            pure_ontology_adj = (
                ontology_grp["pure_ontology_adj"][:]
                if "pure_ontology_adj" in ontology_grp
                else None
            )
            full_pure_ontology_adj = (
                ontology_grp["full_pure_ontology_adj"][:]
                if "full_pure_ontology_adj" in ontology_grp
                else None
            )
            level_array = (
                ontology_grp["level_array"][:]
                if "level_array" in ontology_grp
                else None
            )
            full_level_array = (
                ontology_grp["full_level_array"][:]
                if "full_level_array" in ontology_grp
                else None
            )

        self.pure_ontology_adj = pure_ontology_adj
        self.full_pure_ontology_adj = full_pure_ontology_adj
        self.full_class_names = full_class_names

        # Build id_to_name_map
        self.id_to_name_map = {}
        if full_class_names and full_classes:
            if len(full_class_names) == len(full_classes):
                self.id_to_name_map = dict(zip(full_classes, full_class_names))

        # Lightweight SimpleNamespace that mimics the TF CellTypeEmbedder
        # just enough for _format_prediction_results_from_core_result,
        # _get_active_class_ids, _indices_to_names, _compute_seen_indices,
        # _build_visualization_state, and the GRIT path.

        # get_ppr_matrix fallback: compute symmetric PPR from full_ontology_adj
        # (used only when use_asymmetric_ppr is False).
        def _get_ppr_matrix_mlx(for_full_ontology=False):
            from .backend_mlx import compute_ppr_matrix as _ppr
            adj = full_ontology_adj if for_full_ontology else ontology_adj
            if adj is None:
                n = len(full_classes) if for_full_ontology else len(seen_classes)
                return np.eye(n, dtype=np.float32)
            return _ppr(adj, alpha=self._mlx_config.get("ppr_alpha", 0.15))

        self.celltype_gat = SimpleNamespace(
            classes=seen_classes,
            full_classes=full_classes,
            ontology_adj_np=ontology_adj,
            full_ontology_adj_np=full_ontology_adj,
            zero_shot_mode=self.config.use_full_ontology,
            level_array_np=level_array,
            full_level_array_np=full_level_array,
            ppr_alpha=self._mlx_config.get("ppr_alpha", 0.15),
            # get_gat_prototypes is called by _build_visualization_state and
            # _score_embedding_batch.  On the MLX path we delegate to the
            # numpy CellTypeEmbedder.
            get_gat_prototypes=lambda for_full_ontology=False, training=False: (
                self._mlx_celltype_embedder.get_gat_prototypes(
                    for_full_ontology=for_full_ontology, normalize=True
                )
            ),
            get_ppr_matrix=_get_ppr_matrix_mlx,
        )

        # ------------------------------------------------------------------ #
        # 7. Memory Bank (anchor features) — stored as numpy, NOT tf.Tensor
        # ------------------------------------------------------------------ #
        with h5py.File(self.model_path, "r") as f:
            if "memory_bank" in f:
                mem_grp = f["memory_bank"]
                mem_size = int(mem_grp["mem_weight_1"][()])
                self.anchor_features = (
                    mem_grp["mem_weight_2"][:mem_size].astype(np.float32)
                )
                self._log(
                    f"  📂 Loaded Memory Bank: {mem_size:,} anchor cells"
                )
            else:
                self.anchor_features = None
                self._log(
                    "  ℹ️  No Memory Bank in checkpoint (batch-only graphs)"
                )

        # ------------------------------------------------------------------ #
        # 8. TF models are NOT loaded on the MLX path
        # ------------------------------------------------------------------ #
        self.vgae_encoder = None
        self.vgae_decoder = None
        self.aux_classifier = None

    def _load_model(self):
        """Load model checkpoint and extract metadata.

        Loads the Procrustes rotation matrix (and optional mean-centering
        vectors) when the checkpoint is Procrustes-aligned; otherwise the
        scorer falls back to raw GAT prototypes.
        """
        # MLX model loading path.
        if _select_encoder_backend() == 'mlx':
            try:
                self._load_model_mlx()
                from .backend_mlx import mlx_device_name
                chip = mlx_device_name() or "Apple Silicon"
                self._log(" 🎉 Model loaded successfully! 🥳")
                self._log(f"   Prediction hardware: GPU ({chip}, MLX)")
                self._log(
                    f"   Supported cell types: "
                    f"{len(self._mlx_ontology.full_classes)}"
                )
                return
            except Exception as exc:
                import traceback as _tb
                self._log(
                    f"  ⚠️ MLX model loading failed ({exc}); falling back to TF"
                )
                _tb.print_exc()

        # TensorFlow model loading path.
        # Load checkpoint metadata first
        checkpoint_data = self._load_checkpoint_metadata()
        self.checkpoint_version = checkpoint_data.get("checkpoint_version")

        # Extract gene order
        self._extract_gene_order(checkpoint_data)

        # Extract normalization config
        self._extract_normalization_config(checkpoint_data)

        # Load model components from HDF5 checkpoint

        self.vgae_encoder, self.vgae_decoder, self.celltype_gat = self._load_models_from_hdf5()

        # Set zero-shot mode if requested
        if self.config.use_full_ontology:
            self.celltype_gat.zero_shot_mode = True


        # Load HPL head (aux_classifier) from checkpoint
        self.aux_classifier = self._load_hpl_head()

        # Load Memory Bank (anchor cells) from checkpoint

        self.anchor_features = self._load_memory_bank()

        # Procrustes alignment (the only prototype-alignment mode)
        self.is_procrustes_aligned = checkpoint_data.get('procrustes_aligned', False)
        self.procrustes_rotation_matrix = None
        self.alignment_quality = checkpoint_data.get('alignment_quality', None)
        self.alignment_cosine_sim = checkpoint_data.get('alignment_cosine_sim', None)

        # Initialize enhanced Procrustes variables (set by _load_rotation_matrix)
        self.procrustes_use_centering = False
        self.procrustes_hpl_mean = None
        self.procrustes_gat_mean = None

        if self.is_procrustes_aligned:
            self.procrustes_rotation_matrix = self._load_rotation_matrix()
            if self.procrustes_rotation_matrix is None:
                self._log(f"  ⚠️  Procrustes flag set but rotation matrix not found")
                self.is_procrustes_aligned = False
        else:
            self._log(f"  ⚠️  Procrustes alignment NOT available")
            self._log(f"     Zero-shot prediction will use GAT-only fallback")

        self._log(" 🎉 Model loaded successfully! 🥳")
        self._log(f"   Prediction hardware: {_describe_tf_device()}")
        if self.config.use_full_ontology:
             self._log(f"   Supported cell types: {len(self.celltype_gat.full_classes) if hasattr(self.celltype_gat, 'full_classes') and self.celltype_gat.full_classes is not None else 'All'}")
    def _load_models_from_hdf5(self, *args, **kwargs):
        return _predictor_support._load_models_from_hdf5(self, *args, **kwargs)

    
    def _load_hpl_head(self, *args, **kwargs):
        return _predictor_support._load_hpl_head(self, *args, **kwargs)

    
    def _load_memory_bank(self, *args, **kwargs):
        return _predictor_support._load_memory_bank(self, *args, **kwargs)


    def _load_rotation_matrix(self, *args, **kwargs):
        return _predictor_support._load_rotation_matrix(self, *args, **kwargs)


    def _load_checkpoint_metadata(self, *args, **kwargs):
        return _predictor_support._load_checkpoint_metadata(self, *args, **kwargs)

    
    def _extract_gene_order(self, checkpoint_data: Dict[str, Any]):
        """Extract gene order from checkpoint."""
        gene_order = checkpoint_data.get('gene_order')
        
        if gene_order is None:
            raise ValueError(
                "Checkpoint does not contain gene order information! "
                "This checkpoint may be too old. Please retrain your model "
                "or use Gene_df.csv with the correct gene order."
            )
        
        self.gene_ids = [str(g) for g in gene_order]
        self.num_genes = len(self.gene_ids)
        
        # Validate with model config
        model_config = checkpoint_data.get('model_config', {})
        if 'input_dim' in model_config:
            expected_num_genes = model_config['input_dim']
            if self.num_genes != expected_num_genes:
                raise ValueError(
                    f"Gene count mismatch! Checkpoint expects {expected_num_genes} genes, "
                    f"but gene_order contains {self.num_genes} genes."
                )
    
    def _extract_normalization_config(self, checkpoint_data: Dict[str, Any]):
        """Extract normalization configuration from checkpoint."""
        norm_config = checkpoint_data.get('normalization_config')
        
        if norm_config is None:
            warnings.warn(
                "Checkpoint does not contain normalization config. "
                "Using default parameters (may cause prediction errors)."
            )
            self.normalization_config = {
                'method': 'log1p_norm',
                'target_median': 1000.0,
                'global_median': 0.4,
                'global_iqr': 2.0,
                'clip_values': True,
                'clip_percentile': 99.5
            }
            self._log(f"  ⚠️  Using default normalization config:")
        else:
            self.normalization_config = norm_config

    def _normalize_data(self, X):
        """Normalize gene expression data using same method as training.

        Accepts both dense ``np.ndarray`` and ``scipy.sparse.csr_matrix``.
        Sparse input is processed without ever materializing a full dense
        matrix — all operations act on the stored nonzero values.
        """
        import scipy.sparse
        if scipy.sparse.issparse(X):
            return self._normalize_data_sparse(X)

        # --- Dense path (unchanged) ---
        library_sizes = X.sum(axis=1, keepdims=True)
        library_sizes = np.maximum(library_sizes, 1.0)  # Avoid division by zero
        
        target_median = self.normalization_config['target_median']
        X_scaled = X * (target_median / library_sizes)
        
        # Log1p transformation
        X_normalized = np.log1p(X_scaled)
        
        # Clip aberrant values (Strictly matching training pipeline)
        if self.normalization_config.get('clip_values', True):
            clip_percentile = self.normalization_config.get('clip_percentile', 99.5)

            # A pre-computed dataset-wide cutoff (clip_max_override) makes chunked
            # encoding bit-identical to a single in-process run; absent (default)
            # the per-matrix percentile is used exactly as before.
            override = self.normalization_config.get('clip_max_override')
            if override is not None:
                clip_max = np.float32(override)
            else:
                # Compute percentile on NON-ZERO values only
                # For sparse data (90%+ zeros), computing on all values would clip meaningful signals
                nonzero_mask = X_normalized > 0
                if nonzero_mask.any():
                    # Get non-zero values and compute percentile
                    nonzero_values = X_normalized[nonzero_mask]
                    clip_max = np.percentile(nonzero_values, clip_percentile)
                else:
                    # Fallback: if all zeros, use max (no clipping needed)
                    clip_max = X_normalized.max() if X_normalized.size > 0 else 1.0

            # Record the cutoff actually used so compute_global_clip_max() can read it.
            self.normalization_config['_last_clip_max'] = float(clip_max)
            # Apply clipping
            X_normalized = np.clip(X_normalized, 0, clip_max)
        
        return X_normalized.astype(np.float32)

    def _normalize_data_sparse(self, X):
        """Sparse-aware normalization — operates on ``.data`` only.

        Numerically identical to the dense path.  All operations preserve
        sparsity (``log1p(0)=0``, ``clip(0,…)=0``), so the stored-zero
        count never grows.
        """
        import scipy.sparse
        X = X.tocsr().astype(np.float32, copy=True)

        library_sizes = np.asarray(X.sum(axis=1)).ravel()
        library_sizes = np.maximum(library_sizes, np.float32(1.0))

        target_median = np.float32(self.normalization_config['target_median'])
        scale_factors = target_median / library_sizes

        D = scipy.sparse.diags(scale_factors, format="csr", dtype=np.float32)
        X = D @ X

        X.data = np.log1p(X.data)

        if self.normalization_config.get('clip_values', True):
            clip_percentile = self.normalization_config.get('clip_percentile', 99.5)
            override = self.normalization_config.get('clip_max_override')
            if override is not None:
                clip_max = np.float32(override)
            else:
                positive_data = X.data[X.data > 0]
                if positive_data.size > 0:
                    clip_max = np.float32(np.percentile(positive_data, clip_percentile))
                else:
                    clip_max = np.float32(1.0)
            self.normalization_config['_last_clip_max'] = float(clip_max)
            np.clip(X.data, 0, clip_max, out=X.data)

        X.eliminate_zeros()
        return X

    def compute_global_clip_max(self, adata) -> "Optional[float]":
        """Dataset-wide normalization clip cutoff.

        Computes one cutoff through the normal preprocessing path so chunked
        encoding can apply a consistent value. Callers may provide it through
        ``normalization_config['clip_max_override']``. The operation is sparse-safe
        and CPU-only. Returns ``None`` when clipping is disabled.
        """
        if not self.normalization_config.get('clip_values', True):
            return None
        saved_override = self.normalization_config.pop('clip_max_override', None)
        self.normalization_config.pop('_last_clip_max', None)
        try:
            # _prepare_expression -> _preprocess -> _normalize_data records the
            # cutoff it used in normalization_config['_last_clip_max'].
            # Shallow copy: share X/raw.X (read-only in this path), only
            # deep-copy var (_auto_detect_gene_ids writes to it).
            _tmp = anndata.AnnData(
                X=adata.X, obs=adata.obs, var=adata.var.copy(),
                layers={k: v for k, v in adata.layers.items()},
            )
            if adata.raw is not None:
                _raw_proxy = anndata.AnnData(X=adata.raw.X, var=adata.raw.var)
                _tmp.raw = _raw_proxy
                del _raw_proxy
            self._prepare_expression(_tmp)
            del _tmp
        finally:
            if saved_override is not None:
                self.normalization_config['clip_max_override'] = saved_override
        cutoff = self.normalization_config.get('_last_clip_max')
        return float(cutoff) if cutoff is not None else None

    def _get_adaptive_entropy_threshold(self, entropy_scores: np.ndarray) -> float:
        """
        Automatically determine entropy threshold using Multi-Otsu thresholding.
        
        This method identifies uncertain cells for GRIT refinement by finding
        a natural threshold in the entropy distribution.
        
        Strategy:
        1. Check for highly confident datasets (low variance) - skip filtering
        2. Try Multi-Otsu (3-class) to separate Low/Mid/High entropy
        3. Fall back to simple Otsu or statistical threshold
        
        Args:
            entropy_scores: [n_cells] entropy values for each cell
            
        Returns:
            threshold: Entropy threshold above which cells are considered uncertain
        """
        n = len(entropy_scores)
        if n < 10:
            return 0.0
        
        # Quick variance check for highly confident datasets
        # If all cells have very similar (low) entropy, no filtering needed
        if np.std(entropy_scores) < 0.05:
            return float(np.max(entropy_scores) + 1.0)  # No filtering
        
        # Try Multi-Otsu thresholding (3-class separation)
        try:
            from skimage.filters import threshold_multiotsu
            # 3 classes: Low entropy (confident), Mid (borderline), High (uncertain)
            thresholds = threshold_multiotsu(entropy_scores, classes=3)
            # Use the highest threshold to identify truly uncertain cells
            return float(thresholds[-1])
        except ImportError:
            pass
        except Exception:
            pass
        
        # Fallback to standard Otsu
        try:
            from skimage.filters import threshold_otsu
            return float(threshold_otsu(entropy_scores))
        except ImportError:
            pass
        except Exception:
            pass
        
        # Final fallback: mean + 1 std (simple statistical threshold)
        return float(np.mean(entropy_scores) + np.std(entropy_scores))
    
    def _get_variance_threshold(self, cell_variance: np.ndarray) -> float:
        """Compute an adaptive variance threshold using the Kneedle algorithm
        on the log-log rank plot of cell variance values.

        Inspired by Cell Ranger's barcode knee detection: sort values descending,
        transform to log-log space, draw a chord from first to last point, and
        find the point of maximum perpendicular distance (the knee). Restricted
        to above-median values to focus on the steep-to-flat transition.

        Fallback order:
        1. < 10 cells → 0.0
        2. CV < 0.10  → max + 1.0 (uniform quality, skip filtering)
        3. Kneedle on above-median log-log rank plot
        4. IQR outlier fence (Q3 + 1.5·IQR)
        """
        n = len(cell_variance)
        if n < 10:
            return 0.0

        mean_val = float(np.mean(cell_variance))
        std_val = float(np.std(cell_variance))

        # CV guard: uniform distribution means all cells are similar quality
        if mean_val > 0 and std_val / mean_val < 0.10:
            self._log(
                "Cell variance distribution is uniform — skipping quality filtering"
            )
            return float(np.max(cell_variance) + 1.0)

        # Kneedle algorithm on above-median log-log rank plot
        try:
            sorted_desc = np.sort(cell_variance)[::-1]
            median_val = np.median(cell_variance)
            n_above = int(np.sum(sorted_desc > median_val))
            if n_above >= 10:
                y = sorted_desc[:n_above]
                x = np.arange(1, n_above + 1)
                log_x = np.log10(x)
                log_y = np.log10(y + 1e-10)
                # Normalize to [0, 1]
                lx = (log_x - log_x[0]) / (log_x[-1] - log_x[0] + 1e-10)
                ly = (log_y - log_y[0]) / (log_y[-1] - log_y[0] + 1e-10)
                # Perpendicular distance from each point to the chord
                dist = np.abs(
                    (ly[-1] - ly[0]) * lx
                    - (lx[-1] - lx[0]) * ly
                    + lx[-1] * ly[0]
                    - ly[-1] * lx[0]
                )
                knee_idx = int(np.argmax(dist))
                threshold = float(sorted_desc[knee_idx])
                self._log(
                    f"  Variance threshold (Kneedle): {threshold:.6f} "
                    f"(knee at rank {knee_idx + 1}/{n_above})"
                )
                return threshold
        except Exception:
            pass

        # Fallback: IQR outlier fence
        q1 = float(np.percentile(cell_variance, 25))
        q3 = float(np.percentile(cell_variance, 75))
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
        self._log(f"  Variance threshold (IQR fallback): {threshold:.6f}")
        return threshold
    
    def _estimate_encoder_memory_terms(self):
        d = self.vgae_encoder.shared_dim
        k = self.config.k_neighbors
        n_genes = self.vgae_encoder.num_genes
        attn_type = self.vgae_encoder.graph_attention_type
        latent_dim = self.vgae_encoder.latent_dim
        n_anchors = 0
        if self.anchor_features is not None:
            n_anchors = int(self.anchor_features.shape[0])
        return _predictor_support._estimate_encoder_memory_terms(
            n_genes=n_genes,
            shared_dim=d,
            k_neighbors=k,
            graph_attention_type=attn_type,
            latent_dim=latent_dim,
            n_anchors=n_anchors,
        )


    def _estimate_classification_bytes_per_row(self, *args, **kwargs):
        return _predictor_support._estimate_classification_bytes_per_row(self, *args, **kwargs)


    def _resolve_stream_batch_size(self, *args, **kwargs):
        return _predictor_support._resolve_stream_batch_size(self, *args, **kwargs)


    def _get_gpu_aware_batch_size(self, *args, **kwargs):
        return _predictor_support._get_gpu_aware_batch_size(self, *args, **kwargs)

    
    def _build_knn_graph_on_expression(
        self,
        X: tf.Tensor,
        anchor_features: Optional[tf.Tensor] = None,
        silent: bool = False,
        query_indices: Optional[np.ndarray] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Build k-NN graph on normalized gene expression using GPU-accelerated cosine similarity.

        Uses a double-tiled approach: both query and candidate dimensions are
        tiled, with a running top-k buffer per query cell across all candidate
        tiles.  This bounds peak GPU memory to (query_tile × cand_tile × 4)
        bytes rather than scaling with the full dataset size.

        Uses cosine similarity on normalized expression (not embeddings).
        Falls back to an exact CPU cosine k-NN (same metric) when GPU memory
        allocation fails.

        Candidate-pool convention (anchors-only + explicit self-loops)
        --------------------------------------------------------------
        When ``anchor_features`` is provided, the k-NN candidate pool is
        ``anchor_features`` ONLY — ``X`` is not concatenated in.  Output
        target indices are then shifted by ``len(X)`` so they remain valid
        offsets into the ``[X, anchor_features]`` combined tensor that the
        encoder caller assembles.  An explicit ``(q, q)`` self-edge is
        appended for every query node ``q`` so each cell's neighborhood
        always includes itself.

        This convention makes per-cell embeddings a function of
        ``(cell, anchors, model)`` only, eliminating the structural
        batch-composition dependence across hardware, subsamples, and
        orderings.

        When ``anchor_features`` is ``None`` (e.g., the GRIT call site),
        the candidate pool is ``X`` itself and no self-loops are added.

        Args:
            X: [n_cells, n_genes] normalized gene expression
            anchor_features: [n_anchors, n_genes] anchor cell features.
                When provided (encoder path), triggers the anchors-only +
                self-loop convention described above.  When ``None`` (GRIT
                path), candidates are ``X`` only with no self-loops.
            silent: If True, suppress logging (used during batch processing
                to avoid tqdm interruption).
            query_indices: Optional array of row indices into X to use as
                queries.  When provided, only these rows are used as query
                cells.  When None, all rows of X are used as queries.

        Returns:
            edge_index: [2, n_edges] edge indices into ``[X, anchor_features]``
                (or into ``X`` if ``anchor_features`` is None).
            edge_weights: [n_edges] edge weights (all ones).
        """
        # Preserve host arrays before accelerator work so CPU recovery does not
        # depend on converting tensors after a device failure.
        import scipy.sparse
        _X_is_tensor = isinstance(X, tf.Tensor)
        if _X_is_tensor:
            try:
                X_backup = X.numpy()
            except Exception:
                X_backup = None
        else:
            X_backup = X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)
        try:
            anchor_backup = anchor_features.numpy() if anchor_features is not None else None
        except Exception:
            anchor_backup = None

        n_cells = int(X.shape[0])
        k = min(self.config.k_neighbors, n_cells - 1)
        
        try:
            # GPU cosine similarity on normalized expression.
            
            # Anchors-only candidates when anchors are provided (encoder path).
            # X is dropped from the candidate pool so each query cell's k-NN
            # depends only on the fixed anchor set — not on which other cells
            # happen to be batched alongside it.  Indices come back as
            # anchor-relative (0..n_anchors-1) and are shifted by n_cells at
            # edge-emission time to remain valid in the [X, anchors] combined
            # tensor the encoder caller assembles.  When no anchors are
            # provided (GRIT caller), keep batch-only behavior.
            use_anchors_only = anchor_features is not None
            if use_anchors_only:
                candidates_x = anchor_features
            else:
                candidates_x = X
            
            # Stream normalization into similarity computation so a full dense
            # normalized candidate matrix is not required on the tiled path.
            
            def _get_gpu_aware_tile_sizes(n_queries: int, n_candidates: int, n_features: int, k: int) -> Tuple[int, int]:
                """Calculate optimal query and candidate tile sizes for double-tiled
                similarity computation, using TF's true VRAM headroom.

                Headroom = ``OS-free + TF arena slack``, the slack counted only for
                memory THIS process holds — correct under
                both growth-on (HECTOR's default) and eager-grab modes.
                See :meth:`_get_gpu_aware_batch_size` for the derivation;
                this helper mirrors that strategy.

                Memory budget per (query_tile, cand_tile) block:
                    sim_matrix:  qt × ct × 4
                    query_norm:  qt × n_features × 4
                    cand_norm:   ct × n_features × 4
                    topk_buffer: qt × k × 8  (values float32 + indices int32)

                Returns:
                    (query_tile_size, candidate_tile_size) tuple, each clamped to [256, 8192].
                """
                DEFAULT_TILE_SIZE = 4096

                if not tf.config.list_physical_devices('GPU'):
                    return (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE)

                # Primary: shared TF-allocator probe (OS-free + our own arena slack).
                available_bytes = _predictor_support._available_vram_bytes(allocator="tf")

                # Fallback: cupy total × 0.70
                if available_bytes is None:
                    try:
                        import cupy as cp
                        _, cupy_total = cp.cuda.Device().mem_info
                        available_bytes = int(cupy_total * 0.70)
                    except Exception:
                        return (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE)

                # Compute tile sizes (overhead_factor accounts for TF internals)
                if available_bytes <= 0:
                    return (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE)

                overhead = 2.0
                # Heuristic: set ct first, then derive qt from remaining budget
                # ct = min(n_candidates, sqrt(available / (4 * overhead)))
                ct = min(n_candidates, int(math.sqrt(available_bytes / (4 * overhead))))
                # qt = available / ((ct + n_features) * 4 * overhead)
                denominator = (ct + n_features) * 4 * overhead
                if denominator <= 0:
                    return (DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE)
                qt = min(n_queries, int(available_bytes / denominator))

                # Clamp both to [256, 8192]
                qt = max(256, min(qt, 8192))
                ct = max(256, min(ct, 8192))
                return (qt, ct)
            
            # Get dimensions
            n_features = int(X.shape[1])
            n_candidates = int(candidates_x.shape[0])
            
            # Determine query set: use query_indices if provided, else all rows
            if query_indices is not None:
                query_row_indices = np.asarray(query_indices, dtype=np.int32)
            else:
                query_row_indices = np.arange(n_cells, dtype=np.int32)
            n_queries = len(query_row_indices)
            
            # Calculate tile sizes for double-tiled processing
            QUERY_TILE_SIZE, CAND_TILE_SIZE = _get_gpu_aware_tile_sizes(n_queries, n_candidates, n_features, k)
            
            # DOUBLE-TILED APPROACH:
            # Outer loop: iterate query tiles (slices of query_row_indices)
            # Inner loop: iterate candidate tiles (slices of candidates_x)
            # Per (query_tile, cand_tile): L2-normalize both on-the-fly,
            # compute cosine similarity block, update running top-k buffer.
            # Peak memory bounded to query_tile × cand_tile × 4 bytes
            # plus normalization and buffer overhead.
            # This NEVER materializes the full normalized candidates tensor.
            
            # Use one candidate matmul when the complete candidate set fits the
            # estimated budget; otherwise retain the memory-bounded tiled path.
            use_single_shot = False
            candidates_norm_full = None
            try:
                import cupy as _cp
                _free, _ = _cp.cuda.Device().mem_info
                _need_bytes = (
                    QUERY_TILE_SIZE * n_candidates * 4   # similarity block
                    + n_candidates * n_features * 4      # normalized candidates (resident)
                    + QUERY_TILE_SIZE * n_features * 4   # query tile
                )
                use_single_shot = (_need_bytes * 2.0) < _free
            except Exception:
                use_single_shot = False

            if use_single_shot:
                # Normalize candidates once. The anchor pool is fixed across
                # batches, so cache its normalized form (built only on this
                # fast path, i.e. only when memory is ample); the GRIT path
                # (candidates == X) varies per call, so normalize fresh.
                if use_anchors_only:
                    _cached = getattr(self, "_knn_anchor_norm_cache", None)
                    if _cached is None or int(_cached.shape[0]) != n_candidates:
                        _cached = tf.math.l2_normalize(
                            tf.convert_to_tensor(candidates_x, dtype=tf.float32), axis=1)
                        self._knn_anchor_norm_cache = _cached
                    candidates_norm_full = _cached
                else:
                    _cand_dense = (candidates_x.toarray()
                                   if scipy.sparse.issparse(candidates_x) else candidates_x)
                    candidates_norm_full = tf.math.l2_normalize(
                        tf.convert_to_tensor(_cand_dense, dtype=tf.float32), axis=1)

            all_edge_indices = []

            num_query_tiles = (n_queries + QUERY_TILE_SIZE - 1) // QUERY_TILE_SIZE
            query_tile_iter = range(0, n_queries, QUERY_TILE_SIZE)
            if not silent and n_queries > QUERY_TILE_SIZE:
                from tqdm import tqdm as _tqdm
                query_tile_iter = _tqdm(
                    query_tile_iter,
                    total=num_query_tiles,
                    desc="  kNN graph",
                    unit="tile",
                    ncols=80
                )
            for qi in query_tile_iter:
                qi_end = min(qi + QUERY_TILE_SIZE, n_queries)
                q_indices_tile = query_row_indices[qi:qi_end]
                tile_size = qi_end - qi
                
                # Gather and normalize query tile on-the-fly
                if _X_is_tensor:
                    query_tile = tf.gather(X, q_indices_tile)
                else:
                    _q_rows = X[q_indices_tile]
                    if scipy.sparse.issparse(_q_rows):
                        _q_rows = _q_rows.toarray()
                    query_tile = tf.constant(np.asarray(_q_rows, dtype=np.float32))
                query_tile_norm = tf.math.l2_normalize(query_tile, axis=1)
                del query_tile
                
                if use_single_shot:
                    # Fast path: one matmul + one top_k over ALL candidates,
                    # using the once-normalized candidate set. Skips the tiled
                    # candidate loop below (empty range).
                    topk_values, topk_indices = _knn_topk_cosine_full(
                        query_tile_norm, candidates_norm_full, k)
                    _cand_tile_range = []
                else:
                    # Initialize running top-k buffer for this query tile
                    # Start with -infinity values so any real similarity wins
                    topk_values = tf.fill([tile_size, k], -float('inf'))
                    topk_indices = tf.zeros([tile_size, k], dtype=tf.int32)
                    _cand_tile_range = range(0, n_candidates, CAND_TILE_SIZE)

                # Inner loop: iterate candidate tiles (empty on the fast path)
                for cj in _cand_tile_range:
                    cj_end = min(cj + CAND_TILE_SIZE, n_candidates)

                    # Normalize candidate tile on-the-fly
                    _cand_slice = candidates_x[cj:cj_end]
                    if scipy.sparse.issparse(_cand_slice):
                        cand_tile = tf.constant(_cand_slice.toarray(), dtype=tf.float32)
                    elif isinstance(_cand_slice, np.ndarray):
                        cand_tile = tf.constant(_cand_slice, dtype=tf.float32)
                    else:
                        cand_tile = _cand_slice
                    cand_tile_norm = tf.math.l2_normalize(cand_tile, axis=1)
                    del cand_tile
                    
                    # Compute cosine similarity block: [tile_size, cand_tile_size]
                    sim_block = tf.matmul(query_tile_norm, cand_tile_norm, transpose_b=True)
                    del cand_tile_norm

                    # Deterministic tie-break: bias each column by a tiny epsilon
                    # that decreases with the GLOBAL candidate index. Identical
                    # cosine similarities then resolve to the lower index. The
                    # bias is monotonic in global index, so it survives the
                    # cross-tile merge below.
                    global_idx_bias = -1e-7 * tf.cast(tf.range(cj, cj_end), tf.float32) / float(max(n_candidates, 1))
                    sim_block = sim_block + global_idx_bias[tf.newaxis, :]

                    # Get top-k from this candidate tile
                    cand_tile_k = min(k, cj_end - cj)
                    tile_topk_vals, tile_topk_local_idx = tf.math.top_k(sim_block, k=cand_tile_k)
                    del sim_block
                    
                    # Map local indices to global candidate indices
                    tile_topk_global_idx = tile_topk_local_idx + cj
                    
                    # If this candidate tile had fewer than k candidates, pad to k
                    if cand_tile_k < k:
                        pad_width = k - cand_tile_k
                        tile_topk_vals = tf.concat([
                            tile_topk_vals,
                            tf.fill([tile_size, pad_width], -float('inf'))
                        ], axis=1)
                        tile_topk_global_idx = tf.concat([
                            tile_topk_global_idx,
                            tf.zeros([tile_size, pad_width], dtype=tf.int32)
                        ], axis=1)
                    
                    # Merge with running buffer: concatenate along axis=1, re-select top-k
                    combined_vals = tf.concat([topk_values, tile_topk_vals], axis=1)  # [tile_size, 2k]
                    combined_indices = tf.concat([topk_indices, tile_topk_global_idx], axis=1)  # [tile_size, 2k]
                    
                    # Re-select overall top-k from combined
                    _, reselect_idx = tf.math.top_k(combined_vals, k=k)
                    
                    # Gather the corresponding global indices and values
                    # reselect_idx: [tile_size, k] — indices into axis=1 of combined tensors
                    batch_idx = tf.repeat(
                        tf.expand_dims(tf.range(tile_size), axis=1), k, axis=1
                    )  # [tile_size, k]
                    gather_idx = tf.stack([batch_idx, reselect_idx], axis=2)  # [tile_size, k, 2]
                    
                    topk_values = tf.gather_nd(combined_vals, gather_idx)
                    topk_indices = tf.gather_nd(combined_indices, gather_idx)
                    
                    del combined_vals, combined_indices, reselect_idx, gather_idx
                
                del query_tile_norm

                # Anchors-only path: shift anchor-relative indices into the
                # combined [X, anchors] layout the encoder expects, and append
                # an explicit (q, q) self-edge per query so each cell appears
                # in its own neighborhood (replacing the implicit self-loop
                # that previously came from the cell being in candidates).
                # Done per-query so the downstream uniform-k reshape works.
                if use_anchors_only:
                    topk_indices = topk_indices + n_cells
                    self_col = tf.constant(q_indices_tile, dtype=tf.int32)[:, None]
                    topk_indices = tf.concat([topk_indices, self_col], axis=1)
                    edges_per_query = k + 1
                else:
                    edges_per_query = k

                # Emit edges from final top-k buffer for this query tile.
                # q_indices_tile contains the global row indices of the query cells.
                global_sources = tf.repeat(
                    tf.constant(q_indices_tile, dtype=tf.int32), edges_per_query
                )
                global_targets = tf.reshape(topk_indices, [-1])

                chunk_edges = tf.stack([global_sources, global_targets], axis=0)
                all_edge_indices.append(chunk_edges)

                del topk_values, topk_indices
            
            # Combine all edges (this is small: 2 x (n_queries * k) ints)
            edge_index = tf.concat(all_edge_indices, axis=1)
            
            # All edges have weight 1.0
            edge_weights = tf.ones(tf.shape(edge_index)[1], dtype=tf.float32)
            
            return edge_index, edge_weights
            
        except Exception as e:
            # Fallback to CPU-based method for very large datasets
            self._log(f"  ⚠️  GPU method failed ({e}), falling back to CPU-based k-NN")
            
            # Check if we have backup arrays
            if X_backup is None:
                raise RuntimeError(
                    "GPU k-NN failed and CPU fallback not available. "
                    "Could not retrieve numpy arrays from tensors."
                ) from e
            
            # Use pre-saved numpy arrays (from before GPU corruption)
            X_np = X_backup

            # Anchors-only candidates when anchors are provided (mirrors GPU path).
            cpu_use_anchors_only = anchor_backup is not None
            if cpu_use_anchors_only:
                candidates_np = anchor_backup
            else:
                candidates_np = X_np

            # Exact COSINE k-NN on the CPU — same metric as the GPU path.
            # Queries are chunked so peak RAM stays at ~(chunk x n_candidates);
            # the candidate pool is L2-normalized inside the helper.
            CPU_CHUNK_SIZE = 4096
            all_edge_lists = []
            
            # Determine which rows to query
            if query_indices is not None:
                query_rows = X_np[query_indices]
                query_global_indices = query_indices
                n_queries = len(query_indices)
            else:
                query_rows = X_np
                query_global_indices = np.arange(n_cells)
                n_queries = n_cells
            
            num_chunks = (n_queries + CPU_CHUNK_SIZE - 1) // CPU_CHUNK_SIZE
            
            for i in range(num_chunks):
                start_idx = i * CPU_CHUNK_SIZE
                end_idx = min(start_idx + CPU_CHUNK_SIZE, n_queries)
                
                batch_X = query_rows[start_idx:end_idx]
                indices = _encoder_knn_cosine_cpu_topk(batch_X, candidates_np, k)
                
                # Build edge list for this batch.  Mirror the GPU path:
                # shift anchor-relative indices into the combined [X, anchors]
                # layout, append a (q, q) self-edge per query.  Skip both
                # operations when there are no anchors (GRIT path).
                shift = n_cells if cpu_use_anchors_only else 0
                for batch_row, neighbor_indices in enumerate(indices):
                    global_source_idx = int(query_global_indices[start_idx + batch_row])
                    for neighbor_idx in neighbor_indices:
                        all_edge_lists.append(
                            [global_source_idx, int(neighbor_idx) + shift]
                        )
                    if cpu_use_anchors_only:
                        all_edge_lists.append(
                            [global_source_idx, global_source_idx]
                        )
            
            # Convert to tensors
            edge_index = tf.constant(np.array(all_edge_lists).T, dtype=tf.int32)
            edge_weights = tf.ones(edge_index.shape[1], dtype=tf.float32)
            
            return edge_index, edge_weights
    
    
    def _compute_profile_matching_scores(
        self,
        z_latent: tf.Tensor,
        prototypes_seen: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute "Profile-Based Voting" scores using Pearson Correlation.
        
        This implements the Topological Grounding logic:
        1. Compute "Observed Profile": Correlation of cell z with all N_seen Expert Prototypes.
           -> "How similar am I to CD4, CD8, NK, B-cell...?"
        2. Fetch "Theoretical Profiles": The rows of the Prior Matrix (PPR) for every candidate class.
           -> "How similar SHOULD a Neuron be to CD4, CD8, NK, B-cell...?"
        3. Score = Correlation(Observed, Theoretical)
        
        Args:
            z_latent: [batch_size, embedding_dim] latent embeddings
            prototypes_seen: [num_seen_classes, embedding_dim] expert prototypes
            
        Returns:
            profile_scores: [batch_size, num_all_classes] correlation scores (0 to 1)
        """
        # Lazy load/compute PPR matrix — recompute if PPR config changed
        ppr_config_key = (
            self.config.use_asymmetric_ppr,
            self.config.forward_weight,
            self.config.ppr_alpha,
        )
        if not hasattr(self, 'prior_matrix_full') or getattr(self, '_ppr_config_key', None) != ppr_config_key:
            self._compute_enhanced_prior_matrix()
            self._ppr_config_key = ppr_config_key
            self.seen_indices_np = self._compute_seen_indices().numpy()

        # 1. Compute Observed Profile (Cosine Similarity to Seen Prototypes)
        # z_norm: [batch, dim]
        # protos_norm: [N_seen, dim]
        z_norm = tf.nn.l2_normalize(z_latent, axis=1)
        protos_norm = tf.nn.l2_normalize(prototypes_seen, axis=1)
        
        # sim_obs: [batch, N_seen]
        sim_obs = tf.matmul(z_norm, protos_norm, transpose_b=True)
        
        # 2. Prepare Theoretical Profiles
        # We need the PPR profile for ALL candidate classes, but only looking at the columns corresponding to SEEN classes.
        # PPR Matrix: [N_all, N_all]
        # Slice columns: [N_all, N_seen]
        # This tells us: "For every candidate class (row), what is its theoretical similarity to the seen landmarks?"
        theoretical_profiles = tf.gather(
            tf.constant(self.prior_matrix_full, dtype=tf.float32), 
            self.seen_indices_np, 
            axis=1
        ) # [N_all, N_seen]
        
        # 3. Compute Ranked Correlation (Simulated Spearman)
        # True Spearman is hard in TF (requires sorting). We use Pearson on centered data as a proxy, 
        # or just Cosine Similarity on centered profiles (which is Pearson).
        # We center across the "profile" dimension (axis=1) to match relative shapes.
        
        # Center Observed: subtract mean across seen classes
        obs_mean = tf.reduce_mean(sim_obs, axis=1, keepdims=True)
        obs_centered = sim_obs - obs_mean
        obs_normalized = tf.nn.l2_normalize(obs_centered, axis=1)
        
        # Center Theoretical: subtract mean across seen classes
        theo_mean = tf.reduce_mean(theoretical_profiles, axis=1, keepdims=True)
        theo_centered = theoretical_profiles - theo_mean
        theo_normalized = tf.nn.l2_normalize(theo_centered, axis=1)
        
        # Compute Correlation: [batch, N_seen] @ [N_all, N_seen].T -> [batch, N_all]
        profile_scores = tf.matmul(obs_normalized, theo_normalized, transpose_b=True)
        
        # Clip to [0, 1] for safety (though Pearson is [-1, 1])
        # We only care about positive correlation (similar shapes)
        profile_scores = tf.maximum(profile_scores, 0.0)
        
        return profile_scores

    def _compute_asymmetric_ppr(self, adj: np.ndarray, alpha: float = 0.15, forward_weight: float = 0.6) -> np.ndarray:
        """Compute asymmetric (directional) PPR by blending forward and backward PPR.

        Forward PPR uses the original adjacency (child-to-parent edges).
        Backward PPR uses the transposed adjacency (parent-to-child edges).
        Both are row-normalized individually, then blended and row-normalized again.

        Args:
            adj: Adjacency matrix of shape [N, N], non-negative.
            alpha: PPR restart probability (default: 0.15).
            forward_weight: Blending weight for forward PPR in [0.0, 1.0] (default: 0.6).

        Returns:
            Asymmetric PPR matrix of shape [N, N], float32, row-normalized.
        """
        n = adj.shape[0]
        identity = np.eye(n, dtype=np.float64)

        def _ppr_from_adj(a: np.ndarray) -> np.ndarray:
            """Compute PPR matrix from a single directed adjacency matrix.

            Adds self-loops, applies symmetric normalization D^{-1/2} A D^{-1/2},
            then computes alpha * (I - (1-alpha) * A_norm)^{-1} and row-normalizes.
            """
            a_loop = a + identity
            degrees = np.sum(a_loop, axis=1)
            degrees = np.maximum(degrees, 1e-8)
            d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
            norm_adj = d_inv_sqrt @ a_loop @ d_inv_sqrt

            matrix_to_invert = identity - (1 - alpha) * norm_adj
            try:
                ppr = alpha * np.linalg.inv(matrix_to_invert)
            except np.linalg.LinAlgError:
                self._log(" ⚠️ PPR inversion failed in _compute_asymmetric_ppr, returning identity")
                return identity.astype(np.float32)

            # Row-normalize
            row_sums = ppr.sum(axis=1, keepdims=True)
            ppr = ppr / np.maximum(row_sums, 1e-8)
            return ppr.astype(np.float32)

        forward_ppr = _ppr_from_adj(adj.astype(np.float64))
        backward_ppr = _ppr_from_adj(adj.T.astype(np.float64))

        # Blend forward and backward
        combined = forward_weight * forward_ppr + (1.0 - forward_weight) * backward_ppr

        # Row-normalize the combined result
        row_sums = combined.sum(axis=1, keepdims=True)
        combined = combined / np.maximum(row_sums, 1e-8)

        return combined.astype(np.float32)



    def _compute_enhanced_prior_matrix(self) -> np.ndarray:
        """Compute enhanced PPR matrix using asymmetric (directional) PPR.

        When ``use_asymmetric_ppr`` is enabled, computes separate forward and
        backward PPR matrices and blends them with ``forward_weight``. Otherwise
        returns the baseline symmetric PPR.

        The result is validated (NaN/Inf replaced with zeros), cast to float32,
        checked for shape [N_all, N_all], and adjusted for diagonal dominance
        before being stored in ``self.prior_matrix_full``.

        Returns:
            Enhanced PPR matrix of shape [N_all, N_all], float32, row-normalized.
        """
        # --- Asymmetric or baseline ---
        if not self.config.use_asymmetric_ppr:
            ppr = self.celltype_gat.get_ppr_matrix(for_full_ontology=True).astype(np.float32)
            self.prior_matrix_full = ppr
            return ppr

        adj = self.celltype_gat.full_ontology_adj_np
        alpha = self.config.ppr_alpha if self.config.ppr_alpha is not None else getattr(self.celltype_gat, 'ppr_alpha', 0.15)
        ppr = self._compute_asymmetric_ppr(adj, alpha=alpha,
                                           forward_weight=self.config.forward_weight)
        ppr = ppr.astype(np.float32)

        # --- Validation: NaN / Inf sanitization ---
        nan_mask = np.isnan(ppr)
        inf_mask = np.isinf(ppr)
        if np.any(nan_mask) or np.any(inf_mask):
            bad_count = int(np.sum(nan_mask) + np.sum(inf_mask))
            self._log(f" ⚠️ Enhanced PPR contains {bad_count} NaN/Inf entries — replacing with zeros")
            ppr = np.nan_to_num(ppr, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Validate shape and dtype ---
        n_all = len(self.celltype_gat.full_classes)
        assert ppr.shape == (n_all, n_all), (
            f"Enhanced PPR shape mismatch: expected ({n_all}, {n_all}), got {ppr.shape}"
        )
        ppr = ppr.astype(np.float32)

        # --- Ensure diagonal dominance (self-similarity is max per row) ---
        for i in range(ppr.shape[0]):
            row_max = np.max(ppr[i])
            if ppr[i, i] < row_max:
                ppr[i, i] = row_max + 1e-6

        # Re-normalize rows after diagonal boost
        row_sums = ppr.sum(axis=1, keepdims=True)
        ppr = ppr / np.maximum(row_sums, 1e-8)
        ppr = ppr.astype(np.float32)

        self.prior_matrix_full = ppr
        return ppr


    def _compute_seen_indices(self) -> tf.Tensor:
        """
        Compute indices of seen (training) classes in full ontology.
        
        Returns:
            seen_indices: [num_seen_classes] indices
        """
        seen_classes = self.celltype_gat.classes
        all_classes = self.celltype_gat.full_classes
        
        # Create mapping from seen class index to full ontology index
        all_classes_set = {cls: idx for idx, cls in enumerate(all_classes)}
        
        indices = []
        for seen_cls in seen_classes:
            if seen_cls in all_classes_set:
                indices.append(all_classes_set[seen_cls])
            else:
                # Fallback: use 0 (shouldn't happen if ontology is consistent)
                indices.append(0)
        
        return tf.constant(indices, dtype=tf.int32)

    def _prepare_expression(self, adata: anndata.AnnData):
        """Shared entry point for expression preprocessing."""

        return self._preprocess(adata)

    def _resolve_expression_source(self, adata: anndata.AnnData) -> Dict[str, Any]:
        """Select the expression slot used for inference and cache validation.

        Thin wrapper over ``predictor_support.resolve_expression_source`` that
        keeps the inference back-calculation path (``allow_log1p=True``) and
        routes the warning through this instance's logger. Returns a dict with
        keys ``name``, ``matrix``, ``var``, ``matrix_type``, ``use_backcalc``.
        """
        return _predictor_support.resolve_expression_source(
            adata, allow_log1p=True, logger=self._log
        )

    def _preprocess(self, adata: anndata.AnnData):
        """Auto-detect gene IDs, reorder genes to match model order, and normalize.

        Args:
            adata: AnnData object with gene expression data.

        Returns:
            X_normalized: Normalized expression matrix as a numpy array with
                genes reordered (and zero-filled) to match the model's expected
                gene order.
        """
        selected = self._resolve_expression_source(adata)
        use_backcalc = bool(selected.get("use_backcalc", False))

        if selected["name"] != "adata.X":
            adata = anndata.AnnData(
                X=selected["matrix"],
                var=selected["var"],
                obs=adata.obs,
            )

        # Step 0: Auto-detect gene IDs
        self._log("  Step 0: Checking gene IDs...")
        # If adata is a view, copy it first to avoid ImplicitModificationWarning
        # when _auto_detect_gene_ids writes to adata.var
        if adata.is_view:
            adata = adata.copy()
        _gene_col = '_hector_gene_ids'
        self._auto_detect_gene_ids(adata, _gene_col)

        # Step 1: Reorder and fill genes
        self._log("  Step 1: Gene Matching and Reordering ...")
        X_reordered = reorder_and_fill_genes(
            adata,
            required_gene_ids=self.gene_ids,
            gene_id_column=_gene_col,
        )

        # Back-calculation applies only to inputs classified as non-negative
        # log1p-like expression.
        if use_backcalc:
            import scipy.sparse
            if scipy.sparse.issparse(X_reordered):
                X_reordered = X_reordered.copy()
                np.clip(X_reordered.data, 0.0, 10.0, out=X_reordered.data)
                X_reordered.data = np.expm1(X_reordered.data)
                X_reordered.eliminate_zeros()
            else:
                X_reordered = np.expm1(np.clip(X_reordered, 0.0, 10.0))

        # Step 2: Normalize
        self._log("  Step 2: Processing expression data...")
        X_normalized = self._normalize_data(X_reordered)

        return X_normalized

    def _run_encoder_batched_mlx(self, X_normalized) -> Tuple[np.ndarray, np.ndarray]:
        """MLX encoder path — no TF dependency, no subprocess recycling."""
        from .backend_mlx import load_mlx_encoder_from_h5, run_encoder_mlx

        if self._mlx_encoder is None:
            self._mlx_encoder, self._mlx_config = load_mlx_encoder_from_h5(
                self.model_path
            )
        anchor_np = (self.anchor_features.numpy()
                     if hasattr(self.anchor_features, 'numpy')
                     else np.asarray(self.anchor_features))

        n_cells = X_normalized.shape[0]

        # --- Adaptive batch sizing ---
        if self.config.batch_size is not None:
            batch_size = min(int(self.config.batch_size), n_cells)
        else:
            batch_size = self._compute_mlx_batch_size(
                X_normalized, anchor_np, n_cells,
            )

        return run_encoder_mlx(
            self._mlx_encoder, X_normalized, anchor_np,
            k_neighbors=self.config.k_neighbors, batch_size=batch_size,
            verbose=self.verbose,
        )

    def _compute_mlx_batch_size(
        self, X_normalized, anchor_np, n_cells,
    ) -> int:
        """Choose an MLX batch size from the current Metal memory model.

        The peak-memory model is:
            ``peak(B) ≈ max(floor, resident + per_cell × B)``
        ``floor`` is the one-time encode_hidden(anchors) working set, measured
        once via mx.get_peak_memory() and cached; ``per_cell`` is dominated by
        the per-batch ``B × n_anchors`` kNN similarity buffer.
        """
        import gc

        default_batch = 2040
        try:
            cfg = self._mlx_config or {}
            hidden_dims = cfg.get("hidden_dims") or []
            shared_dim = (
                hidden_dims[-1] if hidden_dims
                else cfg.get("shared_dim", 1024)
            )
            latent_dim = getattr(self._mlx_encoder, "latent_dim", 768)
            n_genes = X_normalized.shape[1]
            n_anchors = anchor_np.shape[0]

            fixed_floor, per_cell_bytes = (
                _predictor_support._estimate_encoder_memory_terms_metal(
                    n_genes=n_genes,
                    n_anchors=n_anchors,
                    shared_dim=shared_dim,
                )
            )

            # Snapshot available RAM before probing the anchor-side encoder floor.
            # Operating-system reclamation can lag after MLX releases transient
            # buffers, so a post-probe reading can understate available memory.
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except ImportError:
                pass

            # Budget from the unified helper: it reads the Metal working set,
            # MLX's live footprint, and the free unified RAM, and subtracts the
            # mu_out host array (n_cells x latent_dim) which shares the pool. Read
            # Read before the floor probe because macOS may reclaim released MLX
            # pages asynchronously.
            from .backend_mlx import mlx_gpu_budget
            budget = mlx_gpu_budget(op_host_bytes=n_cells * latent_dim * 4)
            if budget is None:              # off-Metal (no MLX device_info)
                return min(default_batch, n_cells)

            # ── Now measure the floor (safe — after the free snapshot) ─
            if self._mlx_floor_bytes is None:
                from .backend_mlx import measure_encoder_floor_mlx
                measured = measure_encoder_floor_mlx(self._mlx_encoder, anchor_np)
                if measured is not None:
                    self._mlx_floor_bytes = measured
            floor = (
                self._mlx_floor_bytes
                if self._mlx_floor_bytes is not None
                else fixed_floor
            )

            # Fail-safe: the anchor floor alone exceeds the budget. Warn the
            # user (the encode may be jetsam-killed) and fall back to the
            # minimum batch to give it the best chance of completing.
            if budget <= floor:
                self._log(
                    f"  ⚠️ MLX memory warning: the encoder's fixed footprint "
                    f"exceeds the available GPU memory (~{max(budget, 0) / (1024 ** 3):.1f} GB)."
                    f"Falling back to the minimum batch size: 1024."
                )
                return min(1024, n_cells)

            margin = _predictor_support._METAL_SIZING_MARGIN
            optimal = max(
                256, int((budget - floor) / (per_cell_bytes * margin))
            )
            batch_size = min(optimal, n_cells)

            if self.verbose:
                self._log(
                    f"  MLX adaptive batch size: {batch_size} "
                    f"(budget {budget / (1024 ** 3):.1f} GB, "
                    f"floor {floor / (1024 ** 3):.1f} GB)"
                )
            return batch_size
        except Exception as exc:
            if self.verbose:
                self._log(
                    f"  MLX batch size probe failed ({exc!r}), "
                    f"using default={default_batch}"
                )
            return min(default_batch, n_cells)

    def _run_encoder_batched(
        self, X_normalized,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run VGAE encoder + cell-type GAT in batches.

        Handles GPU-aware batch sizing, batch graph construction with anchor
        features, and concatenation of per-batch embeddings.

        Args:
            X_normalized: Normalized expression matrix of shape ``[n_cells, n_genes]``.

        Returns:
            mu_class:      Embeddings of shape ``[n_cells, dim]``.
            cell_variance: Scalar per-cell variance (mean of exp(log_var_class)
                           across dimensions).  Shape ``[n_cells]``.
        """
        if _select_encoder_backend() == 'mlx':
            return self._run_encoder_batched_mlx(X_normalized)

        n_cells = X_normalized.shape[0]

        # GPU-aware batch sizing — split fixed (anchor-side) and per-cell
        # terms so the sizer doesn't treat anchor memory as a per-cell cost.
        fixed_bytes, per_cell_bytes = self._estimate_encoder_memory_terms()
        # Fold in the streamed output array (mu_out, n_cells x emb_dim x 4): a
        # real, batch-independent host allocation the per-op estimator doesn't
        # see. Without it the budget is optimistic by ~n_cells*emb_dim*4 bytes.
        emb_dim = int(getattr(self.vgae_encoder, "latent_dim", 768) or 768)
        fixed_bytes += n_cells * emb_dim * 4
        # The overhead factor accounts for graph buffers, concatenated batch and
        # anchor state, and backend workspace beyond the base row storage.
        batch_size = self._get_gpu_aware_batch_size(
            n_cells, fixed_bytes, per_cell_bytes,
            overhead_factor=2.2, default_batch=2040,
        )

        # Stream embeddings into a preallocated output array to avoid a second
        # full-size concatenation allocation. Determine its width from the first
        # batch rather than assuming the checkpoint's embedding dimension.
        mu_out = None
        var_out = np.empty(n_cells, dtype=np.float32)
        import scipy.sparse
        _is_sparse = scipy.sparse.issparse(X_normalized)
        if _is_sparse:
            X_normalized = X_normalized.tocsr()
            if X_normalized.dtype != np.float32:
                X_normalized = X_normalized.astype(np.float32, copy=False)
            _dense_buf = np.empty(
                (batch_size, X_normalized.shape[1]), dtype=np.float32,
            )
        num_batches = (n_cells + batch_size - 1) // batch_size

        # Anchors' hidden representation is constant across batches (the encoder's
        # hidden stack is per-node), so compute it once and reuse it instead of
        # re-running the hidden layers on all anchors every batch.
        h_anchors = None
        if self.anchor_features is not None and self.anchor_features.shape[0] > 0:
            h_anchors = self.vgae_encoder.encode_hidden(
                self.anchor_features, training=False
            )

        batch_iterator = range(0, n_cells, batch_size)
        if self.verbose:
            batch_iterator = tqdm(
                batch_iterator,
                total=num_batches,
                desc="  Encoding",
                unit="batch",
                ncols=80,
            )

        for i in batch_iterator:
            batch_end = min(i + batch_size, n_cells)
            actual = batch_end - i
            if _is_sparse:
                X_normalized[i:batch_end].toarray(out=_dense_buf[:actual])
                batch_X = _dense_buf[:actual]
            else:
                batch_X = X_normalized[i:batch_end]
            batch_X_tensor = tf.constant(batch_X, dtype=tf.float32)

            # Build per-batch k-NN graph.  Because anchors are passed,
            # _build_knn_graph_on_expression uses the anchors-only candidate
            # convention with explicit per-query self-loops, making each query's
            # neighborhood independent of other cells in the batch.
            batch_edge_index, batch_edge_weights = self._build_knn_graph_on_expression(
                batch_X_tensor,
                self.anchor_features,
                silent=True,
            )

            batch_size_actual = tf.shape(batch_X_tensor)[0]

            # Run the per-node hidden stack on the batch only, then reuse the
            # precomputed anchor hidden states. Node feature ordering is
            # [batch; anchors], matching the edge-index layout.
            h_batch = self.vgae_encoder.encode_hidden(batch_X_tensor, training=False)
            if h_anchors is not None:
                h_full = tf.concat([h_batch, h_anchors], axis=0)
            else:
                h_full = h_batch

            _, _, _, _, mu_class, log_var_class = self.vgae_encoder.encode_from_hidden(
                h_full, batch_edge_index, batch_edge_weights,
                batch_size=batch_size_actual, training=False,
            )

            mu_np = mu_class.numpy()
            if mu_out is None:
                mu_out = np.empty((n_cells, mu_np.shape[1]), dtype=mu_np.dtype)
            mu_out[i:batch_end] = mu_np
            # Reduce 768-dim log_var to scalar variance per cell immediately
            var_out[i:batch_end] = np.mean(
                np.exp(np.clip(log_var_class.numpy(), -20, 20)), axis=1
            )

        if mu_out is None:  # n_cells == 0 edge case
            mu_out = np.empty((0, 0), dtype=np.float32)
        return mu_out, var_out

    def _score_embedding_batch_mlx(
        self,
        mu_class: np.ndarray,
        *,
        top_k: int,
        cell_variance: Optional[np.ndarray],
        use_full_ontology: bool,
        return_filtered_scores: bool,
        capture_diag: bool = False,
    ) -> Dict[str, Any]:
        """MLX/numpy scoring path — mirrors :meth:`_score_embedding_batch`.

        Uses :func:`backend_mlx.score_embeddings_np` for the consensus
        scorer and returns the same dict shape the TF path produces so that
        :meth:`_classify_embeddings_batched` is backend-agnostic.
        """
        from .backend_mlx import score_embeddings_np

        mu = np.asarray(mu_class, dtype=np.float32)
        # GAT prototypes (full ontology always; slicing handled inside scorer)
        gat_prototypes_full = self._mlx_celltype_embedder.get_gat_prototypes(
            for_full_ontology=True, normalize=True
        )

        # PPR matrix — lazy-compute once and cache, same as TF path
        ppr_config_key = (
            self.config.use_asymmetric_ppr,
            getattr(self.config, "forward_weight", 0.6),
            getattr(self.config, "ppr_alpha", None),
        )
        if (
            not hasattr(self, "prior_matrix_full")
            or getattr(self, "_ppr_config_key", None) != ppr_config_key
        ):
            self._compute_enhanced_prior_matrix()
            self._ppr_config_key = ppr_config_key

        scored = score_embeddings_np(
            mu,
            gat_prototypes_full,
            self._mlx_hpl_head,
            procrustes=self._mlx_procrustes,
            ontology=self._mlx_ontology,
            prior_matrix_full=self.prior_matrix_full,
            use_full_ontology=use_full_ontology,
            top_k=top_k,
        )

        # Cell variance
        if cell_variance is not None:
            batch_cell_variance = np.asarray(cell_variance, dtype=np.float32)
        else:
            top1_score = np.max(scored["filtered_scores"], axis=1)
            batch_cell_variance = 1.0 - np.clip(top1_score, 0.0, 1.0)

        result = {
            "top_indices": scored["top_indices"],
            "top_scores": scored["top_scores"],
            "cell_variance": batch_cell_variance,
        }
        if return_filtered_scores:
            result["filtered_scores"] = scored["filtered_scores"]
        else:
            result["filtered_scores"] = None

        if capture_diag:
            result["diag"] = {
                "score_generalist": scored["score_generalist"],
                "score_expert": scored["score_expert"],
                "score_profile": scored["score_profile"],
                "expert_confidence": scored["expert_confidence"],
                "base_final_scores": scored["base_final_scores"],
                "aux_classifier_present": self._mlx_hpl_head is not None,
            }
        return result

    def _score_embedding_batch(
        self,
        mu_class: Union[np.ndarray, tf.Tensor],
        *,
        top_k: int,
        cell_variance: Optional[np.ndarray],
        use_full_ontology: Optional[bool],
        return_filtered_scores: bool,
        capture_diag: bool = False,
    ) -> Dict[str, Any]:
        """Run the shared consensus scorer for one embedding batch.

        When ``capture_diag`` is True, the result dict also contains a
        ``diag`` payload exposing the raw per-cell pathway scores and tier
        masks used to derive golden/silver/bronze. Intended for offline
        diagnostics only — adds memory cost roughly proportional to
        (cells x num_classes).
        """

        if use_full_ontology is None:
            use_full_ontology = self.config.use_full_ontology

        # ---- MLX scoring fast-path ----
        if self._mlx_celltype_embedder is not None:
            return self._score_embedding_batch_mlx(
                mu_class,
                top_k=top_k,
                cell_variance=cell_variance,
                use_full_ontology=use_full_ontology,
                return_filtered_scores=return_filtered_scores,
                capture_diag=capture_diag,
            )

        mu_class_tensor = tf.convert_to_tensor(mu_class, dtype=tf.float32)
        batch_size = tf.shape(mu_class_tensor)[0]

        gat_embeddings_full = self.celltype_gat.get_gat_prototypes(
            for_full_ontology=True, training=False
        )
        z_norm = tf.nn.l2_normalize(mu_class_tensor, axis=-1)
        gat_emb_norm = tf.nn.l2_normalize(gat_embeddings_full, axis=-1)

        if (
            self.is_procrustes_aligned
            and self.procrustes_rotation_matrix is not None
        ):
            R = tf.constant(self.procrustes_rotation_matrix, dtype=tf.float32)
            if (
                self.procrustes_use_centering
                and self.procrustes_hpl_mean is not None
            ):
                gat_mean = tf.constant(
                    self.procrustes_gat_mean, dtype=tf.float32
                )
                hpl_mean = tf.constant(
                    self.procrustes_hpl_mean, dtype=tf.float32
                )
                gat_aligned = (
                    tf.matmul(gat_embeddings_full - gat_mean, R) + hpl_mean
                )
            else:
                gat_aligned = tf.matmul(gat_embeddings_full, R)
            gat_aligned_norm = tf.nn.l2_normalize(gat_aligned, axis=-1)
            score_generalist = tf.matmul(
                z_norm, gat_aligned_norm, transpose_b=True
            )
        else:
            score_generalist = tf.matmul(
                z_norm, gat_emb_norm, transpose_b=True
            )

        score_generalist = tf.clip_by_value(score_generalist, 0.0, 1.0)

        if self.aux_classifier is not None:
            hpl_logits = self.aux_classifier(mu_class_tensor, training=False)
            score_expert_seen = tf.nn.softmax(hpl_logits, axis=-1)
            score_expert = self._expand_hpl_logits(
                score_expert_seen, score_generalist
            )
        else:
            score_expert_seen = None
            score_expert = tf.zeros_like(score_generalist)

        if self.aux_classifier is not None:
            protos_seen = tf.matmul(
                self.aux_classifier.ancestry_matrix,
                self.aux_classifier.node_parts,
            )
            score_profile = self._compute_profile_matching_scores(
                mu_class_tensor, protos_seen
            )
        else:
            protos_seen = None
            score_profile = tf.zeros_like(score_generalist)

        if self.aux_classifier is not None:
            _, expert_top_idx_seen = tf.nn.top_k(score_expert_seen, k=2)
            p1_idx = expert_top_idx_seen[:, 0]
            p2_idx = expert_top_idx_seen[:, 1]

            normalized_protos = tf.nn.l2_normalize(protos_seen, axis=1)
            p1_vec = tf.gather(normalized_protos, p1_idx)
            p2_vec = tf.gather(normalized_protos, p2_idx)

            sim_p1 = tf.reduce_sum(z_norm * p1_vec, axis=1)
            sim_p2 = tf.reduce_sum(z_norm * p2_vec, axis=1)

            expert_confidence = tf.clip_by_value(
                sim_p1 - sim_p2, 0.0, 1.0
            )
            expert_confidence = tf.expand_dims(expert_confidence, -1)
            mask_expert_safety = (
                tf.reduce_sum(
                    tf.one_hot(
                        tf.expand_dims(p1_idx, 1),
                        tf.shape(score_generalist)[1],
                    ),
                    axis=1,
                )
                > 0.5
            )
        else:
            expert_confidence = tf.zeros([batch_size, 1], dtype=tf.float32)
            mask_expert_safety = tf.zeros(
                [batch_size, tf.shape(score_generalist)[1]], dtype=tf.bool
            )

        if not use_full_ontology:
            seen_indices = self._compute_seen_indices()
            score_generalist = tf.gather(score_generalist, seen_indices, axis=1)
            score_expert = tf.gather(score_expert, seen_indices, axis=1)
            score_profile = tf.gather(score_profile, seen_indices, axis=1)
            mask_expert_safety = tf.gather(
                mask_expert_safety, seen_indices, axis=1
            )
        else:
            seen_indices = None

        w_exp = 0.8 * expert_confidence
        w_gen = 0.1 * expert_confidence + 0.6 * (1.0 - expert_confidence)
        w_prof = 0.1 * expert_confidence + 0.4 * (1.0 - expert_confidence)

        base_final_scores = (
            (w_exp * score_expert)
            + (w_gen * score_generalist)
            + (w_prof * score_profile)
        )

        # The generalist uses a relative-score tolerance; the profile head uses
        # fixed top-K membership because its score scale has different semantics.
        gen_max = tf.reduce_max(score_generalist, axis=1, keepdims=True)
        mask_gen = score_generalist >= (0.85 * gen_max)
        mask_prof_topk = _topk_mask_tf(score_profile, k=PROFILE_TOPK)

        if self.aux_classifier is not None:
            _, top_expert_seen_idx = tf.nn.top_k(score_expert_seen, k=3)
            mask_expert_neighborhood = self._get_neighborhood_mask(
                top_expert_seen_idx
            )
            if not use_full_ontology and seen_indices is not None:
                mask_expert_neighborhood = tf.gather(
                    mask_expert_neighborhood, seen_indices, axis=1
                )
        else:
            mask_expert_neighborhood = tf.ones_like(mask_gen)

        # Admit any class endorsed by a scoring head or the expert safety mask.
        mask_final = tf.logical_or(
            tf.logical_or(
                tf.logical_or(mask_gen, mask_prof_topk),
                mask_expert_neighborhood,
            ),
            mask_expert_safety,
        )
        filtered_scores = base_final_scores * tf.cast(mask_final, tf.float32)

        if cell_variance is not None:
            batch_cell_variance = np.asarray(cell_variance, dtype=np.float32)
        else:
            top1_score = tf.reduce_max(filtered_scores, axis=1).numpy()
            batch_cell_variance = 1.0 - np.clip(top1_score, 0.0, 1.0)

        top_scores, top_indices = tf.nn.top_k(filtered_scores, k=top_k)

        result = {
            "top_indices": top_indices.numpy(),
            "top_scores": top_scores.numpy().astype(np.float64),
            "cell_variance": batch_cell_variance,
        }
        if return_filtered_scores:
            result["filtered_scores"] = filtered_scores.numpy().astype(
                np.float32,
                copy=False,
            )
        else:
            result["filtered_scores"] = None

        if capture_diag:
            # Agreement-based tier label at the predicted top-1 class.
            # 3-of-3 -> "gold", 2-of-3 -> "silver", {0,1}-of-3 -> "bronze".
            # Diagnostic only; production code does not branch on tier.
            top1_idx = top_indices[:, 0]
            agrees_gen_t = tf.gather(mask_gen, top1_idx, batch_dims=1)
            agrees_prof_t = tf.gather(mask_prof_topk, top1_idx, batch_dims=1)
            agrees_exp_t = tf.gather(mask_expert_neighborhood, top1_idx, batch_dims=1)
            agree_count_t = (
                tf.cast(agrees_gen_t, tf.int32)
                + tf.cast(agrees_prof_t, tf.int32)
                + tf.cast(agrees_exp_t, tf.int32)
            )
            agree_count_np = agree_count_t.numpy()
            tier_label = np.where(
                agree_count_np == 3, "gold",
                np.where(agree_count_np == 2, "silver", "bronze"),
            )

            result["diag"] = {
                "score_generalist": score_generalist.numpy().astype(np.float32, copy=False),
                "score_expert": score_expert.numpy().astype(np.float32, copy=False),
                "score_profile": score_profile.numpy().astype(np.float32, copy=False),
                "expert_confidence": tf.squeeze(expert_confidence, axis=-1).numpy().astype(np.float32, copy=False),
                "mask_gen": mask_gen.numpy(),
                "mask_prof_topk": mask_prof_topk.numpy(),
                "mask_expert_neighborhood": mask_expert_neighborhood.numpy(),
                "mask_expert_safety": mask_expert_safety.numpy(),
                "mask_final": mask_final.numpy(),
                "agrees_gen": agrees_gen_t.numpy(),
                "agrees_prof": agrees_prof_t.numpy(),
                "agrees_exp": agrees_exp_t.numpy(),
                "agree_count": agree_count_np,
                "tier_label": tier_label,
                "base_final_scores": base_final_scores.numpy().astype(np.float32, copy=False),
                "seen_indices": (None if seen_indices is None else seen_indices.numpy()),
                "aux_classifier_present": (self.aux_classifier is not None),
            }
        return result

    def _classify_embeddings_batched(
        self,
        mu_class: np.ndarray,
        *,
        top_k: int,
        cell_variance: Optional[np.ndarray],
        use_full_ontology: Optional[bool],
        capture_filtered_scores: bool,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        np.ndarray,
        np.ndarray,
        Optional[Callable[[], None]],
    ]:
        """Run the embedding classifier in GPU-aware cell batches."""

        self._log("  Step 3: Predicting cell types...")

        if use_full_ontology is None:
            use_full_ontology = self.config.use_full_ontology

        n_cells = int(mu_class.shape[0])
        # Classification has no anchor-style fixed cost — fixed_bytes=0.
        per_cell_bytes = self._estimate_classification_bytes_per_row(
            use_full_ontology=use_full_ontology
        )
        batch_size = self._resolve_stream_batch_size(
            n_cells,
            0,
            per_cell_bytes,
            overhead_factor=2.0,
            default_batch=8192,
            silent=not self.verbose,
        )

        all_top_indices = np.empty((n_cells, top_k), dtype=np.int64)
        all_top_scores = np.empty((n_cells, top_k), dtype=np.float64)
        all_variance = np.empty(n_cells, dtype=np.float32)

        filtered_scores_array = None
        cleanup = None
        if capture_filtered_scores:
            num_active_classes = len(
                self._get_active_class_ids(use_full_ontology)
            )
            score_buffer = _ScoreMatrixBuffer(
                (n_cells, num_active_classes),
                np.float32,
            )
            filtered_scores_array = score_buffer.array

            _closed = {"done": False}

            def _cleanup_scores() -> None:
                if _closed["done"]:
                    return
                _closed["done"] = True
                score_buffer.close()

            cleanup = _cleanup_scores
            if score_buffer.is_memmap:
                weakref.finalize(filtered_scores_array, cleanup)

        batch_iterator = range(0, n_cells, batch_size)
        if self.verbose and n_cells > batch_size:
            batch_iterator = tqdm(
                batch_iterator,
                total=(n_cells + batch_size - 1) // batch_size,
                desc="  Predicting",
                unit="batch",
                ncols=80,
            )

        for start in batch_iterator:
            end = min(start + batch_size, n_cells)
            batch_variance = None
            if cell_variance is not None:
                batch_variance = np.asarray(
                    cell_variance[start:end],
                    dtype=np.float32,
                )

            batch_result = self._score_embedding_batch(
                mu_class[start:end],
                top_k=top_k,
                cell_variance=batch_variance,
                use_full_ontology=use_full_ontology,
                return_filtered_scores=capture_filtered_scores,
            )
            all_top_indices[start:end] = batch_result["top_indices"]
            all_top_scores[start:end] = batch_result["top_scores"]
            all_variance[start:end] = batch_result["cell_variance"]

            if capture_filtered_scores and filtered_scores_array is not None:
                filtered_scores_array[start:end] = batch_result["filtered_scores"]

        return (
            all_top_indices,
            all_top_scores,
            filtered_scores_array,
            mu_class,
            all_variance,
            cleanup,
        )
    
    def _get_neighborhood_mask(self, top_seen_indices: tf.Tensor) -> tf.Tensor:
        """
        Compute the one-hop neighborhood (parents, children, and self) of the
        given seen-class indices.
        
        Args:
            top_seen_indices: [batch, k] indices of top expert predictions (seen classes)
            
        Returns:
            neighborhood_mask: [batch, num_all_classes] boolean mask
        """
        # Map Seen Indices -> Full Indices
        seen_to_full_map = self._compute_seen_indices() # [N_seen] mapping to N_all
        
        # Flatten batch and k dimensions for lookup
        flat_seen_indices = tf.reshape(top_seen_indices, [-1])
        flat_full_indices = tf.gather(seen_to_full_map, flat_seen_indices)
        
        # Get shape info
        batch_size = tf.shape(top_seen_indices)[0]
        k = tf.shape(top_seen_indices)[1]
        num_all_classes = tf.shape(self.prior_matrix_full)[0] # Infer from PPR matrix shape
        
        # Create one-hot vectors for the seed nodes [batch*k, N_all]
        step1_seeds = tf.one_hot(flat_full_indices, num_all_classes)
        
        # Retrieve Full Adjacency Matrix (A)
        # Cache the dense full-ontology adjacency used for one-hop propagation.
        if not hasattr(self, 'adj_full_tensor'):
             self.adj_full_tensor = tf.constant(self.celltype_gat.full_ontology_adj_np, dtype=tf.float32)
        
        A = self.adj_full_tensor
        # Make symmetric for "Neighborhood" (Parents AND Children)
        # A_sym = A + A.T + I
        A_sym = tf.maximum(A, tf.transpose(A)) + tf.eye(num_all_classes)
        
        # Propagate: Mask = Seeds @ A_sym
        # [batch*k, N_all] @ [N_all, N_all] -> [batch*k, N_all]
        # This finds all 1-hop neighbors
        flattened_mask = tf.matmul(step1_seeds, A_sym)
        
        # Reshape back to [batch, k, N_all] and reduce (Union of k neighborhoods)
        reshaped_mask = tf.reshape(flattened_mask, [batch_size, k, num_all_classes])
        neighborhood_mask = tf.reduce_max(reshaped_mask, axis=1) > 0.001
        
        return neighborhood_mask

    def _expand_hpl_logits(
        self,
        hpl_logits: tf.Tensor,
        gat_scores: tf.Tensor
    ) -> tf.Tensor:
        """
        Expand HPL logits from seen classes to full ontology.
        
        Args:
            hpl_logits: [batch_size, num_seen_classes] (can be logits or probabilities)
            gat_scores: [batch_size, num_total_classes] (used for shape reference)
        
        Returns:
            hpl_logits_expanded: [batch_size, num_total_classes]
        """
        num_cells = tf.shape(hpl_logits)[0]
        num_seen_classes = tf.shape(hpl_logits)[1]
        num_total_classes = tf.shape(gat_scores)[1]
        
        # Create expanded logits with very small values for unseen classes
        # Inputs are probabilities, so unseen entries receive a small positive floor.
        expanded_logits = tf.ones([num_cells, num_total_classes], dtype=tf.float32) * 1e-10
        
        # Get indices of seen classes  
        seen_indices = self._compute_seen_indices()
        
        # Scatter HPL logits into seen class positions
        batch_indices = tf.repeat(tf.range(num_cells), num_seen_classes)
        class_indices = tf.tile(seen_indices, [num_cells])
        indices = tf.stack([batch_indices, class_indices], axis=1)
        
        updates = tf.reshape(hpl_logits, [-1])
        
        hpl_logits_expanded = tf.tensor_scatter_nd_update(
            expanded_logits,
            indices,
            updates
        )
        
        return hpl_logits_expanded
    
    def _get_active_class_ids(
        self, use_full_ontology: Optional[bool] = None
    ) -> List[str]:
        """Return ontology IDs for the predictor's active output universe."""
        if use_full_ontology is None:
            use_full_ontology = self.config.use_full_ontology

        if use_full_ontology:
            return list(self.celltype_gat.full_classes)
        return list(self.celltype_gat.classes)

    def _indices_to_names(
        self,
        indices: np.ndarray,
        use_full_ontology: Optional[bool] = None,
    ) -> List[List[str]]:
        """
        Convert class indices to cell type names.
        
        Args:
            indices: [n_cells, top_k] predicted class indices
        
        Returns:
            names: List of lists of cell type names
        """
        classes = self._get_active_class_ids(use_full_ontology)
        
        names = []
        for cell_indices in indices:
            cell_names = [classes[idx] for idx in cell_indices]
            names.append(cell_names)
        
        return names
    
    def _format_prediction(self, ontology_id: str, label_format: str) -> str: 
        """
        Format a prediction based on label_format preference.
        
        Converts ontology IDs to human-readable formats based on the requested
        output format. Handles missing names gracefully by falling back to IDs.
        
        Args:
            ontology_id: Ontology ID (e.g., 'CL:0000128')
            label_format: Format preference - 'id', 'name', or 'both'
                - 'id': Returns only the ontology ID
                - 'name': Returns only the cell type name (falls back to ID if name unavailable)
                - 'both': Returns "name (id)" format (falls back to ID if name unavailable)
        
        Returns:
            Formatted prediction string according to label_format
            
        Examples:
            >>> predictor._format_prediction('CL:0000128', 'id')
            'CL:0000128'
            
            >>> predictor._format_prediction('CL:0000128', 'name')
            'monocyte'
            
            >>> predictor._format_prediction('CL:0000128', 'both')
            'monocyte (CL:0000128)'
            
            >>> # Fallback when name is not available
            >>> predictor._format_prediction('CL:9999999', 'name')
            'CL:9999999'
            
            >>> predictor._format_prediction('CL:9999999', 'both')
            'CL:9999999'
        """
        if label_format == 'id':
            return ontology_id
        
        # Get name from mapping (empty string if not found)
        name = self.id_to_name_map.get(ontology_id, '')
        
        if label_format == 'name':
            # Return name if available, otherwise fallback to ID
            return name if name else ontology_id
        
        # label_format == 'both'
        if name:
            return f"{name} ({ontology_id})"
        else:
            # Fallback to ID only when name not available
            return ontology_id

    def _compute_low_quality_mask(
        self,
        cell_variance_array: np.ndarray,
        settings: _PredictionSettings,
    ) -> Optional[np.ndarray]:
        """Compute the variance-based low-quality mask once per run."""

        if not settings.filter_low_quality:
            return None

        if settings.variance_threshold is not None:
            effective_var_threshold = settings.variance_threshold
        else:
            effective_var_threshold = self._get_variance_threshold(
                cell_variance_array
            )

        low_quality_mask = cell_variance_array > effective_var_threshold
        num_low_quality = int(np.sum(low_quality_mask))
        self._log(
            "  Variance filter: "
            f"{num_low_quality}/{len(cell_variance_array)} cells marked low-quality "
            f"(threshold={effective_var_threshold:.4f})"
        )
        return low_quality_mask

    def _extract_top_predictions(
        self,
        final_scores: Union[np.ndarray, tf.Tensor],
        top_k: int,
        log_space: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract top-k indices and user-facing scores from score tensors."""

        scores_array = np.asarray(final_scores, dtype=np.float32)
        candidate_indices = np.argpartition(scores_array, -top_k, axis=1)[
            :, -top_k:
        ]
        candidate_scores = np.take_along_axis(
            scores_array,
            candidate_indices,
            axis=1,
        )
        order = np.argsort(-candidate_scores, axis=1)
        top_indices = np.take_along_axis(candidate_indices, order, axis=1)
        top_scores_np = np.take_along_axis(candidate_scores, order, axis=1)
        if log_space:
            top_scores_np = np.exp(top_scores_np)
        return top_indices.astype(np.int64), top_scores_np.astype(np.float64)

    def _compute_score_entropy(
        self,
        filtered_scores: Union[np.ndarray, tf.Tensor],
    ) -> np.ndarray:
        """Compute per-cell entropy from positive filtered scores in class chunks."""

        scores_array = np.asarray(filtered_scores, dtype=np.float32)
        n_cells, n_classes = scores_array.shape
        row_sums = np.sum(scores_array, axis=1, dtype=np.float64) + (
            1e-15 * n_classes
        )
        entropy = np.zeros(n_cells, dtype=np.float64)
        class_chunk_size = _predictor_support._get_cpu_class_chunk_size(
            n_cells,
            n_classes,
            working_buffers=2,
        )
        for start in range(0, n_classes, class_chunk_size):
            end = min(start + class_chunk_size, n_classes)
            block = np.asarray(scores_array[:, start:end], dtype=np.float32)
            probs = (block + 1e-15) / row_sums[:, None]
            entropy -= np.sum(
                probs * np.log(probs + 1e-9),
                axis=1,
                dtype=np.float64,
            )
        return entropy

    def _apply_grit_refinement_to_scores(
        self,
        filtered_scores: Union[np.ndarray, tf.Tensor],
        X_normalized,
        low_quality_mask: Optional[np.ndarray],
        settings: _PredictionSettings,
        preserve_input: bool = False,
    ) -> Tuple[np.ndarray, Optional[Callable[[], None]]]:
        """Apply GRIT refinement to already-classified scores."""

        filtered_scores_array = np.asarray(filtered_scores, dtype=np.float32)
        n_cells = int(filtered_scores_array.shape[0])

        self._log("  [INFO] Applying GRIT refinement...")

        entropy = self._compute_score_entropy(filtered_scores_array)

        if self.verbose:
            self._log("\n  " + "=" * 60)
            self._log("  GRIT REFINEMENT")
            self._log("  " + "=" * 60)
            stats_entropy = np.percentile(entropy, [0, 5, 25, 50, 75, 95, 100])
            self._log(
                f"  {'Metric':<15} {'Min':<8} {'5%':<8} {'25%':<8} "
                f"{'Median':<8} {'75%':<8} {'95%':<8} {'Max':<8}"
            )
            self._log("  " + "-" * 75)
            self._log(
                f"  {'Entropy':<15} {stats_entropy[0]:<8.2f} "
                f"{stats_entropy[1]:<8.2f} {stats_entropy[2]:<8.2f} "
                f"{stats_entropy[3]:<8.2f} {stats_entropy[4]:<8.2f} "
                f"{stats_entropy[5]:<8.2f} {stats_entropy[6]:<8.2f}"
            )
            self._log("  " + "-" * 75)

        effective_threshold = settings.grit_entropy_threshold

        if effective_threshold is None:
            if settings.grit_refinement_percentile is not None:
                target_percentile = 100.0 - settings.grit_refinement_percentile
                effective_threshold = float(
                    np.percentile(entropy, target_percentile)
                )
                pct_above = 100.0 * np.mean(entropy > effective_threshold)
                if self.verbose:
                    self._log("\n  [Manual Percentile Threshold]")
                    self._log(
                        f"    Target:Refining the lower {pct_above:.1f}% of the prediction distribution")
                    self._log(
                        f"    Calculated Threshold: {effective_threshold:.4f}"
                    )
            else:
                effective_threshold = self._get_adaptive_entropy_threshold(
                    entropy
                )
                pct_above = 100.0 * np.mean(entropy > effective_threshold)
                if self.verbose:
                    self._log(f"\n  [Auto-Thresholding]: Detected Threshold: {effective_threshold:.4f}. Refining the lower {pct_above:.1f}% of the prediction distribution")
        elif self.verbose:
            self._log("\n  [Manual Threshold]")
            self._log(
                f"    Using fixed entropy threshold: {effective_threshold}"
            )

        if self.verbose:
            self._log("  " + "=" * 60 + "\n")

        is_uncertain = entropy > effective_threshold
        num_uncertain = int(np.sum(is_uncertain))
        
        if num_uncertain == 0:
            return filtered_scores_array, None

        self._log(
            f"  Step 4: Refining {num_uncertain:,} cells using all {n_cells:,} "
            "cells as neighbors ..."
        )
        uncertain_indices = np.where(is_uncertain)[0]

        # ---- Backend dispatch for kNN + adjacency + GRIT propagation ----
        # Use MLX GRIT when the MLX celltype embedder is loaded, or when TF
        # is unavailable (macOS Apple Silicon with conditional TF import).
        _use_mlx_grit = self._mlx_celltype_embedder is not None or tf is None

        if _use_mlx_grit:
            from .backend_mlx import (
                build_knn_graph_cells_mlx,
                build_normalized_adjacency_scipy,
                apply_grit_refinement_scipy,
            )

            try:
                # Pass the sparse matrix straight through -- build_knn_graph_cells_mlx
                # streams tiles from CSR and never materializes the full dense matrix
                # (host or device), so no .toarray() here.
                edge_index_np, edge_weights_np = build_knn_graph_cells_mlx(
                    X_normalized,
                    k=self.config.k_neighbors,
                    query_indices=uncertain_indices,
                )
            except (MemoryError, RuntimeError) as _grit_oom:
                # GRIT is a Step-4 refinement that runs after predictions already
                # exist; on host/GPU OOM (MLX surfaces it as RuntimeError), skip
                # it and return the unrefined scores -- the same no-op contract as
                # the num_uncertain == 0 path above -- rather than crashing a
                # near-complete run.
                self._log(
                    f"[WARN] GRIT skipped: insufficient memory for k-NN on "
                    f"{num_uncertain:,} cells ({_grit_oom}); returning unrefined "
                    f"predictions."
                )
                return filtered_scores_array, None
        else:
            _has_gpu = tf is not None and bool(
                tf.config.list_physical_devices("GPU")
            )
            if _has_gpu:
                # GPU hardware path: reordered sparse-stream + GPU-densify k-NN.
                # OOM guard: if the VRAM estimate was optimistic, halve the
                # tiles and retry rather than crashing.
                ct = qt = 8192
                while True:
                    try:
                        edge_index_np, edge_weights_np = _grit_knn_gpu_reordered(
                            X_normalized,
                            uncertain_indices,
                            self.config.k_neighbors,
                            self._grit_knn_available_bytes,
                            cand_tile=ct,
                            query_tile=qt,
                        )
                        break
                    except tf.errors.ResourceExhaustedError:
                        if ct <= 512 and qt <= 512:
                            raise
                        ct = max(512, ct // 2)
                        qt = max(512, qt // 2)
                        self._log(
                            f"  GRIT: VRAM tight; retrying k-NN with tiles "
                            f"cand={ct}, query={qt}"
                        )
            else:
                # CPU hardware path (no GPU, no MLX): exact cosine k-NN.
                # Separate first-class path, not a fallback. Returns numpy.
                self._log(
                    "  GRIT: no GPU/MLX detected; using exact CPU cosine k-NN"
                )
                edge_index_np, edge_weights_np = _grit_knn_cosine_cpu(
                    X_normalized,
                    uncertain_indices,
                    self.config.k_neighbors,
                )

        if (
            settings.filter_low_quality
            and low_quality_mask is not None
            and np.any(low_quality_mask)
        ):
            lq_set = set(np.where(low_quality_mask)[0].tolist())
            rows, cols = edge_index_np[0], edge_index_np[1]
            keep = np.array(
                [r not in lq_set and c not in lq_set for r, c in zip(rows, cols)]
            )
            edge_index_np = edge_index_np[:, keep]
            edge_weights_np = edge_weights_np[keep]
            self._log(
                f"  GRIT: Excluded {len(lq_set)} low-quality cells from propagation graph"
            )

        if _use_mlx_grit:
            adj_normalized = build_normalized_adjacency_scipy(
                edge_index_np, edge_weights_np, n_cells
            )
        else:
            adj_normalized = build_normalized_adjacency(
                edge_index_np, edge_weights_np, n_cells
            )

        if _use_mlx_grit:
            refined_probs = apply_grit_refinement_scipy(
                filtered_scores_array,
                adj_normalized,
                num_iterations=settings.grit_iterations,
                alpha=settings.grit_alpha,
                score_mode="positive_scores",
            )
            refined_cleanup = None
        else:
            refined_probs, refined_cleanup = apply_grit_refinement(
                filtered_scores_array,
                adj_normalized,
                num_iterations=settings.grit_iterations,
                alpha=settings.grit_alpha,
                score_mode="positive_scores",
            )

        final_scores_cleanup = refined_cleanup
        if preserve_input:
            final_buffer = _ScoreMatrixBuffer(
                filtered_scores_array.shape,
                np.float32,
                prefer_memmap=isinstance(filtered_scores_array, np.memmap),
            )
            final_scores = final_buffer.array
            final_scores[:] = filtered_scores_array

            def _cleanup() -> None:
                if refined_cleanup is not None:
                    refined_cleanup()
                final_buffer.close()

            final_scores_cleanup = _cleanup
        else:
            final_scores = filtered_scores_array

        final_scores[is_uncertain] = refined_probs[is_uncertain]
        if preserve_input and refined_cleanup is not None:
            refined_cleanup()
        return final_scores, final_scores_cleanup

    def _build_visualization_state(
        self,
        use_full_ontology: bool,
    ) -> Dict[str, Any]:
        """Build ontology vectors and graph metadata for visualization."""

        gat_embeddings_full = self.celltype_gat.get_gat_prototypes(
            for_full_ontology=use_full_ontology,
            training=False,
        )
        # On the MLX path get_gat_prototypes already returns numpy; on TF
        # it returns a tf.Tensor.  Normalise to numpy for downstream math.
        _gat_np = (
            gat_embeddings_full
            if isinstance(gat_embeddings_full, np.ndarray)
            else gat_embeddings_full.numpy()
        )

        if (
            self.is_procrustes_aligned
            and self.procrustes_rotation_matrix is not None
        ):
            node_vectors = _gat_np
            R = self.procrustes_rotation_matrix
            if self.procrustes_use_centering:
                node_vectors = (
                    (node_vectors - self.procrustes_gat_mean) @ R
                    + self.procrustes_hpl_mean
                )
            else:
                node_vectors = node_vectors @ R
        else:
            node_vectors = _gat_np

        node_names = self._get_active_class_ids(use_full_ontology)

        if use_full_ontology:
            adjacency_matrix = self.celltype_gat.full_ontology_adj_np.copy()
            pure_adjacency_matrix = self.full_pure_ontology_adj
        else:
            adjacency_matrix = self.celltype_gat.ontology_adj_np.copy()
            pure_adjacency_matrix = self.pure_ontology_adj

        co_graph = {}
        rows, cols = np.where(adjacency_matrix > 0)
        for child_idx, parent_idx in zip(rows, cols):
            if child_idx == parent_idx:
                continue
            child_name = node_names[child_idx]
            parent_name = node_names[parent_idx]
            if child_name not in co_graph:
                co_graph[child_name] = []
            co_graph[child_name].append(parent_name)

        if use_full_ontology:
            level_array = getattr(
                self.celltype_gat, "full_level_array_np", None
            )
        else:
            level_array = getattr(
                self.celltype_gat, "level_array_np", None
            )

        return {
            "node_vectors": node_vectors,
            "node_names": node_names,
            "adjacency_matrix": adjacency_matrix,
            "co_graph": co_graph,
            "pure_adjacency_matrix": pure_adjacency_matrix,
            "level_array": level_array,
            "gat_embeddings": _gat_np,
            "ontology_adj": adjacency_matrix,
        }

    def _run_prediction_from_embeddings(
        self,
        adata: anndata.AnnData,
        cell_vectors: np.ndarray,
        cell_variance_array: np.ndarray,
        settings: _PredictionSettings,
        prepared_expression=None,
        export_score_matrix: bool = False,
        capture_pre_grit_score_matrix: bool = False,
        force_recompute: bool = False,
    ) -> _PredictionCoreResult:
        """Run the shared prediction stage from cached embeddings."""

        low_quality_mask = self._compute_low_quality_mask(
            cell_variance_array, settings
        )

        X_normalized = prepared_expression

        def _ensure_prepared() -> np.ndarray:
            nonlocal X_normalized
            if X_normalized is None:
                X_normalized = self._prepare_expression(adata)
            return X_normalized

        need_full_scores = bool(
            settings.use_grit
            or export_score_matrix
            or capture_pre_grit_score_matrix
        )
        if settings.use_grit:
            _ensure_prepared()
        (
            top_indices,
            top_scores,
            filtered_scores,
            _,
            _,
            classification_cleanup,
        ) = self._classify_embeddings_batched(
            cell_vectors,
            top_k=settings.top_k,
            cell_variance=cell_variance_array,
            use_full_ontology=settings.use_full_ontology,
            capture_filtered_scores=need_full_scores,
        )

        transient_cleanups: List[Callable[[], None]] = []
        if classification_cleanup is not None:
            transient_cleanups.append(classification_cleanup)

        pre_grit_score_matrix_array = None
        if capture_pre_grit_score_matrix:
            if filtered_scores is None:
                raise RuntimeError(
                    "pre-GRIT score capture requires staged filtered scores."
                )
            pre_grit_score_matrix_array = filtered_scores

        final_scores = filtered_scores
        if settings.use_grit:
            if filtered_scores is None:
                raise RuntimeError(
                    "GRIT refinement requires staged filtered scores."
                )
            final_scores, grit_cleanup = self._apply_grit_refinement_to_scores(
                filtered_scores,
                _ensure_prepared(),
                low_quality_mask,
                settings,
                preserve_input=capture_pre_grit_score_matrix,
            )
            if grit_cleanup is not None:
                transient_cleanups.append(grit_cleanup)
        else:
            self._log("  Step 4: GRIT refinement skipped (use_grit=False).")
        if final_scores is not None:
            top_indices, top_scores = self._extract_top_predictions(
                final_scores,
                settings.top_k,
                log_space=False,
            )
        top_ids = np.asarray(
            self._indices_to_names(
                top_indices,
                use_full_ontology=settings.use_full_ontology,
            ),
            dtype=object,
        )
        prediction_ids = top_ids[:, 0]
        prediction_scores = top_scores[:, 0]

        score_matrix_array = None
        if export_score_matrix:
            if final_scores is None:
                raise RuntimeError("Final score matrix was not captured during prediction.")
            score_matrix_array = final_scores

        cleanup_once = None
        if transient_cleanups:
            cleanup_state = {"done": False}

            def _cleanup_transient_arrays() -> None:
                if cleanup_state["done"]:
                    return
                cleanup_state["done"] = True
                for callback in reversed(transient_cleanups):
                    callback()

            cleanup_once = _cleanup_transient_arrays
            for array in (score_matrix_array, pre_grit_score_matrix_array):
                if array is not None:
                    weakref.finalize(array, cleanup_once)

        return _PredictionCoreResult(
            top_indices=top_indices,
            top_ids=top_ids,
            top_scores=top_scores,
            prediction_ids=prediction_ids,
            prediction_scores=prediction_scores,
            cell_variance_array=cell_variance_array,
            low_quality_mask=low_quality_mask,
            score_matrix_array=score_matrix_array,
            pre_grit_score_matrix_array=pre_grit_score_matrix_array,
            transient_cleanup=cleanup_once,
        )
    
    def _auto_detect_gene_ids(self, adata, target_column: str) -> None:
        """
        Automatically detect Ensembl gene IDs in adata.var and set them to the target column.
        
        Scans:
        1. adata.var.index
        2. adata.var columns
        
        If a match is found (format: ENSG + digits), it populates adata.var[target_column].
        If target_column already exists, it verifies it.

        Any trailing Ensembl version suffix (e.g. ``ENSMUSG00000051951.5``) is
        trimmed when the column is written so versioned IDs still match the
        model's unversioned required gene IDs.
        """
        # Regex for Ensembl ID (human: ENSG00000243485, mouse: ENSMUSG00000000001)
        # We look for the pattern anywhere in the string, but typically it should be the whole string
        ensembl_pattern = re.compile(r'ENS(?:MUS)?G\d+')

        def strip_version(series):
            """Trim a trailing Ensembl version suffix (e.g. ``.5``) from each ID.

            GTF/GENCODE-derived AnnData frequently carry versioned IDs such as
            ``ENSMUSG00000051951.5``. The model's required gene IDs are
            unversioned, so the suffix must be removed before matching —
            otherwise every gene misses the lookup and is filled with zeros.
            Only a literal ``.<digits>`` at the end of the string is removed,
            so non-Ensembl values (gene symbols, blanks) are left untouched.
            Returns a positional numpy array suitable for direct column
            assignment regardless of the source's index.
            """
            return series.astype(str).str.replace(r'\.\d+$', '', regex=True).to_numpy()

        def check_series(series):
            """Check whether a series contains Ensembl IDs.

            Treats empty/whitespace-only strings as missing in addition to NaN,
            and scans the entire column rather than only the first 10 entries
            so a leading block of unmapped genes does not cause a false
            negative.
            """
            sample = series.dropna().astype(str).str.strip()
            sample = sample[sample != '']
            if len(sample) == 0:
                return False
            matches = sample.str.contains(ensembl_pattern, regex=True).sum()
            return matches > (len(sample) * 0.5)

        # 1. Check if target column exists (User might have already set it)
        if target_column in adata.var.columns:
            if check_series(adata.var[target_column]):
                # Normalise in place so any version suffixes are trimmed even
                # when the user pre-populated the column.
                adata.var[target_column] = strip_version(adata.var[target_column])
                return
            else:
                 # It exists but doesn't look like Ensembl IDs.
                 # If user explicitly provided "feature_id", and it exists, maybe they mean it.
                 # But if they passed default "gene_ids" and it doesn't exist...
                 # Let's proceed to search. If provided column doesn't match, we assume it's not the right one and search others.
                 self._log(f"  ℹ️  The'{target_column}' column is present, but its values do not appear to be valid Ensembl IDs. Checking other columns...")

        # 2. Check Index
        if check_series(pd.Series(adata.var.index)):
            adata.var[target_column] = strip_version(pd.Series(adata.var.index))
            return

        # 3. Check all other columns
        for col in adata.var.columns:
            if col == target_column: continue

            if check_series(adata.var[col]):
                adata.var[target_column] = strip_version(adata.var[col])
                return

        # 4. If we get here, no IDs were found
        raise ValueError(
            f"❌ Could not find Ensembl gene IDs (e.g., ENSG00000... for human, ENSMUSG00000... for mouse) in adata.var index or columns.\n"
            f"⚠️ Please ensure your data contains Ensembl IDs to use the model for prediction."
        )

    def _embeddings_cache_valid(self, adata) -> bool:
        """True iff cached embeddings in ``adata`` are usable for this model.

        The gate used by :meth:`compute_cell_embeddings` (and the preflight
        memory guard) to decide whether the encoder can be skipped: ``X_hector``
        and ``hector_cell_variance`` present with matching row counts, AND the
        stored embedding manifest matching this model, checkpoint, expression
        source, and cell order.
        """
        if not self._check_cache_valid(adata, 'X_hector', location='obsm'):
            return False
        if not self._check_cache_valid(adata, 'hector_cell_variance', location='obs'):
            return False
        embedding_manifest = self._get_embedding_manifest(adata)
        source_slot = None
        if embedding_manifest is not None:
            try:
                source_slot = self._resolve_expression_source(adata)["name"]
            except ValueError:
                source_slot = None
        return bool(
            self._embedding_manifest_matches(
                adata, embedding_manifest, source_slot=source_slot
            )
        )

    def compute_cell_embeddings(
        self,
        adata: anndata.AnnData,
        force_recompute: bool = False,
        _return_prepared_expression: bool = False,
    ) -> Optional[np.ndarray]:
        """Compute and store cell embeddings in adata.obsm['X_hector'].

        This is the first step in the HECTOR pipeline. It runs the VGAE encoder
        to produce latent representations of cells.

        Args:
            adata: AnnData object with gene expression data.
            force_recompute: If False and embeddings already exist, skip computation.

        Returns:
            None for normal public usage. Internal callers may request the
            freshly prepared normalized expression matrix so it can be reused
            within the same pipeline call. Results are always stored in:
            - adata.obsm['X_hector']: Cell embeddings (n_cells, latent_dim)
            - adata.obs['hector_cell_variance']: Per-cell variance from VGAE

        Note:
            If *adata* is a view it will be materialised in-place (via
            ``_init_as_actual``) so that the caller's reference receives the
            cached embeddings. This avoids the ``ImplicitModificationWarning``
            and the expensive implicit copy that AnnData otherwise triggers.
        """
        if not force_recompute and self._embeddings_cache_valid(adata):
            return None

        # Materialize view in-place so subsequent writes to .obsm/.obs
        # land on a real AnnData that the caller can see, and avoid the
        # slow implicit copy triggered by writing to a view.
        self._ensure_cache_writable(adata)

        source_slot = self._resolve_expression_source(adata)["name"]
        X_normalized = self._prepare_expression(adata)
        mu_class, cell_variance = self._run_encoder_batched(X_normalized)
        adata.obsm['X_hector'] = mu_class
        # Cache scalar per-cell variance so predict() can use true VGAE
        # uncertainty without re-running the encoder.
        adata.obs['hector_cell_variance'] = cell_variance
        self._write_embedding_manifest(
            adata,
            self._build_embedding_manifest(adata, source_slot=source_slot),
        )
        if _return_prepared_expression:
            return X_normalized
        return None

    def _run_mlx_tsne(self, adata, *, perplexity, learning_rate, random_state,
                      n_pcs):
        """Deterministic MLX t-SNE on adata.obsm['X_hector'] -> obsm['X_tsne']."""
        from .backend_mlx import MlxTSNE

        print("  Computing t-SNE coordinates using the MLX (Metal GPU) engine ...")
        X = np.asarray(adata.obsm['X_hector'], dtype=np.float32)
        coords = MlxTSNE(
            perplexity=perplexity, n_pcs=(n_pcs or 50),
            learning_rate=learning_rate,
            random_state=int(random_state) if random_state is not None else 0,
            verbose=self.verbose,
        ).fit_transform(X)
        adata.obsm['X_tsne'] = np.asarray(coords, dtype=np.float32)

    def _run_cuml_tsne(self, adata, *, perplexity, learning_rate, random_state,
                       n_pcs):
        """cuML GPU t-SNE (NVIDIA) on adata.obsm['X_hector'] -> obsm['X_tsne'].
        Import-guarded; only reached when cuML is importable (not on this Mac)."""
        from cuml.manifold import TSNE as cuTSNE

        X = np.asarray(adata.obsm['X_hector'], dtype=np.float32)
        if n_pcs and X.shape[1] > n_pcs:
            from sklearn.decomposition import PCA
            X = PCA(n_components=n_pcs, svd_solver='randomized',
                    random_state=random_state).fit_transform(X).astype(np.float32)
        lr = 200.0 if learning_rate == 'auto' else learning_rate
        coords = cuTSNE(n_components=2, perplexity=perplexity, learning_rate=lr,
                        random_state=random_state).fit_transform(X)
        adata.obsm['X_tsne'] = np.asarray(coords, dtype=np.float32)

    def _run_scanpy_tsne(self, adata, scanpy, *, perplexity, learning_rate,
                         random_state, use_rep='X_hector'):
        """sklearn t-SNE -> obsm['X_tsne'] (CPU fallback).

        Calls scikit-learn directly with verbose progress output rather than
        routing through ``scanpy.tl.tsne``.
        """
        from sklearn.manifold import TSNE

        X = np.asarray(adata.obsm[use_rep], dtype=np.float32)
        coords = TSNE(
            n_components=2, perplexity=perplexity, learning_rate=learning_rate,
            random_state=random_state, init='pca', verbose=2,
        ).fit_transform(X)
        adata.obsm['X_tsne'] = np.asarray(coords, dtype=np.float32)

    def _project_tsne(self, adata, scanpy, *, perplexity, learning_rate,
                      random_state, n_pcs, backend):
        """t-SNE dispatch for project(method='tsne'): MLX -> cuML -> scanpy
        ladder (auto by hardware when backend is None) with atomic fallback to
        scanpy. Writes adata.obsm['X_tsne']."""
        if backend is None or backend not in ('mlx', 'cuml', 'scanpy'):
            backend = _select_tsne_backend()
        kw = dict(perplexity=perplexity, learning_rate=learning_rate,
                  random_state=random_state, n_pcs=n_pcs)
        if backend == 'mlx':
            try:
                self._run_mlx_tsne(adata, **kw)
                return
            except Exception as exc:
                print(f"  MLX t-SNE failed ({exc}); falling back to scanpy.")
                backend = 'scanpy'
        if backend == 'cuml':
            try:
                self._run_cuml_tsne(adata, **kw)
                return
            except Exception as exc:
                print(f"  cuML t-SNE failed ({exc}); falling back to scanpy.")
                backend = 'scanpy'
        self._run_scanpy_tsne(adata, scanpy, perplexity=perplexity,
                              learning_rate=learning_rate,
                              random_state=random_state)

    def _run_mlx_projection(
        self,
        adata: anndata.AnnData,
        *,
        n_neighbors: int,
        n_pcs: Optional[int],
        min_dist: float,
        spread: float,
        n_epochs: int,
        negative_sample_rate: int,
        random_state: int,
        metric: str = 'cosine',
    ) -> None:
        """Run the in-house deterministic MLX UMAP on ``adata.obsm['X_hector']``.

        The engine PCA-reduces internally (so no transient ``X_hector_pca`` rep)
        and writes only ``adata.obsm['X_umap']``. Apple Silicon only.
        """
        from .backend_mlx import MlxUMAP

        print("  Computing UMAP coordinates using the MLX (Metal GPU) engine ...")
        X = np.asarray(adata.obsm['X_hector'], dtype=np.float32)
        coords = MlxUMAP(
            n_neighbors=n_neighbors,
            n_pcs=(n_pcs or 50),
            min_dist=min_dist,
            spread=spread,
            n_epochs=n_epochs,
            negative_sample_rate=negative_sample_rate,
            random_state=int(random_state) if random_state is not None else 42,
            metric=metric,
            verbose=self.verbose,
        ).fit_transform(X)
        adata.obsm['X_umap'] = np.asarray(coords, dtype=np.float32)

    def _run_scanpy_projection(
        self,
        adata: anndata.AnnData,
        scanpy,
        *,
        n_neighbors: int,
        random_state: int,
        min_dist: float,
        spread: float,
        n_epochs: Optional[int] = None,
        init: str = 'spectral',
        negative_sample_rate: int = 5,
        use_rep: str = 'X_hector',
        metric: str = 'cosine',
    ) -> None:
        """Run the Scanpy CPU projection path on ``adata.obsm[use_rep]``.

        Forwards UMAP-layout knobs to ``scanpy.tl.umap`` so the CPU fallback
        and the cuML GPU path share a single parameter surface in ``reduce_dimensions()``.

        Args:
            adata: AnnData with ``obsm[use_rep]``.
            scanpy: Imported scanpy module (passed in to avoid reimporting).
            n_neighbors: Neighborhood size for ``sc.pp.neighbors``.
            random_state: Seed used by ``sc.pp.neighbors`` and ``sc.tl.umap``.
            min_dist: Effective minimum embedded distance.
            spread: Effective scale of embedded points.
            n_epochs: Maximum optimization epochs (mapped to scanpy's
                ``maxiter`` argument).  ``None`` lets scanpy pick its own
                size-dependent default.
            init: Initialization for the embedding (mapped to scanpy's
                ``init_pos``).  Accepts ``'spectral'``, ``'random'``,
                ``'paga'``, or an explicit ``ndarray``.
            negative_sample_rate: Number of negative samples per positive
                edge in the SGD optimizer.
            use_rep: ``obsm`` key used to build the neighborhood graph.
                ``reduce_dimensions()`` passes the PCA-reduced rep here when
                ``n_pcs`` pre-reduction is active, otherwise ``'X_hector'``.
        """
        print("  Building the cell-cell neighborhood graph using Scanpy...")
        scanpy.pp.neighbors(
            adata,
            use_rep=use_rep,
            n_neighbors=n_neighbors,
            random_state=random_state,
            metric=metric,
        )
        print("  Computing UMAP coordinates using Scanpy ...")
        # Bump verbosity so umap-learn prints its per-epoch tqdm bar (which the
        # Desktop app captures for progress); restore it afterwards.
        _prev_verbosity = scanpy.settings.verbosity
        scanpy.settings.verbosity = max(_prev_verbosity, 4)
        try:
            scanpy.tl.umap(
                adata,
                random_state=random_state,
                min_dist=min_dist,
                spread=spread,
                maxiter=n_epochs,
                init_pos=init,
                negative_sample_rate=negative_sample_rate,
            )
        finally:
            scanpy.settings.verbosity = _prev_verbosity

        # reduce_dimensions() guarantees ONLY the 2-D embedding as output.
        # scanpy.pp.neighbors leaves a kNN graph behind; drop it so the CPU
        # path matches the MLX/cuML backends (which never persist a graph) and
        # no downstream tool silently inherits a backend-specific graph. Users
        # who need a neighbor graph build it explicitly with sc.pp.neighbors.
        for _k in ("distances", "connectivities"):
            if _k in adata.obsp:
                del adata.obsp[_k]
        if "neighbors" in adata.uns:
            del adata.uns["neighbors"]

    def _prepare_project_knn_results(
        self,
        neighbor_indices: np.ndarray,
        neighbor_distances: np.ndarray,
        *,
        n_obs: int,
        target_neighbors: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Drop self-neighbors and keep a fixed number of non-self neighbors per cell."""
        target_neighbors = max(
            0,
            min(int(target_neighbors), max(int(n_obs) - 1, 0)),
        )
        neighbor_indices = np.asarray(neighbor_indices)
        neighbor_distances = np.asarray(neighbor_distances)

        if neighbor_indices.shape != neighbor_distances.shape:
            raise ValueError(
                "cuML returned mismatched kNN shapes: "
                f"indices={neighbor_indices.shape}, "
                f"distances={neighbor_distances.shape}."
            )

        trimmed_indices = np.empty((n_obs, target_neighbors), dtype=np.int64)
        trimmed_distances = np.empty((n_obs, target_neighbors), dtype=np.float32)

        for cell_idx in range(n_obs):
            row_indices = np.asarray(neighbor_indices[cell_idx], dtype=np.int64)
            row_distances = np.asarray(
                neighbor_distances[cell_idx],
                dtype=np.float32,
            )

            keep_mask = row_indices != cell_idx
            row_indices = row_indices[keep_mask]
            row_distances = row_distances[keep_mask]

            if row_indices.shape[0] < target_neighbors:
                raise ValueError(
                    "cuML returned too few non-self neighbors for "
                    f"cell {cell_idx}: expected at least {target_neighbors}, "
                    f"got {row_indices.shape[0]}."
                )

            trimmed_indices[cell_idx] = row_indices[:target_neighbors]
            trimmed_distances[cell_idx] = row_distances[:target_neighbors]

        return trimmed_indices, trimmed_distances

    def _grit_knn_available_bytes(self):
        """Arena-aware free VRAM in bytes for sizing the GRIT k-NN residency.

        ``OS-free + TF arena slack``, where the slack counts only memory this
        process holds — correct under GPU-growth mode (the
        default), where the OS-free number alone understates what TF can still
        allocate. Mirrors ``_get_gpu_aware_tile_sizes``. Returns ``None`` if no
        probe is available (caller assumes a conservative default).
        """
        return _predictor_support._available_vram_bytes(allocator="tf")

    def _get_cuml_gpu_available_bytes(
        self, *, silent: bool = False,
    ) -> Optional[int]:
        """Live free-byte estimate for cuML / cugraph allocations via cupy.

        Unlike :meth:`_get_gpu_aware_batch_size` (which probes TF's pool),
        cuML and cugraph allocate through RMM, which wraps cupy's default
        memory pool.  The real "budget" is therefore whatever
        ``cp.cuda.Device().mem_info`` reports as free **after** asking the
        cupy pool to release any cached-but-unused blocks.

        Returns:
            Free bytes available for a cuML / cugraph allocation, or
            ``None`` when no CUDA device is visible or probing fails.
            Callers should treat ``None`` as "probe unavailable — fall
            back to CPU or use a conservative default."
        """
        free_bytes = _predictor_support._available_vram_bytes(allocator="rmm")
        if free_bytes is not None and not silent:
            self._log(f"  VRAM Info (cuML): free={free_bytes / 1e9:.2f} GB")
        return free_bytes

    def _run_cuml_knn(
        self,
        cell_embeddings: np.ndarray,
        *,
        n_neighbors: int,
        verbose: bool = False,
        metric: str = 'euclidean',
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Run cuML brute-force k-NN on a dense embedding matrix.

        Shared primitive used by both :meth:`_run_cuml_projection` (for UMAP
        layout) and the atypical-detection pipeline's SNN graph construction
        in :meth:`evaluate_cells`.

        Args:
            cell_embeddings: ``(n_obs, d)`` float32/float64 embedding matrix
                (typically ``adata.obsm['X_hector']``).
            n_neighbors: Target number of non-self neighbors per cell.
            verbose: If ``True``, print a single status line (matches the
                progress reporting style used by ``reduce_dimensions()``).
            metric: Distance metric for k-NN (default ``'euclidean'``).
                ``_run_cuml_projection`` passes ``'cosine'`` for UMAP;
                ``evaluate_cells`` uses the default ``'euclidean'``.

        Returns:
            ``(trimmed_indices, trimmed_distances, effective_n_neighbors)`` —
            integer neighbor-index array and float32 distance array, both of
            shape ``(n_obs, effective_n_neighbors)``, with self-neighbors
            removed.  ``effective_n_neighbors`` is the clamped neighbor count
            actually returned (always ``<= n_neighbors`` and ``<= n_obs-1``).

        Raises:
            ValueError: If the underlying trimming detects shape mismatches
                or insufficient non-self neighbors.
        """
        from cuml.neighbors import NearestNeighbors as CumlNearestNeighbors

        cell_embeddings = np.asarray(cell_embeddings, dtype=np.float32)
        n_obs = cell_embeddings.shape[0]
        effective_n_neighbors = min(
            max(int(n_neighbors), 1),
            max(n_obs - 1, 1),
        )
        knn_query_neighbors = min(effective_n_neighbors + 1, n_obs)

        if verbose:
            print("  Building the cell-cell neighborhood graph using cuML...")
        knn_model = CumlNearestNeighbors(
            n_neighbors=knn_query_neighbors,
            algorithm='brute',
            metric=metric,
            output_type='numpy',
        )
        knn_model.fit(cell_embeddings)

        # --- VRAM-adaptive query batch size -----------------------------
        # Brute-force kneighbors allocates an internal distance scratch
        # whose peak is roughly (query_batch × n_index × 4 bytes) for the
        # distance matrix plus (query_batch × k × 4) for the top-k.  We
        # size the query batch from the live cupy free pool with an
        # overhead multiplier tuned empirically for cuML brute-force
        # (distance scratch + partial-sort workspace).
        overhead = float(os.environ.get("HECTOR_CUML_KNN_OVERHEAD", 3.0))
        bytes_per_query_row = int(
            (n_obs + knn_query_neighbors) * 4 * max(overhead, 1.0)
        )
        free_bytes = self._get_cuml_gpu_available_bytes(silent=not verbose)
        if free_bytes is None or bytes_per_query_row <= 0:
            # Probe failed — attempt a single-shot call and let the caller
            # catch OOM if the GPU is truly too small.
            query_batch = n_obs
        else:
            # Target ~85 % of free pool; clamp to [256, n_obs].
            query_batch = max(
                256, min(n_obs, int(0.85 * free_bytes // bytes_per_query_row))
            )
        if verbose:
            self._log(
                f"  cuML k-NN query batch: {query_batch} rows "
                f"(n_obs={n_obs}, k={knn_query_neighbors})"
            )

        # Preallocate outputs on host; fill in place per batch.
        neighbor_distances = np.empty(
            (n_obs, knn_query_neighbors), dtype=np.float32
        )
        neighbor_indices = np.empty(
            (n_obs, knn_query_neighbors), dtype=np.int64
        )

        def _is_oom(exc: BaseException) -> bool:
            """True when ``exc`` looks like a CUDA/RMM out-of-memory."""
            if isinstance(exc, MemoryError):
                return True
            text = f"{type(exc).__name__}: {exc}".lower()
            return ("bad_alloc" in text) or ("out_of_memory" in text) or (
                "out of memory" in text
            )

        start = 0
        min_batch = 256
        while start < n_obs:
            end = min(start + query_batch, n_obs)
            try:
                d_batch, i_batch = knn_model.kneighbors(cell_embeddings[start:end])
                neighbor_distances[start:end] = d_batch
                neighbor_indices[start:end] = i_batch
                start = end
            except Exception as exc:
                if not _is_oom(exc):
                    raise
                if query_batch <= min_batch:
                    raise
                # Halve the batch and retry from the same row.  Drain the
                # cupy pool between attempts so the scratch from the failed
                # call is returned to the allocator.
                query_batch = max(min_batch, query_batch // 2)
                try:
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                self._log(
                    f"  cuML k-NN OOM at row {start}; halving query batch "
                    f"to {query_batch} rows and retrying"
                )

        trimmed_indices, trimmed_distances = self._prepare_project_knn_results(
            neighbor_indices,
            neighbor_distances,
            n_obs=n_obs,
            target_neighbors=effective_n_neighbors,
        )
        return trimmed_indices, trimmed_distances, effective_n_neighbors

    def _run_cuml_projection(
        self,
        adata: anndata.AnnData,
        *,
        n_neighbors: int,
        random_state: int,
        min_dist: float,
        spread: float,
        n_epochs: int = 500,
        init: str = 'random',
        negative_sample_rate: int = 5,
        use_rep: str = 'X_hector',
        metric: str = 'cosine',
    ) -> None:
        """Run cuML kNN + UMAP projection on ``adata.obsm[use_rep]``.

        Writes only ``adata.obsm["X_umap"]``.  Does **not** write any Scanpy
        graph artifacts (``obsp["distances"]``, ``obsp["connectivities"]``,
        ``uns["neighbors"]``).

        Reproducibility notes:
            * ``random_state`` is coerced to a Python ``int``, which forces
              cuML's deterministic SGD kernel (GPU-resident, slightly slower);
              passing ``None`` or a ``RandomState`` re-enables the
              non-deterministic atomic-add path.
            * ``init`` is a deterministic PCA-2D array for the default
              'spectral' request (see ``reduce_dimensions()``), giving clean
              global structure without cuML's spectral-init eigendecomposition;
              an explicit ``'random'`` also stays deterministic under a fixed
              ``random_state``.
            * Exact results can depend on the cuML version, GPU architecture,
              and input row order even when a seed is fixed.

        Args:
            adata: AnnData object with ``obsm['X_hector']`` embeddings.
            n_neighbors: Number of nearest neighbors for kNN and UMAP.
            random_state: Random seed for reproducible UMAP coordinates.
            min_dist: Effective minimum embedded distance.
            spread: Effective scale of embedded points.
            n_epochs: SGD optimization epochs.  Higher values produce
                smoother / better-converged layouts at moderate cost.
            init: Embedding initialization for ``CumlUMAP``: a string
                (``'spectral'``/``'random'``) or an init ARRAY (accepted by
                cuML >= 26.04). ``reduce_dimensions()`` passes a deterministic
                PCA-2D array for the default request, avoiding both cuML's
                spectral-init instability and the ``'random'`` "spider" artifact.
            negative_sample_rate: Number of negative samples per positive
                edge in the SGD optimizer.
            use_rep: ``obsm`` key used as the kNN / UMAP input.
                ``reduce_dimensions()`` passes the PCA-reduced rep here when
                ``n_pcs`` pre-reduction is active, otherwise ``'X_hector'``.

        Returns:
            None. Writes ``adata.obsm['X_umap']``.

        Raises:
            ValueError: If ``_prepare_project_knn_results`` detects shape
                mismatches or too few non-self neighbors.
        """
        from cuml.manifold import UMAP as CumlUMAP

        # Coerce to a Python int so cuML engages its deterministic SGD path.
        # ``None`` or numpy.random.RandomState would silently bypass it.
        random_state_int = int(random_state) if random_state is not None else 0

        # Seed global RNGs that helpers (and cuML internals on some versions)
        # may consult.  Cheap, runs once per reduce_dimensions() call.
        np.random.seed(random_state_int)
        try:
            import cupy as cp  # type: ignore
            cp.random.seed(random_state_int)
        except Exception:
            pass

        cell_embeddings = np.asarray(adata.obsm[use_rep], dtype=np.float32)
        trimmed_indices, trimmed_distances, effective_n_neighbors = (
            self._run_cuml_knn(
                cell_embeddings,
                n_neighbors=n_neighbors,
                verbose=True,
                metric=metric,
            )
        )

        print("  Computing UMAP coordinates using cuML ...")
        umap_model = CumlUMAP(
            n_neighbors=effective_n_neighbors,
            metric=metric,
            random_state=random_state_int,
            min_dist=min_dist,
            spread=spread,
            n_epochs=n_epochs,
            init=init,
            negative_sample_rate=negative_sample_rate,
            precomputed_knn=(trimmed_indices, trimmed_distances),
            output_type='numpy',
        )
        adata.obsm['X_umap'] = np.asarray(
            umap_model.fit_transform(cell_embeddings),
            dtype=np.float32,
        )

        return None

    def reduce_dimensions(
        self,
        adata: anndata.AnnData,
        method: str = 'umap',
        perplexity: float = 30.0,
        learning_rate="auto",
        n_neighbors: int = 30,
        random_state: int = 0,
        min_dist: Optional[float] = None,
        spread: Optional[float] = None,
        n_epochs: Optional[int] = None,
        init: Optional[str] = None,
        negative_sample_rate: Optional[int] = None,
        n_pcs: Optional[int] = 50,
        backend: Optional[str] = None,
        metric: str = 'cosine',
    ) -> None:
        """Reduce HECTOR cell embeddings to a 2-D layout for plotting.

        Computes a 2-D representation of ``adata.obsm['X_hector']`` using UMAP
        (default) or t-SNE and writes it to ``adata.obsm['X_umap']`` (UMAP) or
        ``adata.obsm['X_tsne']`` (t-SNE). If ``X_hector`` is absent it is
        computed first via :meth:`compute_cell_embeddings`.

        Uses a GPU-accelerated path by default (MLX on Apple Silicon, cuML on
        NVIDIA) with automatic atomic fallback to the Scanpy CPU path when the
        accelerator is unavailable or fails.

        **Output contract:** every backend writes ONLY the 2-D embedding. The
        method does not persist a kNN neighbor graph. If you need a neighbor
        graph for your own clustering, RNA velocity, PAGA, etc., build it
        explicitly with ``scanpy.pp.neighbors(adata, use_rep='X_hector')`` —
        this is deterministic and independent of which backend drew the layout.

        .. note::
            GPU and CPU backends are **not** numerically equivalent: they use
            different UMAP/t-SNE implementations, so coordinates can differ at
            the same parameters, including after a GPU->CPU fallback.

        Args:
            adata: AnnData object with gene expression data.
            method: ``'umap'`` (default) or ``'tsne'``.
            perplexity: t-SNE perplexity (ignored for UMAP).
            learning_rate: t-SNE learning rate (ignored for UMAP).
            n_neighbors: Neighbors for the UMAP neighborhood graph. Default ``30``.
            random_state: Random seed for reproducible coordinates. On the cuML
                path this is coerced to a Python ``int`` to engage the
                deterministic SGD kernel; passing ``None`` is not recommended.
            min_dist: Effective minimum distance between embedded points;
                lower values yield tighter clusters. ``None`` selects the
                backend default: ``1.0`` for MLX and ``0.5`` otherwise.
            spread: Effective scale of embedded points. ``None`` selects the
                backend default of ``1.0``.
            n_epochs: SGD epochs for the UMAP layout. ``None`` selects ``300``
                for MLX and ``200`` for cuML or Scanpy.
            init: Embedding initialization. ``None`` uses the MLX engine's
                internal initialization on MLX and ``'spectral'`` on cuML or
                Scanpy. On supported cuML versions, a spectral request is
                realized as a deterministic PCA-2D initialization; older cuML
                versions fall back to Scanpy. Pass ``'random'`` or a custom
                initialization array to override this behavior.
            negative_sample_rate: Negative samples per positive edge in the SGD
                optimizer. ``None`` selects the backend default of ``5``.
            n_pcs: If set (default ``50``) and the embedding has more dims, it is
                PCA-reduced to ``n_pcs`` components before building the neighbor
                graph to reduce search dimension. Used transiently,
                **not** persisted in ``adata``. Pass ``None``/``0`` to run on the
                full-dimensional ``X_hector``.
            backend: ``'mlx'``, ``'cuml'``, ``'scanpy'``, or ``None`` (auto).
            metric: kNN distance metric. Defaults to ``'cosine'``.

        Returns:
            None.

        Notes:
            ``reduce_dimensions()`` does not write prediction annotations into
            ``adata.obs``. Use ``predict()`` + ``write_predictions()`` for
            persistent ``hector_prediction`` columns to color the plot by.

        Raises:
            ImportError: If Scanpy is not installed. The current dispatch uses
                Scanpy for shared setup and CPU fallback.
            ValueError: If ``method`` is not ``'umap'`` or ``'tsne'``.
        """
        try:
            import scanpy
        except ImportError:
            raise ImportError(
                "scanpy is required for reduce_dimensions() to build the standard "
                "neighbors graph. Install with: pip install scanpy. cuML can "
                "optionally accelerate reduce_dimensions() when available."
            )

        if method == 'tsne':
            if 'X_hector' not in adata.obsm:
                self.compute_cell_embeddings(adata)
            self._project_tsne(adata, scanpy, perplexity=perplexity,
                               learning_rate=learning_rate,
                               random_state=random_state, n_pcs=n_pcs,
                               backend=backend)
            return None
        if method != 'umap':
            raise ValueError(
                f"Unknown method {method!r}; use 'umap' or 'tsne'."
            )

        if 'X_hector' not in adata.obsm:
            self.compute_cell_embeddings(adata)

        # --- Backend: explicit override, else auto-select by hardware ---
        # ``backend`` may be 'mlx' (Apple Silicon), 'cuml' (NVIDIA), or 'scanpy'
        # (CPU). When None, auto-select MLX -> cuML -> scanpy. Each backend has
        # its own tuned layout recipe; sentinel (None) params are filled here.
        if backend is None:
            backend = _select_umap_backend()
        if backend not in _UMAP_BACKEND_DEFAULTS:
            raise ValueError(
                f"Unknown UMAP backend {backend!r}; "
                "choose 'mlx', 'cuml', 'scanpy', or None (auto)."
            )
        _d = _UMAP_BACKEND_DEFAULTS[backend]
        if min_dist is None:
            min_dist = _d['min_dist']
        if spread is None:
            spread = _d['spread']
        if n_epochs is None:
            n_epochs = _d['n_epochs']
        if init is None:
            init = _d['init']
        if negative_sample_rate is None:
            negative_sample_rate = _d['negative_sample_rate']

        # --- MLX engine path (Apple Silicon): own PCA, no temp rep / scanpy graph ---
        if backend == 'mlx':
            try:
                self._run_mlx_projection(
                    adata,
                    n_neighbors=n_neighbors,
                    n_pcs=n_pcs,
                    min_dist=min_dist,
                    spread=spread,
                    n_epochs=n_epochs,
                    negative_sample_rate=negative_sample_rate,
                    random_state=random_state,
                    metric=metric,
                )
                return None
            except Exception as exc:
                warnings.warn(
                    f"MLX UMAP engine failed ({exc}); falling back to the "
                    "Scanpy CPU path.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                backend = 'scanpy'

        # --- Pre-flight (cuML only; selection already confirmed availability) ---
        use_gpu = backend == 'cuml'

        # --- PCA pre-reduction (transient; never persisted in adata) ---
        # Reduce the HECTOR embedding to ``n_pcs`` components before building
        # the neighbor graph. This denoises the layout and reduces neighbor-
        # search cost. The reduced representation
        # is removed in the ``finally`` block below so it never lands in a
        # saved ``.h5ad``.
        rep_key = 'X_hector'
        if (
            n_pcs
            and 'X_hector' in adata.obsm
            and np.asarray(adata.obsm['X_hector']).shape[1] > n_pcs
        ):
            from sklearn.decomposition import PCA

            source_dims = adata.obsm['X_hector'].shape[1]
            print(
                f"  PCA pre-reduction: {source_dims} -> {n_pcs} dims "
                "before neighbor graph ..."
            )
            pca_seed = int(random_state) if random_state is not None else 0
            x_hector_mat = np.asarray(adata.obsm['X_hector'], dtype=np.float32)
            adata.obsm['X_hector_pca'] = PCA(
                n_components=n_pcs,
                svd_solver='randomized',
                random_state=pca_seed,
            ).fit_transform(x_hector_mat)
            rep_key = 'X_hector_pca'

        # Resolve the cuML init for the default 'spectral' request. cuML's own
        # spectral eigendecomposition is spikier than umap-learn's (and has
        # upstream convergence bugs), while plain 'random' leaves "spider"
        # filament artifacts. So: seed the GPU layout with a deterministic
        # PCA-2D init ARRAY when cuML is new enough to accept one (>= 26.04);
        # on older cuML, route the layout to the Scanpy CPU path (clean,
        # deterministic spectral) rather than degrade to 'random'. An explicit
        # 'random' or a caller-supplied init array is passed through unchanged.
        cuml_init = init
        if use_gpu and isinstance(init, str) and init == 'spectral':
            cuml_ver = _cuml_version()
            if cuml_ver is None:
                # cuML not importable here; leave use_gpu=True so the try/except
                # below attempts cuML, fails, warns, and falls back to Scanpy
                # (unchanged behavior). The init value is irrelevant then.
                cuml_init = 'random'
            elif cuml_ver >= (26, 4):
                cuml_init = _pca2d_init(
                    np.asarray(adata.obsm[rep_key], dtype=np.float32),
                    random_state,
                )
            else:
                print(
                    f"  cuML {cuml_ver[0]}.{cuml_ver[1]:02d} lacks array UMAP "
                    "init; using the Scanpy CPU path for a clean spectral layout."
                )
                use_gpu = False

        try:
            # --- GPU path (atomic: any failure falls back to full CPU path) ---
            if use_gpu:
                try:
                    self._run_cuml_projection(
                        adata,
                        n_neighbors=n_neighbors,
                        random_state=random_state,
                        min_dist=min_dist,
                        spread=spread,
                        n_epochs=n_epochs,
                        init=cuml_init,
                        negative_sample_rate=negative_sample_rate,
                        use_rep=rep_key,
                        metric=metric,
                    )
                    return None
                except Exception as exc:
                    warnings.warn(
                        "cuML acceleration for reduce_dimensions() is unavailable or failed "
                        f"({exc}). \nFalling back to the slower Scanpy pipeline.\n "
                        "CPU fallback may yield different neighbors and UMAP coordinates from the GPU path.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            # --- CPU path ---
            self._run_scanpy_projection(
                adata,
                scanpy,
                n_neighbors=n_neighbors,
                random_state=random_state,
                min_dist=min_dist,
                spread=spread,
                n_epochs=n_epochs,
                init=init,
                negative_sample_rate=negative_sample_rate,
                use_rep=rep_key,
                metric=metric,
            )

            return None
        finally:
            # Drop the transient PCA rep so it is never persisted, even on
            # the cuML -> Scanpy fallback path.
            adata.obsm.pop('X_hector_pca', None)

    def predict(
        self,
        adata,
        cell_id_column: Optional[str] = None,
        label_format: str = 'id',
        export_score_matrix: bool = False,
        # Runtime prediction settings (override config defaults)
        top_k: Optional[int] = None,
        use_full_ontology: Optional[bool] = None,
        use_grit: Optional[bool] = None,
        grit_iterations: Optional[int] = None,
        grit_alpha: Optional[float] = None,
        grit_entropy_threshold: Optional[float] = None,
        grit_refinement_percentile: Optional[float] = None,
        # PPR settings (override config defaults)
        use_asymmetric_ppr: Optional[bool] = None,
        forward_weight: Optional[float] = None,
        ppr_alpha: Optional[float] = None,
        # Variance-based quality filtering (override config defaults)
        filter_low_quality: Optional[bool] = None,
        variance_threshold: Optional[float] = None,
        # Cache control
        force_recompute: bool = False,
    ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Make predictions on new data.

        Every argument from ``top_k`` onward overrides this predictor's config
        for this call only; ``None`` means "use the config default".

        Args:
            adata: AnnData object with gene expression data
                   - adata.X: raw counts. Normalized or scaled input is
                     rejected; log-transformed input is detected and reversed.
                   - adata.var: must contain Ensembl gene IDs (auto-detected)
                   - adata.obs_names or adata.obs[cell_id_column]: cell identifiers
            cell_id_column: Column in adata.obs holding cell IDs (uses obs_names if None)
            label_format: Format for prediction output (default: 'id')
                          - 'id': ontology IDs, e.g. 'CL:0000128'
                          - 'name': cell type names, e.g. 'monocyte'
                          - 'both': "name (id)", e.g. 'monocyte (CL:0000128)'
                          Falls back to the ID where no name is available.
            export_score_matrix: Also return the full per-cell score matrix used
                          for ranking.
            top_k: Number of top predictions to return.
            use_full_ontology: Score against the full ontology (zero-shot).
            use_grit: Apply GRIT graph-based label refinement.
            grit_iterations: Number of GRIT propagation iterations.
            grit_alpha: GRIT damping factor, 0-1.
            grit_entropy_threshold: Fixed entropy threshold; None auto-detects.
            grit_refinement_percentile: Percentage of the lower end of the
                          prediction distribution to refine, in auto mode.
            use_asymmetric_ppr: Use directional PPR for profile scoring.
            forward_weight: Blending weight for forward PPR, in [0.0, 1.0].
            ppr_alpha: PPR restart probability; None uses the checkpoint value.
            filter_low_quality: Flag cells whose encoder variance exceeds the
                          threshold as low quality.
            variance_threshold: The threshold that flag uses; None auto-detects it.
            force_recompute: Ignore anything cached on *adata* and re-run the
                          encoder and classification from scratch.

        Returns:
            predictions: Barcode-indexed DataFrame with ``top_1_prediction`` /
                ``top_1_score`` through ``top_k``.
            score_matrix_df: Returned only when ``export_score_matrix=True``.
                DataFrame indexed by cell ID with ontology IDs as columns.

        """
        settings = self._resolve_prediction_settings(
            top_k=top_k,
            use_full_ontology=use_full_ontology,
            use_grit=use_grit,
            grit_iterations=grit_iterations,
            grit_alpha=grit_alpha,
            grit_entropy_threshold=grit_entropy_threshold,
            grit_refinement_percentile=grit_refinement_percentile,
            use_asymmetric_ppr=use_asymmetric_ppr,
            forward_weight=forward_weight,
            ppr_alpha=ppr_alpha,
            filter_low_quality=filter_low_quality,
            variance_threshold=variance_threshold,
        )

        valid_formats = ['id', 'name', 'both']
        if label_format not in valid_formats:
            raise ValueError(f"Invalid label_format '{label_format}'. Must be one of: {valid_formats}")

        self._log(f"  [INFO] Making predictions on {adata.shape[0]} cells...")

        self._log(
            "  Settings: "
            f"top_k={settings.top_k}, use_grit={settings.use_grit}, "
            f"zero_shot={settings.use_full_ontology}"
        )

        # Preflight memory guard (default-on, warn-only): silent when the run
        # fits, one line when projected peak may exceed free memory.
        try:
            run_preflight(
                self, adata,
                use_grit=settings.use_grit,
                use_full_ontology=settings.use_full_ontology,
                verbose=True,
            )
        except Exception as exc:  # a guard failure must never block prediction
            self._log(f"  (preflight memory check skipped: {exc!r})")

        if settings.top_k == 1 and not export_score_matrix:
            cell_vectors, _, core_result = self._ensure_top1_predictions(
                adata,
                use_full_ontology=settings.use_full_ontology,
                use_grit=settings.use_grit,
                grit_iterations=settings.grit_iterations,
                grit_alpha=settings.grit_alpha,
                grit_entropy_threshold=settings.grit_entropy_threshold,
                grit_refinement_percentile=settings.grit_refinement_percentile,
                use_asymmetric_ppr=settings.use_asymmetric_ppr,
                forward_weight=settings.forward_weight,
                ppr_alpha=settings.ppr_alpha,
                filter_low_quality=settings.filter_low_quality,
                variance_threshold=settings.variance_threshold,
                force_recompute=force_recompute,
                allow_annotation_reuse=True,
            )
        else:
            cell_vectors, cell_variance_array, prepared_expression = self._ensure_embeddings(
                adata,
                force_recompute=force_recompute,
            )
            core_result = self._run_prediction_from_embeddings(
                adata,
                cell_vectors=cell_vectors,
                cell_variance_array=cell_variance_array,
                settings=settings,
                prepared_expression=prepared_expression,
                export_score_matrix=export_score_matrix,
                force_recompute=force_recompute,
            )

        try:
            self._log("  ✅ Predictions complete!")
            self._log("  Step 5: Formatting results...")
            prediction_fingerprint = self._build_prediction_fingerprint(settings)
            results_df = self._format_prediction_results_from_core_result(
                adata,
                core_result=core_result,
                settings=settings,
                prediction_fingerprint=prediction_fingerprint,
                label_format=label_format,
                cell_id_column=cell_id_column,
            )

            self._log(f"  ✅ Results formatted as DataFrame with shape {results_df.shape}")
            if not export_score_matrix:
                return results_df

            if cell_id_column is not None:
                cell_ids = adata.obs[cell_id_column].tolist()
            else:
                cell_ids = adata.obs_names.tolist()

            active_class_ids = self._get_active_class_ids(
                settings.use_full_ontology
            )
            if core_result.score_matrix_array is None:
                raise RuntimeError("score_matrix_array was not captured during prediction.")
            if core_result.score_matrix_array.shape != (
                len(cell_ids), len(active_class_ids)
            ):
                raise ValueError(
                    "Score matrix shape does not match active class IDs: "
                    f"{core_result.score_matrix_array.shape} vs "
                    f"({len(cell_ids)}, {len(active_class_ids)})"
                )

            score_matrix_df = pd.DataFrame(
                core_result.score_matrix_array,
                index=pd.Index(cell_ids, name='cell_id'),
                columns=active_class_ids,
            )
            return results_df, score_matrix_df
        finally:
            core_result.close_transient_arrays()

    def _run_rollup(
        self,
        predictions: pd.Series,
        min_cells_pct: float = 1.0,
        max_cells: int = 20,
        rare_label: str = 'Rare Cells',
    ) -> Tuple[pd.Series, pd.Series]:
        """Bucket rare cell-type annotations below a population threshold.

        Cell types whose count falls below the effective threshold are
        relabeled to *rare_label*.  The threshold is computed as
        ``min(ceil(min_cells_pct / 100 * total_cells), max_cells)``.

        Args:
            predictions: Series of cell-type labels (``top_1_prediction``),
                indexed by cell barcode.
            min_cells_pct: Minimum percentage of total cells a type must reach
                to avoid being bucketed.  Must be in ``[0.0, 100.0]``.
            max_cells: Hard cap on the threshold so that large datasets do not
                require an unreasonably high cell count.
            rare_label: Label assigned to cell types below the threshold.

        Returns:
            A tuple ``(bucketed_annotation, is_bucketed)`` of two
            :class:`pd.Series` sharing the same index as *predictions*.
            ``bucketed_annotation`` holds the (possibly relabeled) label;
            ``is_bucketed`` is ``True`` for cells whose label was moved into
            the rare-cell bucket.

        Raises:
            ValueError: If *min_cells_pct* is outside ``[0.0, 100.0]``.
        """
        if min_cells_pct < 0.0 or min_cells_pct > 100.0:
            raise ValueError(
                f"min_cells_pct must be between 0.0 and 100.0, got {min_cells_pct}"
            )

        # Short-circuit: no rollup requested
        if min_cells_pct == 0.0:
            bucketed = predictions.copy()
            is_bucketed = pd.Series(False, index=predictions.index)
            return bucketed, is_bucketed

        total_cells = len(predictions)
        pct_threshold = math.ceil(min_cells_pct / 100.0 * total_cells)
        threshold = min(pct_threshold, max_cells)

        cell_counts = predictions.value_counts()
        rare_types = set(cell_counts[cell_counts < threshold].index)

        bucketed = predictions.copy()
        if rare_types:
            bucketed[bucketed.isin(rare_types)] = rare_label

        is_bucketed = bucketed != predictions

        n_rare = len(rare_types)
        n_cells = int(is_bucketed.sum())
        self._log(
            f"  Rare-cell rollup summary: threshold={threshold} cells "
            f"(min({min_cells_pct}% of {total_cells}, {max_cells})), "
            f"{n_rare} type(s) bucketed as '{rare_label}', "
            f"{n_cells} cell(s) affected"
        )

        return bucketed, is_bucketed

    def _resolve_results_join_key(
        self,
        adata,
        results: pd.DataFrame,
    ) -> Tuple[pd.Index, pd.Index]:
        """Pick the right cell-ID join key for :meth:`write_predictions`.

        Returns a ``(results_idx, obs_names_str)`` pair of string-typed
        indices to feed into the reindex below.

        Resolution order:

        1. ``results.index`` directly (canonical: this is what ``predict()``
           sets and the documented contract).
        2. If the index matches nothing, scan every non-numeric column and
           pick the one whose values overlap ``adata.obs_names`` best. We
           use *value-based* detection rather than hardcoded column names
           because callers may rename the cell-ID column (CSV round-trips,
           merges, custom pipelines) and we don't want a silent failure
           the moment the name drifts from any one convention.
        3. Whitespace-stripped retry on the same set of candidates.

        Raises ``ValueError`` if no cell can be matched at all and emits a
        ``RuntimeWarning`` for every non-canonical fallback or partial match.
        """
        obs_names_str = pd.Index(adata.obs_names).astype(str)
        n_obs = adata.n_obs

        def _build_candidates(
            obs: pd.Index,
            strip: bool,
        ) -> List[Tuple[str, pd.Index, int]]:
            """Score each candidate cell-ID source against *obs*."""
            scored: List[Tuple[str, pd.Index, int]] = []

            def _score(label: str, raw: pd.Index) -> None:
                try:
                    idx = raw.astype(str)
                except (TypeError, ValueError):
                    return
                if strip:
                    idx = idx.str.strip()
                scored.append((label, idx, int(obs.isin(idx).sum())))

            _score("results.index", pd.Index(results.index))
            for col in results.columns:
                # Skip dtypes that cannot be cell barcodes. Object/string
                # columns ('O', 'U', 'S') are the typical case; integer
                # columns ('i', 'u') are allowed in case someone uses
                # numeric IDs.
                if results[col].dtype.kind not in ("O", "U", "S", "i", "u"):
                    continue
                _score(f"results[{col!r}]", pd.Index(results[col]))
            return scored

        # First pass: exact string match.
        candidates = _build_candidates(obs_names_str, strip=False)
        canonical_label = "results.index"
        canonical_matched = next(
            (m for label, _, m in candidates if label == canonical_label),
            0,
        )

        chosen_label, chosen_idx, matched = max(
            candidates, key=lambda c: c[2]
        ) if candidates else (canonical_label, pd.Index([], dtype=str), 0)

        # Prefer the canonical index when it matches at all — even a
        # partial match on the index beats a column match, because the
        # index is the documented contract and a column's match could be
        # coincidental.
        if canonical_matched > 0:
            chosen_label = canonical_label
            chosen_idx = next(idx for label, idx, _ in candidates if label == canonical_label)
            matched = canonical_matched

        used_obs = obs_names_str

        # Second pass: whitespace-stripped retry only if nothing matched.
        if matched == 0:
            stripped_obs = obs_names_str.str.strip()
            stripped_candidates = _build_candidates(stripped_obs, strip=True)
            if stripped_candidates:
                s_label, s_idx, s_matched = max(
                    stripped_candidates, key=lambda c: c[2]
                )
                if s_matched > 0:
                    warnings.warn(
                        f"write_predictions: cell IDs matched only after "
                        f"stripping whitespace (source: {s_label}) — "
                        f"consider cleaning barcodes upstream.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    chosen_label = s_label
                    chosen_idx = s_idx
                    matched = s_matched
                    used_obs = stripped_obs

        if matched == 0:
            scored_summary = ", ".join(
                f"{label}={m}" for label, _, m in candidates
            ) or "no candidate columns"
            sample_obs = list(used_obs[:3])
            raise ValueError(
                f"write_predictions: could not match any of {n_obs} cells "
                f"in adata to the predictions DataFrame. Tried "
                f"results.index and every non-numeric column "
                f"({scored_summary}). The predictions DataFrame must "
                f"either be indexed by cell barcodes (as returned by "
                f"predict()) or contain a column whose values are cell "
                f"barcodes from the same AnnData. Sample adata.obs_names: "
                f"{sample_obs!r}."
            )

        if chosen_label != canonical_label:
            warnings.warn(
                f"write_predictions: results.index did not match "
                f"adata.obs_names; detected cell barcodes in "
                f"{chosen_label} instead ({matched}/{n_obs} cells "
                f"matched). This usually means the predictions DataFrame "
                f"was passed through reset_index() or read back from a "
                f"flat file.",
                RuntimeWarning,
                stacklevel=3,
            )

        if matched < n_obs:
            warnings.warn(
                f"write_predictions: only {matched}/{n_obs} adata cells "
                f"matched (using {chosen_label}) — unmatched cells will "
                f"receive NaN values in adata.obs.",
                RuntimeWarning,
                stacklevel=3,
            )

        return chosen_idx, used_obs

    def write_predictions(
        self,
        adata,
        results: pd.DataFrame,
        columns: Optional[List[str]] = None,
        prefix: str = 'hector_',
        overwrite: bool = True,
        rare_rollup: bool = True,
        min_cells_pct: float = 1.0,
        max_cells: int = 20,
        rare_label: str = 'Rare Cells',
    ) -> None:
        """Write prediction results back into ``adata.obs`` in-place.

        Joins on cell barcodes (``results.index`` matched against
        ``adata.obs_names``).

        Args:
            adata: AnnData object (must match the one used for ``predict()``).
            results: DataFrame returned by ``predict()`` (barcode-indexed).
            columns: Subset of result columns to write. ``None`` writes all
                default columns (prediction, score, cell variance, and
                ``is_low_quality``).
            prefix: String prepended to each column name in ``adata.obs``.
            overwrite: If ``False``, skip columns that already exist in
                ``adata.obs`` and log a warning. If ``True`` (default),
                overwrite silently.
            rare_rollup: When ``True`` (default), bucket rare cell types below
                the population threshold into *rare_label* and write
                ``bucketed_prediction`` and ``is_bucketed`` columns.
                ``{prefix}prediction`` always holds the raw top-1 label
                regardless of this flag; the bucketed view lives only in
                ``bucketed_prediction``.
            min_cells_pct: Minimum percentage of total cells a cell type must
                reach to avoid being bucketed.  Must be in ``[0.0, 100.0]``.
                Only used when *rare_rollup* is ``True``.
            max_cells: Hard cap on the computed threshold so that large
                datasets do not require an unreasonably high cell count.
                Effective threshold is ``min(ceil(pct * total), max_cells)``.
            rare_label: Label assigned to cell types below the threshold.

        Column mapping (internal → adata.obs):
            top_1_prediction      → {prefix}prediction  (always the raw top-1 label)
            top_1_score           → {prefix}prediction_confidence
            hector_cell_variance  → {prefix}cell_variance
            is_low_quality        → is_low_quality (no prefix, defaults to False)

        Additional columns when rare_rollup=True:
            bucketed_prediction     — top-1 label with rare types replaced
                                      by *rare_label*; the only column
                                      that may contain *rare_label*.
            is_bucketed             — True if the cell was moved into the rare bucket
        """
        self._ensure_cache_writable(adata)

        # --- resolve cell-ID join key (results -> adata.obs_names) -----
        # Canonical: results.index holds barcodes (as set by predict()).
        # Also accept reset-index results with a ``cell_id`` column and normalize
        # whitespace/dtypes before matching barcodes.
        results_idx, obs_names_str = self._resolve_results_join_key(
            adata, results
        )

        # --- apply rare-label roll-up if requested ---------------------
        bucketed_annotation = None
        is_bucketed_series = None
        if rare_rollup and 'top_1_prediction' in results.columns:
            bucketed_annotation, is_bucketed_series = self._run_rollup(
                results['top_1_prediction'], min_cells_pct, max_cells, rare_label
            )

        col_map = {
            'top_1_prediction': f'{prefix}prediction',
            'top_1_score': f'{prefix}prediction_confidence',
            'hector_cell_variance': f'{prefix}cell_variance',
            'is_low_quality': 'is_low_quality',
        }

        if columns is not None:
            col_map = {k: v for k, v in col_map.items() if k in columns}

        n_written = 0
        prediction_col_written = False
        score_col_written = False
        for src_col, obs_col in col_map.items():
            if src_col not in results.columns:
                continue
            if obs_col in adata.obs.columns and not overwrite:
                self._log(f"  ℹ️ '{obs_col}' already exists in adata.obs — skipping (use overwrite=True)")
                continue

            series = pd.Series(results[src_col].values, index=results_idx)
            adata.obs[obs_col] = series.reindex(obs_names_str).values
            n_written += 1
            if src_col == 'top_1_prediction':
                prediction_col_written = True
            elif src_col == 'top_1_score':
                score_col_written = True

        # --- write bucketed columns when rare_rollup=True --------------
        if bucketed_annotation is not None:
            sa_col = 'bucketed_prediction'
            is_col = 'is_bucketed'

            sa_series = pd.Series(bucketed_annotation.values, index=results_idx)
            is_series = pd.Series(is_bucketed_series.values, index=results_idx)

            for col_name, col_data in [(sa_col, sa_series), (is_col, is_series)]:
                if col_name in adata.obs.columns and not overwrite:
                    self._log(f"  ℹ️ '{col_name}' already exists in adata.obs — skipping (use overwrite=True)")
                    continue
                adata.obs[col_name] = col_data.reindex(obs_names_str).values
                n_written += 1

        matched = int(obs_names_str.isin(results_idx).sum())
        self._log(f"  Annotated adata.obs with {n_written} columns (prefix='{prefix}', {matched}/{adata.n_obs} cells matched)")

        prediction_col = f'{prefix}prediction'
        score_col = f'{prefix}prediction_confidence'
        cell_variance_col = f'{prefix}cell_variance'
        if prediction_col_written and score_col_written:
            metadata = self._extract_prediction_metadata(results)
            manifest = self._build_annotation_manifest(
                prediction_col=prediction_col,
                score_col=score_col,
                cell_variance_col=(
                    cell_variance_col
                    if cell_variance_col in adata.obs.columns
                    else None
                ),
                label_format=metadata.get("label_format")
                or self._infer_label_format(adata.obs[prediction_col]),
                prediction_fingerprint=metadata.get("prediction_fingerprint"),
            )
            self._write_annotation_manifest(adata, manifest)
    
    def _get_visualization_vectors_from_prediction_session(
        self,
        prediction_session: _PredictionSession,
    ) -> Dict[str, Any]:
        """Build visualization vectors from a resolved prediction session."""

        cell_vectors = prediction_session.cell_vectors
        settings = prediction_session.settings
        core_result = prediction_session.core_result
        visualization_state = self._build_visualization_state(
            settings.use_full_ontology
        )

        result_dict = {
            'cell_vectors': cell_vectors,
            'predictions': core_result.top_indices[:, 0],
            'prediction_scores': core_result.prediction_scores,
            'hector_cell_variance': core_result.cell_variance_array,
            **visualization_state,
        }
        if core_result.low_quality_mask is not None:
            result_dict['is_low_quality'] = core_result.low_quality_mask
        return result_dict

    def get_visualization_vectors(
        self,
        adata,
        # GRIT parameters for Unity Mode
        use_grit: bool = True,
        grit_refinement_percentile: Optional[float] = None,
        # Variance-based quality filtering (override config defaults)
        filter_low_quality: Optional[bool] = None,
        variance_threshold: Optional[float] = None,
        force_recompute: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract cell embeddings and ontology node embeddings for visualization.
        
        When use_grit=True (Unity Mode), runs full GRIT refinement pipeline
        identical to predict() to ensure visualization matches prediction reports.
        
        Args:
            adata: AnnData object with gene expression data
            use_grit: Enable GRIT refinement for Unity Mode (default: True)
            grit_refinement_percentile: Controls which cells get GRIT refinement
                - None (default): Auto-detect threshold using Multi-Otsu
                  (finds natural boundary in entropy distribution)
                - 0-100: Manual override - refine top X% lower end prediction
                  (e.g., 10 = refine cells with entropy in top 10%)
            filter_low_quality: Override config's filter_low_quality for this call only (None = use config default)
            variance_threshold: Override config's variance_threshold for this call only (None = use config default)
            force_recompute: If True, bypass all cache checks and re-run the full
                encoder and classification pipelines regardless of any pre-existing
                cached embeddings or predictions in adata (default: False)
            
        Returns:
            Dictionary with keys:
            - 'cell_vectors': np.ndarray [n_cells, latent_dim] - encoder means
            - 'predictions': np.ndarray [n_cells] - GRIT-refined class indices
            - 'prediction_scores': np.ndarray [n_cells] - confidence scores
            - 'hector_cell_variance': np.ndarray [n_cells] - per-cell variance scores
            - 'node_vectors': np.ndarray [n_nodes, latent_dim] - ontology embeddings
            - 'node_names': List[str] - cell type names for all nodes
            - 'adjacency_matrix': np.ndarray - ontology adjacency matrix
            - 'pure_adjacency_matrix': np.ndarray - direct unweighted ontology adjacency
            - 'co_graph': Dict[str, List[str]] - child→parents mapping for NetworkX
            - 'is_low_quality': np.ndarray[bool] [n_cells] - (only when filter_low_quality=True)
        """
        unity_mode = "On" if use_grit else "Off"
        self._log(f"[INFO] Extracting visualization vectors (Unity Mode: GRIT={unity_mode})...")
        self._log(f"[INFO] Processing {adata.shape[0]} cells...")

        prediction_session = self._resolve_prediction_session(
            adata,
            use_grit=use_grit,
            grit_refinement_percentile=grit_refinement_percentile,
            filter_low_quality=filter_low_quality,
            variance_threshold=variance_threshold,
            force_recompute=force_recompute,
            require_pre_grit_scores=False,
            allow_cached_prediction_reuse=True,
        )
        try:
            return self._get_visualization_vectors_from_prediction_session(
                prediction_session
            )
        finally:
            prediction_session.core_result.close_transient_arrays()

    def evaluate_cells(
        self,
        adata: anndata.AnnData,
        *,
        # Trajectory scoring
        min_cells_number: int = 20,
        top_k_neighbors: int = 15,
        temperature: float = 0.1,
        class_relative_adherence: bool = True,
        class_relative_blend_weight: float = 0.5,
        # SNN graph
        snn_n_neighbors: int = 30,
        snn_min_shared_fraction: float = 0.1,
        # Lineage pruning
        ontology_prune_lineage_levels: Optional[int] = 4,
        ontology_prune_soft_power: float = 1.0,
        # Community gate (no leiden_resolution; G1 picks it via ARI scan)
        min_community_size: Optional[int] = None,
        effect_floor: float = 0.10,
        # G1 auto-resolution Leiden scan
        g1_resolution_range: Tuple[float, float] = (0.5, 2.0),
        g1_resolution_step: float = 0.1,
        # G5 multi-resolution ontology rollup
        g5_d_cuts: Tuple[float, ...] = (0.5, 1.5, 2.5, 3.5, 4.5),
        g5_min_lca_level: int = 2,
        # 1-D BGMM seed selection (only the threshold is user-tunable;
        # min_weight / max_std / n_components / train_subsample / fdr_alpha
        # are hard-coded -- see module-level constants).
        bgmm_threshold: float = 0.38,
        # Diagnostic plot
        diagnostic_top_k_classes: int = 30,
        diagnostic_min_class_size: int = 20,
        # Standard
        backend: str = "auto",
        random_seed: int = 42,
        pynndescent_n_jobs: Optional[int] = None,
        force_recompute: bool = False,
        diagnostic_output_path: Optional[str] = None,
    ) -> EvaluateCellsResult:
        """Detect cells with atypical model behavior using ontology-routed gates.

        The method derives four abnormality features, selects seed cells with a
        one-dimensional Bayesian Gaussian mixture, constructs an ontology-pruned
        shared-nearest-neighbor graph, and chooses a single- or multi-lineage
        route from the lowest common ancestor of the predicted classes. G1 runs
        Leiden at the resolution that maximizes agreement with predicted classes.
        On the multi-lineage route, G5 additionally evaluates ontology rollups at
        several distance cuts. A per-class enrichment test prevents a class with
        below-baseline seed prevalence from inheriting a group-level verdict.

        The detector targets broad shifts relative to the model and reference
        population. It is not a substitute for direct class identification with
        :meth:`predict`. Leiden-based gates can also become less stable when the
        input contains several hundred distinct predicted classes.

        Results are written to the canonical atypical-cell columns in
        ``adata.obs``. Route-level values such as ``route``, ``lca``,
        ``lca_is_root_only``, ``g1_resolution``, and ``g5_per_depth_summary``
        are stored in ``adata.uns['hector_atypical_manifest']`` and exposed by
        :attr:`EvaluateCellsResult.summary`.

        Args:
            adata: AnnData containing predictions and ``obsm['X_hector']``.
            min_cells_number: Minimum number of cells required for evaluation.
            top_k_neighbors: Neighbor count used for trajectory scoring.
            temperature: Temperature used when converting similarities to
                adherence scores.
            class_relative_adherence: Whether to normalize adherence relative
                to each predicted class.
            class_relative_blend_weight: Weight assigned to class-relative
                adherence when it is blended with the global score.
            snn_n_neighbors: Neighbor count used to construct the SNN graph.
            snn_min_shared_fraction: Minimum shared-neighbor fraction retained
                as an SNN edge.
            ontology_prune_lineage_levels: Maximum lineage distance retained by
                hard ontology pruning, or ``None`` to disable the hard cutoff.
            ontology_prune_soft_power: Exponent applied to soft ontology edge
                weights.
            min_community_size: Minimum community size for gate evaluation, or
                ``None`` to choose it automatically.
            effect_floor: Minimum enrichment effect required by a community gate.
            g1_resolution_range: Inclusive ``(low, high)`` bounds for the G1
                Leiden-resolution scan.
            g1_resolution_step: Step size for the G1 resolution scan.
            g5_d_cuts: Ontology-distance cuts used to create G5 partitions.
            g5_min_lca_level: Minimum LCA depth at which predicted classes may
                share a G5 bucket. ``0`` disables this branch barrier.
            bgmm_threshold: Posterior-probability threshold used to select BGMM
                seed cells.
            diagnostic_top_k_classes: Maximum number of classes shown in the
                diagnostic class-level panel.
            diagnostic_min_class_size: Minimum class size shown in that panel.
            backend: Neighbor-search backend: ``'auto'`` or a supported explicit
                backend name.
            random_seed: Seed for stochastic graph and mixture operations.
            pynndescent_n_jobs: Worker count for PyNNDescent, or ``None`` for its
                default.
            force_recompute: Ignore a compatible cached evaluation when true.
            diagnostic_output_path: Optional path for the diagnostic PNG grid.

        Returns:
            Evaluation result containing per-cell outputs and a route summary.
        """
        from .trajectory_support import (
            _build_co_graph_from_directed_adj,
            _build_snn_graph_v2,
            _build_trajectory_scoring_contexts,
            _fit_bgmm_aberrant_1d,
            _label_communities_by_enrichment_fdr,
            _lca_routed_build_g5_partitions,
            _lca_routed_predicted_classes_lca,
            _lca_routed_scan_leiden_by_ari,
            _per_class_fairness_mask,
            _prune_snn_by_ontology_v1,
        )
        from .trajectory_render import _plot_atypical_lca_routed_diagnostic_grid

        # --- Validation ---------------------------------------------------
        if ontology_prune_lineage_levels is not None:
            if (not isinstance(
                    ontology_prune_lineage_levels, (int, np.integer))
                    or ontology_prune_lineage_levels < 0):
                raise ValueError(
                    "evaluate_cells: ontology_prune_lineage_levels "
                    "must be a non-negative integer or None, got "
                    f"{ontology_prune_lineage_levels!r}"
                )
        if ontology_prune_soft_power < 0:
            raise ValueError(
                "evaluate_cells: ontology_prune_soft_power must be "
                f">= 0, got {ontology_prune_soft_power!r}"
            )
        if not (0.0 <= float(bgmm_threshold) <= 1.0):
            raise ValueError(
                "evaluate_cells: bgmm_threshold must be in "
                f"[0, 1], got {bgmm_threshold!r}"
            )
        if (
            not isinstance(g1_resolution_range, tuple)
            or len(g1_resolution_range) != 2
            or not (g1_resolution_range[0] > 0)
            or not (g1_resolution_range[1] > g1_resolution_range[0])
        ):
            raise ValueError(
                "evaluate_cells: g1_resolution_range must be a "
                "(lo, hi) tuple with 0 < lo < hi, got "
                f"{g1_resolution_range!r}"
            )
        if not (g1_resolution_step > 0):
            raise ValueError(
                "evaluate_cells: g1_resolution_step must be > 0, "
                f"got {g1_resolution_step!r}"
            )
        if not g5_d_cuts:
            raise ValueError(
                "evaluate_cells: g5_d_cuts must be non-empty"
            )
        if any(d <= 0 for d in g5_d_cuts):
            raise ValueError(
                "evaluate_cells: g5_d_cuts must all be > 0, got "
                f"{g5_d_cuts!r}"
            )
        if (not isinstance(g5_min_lca_level, (int, np.integer))
                or g5_min_lca_level < 0):
            raise ValueError(
                "evaluate_cells: g5_min_lca_level must be a "
                f"non-negative integer, got {g5_min_lca_level!r}"
            )

        # Hard-coded BGMM + community-gate constants.  See module-level
        # ``_ATYPICAL_FDR_ALPHA`` etc. for rationale.
        fdr_alpha            = _ATYPICAL_FDR_ALPHA
        bgmm_n_components    = _BGMM_N_COMPONENTS
        bgmm_min_weight      = _BGMM_MIN_WEIGHT
        bgmm_max_std         = _BGMM_MAX_STD
        bgmm_train_subsample = _BGMM_TRAIN_SUBSAMPLE

        n_cells = int(adata.shape[0])
        self._log(
            f"[INFO] Atypical detection for {n_cells} cells..."
        )

        if min_community_size is None:
            min_community_size = max(50, int(0.002 * n_cells))

        settings = self._resolve_prediction_settings(
            top_k=1,
            use_grit=None,
            grit_refinement_percentile=None,
        )
        prediction_fingerprint = self._build_prediction_fingerprint(settings)
        atypical_fingerprint = self._build_atypical_fingerprint(
            snn_n_neighbors=snn_n_neighbors,
            snn_min_shared_fraction=snn_min_shared_fraction,
            backend=backend,
            min_community_size=min_community_size,
            fdr_alpha=fdr_alpha,
            effect_floor=effect_floor,
            temperature=temperature,
            top_k_neighbors=top_k_neighbors,
            min_cells_number=min_cells_number,
            class_relative_adherence=class_relative_adherence,
            class_relative_blend_weight=class_relative_blend_weight,
            bgmm_n_components=bgmm_n_components,
            bgmm_min_weight=bgmm_min_weight,
            bgmm_max_std=bgmm_max_std,
            bgmm_train_subsample=bgmm_train_subsample,
            bgmm_threshold=bgmm_threshold,
            random_seed=random_seed,
            pynndescent_n_jobs=pynndescent_n_jobs,
            ontology_prune_lineage_levels=ontology_prune_lineage_levels,
            ontology_prune_soft_power=float(ontology_prune_soft_power),
            g1_resolution_range=tuple(g1_resolution_range),
            g1_resolution_step=float(g1_resolution_step),
            g5_d_cuts=tuple(g5_d_cuts),
            g5_min_lca_level=int(g5_min_lca_level),
        )

        if not force_recompute:
            cached = self._resolve_atypical_manifest(
                adata,
                expected_prediction_fingerprint=prediction_fingerprint,
                expected_atypical_fingerprint=atypical_fingerprint,
            )
            if cached is not None:
                self._log(
                    "[INFO] Using cached evaluate_cells results "
                    "from adata.uns['hector_atypical_manifest']"
                )
                return self._build_result_from_adata(adata, cached)

        prediction_session = self._resolve_prediction_session(
            adata,
            use_grit=None,
            grit_refinement_percentile=None,
            force_recompute=force_recompute,
            require_pre_grit_scores=True,
            allow_cached_prediction_reuse=False,
        )
        core_result = prediction_session.core_result
        cell_embeddings_scoring = prediction_session.cell_vectors
        cell_embeddings_full = np.asarray(
            cell_embeddings_scoring, dtype=np.float32
        )

        predictive_score_matrix = core_result.pre_grit_score_matrix_array
        if predictive_score_matrix is None:
            raise RuntimeError(
                "evaluate_cells requires pre-GRIT score capture."
            )

        visualization_state = self._build_visualization_state(
            settings.use_full_ontology
        )
        node_embeddings = visualization_state["node_vectors"]
        node_names = list(visualization_state["node_names"])
        predictions = np.asarray(
            core_result.top_indices[:, 0], dtype=np.int64
        )

        # --- Trajectory context + 4-D feature stack ----------------------
        scoring_contexts = _build_trajectory_scoring_contexts(
            adjacency_matrix=visualization_state["pure_adjacency_matrix"],
            node_names=node_names,
            predicted_indices=predictions,
            min_cells_number=min_cells_number,
        )
        canonical_context = scoring_contexts["canonical"]
        keep_mask = np.asarray(canonical_context["keep_mask"], dtype=bool)
        n_kept = int(np.sum(keep_mask))
        self._log(
            f"  Canonical trajectory context keeps {n_kept}/{n_cells} cells "
            f"(min_cells_number={min_cells_number})"
        )

        (
            features,
            _,
            _,
            original_adherence,
            class_relative_adherence_arr,
        ) = self._hybrid_compute_per_cell_features(
            cell_embeddings=np.asarray(cell_embeddings_scoring),
            node_embeddings=node_embeddings,
            predictions=predictions,
            pre_grit_score_matrix=predictive_score_matrix,
            active_indices=canonical_context["active_indices"],
            tree_edges=canonical_context["tree_edges"],
            keep_mask=keep_mask,
            temperature=temperature,
            top_k_neighbors=top_k_neighbors,
            class_relative_adherence=class_relative_adherence,
            class_relative_blend_weight=class_relative_blend_weight,
            node_names=node_names,
        )

        abnormality = self._hybrid_compute_per_cell_abnormality(
            features=features,
        )

        # --- Auxiliary diagnostic feature --------------------------------
        from .trajectory_support import _compute_jsd_knn_std
        jsd_knn_std = np.full(n_cells, np.nan, dtype=np.float64)
        if n_kept > 0:
            kept_emb = np.asarray(
                cell_embeddings_full[keep_mask], dtype=np.float32,
            )
            jsd_knn_std[keep_mask] = _compute_jsd_knn_std(
                cell_embeddings=kept_emb,
                jsd_values=features[keep_mask, 1],
                k=30,
            )

        # --- 1-D BGMM seed selection -------------------------------------
        bgmm_result = _fit_bgmm_aberrant_1d(
            abnormality,
            n_components=bgmm_n_components,
            min_weight=bgmm_min_weight,
            max_std=bgmm_max_std,
            train_subsample=bgmm_train_subsample,
            threshold=bgmm_threshold,
            random_state=random_seed,
            weight_concentration_prior=0.4,
            max_iter=1000,
            covariance_type="diag",
            reg_covar=1e-6,
        )
        is_aberrant = np.asarray(bgmm_result["is_aberrant"], dtype=bool)
        bgmm_component = np.asarray(
            bgmm_result["bgmm_component"], dtype=np.int64
        )
        bgmm_halted = bool(bgmm_result["bgmm_halted"])
        self._log(
            f"  BGMM 1-D seed: {int(is_aberrant.sum())} / {n_cells} "
            f"aberrant cells (threshold={float(bgmm_threshold):.2f}, "
            f"seed_components={bgmm_result['aberrant_component_indices']}, "
            f"bgmm_halted={bgmm_halted}, "
            f"n_fit_cells={int(bgmm_result['n_fit_cells'])})"
        )

        # --- SNN graph ---------------------------------------------------
        cuml_knn_fn = None
        if backend in {"auto", "gpu"}:
            def _cuml_knn_adapter(emb, k):
                return self._run_cuml_knn(emb, n_neighbors=k, verbose=False)
            cuml_knn_fn = _cuml_knn_adapter

        snn_result = _build_snn_graph_v2(
            cell_embeddings_full,
            n_neighbors=snn_n_neighbors,
            min_shared_fraction=snn_min_shared_fraction,
            backend=backend,
            cuml_knn_fn=cuml_knn_fn,
            random_state=random_seed,
            pynndescent_n_jobs=pynndescent_n_jobs,
            log_fn=self._log,
        )
        snn_adj = snn_result["snn_adj"]

        # --- Optional ontology-aware SNN edge pruning --------------------
        # Capture full-ontology state regardless of pruning so the LCA
        # routing decision and G5 rollup have access to the full DAG.
        full_state = self._build_visualization_state(use_full_ontology=True)
        full_node_names = list(full_state["node_names"])
        full_co_graph = _build_co_graph_from_directed_adj(
            full_state["pure_adjacency_matrix"],
            full_node_names,
        )
        full_name_to_idx = {
            name: i for i, name in enumerate(full_node_names)
        }
        active_to_full = np.full(len(node_names), -1, dtype=np.int64)
        for active_idx, name in enumerate(node_names):
            full_idx = full_name_to_idx.get(name, -1)
            active_to_full[active_idx] = full_idx
        if int(active_to_full.min()) < 0:
            missing = [
                name
                for i, name in enumerate(node_names)
                if active_to_full[i] < 0
            ]
            raise RuntimeError(
                "evaluate_cells: every active class must be "
                "present in the full ontology; "
                f"{len(missing)} were missing (e.g., {missing[:3]})."
            )
        predictions_full = active_to_full[predictions]

        prune_info: Optional[Dict[str, Any]] = None
        if ontology_prune_lineage_levels is not None:
            snn_adj, prune_info = _prune_snn_by_ontology_v1(
                snn_adj,
                predictions=predictions_full,
                co_graph=full_co_graph,
                node_names=full_node_names,
                lineage_levels=int(ontology_prune_lineage_levels),
                soft_power=float(ontology_prune_soft_power),
                log_fn=self._log,
            )

        # --- LCA routing decision ----------------------------------------
        lca_info = _lca_routed_predicted_classes_lca(
            full_co_graph, full_node_names, predictions_full,
        )
        is_multi_lineage = bool(lca_info["lca_is_root_only"])
        route = "multi_lineage" if is_multi_lineage else "single_lineage"
        # --- G1: auto-resolution Leiden + enrichment_fdr (always run) ----
        g1_scan = _lca_routed_scan_leiden_by_ari(
            snn_adj,
            predictions_full,
            resolution_range=tuple(g1_resolution_range),
            resolution_step=float(g1_resolution_step),
            backend=backend,
            random_seed=random_seed,
            log_fn=self._log,
        )
        g1_labels = np.asarray(g1_scan["community_labels"], dtype=np.int64)
        g1_sizes = np.asarray(g1_scan["community_size_per_cell"], dtype=np.int64)
        g1_resolution_chosen = float(g1_scan["best_resolution"])
        g1_ari_scores = dict(g1_scan["ari_scores"])

        g1_label_result = _label_communities_by_enrichment_fdr(
            g1_labels,
            g1_sizes,
            is_aberrant,
            adherence=None,
            jsd=None,
            fdr_alpha=fdr_alpha,
            effect_floor=effect_floor,
            min_community_size=int(min_community_size),
            log_fn=self._log,
        )
        g1_atypical = np.asarray(g1_label_result["is_atypical"], dtype=bool)
        g1_halted = bool(g1_label_result["halted"])
        g1_n_suspicious = int(g1_label_result["n_suspicious_communities"])
        g1_community_stats = g1_label_result["community_stats"]
        g1_community_frac_per_cell = np.asarray(
            g1_label_result["community_fraction_aberrant"], dtype=np.float64
        )
        g1_q_by_community = {
            int(row["label"]): float(row.get("hypergeom_q", float("nan")))
            for row in g1_community_stats
        }
        g1_community_hypergeom_q = np.array(
            [g1_q_by_community.get(int(c), float("nan")) for c in g1_labels],
            dtype=np.float64,
        )

        # --- G5: multi-resolution rollup (only when multi-lineage) -------
        g5_atypical = np.zeros(n_cells, dtype=bool)
        g5_halted = True  # vacuous halt when the path doesn't run
        g5_per_depth_summary: List[Dict[str, Any]] = []
        if is_multi_lineage:
            full_level_array = full_state.get("level_array")
            if g5_min_lca_level >= 1 and full_level_array is None:
                raise RuntimeError(
                    "evaluate_cells: g5_min_lca_level >= 1 requires "
                    "full_level_array on the loaded model — your model "
                    "checkpoint does not expose level_array. Re-export the "
                    "model with level_array, or pass g5_min_lca_level=0 to "
                    "disable the branch barrier."
                )
            g5_partitions = _lca_routed_build_g5_partitions(
                full_co_graph, full_node_names, predictions_full,
                d_cuts=tuple(g5_d_cuts),
                level_array=full_level_array,
                min_lca_level=int(g5_min_lca_level),
            )
            for part in g5_partitions:
                d_label_result = _label_communities_by_enrichment_fdr(
                    part["community_labels"],
                    part["community_size_per_cell"],
                    is_aberrant,
                    adherence=None,
                    jsd=None,
                    fdr_alpha=fdr_alpha,
                    effect_floor=effect_floor,
                    min_community_size=int(min_community_size),
                    log_fn=None,  # avoid 5 noisy log lines
                )
                d_atypical = np.asarray(
                    d_label_result["is_atypical"], dtype=bool,
                )
                g5_atypical |= d_atypical
                g5_per_depth_summary.append({
                    "d_cut": float(part["d_cut"]),
                    "n_communities": int(part["n_communities"]),
                    "n_suspicious_communities": int(
                        d_label_result["n_suspicious_communities"]
                    ),
                    "n_atypical": int(d_atypical.sum()),
                    "halted": bool(d_label_result["halted"]),
                })
            g5_halted = all(d["halted"] for d in g5_per_depth_summary)
            self._log(
                f"  [LCA-G5] multi-resolution rollup ..."
            )

        # --- Combine -----------------------------------------------------
        if is_multi_lineage:
            is_atypical = g1_atypical | g5_atypical
            halted = g1_halted and g5_halted
        else:
            is_atypical = g1_atypical.copy()
            halted = g1_halted

        # --- Per-class fairness ------------------------------------------
        # Symmetric complement to the per-community gate: a class whose
        # BGMM-aberrant count is statistically below dataset baseline
        # (left-tail hypergeometric + BH-FDR over predicted classes,
        # alpha shared with the community gate) cannot inherit a
        # bucket-level atypical verdict. Always-on, no opt-out.
        admissible = _per_class_fairness_mask(
            predictions_full, is_aberrant,
            fdr_alpha=float(fdr_alpha), log_fn=self._log,
        )
        is_atypical = is_atypical & admissible

        # BGMM-halt short-circuit: same semantics as evaluate_cells.
        if bgmm_halted:
            is_atypical = np.zeros(n_cells, dtype=bool)
            halted = True

        # Suspicious-community per-cell flag — derived from the union of
        # G1 and G5 atypical masks. (G5 doesn't carry community labels in
        # obs, so this is the cleanest per-cell summary; the per-depth
        # detail is in summary["g5_per_depth_summary"].)
        is_suspicious_community = is_atypical.copy()

        # n_suspicious_communities reported in the manifest summarises
        # G1's count (the obs partition) plus G5's per-depth counts.
        n_suspicious_total = g1_n_suspicious + sum(
            d["n_suspicious_communities"] for d in g5_per_depth_summary
        )

        self._log(
            f"  atypical={int(is_atypical.sum())} / {n_cells} "
            f"(route={route}, bgmm_halted={bgmm_halted}, "
            f"halted={halted}, "
            f"g1_n_susp={g1_n_suspicious}, "
            f"g5_per_depth_n_susp="
            f"{[d['n_suspicious_communities'] for d in g5_per_depth_summary]})"
        )

        # --- Write obs columns -------------------------------------------
        # Per-cell community columns come from G1 (the most representative
        # single partition); G5's per-depth structure is summary-only.
        adata.obs["abnormality"] = abnormality
        adata.obs["bgmm_component"] = bgmm_component
        adata.obs["is_aberrant"] = is_aberrant
        adata.obs["snn_community"] = g1_labels
        adata.obs["snn_community_size"] = g1_sizes
        adata.obs["community_fraction_aberrant"] = g1_community_frac_per_cell
        adata.obs["community_hypergeom_q"] = g1_community_hypergeom_q
        adata.obs["is_suspicious_community"] = is_suspicious_community
        adata.obs["is_atypical"] = is_atypical
        adata.obs["adherence_score"] = features[:, 0]
        adata.obs["jsd"] = features[:, 1]
        adata.obs["predicted_node_similarity"] = features[:, 2]
        adata.obs["predicted_class_similarity_margin"] = features[:, 3]
        adata.obs["original_adherence_score"] = original_adherence
        adata.obs["class_relative_adherence_score"] = class_relative_adherence_arr
        adata.obs["jsd_knn_std"] = jsd_knn_std

        # --- Diagnostic info dicts ---------------------------------------
        snn_info = {
            "backend_used": snn_result["backend_used"],
            "n_neighbors_effective": snn_result["n_neighbors_effective"],
            "nnz": int(snn_adj.nnz),
        }
        if prune_info is not None:
            snn_info["ontology_prune"] = prune_info
        bgmm_info = {
            "aberrant_component_indices": list(
                bgmm_result["aberrant_component_indices"]
            ),
            "threshold": float(bgmm_result["threshold"]),
            "n_fit_cells": int(bgmm_result["n_fit_cells"]),
            "min_weight": float(bgmm_min_weight),
            "max_std": float(bgmm_max_std),
            "n_components_max": int(bgmm_n_components),
            "component_report": list(bgmm_result["component_report"]),
        }

        manifest = self._build_atypical_manifest(
            prediction_fingerprint=prediction_fingerprint,
            atypical_fingerprint=atypical_fingerprint,
            community_stats=g1_community_stats,
            bgmm_info=bgmm_info,
            snn_info=snn_info,
            halted=halted,
            bgmm_halted=bgmm_halted,
            n_suspicious_communities=n_suspicious_total,
            fdr_alpha=float(fdr_alpha),
            effect_floor=float(effect_floor),
            min_community_size=int(min_community_size),
            bgmm_threshold=float(bgmm_threshold),
            route=route,
            lca=sorted(lca_info["lca"]),
            lca_is_root_only=is_multi_lineage,
            g1_resolution=g1_resolution_chosen,
            g1_ari_scores=g1_ari_scores,
            g5_per_depth_summary=g5_per_depth_summary,
        )
        self._write_atypical_manifest(adata, manifest)

        if diagnostic_output_path is not None:
            predicted_class_names = np.asarray(
                [
                    node_names[int(p)] if 0 <= int(p) < len(node_names)
                    else None
                    for p in predictions
                ],
                dtype=object,
            )
            _plot_atypical_lca_routed_diagnostic_grid(
                abnormality=abnormality,
                is_aberrant=is_aberrant,
                bgmm=bgmm_result["bgmm"],
                component_report=bgmm_result.get("component_report", []),
                aberrant_component_indices=list(
                    bgmm_result["aberrant_component_indices"]
                ),
                threshold=float(bgmm_threshold),
                bgmm_halted=bgmm_halted,
                predicted_class_names=predicted_class_names,
                community_stats=g1_community_stats,
                min_community_size=int(min_community_size),
                halted=halted,
                route=route,
                lca=sorted(lca_info["lca"]),
                lca_is_root_only=is_multi_lineage,
                g1_resolution=g1_resolution_chosen,
                g1_ari_scores=g1_ari_scores,
                g5_per_depth_summary=g5_per_depth_summary,
                output_path=diagnostic_output_path,
                boxplot_top_k=int(diagnostic_top_k_classes),
                boxplot_min_class_size=int(diagnostic_min_class_size),
            )

        self._log("✅ Cell evaluation complete!")
        core_result.close_transient_arrays()

        return self._build_result_from_adata(adata, manifest)

    # ============================================================================
    # Marker-gene discovery (attribution-based)
    # ============================================================================

    def _marker_gat_prototypes(self, use_full_ontology: bool):
        """Cached, L2-normalizable GAT prototypes [n_classes, dim] (float32).

        Input-independent, so the ontology GAT runs once instead of per batch.
        """
        cache = getattr(self, "_marker_proto_cache", None)
        if cache is None:
            cache = {}
            self._marker_proto_cache = cache
        if use_full_ontology not in cache:
            cache[use_full_ontology] = tf.cast(
                self.celltype_gat.get_gat_prototypes(
                    for_full_ontology=use_full_ontology, training=False),
                tf.float32)
        return cache[use_full_ontology]

    # NOTE: eager-only (uses tf.GradientTape + .numpy()); do not wrap in tf.function.
    def _marker_attribute(self, X, type_idx, *, k_neighbors, use_full_ontology,
                          batch_size, verbose):
        """Signed Input x Gradient of each cell's own-type generalist-cosine
        score, batched. Returns ``[n_cells, n_model_genes]`` (positive = pushes
        the cell toward its type; negative = away).
        """
        pn = tf.nn.l2_normalize(self._marker_gat_prototypes(use_full_ontology), axis=-1)

        def _knn(batch):
            Xb = tf.constant(batch, tf.float32)
            n = batch.shape[0]
            k = min(k_neighbors, n - 1)
            if k < 1:
                return tf.constant([[0], [0]], tf.int32), tf.ones((1,), tf.float32)
            Xn = tf.nn.l2_normalize(Xb, axis=1)
            _, idx = tf.math.top_k(tf.matmul(Xn, Xn, transpose_b=True), k=k)
            src = tf.repeat(tf.range(n), k)
            tgt = tf.reshape(idx, (-1,))
            ei = tf.stack([src, tgt], 0)
            return ei, tf.ones(tf.shape(ei)[1], tf.float32)

        def _attribute_batch(xb, tib):
            x = tf.constant(xb, tf.float32)
            ei, ew = _knn(xb)
            ti = tf.constant(tib, tf.int32)
            with tf.GradientTape() as tape:
                tape.watch(x)
                mu = self.vgae_encoder(
                    [x, tf.stop_gradient(ei), tf.stop_gradient(ew)], training=False)[4]
                z = tf.nn.l2_normalize(mu, axis=-1)
                s = tf.matmul(z, pn, transpose_b=True)
                r = tf.range(tf.shape(x)[0])
                own = tf.gather_nd(s, tf.stack([r, ti], axis=1))
                total = tf.reduce_sum(own)
            g = tape.gradient(total, x)
            return (x * g).numpy()

        out = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
        starts = list(range(0, X.shape[0], batch_size))
        if verbose:
            try:
                from tqdm import tqdm
                starts = tqdm(starts, desc="  IxG attribution", unit="batch")
            except ImportError:
                pass
        for s0 in starts:
            e0 = min(s0 + batch_size, X.shape[0])
            out[s0:e0] = _attribute_batch(X[s0:e0], type_idx[s0:e0])
        return out

    def _marker_attribute_mlx(self, X, type_idx, *, k_neighbors,
                               use_full_ontology, batch_size, verbose):
        """MLX-native Input×Gradient attribution, mirroring _marker_attribute."""
        import mlx.core as mx

        encoder = self._mlx_encoder
        proto_np = self._mlx_celltype_embedder.get_gat_prototypes(
            for_full_ontology=use_full_ontology, normalize=True)
        proto_mx = mx.array(proto_np.astype(np.float32))

        def _knn_np(batch):
            n = batch.shape[0]
            k = min(k_neighbors, n - 1)
            if k < 1:
                src = np.array([0, 0], dtype=np.int32)
                tgt = np.array([0, 0], dtype=np.int32)
                return np.stack([src, tgt], axis=0)
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            xn = batch / np.maximum(norms, 1e-10)
            sim = xn @ xn.T
            idx = np.argsort(-sim, axis=1)[:, :k]
            src = np.repeat(np.arange(n, dtype=np.int32), k)
            tgt = idx.ravel().astype(np.int32)
            return np.stack([src, tgt], axis=0)

        def _attribute_batch(xb, tib):
            ei_np = _knn_np(xb)
            n = xb.shape[0]
            x_mx = mx.array(xb)
            ei_mx = mx.array(ei_np)
            ti_mx = mx.array(tib)

            def _score_fn(x):
                mu, _ = encoder(x, ei_mx, n)
                z = mu / (mx.linalg.norm(mu, axis=-1, keepdims=True) + 1e-10)
                s = z @ proto_mx.T
                return mx.sum(s[mx.arange(n), ti_mx])

            grad_fn = mx.grad(_score_fn)
            g = grad_fn(x_mx)
            ixg = x_mx * g
            mx.eval(ixg)
            return np.array(ixg, dtype=np.float32)

        out = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
        starts = list(range(0, X.shape[0], batch_size))
        if verbose:
            try:
                from tqdm import tqdm
                starts = tqdm(starts, desc="  IxG attribution (MLX)",
                              unit="batch")
            except ImportError:
                pass
        for s0 in starts:
            e0 = min(s0 + batch_size, X.shape[0])
            out[s0:e0] = _attribute_batch(X[s0:e0], type_idx[s0:e0])
        return out

    def find_marker_genes(self, adata, label_key, *, sample_key=None,
                          cell_types=None, confidence="auto", top_n=100,
                          fdr_alpha=None, return_all=False, batch_size=256,
                          k_neighbors=30, use_full_ontology=True, verbose=True):
        """Rank each cell type's model genes by attribution-based importance.

        Signed Input x Gradient -> per-type signed mean, directional specificity
        ``spec`` (the ranking metric), and a clipped contrast. ``confidence`` is
        an across-sample test when ``sample_key`` is given and a type has >= 2
        qualifying samples, else the within-sample one-vs-rest test (recorded in
        ``confidence_basis``). Output is contained by ``spec > 0`` + ``top_n``
        (set ``return_all=True`` for the full ranked list). Covers genes represented
        by the loaded checkpoint.

        Args:
            adata: AnnData with raw counts in a resolvable slot and Ensembl gene
                IDs detectable in ``var``.
            label_key: ``adata.obs`` column whose values are the model's class IDs
                (e.g. 'CL:0000623') or their readable names (e.g. 'natural killer
                cell'); names are resolved via the model's id_to_name_map.
            sample_key: optional ``adata.obs`` column naming the replicate unit
                (donor/patient); enables the across-sample confidence.
            cell_types: optional list restricting which types are reported (the
                specificity comparison field stays all present types).
            confidence: ``'auto'`` (across-sample when possible), ``'within_sample'``,
                or ``'permutation'``.
            top_n: per-type row cap on the contained output (None = no cap).
            fdr_alpha: optional BH-FDR cutoff applied within each type's spec>0 set.
            return_all: return the full ranked list with no containment.
            batch_size / k_neighbors / use_full_ontology / verbose: runtime knobs.

        Returns:
            DataFrame with one block per reported type; columns: cell_type,
            cell_type_name, gene_id, gene_name, attribution_score, spec,
            contrast_score, confidence, confidence_basis, sample_reproducibility,
            rank.
        """
        if self.vgae_encoder is None and self._mlx_encoder is None:
            raise RuntimeError(
                "No backend available for attribution: both vgae_encoder (TF) "
                "and _mlx_encoder (MLX) are None.")
        use_mlx = self.vgae_encoder is None
        from scipy.sparse import issparse
        from .predictor_support import (
            aggregate_attributions, onevsrest_confidence, permutation_confidence,
            across_sample_confidence, combine_across_samples, assemble_marker_table,
            contain_markers, gene_symbols_for_ids, _find_ensembl_id_array_in_var,
        )
        _p = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)

        classes = list(self.celltype_gat.full_classes if use_full_ontology
                       else self.celltype_gat.classes)
        cls_index = {c: i for i, c in enumerate(classes)}
        # Accept class IDs, readable names, or "name (ID)" (via _parse_ontology_id).
        raw_labels = np.asarray(adata.obs[label_key].values, dtype=str)
        resolved = self._resolve_class_labels(raw_labels, cls_index.keys())
        labels = np.array([r if r is not None else "" for r in resolved], dtype=str)
        samples = (np.asarray(adata.obs[sample_key].values, dtype=str)
                   if sample_key is not None else None)

        # Attribute over ALL cells of model-known types so specificity is judged
        # against the full competitive field; cell_types filters the REPORTED
        # output later, not the attribution/aggregation set.
        valid = labels != ""
        rows = np.where(valid)[0]
        if rows.size == 0:
            ex = list(dict.fromkeys(raw_labels.tolist()))[:5]
            raise ValueError(
                f"No cells' '{label_key}' values match the model's classes.\n"
                f"  example labels:  {ex}\n  example classes: {classes[:5]}\n"
                "Provide an obs column of model class IDs (e.g. 'CL:0000084') "
                "or their readable names (e.g. 'natural killer cell').")
        labels_v = labels[rows]
        type_idx = np.array([cls_index[l] for l in labels_v], dtype=np.int32)

        X_model = self._prepare_expression(adata)
        if issparse(X_model):
            X_model = X_model.toarray()
        X_model = np.asarray(X_model, dtype=np.float32)[rows]

        _p(f"  [markers] attributions: {len(rows)} cells, "
           f"{len(set(labels_v))} types...")
        _attr_fn = (self._marker_attribute_mlx if use_mlx
                    else self._marker_attribute)
        attr = _attr_fn(
            X_model, type_idx, k_neighbors=k_neighbors,
            use_full_ontology=use_full_ontology, batch_size=batch_size,
            verbose=verbose)

        _p(f"  [markers] aggregating + confidence ({confidence})...")
        agg = aggregate_attributions(attr, labels_v)

        asp, nsamp = ({}, {})
        if samples is not None and confidence == "auto":
            asp, nsamp = across_sample_confidence(attr, labels_v, samples[rows])
        if confidence == "permutation":
            wp = permutation_confidence(attr, labels_v, score="mean")
            within_basis = "permutation"
        else:
            wp = onevsrest_confidence(attr, labels_v)
            within_basis = "within_sample"
        conf, conf_basis = {}, {}
        for t in agg:
            if samples is not None and confidence == "auto" and nsamp.get(t, 0) >= 2:
                conf[t] = asp[t]
                conf_basis[t] = f"across_sample(n={nsamp[t]})"
            else:
                conf[t] = wp[t]
                conf_basis[t] = within_basis

        rep = None
        if samples is not None:
            sv = samples[rows]
            per_sample = {}
            for s in np.unique(sv):
                sm = sv == s
                agg_s = aggregate_attributions(attr[sm], labels_v[sm])
                per_sample[s] = {t: agg_s[t]["spec"] for t in agg_s}
            rep = combine_across_samples(per_sample, thresh=0.0)

        _p("  [markers] assembling + containing...")
        gene_ids = [str(g) for g in self.gene_ids]
        var = adata.var
        if (getattr(adata, "raw", None) is not None
                and _find_ensembl_id_array_in_var(var) is None):
            var = adata.raw.var
        gene_names = gene_symbols_for_ids(gene_ids, var)
        name_map = getattr(self, "id_to_name_map", None) or {}
        df = assemble_marker_table(agg, conf, conf_basis, rep, gene_ids,
                                   gene_names, name_map)
        if cell_types is not None:
            resolved_ct = self._resolve_class_labels(cell_types, cls_index.keys())
            want = set(r for r in resolved_ct if r is not None)
            df = df[df["cell_type"].isin(want)].reset_index(drop=True)
            if df.empty:
                present = sorted(set(labels_v))[:5]
                raise ValueError(
                    f"None of cell_types={list(cell_types)} are present/resolvable "
                    f"under '{label_key}'. Example present types: {present}")
        return contain_markers(df, spec_min=0.0, top_n=top_n,
                               fdr_alpha=fdr_alpha, return_all=return_all)

    def find_gene_networks(self, adata, markers_df, label_key, *, top_k=300,
                           correlation_threshold=0.3, leiden_resolution=1.0,
                           min_module_size=3, use_gprofiler=True, organism=None,
                           gprofiler_sources=None, go_gene_sets=None,
                           reactome_gene_sets=None, fdr_threshold=0.05):
        """Co-expression modules + pathway enrichment over each type's top markers.

        For each cell type in ``markers_df``, take the top ``top_k`` ranked genes,
        build a Spearman co-expression graph on the resolved count-like expression,
        modules (Leiden -> connected-components fallback), and annotate each module
        with pathway enrichment (g:Profiler online -> local GMT -> none). Model
        gene IDs in ``markers_df`` are matched to the data's count columns with
        Ensembl version suffixes stripped on both sides.

        Args:
            adata: AnnData with raw counts in a resolvable slot and Ensembl gene
                IDs detectable in ``var`` (index or a column).
            markers_df: the DataFrame returned by :meth:`find_marker_genes`.
            label_key: ``adata.obs`` column giving each cell's type (same values
                used in ``markers_df['cell_type']``).
            top_k: number of top-ranked genes per type to build the network on.
            correlation_threshold: absolute Spearman cutoff for an edge.
            leiden_resolution: higher = more, smaller modules.
            min_module_size: minimum genes per reported module.
            use_gprofiler: try online g:Profiler enrichment first.
            organism: g:Profiler organism; auto-detected from gene IDs if None.
            gprofiler_sources: e.g. ``['GO:BP', 'REAC', 'KEGG']``.
            go_gene_sets / reactome_gene_sets: local GMT fallback gene sets.
            fdr_threshold: enrichment significance threshold.

        Returns:
            ``{cell_type: {'edge_list', 'modules'} or None}`` where each module
            carries ``gene_indices``, ``gene_ids``, ``gene_names``,
            ``attribution_mass`` and ``enriched_terms``.
        """
        from scipy.sparse import issparse
        from .predictor_support import (
            GeneNetworkBuilder, PathwayAnnotator, GProfilerAnnotator,
            resolve_expression_source, _find_ensembl_id_array_in_var,
            _strip_ensembl_version, gene_symbols_for_ids,
        )
        # Accept class IDs OR readable names so label_key matches markers_df cell_type ids.
        # Derive valid IDs from the model's GAT (when loaded) and from markers_df itself.
        gat = getattr(self, "celltype_gat", None)
        valid_class_ids = set(markers_df["cell_type"].astype(str).unique())
        if gat is not None:
            valid_class_ids |= set(getattr(gat, "full_classes", []))
            valid_class_ids |= set(getattr(gat, "classes", []))
        resolved = self._resolve_class_labels(
            np.asarray(adata.obs[label_key].values, dtype=str), valid_class_ids)
        labels = np.array([r if r is not None else "" for r in resolved], dtype=str)
        sel = resolve_expression_source(adata, allow_log1p=True, logger=None)
        counts = sel["matrix"]
        if sel.get("use_backcalc", False):
            if issparse(counts):
                counts = counts.copy().astype(np.float64)
                counts.data = np.expm1(np.clip(counts.data, 0.0, 10.0))
            else:
                counts = np.expm1(np.clip(np.asarray(counts, np.float64), 0.0, 10.0))
        else:
            counts = (counts.astype(np.float64) if issparse(counts)
                      else np.asarray(counts, dtype=np.float64))
        counts = counts.tocsr() if issparse(counts) else counts
        ids = _find_ensembl_id_array_in_var(sel["var"])
        if ids is None:
            ids = np.asarray(sel["var"].index.astype(str))
        counts_gene_ids = [str(g) for g in ids]
        # Same symbol resolution as find_marker_genes: the array holding symbols
        # is found by agreement with the packaged table, not by column name.
        sym = dict(zip(counts_gene_ids,
                       gene_symbols_for_ids(counts_gene_ids, sel["var"])))

        builder = GeneNetworkBuilder(
            top_k_genes=top_k, correlation_threshold=correlation_threshold,
            leiden_resolution=leiden_resolution, min_module_size=min_module_size)
        gp = None
        if use_gprofiler:
            org = organism or ("mmusculus"
                               if (counts_gene_ids and counts_gene_ids[0].startswith("ENSMUSG"))
                               else "hsapiens")
            gp = GProfilerAnnotator(organism=org, sources=gprofiler_sources,
                                    fdr_threshold=fdr_threshold)
            if not gp._ensure_client():
                gp = None
        local = None
        if gp is None and (go_gene_sets is not None or reactome_gene_sets is not None):
            local = PathwayAnnotator(go_gene_sets=go_gene_sets,
                                     reactome_gene_sets=reactome_gene_sets,
                                     fdr_threshold=fdr_threshold)

        id_to_col = {_strip_ensembl_version(g): i for i, g in enumerate(counts_gene_ids)}
        networks = {}
        for t, grp in markers_df.groupby("cell_type", sort=False):
            cells = labels == t
            if int(cells.sum()) < 3:
                networks[t] = None
                continue
            top = grp.sort_values("rank").head(top_k)
            cols, scores = [], []
            for _, row in top.iterrows():
                ci = id_to_col.get(_strip_ensembl_version(str(row["gene_id"])))
                if ci is None:
                    continue
                s = row.get("attribution_score", 0.0)
                cols.append(ci)
                scores.append(abs(float(s)) if np.isfinite(s) else 0.0)
            if len(cols) < min_module_size:
                networks[t] = None
                continue
            cols = np.array(cols)
            importance = np.zeros(len(counts_gene_ids))
            importance[cols] = scores
            net = builder.build_network(counts[cells], cols, importance)
            if net and net.get("modules"):
                bg_ids = [counts_gene_ids[c] for c in cols]
                for m in net["modules"]:
                    gi = m["gene_indices"]
                    m["gene_ids"] = [counts_gene_ids[g] for g in gi]
                    m["gene_names"] = [sym.get(counts_gene_ids[g],
                                               counts_gene_ids[g]) for g in gi]
                    if gp is not None:
                        m["enriched_terms"] = gp.annotate_module(m["gene_ids"], bg_ids)
                    elif local is not None:
                        m["enriched_terms"] = local.annotate_module(
                            m["gene_names"], [sym.get(i, i) for i in bg_ids])
                    else:
                        m["enriched_terms"] = []
            networks[t] = net
        return networks

    # ============================================================================
    # ============================================================================
    # Helpers shared by evaluate_cells
    # ============================================================================

    def _hybrid_compute_per_cell_features(self, *args, **kwargs):
        return _predictor_support._hybrid_compute_per_cell_features(self, *args, **kwargs)


    @staticmethod
    def _hybrid_compute_per_cell_abnormality(*args, **kwargs):
        return _predictor_support._hybrid_compute_per_cell_abnormality(*args, **kwargs)





# Expose convenience helpers on the public predictor surface.
_predictor_support.HECTOR = HECTOR
_predictor_support.InferenceConfig = InferenceConfig
rollup_rare_annotations = _predictor_support.rollup_rare_annotations
predict_from_anndata = _predictor_support.predict_from_anndata
diagnose_cellranger_directory = _predictor_support.diagnose_cellranger_directory
load_cellranger_data = _predictor_support.load_cellranger_data
_ATYPICAL_FEATURE_NAMES = _predictor_support._ATYPICAL_FEATURE_NAMES
