"""Ontology graph infrastructure, layout engines, coloring, and simplification.

Contains: elastic river layout, soft edge discovery, ontology graph utilities
(SimpleOBOParser, AdaptiveOntologySubgraph, DAGToTreeConverter), layout engines
(TreeLayoutEngine, RadialTreeLayoutEngine), adaptive lineage coloring,
TrajectoryAnalysisConfig, OntologySimplifier, and the simplify/plot APIs.
"""

import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

# Configure module logger
logger = logging.getLogger(__name__)



# Configuration, graph utilities, layout engines, color handling, and
# ontology simplification.

@dataclass
class TrajectoryAnalysisConfig:
    """
    Configuration for the hierarchical trajectory analysis.
    
    Adaptive lineage coloring groups active cell types with a top-down set-cover
    procedure over the ontology DAG, assigns colors to the resulting group roots,
    and propagates each color to its represented descendants. This is the
    supported lineage-coloring method.
    
    - adaptive_color_palette: Custom list of hex colors for branch assignment.
      If None, uses the default ADAPTIVE_PALETTE (10 visually distinct colors).
    """
    
    # Layout parameters
    y_scale: float = 1  # Vertical spacing multiplier (higher = more separate paths)
    x_scale: float = 3.5  # Horizontal spacing multiplier (higher = more space between levels)    
    # Barycentric placement parameters
    temperature: float = 0.1  # Softmax temperature for barycentric weights (lower = sharper)
    
    # Visual styling
    node_size: int = 4
    cell_size: int = 1  # Size of cell markers
    cell_opacity: float = 0.5  # Opacity for transitioning cells
    edge_width: float = 1.5  # Width of tree edges
    
    # Trajectory mode control
    show_transitions: bool = True  # If True (default), transitioning cells are rendered as dots on edges
                                   # and clouds contain only stable cells. If False, all cells stay in
                                   # their predicted node's cloud (raw prediction mode) — no edge dots.
    
    # Contour settings
    display_contours: bool = True  # Show density contours for stable cells
    contour_bandwidth: Optional[float] = None  # KDE bandwidth (None = auto via Scott's rule)
    
    # Ontology source
    ontology_path: Optional[str] = None  # Path to cl.obo file. If provided, overrides model adjacency.
    
    # Trajectory Analysis Parameters (Stable vs Transitioning)
    # Minimum lateral spread of transitioning cells on edges, as a sigma of
    # ``value * 15`` display pixels. Branch width sets the spread on thick
    # branches; this floors it on thin distal ones, which are narrower than the
    # cell markers themselves. Raise for a broader stream, lower to hug the
    # branch centreline.
    transition_stream_width: float = 0.5
    top_k_neighbors: int = 15  # Top-K masking for robustness (auto-adapts to atlas size)
    # Stable zones flank the [0, 1] developmental arc: cells with t_raw <=
    # stable_source_zone are Stable at the source; cells with t_raw >=
    # 1 - stable_target_zone are Stable at the target; everything in between is
    # Transitioning. The two zones are independent so users can express
    # asymmetric biology along the source→target arc.
    # ``None`` (the default) means "auto-detect via BGMM in
    # ``LatentBarycentricPlacer.place_cells``"; a numeric value overrides
    # auto-detection on that side. Either side can be ``None`` independently.
    stable_source_zone: Optional[float] = None
    stable_target_zone: Optional[float] = None
    
    # Count-scaled clouds interpolate between configured minimum and maximum sizes.
    # Cloud radii are expressed in display pixels under the pixel-space sizing
    # model, so clouds render isotropically regardless of axis aspect.
    scale_cloud_by_count: bool = True  # If True, stable cloud area scales with per-node stable cell counts
    cloud_scale_exponent: float = 1  # Higher = larger small clouds after count normalization
    cloud_size_multiplier: float = 1.0  # Global post-multiplier applied on top of the lerp result.
    cloud_size_min: float = 0.35         # Cloud radius multiplier at the smallest-count node
                                        # (also used for below-threshold counts). Lerp lower bound.
    cloud_size_max: float = 1.2         # Cloud radius multiplier at the largest-count node. Lerp upper bound.
    contour_min_cells: int = 30         # Minimum stable-cell count required for contour/cloud rendering
    stable_scatter_jitter_scale: float = 0.25  # Contract stable scatter offsets when contours are not used. Also applied to highlight scatter per-anchor so dots stay distinguishable at low cell counts.
    
    # Filtering
    min_cells_number: int = 20        # Minimum absolute cell count per node (default: 20 cells)
                                      # Nodes with fewer cells are excluded from visualization
    
    # Title
    title: str = "HECTOR: Trajectory Analysis"
    
    # PDF-specific settings
    pdf_dpi: int = 400 # Higher DPI for publication
    pdf_figsize: Tuple[int, int] = (18, 10) 
    label_font_size: int = 8  # Font size for node labels (points)
    show_cloud_cell_counts: bool = True  # Display "n=X" count labels on cloud nodes

    # HTML-specific settings
    html_plot_width: int = 1800   # Width of the HTML Plotly canvas in pixels
    html_plot_height: int = 1200  # Height of the HTML Plotly canvas in pixels
    
    # Atypical-cell clustering (HDBSCAN + anchor-based placement)
    # NOTE: Atypical-cell clustering creates pie charts. It should only run when atypical_clusters=True
    # to prevent pie charts from appearing when atypical cells should be merged into stable/transitioning.
    enable_atypical_clustering: bool = True  # Enable atypical-cell clustering (HDBSCAN-based)
    atypical_anchor_count: int = 3       # Number of anchor nodes for atypical-cluster hub positioning (3-6 typical)
    
    # Two-Level Atypical Cell Control
    # Level 1: Detection - Controls whether to run BGMM classification to detect atypical cells
    enable_atypical_detection: bool = False  # If False, skip BGMM - all cells classified as Stable/Transitioning only
    # Level 2: Visualization - Controls how detected atypical cells are displayed (only matters if detection is enabled)
    atypical_clusters: bool = False  # Cluster atypical cells and render each cluster as a pie chart
    atypical_scatter: bool = False  # Highlight atypical cells by anchor cell type
    
    # Highlight overlay controls
    highlight_obs: Optional[str] = None  # adata.obs column name for custom cell highlighting
    highlight_groups: Optional[List[str]] = None  # Subset of categories within highlight_obs to highlight
    highlight_palette: Optional[List[str]] = None  # Custom color palette for HighlightColorMapper

    # Diagnostic plot: when True, saves a PNG showing the routed
    # evaluate_cells diagnostic grid at the path derived from ``output_file``.
    save_diagnostic_plot: bool = True


    # Pie Chart Visualization (for atypical clusters)
    show_pie_charts: bool = True                # Show pie charts instead of dot clouds
    pie_chart_edge_color: str = 'white'         # Edge color between pie slices
    pie_chart_edge_width: float = 0.5           # Edge width between pie slices
    
    # Layout mode: 'horizontal' (left-to-right tree) or 'radial' (center-out circular tree)
    layout_mode: str = 'horizontal'
    
    # Adaptive lineage coloring
    # Uses LCA-based algorithm to dynamically determine coloring based on dataset composition
    # This is the only supported coloring method (always enabled)
    adaptive_color_palette: Optional[List[str]] = None  # Custom color list (None = use ADAPTIVE_PALETTE)
    
    # Radial layout specific parameters
    radial_max_radius: float = 12     # Outer radius of the tree
    radial_start_width: float = 14.0  # Starting width at root for dynamic tapering
    radial_min_width: float = 0.8     # Minimum width at leaves for dynamic tapering
    radial_sweep_angle_degrees: Optional[float] = None  # Manual override for total sweep angle (degrees). None = auto-adaptive
    radial_html_branch_mode: Literal['auto', 'raster', 'vector'] = 'auto'  # Rasterize static radial HTML branches for faster browser rendering
    
    # ==========================================================================
    # Class-relative adherence scoring
    # ==========================================================================
    class_relative_adherence: bool = True       # Enable class-relative adherence blending
    class_relative_blend_weight: float = 0.4    # Blend weight (0.0 = original only, 1.0 = class-relative only)

    # ==========================================================================
    # HECTOR Elastic River Configuration
    # ==========================================================================
    # Inferred-path discovery toggle. Shared parameters are defined below.
    enable_inferred_paths: bool = False          # Shortcut links (inferred paths) are opt-in; family-tree is the default
    
    # Elastic Layout: Continuous depths based on latent pseudotime
    enable_elastic_layout: bool = False       # Enable elastic accordion layout
    elasticity_factor: float = 0.75           # 0.0=rigid grid, 1.0=pure pseudotime
    elastic_max_deviation: float = 0.4       # Max drift from layer anchor
    elastic_min_layer_gap: float = 1.0       # Minimum separation between layers
    elastic_min_node_dist: float = 0.25      # Causality enforcement (child > parent)
    
    # Soft Edge Visualization (visual styling only)
    inferred_path_min_cells: int = 20            # Minimum transitioning cells required to show a inferred path
    inferred_path_gate1_blob_ratio: float = 1.5  # Gate 1: blob if edge t-IQR < ratio * within-type null

    # Ribbon rendering parameters
    ribbon_taper_ratio: float = 0.6       # end_width / start_width, clamped to [0.3, 1.0]
    ribbon_opacity: float = 0.7           # Center-line base opacity for ribbon fill
    cloud_blend_enabled: bool = True      # Toggle cloud-ribbon terminus blending

    def __post_init__(self):
        """Validate configuration."""
        self.ribbon_taper_ratio = max(0.3, min(1.0, self.ribbon_taper_ratio))

        # Validate min_cells_number: must be a positive integer (no floats, no booleans, no zero/negative)
        val = self.min_cells_number
        is_real_number = isinstance(val, (int, float, np.integer, np.floating))
        has_fractional_part = isinstance(val, (float, np.floating)) and val != int(val)
        if isinstance(val, bool) or not is_real_number or has_fractional_part or int(val) < 1:
            raise ValueError(
                f"min_cells_number must be a positive integer (got {val!r}). "
                "Example: min_cells_number=20"
            )
        self.min_cells_number = int(val)

        scatter_scale = self.stable_scatter_jitter_scale
        is_real_scale = isinstance(scatter_scale, (int, float, np.integer, np.floating))
        if (
            isinstance(scatter_scale, bool)
            or not is_real_scale
            or not np.isfinite(float(scatter_scale))
            or float(scatter_scale) <= 0
        ):
            raise ValueError(
                "stable_scatter_jitter_scale must be a positive number "
                f"(got {scatter_scale!r})."
            )
        self.stable_scatter_jitter_scale = float(scatter_scale)

        # ``None`` is the auto-detect sentinel — it bypasses the range
        # and sum guards here. ``LatentBarycentricPlacer.place_cells``
        # resolves None to a concrete value via BGMM and re-applies the
        # ``src + tgt < 1.0`` guard (with warn-and-fall-back) once both
        # sides are concrete.
        for name, val in (
            ("stable_source_zone", self.stable_source_zone),
            ("stable_target_zone", self.stable_target_zone),
        ):
            if val is None:
                continue
            is_real = isinstance(val, (int, float, np.integer, np.floating))
            if isinstance(val, bool) or not is_real or not (0.0 <= float(val) < 1.0):
                raise ValueError(
                    f"{name} must be None (auto-detect) or a number in "
                    f"[0, 1) (got {val!r})."
                )
        if (
            self.stable_source_zone is not None
            and self.stable_target_zone is not None
            and self.stable_source_zone + self.stable_target_zone >= 1.0
        ):
            raise ValueError(
                "stable_source_zone + stable_target_zone must be < 1.0 "
                f"(got {self.stable_source_zone} + {self.stable_target_zone}); "
                "otherwise the Transitioning band collapses."
            )
        if self.stable_source_zone is not None:
            self.stable_source_zone = float(self.stable_source_zone)
        if self.stable_target_zone is not None:
            self.stable_target_zone = float(self.stable_target_zone)


# Shared inferred-path parameters used by SoftEdgeConfig defaults.
INFERRED_PATH_EMBEDDING_WEIGHT = 0.45   # beta: cosine (transcriptomic similarity) weight
INFERRED_PATH_ONTOLOGY_WEIGHT = 0.20    # gamma: ontology-distance prior weight
INFERRED_PATH_DECAY_CONSTANT = 2.0      # lambda: exp(-hops/lambda) ontology-distance decay
INFERRED_PATH_MIN_SUMMED_SCORE = 0.28   # summed-score floor


# NOTE: trajectory_support imports are deferred to avoid a circular import
# (trajectory_ontology -> trajectory_support -> trajectory_ontology).
# Each function/method that needs these symbols does a local import.

# =============================================================================
# Performance Caching
# =============================================================================

class VisualizationCache:
    """Cache expensive computations for reuse across visualization calls.

    Caches ontology distance matrices (O(n³)) and embedding similarity
    matrices (O(n² × d)) so that repeated calls to
    ``discover_inferred_paths`` — e.g. from both ``visualize()`` and
    ``simplify_ontology_tree()`` on the same data — avoid redundant work.

    Not thread-safe.  Use one instance per thread or add external locking.
    """

    def __init__(self) -> None:
        """Initialize empty cache."""
        self._ontology_distances: Optional[np.ndarray] = None
        self._ontology_distances_hash: Optional[int] = None
        self._embedding_similarity: Optional[np.ndarray] = None
        self._embedding_hash: Optional[int] = None
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        logger.debug("VisualizationCache initialized")

    def _compute_array_hash(self, arr: np.ndarray) -> int:
        """Compute a hash for numpy array to detect changes."""
        return hash((arr.shape, arr.dtype.str, arr.tobytes()[:1000]))

    def get_ontology_distances(
        self,
        pure_ontology_adj: np.ndarray,
        force_recompute: bool = False
    ) -> np.ndarray:
        """Get cached ontology distances or compute if not cached."""
        current_hash = self._compute_array_hash(pure_ontology_adj)

        if (not force_recompute and
            self._ontology_distances is not None and
            self._ontology_distances_hash == current_hash):
            self._cache_hits += 1
            logger.debug("Cache hit for ontology distances")
            return self._ontology_distances

        self._cache_misses += 1
        logger.debug("Cache miss for ontology distances, computing...")

        start_time = time.time()
        self._ontology_distances = compute_ontology_distances(pure_ontology_adj)
        self._ontology_distances_hash = current_hash
        elapsed = time.time() - start_time

        logger.info(f"Ontology distances computed in {elapsed:.3f}s "
                   f"(shape: {self._ontology_distances.shape})")

        return self._ontology_distances

    def get_embedding_similarity(
        self,
        node_embeddings: np.ndarray,
        force_recompute: bool = False
    ) -> np.ndarray:
        """Get cached embedding similarity matrix or compute if not cached."""
        current_hash = self._compute_array_hash(node_embeddings)

        if (not force_recompute and
            self._embedding_similarity is not None and
            self._embedding_hash == current_hash):
            self._cache_hits += 1
            logger.debug("Cache hit for embedding similarity")
            return self._embedding_similarity

        self._cache_misses += 1
        logger.debug("Cache miss for embedding similarity, computing...")

        start_time = time.time()

        norms = np.linalg.norm(node_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_norm = node_embeddings / norms

        self._embedding_similarity = embeddings_norm @ embeddings_norm.T
        self._embedding_hash = current_hash
        elapsed = time.time() - start_time

        logger.info(f"Embedding similarity computed in {elapsed:.3f}s "
                   f"(shape: {self._embedding_similarity.shape})")

        return self._embedding_similarity

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        }


def compute_ontology_distances(pure_ontology_adj: np.ndarray) -> np.ndarray:
    """
    Compute shortest path distances on undirected ontology graph.

    Symmetrizes the directed IS_A hierarchy (child->parent) and computes
    all-pairs shortest paths, enabling bidirectional transitions for inferred
    path weighting.

    Args:
        pure_ontology_adj: [n_nodes, n_nodes] directed adjacency matrix representing
            the pure IS_A hierarchy (child->parent edges have value 1.0)
    
    Returns:
        distances: [n_nodes, n_nodes] shortest path distances (undirected).
            - Distance from node i to node j is distances[i, j]
            - Self-distances (diagonal) are 0
            - Disconnected components have distance = infinity (np.inf)
    
    Raises:
        ValueError: If input is not a 2D square matrix
    
    Notes:
        The dense result requires ``O(n_nodes²)`` memory. SciPy selects the
        shortest-path algorithm automatically for the supplied graph.
    """
    # Input validation
    if pure_ontology_adj.ndim != 2:
        raise ValueError(f"Expected 2D array, got {pure_ontology_adj.ndim}D")
    if pure_ontology_adj.shape[0] != pure_ontology_adj.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {pure_ontology_adj.shape}")
    
    n_nodes = pure_ontology_adj.shape[0]
    logger.debug(f"Computing ontology distances for {n_nodes} nodes")
    
    # Convert to undirected by symmetrizing the adjacency matrix
    # This allows traversal in both directions (child->parent and parent->child)
    adj_undirected = pure_ontology_adj + pure_ontology_adj.T
    
    # Binarize: any edge (regardless of original direction) becomes 1
    adj_undirected = (adj_undirected > 0).astype(float)
    
    # Compute all-pairs shortest paths using Floyd-Warshall
    # directed=False ensures we treat the graph as undirected
    # unweighted=True means all edges have weight 1 (hop count)
    distances = shortest_path(
        csgraph=adj_undirected,
        directed=False,
        unweighted=True
    )
    
    # Log statistics about distances
    finite_distances = distances[np.isfinite(distances)]
    if len(finite_distances) > 0:
        logger.debug(f"Distance statistics: min={np.min(finite_distances):.1f}, "
                    f"max={np.max(finite_distances):.1f}, "
                    f"mean={np.mean(finite_distances):.2f}")
    
    # Count disconnected pairs
    n_disconnected = np.sum(np.isinf(distances))
    if n_disconnected > 0:
        logger.warning(f"Found {n_disconnected} disconnected node pairs "
                      f"({n_disconnected / (n_nodes * n_nodes) * 100:.1f}%)")
    
    return distances


def _terminal_sibling_mask(pure_adj: np.ndarray) -> np.ndarray:
    """[n,n] bool: True where i,j are both terminal (leaf) AND share an immediate parent.

    ``pure_adj`` is the binary IS-A tree with ``pure_adj[child, parent] = 1``. Grouped by parent so the
    cost is sum-of-(siblings^2), not n^2.
    """
    from collections import defaultdict
    n = pure_adj.shape[0]
    is_terminal = (pure_adj.sum(axis=0) == 0)          # no children -> leaf
    out = np.zeros((n, n), dtype=bool)
    by_parent = defaultdict(list)
    for c in np.where(is_terminal)[0]:
        for p in np.where(pure_adj[c] > 0)[0]:
            by_parent[p].append(int(c))
    for kids in by_parent.values():
        for a in kids:
            for b in kids:
                if a != b:
                    out[a, b] = True
    return out



@dataclass
class SoftEdgeConfig:
    """
    Configuration for inferred path discovery.

    Inferred paths are data-driven transitions that bypass the rigid ontology hierarchy,
    discovered using a two-component weighted combination:

    inferred_path_weight = embedding_weight * embedding_similarity + ontology_weight * ontology_proximity_bonus

    Model-graph membership is a hard eligibility gate. A data-derived Otsu floor
    on cosine similarity is applied by :func:`discover_inferred_paths`.

    Attributes:
        embedding_weight: β - Weight for embedding similarity (cosine similarity).
            Higher values favor edges between transcriptomically similar nodes.
            Default: 0.45.
            Recommended range: [0.2, 0.5]

        ontology_weight: γ - Weight for ontology proximity prior.
            Higher values favor edges between developmentally related nodes.
            Default: 0.20.
            Recommended range: [0.1, 0.3]

        decay_constant: λ - How quickly ontology bonus decays with distance,
            bonus = exp(-distance / λ). Lower = faster decay. Default: 2.0.
            Recommended range: [1.0, 3.0]

        min_summed_score: Minimum combined score retained after structural and
            cosine-similarity gates. Default: 0.28.
        enable_inferred_paths: Toggle inferred path discovery. Default: True

    The ontology bonus decays exponentially with hop distance and therefore acts
    as a structural proximity prior rather than evidence of a transition.
    """
    
    # Weighting parameters
    embedding_weight: float = INFERRED_PATH_EMBEDDING_WEIGHT    # β - embedding similarity
    ontology_weight: float = INFERRED_PATH_ONTOLOGY_WEIGHT      # γ - ontology proximity prior
    decay_constant: float = INFERRED_PATH_DECAY_CONSTANT        # λ - ontology distance decay
    min_summed_score: float = INFERRED_PATH_MIN_SUMMED_SCORE    # user-tuned summed-score floor (see discover_inferred_paths)
    
    # Feature toggle
    enable_inferred_paths: bool = True

    def __post_init__(self):
        """Run validate() automatically at construction time."""
        self.validate()

    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is out of valid range.
        """
        # Validate weights are non-negative
        if self.embedding_weight < 0:
            raise ValueError(f"embedding_weight must be non-negative, got {self.embedding_weight}")
        if self.ontology_weight < 0:
            raise ValueError(f"ontology_weight must be non-negative, got {self.ontology_weight}")
        
        # Validate decay constant is positive
        if self.decay_constant <= 0:
            raise ValueError(f"decay_constant must be positive, got {self.decay_constant}")


# =============================================================================
# Semantic name-based safeguard for soft-edge direction
# =============================================================================
# A small set of name-string patterns captures the developmental-precursor /
# end-stage cell naming conventions used in CL. Pairs whose direction the
# default depth/gradient rule would orient incorrectly are flipped by counting
# how many precursor patterns each side matches: the higher-count side becomes
# the source, the lower-count side the target. Terminal patterns act in
# reverse (the side that matches more terminal patterns becomes the target).
#
_PRECURSOR_PREFIXES: Tuple[str, ...] = (
    "immature ",
    "early ",
    "pre-",
    "primitive ",
    "embryonic ",
    "fetal pre-",   # kept; required for 'fetal pre-type II ...'
    "fetal ",       # broader fetal-precursor convention
    "neonatal ",
    "primary ",
)

# Plain substrings (literal `in` check after lowercasing).
_PRECURSOR_SUBSTRINGS: Tuple[str, ...] = (
    "pluripotent",
    "progenitor",
    "precursor",
    "poietic",
)

# Substrings that need word-boundary matching to avoid false positives like
# 'system cell' containing 'stem cell', or 'blastocyst' containing 'blast'.
_PRECURSOR_REGEXES: Tuple[re.Pattern, ...] = (
    re.compile(r"\bstem cell\b"),
    re.compile(r"blast\b"),   # matches word-final 'blast' (lymphoblast, myeloblast,
                               # "... blast") but NOT 'blastocyst' where 'blast' is
                               # followed by another word character.
)

_TERMINAL_SUBSTRINGS: Tuple[str, ...] = (
    "terminally differentiated",
)

# Names that textually match a precursor pattern but are NOT developmental
# precursors — the matching token is anatomical or positional. Lowercased.
_PRECURSOR_NAME_DENYLIST: frozenset = frozenset({
    "pre-venule capillary cell",
    "primary motor cortex pyramidal cell",
    "primary neuron (sensu teleostei)",
    "primary sensory neuron (sensu teleostei)",
})


def _precursor_count(name: Optional[str]) -> int:
    """Number of precursor naming patterns the name matches.

    Counts at most one matching prefix plus each matching substring/regex.
    Names in the denylist always return 0. Empty/None returns 0.
    Case-insensitive.
    """
    if not name:
        return 0
    n = name.lower()
    if n in _PRECURSOR_NAME_DENYLIST:
        return 0
    count = 0
    if any(n.startswith(p) for p in _PRECURSOR_PREFIXES):
        count += 1
    for s in _PRECURSOR_SUBSTRINGS:
        if s in n:
            count += 1
    for pat in _PRECURSOR_REGEXES:
        if pat.search(n) is not None:
            count += 1
    return count


def _terminal_count(name: Optional[str]) -> int:
    """Number of terminal naming patterns the name matches.

    Case-insensitive. Empty/None returns 0.
    """
    if not name:
        return 0
    n = name.lower()
    return sum(1 for s in _TERMINAL_SUBSTRINGS if s in n)


def _is_likely_precursor_name(name: Optional[str]) -> bool:
    """Return whether a name matches a developmental-precursor pattern."""
    return _precursor_count(name) > 0


def _is_terminally_differentiated_name(name: Optional[str]) -> bool:
    """Return True if the cell-type name marks an end-stage/terminal cell."""
    return _terminal_count(name) > 0


def _semantic_direction(name_a: str, name_b: str) -> Optional[Tuple[int, int]]:
    """Decide direction between two cell types from their names alone.

    Uses the precursor/terminal pattern counts:

      - Higher precursor count wins as source (more precursor signals → more
        primitive cell type).
      - Tie on precursor count: higher terminal count is the target
        (terminal/end-stage cell receives, does not emit).
      - All tied: return None (caller falls back to depth or gradient).

    Returns (0, 1) if A → B, (1, 0) if B → A, None otherwise.
    """
    pa, pb = _precursor_count(name_a), _precursor_count(name_b)
    if pa > pb:
        return (0, 1)
    if pb > pa:
        return (1, 0)
    ta, tb = _terminal_count(name_a), _terminal_count(name_b)
    if ta > tb:
        return (1, 0)
    if tb > ta:
        return (0, 1)
    return None


def discover_inferred_paths(
    pure_ontology_adj: np.ndarray,
    ontology_adj: np.ndarray,
    node_embeddings: np.ndarray,
    config: SoftEdgeConfig,
    cache: Optional[VisualizationCache] = None
) -> Tuple[np.ndarray, Dict[Tuple[int, int], float], Dict[Tuple[int, int], Tuple[int, int]]]:
    """
    Discover inferred paths via a scored, gated candidate generator.

    Inferred paths are data-driven transitions that bypass the rigid ontology hierarchy.
    Every candidate pair (i, j) is first assigned a score, then kept only if it survives
    a set of structural eligibility gates — the score ranks candidates, it does not by
    itself decide inclusion:

        score(i, j) = embedding_weight * cosine(node_embeddings[i], node_embeddings[j])
                    + ontology_weight * ontology_bonus(i, j)

    where ``ontology_bonus`` is the exponentially decaying proximity term based
    on ontology distance. Model-graph membership is an eligibility gate rather
    than a weighted score component.

    A candidate must pass all of the following gates to be kept, regardless of score:

    1. Model-graph membership: ontology_adj[i, j] > 0 (the model actually drew this edge,
       e.g. semantic k-NN)
    2. Undirected novelty: (pure_ontology_adj + pure_ontology_adj.T)[i, j] == 0, i.e.
       ontology distance >= 2 — this excludes both the canonical edge itself and its
       reverse-is-a mirror (parent->child when the canonical edge is child->parent)
    3. No self-loops
    4. Gate 2 — terminal-sibling drop: pairs where both i and j are leaves sharing an
       immediate parent are excluded (see `_terminal_sibling_mask`)
    5. Floor 1 (data-derived): an Otsu floor on cosine similarity over the candidates
       that passed gates 1-4 — cosine is bimodal, so this is a real valley
    6. Floor 2 (user-tuned): config.min_summed_score on the summed score — the summed score
       is unimodal (no valley), so this cut is a human-tuned knob (see Step 7)

    Args:
        pure_ontology_adj: [n_nodes, n_nodes] Pure IS_A hierarchy adjacency matrix.
            Contains only the canonical ontology edges (child->parent = 1.0).

        ontology_adj: [n_nodes, n_nodes] Model's graph structure adjacency matrix.
            Includes semantic k-NN edges added during graph construction.
            The difference (ontology_adj - pure_ontology_adj) gives model-added edges.

        node_embeddings: [n_nodes, embedding_dim] GAT embeddings from checkpoint.
            Used to compute cosine similarity between nodes.

        config: SoftEdgeConfig with weighting parameters and thresholds.

        cache: Optional VisualizationCache for caching expensive computations.
            If None, computations are performed without caching.
    
    Returns:
        augmented_adj: [n_nodes, n_nodes] Adjacency matrix with inferred paths added.
            Binary matrix where 1.0 indicates an edge (canonical or soft).
        
        edge_weights: Dict mapping (source, target) tuples to inferred path weights.
            Only contains inferred paths (not canonical edges).
            Each weight is the summed score (embedding_weight*cosine + ontology_weight*bonus);
            higher means more confident. NOT normalized to [0, 1].
        
        lateral_directions: Dict mapping each lateral inferred path's canonical key
            (whichever of (i,j) or (j,i) appears in edge_weights) to the canonical
            (src, tgt) tuple where src is the less-differentiated node and tgt is the
            more-differentiated node, as determined by the embedding gradient.
            Only contains entries for lateral pairs (nodes at the same ontology depth).
            This is the single source of truth for lateral direction throughout the pipeline.
    
    Raises:
        ValueError: If input matrices have incompatible shapes or invalid values.

    Notes:
        - Inferred paths are only added between nodes without existing ontology edges
        - Self-loops are never added
        - Scores are NOT bounded to [0, 1]: cosine similarity ranges over [-1, 1], so at
          default weights (embedding_weight=0.45, ontology_weight=0.20) the score ranges
          roughly [-0.45, 0.52], not [0, 1]
        - Disconnected components in the ontology have ontology_bonus = 0
    """
    start_time = time.time()
    
    # Input validation
    if pure_ontology_adj.shape != ontology_adj.shape:
        raise ValueError(
            f"Shape mismatch: pure_ontology_adj {pure_ontology_adj.shape} "
            f"vs ontology_adj {ontology_adj.shape}"
        )
    
    n_nodes = len(pure_ontology_adj)
    
    if node_embeddings.shape[0] != n_nodes:
        raise ValueError(
            f"Embedding count mismatch: expected {n_nodes} nodes, "
            f"got {node_embeddings.shape[0]} embeddings"
        )
    
    logger.info(f"Starting inferred path discovery for {n_nodes} nodes")
    logger.debug(f"Config: embedding_weight={config.embedding_weight}, "
                f"ontology_weight={config.ontology_weight}")
    
    # Step 1: Precompute ontology distances (use cache if available)
    if cache is not None:
        ontology_distances = cache.get_ontology_distances(pure_ontology_adj)
    else:
        ontology_distances = compute_ontology_distances(pure_ontology_adj)

    # Step 2: Compute embedding similarity (use cache if available)
    if cache is not None:
        embedding_similarity = cache.get_embedding_similarity(node_embeddings)
    else:
        norms = np.linalg.norm(node_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_norm = node_embeddings / norms
        embedding_similarity = embeddings_norm @ embeddings_norm.T
    
    # Compute ontology proximity bonus.
    ontology_bonus = np.exp(-ontology_distances / config.decay_constant)
    ontology_bonus[np.isinf(ontology_distances)] = 0
    
    # Combine embedding similarity with the ontology-distance prior.
    inferred_path_weights = (
        config.embedding_weight * embedding_similarity +
        config.ontology_weight * ontology_bonus
    )

    # Step 6: structural masks
    mask_model_added = (ontology_adj > 0)                                  # edge present in the model graph (canonical OR k-NN-added); the novelty mask below strips the canonical ones
    mask_novel = ((pure_ontology_adj + pure_ontology_adj.T) == 0)         # UNDIRECTED -> excludes reverse-is-a (dist>=2)
    mask_no_self = ~np.eye(n_nodes, dtype=bool)
    mask_not_sibling = ~_terminal_sibling_mask(pure_ontology_adj)          # Gate 2: drop terminal same-parent siblings
    structural = mask_model_added & mask_novel & mask_no_self & mask_not_sibling

    # Apply two complementary floors.
    #   Floor 1 (data-derived): Otsu on the COSINE similarity over the candidate pool. Cosine is
    #     genuinely bimodal (unrelated vs. similar pairs), so its Otsu is a real valley — unlike the
    #     summed score, which is unimodal and gives only a meaningless central split near the peak.
    #   Floor 2 (configured): config.min_summed_score on the full summed score.
    # Both are required. scikit-image is optional, so degrade the cosine floor gracefully if absent
    # (the min_summed_score floor and Gate 1 / Gate 2 still apply).
    cosine_cand = embedding_similarity[structural]
    cosine_floor = -np.inf   # fallback: no cosine floor if skimage is unavailable / pool degenerate
    if cosine_cand.size > 1 and cosine_cand.min() < cosine_cand.max():
        try:
            from skimage.filters import threshold_otsu
            cosine_floor = float(threshold_otsu(cosine_cand))
        except ImportError:
            logger.warning(
                "scikit-image not installed; skipping the Otsu cosine floor for inferred paths. "
                "min_summed_score / Gate 1 / Gate 2 still apply."
            )
    inferred_path_mask = (
        structural
        & (embedding_similarity >= cosine_floor)
        & (inferred_path_weights >= config.min_summed_score)
    )

    # Step 7b: augmented adjacency
    augmented_adj = pure_ontology_adj.copy()
    augmented_adj[inferred_path_mask] = 1.0

    # Step 8: extract inferred-path weights for visualization
    edge_weights = {}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if inferred_path_mask[i, j]:
                edge_weights[(i, j)] = float(inferred_path_weights[i, j])

    # =========================================================================
    # Step 9: Compute embedding-gradient direction for lateral pairs
    # =========================================================================
    # Lateral pairs are inferred path edges where both nodes share the same
    # ontology depth.  For these edges the geometry-based direction used by the
    # placer is arbitrary (both nodes may share the same x/r coordinate), so we
    # compute a single authoritative direction here from cell-state evidence and
    # carry it as metadata through the rest of the pipeline.
    #
    # Algorithm:
    #   1. Compute node depths from pure_ontology_adj via BFS (roots = depth 0,
    #      depth = max(parent depths) + 1).
    #   2. Build a reference direction = mean of (emb[parent] - emb[child]) over
    #      all vertical edges in pure_ontology_adj. In pure_ontology_adj,
    #      pure_ontology_adj[child, parent] = 1.0 means child->parent.
    #   3. For each lateral pair (i, j) in edge_weights, project emb[i] and
    #      emb[j] onto the reference direction.  The node with the higher
    #      projection is the target (more differentiated).
    #   4. Fall back to L2-norm comparison when the reference direction is zero.

    # 9a. Compute node depths via BFS
    from collections import deque as _deque

    _parents: Dict[int, List[int]] = {k: [] for k in range(n_nodes)}
    _children_map: Dict[int, List[int]] = {k: [] for k in range(n_nodes)}
    for _i in range(n_nodes):
        for _j in range(n_nodes):
            if pure_ontology_adj[_i, _j] > 0:
                # pure_ontology_adj[child, parent] = 1  =>  _j is parent of _i
                _parents[_i].append(_j)
                _children_map[_j].append(_i)

    _node_depths: Dict[int, int] = {k: 0 for k in range(n_nodes)}
    _roots = [k for k in range(n_nodes) if len(_parents[k]) == 0]
    _visited: set = set(_roots)
    _bfs_queue: _deque = _deque(_roots)

    while _bfs_queue:
        _node = _bfs_queue.popleft()
        _cur_depth = _node_depths[_node]
        for _child in _children_map[_node]:
            _new_depth = _cur_depth + 1
            if _new_depth > _node_depths[_child]:
                _node_depths[_child] = _new_depth
            if all(p in _visited for p in _parents[_child]) and _child not in _visited:
                _visited.add(_child)
                _bfs_queue.append(_child)

    # 9b. Build reference direction from vertical edges
    # Vertical edge: pure_ontology_adj[child, parent] = 1 where
    # node_depths[parent] < node_depths[child]  (parent is more primitive).
    # Gradient direction: emb[child] - emb[parent]  (primitive -> differentiated).
    # Higher projection along this direction = more differentiated = target.
    _gradient_vecs: List[np.ndarray] = []
    for _child in range(n_nodes):
        for _parent in range(n_nodes):
            if pure_ontology_adj[_child, _parent] > 0:
                if _node_depths[_parent] < _node_depths[_child]:
                    _gradient_vecs.append(node_embeddings[_child] - node_embeddings[_parent])

    if _gradient_vecs:
        _ref_dir: Optional[np.ndarray] = np.mean(_gradient_vecs, axis=0)
        _ref_norm = float(np.linalg.norm(_ref_dir))
        if _ref_norm > 1e-9:
            _ref_dir = _ref_dir / _ref_norm
        else:
            _ref_dir = None  # zero reference direction -> fall back to L2 norm
    else:
        _ref_dir = None  # no vertical edges -> fall back to L2 norm

    # 9c. Assign canonical direction for each lateral inferred path
    lateral_directions: Dict[Tuple[int, int], Tuple[int, int]] = {}

    for (i, j) in edge_weights:
        if _node_depths.get(i, 0) != _node_depths.get(j, 0):
            # Not a lateral pair — skip
            continue

        if _ref_dir is not None:
            proj_i = float(np.dot(node_embeddings[i], _ref_dir))
            proj_j = float(np.dot(node_embeddings[j], _ref_dir))
            # Node with higher projection is more differentiated -> becomes target
            canonical: Tuple[int, int] = (i, j) if proj_j >= proj_i else (j, i)
        else:
            # Fall back: node with higher L2 norm is the target
            norm_i = float(np.linalg.norm(node_embeddings[i]))
            norm_j = float(np.linalg.norm(node_embeddings[j]))
            canonical = (i, j) if norm_j >= norm_i else (j, i)

        lateral_directions[(i, j)] = canonical
        # Store reverse key too so downstream code can look up either direction
        lateral_directions[(j, i)] = canonical

    n_lateral = len(lateral_directions)
    if n_lateral > 0:
        logger.info(f"  Lateral inferred paths with embedding-gradient direction: {n_lateral}")

    elapsed = time.time() - start_time

    # Log statistics
    n_inferred_paths = len(edge_weights)
    n_canonical_edges = int(np.sum(pure_ontology_adj > 0))

    logger.info(f"Inferred path discovery completed in {elapsed:.3f}s")
    logger.info(f"  Canonical edges: {n_canonical_edges}")
    logger.info(f"  Inferred paths discovered: {n_inferred_paths}")
    logger.info(f"  Total edges: {n_canonical_edges + n_inferred_paths}")

    if n_inferred_paths > 0:
        weights_array = np.array(list(edge_weights.values()))
        logger.info(f"  Inferred path weight stats: min={np.min(weights_array):.3f}, "
                   f"max={np.max(weights_array):.3f}, mean={np.mean(weights_array):.3f}")

    return augmented_adj, edge_weights, lateral_directions


def get_edge_style(edge_type: str, weight: float = 1.0) -> Dict[str, Any]:
    """
    Get visual styling for edge based on type and weight.
    
    This function returns a style dictionary that can be used to render edges
    in Plotly visualizations. Canonical edges (from the ontology) are rendered
    as solid black lines, while inferred paths (data-driven transitions) are rendered
    as dotted gray lines with opacity proportional to their confidence weight.
    
    Args:
        edge_type: Type of edge, either 'canonical' or 'soft'.
            - 'canonical': Edges from the pure IS_A ontology hierarchy
            - 'soft': Data-driven edges discovered through inferred path discovery
        
        weight: Inferred-path score. Only used for inferred paths, whose opacity
            is computed as ``0.6 * weight``. Default: 1.0.
    
    Returns:
        Style dictionary with the following keys:
            - 'line_dash': Line style ('solid' for canonical, 'dot' for soft)
            - 'line_color': Hex color code ('#000000' for canonical, '#888888' for soft)
            - 'line_width': Line width in pixels (1.5 for canonical, 2.5 for soft)
            - 'opacity': ``1.0`` for canonical edges or ``0.6 * weight`` for
              inferred paths.
    
    Notes:
        - Inferred-path opacity scales with confidence (opacity = 0.6 * weight),
          and they use thicker dotted lines so edge confidence is visually clear.
    """
    if edge_type == 'canonical':
        return {
            'line_dash': 'solid',
            'line_color': '#000000',
            'line_width': 1.5,
            'opacity': 1.0
        }
    else:  # inferred path
        # Opacity scales with confidence: 0.0 (low) to 0.6 (high)
        # Formula: opacity = 0.6 * weight (allows very faint edges)
        opacity = 0.6 * weight
        
        return {
            'line_dash': 'dot',
            'line_color': '#888888',
            'line_width': 2.5,
            'opacity': opacity
        }


def identify_roots(adjacency_matrix: np.ndarray) -> List[int]:
    """
    Identify root nodes (nodes with no incoming edges) in the graph.
    
    Root nodes are the reference points for the embedding-distance coordinate.
    
    Args:
        adjacency_matrix: [n_nodes, n_nodes] Adjacency matrix where entry [i, j] = 1
            indicates an edge from node ``i`` to node ``j``. The function treats
            the column sum as in-degree, so callers must supply parent-to-child
            orientation when roots represent nodes with no parents.
    
    Returns:
        List of node indices that have no incoming edges (in-degree = 0).
        If no roots are found (e.g., cyclic graph), returns [0] as fallback.
    
    Notes:
        - In-degree is the column sum of the adjacency matrix.
        - With multiple roots, distance is measured from the nearest root.
        - Fallback to [0] ensures at least one root is always returned.
    """
    # Compute in-degree for each node (sum of incoming edges)
    # In adjacency_matrix[i, j] = 1 means edge from i to j, so column sum gives in-degree
    in_degree = np.sum(adjacency_matrix, axis=0)
    
    # Find nodes with no incoming edges
    roots = np.where(in_degree == 0)[0].tolist()
    
    # Fallback to node 0 if no roots found (e.g., cyclic graph)
    return roots if roots else [0]


def compute_pseudotimes(
    node_embeddings: np.ndarray,
    roots: List[int]
) -> np.ndarray:
    """
    Compute pseudotime as distance from nearest root in embedding space.
    
    This operational coordinate is the Euclidean distance from the nearest root
    in the
    latent embedding space. This allows multi-root datasets to maintain
    independent pseudotime scales for each lineage.
    
    Args:
        node_embeddings: [n_nodes, embedding_dim] GAT embeddings from checkpoint.
            Each row is the latent representation of a cell type node.
        
        roots: Root node indices returned by ``identify_roots()``.
    
    Returns:
        pseudotimes: [n_nodes] array of root-distance values.
            - Root nodes have value 0 (distance to self)
            - Other nodes have the distance to the nearest root
            - Values are non-negative (Euclidean distance)
    
    Notes:
        - Euclidean (L2) distance in latent space, not normalized.
        - With multiple roots, each node maps to its nearest root.
    """
    n_nodes = len(node_embeddings)
    pseudotimes = np.zeros(n_nodes)
    
    # Get root embeddings
    root_embeddings = node_embeddings[roots]
    
    for i in range(n_nodes):
        # Compute Euclidean distance to each root
        distances = np.linalg.norm(
            node_embeddings[i] - root_embeddings,
            axis=1
        )
        # Use minimum distance (nearest root)
        pseudotimes[i] = np.min(distances)
    
    return pseudotimes


@dataclass
class ElasticLayoutConfig:
    """
    Configuration for elastic accordion layout.
    
    The elastic layout transforms rigid ontology depths (integers: 1, 2, 3...) into
    continuous depths based on latent pseudotime. This allows edge length to represent
    transcriptomic distance rather than annotation depth.
    
    Attributes:
        elasticity_factor: Controls blending between rigid ontology depth and flexible
            pseudotime. Range [0.0, 1.0]:
            - 0.0 = rigid grid (pure ontology depth)
            - 1.0 = pure pseudotime (maximum flexibility)
            - 0.6 (default) = 60% biological signal, 40% structural anchoring
            Recommended range: [0.4, 0.8]
        
        max_deviation: Maximum drift from layer anchor in depth units.
            Prevents nodes from escaping their ontological "lane".
            Default: 0.4
            Recommended range: [0.2, 0.6]
        
        min_layer_gap: Minimum visual separation between adjacent ontology layers.
            Ensures hierarchical readability is preserved.
            Default: 1.0
            Recommended range: [0.5, 2.0]
        
        min_node_dist: Minimum depth difference for causality enforcement.
            Ensures child depth > parent depth by at least this amount.
            Default: 0.25
            Recommended range: [0.1, 0.5]
        
        enable_elastic_layout: Feature toggle to enable/disable elastic layout.
            When False, uses rigid integer depths.
            Default: True
    
    Interpretation:
        Ontology depth counts annotation steps rather than transcriptomic
        separation. Elastic depths blend that structural depth with distance from
        lineage roots in embedding space. ``elasticity_factor`` controls the
        tradeoff between ontology anchoring and embedding-derived separation.
    """
    
    # Elasticity parameters
    elasticity_factor: float = 0.6      # 0.0=rigid grid, 1.0=pure pseudotime
    max_deviation: float = 0.4          # Max drift from layer anchor
    min_layer_gap: float = 1.0          # Minimum visual separation between layers
    min_node_dist: float = 0.25         # Causality enforcement (child > parent)
    
    # Feature toggle
    enable_elastic_layout: bool = True

    def __post_init__(self):
        """Run validate() automatically at construction time."""
        self.validate()

    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is out of valid range.
        """
        if not 0 <= self.elasticity_factor <= 1:
            raise ValueError(f"elasticity_factor must be in [0, 1], got {self.elasticity_factor}")
        if self.max_deviation < 0:
            raise ValueError(f"max_deviation must be non-negative, got {self.max_deviation}")
        if self.min_layer_gap < 0:
            raise ValueError(f"min_layer_gap must be non-negative, got {self.min_layer_gap}")
        if self.min_node_dist < 0:
            raise ValueError(f"min_node_dist must be non-negative, got {self.min_node_dist}")


def compute_layer_anchors(
    pseudotimes: np.ndarray,
    node_depths: Dict[int, int]
) -> Dict[int, float]:
    """
    Compute average pseudotime for each ontology depth level.
    
    Layer anchors serve as reference points for elastic positioning. Each node's
    elastic depth is constrained to stay within max_deviation of its layer anchor,
    ensuring nodes don't escape their ontological "lane".
    
    Args:
        pseudotimes: [n_nodes] array of pseudotime values from compute_pseudotimes().
            Represents distance from nearest root in embedding space.
        
        node_depths: Dictionary mapping node index to ontology depth (integer).
            Depth 0 = root level, depth 1 = first level children, etc.
    
    Returns:
        layer_anchors: Dictionary mapping ontology depth to average pseudotime.
            Monotonicity is enforced: layer_anchors[N+1] > layer_anchors[N].
    
    Notes:
        - If a layer's average pseudotime is <= the previous layer's, it is
          bumped to previous_anchor + 0.1 to keep monotonicity, preserving visual
          hierarchy even when pseudotimes don't align with ontology structure.
    """
    from collections import defaultdict
    
    # Group nodes by depth
    depth_groups = defaultdict(list)
    for node_idx, depth in node_depths.items():
        depth_groups[depth].append(pseudotimes[node_idx])
    
    # Compute average pseudotime per depth
    layer_anchors = {}
    for depth, times in depth_groups.items():
        layer_anchors[depth] = float(np.mean(times))
    
    # Enforce monotonicity in anchors (Level N+1 > Level N)
    sorted_depths = sorted(layer_anchors.keys())
    for i in range(1, len(sorted_depths)):
        prev_depth = sorted_depths[i-1]
        curr_depth = sorted_depths[i]
        if layer_anchors[curr_depth] <= layer_anchors[prev_depth]:
            layer_anchors[curr_depth] = layer_anchors[prev_depth] + 0.1
    
    return layer_anchors


def enforce_causality(
    elastic_depths: Dict[int, float],
    adjacency_matrix: np.ndarray,
    min_gap: float = 0.25
) -> Dict[int, float]:
    """
    Enforce causality: child depth > parent depth.

    Iteratively adjusts elastic depths toward a minimum parent-child gap.
    Constraints that would exceed the depth safety threshold are skipped.

    Args:
        elastic_depths: Dictionary mapping node index to elastic depth (float).
            Initial depths from blending ontology depth with pseudotime.
        
        adjacency_matrix: [n_nodes, n_nodes] parent-to-child adjacency matrix,
            where entry ``[parent, child]`` denotes an edge.
        
        min_gap: Minimum depth difference between parent and child.
            Default: 0.25 (from requirements)
    
    Returns:
        depths: Dictionary mapping node index to adjusted elastic depth. For
            processed constraints, a child is placed at least ``min_gap`` beyond
            its parent; safety limits and cycles can prevent full convergence.
    
    Notes:
        - At most 100 adjustment passes are performed.
        - Edges that would exceed the depth safety threshold are skipped.
    """
    n_nodes = len(adjacency_matrix)
    depths = elastic_depths.copy()
    
    # Iterate until all constraints satisfied
    max_iterations = 100
    max_depth_threshold = 50.0  # Safety threshold for cycle detection
    
    for iteration in range(max_iterations):
        violations = 0
        
        for child in range(n_nodes):
            if child not in depths:
                continue
            # Find parents (nodes with edge to this child)
            # adjacency_matrix[parent, child] = 1 means parent -> child
            parents = np.where(adjacency_matrix[:, child] > 0)[0]
            
            for parent in parents:
                if parent not in depths:
                    continue
                if depths[child] <= depths[parent] + min_gap:
                    # Violation: push child forward
                    new_depth = depths[parent] + min_gap
                    
                    # SAFETY CHECK: Detect runaway depth explosion
                    if new_depth > max_depth_threshold:
                        logger.warning(
                            f"Depth explosion detected at node {child}: "
                            f"{new_depth:.2f} > {max_depth_threshold}. "
                            f"Possible cycle in adjacency matrix. Skipping edge."
                        )
                        continue
                    
                    depths[child] = new_depth
                    violations += 1
        
        if violations == 0:
            break
    
    # Final safety check for depth explosion
    max_depth = max(depths.values()) if depths else 0
    if max_depth > max_depth_threshold:
        logger.error(
            f"Elastic depth range explosion: max={max_depth:.2f}. "
            f"This indicates cycles in the adjacency matrix. "
            f"Ensure pure ontology DAG is used for layout pipeline."
        )
    
    return depths


def enforce_layer_gaps(
    elastic_depths: Dict[int, float],
    node_depths: Dict[int, int],
    min_gap: float = 1.0
) -> Dict[int, float]:
    """
    Ensure minimum separation between adjacent ontology layers.

    Enforces visual hierarchy: nodes in layer N+1 are kept at least min_gap from
    layer N. When the gap is too small, layer N+1 and all subsequent layers are
    shifted forward.

    Args:
        elastic_depths: Dictionary mapping node index to elastic depth (float).
            Depths after causality enforcement.
        
        node_depths: Dictionary mapping node index to ontology depth (integer).
            Used to group nodes by layer.
        
        min_gap: Minimum separation between the maximum depth of layer N and
            the minimum depth of layer N+1.
            Default: 1.0 (from requirements)
    
    Returns:
        depths: Dictionary mapping node index to adjusted elastic depth.
            Guarantees: min(layer[N+1]) - max(layer[N]) >= min_gap
    
    Notes:
        - Processes layers from lowest to highest depth; shifting a layer also
          shifts all subsequent layers while preserving intra-layer positions.
    """
    from collections import defaultdict
    
    # Group nodes by ontology depth
    depth_groups = defaultdict(list)
    for node_idx, depth in node_depths.items():
        if node_idx in elastic_depths:
            depth_groups[depth].append(node_idx)
    
    # Handle empty case
    if not depth_groups:
        return elastic_depths.copy()
    
    # Compute layer boundaries (min/max elastic depth per layer)
    layer_bounds = {}
    for depth, nodes in depth_groups.items():
        elastic_vals = [elastic_depths[n] for n in nodes]
        layer_bounds[depth] = (min(elastic_vals), max(elastic_vals))
    
    # Enforce gaps between adjacent layers
    sorted_depths = sorted(layer_bounds.keys())
    depths = elastic_depths.copy()
    
    # Track cumulative shift for each layer
    cumulative_shift = 0.0
    
    for i in range(len(sorted_depths)):
        curr_depth = sorted_depths[i]
        
        # Apply cumulative shift to current layer
        if cumulative_shift > 0:
            for node in depth_groups[curr_depth]:
                depths[node] += cumulative_shift
        
        # Check gap to next layer (if exists)
        if i < len(sorted_depths) - 1:
            next_depth = sorted_depths[i + 1]
            
            # Recompute bounds after shift
            curr_max = max(depths[n] for n in depth_groups[curr_depth])
            next_min = min(elastic_depths[n] + cumulative_shift for n in depth_groups[next_depth])
            
            gap = next_min - curr_max
            if gap < min_gap:
                # Need to shift next layer (and all subsequent)
                additional_shift = min_gap - gap
                cumulative_shift += additional_shift
    
    # Apply final cumulative shift to last layer if needed
    if len(sorted_depths) > 0:
        last_depth = sorted_depths[-1]
        for node in depth_groups[last_depth]:
            if depths[node] == elastic_depths[node]:  # Not yet shifted
                depths[node] += cumulative_shift
    
    return depths


def compute_elastic_depths(
    adjacency_matrix: np.ndarray,
    node_embeddings: np.ndarray,
    node_depths: Dict[int, int],
    config: ElasticLayoutConfig
) -> Dict[int, float]:
    """
    Compute elastic depths using an embedding-distance coordinate.
    
    This is the main function for elastic accordion layout. It transforms rigid
    ontology depths into continuous depths influenced by embedding distance while
    retaining explicit hierarchy constraints.
    
    The algorithm:
    1. Identify roots (nodes with no parents)
    2. Compute distance from the nearest root in embedding space
    3. Compute layer anchors (average pseudotime per ontology level)
    4. Blend ontology depth with pseudotime using elasticity_factor
    5. Constrain deviation from layer anchor
    6. Enforce causality (child > parent + min_gap)
    7. Enforce minimum layer separation
    
    Args:
        adjacency_matrix: [n_nodes, n_nodes] HECTOR ontology adjacency matrix,
            where ``adjacency_matrix[child, parent] = 1``.
        
        node_embeddings: [n_nodes, embedding_dim] GAT embeddings from checkpoint.
            Used to compute pseudotime (distance from roots).
        
        node_depths: Dictionary mapping node index to ontology depth (integer).
            Depth 0 = root level, depth 1 = first level children, etc.
        
        config: ElasticLayoutConfig with elasticity parameters.
    
    Returns:
        elastic_depths: Dictionary mapping node index to elastic depth (float).
            Continuous depths that can be used for X-coordinates in visualization.
    
    Raises:
        ValueError: If input shapes are incompatible.

    Edge length represents the adjusted embedding separation; it does not by
    itself measure elapsed time or transition velocity.
    """
    start_time = time.time()
    n_nodes = len(adjacency_matrix)
    
    # Handle empty case
    if n_nodes == 0:
        logger.warning("Empty adjacency matrix, returning empty elastic depths")
        return {}
    
    # Input validation
    if node_embeddings.shape[0] != n_nodes:
        raise ValueError(
            f"Embedding count mismatch: adjacency has {n_nodes} nodes, "
            f"embeddings have {node_embeddings.shape[0]} rows"
        )

    # The low-level root and causality helpers operate on parent-to-child
    # adjacency, while HECTOR stores ontology edges as child-to-parent.
    parent_child_adjacency = adjacency_matrix.T
    
    logger.info(f"Computing elastic depths for {n_nodes} nodes")
    logger.debug(f"Config: elasticity_factor={config.elasticity_factor}, "
                f"max_deviation={config.max_deviation}, "
                f"min_layer_gap={config.min_layer_gap}, "
                f"min_node_dist={config.min_node_dist}")
    
    # Step 1: Identify roots (nodes with no incoming edges)
    roots = identify_roots(parent_child_adjacency)
    logger.debug(f"Identified {len(roots)} root node(s): {roots[:5]}{'...' if len(roots) > 5 else ''}")
    
    # Step 2: Compute pseudotime for each node
    pseudotimes = compute_pseudotimes(node_embeddings, roots)
    logger.debug(f"Pseudotime range: [{np.min(pseudotimes):.3f}, {np.max(pseudotimes):.3f}]")
    
    # Step 3: Compute layer anchors (average pseudotime per depth level)
    layer_anchors = compute_layer_anchors(pseudotimes, node_depths)
    n_layers = len(layer_anchors)
    logger.debug(f"Computed {n_layers} layer anchors")
    
    # Step 4: Assign elastic positions with blending
    elastic_depths: Dict[int, float] = {}
    for node_idx in range(n_nodes):
        if node_idx not in node_depths:
            # Node not in node_depths, use pseudotime directly
            elastic_depths[node_idx] = float(pseudotimes[node_idx])
            continue
            
        ontology_depth = node_depths[node_idx]
        layer_anchor = layer_anchors.get(ontology_depth, float(ontology_depth))
        node_pseudotime = pseudotimes[node_idx]
        
        # Blend between rigid ontology depth and flexible pseudotime
        elastic_depth = (
            (1 - config.elasticity_factor) * ontology_depth +
            config.elasticity_factor * node_pseudotime
        )
        
        # Constrain deviation from layer anchor
        max_deviation = config.max_deviation
        elastic_depth = np.clip(
            elastic_depth,
            layer_anchor - max_deviation,
            layer_anchor + max_deviation
        )
        
        elastic_depths[node_idx] = float(elastic_depth)
    
    # Step 5: Enforce monotonicity (child > parent)
    elastic_depths = enforce_causality(
        elastic_depths,
        parent_child_adjacency,
        min_gap=config.min_node_dist
    )
    
    # Step 6: Enforce minimum layer separation
    elastic_depths = enforce_layer_gaps(
        elastic_depths,
        node_depths,
        min_gap=config.min_layer_gap
    )
    
    elapsed = time.time() - start_time
    
    # Log statistics
    depths_array = np.array(list(elastic_depths.values()))
    logger.info(f"Elastic layout completed in {elapsed:.3f}s")
    logger.info(f"  Nodes processed: {len(elastic_depths)}")
    logger.info(f"  Depth range: [{np.min(depths_array):.3f}, {np.max(depths_array):.3f}]")
    logger.info(f"  Depth std: {np.std(depths_array):.3f}")
    logger.info(f"  Layers: {n_layers}")
    
    return elastic_depths



def apply_elastic_to_radial(
    elastic_depths: Dict[int, float],
    angular_positions: Dict[int, float],
    center_radius: float,
    max_radius: float
) -> Dict[int, Tuple[float, float]]:
    """
    Apply elastic depths to radial layout.

    Transforms elastic depths (continuous X-coordinates from the elastic
    accordion layout) into radial (theta, radius): depths normalized to [0, 1]
    then mapped to [center_radius, max_radius]. Radial separation therefore
    reflects adjusted embedding distance rather than transition velocity.

    Strategy:
    1. Normalize elastic depths to [0, 1] range
    2. Map normalized depths to radius range [center_radius, max_radius]
    3. Combine with angular positions from spectral sorting
    
    Args:
        elastic_depths: Dictionary mapping node indices to elastic depths (float).
            These are continuous depths computed by compute_elastic_depths().
            Values can be any positive float (not necessarily in [0, 1]).
        
        angular_positions: Dictionary mapping node indices to angular positions (float).
            These are theta values in radians from the radial layout's sector
            partitioning algorithm.
        
        center_radius: Inner radius of the radial layout (where root nodes are placed).
            Typically a small positive value like 0.2 or 0.5.
        
        max_radius: Outer radius of the radial layout (where leaf nodes are placed).
            Typically a larger value like 5.0 or 10.0.
    
    Returns:
        radial_positions: Dictionary mapping node indices to (theta, radius) tuples.
            - theta: Angular position in radians (unchanged from input)
            - radius: Radial position computed from normalized elastic depth
    
    Notes:
        - If all elastic depths are equal, all nodes get center_radius (avoids
          division by zero).
        - Only nodes present in both input dicts are processed.
        - Angular positions are preserved; only radius is computed.
    """
    # Handle empty input
    if not elastic_depths:
        return {}
    
    # Step 1: Normalize elastic depths to [0, 1]
    depth_values = list(elastic_depths.values())
    min_depth = min(depth_values)
    max_depth = max(depth_values)
    depth_range = max_depth - min_depth if max_depth > min_depth else 1.0
    
    normalized_depths = {
        node: (elastic_depths[node] - min_depth) / depth_range
        for node in elastic_depths
    }
    
    # Step 2: Map to radius range [center_radius, max_radius]
    radius_range = max_radius - center_radius
    
    # Step 3: Combine with angular positions
    radial_positions = {}
    for node in elastic_depths:
        if node not in angular_positions:
            continue
        
        theta = angular_positions[node]
        normalized_depth = normalized_depths[node]
        radius = center_radius + radius_range * normalized_depth
        radial_positions[node] = (theta, radius)
    
    return radial_positions


# =============================================================================
# Checkpoint Data Extraction
# =============================================================================

def extract_visualization_data(checkpoint_path: str) -> Dict[str, Any]:
    """
    Extract required data from checkpoint for visualization.

    Loads the matrices and embeddings from a HECTOR checkpoint for inferred-path
    discovery and elastic layout. Handles formats with and without 'full_' prefix.

    Args:
        checkpoint_path: Path to the HDF5 checkpoint file (.h5).
            The checkpoint must contain:
            - Pure ontology adjacency matrix (IS_A hierarchy)
            - Model's graph structure (includes semantic k-NN edges)
            - GAT embeddings snapshot
            - Node/class names
    
    Returns:
        viz_data: Dictionary containing:
            - 'pure_adjacency_matrix': np.ndarray [n_nodes, n_nodes]
                Pure IS_A hierarchy from Cell Ontology
            - 'ontology_adj': np.ndarray [n_nodes, n_nodes]
                Model's graph structure (includes semantic k-NN edges if enabled)
            - 'gat_embeddings': np.ndarray [n_nodes, embedding_dim]
                Node embeddings in latent space
            - 'node_names': List[str]
                Cell type names (e.g., 'CL:0000630')
    
    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        KeyError: If required data is missing from checkpoint.
        ValueError: If checkpoint data is corrupted or invalid.
    
    Notes:
        - Supports both 'ontology/pure_ontology_adj' and
          'ontology/full_pure_ontology_adj' keys.
        - (ontology_adj - pure_adjacency_matrix) gives the semantic k-NN edges
          added during graph construction.
    """
    import h5py
    import os
    
    start_time = time.time()
    logger.info(f"Loading visualization data from: {checkpoint_path}")
    
    # Validate file exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    viz_data: Dict[str, Any] = {}
    
    try:
        with h5py.File(checkpoint_path, 'r') as f:
            # Load pure ontology adjacency (IS_A hierarchy)
            if 'ontology/pure_ontology_adj' in f:
                viz_data['pure_adjacency_matrix'] = f['ontology/pure_ontology_adj'][:]
                logger.debug("Loaded pure_ontology_adj from ontology/pure_ontology_adj")
            elif 'ontology/full_pure_ontology_adj' in f:
                viz_data['pure_adjacency_matrix'] = f['ontology/full_pure_ontology_adj'][:]
                logger.debug("Loaded pure_ontology_adj from ontology/full_pure_ontology_adj")
            else:
                logger.error("Checkpoint missing pure_ontology_adj")
                raise KeyError(
                    "Checkpoint missing pure_ontology_adj. "
                    "Expected 'ontology/pure_ontology_adj' or 'ontology/full_pure_ontology_adj'. "
                    "This model may have been trained before this feature was added."
                )
            
            # Load model's graph structure (includes semantic k-NN edges)
            if 'ontology/ontology_adj' in f:
                viz_data['ontology_adj'] = f['ontology/ontology_adj'][:]
                logger.debug("Loaded ontology_adj from ontology/ontology_adj")
            elif 'ontology/full_ontology_adj' in f:
                viz_data['ontology_adj'] = f['ontology/full_ontology_adj'][:]
                logger.debug("Loaded ontology_adj from ontology/full_ontology_adj")
            else:
                logger.error("Checkpoint missing ontology_adj")
                raise KeyError(
                    "Checkpoint missing ontology_adj. "
                    "Expected 'ontology/ontology_adj' or 'ontology/full_ontology_adj'."
                )
            
            # Load GAT embeddings
            if 'embeddings/gat_embeddings_snapshot' in f:
                viz_data['gat_embeddings'] = f['embeddings/gat_embeddings_snapshot'][:]
                logger.debug("Loaded gat_embeddings from embeddings/gat_embeddings_snapshot")
            else:
                logger.error("Checkpoint missing gat_embeddings_snapshot")
                raise KeyError(
                    "Checkpoint missing gat_embeddings_snapshot. "
                    "Expected 'embeddings/gat_embeddings_snapshot'."
                )
            
            # Load node names (cell type identifiers)
            if 'ontology/classes' in f:
                viz_data['node_names'] = [s.decode('utf-8') if isinstance(s, bytes) else s 
                                           for s in f['ontology/classes'][:]]
                logger.debug("Loaded node_names from ontology/classes")
            elif 'ontology/full_classes' in f:
                viz_data['node_names'] = [s.decode('utf-8') if isinstance(s, bytes) else s 
                                           for s in f['ontology/full_classes'][:]]
                logger.debug("Loaded node_names from ontology/full_classes")
            else:
                logger.error("Checkpoint missing class names")
                raise KeyError(
                    "Checkpoint missing class names. "
                    "Expected 'ontology/classes' or 'ontology/full_classes'."
                )
    
    except OSError as e:
        logger.error(f"Failed to open checkpoint file: {e}")
        raise ValueError(f"Failed to open checkpoint file: {e}") from e
    
    # Validate loaded data
    try:
        validate_checkpoint_data(viz_data)
    except ValueError as e:
        logger.error(f"Checkpoint validation failed: {e}")
        raise
    
    elapsed = time.time() - start_time
    n_nodes = len(viz_data['node_names'])
    embedding_dim = viz_data['gat_embeddings'].shape[1]
    
    logger.info(f"Checkpoint loaded in {elapsed:.3f}s")
    logger.info(f"  Nodes: {n_nodes}")
    logger.info(f"  Embedding dimension: {embedding_dim}")
    logger.info(f"  Pure ontology edges: {int(np.sum(viz_data['pure_adjacency_matrix'] > 0))}")
    logger.info(f"  Model graph edges: {int(np.sum(viz_data['ontology_adj'] > 0))}")
    
    return viz_data


def validate_checkpoint_data(viz_data: Dict[str, Any]) -> None:
    """
    Validate checkpoint contains required data for visualization.
    
    This function performs shape validation to ensure all matrices and arrays
    are consistent and can be used together for inferred path discovery and
    elastic layout computation.
    
    Args:
        viz_data: Dictionary from extract_visualization_data() containing:
            - 'pure_adjacency_matrix': np.ndarray
            - 'ontology_adj': np.ndarray
            - 'gat_embeddings': np.ndarray
            - 'node_names': List[str]
    
    Raises:
        ValueError: If required data is missing or shapes don't match.
    
    Example:
        >>> viz_data = extract_visualization_data('checkpoint.h5')
        >>> validate_checkpoint_data(viz_data)  # Raises if invalid
    
    Notes:
        - All matrices must have shape (n_nodes, n_nodes)
        - Embeddings must have shape (n_nodes, embedding_dim)
        - Node names list must have length n_nodes
    """
    required_keys = [
        'pure_adjacency_matrix',
        'ontology_adj',
        'gat_embeddings',
        'node_names'
    ]
    
    # Check all required keys present
    for key in required_keys:
        if key not in viz_data:
            logger.error(f"Checkpoint missing required key: {key}")
            raise ValueError(f"Checkpoint missing required key: {key}")
    
    # Validate shapes match
    n_nodes = len(viz_data['node_names'])
    
    if viz_data['pure_adjacency_matrix'].shape != (n_nodes, n_nodes):
        raise ValueError(
            f"pure_adjacency_matrix shape mismatch: "
            f"expected ({n_nodes}, {n_nodes}), "
            f"got {viz_data['pure_adjacency_matrix'].shape}"
        )
    
    if viz_data['ontology_adj'].shape != (n_nodes, n_nodes):
        raise ValueError(
            f"ontology_adj shape mismatch: "
            f"expected ({n_nodes}, {n_nodes}), "
            f"got {viz_data['ontology_adj'].shape}"
        )
    
    if viz_data['gat_embeddings'].shape[0] != n_nodes:
        raise ValueError(
            f"gat_embeddings shape mismatch: "
            f"expected {n_nodes} nodes, "
            f"got {viz_data['gat_embeddings'].shape[0]}"
        )
    
    # Additional validation: check for NaN/Inf values
    if np.any(np.isnan(viz_data['gat_embeddings'])):
        logger.warning("GAT embeddings contain NaN values")
    
    if np.any(np.isinf(viz_data['gat_embeddings'])):
        logger.warning("GAT embeddings contain Inf values")
    
    embedding_dim = viz_data['gat_embeddings'].shape[1]
    logger.debug(f"Checkpoint validation passed: {n_nodes} nodes, {embedding_dim}-dim embeddings")




def compute_node_depths(adjacency_matrix: np.ndarray) -> Dict[int, int]:
    """Compute shortest-path ontology depth for each node via BFS from roots.

    Performs a breadth-first traversal starting from all root nodes (nodes
    with no incoming edges) and assigns each node the length of the shortest
    path from any root.  First-visit semantics guarantee shortest-path depth
    in a DAG.

    Args:
        adjacency_matrix: [n_nodes, n_nodes] array where
            ``adjacency_matrix[i, j] = 1`` means there is an edge from node
            ``i`` to node ``j`` (parent→child convention).  Roots are nodes
            whose column sum is 0 (no incoming edges).

    Returns:
        Dict mapping node index → shortest-path depth from the nearest root.
        Disconnected nodes are assigned depth 0.
    """
    n_nodes = len(adjacency_matrix)
    in_degree = np.sum(adjacency_matrix, axis=0)
    roots = np.where(in_degree == 0)[0]

    depths: Dict[int, int] = {int(r): 0 for r in roots}
    queue = list(roots)

    while queue:
        node = queue.pop(0)
        children = np.where(adjacency_matrix[node] > 0)[0]
        for child in children:
            child = int(child)
            if child not in depths:
                depths[child] = depths[int(node)] + 1
                queue.append(child)

    # Handle any remaining nodes (disconnected)
    for i in range(n_nodes):
        if i not in depths:
            depths[i] = 0

    return depths


def _apply_direction_filter(
    inferred_path_weights: Dict[Tuple[int, int], float],
    lateral_directions: Dict[Tuple[int, int], Tuple[int, int]],
    node_depths_dag: Dict[int, int],
    node_names: List[str],
    id_to_name_map: Dict[str, str],
) -> Tuple[Dict[Tuple[int, int], float], Dict[str, Any]]:
    """Filter inferred-path edges to a single canonical direction per pair.

    Combines three direction sources:
      1. Cross-level (different depths): keep low-depth → high-depth orientation.
      2. Lateral (same depth): keep the embedding-gradient direction stored in
         ``lateral_directions``.
      3. Semantic name override: a count-based predicate
         (`hector.trajectory_ontology._semantic_direction`) scores each side by how
         many precursor naming patterns it matches (e.g., "immature ", "early ",
         "pre-", "fetal ", "stem cell", "progenitor", "precursor", "poietic",
         "blast" — see `hector.trajectory_ontology` for the full list). The side with
         the higher precursor count becomes the source, overriding the depth
         and gradient defaults. Terminal patterns ("terminally differentiated")
         act symmetrically — higher terminal count becomes the target.

    Inputs ``inferred_path_weights`` contains both ``(i, j)`` and ``(j, i)`` for
    every discovered pair (symmetric k-NN); this function collapses each pair to
    one kept entry.

    Returns the filtered weight dict plus a counters dict for transparency:
    ``{"n_wrong_direction", "n_same_depth", "n_lateral_skipped",
       "n_semantic_overrides_cross", "n_semantic_overrides_lateral",
       "semantic_examples"}``.
    """

    def _name_of(idx: int) -> str:
        cl_id = node_names[idx]
        return id_to_name_map.get(cl_id, cl_id) if id_to_name_map else cl_id

    direction_filtered: Dict[Tuple[int, int], float] = {}
    n_wrong_direction = 0
    n_same_depth = 0
    n_lateral_skipped = 0
    n_semantic_overrides_cross = 0
    n_semantic_overrides_lateral = 0
    semantic_examples: List[str] = []

    # Cross-level pairs whose direction has already been resolved by the semantic
    # override — used to skip the symmetric reciprocal entry on its own iteration.
    overridden_cross: Set[frozenset] = set()
    # Lateral pairs whose direction has already been resolved by the semantic
    # override — bypasses the lateral-canonical lookup for the reciprocal.
    overridden_lateral: Set[frozenset] = set()

    def _record_example(src_idx: int, tgt_idx: int) -> None:
        if len(semantic_examples) < 5:
            semantic_examples.append(f"{_name_of(src_idx)} -> {_name_of(tgt_idx)}")

    for (src, tgt), weight in inferred_path_weights.items():
        src_depth = node_depths_dag.get(src, 0)
        tgt_depth = node_depths_dag.get(tgt, 0)
        pair_key = frozenset((src, tgt))

        if src_depth != tgt_depth:
            # --- Cross-level pair ---
            if pair_key in overridden_cross:
                # Reciprocal already handled by the override branch below.
                continue

            verdict = _semantic_direction(_name_of(src), _name_of(tgt))
            if verdict is not None:
                # Semantic verdict overrides the depth rule.
                if verdict == (0, 1):
                    kept_src, kept_tgt = src, tgt
                else:
                    kept_src, kept_tgt = tgt, src
                kept_weight = inferred_path_weights.get(
                    (kept_src, kept_tgt), weight
                )
                direction_filtered[(kept_src, kept_tgt)] = kept_weight
                overridden_cross.add(pair_key)
                n_semantic_overrides_cross += 1
                _record_example(kept_src, kept_tgt)
                continue

            # Default depth-based behavior.
            if src_depth < tgt_depth:
                direction_filtered[(src, tgt)] = weight
            else:
                n_wrong_direction += 1
        else:
            # --- Lateral pair (same depth) ---
            if pair_key in overridden_lateral:
                # Already resolved by the semantic override; skip the reciprocal.
                n_lateral_skipped += 1
                continue

            verdict = _semantic_direction(_name_of(src), _name_of(tgt))
            if verdict is not None:
                # Semantic verdict overrides the embedding-gradient direction.
                if verdict == (0, 1):
                    kept_src, kept_tgt = src, tgt
                else:
                    kept_src, kept_tgt = tgt, src
                kept_weight = inferred_path_weights.get(
                    (kept_src, kept_tgt), weight
                )
                direction_filtered[(kept_src, kept_tgt)] = kept_weight
                overridden_lateral.add(pair_key)
                n_semantic_overrides_lateral += 1
                n_same_depth += 1
                _record_example(kept_src, kept_tgt)
                continue

            # Default lateral behavior: keep entry that matches the canonical
            # embedding-gradient direction stored in lateral_directions.
            canonical = lateral_directions.get((src, tgt))
            if canonical is None:
                direction_filtered[(src, tgt)] = weight
                n_same_depth += 1
            elif canonical == (src, tgt):
                direction_filtered[(src, tgt)] = weight
                n_same_depth += 1
            else:
                n_lateral_skipped += 1

    counters = {
        "n_wrong_direction": n_wrong_direction,
        "n_same_depth": n_same_depth,
        "n_lateral_skipped": n_lateral_skipped,
        "n_semantic_overrides_cross": n_semantic_overrides_cross,
        "n_semantic_overrides_lateral": n_semantic_overrides_lateral,
        "semantic_examples": semantic_examples,
    }
    return direction_filtered, counters


# =============================================================================
# Relocated Contiguous Blocks From trajectory_support.py
# =============================================================================
class SimpleOBOParser:
    """
    Dependency-free parser for OBO files to extract strict IS_A hierarchy.
    """
    @staticmethod
    def parse(obo_path: str) -> nx.DiGraph:
        G = nx.DiGraph()
        
        current_id = None
        
        with open(obo_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                if line == '[Term]':
                    current_id = None
                    continue
                
                # Parse ID
                if line.startswith('id:'):
                    # Format: id: CL:0000000
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        current_id = parts[1].strip().split(' ')[0] # Remove comments if any
                        G.add_node(current_id)
                
                # Parse IS_A (Parent)
                elif line.startswith('is_a:') and current_id:
                    # Format: is_a: CL:0000115 ! endothelial cell
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        parent_raw = parts[1].strip()
                        parent_id = parent_raw.split('!')[0].strip().split(' ')[0]
                        
                        # Add Edge: Parent -> Child (Standard Tree Flow)
                        # OBO says: Child IS_A Parent.
                        # Visualization Flow: Parent -> Child.
                        G.add_edge(parent_id, current_id)
                        
        return G


class CanonicalOrderComputer:
    """Computes and caches a deterministic total ordering of ontology nodes.
    
    Reuses the existing Child→Parent graph from AdaptiveOntologySubgraph.nx_graph
    directly — no graph reversal or rebuilding needed. Children are accessed via
    graph.predecessors(node), consistent with get_subtree_nodes() and other
    existing traversal functions.
    """
    
    def __init__(self, nx_graph: nx.DiGraph, node_names: List[str]):
        self.node_names = node_names
        self.nx_graph = nx_graph
        self.canonical_rank: Dict[int, int] = {}
        self.structural_rank: Dict[int, int] = {}
        self._subtree_sizes: Dict[int, int] = {}
        self._subtree_depths: Dict[int, int] = {}
        self._compute()
        self._compute_structural()

    def _compute(self) -> None:
        visited = set()
        rank = 0

        roots = [n for n in self.nx_graph.nodes() if self.nx_graph.out_degree(n) == 0]
        roots = sorted(roots, key=lambda n: self.node_names[n] if n < len(self.node_names) else "")

        def dfs(node):
            nonlocal rank
            if node in visited:
                return
            visited.add(node)
            self.canonical_rank[node] = rank
            rank += 1

            children = list(self.nx_graph.predecessors(node))
            children = sorted(children, key=lambda c: self.node_names[c] if c < len(self.node_names) else "")
            for child in children:
                dfs(child)

        for root in roots:
            dfs(root)

        all_nodes = sorted(self.nx_graph.nodes())
        unreachable = [n for n in all_nodes if n not in visited]
        unreachable = sorted(unreachable, key=lambda n: self.node_names[n] if n < len(self.node_names) else "")
        for node in unreachable:
            if node not in self.canonical_rank:
                self.canonical_rank[node] = rank
                rank += 1

    def _compute_structural(self) -> None:
        """Compute subtree sizes and depths, then assign structural ranks.

        Structural ordering: larger subtrees first (more angular space on
        the left), deeper subtrees later, alphabetical tie-break.
        """
        # nx_graph uses Child→Parent edges, so predecessors = children
        def _subtree_size(node, visited):
            if node in visited:
                return 0
            visited.add(node)
            size = 1
            for child in self.nx_graph.predecessors(node):
                size += _subtree_size(child, visited)
            return size

        def _subtree_depth(node, visited):
            if node in visited:
                return 0
            visited.add(node)
            children = list(self.nx_graph.predecessors(node))
            if not children:
                return 0
            return 1 + max(_subtree_depth(c, visited) for c in children)

        for n in self.nx_graph.nodes():
            self._subtree_sizes[n] = _subtree_size(n, set())
            self._subtree_depths[n] = _subtree_depth(n, set())

        # DFS with structural child ordering
        visited: Set[int] = set()
        rank = 0
        roots = [n for n in self.nx_graph.nodes() if self.nx_graph.out_degree(n) == 0]
        roots = self._structural_sort(roots)

        def dfs_structural(node):
            nonlocal rank
            if node in visited:
                return
            visited.add(node)
            self.structural_rank[node] = rank
            rank += 1
            children = list(self.nx_graph.predecessors(node))
            children = self._structural_sort(children)
            for child in children:
                dfs_structural(child)

        for root in roots:
            dfs_structural(root)

        for n in self.nx_graph.nodes():
            if n not in self.structural_rank:
                self.structural_rank[n] = rank
                rank += 1

    def _structural_sort(self, nodes: List[int]) -> List[int]:
        """Sort nodes by subtree size (desc), depth (asc), name (asc)."""
        def _key(n):
            name = self.node_names[n] if n < len(self.node_names) else ""
            return (-self._subtree_sizes.get(n, 1),
                    self._subtree_depths.get(n, 0),
                    name)
        return sorted(nodes, key=_key)

    def get_rank(self, node_index: int) -> int:
        return self.canonical_rank.get(node_index, len(self.node_names))

    def get_structural_rank(self, node_index: int) -> int:
        """Rank based on ontology subtree structure rather than alphabetical order."""
        return self.structural_rank.get(node_index, len(self.node_names))

@dataclass
class SiblingOrder:
    """Deterministic sibling ordering for radial tree layout.

    Stores the resolved ordering of children for each parent node,
    keyed by original DAG indices. This ordering survives DAG-to-tree
    conversion and can be shared across samples for comparison plots.
    """
    parent_to_children: Dict[int, List[int]]


class TreeLayoutEngine:
    """
    Compute two-dimensional coordinates for an ontology DAG.

    The layout locks X to topological or elastic depth and obtains Y through
    iterative force-directed relaxation, allowing nodes with multiple parents.
    """
    
    def __init__(self, adjacency_matrix: np.ndarray, node_names: List[str], config: Optional['TrajectoryAnalysisConfig'] = None):
        """
        Initialize the layout engine.

        Args:
            adjacency_matrix: [n_nodes, n_nodes] ontology adjacency matrix
            node_names: List of node names
            config: TrajectoryAnalysisConfig object
        """
        self.adjacency_matrix = adjacency_matrix
        self.node_names = node_names
        self.config = config
        self.n_nodes = len(node_names)
        self.node_embeddings = None # Set later
        self.subtree_sizes = {}
        self.canonical_order: Optional[CanonicalOrderComputer] = None
        self.original_indices: Optional[List[int]] = None
        # Per-node rendered branch widths (in HTML display pixels — the
        # value the renderer would draw for a non-PDF backend). Populated
        # in compute_layout() so the placer can scale transition-cell
        # jitter to a fraction of the local branch width, matching the
        # convention used by RadialTreeLayoutEngine.
        self.node_widths: Dict[int, float] = {}
        
    def set_embeddings(self, embeddings: np.ndarray):
        """Set embeddings for spectral sorting."""
        self.node_embeddings = embeddings
        
    def compute_layout(
        self, 
        active_indices: List[int],
        augmented_adjacency: Optional[np.ndarray] = None
    ) -> Dict[int, Tuple[float, float]]:
        """
        Compute 2D positions using a PHYSICS-BASED LAYOUT on a DAG.
        
        This is a "Lock X, Float Y" simulation:
        - X-coordinate = topological depth (locked, determined by longest path from root)
          OR elastic depths if self.elastic_depths is set
        - Y-coordinate = computed via iterative force-directed relaxation
        
        This properly handles DAGs where a node may have multiple parents (convergent evolution).
        
        Args:
            active_indices: Nodes to include in layout
            augmented_adjacency: Optional adjacency with inferred paths (default: self.adjacency_matrix)
        """
        from .trajectory_support import RIBBON_BASE_HW_PX
        if not active_indices:
            return {}
        if len(active_indices) == 1:
            return {active_indices[0]: (0.0, 0.0)}
        
        # Use augmented adjacency if provided, else use self.adjacency_matrix
        adjacency = augmented_adjacency if augmented_adjacency is not None else self.adjacency_matrix
            
        # 1. Build Full Graph from Adjacency (Parent -> Child flow for layout)
        G_full = nx.DiGraph()
        rows, cols = np.where(adjacency > 0)
        for r, c in zip(rows, cols):
            # adjacency[row, col] = 1 means row -> col edge
            # Our convention: Child -> Parent in adjacency
            # So r is Child, c is Parent
            # For layout, we want Parent -> Child flow
            G_full.add_edge(int(c), int(r))  # Flip: Parent -> Child

        # Add all nodes
        for i in range(len(self.node_names)):
            if i not in G_full:
                G_full.add_node(i)

        # Diagnose cycles before projecting the active subgraph.
        _cycles_in_G_full = list(nx.simple_cycles(G_full))
        if _cycles_in_G_full:
            logger.warning(
                "Ontology layout graph contains %d cycle(s); cyclic "
                "reachability pairs will be skipped.",
                len(_cycles_in_G_full),
            )
            logger.debug("First ontology layout cycle: %s", _cycles_in_G_full[0])

        # --- PROJECT ACTIVE SUBGRAPH (Infer Transitive Edges) ---
        # Add an edge only when reachability is one-way, which avoids carrying
        # cycles into the projected active graph.
        active_set = set(active_indices)
        G = nx.DiGraph()
        G.add_nodes_from(active_indices)

        sorted_active = sorted(active_indices)

        for u in sorted_active:
            for v in sorted_active:
                if u == v: continue
                try:
                    has_path_uv = nx.has_path(G_full, u, v)
                    has_path_vu = nx.has_path(G_full, v, u)

                    # Only add edge if path exists in ONE direction (not both)
                    # If both directions have paths, it's a cycle - skip both
                    if has_path_uv and not has_path_vu:
                        G.add_edge(u, v)
                    # Note: if has_path_vu and not has_path_uv, the edge v->u
                    # will be added when we iterate with v as u
                except nx.NetworkXError:
                    pass

        # Diagnose cycles before transitive reduction.
        _cycles_before_reduction = list(nx.simple_cycles(G))
        if _cycles_before_reduction:
            logger.warning(
                "Projected layout graph contains %d cycle(s) before "
                "transitive reduction.",
                len(_cycles_before_reduction),
            )
            logger.debug(
                "First projected layout cycle: %s",
                _cycles_before_reduction[0],
            )

        # Transitive reduction to keep only direct edges
        try:
            G = nx.transitive_reduction(G)
        except Exception as e:
            logger.warning("Transitive reduction failed: %s", e)

        # Diagnose cycles after transitive reduction.
        _cycles_in_G = list(nx.simple_cycles(G))
        if _cycles_in_G:
            logger.error(
                "Projected layout graph contains %d cycle(s) after "
                "transitive reduction.",
                len(_cycles_in_G),
            )
            logger.debug("First remaining layout cycle: %s", _cycles_in_G[0])
            if hasattr(self, 'node_names'):
                cycle_names = [self.node_names[n] if n < len(self.node_names) else f"idx_{n}" for n in _cycles_in_G[0]]
                logger.debug("First remaining layout cycle names: %s", cycle_names)

        # Store tree edges for drawing (these are the DAG edges)
        self.tree_edges = list(G.edges())
        self.tree = G

        # --- PHASE 1: ASSIGN DEPTHS (X-Coordinates) ---
        # Use topological_generations which works with DAGs
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        if not roots:
            degrees = dict(G.out_degree())
            roots = [max(degrees, key=degrees.get)] if degrees else [active_indices[0]]
        
        # Compute depth as longest path from any root to each node
        node_depths = {}
        for node in G.nodes():
            node_depths[node] = 0
        
        # BFS-based depth calculation for DAGs (longest path = generation)
        # For DAGs with multiple paths, we want the LONGEST path to preserve hierarchy
        try:
            for generation, nodes in enumerate(nx.topological_generations(G)):
                for node in nodes:
                    node_depths[node] = generation
        except nx.NetworkXError as e:
            logger.error(
                "Topological generation failed for a graph with %d nodes "
                "and %d edges: %s",
                G.number_of_nodes(),
                G.number_of_edges(),
                e,
            )
            # Fallback if there are cycles (shouldn't happen after transitive reduction)
            for i, node in enumerate(active_indices):
                node_depths[node] = i % 5
        
        self.node_depths = node_depths
        max_depth = max(node_depths.values()) if node_depths else 0
        
        # --- PHASE 2: CROSSING MINIMIZATION (Sugiyama Style) ---
        # Instead of random initialization, we optimize the order of nodes at each depth
        # to minimize edge crossings.
        
        # 1. Group nodes by depth
        nodes_by_depth = {}
        for node in G.nodes():
            d = node_depths[node]
            if d not in nodes_by_depth:
                nodes_by_depth[d] = []
            nodes_by_depth[d].append(node)
            
        # 2. Add Virtual Nodes for long edges (needed for proper crossing minimization)
        # We modify the graph effectively for this phase
        virtual_nodes = {} # map id -> node
        layout_edges = []  # List of (u, v) for the layout graph
        
        # Track effective parents/children including virtuals
        effective_parents = defaultdict(list)
        effective_children = defaultdict(list)
        
        # Helper to add edge for layout
        def add_layout_edge(u, v):
            layout_edges.append((u, v))
            effective_children[u].append(v)
            effective_parents[v].append(u)
        
        # Process edges and insert virtuals
        for u, v in G.edges():
            d_u = node_depths[u]
            d_v = node_depths[v]
            
            if d_v > d_u + 1:
                # Long edge: u -> virt1 -> ... -> v
                prev = u
                for d in range(d_u + 1, d_v):
                    v_node = f"virt_{u}_{v}_{d}"
                    if v_node not in virtual_nodes:
                        virtual_nodes[v_node] = {'depth': d}
                        if d not in nodes_by_depth: nodes_by_depth[d] = []
                        nodes_by_depth[d].append(v_node)
                        node_depths[v_node] = d # Add to global lookup
                    
                    add_layout_edge(prev, v_node)
                    prev = v_node
                add_layout_edge(prev, v)
            else:
                add_layout_edge(u, v)
        
        # 3. Minimize Crossings (Barycenter Sweeps)
        # Initial order: Spectral check or just current order
        for d, nodes in nodes_by_depth.items():
            if len(nodes) > 1:
                 real_nodes = [n for n in nodes if not isinstance(n, str) or not n.startswith("virt_")]
                 virt_nodes = [n for n in nodes if isinstance(n, str) and n.startswith("virt_")]

                 if self.canonical_order is not None:
                     # Map tree indices to original ontology indices for canonical lookup
                     def _canonical_rank(n):
                         orig = self.original_indices[n] if self.original_indices else n
                         return self.canonical_order.get_rank(orig)
                     sorted_real = sorted(real_nodes, key=_canonical_rank)
                 else:
                     sorted_real = self._sort_children_spectral(real_nodes, None)

                 nodes_by_depth[d] = sorted_real + virt_nodes

        # Iterative Sweeps
        for _ in range(15): # 15 iterations usually sufficient
            # Forward Sweep (Root -> Leaves)
            for d in range(1, max_depth + 1):
                if d not in nodes_by_depth: continue
                
                # Sort nodes at depth d based on avg position of parents at d-1
                layer_nodes = nodes_by_depth[d]
                
                # Get current order of previous layer
                prev_layer = nodes_by_depth.get(d-1, [])
                prev_order = {n: i for i, n in enumerate(prev_layer)}
                
                def get_barycenter_parents(n):
                    parents = effective_parents[n]
                    if not parents: return 999999 # Push to end if no parents
                    avg_pos = np.mean([prev_order.get(p, 0) for p in parents])
                    return avg_pos
                
                nodes_by_depth[d] = sorted(layer_nodes, key=get_barycenter_parents)
                
            # Backward Sweep (Leaves -> Root)
            for d in range(max_depth - 1, -1, -1):
                if d not in nodes_by_depth: continue
                
                layer_nodes = nodes_by_depth[d]
                next_layer = nodes_by_depth.get(d+1, [])
                next_order = {n: i for i, n in enumerate(next_layer)}
                
                # Build current order for fallback (nodes without children keep their position)
                current_order = {n: i for i, n in enumerate(layer_nodes)}
                
                def get_barycenter_children(n):
                    children = effective_children[n]
                    if not children:
                        # Fallback: use current position to avoid shuffling leaves
                        return current_order.get(n, 0)
                    avg_pos = np.mean([next_order.get(c, 0) for c in children])
                    return avg_pos
                
                nodes_by_depth[d] = sorted(layer_nodes, key=get_barycenter_children)

        # --- PHASE 2b: ADJACENT-SWAP CROSSING REDUCTION ---
        # The barycenter heuristic above seeds a good initial order but can't
        # resolve ties (siblings sharing the same parent get identical
        # barycenters, tiebroken by alphabetical canonical order which is
        # topology-unaware). This phase completes Sugiyama step 2 by trying
        # adjacent swaps and keeping those that reduce actual crossing counts.
        self._swap_reduce_crossings(
            nodes_by_depth, effective_parents, effective_children, max_depth
        )

        # --- PHASE 3: PHYSICS SIMULATION FOR Y-Coordinates ---
        x_scale = self.config.x_scale if self.config else 4.0
        base_y_scale = self.config.y_scale if self.config else 0.5
        
        # Initial min_node_sep. Phase 6 refines this automatically from label geometry.
        min_node_sep = 3.0  # Reasonable default fallback
        
        # Use base_y_scale for initial layout, Phase 6 will enforce final separation
        y_scale = base_y_scale
        
        # Initialize positions based on Optimized Order
        positions = {}
        for d, nodes in nodes_by_depth.items():
            # Center the layer around 0
            layer_height = (len(nodes) - 1) * y_scale
            start_y = -layer_height / 2
            
            for i, node in enumerate(nodes):
                # Use elastic depths if available, else use rigid ontology depths
                if hasattr(self, 'elastic_depths') and self.elastic_depths is not None:
                    x = self.elastic_depths.get(node, node_depths[node]) * x_scale
                else:
                    x = node_depths[node] * x_scale
                y = start_y + i * y_scale
                positions[node] = [x, y]
        
        # Physics parameters
        iterations = 100
        alpha = 0.3  # Learning rate
        k_spring = 0.5  # Spring constant
        k_repulsion = 0.5  # Repulsion constant
        min_separation = min_node_sep  # Minimum Y separation at same depth (from config)
        
        # Build neighbor lookup (both parents and children)
        # We reuse the "layout_edges" logic which includes virtual nodes
        neighbors = {node: [] for node in positions}
        all_physics_nodes = list(positions.keys()) # Includes virtuals
        
        for u, v in layout_edges:
             neighbors[u].append(v)
             neighbors[v].append(u)
        
        # --- PHYSICS LOOP ---
        # We run the physics loop to "relax" the layout, but we must respect the 
        # ordering we established to prevent re-crossing.
        # Actually, standard forces might flip nodes if not careful. 
        # But if initialized well, they should settle in the local minimum (no crossing).
        
        for iteration in range(iterations):
            forces = {node: 0.0 for node in all_physics_nodes}
            
            # 1. SPRING FORCE: Pull towards average Y of neighbors
            for node in all_physics_nodes:
                node_neighbors = neighbors[node]
                if not node_neighbors:
                    continue
                
                avg_y = np.mean([positions[n][1] for n in node_neighbors])
                forces[node] += k_spring * (avg_y - positions[node][1])
            
            # 2. REPULSION FORCE: Push apart nodes at same depth
            # Preserve the optimized order during repulsion. Re-sorting by the
            # evolving Y coordinate would allow nodes to swap across branches.
            
            for d, depth_nodes in nodes_by_depth.items():
                if len(depth_nodes) < 2:
                    continue
                
                # Iterate adjacent pairs in the OPTIMIZED order
                for i in range(len(depth_nodes) - 1):
                    n1 = depth_nodes[i]
                    n2 = depth_nodes[i+1] # n2 should be strictly 'below' n1 (higher Y)
                    
                    dy = positions[n2][1] - positions[n1][1]
                    
                    # If they are too close OR if they have crossed (dy < 0), push them apart
                    # This acts as a hard topological constraint
                    if dy < min_separation:
                        # Repel
                        # If crossed (dy < 0), the force will be large and corrective
                        # Avoid singularity
                        if abs(dy) < 0.01: dy = 0.01 if dy >= 0 else -0.01 
                        
                        # Force magnitude: proportional to violation
                        # If dy < min_separation, this is positive
                        repel_force = k_repulsion * (min_separation - dy)
                        
                        forces[n1] -= repel_force
                        forces[n2] += repel_force
            
            # 3. Apply forces (only update Y, X is locked)
            for node in all_physics_nodes:
                if isinstance(node, str) and node.startswith("virt_"):
                     # Virtual nodes move freely
                     positions[node][1] += alpha * forces[node]
                else:
                     # Real nodes move freely too
                     positions[node][1] += alpha * forces[node]
            
            # Decay learning rate
            alpha *= 0.98
        
        # --- PHASE 4: NORMALIZE Y COORDINATES ---
        all_y = [positions[n][1] for n in G.nodes()]
        if not all_y: # Safety
             return {n: (0.0, 0.0) for n in active_indices}
             
        y_min, y_max = min(all_y), max(all_y)
        y_range = y_max - y_min if y_max > y_min else 1.0
        
        # Scale to fit nicely.
        # Use max-per-level rather than total node count: the widest depth
        # layer sets the true minimum y-span required to avoid same-level
        # overlap. Using len(active_indices) here over-allocates y-span by
        # an order of magnitude on typical ontologies (e.g. 273 for a
        # 273-node tree where at most ~40 nodes sit at the widest depth),
        # which fed directly into the horizontal-layout data-aspect blowup.
        max_per_level = max((len(nodes) for nodes in nodes_by_depth.values()), default=1)
        target_height = max_per_level * y_scale
        
        for node in G.nodes():
            normalized_y = (positions[node][1] - y_min) / y_range
            positions[node][1] = normalized_y * target_height
        
        # --- PHASE 5: POST-PHYSICS ORDER ENFORCEMENT ---
        # The physics simulation determines good *spacing* but may have caused node swaps
        # (crossings) within a depth layer. We enforce the Sugiyama-optimized order
        # by reassigning Y-coordinates.
        #
        # Strategy:
        # 1. For each depth layer, get the Y-coordinates assigned by physics.
        # 2. Sort these Y-coordinates numerically (these are the "slots").
        # 3. Assign the sorted slots to nodes in the optimized_order from Sugiyama.
        # This preserves physics spacing while guaranteeing crossing-free topology.
        
        for d, optimized_node_order in nodes_by_depth.items():
            if len(optimized_node_order) < 2:
                continue
            
            # Get the physics-computed Y-coords for these nodes
            physics_y_coords = [positions[n][1] for n in optimized_node_order if n in positions]
            
            if len(physics_y_coords) != len(optimized_node_order):
                # Some nodes missing (shouldn't happen), skip enforcement for this layer
                continue
            
            # Sort the Y-coords to get the "slots"
            sorted_slots = sorted(physics_y_coords)
            
            # Reassign: Give the lowest slot to the first node in optimized_order, etc.
            for i, node in enumerate(optimized_node_order):
                if node in positions:
                    positions[node][1] = sorted_slots[i]
        
        # --- PHASE 6: OVERLAP RESOLUTION (FINAL GUARANTEE) ---
        # Group REAL nodes (from active_indices only) by their X-coordinate (depth),
        # then enforce minimum Y-separation within each group.
        #
        # This completely ignores the nodes_by_depth dict which contains virtual nodes.
        
        # Step 0: AUTO-CALCULATE min_node_sep based on font size and figure dimensions
        if self.config:
            # Get current Y-range from real nodes
            real_y_values = [positions[n][1] for n in positions if n in set(active_indices)]
            
            if real_y_values:
                y_min_current = min(real_y_values)
                y_max_current = max(real_y_values)
                y_range_current = y_max_current - y_min_current
                
                if y_range_current > 0:
                    # Figure dimensions
                    fig_width, fig_height = self.config.pdf_figsize
                    
                    # Font size in points (1 point = 1/72 inch)
                    font_size_pt = self.config.label_font_size
                    
                    # Calculate how many layout units = 1 inch
                    # layout_units_per_inch = y_range_current / fig_height
                    
                    # Required separation in inches:
                    # - Font height (~1.5x font size for line height)
                    # - Label offset (10 points = 10/72 inch)
                    # - Margin buffer (50% extra)
                    required_inches = (font_size_pt * 1.5 + 10 + font_size_pt ) / 72.0
                    
                    # Convert inches to layout units
                    layout_units_per_inch = y_range_current / fig_height
                    min_node_sep = required_inches * layout_units_per_inch
                    
        # Step 1: Group real nodes by X-coordinate
        real_nodes_by_x = {}
        active_set = set(active_indices)
        
        for node in positions:
            if node not in active_set:
                continue  # Skip virtual nodes
            
            x_coord = positions[node][0]
            # Round X to handle floating point precision
            x_key = round(x_coord, 2)
            
            if x_key not in real_nodes_by_x:
                real_nodes_by_x[x_key] = []
            real_nodes_by_x[x_key].append(node)
        
        # Step 2: For each X-group, enforce minimum Y-separation
        for x_key, nodes_at_x in real_nodes_by_x.items():
            if len(nodes_at_x) < 2:
                continue
            
            # Sort by current Y position
            sorted_nodes = sorted(nodes_at_x, key=lambda n: positions[n][1])
            
            # Iteratively push nodes apart until all gaps satisfy min_node_sep
            for iteration in range(100):  # More iterations for safety
                violations_fixed = 0
                
                for i in range(len(sorted_nodes) - 1):
                    n1 = sorted_nodes[i]
                    n2 = sorted_nodes[i + 1]
                    
                    gap = positions[n2][1] - positions[n1][1]
                    
                    if gap < min_node_sep:
                        # Push n2 (and all nodes below it) down
                        deficit = min_node_sep - gap
                        
                        # Push all subsequent nodes down to maintain relative order
                        for j in range(i + 1, len(sorted_nodes)):
                            positions[sorted_nodes[j]][1] += deficit
                        
                        violations_fixed += 1
                
                if violations_fixed == 0:
                    break
        
        # Convert to tuples and ensure all nodes placed
        final_positions = {}
        for idx in active_indices:
            if idx in positions:
                final_positions[idx] = tuple(positions[idx])
            else:
                # Fallback for any missing nodes
                final_positions[idx] = (0.0, len(final_positions) * y_scale)

        # Per-node rendered branch widths (HTML display pixels), mirroring
        # the convention used by RadialTreeLayoutEngine. Horizontal ribbons
        # have uniform width per edge — taper is applied per-edge in the
        # renderer, not per-node — so every active node carries the same
        # base width. The placer then scales transition-cell jitter to a
        # fraction of this branch width, matching the radial behaviour.
        if self.config is not None:
            ribbon_full_hw_px = (
                2.0 * RIBBON_BASE_HW_PX * (self.config.edge_width / 1.5)
            )
            for idx in final_positions:
                self.node_widths[idx] = ribbon_full_hw_px

        return final_positions
    
    # -----------------------------------------------------------------
    # Adjacent-swap crossing reduction (Sugiyama step 2b)
    # -----------------------------------------------------------------

    def _swap_crossing_delta(
        self, n1, n2, depth, nodes_by_depth, effective_parents, effective_children
    ):
        """Count crossing pairs between edges of n1 and n2 before/after swap.

        n1 is currently above n2 (lower index). Returns (before, after) where
        'before' is the number of (n1-edge, n2-edge) crossings in the current
        order and 'after' is the count if the two were swapped.
        O(d1 * d2) per call where d1, d2 are degrees into adjacent layers.
        """
        before = 0
        after = 0

        for adj_depth, get_neighbors in (
            (depth - 1, lambda n: effective_parents[n]),
            (depth + 1, lambda n: effective_children[n]),
        ):
            adj_layer = nodes_by_depth.get(adj_depth)
            if not adj_layer:
                continue
            adj_pos = {n: i for i, n in enumerate(adj_layer)}

            t1s = [adj_pos[p] for p in get_neighbors(n1) if p in adj_pos]
            t2s = [adj_pos[p] for p in get_neighbors(n2) if p in adj_pos]

            for a in t1s:
                for b in t2s:
                    if a > b:
                        before += 1
                    elif b > a:
                        after += 1
        return before, after

    def _count_total_crossings(self, nodes_by_depth, effective_parents, max_depth):
        """Count total edge crossings across all adjacent layer pairs."""
        total = 0
        for d in range(1, max_depth + 1):
            layer = nodes_by_depth.get(d, [])
            prev_layer = nodes_by_depth.get(d - 1, [])
            if not layer or not prev_layer:
                continue
            prev_pos = {n: i for i, n in enumerate(prev_layer)}
            curr_pos = {n: i for i, n in enumerate(layer)}

            edges = []
            for n in layer:
                for p in effective_parents[n]:
                    if p in prev_pos:
                        edges.append((prev_pos[p], curr_pos[n]))
            for i in range(len(edges)):
                for j in range(i + 1, len(edges)):
                    if (edges[i][0] < edges[j][0]) != (edges[i][1] < edges[j][1]):
                        total += 1
        return total

    def _swap_reduce_crossings(
        self, nodes_by_depth, effective_parents, effective_children, max_depth
    ):
        """Adjacent-swap crossing reduction — completes Sugiyama step 2."""
        total_before = self._count_total_crossings(
            nodes_by_depth, effective_parents, max_depth
        )

        max_passes = 24
        for pass_num in range(max_passes):
            improved = False

            if pass_num % 2 == 0:
                depth_range = range(max_depth + 1)
            else:
                depth_range = range(max_depth, -1, -1)

            for d in depth_range:
                layer = nodes_by_depth.get(d, [])
                if len(layer) < 2:
                    continue
                for i in range(len(layer) - 1):
                    n1, n2 = layer[i], layer[i + 1]
                    before, after = self._swap_crossing_delta(
                        n1, n2, d, nodes_by_depth,
                        effective_parents, effective_children,
                    )
                    if after < before:
                        layer[i], layer[i + 1] = n2, n1
                        improved = True

            if not improved:
                break

        total_after = self._count_total_crossings(
            nodes_by_depth, effective_parents, max_depth
        )

    def _sort_children_spectral(self, children: List[int], parent: int) -> List[int]:
        """
        Sort children based on 1st Principal Component of their embeddings.
        This ensures similar siblings are placed adjacent to each other angularly.
        """
        if len(children) < 2:
            return children
        
        # If no embeddings, fallback to subtree size
        if self.node_embeddings is None:
             return sorted(children, key=lambda x: self.subtree_sizes.get(x, 1), reverse=True)

        try:
            # 1. Get embeddings
            child_embs = self.node_embeddings[children]

            # 2. PCA sorting (1D)
            if len(children) >= 3:
                # Skip PCA when child embeddings have no variance — common after
                # DAG-to-tree splitting where copies of the same DAG node inherit
                # identical vectors. PCA would emit a divide-by-zero warning and
                # return all-zero scores, which preserves input order anyway.
                if not np.any(np.std(child_embs, axis=0) > 1e-12):
                    return sorted(children, key=lambda x: self.subtree_sizes.get(x, 1), reverse=True)
                # Center data first
                # We want to capture the variation *orthogonal* to the parent vector?
                # Simpler: Just 1st PC of the cloud of children.
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                scores = pca.fit_transform(child_embs).flatten()

                # Pair zip and sort
                sorted_pairs = sorted(zip(children, scores), key=lambda x: x[1])
                return [c for c, s in sorted_pairs]
            else:
                # For 2 children, sort by similarity to parent?
                # Or just by similarity to each other? (doesn't matter)
                # Let's sort by distance to Grandparent?
                # Fallback to subtree size
                return sorted(children, key=lambda x: self.subtree_sizes.get(x, 1), reverse=True)

        except Exception as e:
            pass
            return children

# =============================================================================
# DAG to Tree Converter (Path Replication for Clean Radial Layout)
# =============================================================================

class DAGToTreeConverter:
    """
    Converts a DAG (Directed Acyclic Graph) into a Strict Tree by replicating nodes
    that have multiple parents. This creates 'clean independent sub-trees' for visualization.
    
    Key Logic:
    1. Traverses the DAG from roots to leaves.
    2. If a node is reached via multiple parents, it is DUPLICATED (Split).
       e.g., 'Macrophage' might appear in the 'Monocyte' branch AND the 'Tissue-Resident' branch.
    3. Cells assigned to a split node are re-assigned to the specific instance
       that best matches their trajectory (closest parent affinity).
    """
    
    def __init__(self, adjacency_matrix: np.ndarray, node_names: List[str], 
                 node_vectors: np.ndarray):
        self.raw_adj = adjacency_matrix
        self.raw_names = node_names
        self.node_vectors = node_vectors
        self.n_raw_nodes = len(node_names)
        
    def convert(self, active_indices: List[int], cell_predictions: np.ndarray, 
                cell_vectors: np.ndarray, min_cells_number: int = 10) -> Dict[str, Any]:
        """
        Execute the conversion.
        
        Args:
            active_indices: DAG node indices surviving SCC pruning.
            cell_predictions: Per-cell predicted node index (DAG space).
            cell_vectors: Per-cell embedding vectors.
            min_cells_number: Minimum subtree cell count for a tree node to be kept.
                Nodes whose entire subtree (direct + descendants) has fewer cells than
            this threshold are pruned. Structural ancestors with 0 direct cells are
                kept as long as their subtree total meets the threshold. Default: 10.

        Returns a dictionary containing:
        - tree_adj: Adjacency matrix of the new Strict Tree
        - tree_names: Names of the new nodes (some duplicated)
        - tree_vectors: Embeddings for new nodes
        - tree_predictions: Cell predictions mapped to new node indices
        - active_indices: The new active indices (0 to N_new)
        - original_indices: Mapping back to original DAG node IDs
        """
        from collections import defaultdict
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 1. Build NetworkX Graph of the ACTIVE subgraph only
        # 1. Build FULL Graph to Infer Transitive Edges
        # This is critical: if A -> B -> C and B is filtered out, we must find A -> C
        G_full = nx.DiGraph()
        rows, cols = np.where(self.raw_adj > 0)
        for r, c in zip(rows, cols):
             # Adjacency: Child(r), Parent(c) -> Edge: Parent(c) -> Child(r)
             G_full.add_edge(int(c), int(r))

        # Add all nodes to ensure connectivity checks work for everyone
        for i in range(self.n_raw_nodes):
            if i not in G_full:
                G_full.add_node(i)

        # 2. Project Active Subgraph
        G_raw = nx.DiGraph()
        G_raw.add_nodes_from(active_indices)

        sorted_active = sorted(active_indices)
        active_set = set(active_indices)

        # Check for paths between all pairs of active nodes
        for u in sorted_active:
            for v in sorted_active:
                if u == v: continue
                try:
                    if nx.has_path(G_full, u, v):
                        G_raw.add_edge(u, v)
                except nx.NetworkXError:
                    pass

        # Transitive Transitive Reduction
        try:
            G_raw = nx.transitive_reduction(G_raw)
        except Exception:
            pass

        # 3. Identify Roots
        roots = [n for n in G_raw.nodes() if G_raw.in_degree(n) == 0]
        if not roots:
             # Fallback for cycles
             roots = [active_indices[0]] if active_indices else []

        if not roots:
            # Empty graph fallback
            return {
                'tree_adj': np.zeros((0, 0), dtype=np.float32),
                'tree_names': [],
                'tree_vectors': np.zeros((0, self.node_vectors.shape[1])),
                'tree_predictions': cell_predictions,
                'active_indices': [],
                'original_indices': []
            }

        # 3. BFS Path Expansion (The "Unrolling")
        # We create a new node for EVERY edge traversal
        # This ensures that if we arrive at a node from a specific parent, we create a unique instance
        
        new_nodes = []  # List of dicts: {'orig_idx': int, 'new_parent': int, 'depth': int}
        queue = []  # (original_idx, new_parent_idx, depth)
        
        # Initialize roots
        for r in roots:
            new_idx = len(new_nodes)
            new_nodes.append({'orig_idx': r, 'new_parent': -1, 'depth': 0})
            queue.append((r, new_idx, 0))
            
        # To prevent infinite loops in cyclic graphs, track path length
        MAX_DEPTH = 20 
        
        while queue:
            orig_curr, new_curr, depth = queue.pop(0)
            
            if depth >= MAX_DEPTH: 
                continue
            
            # Find children in original DAG
            children = list(G_raw.successors(orig_curr))
            
            for child in children:
                # ALWAYS create a new node for the child (Replication Strategy)
                # This unrolls every edge into a tree branch
                new_child_idx = len(new_nodes)
                new_nodes.append({'orig_idx': child, 'new_parent': new_curr, 'depth': depth + 1})
                
                queue.append((child, new_child_idx, depth + 1))
        
        # 4. Construct New Tree Adjacency & Metadata
        n_tree_nodes = len(new_nodes)
        tree_adj = np.zeros((n_tree_nodes, n_tree_nodes), dtype=np.float32)
        tree_names = []
        tree_vectors = np.zeros((n_tree_nodes, self.node_vectors.shape[1]))
        original_map = []
        
        # Build lookup: original_idx -> list of [new_indices]
        orig_to_new_map = defaultdict(list)
        
        for i, node_info in enumerate(new_nodes):
            orig_idx = node_info['orig_idx']
            parent_idx = node_info['new_parent']
            
            # Adjacency: Child(i), Parent(parent_idx) -> Edge: Parent->Child
            # Our matrix convention: adj[child, parent] = 1
            if parent_idx != -1:
                tree_adj[i, parent_idx] = 1.0
            
            tree_names.append(self.raw_names[orig_idx])
            tree_vectors[i] = self.node_vectors[orig_idx]
            original_map.append(orig_idx)
            orig_to_new_map[orig_idx].append(i)

        # 5. Re-assign Cells to Specific Split Nodes
        # For a cell originally predicted as Node X:
        # We must decide: Does it belong to Node X (child of A) or Node X (child of B)?
        # Strategy: Assign to the instance whose PARENT is closest to the cell.
        
        new_predictions = np.zeros_like(cell_predictions)
        
        # Pre-compute parent vectors for every new node
        new_node_parent_vectors = np.zeros((n_tree_nodes, self.node_vectors.shape[1]))
        
        for i, info in enumerate(new_nodes):
            p_idx = info['new_parent']
            if p_idx != -1:
                new_node_parent_vectors[i] = tree_vectors[p_idx]
            else:
                new_node_parent_vectors[i] = tree_vectors[i]  # Root falls back to self
        
        # Process unique original predictions to batch operations
        unique_preds = np.unique(cell_predictions)
        
        for orig_pred in unique_preds:
            if orig_pred not in orig_to_new_map: 
                continue
            
            possible_targets = orig_to_new_map[orig_pred]
            
            # If only one option, easy
            if len(possible_targets) == 1:
                mask = (cell_predictions == orig_pred)
                new_predictions[mask] = possible_targets[0]
                continue
            
            # If multiple options (split node), use Parent Affinity
            mask = (cell_predictions == orig_pred)
            cells_in_class = cell_vectors[mask]
            
            # Candidate parent vectors
            candidate_parent_vecs = new_node_parent_vectors[possible_targets]
            
            # Compute similarity: Cells x Candidates
            sims = cosine_similarity(cells_in_class, candidate_parent_vecs)
            
            # Pick best match
            best_candidate_local_indices = np.argmax(sims, axis=1)
            best_candidate_global_indices = np.array(possible_targets)[best_candidate_local_indices]
            
            new_predictions[mask] = best_candidate_global_indices
        
        # 6. Prune Underpopulated Branches
        # A node is kept only if its subtree (direct cells + all descendants) meets
        # min_cells_number. Structural ancestors with 0 direct cells are preserved as
        # long as their subtree total is sufficient — this keeps the tree connected.

        # Count direct cells per tree node
        node_cell_counts = np.zeros(n_tree_nodes, dtype=int)
        for pred in new_predictions:
            node_cell_counts[pred] += 1

        # Build parent / children lookups
        node_parents = [-1] * n_tree_nodes
        for i, info in enumerate(new_nodes):
            node_parents[i] = info['new_parent']

        node_children = [[] for _ in range(n_tree_nodes)]
        for i in range(n_tree_nodes):
            p = node_parents[i]
            if p != -1:
                node_children[p].append(i)

        # Iterative leaf pruning: repeatedly remove leaf nodes whose direct
        # cell count is below the threshold.  A single-pass subtree approach
        # leaves orphan leaves when children that inflated a parent's subtree
        # total are themselves pruned.
        alive: Set[int] = set(range(n_tree_nodes))
        alive_children: Dict[int, Set[int]] = {i: set() for i in range(n_tree_nodes)}
        for i in range(n_tree_nodes):
            p = node_parents[i]
            if p != -1:
                alive_children[p].add(i)

        changed = True
        while changed:
            changed = False
            leaves_to_remove = [
                i for i in alive
                if not alive_children[i]
                and node_cell_counts[i] < min_cells_number
            ]
            for leaf in leaves_to_remove:
                alive.discard(leaf)
                p = node_parents[leaf]
                if p != -1:
                    alive_children[p].discard(leaf)
                changed = True

        keep_indices = sorted(alive)

        if len(keep_indices) < n_tree_nodes:
            # Re-index: Create mapping from old indices to new indices
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}

            # Drop cells whose assigned tree node was pruned
            keep_set = set(keep_indices)
            cell_keep_mask = np.array([int(p) in keep_set for p in new_predictions])
            new_predictions = new_predictions[cell_keep_mask]
            cell_vectors = cell_vectors[cell_keep_mask]

            # Build pruned tree
            n_pruned = len(keep_indices)
            pruned_adj = np.zeros((n_pruned, n_pruned), dtype=np.float32)
            pruned_names = []
            pruned_vectors = np.zeros((n_pruned, tree_vectors.shape[1]))
            pruned_original_map = []
            
            for new_idx, old_idx in enumerate(keep_indices):
                pruned_names.append(tree_names[old_idx])
                pruned_vectors[new_idx] = tree_vectors[old_idx]
                pruned_original_map.append(original_map[old_idx])
                
                # Update adjacency
                old_parent = node_parents[old_idx]
                if old_parent != -1 and old_parent in old_to_new:
                    new_parent = old_to_new[old_parent]
                    pruned_adj[new_idx, new_parent] = 1.0
            
            # Re-index predictions to new node indices (all pruned nodes already removed)
            pruned_predictions = np.array([old_to_new[int(p)] for p in new_predictions])
            return {
                'tree_adj': pruned_adj,
                'tree_names': pruned_names,
                'tree_vectors': pruned_vectors,
                'tree_predictions': pruned_predictions,
                'active_indices': list(range(n_pruned)),
                'original_indices': pruned_original_map,
                'cell_keep_mask': cell_keep_mask,
            }

        return {
            'tree_adj': tree_adj,
            'tree_names': tree_names,
            'tree_vectors': tree_vectors,
            'tree_predictions': new_predictions,
            'active_indices': list(range(n_tree_nodes)),
            'original_indices': original_map,
            'cell_keep_mask': np.ones(len(new_predictions), dtype=bool),
        }


# =============================================================================
# Radial Tree Layout Engine (Recursive Sector Partitioning)
# =============================================================================


def _generate_color_variants(base_hex: str, mcolors) -> Dict[str, str]:
    """
    Generate color variants for gradient and background from a base color.
    
    Returns a dictionary with:
    - 'start': The base color (saturated, for shallow nodes)
    - 'end': A lighter variant (for deep nodes)  
    - 'bg': A very light variant (for clade background sectors)
    
    Args:
        base_hex: Base color in hex format (e.g., "#E64B35")
        mcolors: matplotlib.colors module
        
    Returns:
        Dictionary with 'start', 'end', 'bg' color variants
    """
    rgb = mcolors.to_rgb(base_hex)
    
    # Generate lighter variant for gradient end (blend toward white)
    light_factor = 0.7
    end_rgb = tuple(c + (1.0 - c) * light_factor for c in rgb)
    
    # Generate background variant (even lighter, for sector backgrounds)
    bg_factor = 0.85
    bg_rgb = tuple(c + (1.0 - c) * bg_factor for c in rgb)
    
    return {
        'start': base_hex,
        'end': mcolors.to_hex(end_rgb),
        'bg': mcolors.to_hex(bg_rgb)
    }


def _build_placer_pixel_geometry(config, node_positions, active_indices, is_polar, output_format):
    """Build the ``PixelGeometry`` used by ``LatentBarycentricPlacer``.

    Mirrors the (x,y)-range + margin logic inside each renderer so the
    placer and the renderer size their glyphs against the same viewport.
    Returns ``None`` if the lazy imports aren't available — the placer
    has a safe data-diagonal-based fallback for that case.
    """
    try:
        from .trajectory_render import (
            pixel_geometry_from_matplotlib,
            pixel_geometry_from_plotly,
        )
    except Exception:  # pragma: no cover
        return None

    if is_polar:
        outer_r = getattr(config, 'radial_max_radius', 12.0)
        plot_range = outer_r * 1.15
        x_range = (-plot_range, plot_range)
        y_range = (-plot_range, plot_range)
    else:
        node_x = [node_positions[i][0] for i in active_indices if i in node_positions]
        node_y = [node_positions[i][1] for i in active_indices if i in node_positions]
        if not node_x or not node_y:
            return None
        x_pad = (max(node_x) - min(node_x)) * 0.08
        y_pad = (max(node_y) - min(node_y)) * 0.05
        x_range = (min(node_x) - x_pad, max(node_x) + x_pad)
        y_range = (min(node_y) - y_pad, max(node_y) + y_pad)

    if output_format == 'pdf':
        if is_polar:
            fig_in = getattr(config, 'radial_pdf_figsize', (20.0, 20.0))
            margins_in = (0.0, 0.0, 0.0, 0.0)
        else:
            fig_in = config.pdf_figsize
            margins_in = (0.9, 0.6, 1.2, 1.2)
        return pixel_geometry_from_matplotlib(
            figsize_inches=fig_in,
            dpi=config.pdf_dpi,
            x_range=x_range,
            y_range=y_range,
            margins_inches=margins_in,
        )

    # HTML / Plotly — margins MUST match the renderer's update_layout
    # margin dict so plot-area pixels are identical on both sides.
    if is_polar:
        margins_px = (10, 10, 40, 60)
    else:
        margins_px = (20, 20, 120, 200)
    return pixel_geometry_from_plotly(
        html_plot_width=config.html_plot_width,
        html_plot_height=config.html_plot_height,
        x_range=x_range,
        y_range=y_range,
        margins=margins_px,
    )


def _place_simplified_cells(
    config: 'TrajectoryAnalysisConfig',
    node_positions: Dict[int, Tuple[float, float]],
    node_vectors: np.ndarray,
    predictions: np.ndarray,
    cell_vectors: np.ndarray,
    visual_tree_edges: List[Tuple[int, int]],
    active_indices: List[int],
    node_counts: Dict[int, int],
    is_polar: bool,
    output_format: str,
    transition_state: Optional[np.ndarray] = None,
    transition_source_idx: Optional[np.ndarray] = None,
    transition_target_idx: Optional[np.ndarray] = None,
    transition_t: Optional[np.ndarray] = None,
    node_widths: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, float], np.ndarray]:
    """Place cells for ``plot_simplified_tree`` via ``LatentBarycentricPlacer``.

    First passes every cell through the placer with ``show_transitions=False``
    so each cell parks inside its anchor's stable cloud (using the renderers'
    pixel-space sizing). Then, if the caller supplies per-cell trajectory state
    via the ``transition_*`` arrays, those cells' positions are **overridden**
    by sampling the placer's pre-built edge curves
    (``LatentBarycentricPlacer.place_on_edge``) at the requested fractional
    position ``t``, routing transitioning cells onto the same Bezier curves the
    edge ribbons follow.

    Args:
        config: Active ``TrajectoryAnalysisConfig`` (cloud_size_*, y_scale,
            etc. are read directly).
        node_positions: Layout-engine output. For radial mode this MUST be
            in tree-index space (post ``DAGToTreeConverter.convert``).
        node_vectors: Node embeddings aligned with ``node_positions``.
        predictions: Per-cell anchor indices in the same index space.
        cell_vectors: Per-cell embeddings; for ``plot_simplified_tree`` the
            caller passes ``node_vectors[predictions]``.
        visual_tree_edges: Edges used by both the renderers and the placer.
        active_indices: Active node indices used for pixel-geometry bounds.
        node_counts: Per-node stable-cell counts (drives cloud sizing).
        is_polar: True for radial layouts.
        output_format: ``'pdf'`` or ``'html'``; selects the PDF / HTML
            backend cloud-radius scale.
        transition_state: Optional bool array, shape ``(n_cells,)``. When
            ``True`` for a cell, its stable position is overridden with an
            on-edge placement using the matching ``transition_source_idx``,
            ``transition_target_idx``, ``transition_t`` entries. ``None``
            disables transition placement.
        transition_source_idx: Optional int64 array, shape ``(n_cells,)``.
            Tree-space source node index per cell. Caller is responsible for
            resolving any DAG-to-tree split-node ambiguity. Ignored where
            ``transition_state`` is ``False``.
        transition_target_idx: Optional int64 array, shape ``(n_cells,)``.
            Tree-space target node index per cell.
        transition_t: Optional float array, shape ``(n_cells,)``. Position
            parameter in ``[0, 1]`` along the edge from source to target.
        node_widths: Optional dict of per-node branch widths in **pixels**.
            When supplied, transitioning-cell jitter on both radial and
            horizontal layouts scales to ~40% of the local branch width
            (with a 4 px floor) so cells visibly fan out within the
            branch's footprint instead of stacking on its centerline.
            Pass ``None`` to use ``transition_stream_width``-based jitter.

    Returns:
        Tuple ``(cell_positions, refreshed_node_cell_counts, cloud_extents,
        is_transitioning)``:
            * ``cell_positions``: ``(n_cells, 2)`` in the layout's data space.
            * ``refreshed_node_cell_counts``: counts after the placer's Pass 1
              recomputation; consumed downstream as the count source so cloud
              labels and scaling stay in sync.
            * ``cloud_extents``: per-node data-space y half-extent (used by
              contour bounding and ribbon-blend logic).
            * ``is_transitioning``: bool array, ``True`` for cells whose
              positions were overridden onto an edge. Cells with
              ``transition_state[i] = True`` but whose
              ``(source_idx, target_idx)`` is not in the placer's
              ``edge_paths`` (e.g., chain-collapse removed the visible edge)
              are silently demoted to ``False`` and keep their stable
              position.
    """
    from .trajectory_support import LatentBarycentricPlacer
    pixel_geometry = _build_placer_pixel_geometry(
        config, node_positions, active_indices, is_polar, output_format,
    )

    placer = LatentBarycentricPlacer(
        node_positions=node_positions,
        node_embeddings=node_vectors,
        tree_edges=visual_tree_edges,
        temperature=config.temperature,
        transition_stream_width=config.transition_stream_width,
        top_k_neighbors=config.top_k_neighbors,
        stable_source_zone=config.stable_source_zone,
        stable_target_zone=config.stable_target_zone,
        node_cell_counts=node_counts,
        scale_cloud_by_count=config.scale_cloud_by_count,
        cloud_scale_exponent=config.cloud_scale_exponent,
        cloud_size_multiplier=config.cloud_size_multiplier,
        cloud_size_min=config.cloud_size_min,
        cloud_size_max=config.cloud_size_max,
        y_scale=config.y_scale,
        is_polar=is_polar,
        is_pdf_output=(output_format == 'pdf'),
        inferred_path_weights={},
        lateral_directions={},
        pixel_geometry=pixel_geometry,
        node_widths=node_widths,
    )

    n_cells = int(predictions.shape[0])
    external_atypical_mask = np.zeros(n_cells, dtype=bool)
    external_adherence_scores = np.ones(n_cells, dtype=float)

    cell_positions, _is_transitioning, _parent_nodes, _child_nodes, \
        _transition_scores, _raw_transition_scores, _anchor_nodes, \
        _adherence_scores, _is_atypical, \
        _absolute_transition_scores = placer.place_cells(
            cell_vectors,
            external_atypical_mask=external_atypical_mask,
            external_adherence_scores=external_adherence_scores,
            prior_anchors=predictions,
            show_transitions=False,
        )

    refreshed_counts = dict(placer.node_cell_counts)
    cloud_extents = placer.compute_cloud_extents()

    is_transitioning = np.zeros(n_cells, dtype=bool)
    if (
        transition_state is not None
        and transition_source_idx is not None
        and transition_target_idx is not None
        and transition_t is not None
        and bool(transition_state.any())
    ):
        # Global seed: place_on_edge uses np.random.randn for normal-direction jitter.
        np.random.seed(42)
        demoted = 0
        for ci in range(n_cells):
            if not bool(transition_state[ci]):
                continue
            s = int(transition_source_idx[ci])
            t_node = int(transition_target_idx[ci])
            if (s, t_node) not in placer.edge_paths:
                demoted += 1
                continue
            is_transitioning[ci] = True
            cell_positions[ci] = placer.place_on_edge(
                s, t_node, float(transition_t[ci]),
            )
        if demoted:
            import logging
            logging.getLogger(__name__).warning(
                "plot_simplified_tree: %d transitioning cell(s) drawn in their "
                "type's cloud as if stable — their (source, target) edge was "
                "removed by chain-collapse and has no drawn replacement.",
                demoted,
            )

    return cell_positions, refreshed_counts, cloud_extents, is_transitioning


class RadialTreeLayoutEngine:
    """
    Compute polar coordinates using active-subtree sector partitioning.

    Angular sectors are allocated from active-node subtree weights. Canonical
    ordering and local crossing reduction arrange siblings within those sectors.
    """
    
    def __init__(self, adjacency_matrix: np.ndarray, node_names: List[str], 
                 config: Optional['TrajectoryAnalysisConfig'] = None):
        from .trajectory_render import RADIAL_CENTER_RADIUS
        self.adjacency_matrix = adjacency_matrix
        self.node_names = node_names
        self.config = config
        self.n_nodes = len(node_names)
        
        # Layout parameters
        self.max_radius = config.radial_max_radius 
        self.center_radius = RADIAL_CENTER_RADIUS
        self.n_segments = 100
        
        # Branch styling
        self.start_width = config.radial_start_width
        self.min_width = config.radial_min_width 
        
        # Internal state
        self.tree = None
        self.tree_edges = []
        self.node_depths = {}
        self.max_depth = 0
        self.node_colors = {}
        self.node_widths = {}
        self.node_clade_bg = {}
        self.clade_boundaries = {}
        self.node_sectors = {}       # angular (start, width) for EVERY node
        self._lineage_colored = set()  # nodes carrying a real lineage colour
        self.spanning_tree = None
        self.subtree_weights = {} 
        self.node_embeddings = None # Added for spectral sorting
        self.external_color_mapper = None  # Color mapper for adaptive lineage coloring
        self.canonical_order: Optional[CanonicalOrderComputer] = None
        self.original_indices: Optional[List[int]] = None
        self._resolved_sibling_order: Dict[int, List[int]] = {}
        self._resolved_root_order: Optional[List[int]] = None
        self._roots: List[int] = []
        self._total_weight: float = 1
        self._total_sweep: float = 0.0
        self._layout_weights: Dict[int, int] = {}
    
    def set_embeddings(self, embeddings: np.ndarray):
        """Store embeddings for spectral sorting of branches."""
        self.node_embeddings = embeddings
    
    def set_color_mapper(self, color_mapper) -> None:
        """
        Set the color mapper for adaptive lineage coloring.
        
        The mapper's get_node_color() method provides the base color for each node,
        which is then used to generate depth-based gradients.
        
        The mapper should be a TreeColorMapper instance that handles
        DAG-to-Tree index mapping (tree indices → original DAG indices → colors).
        
        Args:
            color_mapper: A color mapper with get_node_color(node_idx) -> hex color.
                         Typically a TreeColorMapper wrapping AdaptiveLineageColorMapper.
        """
        self.external_color_mapper = color_mapper
    
    def compute_layout(
        self,
        active_indices: List[int],
        augmented_adjacency: Optional[np.ndarray] = None,
        locked_sibling_order: Optional['SiblingOrder'] = None,
        reference_subtree_weights: Optional[Dict[int, int]] = None,
    ) -> Dict[int, Tuple[float, float]]:
        """Compute layout allocating sectors based on ACTIVE NODE COUNT.

        Uses a 2-stage crossing-reduction pipeline on sector-midpoint
        positions (no centering), then applies bottom-up parent centering
        as a final cosmetic pass:

        1. Ontology-based canonical ordering (deterministic, data-independent)
        2. Local swap polish (greedy pairwise swaps: crossings → spread →
           node-edge proximity)
        3. Bottom-up centering (post-processing, not part of swap evaluation)

        Args:
            active_indices: Nodes to include in layout.
            augmented_adjacency: Optional adjacency with inferred paths.
            locked_sibling_order: Pre-computed ordering from union tree
                (comparison mode). Overrides the crossing-reduction pipeline.
            reference_subtree_weights: Subtree weights from union tree for
                sector allocation in comparison mode.
        """
        import matplotlib.colors as mcolors

        if not active_indices:
            return {}
        if len(active_indices) == 1:
            return {active_indices[0]: (0.0, 1.0)}

        adjacency = augmented_adjacency if augmented_adjacency is not None else self.adjacency_matrix

        # 1. Build Full Graph from Adjacency (Parent -> Child flow)
        G_full = nx.DiGraph()
        rows, cols = np.where(adjacency > 0)
        for r, c in zip(rows, cols):
            G_full.add_edge(int(c), int(r))

        for i in range(self.n_nodes):
            if i not in G_full: G_full.add_node(i)

        # 2. Project Active Subgraph
        G = nx.DiGraph()
        G.add_nodes_from(active_indices)

        sorted_active = sorted(active_indices)

        for u in sorted_active:
            for v in sorted_active:
                if u == v: continue
                try:
                    if nx.has_path(G_full, u, v):
                        G.add_edge(u, v)
                except nx.NetworkXError:
                    pass

        try:
            G = nx.transitive_reduction(G)
        except Exception:
            pass

        self.tree = G
        self.tree_edges = list(G.edges())

        # 3. Build Spanning Tree
        self.spanning_tree = nx.DiGraph()
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]

        self.spanning_tree.add_nodes_from(G.nodes())

        queue = roots[:]
        visited = set(roots)
        while queue:
            u = queue.pop(0)
            children = list(G.successors(u))
            for v in children:
                if v not in visited:
                    visited.add(v)
                    self.spanning_tree.add_edge(u, v)
                    queue.append(v)

        # 4. Compute Subtree Weights
        self.subtree_weights = {}
        nodes_postorder = list(nx.dfs_postorder_nodes(self.spanning_tree))

        for n in nodes_postorder:
            children = list(self.spanning_tree.successors(n))
            if not children:
                self.subtree_weights[n] = 1
            else:
                self.subtree_weights[n] = 1 + sum(self.subtree_weights[c] for c in children)

        # Use reference weights for sector allocation in comparison mode
        layout_weights = reference_subtree_weights if reference_subtree_weights is not None else self.subtree_weights

        total_weight = sum(layout_weights.get(r, self.subtree_weights.get(r, 1)) for r in roots)
        if total_weight == 0: total_weight = 1

        # 5. Adaptive Sweep Angle
        if self.config and self.config.radial_sweep_angle_degrees is not None:
            total_sweep = np.radians(self.config.radial_sweep_angle_degrees)
        else:
            min_angle = 40
            max_angle = 240
            if total_weight <= 10:
                total_sweep = np.radians(min_angle)
            else:
                log_weight = np.log10(total_weight)
                log_min = np.log10(10)
                log_max = np.log10(200)
                scale = (log_weight - log_min) / (log_max - log_min)
                scale = np.clip(scale, 0.0, 1.0)
                angle_degrees = min_angle + (max_angle - min_angle) * scale
                total_sweep = np.radians(angle_degrees)

        self._roots = list(roots)
        self._total_weight = total_weight
        self._total_sweep = total_sweep
        self._layout_weights = layout_weights

        # 6. Crossing-reduction pipeline (2-stage: canonical order → swap polish)
        if locked_sibling_order is not None:
            self._apply_locked_order(locked_sibling_order)
        else:
            self._apply_structural_canonical_order()
            radii = self._precompute_radii(list(roots))
            positions = self._compute_initial_layout()
            c1 = self._count_edge_crossings(positions, radii)
            c2 = self._local_swap_polish(radii)

        # Final layout — apply centering after crossing reduction is locked in
        positions = self._center_parents_on_children(self._compute_initial_layout())

        # 7. Map Depth to Radius
        max_depth = 0
        for _, d in positions.values():
            if d > max_depth: max_depth = d
        self.max_depth = max_depth

        if hasattr(self, '_precomputed_node_heights') and self._precomputed_node_heights:
            node_heights = self._precomputed_node_heights
        else:
            node_heights = {}
            def calculate_height(node):
                if node in node_heights: return node_heights[node]
                children = list(self.spanning_tree.successors(node))
                if not children:
                    node_heights[node] = 0
                    return 0
                max_child_height = max(calculate_height(child) for child in children)
                node_heights[node] = max_child_height + 1
                return node_heights[node]
            for root in roots: calculate_height(root)

        final_positions = {}

        use_elastic = (
            hasattr(self, 'elastic_depths') and
            self.elastic_depths is not None and
            len(self.elastic_depths) > 0
        )

        if use_elastic:
            from .trajectory_ontology import apply_elastic_to_radial

            angular_positions = {node: theta for node, (theta, _) in positions.items()}

            final_positions = apply_elastic_to_radial(
                elastic_depths=self.elastic_depths,
                angular_positions=angular_positions,
                center_radius=self.center_radius,
                max_radius=self.max_radius
            )

            for node, (theta, depth) in positions.items():
                self.node_depths[node] = depth
        else:
            for node, (theta, depth) in positions.items():
                height = node_heights.get(node, 0)
                branch_depth = depth + height

                if depth == 0:
                    r = 0.0
                elif branch_depth > 0:
                    local_progress = depth / branch_depth
                    r = self.center_radius + (self.max_radius - self.center_radius) * local_progress
                else:
                    r = self.center_radius

                final_positions[node] = (theta, r)
                self.node_depths[node] = depth

        self.node_heights = node_heights
        self._compute_radial_colors(G, roots, mcolors)
        self._compute_dynamic_widths(roots)

        return final_positions

    # -----------------------------------------------------------------
    # Crossing-reduction pipeline helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4):
        """Test whether segments (p1-p2) and (p3-p4) properly cross."""
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
        d1 = ccw(p3, p4, p1)
        d2 = ccw(p3, p4, p2)
        d3 = ccw(p1, p2, p3)
        d4 = ccw(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def _apply_structural_canonical_order(self) -> None:
        """Set sibling ordering using ontology subtree structure."""
        self._resolved_sibling_order = {}
        for node in self.spanning_tree.nodes():
            children = list(self.spanning_tree.successors(node))
            if not children:
                continue
            if self.canonical_order is not None:
                def _struct_rank(n, _co=self.canonical_order, _oi=self.original_indices):
                    orig = _oi[n] if _oi else n
                    return _co.get_structural_rank(orig)
                self._resolved_sibling_order[node] = sorted(children, key=_struct_rank)
            else:
                self._resolved_sibling_order[node] = sorted(
                    children, key=lambda x: self.subtree_weights.get(x, 1), reverse=True
                )

    def _compute_initial_layout(self) -> Dict[int, Tuple[float, float]]:
        """Allocate angular sectors via top-down weight-proportional subdivision.

        Each node is placed at the midpoint of its sector.  This produces
        stable, deterministic positions that depend only on sibling order
        and subtree weights — no cascading repositioning.
        """
        positions: Dict[int, Tuple[float, float]] = {}
        self.clade_boundaries = {}
        self.node_sectors = {}
        current_angle = 0.0

        roots = self._get_ordered_roots()

        for root in roots:
            weight = self._layout_weights.get(root, self.subtree_weights.get(root, 1))
            sector_width = (weight / self._total_weight) * self._total_sweep

            mid_angle = current_angle + (sector_width / 2.0)
            positions[root] = (mid_angle, 0.0)

            self.clade_boundaries[root] = (current_angle, sector_width)
            self.node_sectors[root] = (current_angle, sector_width)
            self._layout_sector(root, current_angle, sector_width, 1, positions)
            current_angle += sector_width

        return positions

    def _center_parents_on_children(
        self, positions: Dict[int, Tuple[float, float]]
    ) -> Dict[int, Tuple[float, float]]:
        """Bottom-up centering: reposition each internal node at the midpoint
        of its children's angle range.

        This prevents long parent-to-leaf edges from sweeping across sibling
        subtrees — a common node-edge proximity artefact in sector-based
        radial layouts.  Applied as a post-processing step AFTER crossing
        reduction so that swap evaluation is not destabilised by cascading
        position changes.
        """
        centered = dict(positions)
        max_depth = max(d for _, d in centered.values()) if centered else 0
        for d in range(int(max_depth) - 1, -1, -1):
            for node in list(self.spanning_tree.nodes()):
                if centered.get(node, (0, -1))[1] != d:
                    continue
                children = list(self.spanning_tree.successors(node))
                if not children:
                    continue
                child_angles = [centered[c][0] for c in children]
                new_angle = (min(child_angles) + max(child_angles)) / 2.0
                centered[node] = (new_angle, centered[node][1])
        return centered

    def _get_ordered_roots(self) -> List[int]:
        """Return roots in resolved order."""
        roots = self._roots
        if hasattr(self, '_resolved_root_order') and self._resolved_root_order is not None:
            return self._resolved_root_order
        if self.canonical_order is not None:
            def _struct_rank(n):
                orig = self.original_indices[n] if self.original_indices else n
                return self.canonical_order.get_structural_rank(orig)
            return sorted(roots, key=_struct_rank)
        return sorted(roots, key=lambda x: self.subtree_weights.get(x, 1), reverse=True)

    def _precompute_radii(self, roots) -> Dict[int, float]:
        """Compute node radii from tree structure (order-independent)."""
        node_heights: Dict[int, int] = {}
        def _calc_height(node):
            if node in node_heights:
                return node_heights[node]
            children = list(self.spanning_tree.successors(node))
            if not children:
                node_heights[node] = 0
                return 0
            h = max(_calc_height(c) for c in children) + 1
            node_heights[node] = h
            return h
        for root in roots:
            _calc_height(root)

        positions = self._compute_initial_layout()

        use_elastic = (
            hasattr(self, 'elastic_depths') and
            self.elastic_depths is not None and
            len(self.elastic_depths) > 0
        )

        radii: Dict[int, float] = {}
        if use_elastic:
            from .trajectory_ontology import apply_elastic_to_radial
            angular_positions = {n: theta for n, (theta, _) in positions.items()}
            elastic_pos = apply_elastic_to_radial(
                elastic_depths=self.elastic_depths,
                angular_positions=angular_positions,
                center_radius=self.center_radius,
                max_radius=self.max_radius,
            )
            for n, (_, r) in elastic_pos.items():
                radii[n] = r
        else:
            for n, (_, depth) in positions.items():
                height = node_heights.get(n, 0)
                branch_depth = depth + height
                if depth == 0:
                    radii[n] = 0.0
                elif branch_depth > 0:
                    local_progress = depth / branch_depth
                    radii[n] = self.center_radius + (self.max_radius - self.center_radius) * local_progress
                else:
                    radii[n] = self.center_radius

        self._precomputed_node_heights = node_heights
        return radii

    def _count_edge_crossings(self, positions: Dict[int, Tuple[float, float]],
                              precomputed_radii: Dict[int, float]) -> int:
        """Count edge crossings using Cartesian segment intersection on post-mapping coordinates."""
        edges = list(self.spanning_tree.edges())
        if len(edges) < 2:
            return 0

        cart = {}
        for n, (theta, _) in positions.items():
            r = precomputed_radii.get(n, 0.0)
            cart[n] = (r * np.cos(theta), r * np.sin(theta))

        r_ranges = []
        for u, v in edges:
            ru = precomputed_radii.get(u, 0.0)
            rv = precomputed_radii.get(v, 0.0)
            r_ranges.append((min(ru, rv), max(ru, rv)))

        crossings = 0
        for i in range(len(edges)):
            u1, v1 = edges[i]
            for j in range(i + 1, len(edges)):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    continue
                if r_ranges[i][1] < r_ranges[j][0] or r_ranges[j][1] < r_ranges[i][0]:
                    continue
                if self._segments_intersect(cart[u1], cart[v1], cart[u2], cart[v2]):
                    crossings += 1

        return crossings

    # -----------------------------------------------------------------
    # Vectorised metric evaluator for swap polish
    # -----------------------------------------------------------------

    class _SwapPolishState:
        """Pre-computed arrays for vectorised crossing / proximity / sweep evaluation.

        Built once at the start of ``_local_swap_polish`` and reused for
        every trial swap.  The edge list, shared-endpoint mask, and
        radius-overlap mask are *static* (they depend on tree topology and
        radii, not on angular positions) and are computed in ``__init__``.
        Per-trial, only a lightweight dict→array conversion and pure numpy
        arithmetic are needed.
        """

        def __init__(self, edges, all_nodes, radii_dict, spanning_tree):
            self.edges = edges
            self.E = len(edges)
            self.all_nodes = all_nodes
            self.N = len(all_nodes)
            self._node_to_idx = {n: i for i, n in enumerate(all_nodes)}
            self._tree = spanning_tree

            self._edge_u = np.array([self._node_to_idx[u] for u, v in edges])
            self._edge_v = np.array([self._node_to_idx[v] for u, v in edges])
            self._radii = np.array([radii_dict.get(n, 0.0) for n in all_nodes])

            # --- static crossing-pair masks ---
            if self.E >= 2:
                ii, jj = np.triu_indices(self.E, k=1)
                en = np.empty((self.E, 2), dtype=np.intp)
                for ei, (u, v) in enumerate(edges):
                    en[ei, 0] = u; en[ei, 1] = v
                shared = ((en[ii, 0] == en[jj, 0]) | (en[ii, 0] == en[jj, 1]) |
                          (en[ii, 1] == en[jj, 0]) | (en[ii, 1] == en[jj, 1]))

                r_lo = np.minimum(self._radii[self._edge_u], self._radii[self._edge_v])
                r_hi = np.maximum(self._radii[self._edge_u], self._radii[self._edge_v])
                no_overlap = (r_hi[ii] < r_lo[jj]) | (r_hi[jj] < r_lo[ii])

                check = ~(shared | no_overlap)
                self._ci = ii[check]
                self._cj = jj[check]
            else:
                self._ci = np.array([], dtype=int)
                self._cj = np.array([], dtype=int)

            # --- static adjacency mask for proximity (N × E) ---
            adj = np.zeros((self.N, self.E), dtype=bool)
            for ei, (u, v) in enumerate(edges):
                adj[self._node_to_idx[u], ei] = True
                adj[self._node_to_idx[v], ei] = True
            self._adj = adj

            # --- parent lookup for ancestor-walking ---
            self._parent = {}
            for u, v in spanning_tree.edges():
                self._parent[v] = u

        # -- helpers --

        def _thetas(self, positions):
            return np.array([positions[n][0] for n in self.all_nodes])

        def _cart(self, thetas):
            return np.column_stack([self._radii * np.cos(thetas),
                                    self._radii * np.sin(thetas)])

        def _ccw_mask(self, p1, p2):
            """Vectorised CCW crossing test on pre-filtered pairs."""
            ci, cj = self._ci, self._cj
            dx_j = p2[cj, 0] - p1[cj, 0]; dy_j = p2[cj, 1] - p1[cj, 1]
            dx_i = p2[ci, 0] - p1[ci, 0]; dy_i = p2[ci, 1] - p1[ci, 1]
            d1 = (p1[ci, 1] - p1[cj, 1]) * dx_j - dy_j * (p1[ci, 0] - p1[cj, 0])
            d2 = (p2[ci, 1] - p1[cj, 1]) * dx_j - dy_j * (p2[ci, 0] - p1[cj, 0])
            d3 = (p1[cj, 1] - p1[ci, 1]) * dx_i - dy_i * (p1[cj, 0] - p1[ci, 0])
            d4 = (p2[cj, 1] - p1[ci, 1]) * dx_i - dy_i * (p2[cj, 0] - p1[ci, 0])
            return (((d1 > 0) & (d2 < 0)) | ((d1 < 0) & (d2 > 0))) & \
                   (((d3 > 0) & (d4 < 0)) | ((d3 < 0) & (d4 > 0)))

        # -- metrics --

        def count_crossings(self, positions):
            if self.E < 2 or len(self._ci) == 0:
                return 0
            thetas = self._thetas(positions)
            cart = self._cart(thetas)
            p1 = cart[self._edge_u]; p2 = cart[self._edge_v]
            return int(np.sum(self._ccw_mask(p1, p2)))

        def get_crossing_edge_indices(self, positions):
            """Return set of edge indices that participate in at least one crossing."""
            if self.E < 2 or len(self._ci) == 0:
                return set()
            thetas = self._thetas(positions)
            cart = self._cart(thetas)
            p1 = cart[self._edge_u]; p2 = cart[self._edge_v]
            cross = self._ccw_mask(p1, p2)
            if not np.any(cross):
                return set()
            result = set()
            result.update(self._ci[cross].tolist())
            result.update(self._cj[cross].tolist())
            return result

        def _ancestors_of(self, seed_nodes):
            """Walk up from *seed_nodes* to roots, return all ancestors."""
            ancestors = set()
            for node in seed_nodes:
                cur = node
                while cur is not None:
                    if cur in ancestors:
                        break
                    ancestors.add(cur)
                    cur = self._parent.get(cur)
            return ancestors

        def blame_crossings(self, crossing_edge_indices, internal_nodes):
            """Phase 1 blame set: ancestors of crossing-edge endpoints."""
            if not crossing_edge_indices:
                return []
            seeds = set()
            for ei in crossing_edge_indices:
                seeds.add(self.edges[ei][0])
                seeds.add(self.edges[ei][1])
            anc = self._ancestors_of(seeds)
            return [n for n in internal_nodes if n in anc]

        def blame_sweep(self, positions, internal_nodes):
            """Phase 2 blame set: ancestors of above-median-sweep edges."""
            thetas = self._thetas(positions)
            s = np.abs(thetas[self._edge_u] - thetas[self._edge_v])
            s = np.where(s > np.pi, 2 * np.pi - s, s)
            s_sq = s * s
            if len(s_sq) == 0:
                return []
            thresh = np.median(s_sq)
            bad = np.where(s_sq >= thresh)[0]
            seeds = set()
            for ei in bad.tolist():
                seeds.add(self.edges[ei][0])
                seeds.add(self.edges[ei][1])
            anc = self._ancestors_of(seeds)
            return [n for n in internal_nodes if n in anc]

        def blame_proximity(self, positions, internal_nodes):
            """Phase 3 blame set: ancestors of worst-proximity nodes."""
            thetas = self._thetas(positions)
            cart = self._cart(thetas)
            a = cart[self._edge_u]; b = cart[self._edge_v]
            ab = b - a
            ab_sq = np.einsum('ij,ij->i', ab, ab)
            p_exp = cart[:, np.newaxis, :]
            a_exp = a[np.newaxis, :, :]
            ab_exp = ab[np.newaxis, :, :]
            ab_sq_exp = ab_sq[np.newaxis, :]
            ap = p_exp - a_exp
            dot_ap_ab = np.einsum('ijk,ijk->ij', ap,
                                  np.broadcast_to(ab_exp, ap.shape))
            safe_sq = np.maximum(ab_sq_exp, 1e-12)
            t = np.clip(dot_ap_ab / safe_sq, 0.0, 1.0)
            closest = a_exp + t[:, :, np.newaxis] * ab_exp
            diff = p_exp - closest
            d_sq = np.einsum('ijk,ijk->ij', diff, diff)
            d_sq = np.where(self._adj | (ab_sq_exp < 1e-12), np.inf, d_sq)
            best = np.min(d_sq, axis=1)                    # (N,)
            # Worst-proximity nodes: smallest best_d_sq (closest to an edge)
            thresh = np.median(best[best < np.inf]) if np.any(best < np.inf) else np.inf
            bad_idx = np.where(best <= thresh)[0]
            seeds = {self.all_nodes[i] for i in bad_idx.tolist()}
            anc = self._ancestors_of(seeds)
            return [n for n in internal_nodes if n in anc]

        # -- metrics --

        def max_edge_sweep(self, positions):
            thetas = self._thetas(positions)
            s = np.abs(thetas[self._edge_u] - thetas[self._edge_v])
            s = np.where(s > np.pi, 2 * np.pi - s, s)
            return float(np.sum(s * s))

        def min_node_edge_proximity(self, positions):
            thetas = self._thetas(positions)
            cart = self._cart(thetas)
            a = cart[self._edge_u]                         # (E, 2)
            b = cart[self._edge_v]                         # (E, 2)
            ab = b - a                                     # (E, 2)
            ab_sq = np.einsum('ij,ij->i', ab, ab)          # (E,)
            p_exp = cart[:, np.newaxis, :]                  # (N, 1, 2)
            a_exp = a[np.newaxis, :, :]                    # (1, E, 2)
            ab_exp = ab[np.newaxis, :, :]                  # (1, E, 2)
            ab_sq_exp = ab_sq[np.newaxis, :]               # (1, E)
            ap = p_exp - a_exp                             # (N, E, 2)
            dot_ap_ab = np.einsum('ijk,ijk->ij', ap,
                                  np.broadcast_to(ab_exp, ap.shape))
            safe_sq = np.maximum(ab_sq_exp, 1e-12)
            t = np.clip(dot_ap_ab / safe_sq, 0.0, 1.0)    # (N, E)
            closest = a_exp + t[:, :, np.newaxis] * ab_exp # (N, E, 2)
            diff = p_exp - closest                         # (N, E, 2)
            d_sq = np.einsum('ijk,ijk->ij', diff, diff)    # (N, E)
            d_sq = np.where(self._adj | (ab_sq_exp < 1e-12), np.inf, d_sq)
            best = np.min(d_sq, axis=1)                    # (N,)
            valid = best > 1e-12
            return float(-np.sum(1.0 / best[valid]))

    # -----------------------------------------------------------------

    def _local_swap_polish(self, precomputed_radii: Dict[int, float],
                           max_swaps: int = 200) -> int:
        """Pairwise sibling swaps on post-mapping Cartesian coordinates.

        Phase 1 — crossing minimisation (targeted):
            Only tries swaps at nodes whose subtrees contain endpoints of
            currently-crossing edges.  Sound because a swap at node P can
            only resolve crossings involving edges with endpoints in P's
            subtrees.
        Phase 2 — spread optimisation (capped):
            Minimise sum-of-squared angular sweeps without re-introducing
            crossings.  Capped at 5 convergence rounds.
        Phase 3 — node-edge proximity (capped):
            Push nodes away from non-adjacent edges without re-introducing
            crossings or worsening spread by >1%.  Capped at 10 rounds.
        """
        edges = list(self.spanning_tree.edges())
        all_nodes = list(self.spanning_tree.nodes())

        internal_nodes = [
            n for n in all_nodes
            if len(self._resolved_sibling_order.get(n) or []) >= 2
        ]

        st = self._SwapPolishState(edges, all_nodes, precomputed_radii,
                                   self.spanning_tree)

        positions = self._compute_initial_layout()
        best_crossings = st.count_crossings(positions)

        # Phase 1: crossing minimisation — targeted to blame-set nodes
        if best_crossings > 0:
            swaps_made = 0
            improved = True
            while improved and swaps_made < max_swaps:
                improved = False
                crossing_ei = st.get_crossing_edge_indices(positions)
                blame = st.blame_crossings(crossing_ei, internal_nodes)
                if not blame:
                    break
                for node in blame:
                    children = self._resolved_sibling_order[node]
                    for i in range(len(children) - 1):
                        for j in range(i + 1, len(children)):
                            trial = list(children)
                            trial[i], trial[j] = trial[j], trial[i]
                            self._resolved_sibling_order[node] = trial
                            trial_positions = self._compute_initial_layout()
                            trial_crossings = st.count_crossings(trial_positions)
                            if trial_crossings < best_crossings:
                                best_crossings = trial_crossings
                                children = trial
                                positions = trial_positions
                                swaps_made += 1
                                improved = True
                                if best_crossings == 0:
                                    break
                            else:
                                self._resolved_sibling_order[node] = children
                        if best_crossings == 0:
                            break
                    if best_crossings == 0:
                        break

        # Phase 2: spread optimisation (0 crossings required, max 5 rounds)
        if best_crossings == 0:
            positions = self._compute_initial_layout()
            best_sweep = st.max_edge_sweep(positions)
            for _round in range(5):
                round_improved = False
                blame = st.blame_sweep(positions, internal_nodes)
                if not blame:
                    break
                for node in blame:
                    children = self._resolved_sibling_order[node]
                    for i in range(len(children) - 1):
                        for j in range(i + 1, len(children)):
                            trial = list(children)
                            trial[i], trial[j] = trial[j], trial[i]
                            self._resolved_sibling_order[node] = trial
                            trial_positions = self._compute_initial_layout()
                            tc = st.count_crossings(trial_positions)
                            if tc > 0:
                                self._resolved_sibling_order[node] = children
                                continue
                            trial_sweep = st.max_edge_sweep(trial_positions)
                            if trial_sweep < best_sweep:
                                best_sweep = trial_sweep
                                children = trial
                                positions = trial_positions
                                round_improved = True
                            else:
                                self._resolved_sibling_order[node] = children
                if not round_improved:
                    break

        # Phase 3: node-edge proximity (0 crossings, ≤1% sweep regression, max 10 rounds)
        if best_crossings == 0:
            positions = self._compute_initial_layout()
            best_sweep = st.max_edge_sweep(positions)
            best_prox = st.min_node_edge_proximity(positions)
            for _round in range(10):
                round_improved = False
                blame = st.blame_proximity(positions, internal_nodes)
                if not blame:
                    break
                for node in blame:
                    children = self._resolved_sibling_order[node]
                    for i in range(len(children) - 1):
                        for j in range(i + 1, len(children)):
                            trial = list(children)
                            trial[i], trial[j] = trial[j], trial[i]
                            self._resolved_sibling_order[node] = trial
                            trial_positions = self._compute_initial_layout()
                            tc = st.count_crossings(trial_positions)
                            if tc > 0:
                                self._resolved_sibling_order[node] = children
                                continue
                            ts = st.max_edge_sweep(trial_positions)
                            if ts > best_sweep * 1.01:
                                self._resolved_sibling_order[node] = children
                                continue
                            tp = st.min_node_edge_proximity(trial_positions)
                            if tp > best_prox:
                                best_prox = tp
                                best_sweep = ts
                                children = trial
                                positions = trial_positions
                                round_improved = True
                            else:
                                self._resolved_sibling_order[node] = children
                if not round_improved:
                    break

        return best_crossings

    def _apply_locked_order(self, locked_order: 'SiblingOrder') -> None:
        """Apply a pre-computed SiblingOrder (DAG-index space) to this tree."""
        self._resolved_sibling_order = {}
        self._resolved_root_order = None

        for tree_parent in self.spanning_tree.nodes():
            tree_children = list(self.spanning_tree.successors(tree_parent))
            if not tree_children:
                continue

            dag_parent = self.original_indices[tree_parent] if self.original_indices else tree_parent

            if dag_parent in locked_order.parent_to_children:
                dag_child_order = locked_order.parent_to_children[dag_parent]
                ordered: List[int] = []
                remaining = set(tree_children)
                for dag_child in dag_child_order:
                    matches = [tc for tc in remaining
                               if (self.original_indices[tc] if self.original_indices else tc) == dag_child]
                    for m in sorted(matches):
                        ordered.append(m)
                        remaining.discard(m)
                ordered.extend(sorted(remaining))
                self._resolved_sibling_order[tree_parent] = ordered
            else:
                self._resolved_sibling_order[tree_parent] = sorted(
                    tree_children, key=lambda x: self.subtree_weights.get(x, 1), reverse=True
                )

        # Lock root order from the SiblingOrder's special root entry
        if -1 in locked_order.parent_to_children:
            dag_root_order = locked_order.parent_to_children[-1]
            ordered_roots: List[int] = []
            remaining_roots = set(self._roots)
            for dag_r in dag_root_order:
                matches = [r for r in remaining_roots
                           if (self.original_indices[r] if self.original_indices else r) == dag_r]
                for m in sorted(matches):
                    ordered_roots.append(m)
                    remaining_roots.discard(m)
            ordered_roots.extend(sorted(remaining_roots))
            self._resolved_root_order = ordered_roots

    def get_resolved_sibling_order(self) -> 'SiblingOrder':
        """Extract current sibling ordering in DAG-index space."""
        parent_to_children: Dict[int, List[int]] = {}
        for tree_parent, tree_children in self._resolved_sibling_order.items():
            dag_parent = self.original_indices[tree_parent] if self.original_indices else tree_parent
            dag_children: List[int] = []
            seen: Set[int] = set()
            for tc in tree_children:
                dag_c = self.original_indices[tc] if self.original_indices else tc
                if dag_c not in seen:
                    dag_children.append(dag_c)
                    seen.add(dag_c)
            parent_to_children[dag_parent] = dag_children

        # Store root ordering under special key -1
        ordered_roots = self._get_ordered_roots()
        dag_roots: List[int] = []
        seen_roots: Set[int] = set()
        for r in ordered_roots:
            dag_r = self.original_indices[r] if self.original_indices else r
            if dag_r not in seen_roots:
                dag_roots.append(dag_r)
                seen_roots.add(dag_r)
        parent_to_children[-1] = dag_roots

        return SiblingOrder(parent_to_children=parent_to_children)

    # -----------------------------------------------------------------
    # Layout helpers
    # -----------------------------------------------------------------

    def _sort_children_optimal(self, children: List[int]) -> List[int]:
        """Sort children by Embedding Similarity (Spectral Sorting).

        The current crossing-reduction pipeline ordinarily supplies
        ``_resolved_sibling_order`` instead.
        """
        if len(children) < 2:
            return children

        if self.node_embeddings is None:
             return sorted(children, key=lambda x: self.subtree_weights.get(x, 1), reverse=True)

        try:
            child_embs = self.node_embeddings[children]

            if len(children) >= 3:
                if not np.any(np.std(child_embs, axis=0) > 1e-12):
                    return sorted(children, key=lambda x: self.subtree_weights.get(x, 1), reverse=True)
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1)
                scores = pca.fit_transform(child_embs).flatten()

                sorted_pairs = sorted(zip(children, scores), key=lambda x: x[1])
                return [c for c, s in sorted_pairs]
            else:
                return sorted(children, key=lambda x: self.subtree_weights.get(x, 1), reverse=True)

        except Exception:
            return sorted(children, key=lambda x: self.subtree_weights.get(x, 1), reverse=True)

    def _layout_sector(self, node, start_angle, width, depth, positions):
        """Recursively divide sector using resolved sibling ordering."""
        children = list(self.spanning_tree.successors(node))
        if not children:
            return

        # Use resolved order if available
        if hasattr(self, '_resolved_sibling_order') and node in self._resolved_sibling_order:
            children = self._resolved_sibling_order[node]
        else:
            if self.node_embeddings is None and self.canonical_order is not None:
                def _canonical_rank(n):
                    orig = self.original_indices[n] if self.original_indices else n
                    return self.canonical_order.get_rank(orig)
                children = sorted(children, key=_canonical_rank)
            else:
                children = self._sort_children_optimal(children)

        total_child_weight = sum(
            self._layout_weights.get(c, self.subtree_weights.get(c, 1)) for c in children
        )
        current_child_angle = start_angle

        for child in children:
            child_weight = self._layout_weights.get(child, self.subtree_weights.get(child, 1))

            if total_child_weight > 0:
                child_width = (child_weight / total_child_weight) * width
            else:
                child_width = width / len(children)

            child_mid_angle = current_child_angle + (child_width / 2.0)
            positions[child] = (child_mid_angle, depth)

            self.node_sectors[child] = (current_child_angle, child_width)
            if depth == 1:
                self.clade_boundaries[child] = (current_child_angle, child_width)

            self._layout_sector(child, current_child_angle, child_width, depth + 1, positions)
            current_child_angle += child_width
    
    def get_node_color(self, node_idx): return self.node_colors.get(node_idx, RADIAL_BRANCH_GRAY)
    
    def is_safe_for_legend(self, margin_degrees: float = 15.0) -> bool:
        """
        Check if the 270-360° quadrant is safe for legend placement.
        
        Args:
            margin_degrees: Safety margin in degrees (default: 15°)
        
        Returns:
            bool: True if tree ends before 270°, False otherwise
        """
        if not self.clade_boundaries:
            return False
        
        # Find the maximum angle used by the tree
        max_angle = 0.0
        for theta_start, theta_width in self.clade_boundaries.values():
            theta_end = theta_start + theta_width
            max_angle = max(max_angle, theta_end)
        
        # Convert to degrees and check if it's before 270°
        max_angle_deg = np.degrees(max_angle)
        
        return max_angle_deg < 270.0

    def _compute_radial_colors(self, G, roots, mcolors):
        """Compute colors for radial layout nodes using adaptive lineage coloring."""
        self._lineage_colored = set()
        if self.external_color_mapper is None:
            # Fallback: assign gray colors to all nodes
            for root in roots:
                self.node_colors[root] = RADIAL_ROOT_GRAY
                self.node_clade_bg[root] = None
            for node in G.nodes():
                if node not in roots:
                    self.node_colors[node] = RADIAL_BRANCH_GRAY
                    self.node_clade_bg[node] = None
            return
        self._compute_radial_colors_adaptive(G, roots, mcolors)

    def _compute_radial_colors_adaptive(self, G, roots, mcolors):
        """Compute radial colors using external color mapper (adaptive lineage colors).
        
        Algorithm:
        1. For each node, get base color from external_color_mapper.get_node_color(node_idx)
        2. Group nodes by their base color (same lineage = same base color)
        3. For each color group, generate gradient variants (start → end)
        4. Apply depth-based interpolation within each group
        5. Root nodes always get a light neutral gray
        
        This preserves the depth-based fading effect while using lineage-appropriate colors.
        
        Args:
            G: NetworkX DiGraph representing the tree structure
            roots: List of root node indices
            mcolors: matplotlib.colors module for color manipulation
        """
        # Cache color variants to avoid recomputation
        color_variants_cache = {}
        
        def get_color_variants(base_hex):
            if base_hex not in color_variants_cache:
                color_variants_cache[base_hex] = _generate_color_variants(base_hex, mcolors)
            return color_variants_cache[base_hex]
        
        # Assign colors to root nodes (always gray)
        for root in roots:
            self.node_colors[root] = RADIAL_ROOT_GRAY
            self.node_clade_bg[root] = None
        
        # Assign colors to all other nodes
        for node in G.nodes():
            if node in roots:
                continue
            
            # Get base color from external mapper (already handles tree→DAG mapping)
            base_color = self.external_color_mapper.get_node_color(node)
            
            depth = self.node_depths.get(node, 0)
            
            if _is_shared_trunk_gray(base_color):
                # Unknown node (NEUTRAL_GRAY), use gray color
                self.node_colors[node] = RADIAL_BRANCH_GRAY
                # For depth-1 nodes, still generate a gray background for sector visibility
                if depth == 1:
                    gray_variants = get_color_variants(RADIAL_BRANCH_GRAY)
                    self.node_clade_bg[node] = gray_variants["bg"]
                else:
                    self.node_clade_bg[node] = None
                continue
            
            # Get color variants for gradient
            variants = get_color_variants(base_color)
            
            # Apply depth-based gradient (same logic as original)
            max_d = float(self.max_depth) if self.max_depth > 0 else 1.0
            ratio = min(depth / max_d, 1.0)
            
            # Interpolate between start (saturated) and end (light)
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "adaptive", [variants["start"], variants["end"]]
            )
            final_color = mcolors.to_hex(cmap(ratio))
            self.node_colors[node] = final_color
            self.node_clade_bg[node] = variants["bg"]
            self._lineage_colored.add(node)

    def get_clade_background_sectors(
        self, visible_nodes: Optional[Set[int]] = None,
    ) -> List[Tuple[float, float, str]]:
        """Angular wedges to shade behind each clade, as (start, width, colour).

        The wedges exist to make clades easy to pick out, so they follow the
        *coloured* clades rather than the root's depth-1 children. The lineage
        colour mapper paints any node whose descendants span several lineages
        in shared-trunk gray, and depth-1 nodes nearly always do — which left
        every wedge the same near-white gray once branch colouring changed.

        A colour clade is the topmost node of a single-lineage subtree: it
        carries a lineage colour while its parent does not. Such subtrees can
        never nest (a coloured node's descendants are all the same lineage, so
        they are all coloured), so the wedges never overlap.

        When no node carries a lineage colour, depth-1 wedges provide the
        fallback, for example when no external colour mapper was supplied.

        Args:
            visible_nodes: If given, only shade clades whose root is drawn.
        """
        sectors = self.node_sectors or self.clade_boundaries
        colored = self._lineage_colored
        parent_of: Dict[int, int] = {}
        if self.spanning_tree is not None:
            for parent, child in self.spanning_tree.edges():
                parent_of[child] = parent

        def _drawn(node: int) -> bool:
            return visible_nodes is None or node in visible_nodes

        out: List[Tuple[float, float, str]] = []
        for node in colored:
            if node not in sectors or not _drawn(node):
                continue
            if parent_of.get(node) in colored:
                continue  # already inside a larger colour clade
            bg_color = self.node_clade_bg.get(node)
            if not bg_color:
                continue
            theta_start, theta_width = sectors[node]
            out.append((theta_start, theta_width, bg_color))

        if not out:
            for node, (theta_start, theta_width) in self.clade_boundaries.items():
                bg_color = self.node_clade_bg.get(node)
                if bg_color and self.node_depths.get(node, 0) == 1 and _drawn(node):
                    out.append((theta_start, theta_width, bg_color))

        out.sort(key=lambda s: s[0])
        return out


    def _compute_dynamic_widths(self, roots: List[int]):
        node_heights = self.node_heights
        start_width = self.config.radial_start_width
        min_width = self.config.radial_min_width
        
        for node in self.spanning_tree.nodes():
            depth = self.node_depths.get(node, 0)
            height = node_heights.get(node, 0)
            path_length = depth + height
            
            if path_length == 0:
                ratio = 0.0
            else:
                ratio = depth / path_length
            
            width = start_width * (1 - ratio) + min_width * ratio
            self.node_widths[node] = max(min_width, width)

    def compute_edge_curvature_scales(self, positions) -> Dict[Tuple[int, int], float]:
        """Targeted curvature reduction for edges involved in Bezier crossings.

        Starts with full curvature for all edges, then iteratively reduces
        only the edges whose Bezier curves actually cross.  Most edges keep
        ``scale=1.0``.  Call after ``compute_layout()``; the result is cached
        in ``self._edge_curvature_scales`` and reused by both
        ``get_radial_edges()`` and ``LatentBarycentricPlacer``.
        """
        from .trajectory_support import calculate_radial_bezier_curve

        roots = set(n for n in self.tree.nodes() if self.tree.in_degree(n) == 0)
        non_root = [(u, v) for u, v in self.tree_edges
                     if u not in roots and u in positions and v in positions]

        scales: Dict[Tuple[int, int], float] = {e: 1.0 for e in non_root}

        def _make_xy(u, v, sc):
            pt, pr = positions[u]
            ct, cr = positions[v]
            ts, te = pt, ct
            d = te - ts
            if d > np.pi: te -= 2 * np.pi
            elif d < -np.pi: te += 2 * np.pi
            pts = calculate_radial_bezier_curve(ts, pr, te, cr,
                                                num_points=21,
                                                curvature_scale=sc)
            return [(r * np.cos(t), r * np.sin(t)) for t, r in pts[::2]]

        # Precompute radius ranges for bounding-box pruning
        r_ranges = {}
        for u, v in non_root:
            ru, rv = positions[u][1], positions[v][1]
            r_ranges[(u, v)] = (min(ru, rv), max(ru, rv))

        for _iteration in range(12):
            curves = {e: _make_xy(e[0], e[1], scales[e]) for e in non_root}

            crossing_count: Dict[Tuple[int, int], int] = {}
            for i in range(len(non_root)):
                e1 = non_root[i]
                for j in range(i + 1, len(non_root)):
                    e2 = non_root[j]
                    if e1[0] == e2[0] or e1[0] == e2[1] or \
                       e1[1] == e2[0] or e1[1] == e2[1]:
                        continue
                    if r_ranges[e1][1] < r_ranges[e2][0] or \
                       r_ranges[e2][1] < r_ranges[e1][0]:
                        continue
                    c1, c2 = curves[e1], curves[e2]
                    crossed = False
                    for si in range(len(c1) - 1):
                        for sj in range(len(c2) - 1):
                            if self._segments_intersect(
                                    c1[si], c1[si + 1], c2[sj], c2[sj + 1]):
                                crossing_count[e1] = crossing_count.get(e1, 0) + 1
                                crossing_count[e2] = crossing_count.get(e2, 0) + 1
                                crossed = True
                                break
                        if crossed:
                            break

            if not crossing_count:
                break

            for edge in crossing_count:
                sweep = abs(positions[edge[0]][0] - positions[edge[1]][0])
                reduction = 0.15 * min(1.0, sweep / np.radians(10.0))
                scales[edge] = max(0.15, scales[edge] - reduction)

        self._edge_curvature_scales = scales
        return scales

    def get_radial_edges(self, positions):
        """Generate edge data for radial layout with color gradients.

        Root edges start from center (radius 0) and use child's color.
        Gray parent nodes inherit child's color to avoid gray branches.
        """
        edges_data = []
        roots = [n for n in self.tree.nodes() if self.tree.in_degree(n) == 0]
        
        from .trajectory_render import _hex_to_rgb, _rgb_to_hex
        from .trajectory_support import calculate_radial_bezier_curve

        def get_color_array(c1, c2, n):
            c1_rgb = np.array(_hex_to_rgb(c1), dtype=float)
            c2_rgb = np.array(_hex_to_rgb(c2), dtype=float)

            if n <= 1:
                return [c1]

            colors = []
            for i in range(n):
                t = i / (n - 1)
                rgb = (1 - t) * c1_rgb + t * c2_rgb
                colors.append(_rgb_to_hex(tuple(rgb)))
            return colors

        # Compute or reuse cached curvature scales
        if not hasattr(self, '_edge_curvature_scales') or not self._edge_curvature_scales:
            self.compute_edge_curvature_scales(positions)
        curvature_scales = self._edge_curvature_scales

        for u, v in self.tree_edges:
            if u not in positions or v not in positions: continue

            parent_theta, parent_r = positions[u]
            child_theta, child_r = positions[v]

            if u in roots:
                p_r = 0.0
                theta_start = child_theta
                theta_end = child_theta
            else:
                p_r = parent_r
                theta_start = parent_theta
                theta_end = child_theta

            c_r = child_r
            diff = theta_end - theta_start
            if diff > np.pi: theta_end -= 2 * np.pi
            elif diff < -np.pi: theta_end += 2 * np.pi

            parent_color_raw = self.node_colors.get(u, RADIAL_ROOT_GRAY)
            c_color = self.node_colors.get(v, RADIAL_BRANCH_GRAY)
            is_gray_parent = _is_shared_trunk_gray(parent_color_raw)

            if u in roots or is_gray_parent:
                p_color = c_color
            else:
                p_color = parent_color_raw

            parent_width = self.node_widths.get(u, 1.0)
            child_width = self.node_widths.get(v, 1.0)

            if u in roots:
                curve_points = np.array([
                    [theta_start, p_r + i * (c_r - p_r) / self.n_segments]
                    for i in range(self.n_segments + 1)
                ])
            else:
                scale = curvature_scales.get((u, v), 1.0)
                curve_points = calculate_radial_bezier_curve(
                    theta_start, p_r, theta_end, c_r,
                    num_points=self.n_segments + 1,
                    curvature_scale=scale,
                )

            segments = []
            cs = get_color_array(p_color, c_color, self.n_segments)
            ws = np.linspace(parent_width, child_width, self.n_segments)

            for i in range(self.n_segments):
                segments.append({
                    'start': tuple(curve_points[i]),
                    'end': tuple(curve_points[i+1]),
                    'color': cs[i],
                    'width': ws[i]
                })
            edges_data.append({'parent': u, 'child': v, 'segments': segments})
            
        return edges_data

class AdaptiveOntologySubgraph:
    """
    Filter an ontology to the nodes relevant to the active cell types.
    """
    
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        node_names: List[str],
        co_graph: Optional[Dict[str, List[str]]] = None
    ):
        self.node_names = node_names
        self.n_nodes = len(node_names)
        self.co_graph = co_graph
        self.input_adjacency = adjacency_matrix
        
        # Build NetworkX graph (Robust check)
        self.nx_graph = self._build_nx_graph()
        
        # Check if adjacency is symmetric (undirected)
        is_symmetric = np.allclose(self.input_adjacency, self.input_adjacency.T) if self.input_adjacency.shape[0] > 0 else False
        
        if co_graph:
            # Using co_graph to enforce directed DAG
            # Rebuild adjacency from strictly directed graph to ensure layout engine works
            self.adjacency_matrix = nx.to_numpy_array(self.nx_graph, nodelist=range(self.n_nodes))
            # Note: nx_graph is Child->Parent. 
            # nx.to_numpy_array[i,j] = 1 if edge i->j exists.
            # So adj[child, parent] = 1.
        else:
            if is_symmetric:
                pass
            self.adjacency_matrix = self.input_adjacency
        
    def _build_nx_graph(self) -> nx.DiGraph:
        """
        Build NetworkX DiGraph.
        Prioritizes co_graph (child->parents) for strict directionality.
        """
        G = nx.DiGraph()
        
        # Initialize all nodes
        for i in range(self.n_nodes):
            G.add_node(i, name=self.node_names[i])
            
        if self.co_graph:
            # Use co_graph: child_name -> [parent_names]
            name_to_idx = {name: i for i, name in enumerate(self.node_names)}
            
            for child_name, parent_names in self.co_graph.items():
                if child_name not in name_to_idx: continue
                child_idx = name_to_idx[child_name]
                
                for parent_name in parent_names:
                    if parent_name in name_to_idx:
                        parent_idx = name_to_idx[parent_name]
                        # Edge: Child -> Parent
                        G.add_edge(child_idx, parent_idx)
        else:
            # Fallback to adjacency matrix
            # Assume rows=child, cols=parent (typical)
            # NOTE: Use input_adjacency since adjacency_matrix isn't set yet during __init__
            rows, cols = np.where(self.input_adjacency > 0)
            for child, parent in zip(rows, cols):
                G.add_edge(int(child), int(parent))
        
        return G
    
    def get_relevant_node_indices(
        self,
        predicted_indices: np.ndarray,
        include_ancestors: bool = True,
        include_siblings: bool = True,
        siblings_rule: str = 'leaves_only'
    ) -> List[int]:
        """
        Get indices of relevant nodes using Shortest Path to Root pruning.
        This avoids the 'explosion' caused by cycles or dense connectivity.
        """
        unique_predictions = set(np.unique(predicted_indices).tolist())
        unique_predictions = {idx for idx in unique_predictions if 0 <= idx < self.n_nodes}
        
        active_nodes = set(unique_predictions)
        leaf_nodes = set(unique_predictions)
        
        leaf_nodes = list(set(predicted_indices) & set(self.nx_graph.nodes()))
        
        # 0. Find Root(s)
        # In Child->Parent graph, Root is a node with Out-Degree 0 (no parents)
        # Or specifically CL:0000000
        roots = [n for n in self.nx_graph.nodes() if self.nx_graph.out_degree(n) == 0]
        if not roots:
             # fallback finding CL:0000000
             cell_root_indices = [i for i, name in enumerate(self.node_names) if 'CL:0000000' in name]
             if cell_root_indices:
                 roots = cell_root_indices
             else:
                 # Last resort: max in-degree
                 degrees = dict(self.nx_graph.in_degree())
                 roots = [max(degrees, key=degrees.get)]
        
        # Add Ancestors (Strict Shortest Paths)
        if include_ancestors:
            ancestors = set()
            for leaf in leaf_nodes:
                for root in roots:
                    if leaf == root: continue
                    try:
                        # Find all shortest paths (or just one shortest path for strictness)
                        # usage: nx.shortest_path(G, source, target)
                        path = nx.shortest_path(self.nx_graph, source=leaf, target=root)
                        ancestors.update(path)
                    except nx.NetworkXNoPath:
                        pass
                    except nx.NetworkXError:
                         pass
            
            if len(ancestors) == 0 and len(leaf_nodes) > 0:
                 for leaf in leaf_nodes:
                     ancestors.update(self.nx_graph.successors(leaf))
            
            active_nodes.update(ancestors)
            
        # 2. Add Siblings (Leaves Only) - RESTRICED
        if include_siblings and siblings_rule == 'leaves_only':
            siblings_count = 0
            # Only if we aren't already huge
            if len(active_nodes) < 200:
                for node_idx in leaf_nodes:
                    try:
                        parents = list(self.nx_graph.successors(node_idx))
                        for parent in parents:
                            # Skip if parent is Root or very high level
                            if parent in roots: continue
                            
                            siblings = list(self.nx_graph.predecessors(parent))
                            # STRICT LIMIT: Max 5 siblings per parent
                            if len(siblings) > 5:
                                siblings = siblings[:5]
                                
                            for sib in siblings:
                                if sib not in active_nodes:
                                    active_nodes.add(sib)
                                    siblings_count += 1
                    except: pass

        valid_active = sorted([idx for idx in active_nodes if 0 <= idx < self.n_nodes])
        
        return valid_active
    
    def get_subgraph_adjacency(self, active_indices: List[int]) -> np.ndarray:
        """Extract adjacency matrix for active subgraph."""
        active_indices = np.array(active_indices)
        return self.adjacency_matrix[np.ix_(active_indices, active_indices)]
    
    def get_active_node_names(self, active_indices: List[int]) -> List[str]:
        """Get names for active nodes."""
        return [self.node_names[i] for i in active_indices]


# =============================================================================
# Color Mapper (Hierarchical Shading)
# =============================================================================

def hex_to_hsl(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex to HSL."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    max_c, min_c = max(r, g, b), min(r, g, b)
    l = (max_c + min_c) / 2.0
    
    if max_c == min_c:
        h = s = 0.0
    else:
        d = max_c - min_c
        s = d / (2.0 - max_c - min_c) if l > 0.5 else d / (max_c + min_c)
        if max_c == r:
            h = (g - b) / d + (6.0 if g < b else 0.0)
        elif max_c == g:
            h = (b - r) / d + 2.0
        else:
            h = (r - g) / d + 4.0
        h /= 6.0
    return h, s, l


def hsl_to_hex(h: float, s: float, l: float) -> str:
    """Convert HSL to hex."""
    def hue_to_rgb(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p
    
    if s == 0:
        r = g = b = l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
    
    return '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))



# =============================================================================
# Adaptive Lineage Coloring - Core LCA Functions
# =============================================================================

def find_lca(graph: nx.DiGraph, nodes: List[int]) -> int:
    """
    Find the lowest common ancestor used for lineage grouping.
    
    The LCA is the deepest node in the ontology tree that is an ancestor of all
    input nodes. For a single node, this function intentionally returns its first
    parent, or the node itself when it has no parent. In a multi-parent DAG, that
    single-node result therefore follows graph successor order.

    Algorithm:
    1. Use BFS from each node to find all ancestors (following child -> parent edges)
    2. Find intersection of all ancestor sets
    3. Return the deepest common ancestor (furthest from root)
    
    Args:
        graph: NetworkX DiGraph with child -> parent edges (is_a relationships).
               In this graph, successors(node) returns the node's parents.
        nodes: List of node indices to find LCA for. Must be non-empty.
        
    Returns:
        Index of the LCA node.
        
    Raises:
        ValueError: If nodes list is empty or if no common ancestor exists.
    """
    from collections import deque
    
    if not nodes:
        raise ValueError("Cannot find LCA of empty node list")
    
    # Handle single node case: return its parent
    if len(nodes) == 1:
        node = nodes[0]
        parents = list(graph.successors(node))
        if parents:
            return parents[0]  # Return first parent
        else:
            # Node has no parent (is root), return itself
            return node
    
    def get_ancestors_with_depths(start_node: int) -> Dict[int, int]:
        """
        BFS to find all ancestors of a node with their depths from the start node.
        
        Depth 0 = the start node itself
        Depth 1 = direct parents
        Depth 2 = grandparents, etc.
        
        Returns:
            Dict mapping ancestor_idx -> depth from start_node
        """
        ancestors = {start_node: 0}  # Include self as ancestor at depth 0
        queue = deque([(start_node, 0)])
        
        while queue:
            current, depth = queue.popleft()
            
            # Get parents (successors in child->parent graph)
            for parent in graph.successors(current):
                if parent not in ancestors:
                    ancestors[parent] = depth + 1
                    queue.append((parent, depth + 1))
        
        return ancestors
    
    # Get ancestors for all input nodes
    ancestor_sets = []
    ancestor_depths_list = []
    
    for node in nodes:
        ancestors_with_depths = get_ancestors_with_depths(node)
        ancestor_sets.append(set(ancestors_with_depths.keys()))
        ancestor_depths_list.append(ancestors_with_depths)
    
    # Find intersection of all ancestor sets (common ancestors)
    common_ancestors = ancestor_sets[0]
    for ancestor_set in ancestor_sets[1:]:
        common_ancestors = common_ancestors.intersection(ancestor_set)
    
    if not common_ancestors:
        raise ValueError(
            f"No common ancestor found for nodes {nodes}. "
            "Graph may be disconnected or malformed."
        )
    
    # Find the deepest common ancestor
    # "Deepest" means furthest from root = highest depth in the ontology
    # We need to compute depth from root for each common ancestor
    
    # First, find root nodes (nodes with no parents = out_degree 0 in child->parent graph)
    roots = [n for n in graph.nodes() if graph.out_degree(n) == 0]
    
    if not roots:
        # Fallback: use node with minimum in-degree as pseudo-root
        roots = [min(graph.nodes(), key=lambda n: graph.in_degree(n))]
    
    # BFS from roots to compute depth from root for all nodes
    depth_from_root = {}
    queue = deque()
    
    for root in roots:
        queue.append((root, 0))
        depth_from_root[root] = 0
    
    while queue:
        current, depth = queue.popleft()
        
        # Get children (predecessors in child->parent graph)
        for child in graph.predecessors(current):
            # Take maximum depth if node reachable from multiple paths
            if child not in depth_from_root or depth + 1 > depth_from_root[child]:
                depth_from_root[child] = depth + 1
                queue.append((child, depth + 1))
    
    # Find the common ancestor with maximum depth from root (deepest = closest to leaves)
    lca = None
    max_depth = -1
    
    for ancestor in common_ancestors:
        ancestor_depth = depth_from_root.get(ancestor, 0)
        if ancestor_depth > max_depth:
            max_depth = ancestor_depth
            lca = ancestor
    
    if lca is None:
        # Fallback: return first common ancestor
        lca = next(iter(common_ancestors))

    return lca


def get_subtree_nodes(graph: nx.DiGraph, root: int) -> Set[int]:
    """
    Get all nodes in the subtree rooted at the given node.
    
    Performs BFS traversal from the root node, following child edges (predecessors
    in the child -> parent graph) to collect all descendant nodes.
    
    Args:
        graph: NetworkX DiGraph with child -> parent edges (is_a relationships).
               In this graph, predecessors(node) returns the node's children.
        root: Root node index of the subtree.
        
    Returns:
        Set of all node indices in the subtree, including the root itself.
        
    """
    from collections import deque
    
    subtree = {root}  # Include root in the subtree
    queue = deque([root])
    
    while queue:
        current = queue.popleft()
        
        # Get children (predecessors in child->parent graph)
        # Since edges are child -> parent, predecessors gives us children
        for child in graph.predecessors(current):
            if child not in subtree:
                subtree.add(child)
                queue.append(child)
    
    return subtree


_MAX_GROUP_SIZE = 25
_MIN_GROUP_SIZE = 2


def _get_active_descendants_map(
    dag: nx.DiGraph, active_indices: Set[int]
) -> Dict[int, Set[int]]:
    """For every node in the DAG, compute its set of active descendants (including itself)."""
    reversed_dag = dag.reverse()
    cache: Dict[int, Set[int]] = {}

    def _get(node: int) -> Set[int]:
        if node in cache:
            return cache[node]
        desc: Set[int] = set()
        if node in active_indices:
            desc.add(node)
        for child in reversed_dag.neighbors(node):
            desc |= _get(child)
        cache[node] = desc
        return desc

    for node in nx.topological_sort(dag):
        _get(node)
    return cache


def _find_group_roots(
    dag: nx.DiGraph,
    active_indices: Set[int],
    min_size: int = _MIN_GROUP_SIZE,
    max_size: int = _MAX_GROUP_SIZE,
    attach_leftovers: bool = True,
) -> List[Tuple[int, List[int]]]:
    """Find natural lineage groups among active types using greedy set-cover.

    Phase 1: Find group roots — nodes whose active descendants form a
    reasonably sized cluster (between min_size and max_size).

    Phase 2: Assign remaining active types to the closest group via
    shortest undirected path in the DAG (max distance 5).  Only runs when
    *attach_leftovers* is True.

    Phase 2 ignores edge direction, and every type reaching it sits under none
    of the groups, so it always files a type somewhere it does not belong —
    sometimes under one of its own descendants.  Pass False to keep those types
    as their own groups instead.

    Returns list of (root_idx, [member_indices]) sorted by member count desc.
    """
    desc_map = _get_active_descendants_map(dag, active_indices)

    candidates = []
    for node, desc in desc_map.items():
        if min_size <= len(desc) <= max_size:
            candidates.append((node, desc))
    candidates.sort(key=lambda x: len(x[1]), reverse=True)

    assigned: Set[int] = set()
    groups: List[Tuple[int, List[int]]] = []
    group_roots: List[int] = []

    for root_idx, members in candidates:
        unassigned = members - assigned
        if len(unassigned) >= min_size:
            groups.append((root_idx, sorted(unassigned)))
            group_roots.append(root_idx)
            assigned |= unassigned

    ungrouped = active_indices - assigned
    if attach_leftovers and ungrouped and group_roots:
        undirected = dag.to_undirected()
        for node in ungrouped:
            best_dist = float('inf')
            best_group = -1
            for gi, root in enumerate(group_roots):
                try:
                    d = nx.shortest_path_length(undirected, node, root)
                except nx.NetworkXNoPath:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_group = gi
            if best_group >= 0 and best_dist <= 5:
                ri, members = groups[best_group]
                groups[best_group] = (ri, sorted(set(members) | {node}))
                assigned.add(node)

    still_ungrouped = active_indices - assigned
    if still_ungrouped:
        groups.append((-1, sorted(still_ungrouped)))

    groups.sort(key=lambda x: len(x[1]), reverse=True)
    return groups


def _generate_leaf_colors(base_hex: str, n: int) -> List[str]:
    """Generate *n* distinct colors varying lightness/saturation around a base hue."""
    import colorsys

    if n <= 1:
        return [base_hex]

    r = int(base_hex[1:3], 16) / 255
    g = int(base_hex[3:5], 16) / 255
    b = int(base_hex[5:7], 16) / 255
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    colors: List[str] = []
    for i in range(n):
        t = i / max(n - 1, 1)
        li = max(0.28, min(0.62, l - 0.17 + t * 0.34))
        si = max(0.40, min(0.85, s - 0.10 + t * 0.20))
        ri, gi, bi = colorsys.hls_to_rgb(h, li, si)
        colors.append(f"#{int(ri * 255):02x}{int(gi * 255):02x}{int(bi * 255):02x}")
    return colors


# =============================================================================
# Cloud Cell Count Helpers
# =============================================================================

def compute_cloud_cell_counts(
    active_indices: List[int],
    anchor_nodes: np.ndarray,
    is_transitioning: np.ndarray,
    is_atypical: np.ndarray,
    enable_atypical_detection: bool,
) -> Dict[int, int]:
    """Compute per-node cell counts for cloud labels.

    Since transitioning cells are always rendered as separate dots on edges,
    cloud labels should always exclude them.  When atypical detection is ON,
    atypical cells are also excluded (they are shown via pie charts or
    scatter overlay).

    Args:
        active_indices: List of node indices that are active in the visualization.
        anchor_nodes: 1-D array of node indices each cell is assigned to.
        is_transitioning: Boolean array indicating transitioning cells.
        is_atypical: Boolean array indicating atypical cells.
        enable_atypical_detection: Whether atypical detection is enabled.

    Returns:
        Dict mapping node index to cell count.  Nodes with zero cells are
        omitted from the output.
    """
    if len(anchor_nodes) == 0:
        return {}

    # Always exclude transitioning cells — they are rendered as dots on edges,
    # not as part of the stable cloud.
    if enable_atypical_detection:
        mask = ~is_transitioning & ~is_atypical
    else:
        mask = ~is_transitioning

    # Count cells per node from the masked subset
    active_set = set(active_indices)
    counts: Dict[int, int] = {}
    for node in anchor_nodes[mask]:
        node_int = int(node)
        if node_int in active_set:
            counts[node_int] = counts.get(node_int, 0) + 1

    return counts


def compute_cloud_transition_counts(
    active_indices: List[int],
    anchor_nodes: np.ndarray,
    is_transitioning: np.ndarray,
) -> Dict[int, int]:
    """Compute per-node transitioning cell counts for cloud labels.

    When show_transitions=True, transitioning cells are rendered on edges
    rather than in clouds.  This counts how many cells per node are
    transitioning so labels can display ``n=14 (+39)``.
    """
    if len(anchor_nodes) == 0:
        return {}

    active_set = set(active_indices)
    counts: Dict[int, int] = {}
    for node in anchor_nodes[is_transitioning]:
        node_int = int(node)
        if node_int in active_set:
            counts[node_int] = counts.get(node_int, 0) + 1

    return counts


ADAPTIVE_PALETTE = [
            '#E64B35', # Red
            '#0072B5', # Blue
            '#00A087', # Teal
            '#3C5488', # Navy
            '#E18727', # Orange
            '#7876B1', # Purple
            '#EFC000', # Yellow
            '#EE4C97', # Pink
            '#7E6148', # Brown
            '#7F7F7F', # Gray
            # Additional High-Contrast for separation
            '#000000', '#FFFF00', '#1CE6FF','#FF4A46', '#008941', '#006FA6', '#A30059',
            '#FFDBE5', '#0000A6', '#63FFAC', '#B79762', '#004D43', '#8FB0FF', '#997D87'
        ]

# Shared-trunk grays: use a light neutral for lineage-ambiguous scaffolds so the
# colored branches remain the visual focus without implying a lineage identity.
NEUTRAL_GRAY = '#C6C6C6'
RADIAL_ROOT_GRAY = '#A6A6A6'
RADIAL_BRANCH_GRAY = '#B8B8B8'

def _is_shared_trunk_gray(color: Optional[str]) -> bool:
    """Return True for any neutral/shared-trunk gray used in tree rendering."""
    if not color:
        return True

    return color.upper() in {
        NEUTRAL_GRAY.upper(),
        RADIAL_ROOT_GRAY.upper(),
        RADIAL_BRANCH_GRAY.upper(),
        '#333333',
        '#555555',
        '#7F7F7F',
    }


class AdaptiveLineageColorMapper:
    """
    Dynamically color active cell types using ontology-derived groups.
    
    Algorithm:
    1. Count active descendants for ontology nodes.
    2. Select group-root candidates within the configured size bounds.
    3. Apply greedy set cover and assign remaining active nodes to nearby groups.
    4. Determine plot roots and assign distinct colors to the resulting groups.
    5. Propagate each group color to represented descendants.
    """
    
    def __init__(
        self,
        ontology_graph: nx.DiGraph,
        node_names: List[str],
        active_cell_types: List[str],
        color_palette: Optional[List[str]] = None
    ):
        """
        Initialize adaptive color mapper.
        
        Args:
            ontology_graph: NetworkX DiGraph with child -> parent edges (is_a relationships).
                           In this graph, successors(node) returns the node's parents,
                           and predecessors(node) returns the node's children.
            node_names: List of all CL IDs in the ontology (index corresponds to node index).
            active_cell_types: List of CL IDs present in the dataset.
            color_palette: Optional custom color list (default: ADAPTIVE_PALETTE).
        """
        self.graph = ontology_graph
        self.node_names = node_names
        self.name_to_idx = {name: i for i, name in enumerate(node_names)}
        self.color_palette = color_palette if color_palette is not None else ADAPTIVE_PALETTE
        
        # Convert active cell types to indices
        self.active_indices: Set[int] = set()
        for cl_id in active_cell_types:
            if cl_id in self.name_to_idx:
                self.active_indices.add(self.name_to_idx[cl_id])
        
        # Initialize state variables
        self.lca_idx: Optional[int] = None
        self.branches: List[Tuple[int, int]] = []
        self.merged_branches: List[Tuple[int, int, List[int]]] = []
        self.node_to_color: Dict[int, str] = {}
        self._node_to_lineage_key: Dict[int, Optional[str]] = {}
        self._branch_color_mapping: Dict[str, str] = {}

        # Pre-compute depth of every node from the root in the full ontology.
        # Roots are nodes with no parents (out_degree == 0 in child→parent graph).
        # Used by the lineage-walk to evaluate lineage_threshold.
        self._node_depths: Dict[int, int] = {}
        roots = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        from collections import deque as _deque
        _q: _deque = _deque()
        for _r in roots:
            self._node_depths[_r] = 0
            _q.append((_r, 0))
        while _q:
            _cur, _d = _q.popleft()
            for _child in self.graph.predecessors(_cur):
                if _child not in self._node_depths or self._node_depths[_child] < _d + 1:
                    self._node_depths[_child] = _d + 1
                    _q.append((_child, _d + 1))

        # Compute coloring if we have active cell types
        if self.active_indices:
            self._compute_adaptive_coloring()
    
    def _compute_adaptive_coloring(self) -> None:
        """Cluster active cell types using top-down set-cover on the IS_A DAG.

        Algorithm:
        1. For every node, count its active descendants.
        2. Find group root candidates with _MIN_GROUP_SIZE <= descendants <= _MAX_GROUP_SIZE.
        3. Greedy set-cover: take largest candidate first, assign its members, repeat.
        4. Orphans assigned to nearest group root via shortest DAG path (max 5).
        5. Determine LCA and plot roots, then assign colors.
        """
        import logging
        logger = logging.getLogger(__name__)

        active_list = list(self.active_indices)
        if not active_list:
            return

        self.lca_idx = find_lca(self.graph, active_list)
        lca_name = (self.node_names[self.lca_idx]
                    if self.lca_idx is not None and self.lca_idx < len(self.node_names)
                    else "None")
        logger.info("Adaptive coloring: LCA=%s active_count=%d", lca_name, len(active_list))

        raw_groups = _find_group_roots(self.graph, self.active_indices)

        plot_roots = []
        group_active_counts = []
        node_groups = []
        for root_idx, members in raw_groups:
            if root_idx == -1:
                pr = members[0] if members else -1
            else:
                pr = root_idx
            plot_roots.append(pr)
            group_active_counts.append(len(members))
            node_groups.append(set(members))

        self.branches = list(zip(plot_roots, group_active_counts))
        self.merged_branches = [
            (pr, count, list(grp))
            for pr, count, grp in zip(plot_roots, group_active_counts, node_groups)
        ]

        self._assign_colors()

    def _assign_colors(self) -> None:
        """
        Assign colors to branches and build node_to_color mapping.
        
        Algorithm:
        1. Sort merged_branches by total_active_count descending (largest first)
        2. Assign colors from ADAPTIVE_PALETTE in order
        3. For each branch, traverse its subtree and assign the branch's color to all nodes
        4. Assign neutral gray (#7F7F7F) to the LCA node itself
        
        Builds:
        - self.node_to_color: Dict[int, str] mapping node index to hex color
        - self._branch_color_mapping: Dict[str, str] mapping branch CL ID to hex color
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Reset mappings
        self.node_to_color = {}
        self._node_to_lineage_key = {}
        self._branch_color_mapping = {}
        
        # Handle edge case: no merged branches
        if not self.merged_branches:
            logger.warning("_assign_colors: No merged branches to assign colors to.")
            return
        
        # Step 1: Sort merged_branches by total_active_count descending (largest
        # first) for deterministic color assignment
        sorted_branches = sorted(
            self.merged_branches,
            key=lambda x: x[1],  # Sort by total_active_count
            reverse=True  # Largest first
        )
        
        # Step 2: Assign colors from palette in order. Every branch group
        # (including merged groups) gets a distinct palette color so all lineages
        # are visually distinguishable.
        palette_idx = 0
        for _sort_idx, (primary_idx, total_count, merged_indices) in enumerate(sorted_branches):
            color = self.color_palette[palette_idx % len(self.color_palette)]
            palette_idx += 1

            # Get branch name for logging and mapping
            primary_name = self.node_names[primary_idx] if primary_idx < len(self.node_names) else f"idx_{primary_idx}"

            # Store branch color mapping (use primary branch name as key)
            self._branch_color_mapping[primary_name] = color
            
            # Log the color assignment
            logger.info(
                "Color assignment: Branch %s (active_count=%d) -> %s",
                primary_name, total_count, color
            )
            
            # Step 3: Painting is done globally after all groups are processed.
            pass
        
        # ── Global painting: deepest-active-descendant rule ──────────────────
        # Build map: active node → its group's color
        active_node_to_color: Dict[int, str] = {}
        active_node_to_lineage_key: Dict[int, str] = {}
        for primary_idx, total_count, member_indices in sorted_branches:
            primary_name = (self.node_names[primary_idx]
                            if primary_idx < len(self.node_names)
                            else f"idx_{primary_idx}")
            base_color = self._branch_color_mapping.get(primary_name, NEUTRAL_GRAY)
            leaf_colors = _generate_leaf_colors(base_color, len(member_indices))
            for m, lc in zip(member_indices, leaf_colors):
                active_node_to_color[m] = lc
                active_node_to_lineage_key[m] = primary_name

        # Active nodes always get their own group's color unconditionally.
        for active_node, color in active_node_to_color.items():
            self.node_to_color[active_node] = color
            self._node_to_lineage_key[active_node] = active_node_to_lineage_key.get(active_node)

        # Non-active nodes inherit the color of their deepest active descendant.
        for node in self.graph.nodes():
            if node in self.node_to_color:
                continue
            subtree = get_subtree_nodes(self.graph, node)
            active_desc = subtree.intersection(self.active_indices)
            if not active_desc:
                continue
            deepest = max(active_desc,
                          key=lambda n: self._node_depths.get(n, 0))
            self.node_to_color[node] = active_node_to_color.get(deepest, NEUTRAL_GRAY)
            self._node_to_lineage_key[node] = active_node_to_lineage_key.get(deepest)

        # Step 4: Assign neutral gray to the LCA node itself
        if self.lca_idx is not None:
            self.node_to_color[self.lca_idx] = NEUTRAL_GRAY
            self._node_to_lineage_key[self.lca_idx] = None
            lca_name = self.node_names[self.lca_idx] if self.lca_idx < len(self.node_names) else f"idx_{self.lca_idx}"
            logger.info(
                "Color assignment: LCA node %s -> %s (neutral gray)",
                lca_name, NEUTRAL_GRAY
            )
    
    def get_node_color(self, node_idx: int) -> str:
        """
        Get hex color for a node index.
        
        Looks up the node index in the node_to_color mapping and returns the
        corresponding hex color string. Returns neutral gray for unknown nodes,
        negative indices, or out-of-bounds indices.
        
        Args:
            node_idx: Index of the node in node_names list.
            
        Returns:
            Hex color string (e.g., '#E64B35') or neutral gray (#7F7F7F) for:
            - Unknown nodes (not in node_to_color mapping)
            - Negative indices
            - Indices out of bounds (>= len(node_names))
        """
        # Handle edge cases: negative indices or out of bounds
        if node_idx < 0 or node_idx >= len(self.node_names):
            return NEUTRAL_GRAY
        
        # Look up node index in node_to_color mapping
        # Return neutral gray if not found
        return self.node_to_color.get(node_idx, NEUTRAL_GRAY)
    
    def get_colors(self, predictions: np.ndarray) -> List[str]:
        """
        Get hex colors for batch of predictions.
        
        Iterates over the predictions array and calls get_node_color() for each
        prediction index. Returns a list of hex colors with the same length as
        the input array.
        
        Args:
            predictions: NumPy array of node indices. Can contain integers or
                        floats (which will be converted to integers). Empty
                        arrays return empty lists.
            
        Returns:
            List of hex colors, one per prediction. Each color is a valid
            7-character hex string (e.g., '#E64B35'). Unknown or out-of-bounds
            indices return neutral gray (#7F7F7F).
        """
        # Handle empty array edge case
        if len(predictions) == 0:
            return []
        
        # Iterate over predictions and get color for each
        return [self.get_node_color(int(p)) for p in predictions]

    def get_node_lineage_key(self, node_idx: int) -> Optional[str]:
        """Return the lineage identifier assigned to a node, if any."""
        if node_idx < 0 or node_idx >= len(self.node_names):
            return None
        return self._node_to_lineage_key.get(node_idx)
    
    @property
    def branch_color_mapping(self) -> Dict[str, str]:
        """Return mapping of branch CL IDs to their assigned colors."""
        return self._branch_color_mapping.copy()

    def get_legend_items(
        self, formatted_names: Optional[Dict[int, str]] = None
    ) -> List[Tuple[str, str]]:
        """Return ``(display_label, hex_color)`` tuples for legend rendering.

        The tuples are keyed by the **branch progenitor** (the LCA-derived
        root of each structural lineage), rather than a downstream leaf.

        Args:
            formatted_names: Optional mapping ``{node_idx: display_name}``.
                If provided, legend labels use these human-readable names
                instead of raw CL IDs, respecting the user's
                ``label_format`` setting (``'name'``, ``'id'``, or
                ``'both'``).

        Returns:
            List of ``(label, color)`` tuples, one per distinct lineage
            branch.  A ``'Miscellaneous'`` entry is emitted for the merged
            small-branch group when present.
        """
        items: List[Tuple[str, str]] = []
        for cl_id, color in self._branch_color_mapping.items():
            idx = self.name_to_idx.get(cl_id)
            if formatted_names is not None and idx is not None:
                # formatted_names may be a list (indexed by node position)
                # or a dict (keyed by node index).  Handle both.
                if isinstance(formatted_names, dict):
                    label = formatted_names.get(idx, cl_id)
                else:
                    # list / tuple — use index bounds check
                    label = formatted_names[idx] if idx < len(formatted_names) else cl_id
            else:
                label = cl_id
            items.append((label, color))
        return items
    
    @property
    def lca_node(self) -> Optional[str]:
        """Return the computed LCA node CL ID."""
        if self.lca_idx is not None and self.lca_idx < len(self.node_names):
            return self.node_names[self.lca_idx]
        return None


def _generate_related_split_colors(base_color: str, count: int) -> List[str]:
    """Generate related colors for disconnected visible components of one lineage."""
    if count <= 1:
        return [base_color]

    h, s, l = hex_to_hsl(base_color)
    variant_specs = [
        (0.0, 0.00, 0.00),
        (-0.03, 0.10, 0.06),
        (0.03, -0.04, -0.06),
        (-0.06, 0.12, -0.02),
        (0.06, -0.02, 0.10),
        (-0.09, 0.14, -0.08),
        (0.09, -0.06, 0.14),
    ]

    colors: List[str] = []
    for idx in range(count):
        hue_shift, sat_shift, light_shift = variant_specs[idx % len(variant_specs)]
        wrap_cycle = idx // len(variant_specs)
        hue = (h + hue_shift + (0.015 * wrap_cycle)) % 1.0
        sat = min(1.0, max(0.25, s + sat_shift - (0.02 * wrap_cycle)))
        light = min(0.75, max(0.25, l + light_shift + (0.015 * wrap_cycle)))
        colors.append(hsl_to_hex(hue, sat, light))
    return colors


def _build_child_to_parent_ontology_graph(
    adjacency_matrix: np.ndarray,
    n_nodes: int,
) -> nx.DiGraph:
    """Build the standard child->parent ontology graph from an adjacency matrix."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n_nodes))

    rows, cols = np.where(adjacency_matrix > 0)
    for child_idx, parent_idx in zip(rows, cols):
        graph.add_edge(int(child_idx), int(parent_idx))

    return graph


class ProjectedLineageColorMapper:
    """Project ontology-derived lineage groups onto the rendered tree topology.

    The base ``AdaptiveLineageColorMapper`` discovers biological lineages in the
    full ontology. This wrapper keeps those lineage assignments, but recomputes
    the visible branch colors on the rendered tree/DAG so that each displayed
    color corresponds to one connected visible component.

    The mapper is layout-agnostic. Callers provide the visible graph that should
    be colored:
    - radial mode passes the DAG-to-tree converted nodes and edges, along with a
      ``visible_to_source`` mapping back to the full ontology
    - horizontal mode passes the plotted DAG edges and ``visible_node_indices``
      so only actually rendered ontology nodes participate in the projection

    Legend entries are derived from family-pure visible components. Mixed/shared
    visible nodes are intentionally left unassigned so the legend does not
    claim that a shared progenitor belongs to only one lineage family.
    """

    def __init__(
        self,
        base_mapper: AdaptiveLineageColorMapper,
        visible_node_names: List[str],
        visible_edges: List[Tuple[int, int]],
        visible_node_indices: Optional[List[int]] = None,
        visible_to_source: Optional[Dict[int, int]] = None,
        legend_label_lookup: Optional[Dict[str, str]] = None,
    ):
        """Initialize a projected mapper on a rendered tree or DAG.

        Args:
            base_mapper: Full-ontology lineage mapper learned from active cell
                types in the original ontology.
            visible_node_names: Node-name array used by the rendered layout.
                In horizontal mode this is usually the full ontology name list;
                in radial mode this is the tree-converted node list.
            visible_edges: Directed edges in the rendered graph, always using
                the node indices expected by ``visible_node_names``.
            visible_node_indices: Optional subset of node indices that are
                truly visible in the rendered graph. This is mainly used by the
                horizontal layout, where ``visible_node_names`` may cover the
                full ontology but only ``active_indices`` are plotted.
            visible_to_source: Optional mapping from rendered-node index to full
                ontology node index. Radial tree conversion uses this to thread
                cloned tree nodes back to their ontology source nodes.
            legend_label_lookup: Optional lineage-family label lookup inherited
                from the base mapper for fallback legend labeling.
        """
        self.base_mapper = base_mapper
        self.visible_node_names = visible_node_names
        if visible_node_indices is None:
            if visible_to_source is not None:
                visible_node_indices = sorted(visible_to_source.keys())
            else:
                visible_node_indices = list(range(len(visible_node_names)))
        else:
            visible_node_indices = sorted(set(visible_node_indices))

        self.visible_node_indices = visible_node_indices
        self.visible_to_source = visible_to_source or {
            idx: idx for idx in self.visible_node_indices
        }
        self.legend_label_lookup = legend_label_lookup or {}

        self.visible_graph = nx.DiGraph()
        self.visible_graph.add_nodes_from(self.visible_node_indices)
        self.visible_graph.add_edges_from(visible_edges)

        self.node_colors: Dict[int, str] = {}
        self.node_lineage_keys: Dict[int, Optional[str]] = {}
        self._legend_entries: List[Tuple[int, str, str, str]] = []
        self._project_colors()

    def _compute_visible_depths(self) -> Dict[int, int]:
        """Compute visible-graph depths used for stable component ranking.

        For DAG inputs, the depth is the topological generation of a node in the
        rendered graph. For non-DAG edge cases, a simple root-to-child BFS depth
        fallback is used so representative selection remains deterministic.
        """
        depths = {node: 0 for node in self.visible_graph.nodes()}
        if self.visible_graph.number_of_nodes() == 0:
            return depths

        try:
            for generation, nodes in enumerate(nx.topological_generations(self.visible_graph)):
                for node in nodes:
                    depths[node] = generation
        except nx.NetworkXError:
            roots = [node for node in self.visible_graph.nodes() if self.visible_graph.in_degree(node) == 0]
            queue = [(root, 0) for root in roots]
            while queue:
                node, depth = queue.pop(0)
                depths[node] = max(depths.get(node, 0), depth)
                for child in self.visible_graph.successors(node):
                    queue.append((child, depth + 1))

        return depths

    def _resolve_component_root(
        self,
        component: Set[int],
        visible_depths: Dict[int, int],
    ) -> int:
        """Choose a stable visible root for one family-pure component.

        The selected node is the shallowest node in the component whose parents,
        if any, are outside the component. This is used for consistent ordering
        and as a stable summary of the component's position in the rendered
        topology.
        """
        component_parents = {
            node: [parent for parent in self.visible_graph.predecessors(node) if parent in component]
            for node in component
        }
        candidate_roots = [node for node, parents in component_parents.items() if not parents]
        if not candidate_roots:
            candidate_roots = list(component)

        candidate_roots.sort(key=lambda node: (visible_depths.get(node, 0), node))
        return candidate_roots[0]

    def _resolve_component_legend_representative(
        self,
        component: Set[int],
        visible_depths: Dict[int, int],
    ) -> Tuple[int, bool]:
        """Choose the legend representative for one family-pure component.

        Returns:
            ``(node_idx, is_progenitor)`` where ``is_progenitor`` is True when
            the chosen node has at least one child in the same component.

        Preference is given to the shallowest non-global progenitor within the
        component. If a component has no internal branching structure, the
        shallowest node in the component is returned and marked as a leaf
        fallback candidate.
        """
        component_set = set(component)
        sorted_nodes = sorted(component_set, key=lambda node: (visible_depths.get(node, 0), node))

        non_global_nodes = [
            node for node in sorted_nodes
            if self.visible_graph.in_degree(node) > 0
        ]
        if non_global_nodes:
            sorted_nodes = non_global_nodes

        progenitor_nodes = [
            node for node in sorted_nodes
            if any(child in component_set for child in self.visible_graph.successors(node))
        ]
        if progenitor_nodes:
            return progenitor_nodes[0], True

        return sorted_nodes[0], False

    def _project_colors(self) -> None:
        """Project ontology lineages onto the visible graph and build legend entries.

        The algorithm has three steps:
        1. propagate lineage-family assignments upward through the visible graph
           and mark a node as resolved only when all visible descendants belong
           to the same lineage family
        2. split each resolved lineage family into connected visible components,
           assigning related shades so disconnected components remain visually
           distinguishable
        3. derive legend entries from family-pure visible components, preferring
           progenitor-like nodes and falling back to a pure leaf only when a
           family has no visible progenitor component
        """
        visible_depths = self._compute_visible_depths()
        base_branch_colors = self.base_mapper.branch_color_mapping
        lineage_order = list(base_branch_colors.keys())

        seed_lineages: Dict[int, Optional[str]] = {}
        for visible_idx in self.visible_graph.nodes():
            source_idx = self.visible_to_source.get(visible_idx)
            if source_idx is None or source_idx not in self.base_mapper.active_indices:
                seed_lineages[visible_idx] = None
                continue
            seed_lineages[visible_idx] = self.base_mapper.get_node_lineage_key(source_idx)

        descendant_lineages: Dict[int, Set[str]] = {}
        topo_nodes = list(nx.topological_sort(self.visible_graph)) if self.visible_graph.number_of_nodes() else []

        for visible_idx in reversed(topo_nodes):
            visible_keys: Set[str] = set()
            seed_key = seed_lineages.get(visible_idx)
            if seed_key is not None:
                visible_keys.add(seed_key)
            for child_idx in self.visible_graph.successors(visible_idx):
                visible_keys.update(descendant_lineages.get(child_idx, set()))
            descendant_lineages[visible_idx] = visible_keys

            if len(visible_keys) == 1:
                self.node_lineage_keys[visible_idx] = next(iter(visible_keys))
            else:
                self.node_lineage_keys[visible_idx] = None

        components_by_lineage: Dict[str, List[Tuple[Set[int], int, int, int, bool]]] = defaultdict(list)
        seed_set = {node for node, lineage_key in seed_lineages.items() if lineage_key is not None}

        for lineage_key in lineage_order:
            lineage_nodes = [
                node for node, node_key in self.node_lineage_keys.items()
                if node_key == lineage_key
            ]
            if not lineage_nodes:
                continue

            lineage_subgraph = self.visible_graph.subgraph(lineage_nodes).to_undirected()
            connected_components = [set(component) for component in nx.connected_components(lineage_subgraph)]

            for component in connected_components:
                component_root = self._resolve_component_root(component, visible_depths)
                legend_root, legend_is_progenitor = self._resolve_component_legend_representative(
                    component,
                    visible_depths,
                )
                active_seed_count = sum(1 for node in component if node in seed_set)
                components_by_lineage[lineage_key].append(
                    (
                        component,
                        component_root,
                        active_seed_count,
                        legend_root,
                        legend_is_progenitor,
                    )
                )

            components_by_lineage[lineage_key].sort(
                key=lambda item: (
                    not item[4],
                    visible_depths.get(item[3], 0),
                    -item[2],
                    -len(item[0]),
                    item[3],
                )
            )

        self.node_colors = {node: NEUTRAL_GRAY for node in self.visible_graph.nodes()}
        self._legend_entries = []

        for lineage_key in lineage_order:
            lineage_components = components_by_lineage.get(lineage_key, [])
            if not lineage_components:
                continue

            base_color = base_branch_colors.get(lineage_key, NEUTRAL_GRAY)
            component_colors = _generate_related_split_colors(base_color, len(lineage_components))

            first_component_entry: Optional[Tuple[int, str]] = None
            lineage_has_progenitor_entry = False

            for (
                component_nodes,
                component_root,
                _seed_count,
                legend_root,
                legend_is_progenitor,
            ), component_color in zip(
                lineage_components,
                component_colors,
            ):
                for node in component_nodes:
                    self.node_colors[node] = component_color

                if first_component_entry is None:
                    first_component_entry = (legend_root, component_color)

                if not legend_is_progenitor:
                    continue

                self._legend_entries.append(
                    (legend_root, component_color, lineage_key, "pure_component_progenitor")
                )
                lineage_has_progenitor_entry = True

            if lineage_has_progenitor_entry or first_component_entry is None:
                continue

            fallback_root, fallback_color = first_component_entry
            self._legend_entries.append(
                (fallback_root, fallback_color, lineage_key, "pure_component_leaf_fallback")
            )

    def get_node_color(self, node_idx: int) -> str:
        """Return the projected color for one rendered node index.

        Unknown, negative, or unprojected nodes fall back to ``NEUTRAL_GRAY``.
        In horizontal mode, non-plotted ontology nodes also fall back to gray
        because they are excluded from ``visible_node_indices``.
        """
        if node_idx < 0 or node_idx >= len(self.visible_node_names):
            return NEUTRAL_GRAY
        return self.node_colors.get(node_idx, NEUTRAL_GRAY)

    def get_colors(self, predictions: np.ndarray) -> List[str]:
        """Return colors for a batch of rendered node indices."""
        return [self.get_node_color(int(pred)) for pred in predictions]

    def get_legend_items(
        self,
        formatted_names: Optional[Dict[int, str]] = None,
    ) -> List[Tuple[str, str]]:
        """Return legend items for family-pure visible branches.

        Each item corresponds to one family-pure visible component chosen by
        ``_project_colors``. Families may contribute multiple legend entries
        when they split into multiple disconnected visible progenitor branches.
        Mixed/shared nodes are intentionally excluded.
        """
        items: List[Tuple[str, str]] = []
        for root_idx, color, lineage_key, selection_mode in self._legend_entries:
            label = None
            label_source = "unset"

            if formatted_names is not None:
                if isinstance(formatted_names, dict):
                    label = formatted_names.get(root_idx)
                    if label is not None:
                        label_source = "formatted_names[dict]"
                else:
                    if 0 <= root_idx < len(formatted_names):
                        label = formatted_names[root_idx]
                        if label is not None:
                            label_source = "formatted_names[list]"

            if label is None:
                if 0 <= root_idx < len(self.visible_node_names):
                    label = self.visible_node_names[root_idx]
                    label_source = "visible_node_name_fallback"

            if label is None:
                label = self.legend_label_lookup.get(lineage_key)
                if label is not None:
                    label_source = "legend_label_lookup"

            if label is None:
                label = lineage_key
                label_source = "lineage_key_fallback"

            items.append((label, color))
        return items


def _create_projected_lineage_color_mapper(
    pure_adjacency: np.ndarray,
    original_node_names: List[str],
    active_source_indices: List[int],
    visible_node_names: List[str],
    visible_edges: List[Tuple[int, int]],
    visible_node_indices: Optional[List[int]] = None,
    color_palette: Optional[List[str]] = None,
    visible_to_source: Optional[Dict[int, int]] = None,
    legend_formatted_names: Optional[Any] = None,
) -> ProjectedLineageColorMapper:
    """Create a projected lineage mapper for a rendered tree or DAG view.

    This helper learns lineage families once on the full ontology, then
    reprojects those families onto a specific rendered topology.

    Args:
        pure_adjacency: Full ontology adjacency in child-to-parent convention.
        original_node_names: Full ontology node-name list.
        active_source_indices: Active ontology node indices represented in the
            current dataset.
        visible_node_names: Node-name array used by the rendered layout.
        visible_edges: Directed edges in the rendered graph.
        visible_node_indices: Optional subset of ``visible_node_names`` that are
            actually plotted. Horizontal mode uses this to restrict projection to
            ``active_indices`` while keeping original ontology indexing.
        color_palette: Optional palette forwarded to
            ``AdaptiveLineageColorMapper``.
        visible_to_source: Optional rendered-node to ontology-node mapping.
            Radial mode uses this when tree conversion duplicates ontology
            nodes.
        legend_formatted_names: Optional formatted full-ontology labels used to
            seed family-level legend name fallbacks.

    Returns:
        A ``ProjectedLineageColorMapper`` configured for the rendered view.
    """
    ontology_graph = _build_child_to_parent_ontology_graph(
        adjacency_matrix=pure_adjacency,
        n_nodes=len(original_node_names),
    )
    active_cell_types = [
        original_node_names[idx]
        for idx in sorted(set(active_source_indices))
        if 0 <= idx < len(original_node_names)
    ]

    base_mapper = AdaptiveLineageColorMapper(
        ontology_graph=ontology_graph,
        node_names=original_node_names,
        active_cell_types=active_cell_types,
        color_palette=color_palette,
    )

    legend_items = base_mapper.get_legend_items(formatted_names=legend_formatted_names)
    legend_label_lookup = {
        lineage_key: label
        for lineage_key, (label, _color) in zip(base_mapper.branch_color_mapping.keys(), legend_items)
    }

    return ProjectedLineageColorMapper(
        base_mapper=base_mapper,
        visible_node_names=visible_node_names,
        visible_edges=visible_edges,
        visible_node_indices=visible_node_indices,
        visible_to_source=visible_to_source,
        legend_label_lookup=legend_label_lookup,
    )


# HierarchicalColorMapper has been removed.
# All visualizations now use AdaptiveLineageColorMapper exclusively.
# If you encounter errors, ensure your model checkpoint contains 'pure_adjacency_matrix'.


class SimplifiedTreeColorMapper:
    """Color mapper for simplified trees where each subtree-of-root is a color family.

    Unlike AdaptiveLineageColorMapper (which discovers lineage groups via
    set-cover on the full ontology), this mapper treats the simplified tree's
    own structure as the grouping: each direct child of the root defines a
    color family, and active nodes within that subtree share a hue with
    per-node shade variation.
    """

    def __init__(
        self,
        tree_edges: List[Tuple[int, int]],
        active_indices: List[int],
        node_names: List[str],
        color_palette: Optional[List[str]] = None,
    ):
        self.node_names = node_names
        self.color_palette = color_palette if color_palette is not None else ADAPTIVE_PALETTE
        self._active_set = set(active_indices)

        self.node_to_color: Dict[int, str] = {}
        self._family_roots: List[int] = []
        self._family_base_colors: Dict[int, str] = {}

        self._assign_colors(tree_edges)

    def _assign_colors(self, tree_edges: List[Tuple[int, int]]) -> None:
        if not tree_edges:
            self._flat_coloring()
            return

        G = nx.DiGraph()
        G.add_edges_from(tree_edges)
        G.add_nodes_from(range(len(self.node_names)))

        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
        if not roots:
            self._flat_coloring()
            return
        root = roots[0]
        self.node_to_color[root] = NEUTRAL_GRAY

        family_roots = self._find_family_roots(G, root)
        self._family_roots = family_roots

        for fi, froot in enumerate(family_roots):
            base_color = self.color_palette[fi % len(self.color_palette)]
            self._family_base_colors[froot] = base_color

            subtree_nodes = set(nx.descendants(G, froot)) | {froot}
            active_in_family = sorted(subtree_nodes & self._active_set)

            if active_in_family:
                shades = _generate_leaf_colors(base_color, len(active_in_family))
                for node_idx, shade in zip(active_in_family, shades):
                    self.node_to_color[node_idx] = shade

            inactive_in_family = subtree_nodes - self._active_set - {root}
            for node_idx in inactive_in_family:
                if node_idx not in self.node_to_color:
                    self.node_to_color[node_idx] = base_color

    def _find_family_roots(self, G: nx.DiGraph, root: int) -> List[int]:
        """Walk down from root until branching is found."""
        current = root
        while True:
            children = list(G.successors(current))
            if len(children) == 0:
                return [current] if current != root else []
            if len(children) == 1:
                current = children[0]
                self.node_to_color[current] = NEUTRAL_GRAY
                continue
            return children

    def _flat_coloring(self) -> None:
        """Fallback: each active node gets a unique color from the palette."""
        active_sorted = sorted(self._active_set)
        if not active_sorted:
            return
        shades = _generate_leaf_colors(
            self.color_palette[0], len(active_sorted)
        )
        for node_idx, shade in zip(active_sorted, shades):
            self.node_to_color[node_idx] = shade
        self._family_roots = active_sorted
        for i, node_idx in enumerate(active_sorted):
            self._family_base_colors[node_idx] = self.color_palette[
                i % len(self.color_palette)
            ]

    def get_node_color(self, node_idx: int) -> str:
        if node_idx < 0 or node_idx >= len(self.node_names):
            return NEUTRAL_GRAY
        return self.node_to_color.get(node_idx, NEUTRAL_GRAY)

    def get_colors(self, predictions: np.ndarray) -> List[str]:
        if len(predictions) == 0:
            return []
        return [self.get_node_color(int(p)) for p in predictions]

    def get_legend_items(
        self, formatted_names: Optional[Dict[int, str]] = None
    ) -> List[Tuple[str, str]]:
        items: List[Tuple[str, str]] = []
        for froot in self._family_roots:
            if formatted_names is not None and isinstance(formatted_names, dict):
                label = formatted_names.get(froot, self.node_names[froot])
            elif formatted_names is not None and not isinstance(formatted_names, dict):
                label = (
                    formatted_names[froot]
                    if 0 <= froot < len(formatted_names)
                    else self.node_names[froot]
                )
            else:
                label = self.node_names[froot]
            color = self._family_base_colors.get(
                froot, self.color_palette[0]
            )
            items.append((label, color))
        return items


def _create_simplified_tree_color_mapper(
    tree_edges: List[Tuple[int, int]],
    active_indices: List[int],
    node_names: List[str],
    color_palette: Optional[List[str]] = None,
) -> SimplifiedTreeColorMapper:
    """Create a color mapper for simplified tree rendering."""
    return SimplifiedTreeColorMapper(
        tree_edges=tree_edges,
        active_indices=active_indices,
        node_names=node_names,
        color_palette=color_palette,
    )


class HighlightColorMapper:
    """Assigns a unique distinct color to each category string."""

    DEFAULT_PALETTE: List[str] = [
        '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
        '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85',
        '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF', '#AEC7E8',
        '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5', '#C49C94',
    ]

    def __init__(self, categories: List[str], palette: Optional[List[str]] = None):
        """
        Args:
            categories: Ordered list of unique category strings.
            palette: Optional custom palette. Falls back to DEFAULT_PALETTE.
        """
        self._palette = palette if palette else self.DEFAULT_PALETTE
        self._color_map: Dict[str, str] = {
            cat: self._palette[i % len(self._palette)]
            for i, cat in enumerate(categories)
        }

    def get_color(self, category: str) -> str:
        """Return hex color for a category. KeyError if unknown."""
        return self._color_map[category]

    def get_legend_items(self) -> List[Tuple[str, str]]:
        """Return (category, color) tuples for legend rendering."""
        return list(self._color_map.items())


# =============================================================================
# Wang Semantic Similarity Engine
# =============================================================================

class WangSimilarityEngine:
    """Compute pairwise Wang Semantic Similarity between cell types.

    Implements the algorithm from Wang et al. (2007) "A new method to measure
    the semantic similarity of GO terms", Bioinformatics 23(10):1274-81.

    The Wang method assigns each ancestor of a term a contribution (S-value)
    that decays with distance from the query term. For IS_A edges the default
    weight factor is 0.8 (the paper's Section 5 recommendation). Similarity
    between terms A and B is the sum of shared ancestor S-values divided by
    the total S-values of both terms.

    S-value computation (Equation 1):
        S_A(A) = 1
        S_A(t) = max{w_is_a * S_A(t') | t' in children_of(t) ∩ DAG_A}

    The max operation ensures that when an ancestor is reachable via multiple
    paths, the highest-weighted path determines its contribution.

    Similarity formula (Equation 3):
        S_GO(A, B) = sum(S_A(t) + S_B(t) for t in T_A ∩ T_B) / (SV(A) + SV(B))

    where SV(X) = sum of all S-values for term X (Equation 2).

    Since cl.obo only has IS_A edges (no part-of), a single weight factor
    w_is_a is used throughout.
    """

    def __init__(self, nx_graph: nx.DiGraph, w_is_a: float = 0.8):
        """Initialize with ontology graph.

        Args:
            nx_graph: NetworkX DiGraph with child→parent edges.
                      ``graph.successors(node)`` returns the node's parents.
            w_is_a: Weight decay factor for IS_A edges. Must be in (0.0, 1.0).
                    The paper recommends 0.8 (default).

        Raises:
            ValueError: If ``w_is_a`` is not strictly in (0.0, 1.0).
        """
        if not (0.0 < w_is_a < 1.0):
            raise ValueError(
                f"w_is_a must be in (0.0, 1.0), got {w_is_a!r}."
            )
        self.graph = nx_graph
        self.w_is_a = w_is_a

    def compute_s_values(self, node_idx: int) -> Dict[int, float]:
        """Compute S-values for all ancestors of a node (Equation 1).

        Starting from the query node (S-value = 1.0), propagates upward
        through IS_A edges via BFS. For each ancestor t the S-value is:

            S_A(t) = max{w_is_a * S_A(t') | t' in children_of(t) ∩ DAG_A}

        When an ancestor is reachable via multiple paths the maximum weighted
        contribution across all paths is kept.

        Args:
            node_idx: Index of the query node.

        Returns:
            Dict mapping ancestor_idx → S-value. Includes the query node
            itself with S-value = 1.0.
        """
        s_values: Dict[int, float] = {node_idx: 1.0}
        queue = [node_idx]

        while queue:
            current = queue.pop(0)
            current_s = s_values[current]
            # graph.successors(current) yields parents (child→parent edges)
            for parent in self.graph.successors(current):
                candidate = self.w_is_a * current_s
                if parent not in s_values:
                    s_values[parent] = candidate
                    queue.append(parent)
                elif candidate > s_values[parent]:
                    # Better path found — update and re-enqueue to propagate
                    s_values[parent] = candidate
                    queue.append(parent)

        return s_values

    def semantic_value(self, node_idx: int) -> float:
        """Compute the total semantic value SV(A) = sum of all S-values (Equation 2).

        Args:
            node_idx: Index of the query node.

        Returns:
            SV(A) as a float.
        """
        return sum(self.compute_s_values(node_idx).values())

    def similarity(self, node_a: int, node_b: int) -> float:
        """Compute Wang similarity between two nodes (Equation 3).

        S_GO(A, B) = sum(S_A(t) + S_B(t) for t in T_A ∩ T_B)
                     / (SV(A) + SV(B))

        Note: S_A(t) may differ from S_B(t) for the same ancestor t because
        the paths from A and B to t may differ in length.

        Args:
            node_a: Index of the first node.
            node_b: Index of the second node.

        Returns:
            Similarity score in [0.0, 1.0]. Returns 0.0 when SV(A) + SV(B) == 0.
        """
        s_a = self.compute_s_values(node_a)
        s_b = self.compute_s_values(node_b)

        sv_a = sum(s_a.values())
        sv_b = sum(s_b.values())
        denom = sv_a + sv_b
        if denom == 0.0:
            return 0.0

        shared = s_a.keys() & s_b.keys()
        numerator = sum(s_a[t] + s_b[t] for t in shared)
        return numerator / denom

    def compute_similarity_matrix(self, node_indices: List[int]) -> np.ndarray:
        """Compute pairwise similarity matrix for a set of nodes.

        Precomputes S-values for all nodes once, then computes pairwise
        similarities. For M nodes this is O(M * ancestors) precomputation
        followed by O(M² * shared_ancestors) pairwise computation.

        Args:
            node_indices: List of node indices to compare.

        Returns:
            N×N symmetric numpy array with dtype float64 and diagonal = 1.0,
            where N = len(node_indices).
        """
        n = len(node_indices)
        # Precompute S-values and semantic values for every node once
        all_s: Dict[int, Dict[int, float]] = {
            idx: self.compute_s_values(idx) for idx in node_indices
        }
        all_sv: Dict[int, float] = {
            idx: sum(s.values()) for idx, s in all_s.items()
        }

        matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            matrix[i, i] = 1.0
            idx_i = node_indices[i]
            for j in range(i + 1, n):
                idx_j = node_indices[j]
                denom = all_sv[idx_i] + all_sv[idx_j]
                if denom == 0.0:
                    sim = 0.0
                else:
                    shared = all_s[idx_i].keys() & all_s[idx_j].keys()
                    numerator = sum(all_s[idx_i][t] + all_s[idx_j][t] for t in shared)
                    sim = numerator / denom
                matrix[i, j] = sim
                matrix[j, i] = sim  # Symmetric

        return matrix


# =============================================================================
# OntologySimplifier
# =============================================================================

class OntologySimplifier:
    """Orchestrate cell type grouping, hierarchical clustering, and LCA collapse.

    Reduces a complex cell ontology tree (potentially hundreds of cell types)
    into fewer, clearer groups.

    Three algorithms are supported:
      - ``"lineage"`` (default): top-down set-cover on the IS_A DAG, groups
        by natural subtree structure bounded by min/max group size.
      - ``"wang"``: Wang Semantic Similarity (Wang et al., 2007) using IS_A
        edge weights, followed by agglomerative clustering and LCA collapse.
      - ``"level"``: rolls up all cell types deeper than a target ontology
        depth to their ancestor at that depth.
    """

    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        pure_ontology_adj: np.ndarray,
        node_names: List[str],
        node_vectors: np.ndarray,
        id_to_name_map: Dict[str, str],
        co_graph: Dict[str, List[str]],
    ) -> None:
        """Initialize with ontology data from the predictor.

        Builds a NetworkX DiGraph from ``pure_ontology_adj`` using the same
        child→parent convention as ``AdaptiveOntologySubgraph._build_nx_graph()``:
        ``adj[child, parent] = 1`` means add edge ``child → parent``.

        Args:
            adjacency_matrix: Full ontology adjacency [n_nodes, n_nodes],
                ``adj[child, parent] = 1``.
            pure_ontology_adj: Pure IS_A adjacency [n_nodes, n_nodes].
            node_names: List of all CL IDs in the ontology (index → CL ID).
            node_vectors: GAT embeddings [n_nodes, embed_dim].
            id_to_name_map: CL ID → human-readable name mapping.
            co_graph: Child→parent graph dict (not used for NX construction
                here; ``pure_ontology_adj`` is used directly).
        """
        self.adjacency_matrix = adjacency_matrix
        self.pure_ontology_adj = pure_ontology_adj
        self.node_names = node_names
        self.node_vectors = node_vectors
        self.id_to_name_map = id_to_name_map
        self.co_graph = co_graph

        # Build name → index lookup (CL ID → integer index)
        self.name_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(node_names)
        }

        # Build NetworkX DiGraph from pure_ontology_adj (child→parent edges)
        n = len(node_names)
        G = nx.DiGraph()
        for i in range(n):
            G.add_node(i, name=node_names[i])
        rows, cols = np.where(pure_ontology_adj > 0)
        for child, parent in zip(rows, cols):
            if child != parent:
                G.add_edge(int(child), int(parent))
        self.nx_graph: nx.DiGraph = G

    def simplify(
        self,
        cell_type_counts: Dict[str, int],
        similarity_threshold: float = 0.85,
        algorithm: str = "lineage",
        target_level: Optional[int] = None,
        min_group_size: int = 2,
        max_group_size: int = 8,
        cell_embeddings: Optional[np.ndarray] = None,
        prediction_indices: Optional[np.ndarray] = None,
        min_cells_number: int = 1,
    ) -> Dict[str, Any]:
        """Run the full simplification pipeline.

        Args:
            cell_type_counts: Dict mapping CL_ID -> cell count.
            similarity_threshold: Clustering cutoff in (0.0, 1.0).  Only
                used by ``"wang"``; ignored by other algorithms.
            algorithm: ``"lineage"`` (default), ``"wang"``, or ``"level"``.
            target_level: Target ontology depth for ``"level"`` algorithm.
            min_group_size: Minimum members per group for ``"lineage"``
                algorithm (default 2).
            max_group_size: Maximum members per group for ``"lineage"``
                algorithm (default 8).  Counted on the split tree, so a node
                covers fewer types than it would on the raw ontology.
            cell_embeddings: Cell embeddings, ``"lineage"`` only — see
                :meth:`_lineage_grouping`.
            prediction_indices: Per-cell predicted node index, ``"lineage"``
                only.
            min_cells_number: Branch pruning threshold, ``"lineage"`` only.

        Returns:
            Dict with keys:
                - ``'simplification_map'``: Dict[str, str]
                - ``'simplified_counts'``: Dict[str, int]
                - ``'similarity_matrix'``: np.ndarray or None
                - ``'tree_nodes'``: Set[str], ``"lineage"`` only — the nodes to
                  build the simplified tree from, including ancestors that hold
                  no cells but join two groups.  Absent for the other
                  algorithms, whose callers fall back to the map's values.

        Raises:
            ValueError: If ``algorithm`` is not a supported value.
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # ------------------------------------------------------------------
        # Validate algorithm
        # ------------------------------------------------------------------
        valid_algorithms = ("lineage", "wang", "level")
        if algorithm == "shortest_path":
            raise ValueError(
                "algorithm='shortest_path' has been removed. "
                "Use algorithm='lineage' instead."
            )
        if algorithm not in valid_algorithms:
            raise ValueError(
                f"algorithm must be one of {valid_algorithms}, got '{algorithm}'"
            )

        # ------------------------------------------------------------------
        # Early dispatch for level-based roll-up
        # ------------------------------------------------------------------
        if algorithm == "level":
            return self._level_rollup(cell_type_counts, target_level)

        # ------------------------------------------------------------------
        # Early dispatch for lineage-based grouping
        # ------------------------------------------------------------------
        if algorithm == "lineage":
            return self._lineage_grouping(
                cell_type_counts, min_group_size, max_group_size,
                cell_embeddings=cell_embeddings,
                prediction_indices=prediction_indices,
                min_cells_number=min_cells_number,
            )

        # ------------------------------------------------------------------
        # "wang" algorithm: similarity matrix + agglomerative clustering
        # ------------------------------------------------------------------
        cl_ids = list(cell_type_counts.keys())
        active_indices: List[int] = [self.name_to_idx[cl] for cl in cl_ids]
        n = len(active_indices)

        if n == 1:
            similarity_matrix = np.array([[1.0]])
        else:
            engine = WangSimilarityEngine(self.nx_graph)
            similarity_matrix = engine.compute_similarity_matrix(active_indices)

        # ------------------------------------------------------------------
        # Stage 3: Hierarchical clustering
        # ------------------------------------------------------------------
        if n == 1:
            # Singleton — skip clustering
            cluster_labels = np.array([1])
        else:
            distance_matrix = 1.0 - similarity_matrix
            # Clip to [0, 1] to guard against tiny floating-point negatives
            np.clip(distance_matrix, 0.0, 1.0, out=distance_matrix)
            np.fill_diagonal(distance_matrix, 0.0)

            condensed = squareform(distance_matrix, checks=False)
            Z = linkage(condensed, method="average")
            cluster_labels = fcluster(
                Z, t=(1.0 - similarity_threshold), criterion="distance"
            )

        # Group CL IDs by cluster label
        clusters: Dict[int, List[str]] = {}
        for cl_id, label in zip(cl_ids, cluster_labels):
            clusters.setdefault(int(label), []).append(cl_id)

        # ------------------------------------------------------------------
        # Stage 4: LCA collapse and map construction
        # ------------------------------------------------------------------
        simplification_map: Dict[str, str] = {}
        simplified_counts: Dict[str, int] = {}

        for label, members in clusters.items():
            if len(members) == 1:
                # Singleton cluster — map to itself
                cl_id = members[0]
                simplification_map[cl_id] = cl_id
                simplified_counts[cl_id] = (
                    simplified_counts.get(cl_id, 0) + cell_type_counts[cl_id]
                )
            else:
                # Multi-member cluster — find LCA
                member_indices = [self.name_to_idx[cl] for cl in members]
                lca_idx = find_lca(self.nx_graph, member_indices)
                lca_cl_id = self.node_names[lca_idx]

                aggregated_count = sum(cell_type_counts[cl] for cl in members)
                simplified_counts[lca_cl_id] = (
                    simplified_counts.get(lca_cl_id, 0) + aggregated_count
                )
                for cl_id in members:
                    simplification_map[cl_id] = lca_cl_id

        return {
            "simplification_map": simplification_map,
            "simplified_counts": simplified_counts,
            "similarity_matrix": similarity_matrix,
        }

    def _lineage_grouping(
        self,
        cell_type_counts: Dict[str, int],
        min_group_size: int,
        max_group_size: int,
        cell_embeddings: Optional[np.ndarray] = None,
        prediction_indices: Optional[np.ndarray] = None,
        min_cells_number: int = 1,
    ) -> Dict[str, Any]:
        """Group cell types by lineage, splitting shared types onto each branch first.

        1. Duplicate types with several parents, one copy per branch
           (``DAGToTreeConverter``), so no branch loses a type another claimed.
        2. Pick group roots greedily on that tree, bounded by *min_group_size* /
           *max_group_size*.  Types no group claims keep their own label.
        3. Keep the ancestors joining two or more groups, so the simplified tree
           has edges; without them the groups are siblings and no cell can be
           measured as moving between two of them.
        4. Map each type to the nearest surviving ancestor of the branch its
           cells landed on, so a label can only get broader.  A type whose cells
           span branches takes the branch holding most of them.
        5. Fold every group holding fewer than *min_cells_number* cells into the
           nearest node above it that the tree already draws — another group, or
           one of the ancestors kept in step 3.  The walk runs on the split tree,
           where each copy has exactly one parent, so the destination is the
           branch the cells were placed on rather than an arbitrary parent of a
           type with several.  An ancestor kept in step 3 is never folded away,
           however few cells it holds: the tree needs it to have edges.

        Args:
            cell_type_counts: Dict mapping CL_ID -> cell count.
            min_group_size: Minimum active descendants for a valid group root.
            max_group_size: Maximum active descendants for a valid group root.
            cell_embeddings: ``(n_cells, dim)`` cell embeddings, used to pick
                which branch a split type's cells belong to.  When absent or
                shaped incompatibly with the node vectors, one stand-in cell per
                type is used: still deterministic, but the branch is arbitrary.
            prediction_indices: Per-cell predicted full-ontology node index,
                aligned row-for-row with *cell_embeddings*.
            min_cells_number: Two jobs.  Branch pruning threshold for the
                splitter (ignored when stand-in cells are used), and the cell
                count a group must reach in step 5 to survive as a node.

        Returns:
            Dict with:

            - 'simplification_map', 'simplified_counts', 'similarity_matrix'
              (None).
            - 'tree_nodes' — groups plus the ancestors joining them.  Build the
              simplified tree from this, not from the map's values.
            - 'connector_nodes' — the ancestors kept in step 3.  Report these
              rather than working them out from the counts: step 5 can move
              cells onto one, and then it no longer looks like an empty node.
            - 'folded_nodes' — {group that step 5 removed: {'into': the node its
              cells went to, 'cells': how many}}.
            - 'rare_rollup_done' — True, meaning step 5 has already applied
              *min_cells_number*, so the caller must not sweep again.
        """
        from collections import Counter

        active_indices: Set[int] = set()
        for cl_id in cell_type_counts:
            if cl_id in self.name_to_idx:
                active_indices.add(self.name_to_idx[cl_id])

        def _each_to_itself() -> Dict[str, Any]:
            """Degenerate fallback: no grouping is possible, keep every label."""
            return {
                "simplification_map": {cl: cl for cl in cell_type_counts},
                "simplified_counts": dict(cell_type_counts),
                "similarity_matrix": None,
                "tree_nodes": set(cell_type_counts),
            }

        if not active_indices:
            return _each_to_itself()

        # --- 1. split shared types onto each branch -------------------------
        subgraph = AdaptiveOntologySubgraph(
            self.pure_ontology_adj, self.node_names, None
        )
        relevant = list(subgraph.get_relevant_node_indices(
            np.array(sorted(active_indices), dtype=np.int64),
            include_ancestors=True,
            include_siblings=False,
        ))
        if not relevant:
            return _each_to_itself()

        node_dim = (
            self.node_vectors.shape[1] if self.node_vectors.ndim == 2 else 0
        )
        preds: Optional[np.ndarray] = None
        vectors: Optional[np.ndarray] = None
        prune = max(int(min_cells_number), 1)
        if cell_embeddings is not None and prediction_indices is not None:
            p_arr = np.asarray(prediction_indices, dtype=np.int64)
            v_arr = np.asarray(cell_embeddings)
            if (
                v_arr.ndim == 2
                and v_arr.shape[0] == p_arr.shape[0]
                and v_arr.shape[1] == node_dim
            ):
                usable = p_arr >= 0
                preds, vectors = p_arr[usable], v_arr[usable]
        if preds is None or preds.size == 0:
            preds = np.array(sorted(active_indices), dtype=np.int64)
            vectors = self.node_vectors[preds]
            prune = 1

        tree = DAGToTreeConverter(
            self.pure_ontology_adj, self.node_names, self.node_vectors
        ).convert(relevant, preds, vectors, min_cells_number=prune)

        orig = np.asarray(tree["original_indices"], dtype=np.int64)
        tree_preds = np.asarray(tree["tree_predictions"], dtype=np.int64)
        if orig.size == 0 or tree_preds.size == 0:
            return _each_to_itself()
        keep_mask = np.asarray(
            tree.get("cell_keep_mask", np.ones(tree_preds.size, dtype=bool)),
            dtype=bool,
        )
        cell_orig = preds[keep_mask] if keep_mask.size == preds.size else preds
        if cell_orig.size != tree_preds.size:
            return _each_to_itself()

        # tree_adj[child, parent] = 1 (see DAGToTreeConverter.convert)
        parent_of: Dict[int, Optional[int]] = {}
        rows, cols = np.where(tree["tree_adj"] > 0)
        for child, parent in zip(rows, cols):
            parent_of[int(child)] = int(parent)
        for node in range(len(orig)):
            parent_of.setdefault(node, None)

        tree_graph = nx.DiGraph()
        tree_graph.add_nodes_from(range(len(orig)))
        for child, parent in parent_of.items():
            if parent is not None:
                tree_graph.add_edge(child, parent)

        # --- 2. group roots on the split tree ------------------------------
        active_copies = {int(t) for t in np.unique(tree_preds) if t >= 0}
        groups = _find_group_roots(
            tree_graph, active_copies,
            min_size=min_group_size, max_size=max_group_size,
            attach_leftovers=False,
        )
        surviving: Set[int] = set()
        for root_idx, members in groups:
            if root_idx == -1:
                surviving.update(int(m) for m in members)
            else:
                surviving.add(int(root_idx))
        if not surviving:
            return _each_to_itself()

        # --- 3. keep the ancestors that join two or more groups ------------
        below_count: Dict[int, int] = defaultdict(int)
        for node in surviving:
            seen: Set[int] = set()
            cur = parent_of.get(node)
            while cur is not None and cur not in seen:
                seen.add(cur)
                below_count[cur] += 1
                cur = parent_of.get(cur)
        connectors = {
            node for node, count in below_count.items()
            if node not in surviving and (count >= 2 or parent_of.get(node) is None)
        }
        surviving_all = surviving | connectors

        # --- 4. every type to the nearest surviving ancestor ---------------
        nearest_cache: Dict[int, Optional[int]] = {}

        def _nearest(node: int) -> Optional[int]:
            """Nearest node at or above *node* that survived, or None."""
            walked: List[int] = []
            cur: Optional[int] = node
            found: Optional[int] = None
            while cur is not None:
                if cur in surviving_all:
                    found = cur
                    break
                if cur in nearest_cache:
                    found = nearest_cache[cur]
                    break
                walked.append(cur)
                cur = parent_of.get(cur)
            for step in walked:
                nearest_cache[step] = found
            return found

        votes: Dict[int, 'Counter[int]'] = defaultdict(Counter)
        for source_idx, copy_idx in zip(cell_orig.tolist(), tree_preds.tolist()):
            target = _nearest(int(copy_idx))
            if target is not None:
                votes[int(source_idx)][target] += 1

        # --- 5. fold the groups too small to stand on their own ------------
        threshold = max(int(min_cells_number), 1)
        connector_names = {
            self.node_names[int(orig[node])] for node in connectors
        }
        copies_of: Dict[str, List[int]] = defaultdict(list)
        for node in surviving_all:
            copies_of[self.node_names[int(orig[node])]].append(node)

        def _build(kept: Set[int]) -> Tuple[
            Dict[str, str], Dict[str, int], Set[str], Dict[int, int]
        ]:
            """Map every type onto *kept*, and report the counts that follow.

            Returns the map, the per-node counts, the names of *kept*, and where
            each node of the split tree folded to.
            """
            fold_cache: Dict[int, int] = {}

            def _fold_target(node: int) -> int:
                """Nearest node at or above *node* in *kept*, else *node*."""
                walked: List[int] = []
                seen: Set[int] = set()
                cur: Optional[int] = node
                found: Optional[int] = None
                while cur is not None and cur not in seen:
                    if cur in kept:
                        found = cur
                        break
                    if cur in fold_cache:
                        found = fold_cache[cur]
                        break
                    seen.add(cur)
                    walked.append(cur)
                    cur = parent_of.get(cur)
                if found is None:
                    # Nothing above it survives — leave the group where it is
                    # rather than invent a home for it.
                    return node
                for step in walked:
                    fold_cache[step] = found
                return found

            names = {self.node_names[int(orig[node])] for node in kept}
            smap: Dict[str, str] = {}
            scounts: Dict[str, int] = {}
            for cl_id, count in cell_type_counts.items():
                idx = self.name_to_idx.get(cl_id)
                target_name: Optional[str] = None
                if cl_id in names:
                    # A type that survives as a node of its own keeps its own
                    # name. The copies exist to decide which branch a route runs
                    # along, and must not move a type into a different node.
                    target_name = cl_id
                elif idx is not None and votes.get(idx):
                    # Fold first, then count: a type whose cells span branches
                    # goes to whichever kept node most of them end up under.
                    folded_votes: 'Counter[int]' = Counter()
                    for node, n in votes[idx].items():
                        folded_votes[_fold_target(int(node))] += int(n)
                    best_copy = folded_votes.most_common(1)[0][0]
                    target_name = self.node_names[int(orig[best_copy])]
                if target_name is None:
                    # This type's cells were pruned out of the split tree; walk
                    # the original ontology up to the nearest surviving name.
                    target_name = _bfs_to_surviving_ancestor(
                        cl_id, names, self.nx_graph, self.name_to_idx
                    ) or cl_id
                smap[cl_id] = target_name
                scounts[target_name] = scounts.get(target_name, 0) + count
            return smap, scounts, names, _fold_target

        # Judge a group on the count it will be reported with, which means
        # building the map first: a type's cells all follow the branch most of
        # them landed on, so a node can clear the threshold on the cells placed
        # against it and still come in under it once whole types have moved.
        # Dropping a group can move a type off a second group, so repeat until
        # the set settles. Dropped names are never reinstated, so it terminates.
        dropped: Set[str] = set()
        dropped_cells: Dict[str, int] = {}
        for _ in range(10):
            kept = {
                node for node in surviving_all
                # An ancestor kept in step 3 stays whatever it holds: drop it
                # and the groups it joins fall apart into siblings again.
                if node in connectors
                or self.node_names[int(orig[node])] not in dropped
            }
            simplification_map, simplified_counts, kept_names, fold_target = (
                _build(kept)
            )
            small = {
                name for name, cnt in simplified_counts.items()
                if cnt < threshold
                and name in kept_names
                and name not in connector_names
            } - dropped
            if not small:
                break
            for name in small:
                dropped_cells[name] = simplified_counts[name]
            dropped |= small

        folded_nodes: Dict[str, Dict[str, Any]] = {}
        for name in sorted(dropped):
            into = {
                self.node_names[int(orig[fold_target(node)])]
                for node in copies_of.get(name, [])
            } - {name}
            if not into:
                continue
            folded_nodes[name] = {
                "into": sorted(into)[0],
                "cells": int(dropped_cells.get(name, 0)),
            }

        return {
            "simplification_map": simplification_map,
            "simplified_counts": simplified_counts,
            "similarity_matrix": None,
            "tree_nodes": kept_names | set(simplification_map.values()),
            "connector_nodes": connector_names,
            "folded_nodes": folded_nodes,
            "rare_rollup_done": True,
        }

    def _level_rollup(
        self,
        cell_type_counts: Dict[str, int],
        target_level: int,
    ) -> Dict[str, Any]:
        """Perform level-based roll-up of cell types to a target ontology depth.

        For each active cell type:
        - If its depth <= target_level: maps to itself.
        - If its depth > target_level: walks parent edges upward via BFS to find
          the ancestor at the target depth. When multiple ancestors exist at the
          target level (DAG), selects the one reachable via the shortest path
          from the cell type (BFS guarantees this).
        - If no ancestor at exactly target_level exists, selects the deepest
          ancestor at depth < target_level.
        - If the cell type is not in the ontology graph (orphan): skips it
          with a warning.

        Args:
            cell_type_counts: Dict mapping CL_ID -> cell count (already filtered).
            target_level: Non-negative integer target depth.

        Returns:
            Dict with keys:
                - 'simplification_map': Dict[str, str]
                - 'simplified_counts': Dict[str, int]
                - 'similarity_matrix': None
        """
        import logging
        from collections import deque
        from .trajectory_ontology import compute_node_depths

        logger = logging.getLogger(__name__)

        # Compute shortest-path depths for all nodes.
        # pure_ontology_adj has adj[child, parent] = 1 (child->parent edges).
        # compute_node_depths expects adj[parent, child] = 1 (parent->child edges).
        # Transposing flips the convention so roots (no parents) have in-degree 0.
        depths = compute_node_depths(self.pure_ontology_adj.T)

        simplification_map: Dict[str, str] = {}
        simplified_counts: Dict[str, int] = {}

        for cl_id, count in cell_type_counts.items():
            idx = self.name_to_idx.get(cl_id)
            if idx is None or idx not in depths:
                logger.warning(
                    "Skipping CL_ID '%s' not found in ontology graph during level roll-up",
                    cl_id,
                )
                continue

            node_depth = depths[idx]

            if node_depth <= target_level:
                # Already at or above target level — map to itself
                ancestor_cl_id = cl_id
            else:
                # BFS upward through parent edges to find ancestor at target_level.
                # nx_graph has child->parent edges, so successors(node) = parents.
                # BFS level-by-level; first node found at target_level is the
                # shortest-path ancestor (BFS guarantees this in a DAG).
                best_ancestor_idx: Optional[int] = None
                best_ancestor_depth: int = -1

                visited: set = {idx}
                queue: deque = deque([idx])

                while queue:
                    current = queue.popleft()
                    current_depth = depths.get(current, 0)

                    if current != idx:  # Don't count the starting node itself
                        if current_depth == target_level:
                            # Found exact target level — use this ancestor
                            best_ancestor_idx = current
                            break
                        elif current_depth < target_level:
                            # Above target level — track deepest so far as fallback
                            if current_depth > best_ancestor_depth:
                                best_ancestor_depth = current_depth
                                best_ancestor_idx = current
                            # Don't traverse further up from here
                            continue

                    # Traverse parents (successors in child->parent graph)
                    for parent in self.nx_graph.successors(current):
                        if parent not in visited:
                            visited.add(parent)
                            queue.append(parent)

                if best_ancestor_idx is None:
                    # No ancestor found — map to itself as fallback
                    logger.warning(
                        "No ancestor found at or above target_level=%d for CL_ID '%s'; "
                        "mapping to itself.",
                        target_level,
                        cl_id,
                    )
                    ancestor_cl_id = cl_id
                else:
                    ancestor_cl_id = self.node_names[best_ancestor_idx]

            simplification_map[cl_id] = ancestor_cl_id
            simplified_counts[ancestor_cl_id] = (
                simplified_counts.get(ancestor_cl_id, 0) + count
            )

        return {
            "simplification_map": simplification_map,
            "simplified_counts": simplified_counts,
            "similarity_matrix": None,
        }


def _rollup_rare_classes(
    simplification_map: Dict[str, str],
    simplified_counts: Dict[str, int],
    nx_graph: 'nx.DiGraph',
    node_names: List[str],
    min_cells_number: int,
) -> Tuple[Dict[str, str], Dict[str, int], Set[str], Dict[str, Dict[str, Any]]]:
    """Roll up post-merge classes below *min_cells_number* into surviving ancestors.

    For each simplified entry with count < ``min_cells_number``:

    1. BFS upward through ``nx_graph`` (child->parent edges via
       ``successors()``) looking for the nearest ancestor that the simplified
       tree keeps — a class over the threshold, or one of the ancestors joining
       two of them.
    2. If no such ancestor exists, leave the class unchanged. Similarity alone
       is not a valid fallback because it does not require an ancestral
       relationship and can conceal an off-lineage prediction.

    After each small class is assigned a target, all ``simplification_map``
    entries that pointed to the small class are rewritten to point to the
    target, and ``simplified_counts`` is updated.

    Only ``"wang"`` and ``"level"`` reach this.  ``"lineage"`` applies
    *min_cells_number* inside the grouping, on the tree it built, where the
    ancestors joining two groups are available to fold into.

    Args:
        simplification_map: Original CL ID -> simplified CL ID mapping
            from OntologySimplifier.simplify().
        simplified_counts: Simplified CL ID -> aggregated cell count.
        nx_graph: NetworkX DiGraph with child->parent edges.
            ``successors(node)`` returns parents.
        node_names: List of all CL IDs (index -> CL ID) for the full ontology.
        min_cells_number: Threshold below which a class is rolled up.

    Returns:
        Tuple of (updated map, updated counts, the ancestors kept to join two
        classes, and {class removed: {'into': where its cells went, 'cells':
        how many}}).
    """
    # Build name -> index lookup for the nx_graph
    name_to_idx: Dict[str, int] = {
        name: i for i, name in enumerate(node_names)
    }

    # Identify surviving set (classes >= threshold)
    surviving: Set[str] = {
        cl for cl, cnt in simplified_counts.items()
        if cnt >= min_cells_number
    }

    if not surviving:
        # Nothing to merge into — return unchanged
        return dict(simplification_map), dict(simplified_counts), set(), {}

    # The ancestors joining the surviving classes. Without them the only
    # targets are the surviving classes themselves, and a class sitting on its
    # own branch — the off-tissue mispredictions, one cell each — has none of
    # them above it and stays on the figure at whatever size it has.
    joining = _joining_ancestors(surviving, nx_graph, name_to_idx)

    # Identify small classes to roll up
    small_classes = [
        cl for cl, cnt in simplified_counts.items()
        if cnt < min_cells_number and cl not in joining
    ]

    if not small_classes:
        return dict(simplification_map), dict(simplified_counts), joining, {}

    # Map each small class to its rollup target
    targets = surviving | joining
    rollup_targets: Dict[str, str] = {}

    for small_cl in small_classes:
        target = _bfs_to_surviving_ancestor(
            small_cl, targets, nx_graph, name_to_idx,
        )

        if target is not None:
            rollup_targets[small_cl] = target

    # Apply rollup: update simplification_map and simplified_counts
    new_map = dict(simplification_map)
    new_counts = dict(simplified_counts)
    folded_nodes: Dict[str, Dict[str, Any]] = {}

    for small_cl, target_cl in rollup_targets.items():
        # Aggregate count into target
        moved = new_counts.pop(small_cl, 0)
        new_counts[target_cl] = new_counts.get(target_cl, 0) + moved
        folded_nodes[small_cl] = {"into": target_cl, "cells": int(moved)}

        # Rewrite all map entries that pointed to small_cl
        for orig_cl, simp_cl in new_map.items():
            if simp_cl == small_cl:
                new_map[orig_cl] = target_cl

    return new_map, new_counts, joining, folded_nodes


def _joining_ancestors(
    node_ids: Set[str],
    nx_graph: 'nx.DiGraph',
    name_to_idx: Dict[str, int],
) -> Set[str]:
    """Ancestors the tree needs to join two or more of *node_ids*.

    Kept: an ancestor with two or more of its children leading to one of
    *node_ids* — a branch point — and any ancestor with no parents of its own.
    An ancestor with a single such child is passed over, because the tree links
    each node to its nearest ancestor in the set and would only draw it as a
    step in a chain. That matches what ``plot_simplified_tree`` keeps, so the
    picture and the cells are measured against the same nodes.

    Args:
        node_ids: CL IDs of the nodes the simplified tree already has.
        nx_graph: NetworkX DiGraph with child->parent edges, so
            ``successors(node)`` returns parents and ``predecessors(node)``
            returns children.
        name_to_idx: CL ID -> integer index in *nx_graph*.

    Returns:
        Set of CL IDs, disjoint from *node_ids*.
    """
    from collections import deque

    seeds = {
        name_to_idx[name] for name in node_ids
        if name in name_to_idx and name_to_idx[name] in nx_graph
    }
    if not seeds:
        return set()

    # Every node on a path from a seed up to a root.
    above: Set[int] = set()
    queue = deque(seeds)
    while queue:
        cur = queue.popleft()
        for parent in nx_graph.successors(cur):
            if parent not in above:
                above.add(parent)
                queue.append(parent)

    supports = above | seeds
    joining: Set[str] = set()
    for node in above:
        if node in seeds:
            continue
        leading = sum(
            1 for child in nx_graph.predecessors(node) if child in supports
        )
        if leading >= 2 or not any(True for _ in nx_graph.successors(node)):
            joining.add(nx_graph.nodes[node].get('name', ''))

    return {name for name in joining if name and name not in node_ids}


def _bfs_to_surviving_ancestor(
    cl_id: str,
    surviving: Set[str],
    nx_graph: 'nx.DiGraph',
    name_to_idx: Dict[str, int],
) -> Optional[str]:
    """BFS upward from *cl_id* to find the nearest ancestor in *surviving*.

    Returns the CL ID of the nearest surviving ancestor, or None if none
    exists (orphaned lineage).
    """
    from collections import deque

    idx = name_to_idx.get(cl_id)
    if idx is None:
        return None

    visited: Set[int] = {idx}
    queue = deque([idx])

    while queue:
        cur = queue.popleft()
        for parent in nx_graph.successors(cur):
            if parent in visited:
                continue
            visited.add(parent)
            parent_name = nx_graph.nodes[parent].get('name', '')
            if parent_name in surviving:
                return parent_name
            queue.append(parent)

    return None


def _compute_longest_path_depths(parent_child_adj: np.ndarray) -> Dict[int, int]:
    """Longest-path depth from any root for every node in a parent->child DAG.

    Unlike shortest-path BFS, longest-path depth guarantees ``d(parent) <
    d(child)`` along every direct edge in the DAG, including in polyhierarchies
    where shortest-path can place a parent at greater depth than its child
    via a separate shortcut. The simplified-tree trajectory pass needs that
    monotonicity for both placer direction inference (``node_positions[u] <
    node_positions[v]`` decides edge orientation) and depth-space projection
    of a cell's full-tree raw_t onto a simplified edge.

    Self-loops are ignored. If the graph still contains a cycle after removing
    self-loops, the function falls back to shortest-path BFS.

    Args:
        parent_child_adj: ``(n, n)`` array where ``[i, j] = 1`` means node
            ``i`` is a parent of node ``j``. Roots are nodes whose column
            sum is 0.

    Returns:
        Dict mapping node index → length of the longest path from any root.
        Disconnected nodes are assigned depth 0.
    """
    n_nodes = int(parent_child_adj.shape[0])
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    rows, cols = np.where(parent_child_adj > 0)
    for parent, child in zip(rows.tolist(), cols.tolist()):
        if parent == child:
            continue  # skip self-loops; they do not add depth
        g.add_edge(int(parent), int(child))

    if not nx.is_directed_acyclic_graph(g):
        # Synthetic test fixtures occasionally feed in cyclic adjacencies;
        # real ontology data is acyclic. Fall back to shortest-path so we
        # never raise — the polyhierarchy bug only matters on DAG data.
        from .trajectory_ontology import compute_node_depths
        return compute_node_depths(parent_child_adj)

    depths: Dict[int, int] = {i: 0 for i in range(n_nodes)}
    for node in nx.topological_sort(g):
        for child in g.successors(node):
            cand = depths[node] + 1
            if cand > depths[child]:
                depths[child] = cand
    return depths


def _walk_up_to_simplified_label(
    full_name: str,
    simplification_map: Dict[str, str],
    simplified_node_set: Set[str],
    child_parent_adj: np.ndarray,
    name_to_idx: Dict[str, int],
    full_classes: List[str],
) -> str:
    """Map a full-ontology node name to its label in the simplified tree.

    Resolution order:

    1. If ``full_name`` is a key in ``simplification_map`` → return the mapped
       value. (Typical case: predicted cell type collapsed into a simplified
       node by the simplification algorithm.)
    2. Else if ``full_name`` is itself in ``simplified_node_set`` → return it
       unchanged. (Surviving node, possibly a non-predicted ancestor.)
    3. Else BFS upward through ``child_parent_adj`` (rows = child, columns =
       parents) until finding a node whose name is in ``simplified_node_set``.
       Return that ancestor's name.
    4. If no ancestor in ``simplified_node_set`` exists, or ``full_name`` is
       not in ``name_to_idx``, return ``full_name`` unchanged as a last-resort
       fallback.

    Empty-string input is preserved (callers use ``''`` as a sentinel for
    "no edge assigned" / "no target").
    """
    if full_name == '':
        return ''
    if full_name in simplification_map:
        return simplification_map[full_name]
    if full_name in simplified_node_set:
        return full_name

    idx = name_to_idx.get(full_name)
    if idx is None:
        return full_name

    from collections import deque
    visited: Set[int] = {idx}
    queue = deque([idx])
    while queue:
        cur = queue.popleft()
        parents = np.where(child_parent_adj[cur] > 0)[0]
        for p in parents.tolist():
            p = int(p)
            if p in visited:
                continue
            visited.add(p)
            p_name = full_classes[p]
            if p_name in simplified_node_set:
                return p_name
            queue.append(p)

    return full_name


def _build_simplified_tree(
    parent_child_adj: np.ndarray,
    node_names: List[str],
    simplified_node_ids: Set[str],
    node_depths: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """Build a proper tree from a subset of ontology nodes.

    Given the full ontology adjacency (parent→child convention) and a set of
    simplified node IDs, constructs a tree where each simplified node's parent
    is its nearest ancestor that is also in the simplified set.

    Args:
        parent_child_adj: ``(n, n)`` array where ``[i, j] = 1`` means
            node ``i`` is a parent of node ``j``.
        node_names: Full list of ontology node ID strings, indexed by
            the adjacency matrix rows/columns.
        simplified_node_ids: Set of node ID strings that survived
            simplification.
        node_depths: Optional pre-computed Dict[int, int] of node depths
            keyed by index, used to populate the returned ``'depths'``.
            When ``None``, longest-path depths over the full ontology are
            computed on demand. Callers that mix this dict with their own
            depth lookups MUST pass the same dict here to guarantee that
            the two agree on every node (the trajectory-scoring stage in
            ``simplify_ontology_tree`` does exactly that).

    Returns:
        Dict with:
            - ``'depths'``: Dict[str, int] — node ID → depth (longest path
              from any root in the full ontology, or whatever convention
              ``node_depths`` uses if supplied).
            - ``'parent_map'``: Dict[str, str] — node ID → parent node
              ID in the simplified set (root excluded).
            - ``'edges'``: List[Tuple[str, str]] — ``(parent_id,
              child_id)`` edges.
            - ``'node_indices'``: Dict[str, int] — node ID → index in
              ``node_names``.
    """
    from collections import deque

    name_to_idx = {name: i for i, name in enumerate(node_names)}

    if node_depths is None:
        node_depths = _compute_longest_path_depths(parent_child_adj)
    full_depths = node_depths

    child_parent_adj = parent_child_adj.T

    simplified_indices = {name_to_idx[nid] for nid in simplified_node_ids if nid in name_to_idx}

    parent_map: Dict[str, str] = {}
    depths: Dict[str, int] = {}

    for nid in simplified_node_ids:
        idx = name_to_idx.get(nid)
        if idx is None:
            continue
        depths[nid] = full_depths.get(idx, 0)

        visited = {idx}
        queue = deque([idx])
        found_parent = None

        while queue:
            current = queue.popleft()
            parents = np.where(child_parent_adj[current] > 0)[0]
            for p in parents:
                if p in visited:
                    continue
                visited.add(p)
                if p in simplified_indices and p != idx:
                    found_parent = p
                    break
                queue.append(p)
            if found_parent is not None:
                break

        if found_parent is not None:
            parent_map[nid] = node_names[found_parent]

    edges = [(parent_id, child_id) for child_id, parent_id in parent_map.items()]

    node_indices = {nid: name_to_idx[nid] for nid in simplified_node_ids if nid in name_to_idx}

    return {
        'depths': depths,
        'parent_map': parent_map,
        'edges': edges,
        'node_indices': node_indices,
    }



import logging as _logging
_simplify_logger = _logging.getLogger(__name__)

def simplify_ontology_tree(
    predictor: 'HECTOR',
    adata: 'anndata.AnnData',
    min_cells_number: int = 20,
    similarity_threshold: float = 0.85,
    algorithm: str = "lineage",
    target_level: Optional[int] = None,
    label_format: Optional[str] = None,
    min_group_size: int = 2,
    max_group_size: int = 8,
    cache: Optional[VisualizationCache] = None,
    compute_transitions: bool = True,
) -> None:
    """Orchestrate the full ontology simplification pipeline.

    Reads cell type predictions from the public annotation manifest, or from
    ``adata.obs['hector_prediction']`` when no compatible manifest is present,
    normalises CL IDs, filters rare cell types, and groups them using one of
    three algorithms.

    Args:
        predictor: Loaded ``HECTOR`` instance with ontology data.
        adata: AnnData object with reusable HECTOR annotation.
        min_cells_number: Minimum cell count per type to include (default 20).
        similarity_threshold: Clustering cutoff in (0.0, 1.0) (default 0.85).
            Only used by ``"wang"``; ignored by other algorithms.
        algorithm: ``"lineage"`` (default), ``"wang"``, or ``"level"``.
            ``"lineage"`` groups cell types by natural subtree structure,
            bounded by ``min_group_size`` / ``max_group_size``. A cell type
            with several parents is duplicated onto each branch first, so no
            branch loses a type another branch claimed.
            ``"wang"`` uses Wang Semantic Similarity clustering.
            ``"level"`` rolls up to a target ontology depth.
        target_level: Target depth for level-based roll-up. Required when
            ``algorithm="level"``; depth 0 is the root, depth 1 is the root's
            direct children, etc. Ignored for other algorithms.
        label_format: Output format for the simplification map. ``None``
            (default) reuses the annotated prediction column's label format
            when available; otherwise uses raw ontology IDs. Supported
            values are ``"id"``, ``"name"``, and ``"both"``.
        min_group_size: Minimum members per lineage group (default 2).
            Only used when ``algorithm="lineage"``.
        max_group_size: Maximum members per lineage group (default 8).
            Only used when ``algorithm="lineage"``. "Members" are cell types,
            not cells, counted after types with several parents have been
            duplicated onto each branch — so the same node covers fewer types
            than the raw ontology would suggest, and this number is smaller
            than it looks. Lower it for a finer tree with more routes, raise it
            for fewer, broader groups.
        cache: Optional VisualizationCache for reusing precomputed ontology
            distances and embedding similarity. When None, automatically
            retrieved from ``adata._hector_viz_cache`` if available (set by
            ``create_trajectory_analysis``), otherwise computed fresh.
        compute_transitions: If ``True`` (default), also work out for each cell
            whether it sits squarely on one of the simplified groups
            (``"Stable"``) or part of the way between two of them
            (``"Transitioning"``), and write the six columns that record it,
            listed under Returns.

            Every cell is re-measured against the simplified tree. This stage
            scales approximately linearly with the number of cells. Existing
            full-tree positions cannot be reused because simplification changes
            the groups at the ends of each branch.

            If ``False``, skip it: ``simplified_prediction`` becomes the only
            column written and the call is around eighty times faster. Grouping
            and ``adata.uns['hector_simplified']`` are computed either way.

    Returns:
        ``None``. Writes the following to ``adata``:

        ``adata.uns['hector_simplified']`` — dict with three keys always, and
        two more when they have anything to say:

        * ``'simplification_map'`` (``Dict[str, str]``): maps each original
          label to its simplified label in the requested output format.
        * ``'simplified_counts'`` (``Dict[str, int]``): maps each simplified
          label to the number of cells assigned to it.
        * ``'label_format'`` (``str``): the output label format used
          (``"id"``, ``"name"``, or ``"both"``).
        * ``'connector_nodes'`` (``List[str]``): the ancestors kept to join two
          groups, which the tree needs to have edges at all.
        * ``'folded_nodes'`` (``Dict[str, Dict[str, Any]]``): each group that
          held fewer than *min_cells_number* cells and was removed, mapped to
          ``{'into': the label its cells now carry, 'cells': how many}``.
          ``"lineage"`` only.

        ``adata.obs`` columns written. ``simplified_prediction`` is always
        assigned. The remaining columns are assigned when
        ``compute_transitions=True``; disabling computation leaves any
        pre-existing transition columns unchanged.

        * ``simplified_prediction``: simplified cell type label per cell.
        * ``simplified_trajectory_state``: ``"Stable"`` or
          ``"Transitioning"`` for each cell.
        * ``simplified_relative_position``: the full-tree
          ``relative_position`` projected onto the simplified edge.
        * ``simplified_transition_position``: the transitioning band of
          that, stretched back out to ``[0, 1]`` (``0`` = stable).

        There is no ``simplified_absolute_position``: the projection
        divides by how many ontology levels the simplified edge spans,
        which depends on which cell types this dataset contains, so the
        result would lose the cross-dataset comparability that is the
        whole reason ``absolute_position`` exists. Use the full-tree
        column for that.
        * ``simplified_trajectory_source``: source (parent) node label
          for the assigned edge.
        * ``simplified_trajectory_target``: target (child) node label
          for the assigned edge.
        * ``simplified_global_score``: continuous depth score combining
          source depth and intra-edge position.

    Raises:
        ValueError: On missing column, invalid parameters, or all cell types
            filtered out.
    """
    from . import trajectory_support as _ts
    from .trajectory_support import LatentBarycentricPlacer
    # Access through the module so callers can replace the detector if needed.

    if cache is None:
        cache = getattr(adata, '_hector_viz_cache', None)

    # Input validation and preprocessing
    annotation_ref = predictor._resolve_annotation_reference(
        adata,
        require_score=False,
        allow_fallback=True,
    )
    if not predictor._annotation_reference_matches(
        adata,
        annotation_ref,
        expected_fingerprint=None,
        require_score=False,
    ):
        raise ValueError(
            "No usable HECTOR prediction annotation found. "
            "Run predictor.predict(...), then predictor.write_predictions(...), first."
        )
    predictor._backfill_annotation_manifest_if_possible(adata, annotation_ref)
    prediction_col = annotation_ref['prediction_col']
    annotation_label_format = annotation_ref.get("label_format")
    valid_label_formats = {"id", "name", "both"}
    if annotation_label_format not in valid_label_formats:
        annotation_label_format = None

    if label_format is None:
        output_label_format = annotation_label_format or "id"
    else:
        if label_format not in valid_label_formats:
            raise ValueError(
                f"Invalid label_format '{label_format}'. Must be one of: {sorted(valid_label_formats)}"
            )
        output_label_format = label_format

    # 2. Validate similarity_threshold
    if not (0.0 < similarity_threshold < 1.0):
        raise ValueError(
            f"similarity_threshold must be in (0.0, 1.0), got {similarity_threshold}"
        )

    # 3. Validate algorithm
    if algorithm == "shortest_path":
        raise ValueError(
            "algorithm='shortest_path' has been removed. "
            "Use algorithm='lineage' instead."
        )
    if algorithm not in ("lineage", "wang", "level"):
        raise ValueError(
            f"algorithm must be 'lineage', 'wang', or 'level', got '{algorithm}'"
        )

    # 3b. Validate target_level when algorithm="level"
    if algorithm == "level":
        if target_level is None:
            raise ValueError(
                "algorithm='level' requires target_level to be a non-negative integer, got None"
            )
        if not isinstance(target_level, (int, np.integer)) or isinstance(target_level, bool) or target_level < 0:
            raise ValueError(
                f"target_level must be a non-negative integer, got {target_level!r}"
            )

    # 4. Read and normalise predictions
    cell_type_counts: Dict[str, int] = {}
    raw_prediction_ids: Optional[np.ndarray] = None

    # Recover canonical IDs from the cached top-1 prediction core when
    # available so name-only public annotations do not need reverse lookup.
    if (
        hasattr(predictor, "_get_prediction_core_manifest")
        and hasattr(predictor, "_prediction_core_manifest_matches")
        and hasattr(predictor, "_get_active_class_ids")
    ):
        core_manifest = predictor._get_prediction_core_manifest(adata)
        if predictor._prediction_core_manifest_matches(
            adata,
            core_manifest,
            expected_prediction_fingerprint=annotation_ref.get("prediction_fingerprint"),
        ):
            use_full_ontology = True
            prediction_fingerprint = annotation_ref.get("prediction_fingerprint")
            if isinstance(prediction_fingerprint, dict):
                use_full_ontology = bool(
                    prediction_fingerprint.get("use_full_ontology", True)
                )

            active_class_ids = np.asarray(
                predictor._get_active_class_ids(use_full_ontology),
                dtype=object,
            )
            top_indices = np.asarray(core_manifest.get("top_indices"), dtype=np.int64)
            if top_indices.shape == (adata.shape[0], 1):
                flat_top_indices = top_indices[:, 0]
                if (
                    flat_top_indices.size == 0
                    or (
                        np.all(flat_top_indices >= 0)
                        and np.all(flat_top_indices < len(active_class_ids))
                    )
                ):
                    raw_prediction_ids = active_class_ids[flat_top_indices]

    if raw_prediction_ids is None:
        raw_predictions = adata.obs[prediction_col].values
        for raw_val in raw_predictions:
            cl_id = predictor._parse_ontology_id(str(raw_val))
            if cl_id is None:
                _simplify_logger.warning(
                    "Skipping unparseable prediction value: '%s'", raw_val
                )
                continue
            cell_type_counts[cl_id] = cell_type_counts.get(cl_id, 0) + 1
    else:
        for cl_id in raw_prediction_ids.tolist():
            cell_type_counts[cl_id] = cell_type_counts.get(cl_id, 0) + 1

    # 5. Guard: no cell types at all (empty input)
    if not cell_type_counts:
        raise ValueError(
            "No cell types found in predictions. "
            "Check your prediction column or AnnData object."
        )

    # 7. Warn if fewer than 2 cell types remain
    n_remaining = len(cell_type_counts)
    if n_remaining < 2:
        _simplify_logger.warning(
            "Only %d cell type(s) after filtering — simplification will have no effect.",
            n_remaining,
        )

    # Wire OntologySimplifier

    # Extract ontology data from predictor
    full_pure_ontology_adj = predictor.full_pure_ontology_adj
    full_classes: List[str] = list(predictor.celltype_gat.full_classes)
    id_to_name_map: Dict[str, str] = getattr(predictor, 'id_to_name_map', {})

    # Obtain node vectors (GAT embeddings) — try multiple attribute paths
    node_vectors: Optional[np.ndarray] = None
    if hasattr(predictor, 'celltype_gat') and hasattr(predictor.celltype_gat, 'node_embeddings'):
        node_vectors = predictor.celltype_gat.node_embeddings
    if node_vectors is None and hasattr(predictor, 'node_vectors'):
        node_vectors = predictor.node_vectors
    if node_vectors is None:
        # Fallback: zero embeddings (layout will still work via ontology structure)
        n_nodes = len(full_classes)
        node_vectors = np.zeros((n_nodes, 1), dtype=np.float32)

    # Ensure node_vectors is a numpy array
    if not isinstance(node_vectors, np.ndarray):
        node_vectors = np.array(node_vectors)

    # celltype_gat.node_embeddings is often absent on a loaded predictor, leaving
    # node_vectors as the (n_nodes, 1) zero fallback — unusable for anything
    # comparing nodes against cells, including the lineage grouping's branch
    # assignment. Fetch the real GAT prototypes once, for both users below.
    cell_embeddings_for_grouping = adata.obsm.get('X_hector')
    if cell_embeddings_for_grouping is not None:
        _cell_dim = int(np.asarray(cell_embeddings_for_grouping).shape[1])
        if node_vectors.shape[1] != _cell_dim and hasattr(
            predictor, '_build_visualization_state'
        ):
            try:
                node_vectors = np.asarray(
                    predictor._build_visualization_state(
                        use_full_ontology=True
                    )['node_vectors']
                )
            except Exception as exc:  # pragma: no cover — defensive
                _simplify_logger.warning(
                    "Could not fetch GAT node embeddings (%s); the lineage "
                    "grouping will place shared cell types on an arbitrary "
                    "branch.",
                    exc,
                )

    # Per-cell predicted full-ontology index, aligned with adata rows. The
    # lineage grouping needs it to work out which branch each cell's type sits
    # on once shared types have been split.
    _grouping_class_to_idx = {name: i for i, name in enumerate(full_classes)}
    if raw_prediction_ids is not None:
        prediction_indices_for_grouping = np.array([
            _grouping_class_to_idx.get(cl_id, -1)
            for cl_id in raw_prediction_ids.tolist()
        ], dtype=np.int64)
    else:
        prediction_indices_for_grouping = np.full(
            adata.shape[0], -1, dtype=np.int64
        )
        for _row, _raw in enumerate(adata.obs[prediction_col].values):
            _cl = predictor._parse_ontology_id(str(_raw))
            if _cl is not None:
                prediction_indices_for_grouping[_row] = (
                    _grouping_class_to_idx.get(_cl, -1)
                )

    # Wang algorithm relies on the nx_graph built natively from the
    # full_pure_ontology_adj adjacency matrix that is fully saved in the H5 checkpoint.
    # No external OBO file or obo_graph object is required.

    # Build adjacency matrix (full_pure_ontology_adj serves as both adj and pure adj)
    adjacency_matrix = full_pure_ontology_adj

    simplifier = OntologySimplifier(
        adjacency_matrix=adjacency_matrix,
        pure_ontology_adj=full_pure_ontology_adj,
        node_names=full_classes,
        node_vectors=node_vectors,
        id_to_name_map=id_to_name_map,
        co_graph={},
    )

    result = simplifier.simplify(
        cell_type_counts=cell_type_counts,
        similarity_threshold=similarity_threshold,
        algorithm=algorithm,
        target_level=target_level,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
        cell_embeddings=cell_embeddings_for_grouping,
        prediction_indices=prediction_indices_for_grouping,
        min_cells_number=min_cells_number,
    )

    simplification_map: Dict[str, str] = result['simplification_map']

    # ------------------------------------------------------------------
    # Post-merge rare rollup: sweep remaining small classes
    # ------------------------------------------------------------------
    raw_simplified_counts_pre_rollup: Dict[str, int] = result['simplified_counts']

    swept_joining: Optional[Set[str]] = None
    swept_folded: Optional[Dict[str, Dict[str, Any]]] = None
    if result.get('rare_rollup_done'):
        # The grouping applied min_cells_number itself, on the tree it built,
        # where every node has one parent and the ancestors joining groups are
        # available to fold into. Sweeping again here would only walk the full
        # ontology and find nothing.
        rolled_up_counts = raw_simplified_counts_pre_rollup
    else:
        (
            simplification_map, rolled_up_counts, swept_joining, swept_folded,
        ) = _rollup_rare_classes(
            simplification_map=simplification_map,
            simplified_counts=raw_simplified_counts_pre_rollup,
            nx_graph=simplifier.nx_graph,
            node_names=full_classes,
            min_cells_number=min_cells_number,
        )
    # Overwrite result counts with post-rollup counts
    result['simplified_counts'] = rolled_up_counts

    # Ancestors kept only to join two groups. The tree needs them: without them
    # the groups are siblings, the tree has no edges, and every cell comes back
    # Stable. Groups dropped by the rollup fall away with the map; these stay.
    # Take them from the grouping when it reports them — a grouping that folds
    # small groups upward can put cells on one, and then "absent from the
    # counts" no longer picks it out.
    _grouping_connectors = result.get('connector_nodes')
    if _grouping_connectors is not None:
        connector_nodes: Set[str] = set(_grouping_connectors)
    elif swept_joining is not None:
        connector_nodes = set(swept_joining)
    else:
        _grouping_tree_nodes = result.get('tree_nodes')
        connector_nodes = (
            set(_grouping_tree_nodes) - set(raw_simplified_counts_pre_rollup)
            if _grouping_tree_nodes else set()
        )

    formatted_map: Dict[str, str] = {}
    if output_label_format != "id":
        for original_cl_id, simplified_cl_id in simplification_map.items():
            original_label = predictor._format_prediction(
                original_cl_id, output_label_format
            )
            simplified_label = predictor._format_prediction(
                simplified_cl_id, output_label_format
            )
            formatted_map[original_label] = simplified_label

    # ------------------------------------------------------------------
    # Stages 1–4: Trajectory scoring on the simplified tree
    # ------------------------------------------------------------------

    n_cells = adata.shape[0]

    # ------------------------------------------------------------------
    # Trajectory stages (1–3b): skip entirely when compute_transitions
    # is False — all cells are marked Stable with zero-valued positions.
    # ------------------------------------------------------------------
    if not compute_transitions:
        # Fast path: grouping-only mode — no placer, no tree build,
        # no trajectory arrays.
        trajectory_computed = False

    else:
        # Stage 1: Build simplified tree
        # full_pure_ontology_adj uses child→parent convention (adj[child, parent] = 1).
        # _build_simplified_tree expects parent→child, so transpose.
        parent_child_adj = full_pure_ontology_adj.T
        # Compute longest-path depths once over the full ontology. The same dict
        # is reused for `full_depths` below — both inputs to the depth-space
        # projection MUST come from the same depth convention, otherwise
        # cross-boundary cells get clipped into the stable-target zone. Longest
        # path (rather than shortest) is required because polyhierarchy nodes
        # can have shortest-path depth less than their parent's, which would
        # invert the placer's edge-direction inference.
        full_node_depths_longest: Dict[int, int] = _compute_longest_path_depths(
            parent_child_adj,
        )
        simplified_node_set = set(simplification_map.values()) | connector_nodes
        simplified_tree = _build_simplified_tree(
            parent_child_adj, full_classes, simplified_node_set,
            node_depths=full_node_depths_longest,
        )
        tree_depths = simplified_tree['depths']
        tree_edges = simplified_tree['edges']
        tree_node_indices = simplified_tree['node_indices']

        # Stage 2: Gather embeddings
        cell_embeddings = adata.obsm.get('X_hector')
        has_embeddings = cell_embeddings is not None

        # The `node_vectors` built above for OntologySimplifier may be a
        # (n_nodes, 1) zero fallback when celltype_gat.node_embeddings is not
        # populated — fine for ontology-distance-based simplification but
        # unusable for the embedding-based placer. Fetch real GAT prototypes
        # when their shape doesn't match the cell embeddings.
        trajectory_node_vectors = node_vectors
        if has_embeddings:
            cell_dim = int(np.asarray(cell_embeddings).shape[1])
            if trajectory_node_vectors.shape[1] != cell_dim and hasattr(
                predictor, '_build_visualization_state'
            ):
                try:
                    _viz_state = predictor._build_visualization_state(
                        use_full_ontology=True
                    )
                    trajectory_node_vectors = _viz_state['node_vectors']
                except Exception as exc:  # pragma: no cover — defensive
                    _simplify_logger.warning(
                        "Could not fetch GAT node embeddings for trajectory "
                        "scoring (%s); falling back to depth-only scoring.",
                        exc,
                    )
                    has_embeddings = False

        # The projection needs simplified tree edges and cell embeddings;
        # positions are always computed fresh against the simplified tree.
        if len(tree_edges) > 0 and has_embeddings:
            # Use the same longest-path-on-full-ontology dict that
            # `_build_simplified_tree` populated `tree_depths` from. The
            # depth-space projection at the bottom of this stage mixes both
            # dicts in one arithmetic, so they MUST agree on every node.
            full_depths: Dict[int, int] = full_node_depths_longest

            full_class_to_idx = {name: i for i, name in enumerate(full_classes)}

            # ============================================================
            # Place every cell against the SIMPLIFIED tree via the shared
            # helper, which builds the same placer create_trajectory_analysis()
            # builds (inferred-path discovery, direction filtering, GRIT-refined
            # anchors).
            #
            # These positions are always recomputed. A cell's position is its
            # projection onto the segment between the two node prototypes at the
            # ends of its edge, and simplification changes which nodes those
            # are; the projection onto one pair does not determine the
            # projection onto another. ``compute_transitions=False`` is the
            # explicit path for skipping this work.
            # ============================================================
            # Build viz_data from cached state: X_hector (always
            # present after predict()) and trajectory_node_vectors
            # (resolved earlier from _build_visualization_state or
            # celltype_gat.node_embeddings).  Avoids re-running
            # get_visualization_vectors() which triggers a redundant
            # prediction session.
            fb_cell_emb = adata.obsm.get('X_hector')
            if fb_cell_emb is None:
                raise ValueError(
                    "adata.obsm['X_hector'] is missing. "
                    "Run predictor.predict() first."
                )
            # Per-cell predicted full-ontology node index, used as the
            # prior anchors for the recompute placer (via viz_data below).
            if raw_prediction_ids is not None:
                pred_idx = np.array([
                    full_class_to_idx.get(cl_id, -1)
                    for cl_id in raw_prediction_ids.tolist()
                ], dtype=np.int64)
            else:
                pred_idx = np.full(n_cells, -1, dtype=np.int64)
                for _i, _raw in enumerate(adata.obs[prediction_col].values):
                    _cl = predictor._parse_ontology_id(str(_raw))
                    if _cl is not None:
                        pred_idx[_i] = full_class_to_idx.get(_cl, -1)
            viz_data = {
                'cell_vectors': np.asarray(fb_cell_emb, dtype=np.float32),
                'node_vectors': np.asarray(trajectory_node_vectors),
                'predictions': pred_idx,
                'node_names': full_classes,
                'adjacency_matrix': full_pure_ontology_adj,
                'pure_adjacency_matrix': full_pure_ontology_adj,
                'co_graph': {},
                'gat_embeddings': np.asarray(trajectory_node_vectors),
                'ontology_adj': full_pure_ontology_adj,
            }
            # Per design: simplify_ontology_tree annotates the collapsed CANONICAL tree only —
            # inferred paths are a create_trajectory_analysis()/visualize() feature and must never
            # be discovered or placed here. enable_inferred_paths=False makes that contract explicit
            # and future-proof. (It is currently also a no-op: the viz_data below passes the pure
            # ontology as `ontology_adj`, so discover_inferred_paths finds no model-added edges and
            # returns an empty set regardless of this flag.) min_cells_number is matched to the
            # caller so surviving active_indices / canonical tree_edges line up with visualize().
            _config = TrajectoryAnalysisConfig(
                min_cells_number=int(min_cells_number),
                enable_inferred_paths=False,
            )
            ctx = compute_trajectory_placement_context(
                _config, predictor, adata, viz_data, use_grit=True,
                cache=cache,
            )

            # The helper's `prediction_indices` and `cell_embeddings` are
            # restricted to surviving cells via `keep_mask`.  simplify_ontology_tree
            # needs per-row data for ALL adata rows (because trajectory_*
            # columns are written cell-by-cell), so we run the placer on
            # the helper's filtered slice and then re-expand back to full
            # length using keep_mask.
            keep_mask_ctx = ctx['keep_mask']
            ctx_cell_emb = ctx['cell_embeddings']
            ctx_pred_indices = ctx['prediction_indices']
            if ctx_pred_indices is None:
                ctx_pred_indices = np.full(int(keep_mask_ctx.sum()), -1, dtype=np.int64)

            node_position_dict = {
                int(i): (float(full_node_depths_longest.get(int(i), 0)), 0.0)
                for i in ctx['active_indices']
            }
            node_cell_counts_dict: Dict[int, int] = {
                int(i): 0 for i in ctx['active_indices']
            }
            for i in ctx_pred_indices.tolist():
                if i >= 0:
                    node_cell_counts_dict[int(i)] = (
                        node_cell_counts_dict.get(int(i), 0) + 1
                    )

            calibration_placer = LatentBarycentricPlacer(
                node_positions=node_position_dict,
                node_embeddings=ctx['node_embeddings'],
                tree_edges=ctx['tree_edges'],
                temperature=_config.temperature,
                top_k_neighbors=_config.top_k_neighbors,
                stable_source_zone=None,  # zones detected below
                stable_target_zone=None,
                node_cell_counts=node_cell_counts_dict,
                scale_cloud_by_count=False,
                is_polar=False,
                lateral_directions=ctx['lateral_directions'],
                inferred_path_weights=ctx['inferred_path_weights'],
            )

            raw_result = calibration_placer.compute_raw_transitions(
                ctx_cell_emb,
                prior_anchors=ctx_pred_indices,
                show_transitions=True,
            )
            # Re-expand results back to full adata length (cells dropped by
            # keep_mask get parent_full = -1, child_full = -1, raw_t = 0.0
            # — they go through the c_full == -1 branch downstream).
            raw_t_full = np.zeros(n_cells, dtype=np.float64)
            parent_full = np.full(n_cells, -1, dtype=np.int64)
            child_full = np.full(n_cells, -1, dtype=np.int64)
            raw_t_full[keep_mask_ctx] = raw_result['raw_t']
            parent_full[keep_mask_ctx] = raw_result['parent_nodes']
            child_full[keep_mask_ctx] = raw_result['child_nodes']

            # Single auto-detection on full-tree raw_t (algorithm-invariant).
            edge_placed = child_full != -1
            if edge_placed.any():
                zone_info = _ts._auto_detect_stable_zones(raw_t_full[edge_placed])
                src_zone = float(zone_info["stable_source_zone"])
                tgt_zone = float(zone_info["stable_target_zone"])
            else:
                src_zone = 0.1
                tgt_zone = 0.1
                zone_info = {
                    "stable_source_zone": 0.1,
                    "stable_target_zone": 0.1,
                    "mode": "fallback",
                    "components": [],
                    "n_fit_cells": 0,
                    "reason": "no edge-placed cells",
                }

            # ============================================================
            # Stage 3b: Per-cell mapping from full-tree edge → simplified
            # tree label + classification with k-scaling (user's Step 3:
            # cells anchored at intermediate full-tree nodes within a
            # collapsed span are reclassified as Transitioning).
            # ============================================================
            raw_positions = np.zeros(n_cells, dtype=np.float64)
            global_scores = np.zeros(n_cells, dtype=np.float64)
            source_labels = np.full(n_cells, '', dtype=object)
            target_labels = np.full(n_cells, '', dtype=object)
            trajectory_states = np.full(n_cells, 'Stable', dtype=object)
            trajectory_positions = np.zeros(n_cells, dtype=np.float64)

            # Pre-compute the simplified-tree label for every full-ontology node
            # name that any cell's parent_full / child_full points to. Walking
            # up the DAG once per unique name avoids redundant BFS work in the
            # per-cell loop and — critically — resolves ancestor sources that
            # are not predicted cell types (and therefore absent from
            # simplification_map). Without the walk-up, the projection
            # downstream silently uses tree_depths.get(unmapped_name, 0).
            _child_parent_adj_full = parent_child_adj.T
            _name_to_idx_full = {n: i for i, n in enumerate(full_classes)}
            _unique_anchor_indices: Set[int] = set()
            for i in range(n_cells):
                p_full_i = int(parent_full[i])
                c_full_i = int(child_full[i])
                if 0 <= p_full_i < len(full_classes):
                    _unique_anchor_indices.add(p_full_i)
                if 0 <= c_full_i < len(full_classes):
                    _unique_anchor_indices.add(c_full_i)
            _unique_anchor_names = {full_classes[i] for i in _unique_anchor_indices}
            simplified_label_cache: Dict[str, str] = {
                name: _walk_up_to_simplified_label(
                    name, simplification_map, simplified_node_set,
                    _child_parent_adj_full, _name_to_idx_full, full_classes,
                )
                for name in _unique_anchor_names
            }

            # A cell may only be reported on a route the simplified tree
            # actually draws. Two groups with no edge between them cannot carry
            # a position, so a cell whose ends land on such a pair collapses
            # into its own group instead.
            simplified_edge_set = {
                (str(parent_id), str(child_id)) for parent_id, child_id in tree_edges
            }
            own_group_label = np.full(n_cells, '', dtype=object)
            for i in range(n_cells):
                own_idx = int(pred_idx[i])
                if 0 <= own_idx < len(full_classes):
                    own_name = full_classes[own_idx]
                    own_group_label[i] = simplification_map.get(
                        own_name, simplified_label_cache.get(own_name, own_name)
                    )

            for i in range(n_cells):
                p_full = int(parent_full[i])
                c_full = int(child_full[i])

                if c_full == -1:
                    # No edge assigned — cell stable at its anchor's simplified node.
                    if 0 <= p_full < len(full_classes):
                        p_full_name = full_classes[p_full]
                        p_simp = simplified_label_cache.get(p_full_name, p_full_name)
                        source_labels[i] = p_simp
                        global_scores[i] = float(tree_depths.get(p_simp, full_depths.get(p_full, 0)))
                    continue

                p_full_name = full_classes[p_full]
                c_full_name = full_classes[c_full]
                p_simp = simplified_label_cache.get(p_full_name, p_full_name)
                c_simp = simplified_label_cache.get(c_full_name, c_full_name)

                if p_simp == c_simp:
                    # Cell anchored inside a collapsed simplified node → Stable.
                    source_labels[i] = p_simp
                    target_labels[i] = p_simp
                    d_simp = tree_depths.get(p_simp, 0)
                    global_scores[i] = float(d_simp)
                    # raw_positions stays 0 and trajectory_states stays 'Stable'.
                    continue

                if (p_simp, c_simp) not in simplified_edge_set:
                    # The two groups exist but nothing joins them on the tree,
                    # so there is no route to place this cell on: it collapses
                    # into its own group and stays there.
                    home = own_group_label[i] or p_simp
                    source_labels[i] = home
                    global_scores[i] = float(tree_depths.get(home, 0))
                    continue

                # Cell straddles a simplification boundary; project onto the
                # simplified edge by depth-space arithmetic.
                d_p_full = full_depths.get(p_full, 0)
                d_c_full = full_depths.get(c_full, d_p_full + 1)
                d_p_simp = tree_depths.get(p_simp, 0)
                d_c_simp = tree_depths.get(c_simp, d_p_simp + 1)
                k = max(d_c_simp - d_p_simp, 1)

                # raw_t is a 0–1 fraction along the cell's FULL-tree edge, and
                # that edge spans `span_full` ontology levels — more than one
                # whenever intermediate types were pruned from the display tree.
                span_full = max(d_c_full - d_p_full, 1)

                depth_pos = d_p_full + raw_t_full[i] * span_full
                simplified_t = (depth_pos - d_p_simp) / k
                simplified_t = float(np.clip(simplified_t, 0.0, 1.0))

                effective_src = src_zone / k
                effective_tgt = tgt_zone / k

                if simplified_t <= effective_src:
                    trajectory_states[i] = 'Stable'
                    trajectory_positions[i] = 0.0
                elif simplified_t >= (1.0 - effective_tgt):
                    trajectory_states[i] = 'Stable'
                    trajectory_positions[i] = 0.0
                else:
                    trajectory_states[i] = 'Transitioning'
                    band = 1.0 - effective_src - effective_tgt
                    trajectory_positions[i] = (
                        (simplified_t - effective_src) / band if band > 0 else 0.5
                    )

                raw_positions[i] = simplified_t
                source_labels[i] = p_simp
                target_labels[i] = c_simp
                global_scores[i] = d_p_simp + simplified_t * (d_c_simp - d_p_simp)

            trajectory_computed = True
        else:
            # Nothing to measure against: no cell embeddings, or a tree with no
            # parent-child pair to place cells on. Say so — the columns below
            # are all Stable / 0.0 / blank, which reads exactly like a genuine
            # "measured, found no movement" result.
            if not has_embeddings:
                _simplify_logger.warning(
                    "No cell embeddings in adata.obsm['X_hector'] — transitions "
                    "were not measured, so every cell is reported Stable."
                )
            else:
                _simplify_logger.warning(
                    "The simplified tree has %d node(s) and no routes between "
                    "them, so no cell can be measured as moving from one group "
                    "to another; every cell is reported Stable. Try a smaller "
                    "max_group_size, or algorithm='wang' / 'level'.",
                    len(simplified_node_set),
                )
            raw_positions = np.zeros(n_cells, dtype=np.float64)
            global_scores = np.zeros(n_cells, dtype=np.float64)
            source_labels = np.full(n_cells, '', dtype=object)
            target_labels = np.full(n_cells, '', dtype=object)
            trajectory_states = np.full(n_cells, 'Stable', dtype=object)
            trajectory_positions = np.zeros(n_cells, dtype=np.float64)
            zone_info = None

            for i, raw_val in enumerate(adata.obs[prediction_col].values):
                cl_id = predictor._parse_ontology_id(str(raw_val))
                if cl_id is not None:
                    simplified_id = simplification_map.get(cl_id, cl_id)
                    global_scores[i] = float(tree_depths.get(simplified_id, 0))
                    source_labels[i] = simplified_id

            trajectory_computed = False

    # ------------------------------------------------------------------
    # Write simplified_prediction to adata.obs
    # ------------------------------------------------------------------
    if output_label_format == "id":
        output_map = simplification_map
    else:
        output_map = formatted_map

    # Map each cell's prediction to its simplified label.
    # raw_prediction_ids contains canonical CL IDs (from the manifest);
    # use them when available so output_label_format is honored regardless
    # of what format the original prediction column uses.
    if raw_prediction_ids is not None:
        simplified_labels = []
        for cl_id in raw_prediction_ids.tolist():
            # output_map keys are in output_label_format; convert cl_id if needed.
            if output_label_format == "id":
                key = cl_id
            else:
                key = predictor._format_prediction(cl_id, output_label_format)
            simplified_labels.append(output_map.get(key, key))
    else:
        # Fallback: prediction_col values are in annotation_label_format.
        # If output_label_format matches annotation_label_format, output_map
        # keys align with prediction_col values; otherwise we may emit values
        # in annotation_label_format rather than output_label_format.
        simplified_labels = []
        for raw_val in adata.obs[prediction_col].values:
            label = str(raw_val)
            simplified_labels.append(output_map.get(label, label))
    adata.obs['simplified_prediction'] = simplified_labels

    # ------------------------------------------------------------------
    # Stage 5: Write trajectory obs columns
    # ------------------------------------------------------------------
    # Skip trajectory columns entirely when compute_transitions is False
    # — they would only contain placeholder zeros which are meaningless.
    if compute_transitions:
        # Classification + per-edge depth scaling were computed in Stage 3b
        # using algorithm-invariant zones from the full-tree calibration pass.
        # This stage just persists the results to adata.obs.
        if trajectory_computed:
            adata.obs['simplified_trajectory_state'] = trajectory_states
            adata.obs['simplified_relative_position'] = raw_positions
            adata.obs['simplified_transition_position'] = trajectory_positions
        else:
            adata.obs['simplified_trajectory_state'] = 'Stable'
            adata.obs['simplified_relative_position'] = raw_positions
            adata.obs['simplified_transition_position'] = 0.0

        # Format source/target labels to match output_label_format
        if output_label_format != "id":
            for i in range(n_cells):
                if source_labels[i]:
                    source_labels[i] = predictor._format_prediction(
                        source_labels[i], output_label_format,
                    )
                if target_labels[i]:
                    target_labels[i] = predictor._format_prediction(
                        target_labels[i], output_label_format,
                    )

        adata.obs['simplified_trajectory_source'] = source_labels
        adata.obs['simplified_trajectory_target'] = target_labels
        adata.obs['simplified_global_score'] = global_scores

    # ------------------------------------------------------------------
    # Build output counts and return
    # ------------------------------------------------------------------
    raw_simplified_counts: Dict[str, int] = result['simplified_counts']
    if output_label_format == "id":
        output_simplified_counts = raw_simplified_counts
    else:
        output_simplified_counts = {}
        for raw_cl_id, count in raw_simplified_counts.items():
            formatted_label = predictor._format_prediction(
                raw_cl_id, output_label_format
            )
            output_simplified_counts[formatted_label] = (
                output_simplified_counts.get(formatted_label, 0) + count
            )

    uns_payload: Dict[str, Any] = {
        'simplification_map': output_map,
        'simplified_counts': output_simplified_counts,
        'label_format': output_label_format,
    }
    if connector_nodes:
        # Kept to join two groups. plot_simplified_tree cannot rediscover them
        # and would otherwise draw a different tree from the one the cells were
        # measured against.
        uns_payload['connector_nodes'] = sorted(
            predictor._format_prediction(cl_id, output_label_format)
            for cl_id in connector_nodes
        )
    _folded_nodes = result.get('folded_nodes') or swept_folded
    if _folded_nodes:
        # Groups that held fewer than min_cells_number cells, and where those
        # cells went. Recorded rather than logged so the figure's counts can be
        # reconciled with the predictions afterwards.
        uns_payload['folded_nodes'] = {
            predictor._format_prediction(cl_id, output_label_format): {
                'into': predictor._format_prediction(
                    entry['into'], output_label_format
                ),
                'cells': int(entry['cells']),
            }
            for cl_id, entry in _folded_nodes.items()
        }
    if trajectory_computed and zone_info is not None:
        # Surface the algorithm-invariant zones produced by the full-tree
        # calibration so downstream consumers / tests can read them without
        # monkey-patching ``_auto_detect_stable_zones``.
        uns_payload['auto_zones'] = {
            'stable_source_zone': float(zone_info.get('stable_source_zone', 0.1)),
            'stable_target_zone': float(zone_info.get('stable_target_zone', 0.1)),
            'mode': str(zone_info.get('mode', 'fallback')),
            'n_fit_cells': int(zone_info.get('n_fit_cells', 0)),
        }
    adata.uns['hector_simplified'] = uns_payload


def _resolve_simplified_label(
    label: str,
    cl_to_idx: Dict[str, int],
    name_to_cl_ids: Dict[str, List[str]],
) -> Optional[int]:
    """Resolve a simplified-prediction label to a node index in the active ontology.

    Tries (in order): direct CL ID match, parenthesised "name (ID)" extraction,
    then unique name lookup. Returns None if the label cannot be uniquely
    resolved.
    """
    if label in cl_to_idx:
        return cl_to_idx[label]
    if label.endswith(")") and "(" in label:
        cand = label.rsplit("(", 1)[-1][:-1].strip()
        if cand in cl_to_idx:
            return cl_to_idx[cand]
    cands = name_to_cl_ids.get(label, [])
    if len(cands) == 1:
        return cl_to_idx.get(cands[0])
    return None


def _prefer_placed_parents(
    adjacency: np.ndarray,
    node_names: List[str],
    adata: 'anndata.AnnData',
    cl_to_idx: Dict[str, int],
    name_to_cl_ids: Dict[str, List[str]],
    id_to_name_map: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Reduce a child->parent DAG to one parent per node, keeping the parent
    that ``simplify_ontology_tree`` placed transitioning cells against.

    A radial figure needs a strict tree, so any type with more than one
    ontology parent has to lose all but one. Which one is kept used to be
    incidental, and a cell whose ``(source, target)`` link was the discarded
    one could not be drawn on any branch — it silently ended up parked in a
    node cloud, indistinguishable from a cell that really is stable there.

    The rule has no thresholds in it:

    * cells on exactly one of a type's parent links -> use that link;
    * cells on more than one -> the link with the most cells wins;
    * a tie -> the parent with the lowest CL ID;
    * no cells on any link -> leave that type's parents alone, so
      ``DAGToTreeConverter`` resolves it exactly as it does today.

    Every decision is logged with the counts behind it. When cells provide a
    parent preference, that evidence takes precedence over the converter's
    parent affinity; in this workflow, cells of a type use the same node vector
    and therefore score identically against candidate parents.

    Ties go by CL ID rather than by position in the class list because class
    order belongs to the checkpoint while a CL ID belongs to the ontology; an
    index tie-break would let the same data draw a different tree against a
    re-ordered checkpoint.

    Only the radial branch of :func:`plot_simplified_tree` uses this;
    ``DAGToTreeConverter`` itself is left alone because it is shared with
    the full-tree and group-comparison figures.

    Args:
        adjacency: Child->parent matrix, ``adj[child, parent] == 1``.
        node_names: Node names indexed like ``adjacency``.
        adata: AnnData carrying the ``simplified_trajectory_*`` columns.
        cl_to_idx: CL ID -> node index.
        name_to_cl_ids: readable name -> CL IDs, for label resolution.
        id_to_name_map: CL ID -> readable name, for the log lines only.

    Returns:
        A copy of ``adjacency`` with at most one parent set per node. The
        input is never modified.
    """
    reduced = np.array(adjacency, copy=True)

    # Tally placed cells per (parent, child) link.
    link_counts: Dict[Tuple[int, int], int] = {}
    cols = ('simplified_trajectory_state', 'simplified_trajectory_source',
            'simplified_trajectory_target')
    if all(c in adata.obs.columns for c in cols):
        states = adata.obs['simplified_trajectory_state'].values
        sources = adata.obs['simplified_trajectory_source'].values
        targets = adata.obs['simplified_trajectory_target'].values
        for i in range(len(states)):
            if states[i] != 'Transitioning':
                continue
            s = _resolve_simplified_label(str(sources[i]), cl_to_idx, name_to_cl_ids)
            t = _resolve_simplified_label(str(targets[i]), cl_to_idx, name_to_cl_ids)
            if s is None or t is None:
                continue
            link_counts[(s, t)] = link_counts.get((s, t), 0) + 1

    # Only a node that actually carries placed
    # cells on one of its links is pinned to that parent; every other
    # multi-parent node keeps its full parent set, so DAGToTreeConverter
    # resolves it using its normal rule. Reducing every node to one parent would
    # instead rewire ancestor chains through pruned intermediates.
    def _label(idx: int) -> str:
        cl = str(node_names[idx])
        return (id_to_name_map or {}).get(cl, cl)

    for child in range(reduced.shape[0]):
        parents = np.nonzero(reduced[child])[0]
        if parents.size <= 1:
            continue
        counts = {int(p): link_counts.get((int(p), int(child)), 0) for p in parents}
        if max(counts.values()) == 0:
            continue                      # no evidence: leave this node alone
        best = min(counts, key=lambda p: (-counts[p], str(node_names[p])))
        reduced[child, :] = 0.0
        reduced[child, best] = 1.0

        # Report every decision with the evidence behind it. A one-cell
        # decision is as visible as a thousand-cell one; the reader judges,
        # not a threshold.
        rejected = sorted(
            (q for q in counts if q != best),
            key=lambda q: (-counts[q], str(node_names[q])),
        )
        contested = any(counts[q] > 0 for q in rejected)
        tied = any(counts[q] == counts[best] for q in rejected)
        _simplify_logger.info(
            "Radial layout: drawing '%s' under '%s' (%d placed cell(s)) rather "
            "than %s.%s",
            _label(child), _label(best), counts[best],
            ", ".join(f"'{_label(q)}' ({counts[q]})" for q in rejected),
            (" Tied on cell count; resolved by ontology ID." if tied else
             " Cells on the other link(s) are drawn against a substituted "
             "parent." if contested else ""),
        )
    return reduced


def plot_simplified_tree(
    predictor: 'HECTOR',
    adata: 'anndata.AnnData',
    output_file: str,
    layout_mode: str = "radial",
    label_format: str = "id",
    show_transitions: bool = True,
    cloud_scale_exponent: float = 1.0,
    cloud_size_multiplier: float = 1.0,
    cloud_size_min: float = 0.2,
    cloud_size_max: float = 1.0,
    radial_sweep_angle_degrees: Optional[float] = None,
) -> None:
    """Render a simplified ontology tree to a PDF or HTML file.

    Reads ``simplified_counts`` from ``adata.uns['hector_simplified']`` (written
    by ``simplify_ontology_tree()``) and the predictor's ontology data, then calls
    the plot builders directly (not via ``visualize()`` /
    ``get_visualization_vectors()``).

    Args:
        predictor: Loaded ``HECTOR`` instance with ontology data.
        adata: AnnData object that has been processed by
            ``simplify_ontology_tree()``.  Must contain
            ``adata.uns['hector_simplified']`` with at least a
            ``'simplified_counts'`` key mapping ontology labels → cell counts.
        output_file: Destination path.  Extension determines format:
            ``.pdf`` → static matplotlib figure,
            ``.html`` → interactive Plotly figure.
        layout_mode: ``"radial"`` (default) or ``"horizontal"``.
        label_format: ``"id"`` (default), ``"name"``, or ``"both"``.
        show_transitions: If True (default), transitioning cells are placed
            along edges using ``simplified_relative_position``.  If False,
            all cells are placed in stable clouds.
        cloud_scale_exponent: Sensitivity to population differences (default:
            1.0). Used by the shared adaptive cloud-size interpolation.
        cloud_size_multiplier: Global multiplier for cloud sizes (default: 1.0).
        cloud_size_min: Minimum cloud size limit for relative scale factor
            (default: 0.2).
        cloud_size_max: Maximum cloud size limit for relative scale factor
            (default: 1.0).
        radial_sweep_angle_degrees: Manual override for the total angular
            sweep of the radial tree, in degrees. ``None`` (default) picks
            an angle automatically from tree size (40°-240°). Only used
            when ``layout_mode="radial"``; ignored (with a warning) for
            ``layout_mode="horizontal"``.

    Raises:
        ValueError: If ``adata.uns['hector_simplified']`` is missing, if
            ``output_file`` extension is not ``.pdf`` or ``.html``, if
            ``layout_mode`` is not ``"radial"`` or ``"horizontal"``, if
            the resolved counts dict is empty, or if
            ``radial_sweep_angle_degrees`` is not a finite positive number.
    """
    from .trajectory_support import compute_adaptive_pdf_ribbon_scale
    if 'hector_simplified' not in adata.uns:
        raise ValueError(
            "adata.uns['hector_simplified'] not found. "
            "Run simplify_ontology_tree() first."
        )
    if 'simplified_prediction' not in adata.obs.columns:
        raise ValueError(
            "adata.obs['simplified_prediction'] not found. "
            "Run simplify_ontology_tree() first."
        )
    simplified_counts = adata.uns['hector_simplified']['simplified_counts']

    # Input validation
    _ext = os.path.splitext(output_file)[1].lower()
    if _ext not in ('.pdf', '.html', '.htm'):
        raise ValueError(
            f"Unsupported output file extension '{_ext}'. "
            "Use '.pdf' for static matplotlib or '.html' for interactive Plotly."
        )
    _output_format = 'pdf' if _ext == '.pdf' else 'html'

    if layout_mode not in ('radial', 'horizontal'):
        raise ValueError(
            f"layout_mode must be 'radial' or 'horizontal', got '{layout_mode!r}'."
        )

    if radial_sweep_angle_degrees is not None:
        if (
            isinstance(radial_sweep_angle_degrees, bool)
            or not isinstance(radial_sweep_angle_degrees, (int, float, np.integer, np.floating))
            or not np.isfinite(float(radial_sweep_angle_degrees))
            or not (0.0 < float(radial_sweep_angle_degrees) <= 360.0)
        ):
            raise ValueError(
                "radial_sweep_angle_degrees must be a finite number in "
                f"(0, 360], got {radial_sweep_angle_degrees!r}."
            )
        if layout_mode != 'radial':
            _simplify_logger.warning(
                "radial_sweep_angle_degrees=%r is ignored because layout_mode=%r "
                "(it only affects layout_mode='radial').",
                radial_sweep_angle_degrees, layout_mode,
            )

    if not simplified_counts:
        raise ValueError("simplified_counts must be non-empty.")

    # Extract ontology data from predictor
    full_pure_ontology_adj: np.ndarray = predictor.full_pure_ontology_adj
    full_classes: List[str] = list(predictor.celltype_gat.full_classes)
    id_to_name_map: Dict[str, str] = getattr(predictor, 'id_to_name_map', {})

    # Node vectors (GAT embeddings)
    node_vectors: Optional[np.ndarray] = None
    if hasattr(predictor, 'celltype_gat') and hasattr(predictor.celltype_gat, 'node_embeddings'):
        node_vectors = predictor.celltype_gat.node_embeddings
    if node_vectors is None and hasattr(predictor, 'node_vectors'):
        node_vectors = predictor.node_vectors
    if node_vectors is None:
        node_vectors = np.zeros((len(full_classes), 1), dtype=np.float32)
    if not isinstance(node_vectors, np.ndarray):
        node_vectors = np.array(node_vectors)

    # ------------------------------------------------------------------
    # Step 3 — Map simplified labels to node indices
    # ------------------------------------------------------------------
    cl_to_idx: Dict[str, int] = {name: i for i, name in enumerate(full_classes)}

    # Reverse lookup for label_format="name" inputs. Restrict to classes that
    # are present in the loaded ontology so auxiliary names do not interfere.
    name_to_cl_ids: Dict[str, List[str]] = defaultdict(list)
    for cl_id, class_name in id_to_name_map.items():
        if class_name and cl_id in cl_to_idx:
            name_to_cl_ids[class_name].append(cl_id)

    # Build node_counts dict keyed by node index (same role as in visualize()).
    # Accept raw IDs, "name (ID)" labels, and uniquely resolvable name-only labels.
    node_counts: Dict[int, int] = {}
    unmatched_labels: List[str] = []
    ambiguous_labels: Dict[str, List[str]] = {}
    for raw_label, count in simplified_counts.items():
        label = str(raw_label)
        resolved_cl_id: Optional[str] = None

        if label in cl_to_idx:
            resolved_cl_id = label
        elif label.endswith(")") and "(" in label:
            candidate_cl_id = label.rsplit("(", 1)[-1][:-1].strip()
            if candidate_cl_id in cl_to_idx:
                resolved_cl_id = candidate_cl_id

        if resolved_cl_id is None:
            candidate_cl_ids = name_to_cl_ids.get(label, [])
            if len(candidate_cl_ids) == 1:
                resolved_cl_id = candidate_cl_ids[0]
            elif len(candidate_cl_ids) > 1:
                ambiguous_labels[label] = sorted(candidate_cl_ids)

        if resolved_cl_id is None:
            unmatched_labels.append(label)
            continue

        idx = cl_to_idx[resolved_cl_id]
        node_counts[idx] = node_counts.get(idx, 0) + int(count)

    if ambiguous_labels:
        ambiguous_preview = "; ".join(
            f"{label!r} -> {candidate_ids}"
            for label, candidate_ids in list(ambiguous_labels.items())[:3]
        )
        _simplify_logger.warning(
            "Ignoring %d ambiguous simplified_counts label(s) that match multiple "
            "ontology IDs: %s",
            len(ambiguous_labels),
            ambiguous_preview,
        )

    unresolved_labels = [
        label for label in unmatched_labels if label not in ambiguous_labels
    ]
    if unresolved_labels:
        unresolved_preview = ", ".join(repr(label) for label in unresolved_labels[:5])
        _simplify_logger.warning(
            "Ignoring %d simplified_counts label(s) that were not found in the "
            "predictor ontology: %s",
            len(unresolved_labels),
            unresolved_preview,
        )

    if not node_counts:
        accepted_examples = ", ".join(repr(cl_id) for cl_id in full_classes[:3])
        message = (
            "None of the labels in simplified_counts could be matched to the "
            "predictor's ontology. Pass raw ontology IDs or labels in HECTOR's "
            f"'name'/'both' formats. Example ontology IDs: {accepted_examples}"
        )
        if unresolved_labels:
            unmatched_preview = ", ".join(repr(label) for label in unresolved_labels[:5])
            message += f". Unmatched labels: {unmatched_preview}"
        if ambiguous_labels:
            ambiguous_preview = "; ".join(
                f"{label!r} -> {candidate_ids}"
                for label, candidate_ids in list(ambiguous_labels.items())[:3]
            )
            message += f". Ambiguous labels: {ambiguous_preview}"
        raise ValueError(message)

    # Build predictions from adata.obs['simplified_prediction'] so the array's
    # cell ordering matches adata.obs row order. Cells whose simplified
    # prediction can't be resolved to an active node are excluded.
    obs_predictions = adata.obs['simplified_prediction'].values
    predictions_list: List[int] = []
    obs_keep_mask = np.zeros(len(obs_predictions), dtype=bool)
    for obs_i, raw_label in enumerate(obs_predictions):
        idx = _resolve_simplified_label(str(raw_label), cl_to_idx, name_to_cl_ids)
        if idx is None:
            continue
        predictions_list.append(idx)
        obs_keep_mask[obs_i] = True
    predictions = np.array(predictions_list, dtype=np.int64)

    # ------------------------------------------------------------------
    # Step 4 — Build AdaptiveOntologySubgraph and get active indices
    # ------------------------------------------------------------------
    node_names: List[str] = list(full_classes)  # may be mutated for radial
    adjacency_matrix = full_pure_ontology_adj

    subgraph = AdaptiveOntologySubgraph(
        adjacency_matrix=adjacency_matrix,
        node_names=node_names,
        co_graph=None,
    )

    active_indices: List[int] = list(subgraph.get_relevant_node_indices(
        predicted_indices=predictions,
        include_ancestors=True,
        include_siblings=False,
        siblings_rule='leaves_only',
    ))

    # Nodes the grouping kept to join two groups. They hold no cells, so the
    # pruning below would drop them and this function would draw a different
    # tree from the one the cells were measured against.
    protected_indices: Set[int] = set()
    for raw_label in adata.uns['hector_simplified'].get('connector_nodes', []) or []:
        idx = _resolve_simplified_label(str(raw_label), cl_to_idx, name_to_cl_ids)
        if idx is not None:
            protected_indices.add(int(idx))
            if idx not in active_indices:
                active_indices.append(int(idx))

    # ------------------------------------------------------------------
    # Step 5 — SCC pruning and chain collapse (mirrors visualize())
    # ------------------------------------------------------------------
    # Use a small min_cells floor so ancestor-only nodes (count=0) are kept
    # unless they are truly isolated leaf stubs.
    min_cells = 1

    iteration = 0
    while True:
        iteration += 1
        current_g = subgraph.nx_graph.subgraph(active_indices)
        C = nx.condensation(current_g)
        leaf_components = [c_id for c_id in C.nodes() if C.in_degree(c_id) == 0]
        components_to_remove = []
        for c_id in leaf_components:
            members = C.nodes[c_id]['members']
            comp_total = sum(node_counts.get(m, 0) for m in members)
            if comp_total < min_cells:
                components_to_remove.append(c_id)
        if not components_to_remove:
            break
        nodes_to_remove: List[int] = []
        for c_id in components_to_remove:
            nodes_to_remove.extend(C.nodes[c_id]['members'])
        to_remove_set = set(nodes_to_remove) - protected_indices
        if not to_remove_set:
            break
        active_indices = [n for n in active_indices if n not in to_remove_set]
        if not active_indices or iteration > 50:
            break

    # Chain collapse: remove zero-count pass-through nodes
    collapsed = 0
    while True:
        current_g = subgraph.nx_graph.subgraph(active_indices)
        to_collapse = [
            n for n in active_indices
            if current_g.in_degree(n) == 1
            and current_g.out_degree(n) >= 1
            and node_counts.get(n, 0) == 0
            and n not in protected_indices
        ]
        if not to_collapse:
            break
        active_indices = [idx for idx in active_indices if idx not in to_collapse]
        collapsed += len(to_collapse)
        if collapsed > 100:
            break

    # Rebuild predictions to only include active nodes
    active_set = set(active_indices)
    keep_mask = np.array([p in active_set for p in predictions], dtype=bool)
    predictions = predictions[keep_mask]

    # Track which obs rows are still represented after SCC pruning
    obs_indices = np.where(obs_keep_mask)[0][keep_mask]

    # Rebuild node_counts to only include active nodes
    node_counts = {k: v for k, v in node_counts.items() if k in active_set}

    # Synthetic cell vectors: repeat node_vectors rows to match predictions
    cell_vectors = node_vectors[predictions]

    # ------------------------------------------------------------------
    # Step 6 — Layout setup
    # ------------------------------------------------------------------
    is_polar = (layout_mode == 'radial')
    layout_adjacency = adjacency_matrix  # pure DAG

    # Config for layout engines
    config = TrajectoryAnalysisConfig(
        show_transitions=False,
        enable_inferred_paths=False,
        enable_atypical_detection=False,
        layout_mode=layout_mode,
        cloud_scale_exponent=cloud_scale_exponent,
        cloud_size_multiplier=cloud_size_multiplier,
        cloud_size_min=cloud_size_min,
        cloud_size_max=cloud_size_max,
        radial_sweep_angle_degrees=radial_sweep_angle_degrees,
    )
    # Wrap config in a namespace so render functions can access it as
    # ``self.config`` (they were originally methods on HierarchicalTrajectoryAnalyzer).
    import types as _types
    _config_holder = _types.SimpleNamespace(config=config)

    line_sep = '\n' if _output_format == 'pdf' else '<br>'
    formatted_original_node_names = format_node_names(
        list(full_classes), id_to_name_map, label_format, line_sep
    )

    canonical_order = CanonicalOrderComputer(subgraph.nx_graph, node_names)

    # Radial: DAG → strict tree conversion
    tree_data = None
    _tree_to_original: Optional[Dict[int, int]] = None

    if is_polar:
        # A radial figure needs a strict tree, so every type with more than one
        # ontology parent must give one up. Choose the parent the ANALYSIS
        # placed cells against: `simplify_ontology_tree` writes each
        # transitioning cell's (source, target) into adata.obs, and dropping the
        # link those cells sit on would strand them. Ties and unplaced nodes keep
        # the DAG's own first parent.
        #
        # This reduction is deliberately local to the radial branch of
        # plot_simplified_tree. DAGToTreeConverter is shared with
        # HierarchicalTrajectoryAnalyzer.visualize() and the group-comparison
        # path, whose figures must not shift underneath this change.
        layout_adjacency = _prefer_placed_parents(
            layout_adjacency, node_names, adata,
            cl_to_idx=cl_to_idx, name_to_cl_ids=name_to_cl_ids,
            id_to_name_map=id_to_name_map,
        )
        converter = DAGToTreeConverter(layout_adjacency, node_names, node_vectors)
        tree_data = converter.convert(
            active_indices,
            predictions,
            cell_vectors,
            min_cells_number=1,
        )

        # Build original→tree mapping for color remapping
        _debug_original_to_tree_map: Dict[int, List[int]] = {}
        for new_idx, old_idx in enumerate(tree_data['original_indices']):
            _debug_original_to_tree_map.setdefault(old_idx, []).append(new_idx)

        layout_adjacency = tree_data['tree_adj']
        node_names = tree_data['tree_names']
        node_vectors = tree_data['tree_vectors']
        predictions = tree_data['tree_predictions']
        active_indices = tree_data['active_indices']

        _tree_cell_mask = tree_data['cell_keep_mask']
        cell_vectors = cell_vectors[_tree_cell_mask]

        # Filter obs_indices to match the tree-converted cells
        obs_indices = obs_indices[_tree_cell_mask]

        _tree_to_original = {
            new_idx: old_idx
            for new_idx, old_idx in enumerate(tree_data['original_indices'])
        }

        # Rebuild node_counts in tree index space
        tree_node_counts: Dict[int, int] = {}
        for tree_idx, orig_idx in _tree_to_original.items():
            if orig_idx in node_counts:
                tree_node_counts[tree_idx] = node_counts[orig_idx]
        node_counts = tree_node_counts

        # Rebuild predictions-based node_counts from tree predictions
        # (tree_predictions already maps cells to tree indices)
        node_counts_from_preds: Dict[int, int] = {}
        for p in predictions:
            node_counts_from_preds[int(p)] = node_counts_from_preds.get(int(p), 0) + 1
        # Merge: prefer counts from simplified_counts (via tree_node_counts) but
        # fall back to prediction-derived counts for any tree nodes not covered
        for k, v in node_counts_from_preds.items():
            if k not in node_counts:
                node_counts[k] = v

        # Update formatted names for tree structure
        formatted_node_names = format_node_names(
            node_names, id_to_name_map, label_format, line_sep
        )

        layout_engine = RadialTreeLayoutEngine(layout_adjacency, node_names, config)
        layout_engine.canonical_order = canonical_order
        layout_engine.original_indices = list(tree_data['original_indices'])
    else:
        formatted_node_names = format_node_names(
            node_names, id_to_name_map, label_format, line_sep
        )
        layout_engine = TreeLayoutEngine(layout_adjacency, node_names, config)
        layout_engine.canonical_order = canonical_order
        layout_engine.original_indices = None
        _tree_to_original = None

    layout_engine.set_embeddings(node_vectors)

    # ------------------------------------------------------------------
    # Step 7 — Color mapping (mirrors visualize() radial/horizontal paths)
    # ------------------------------------------------------------------
    orig_node_names = list(predictor.celltype_gat.full_classes)
    orig_pure_adj = predictor.full_pure_ontology_adj
    color_mapper = None

    if is_polar and tree_data is not None:
        tree_visible_edges: List[Tuple[int, int]] = []
        rows, cols = np.where(tree_data['tree_adj'] > 0)
        for child_idx, parent_idx in zip(rows, cols):
            tree_visible_edges.append((int(parent_idx), int(child_idx)))

        original_predictions_for_color = [
            int(tree_data['original_indices'][pred_idx])
            for pred_idx in predictions
            if 0 <= int(pred_idx) < len(tree_data['original_indices'])
        ]

        color_mapper = _create_simplified_tree_color_mapper(
            tree_edges=tree_visible_edges,
            active_indices=active_indices,
            node_names=node_names,
            color_palette=config.adaptive_color_palette,
        )
        layout_engine.set_color_mapper(color_mapper)

    # ------------------------------------------------------------------
    # Step 8 — Compute layout
    # ------------------------------------------------------------------
    node_positions = layout_engine.compute_layout(active_indices, augmented_adjacency=layout_adjacency)
    canonical_tree_edges = layout_engine.tree_edges
    visual_tree_edges = list(canonical_tree_edges)  # no inferred paths

    if not is_polar:
        color_mapper = _create_simplified_tree_color_mapper(
            tree_edges=canonical_tree_edges,
            active_indices=active_indices,
            node_names=node_names,
            color_palette=config.adaptive_color_palette,
        )

    # ------------------------------------------------------------------
    # Step 9 — Resolve per-cell trajectory state into tree-space indices
    # ------------------------------------------------------------------
    # Reads simplified_trajectory_state / _source / _target and
    # simplified_relative_position from adata.obs (see ``required_obs``
    # below), and maps the ontology-space source/target labels into the
    # visual tree's index
    # space. For radial layouts, ``DAGToTreeConverter`` may split one DAG
    # node into multiple tree copies; when multiple candidate tree edges
    # match the same (src_orig, tgt_orig) pair we prefer the edge incident
    # to the cell's predicted anchor (``predictions[ci]``) so cells stay on
    # the tree branch their anchor lives on.
    n_cells = len(predictions)
    transition_state = np.zeros(n_cells, dtype=bool)
    transition_source_idx = np.full(n_cells, -1, dtype=np.int64)
    transition_target_idx = np.full(n_cells, -1, dtype=np.int64)
    transition_t = np.zeros(n_cells, dtype=float)

    if show_transitions and n_cells > 0:
        required_obs = [
            'simplified_trajectory_state',
            'simplified_relative_position',
            'simplified_trajectory_source',
            'simplified_trajectory_target',
        ]
        missing = [c for c in required_obs if c not in adata.obs.columns]
        if missing:
            raise ValueError(
                f"adata.obs is missing required trajectory columns {missing}. "
                "Run simplify_ontology_tree() first, or pass show_transitions=False."
            )
        obs_states = adata.obs['simplified_trajectory_state'].values
        # The un-stretched position, matching what the full tree draws from
        # (LatentBarycentricPlacer.place_cells), so the same cell sits at the
        # same fraction along its edge in both pictures.
        obs_positions = np.asarray(adata.obs['simplified_relative_position'].values, dtype=float)
        obs_sources = adata.obs['simplified_trajectory_source'].values
        obs_targets = adata.obs['simplified_trajectory_target'].values

        edges_by_orig: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for s_tree, t_tree in visual_tree_edges:
            so = _tree_to_original.get(s_tree, s_tree) if _tree_to_original is not None else s_tree
            to = _tree_to_original.get(t_tree, t_tree) if _tree_to_original is not None else t_tree
            edges_by_orig.setdefault((int(so), int(to)), []).append((int(s_tree), int(t_tree)))

        # Fallback wiring: the branch DRAWN into each node, whatever its source.
        # A strict tree gives every non-root node exactly one incoming branch,
        # so a cell whose own link is absent can always be drawn against the
        # same target at the same distance from it. That distance is what the
        # number means — how decisively the cell matched the target — so the
        # target and the distance are preserved and only the source, which the
        # layout is entitled to redraw, changes.
        incoming_by_target: Dict[int, List[Tuple[int, int]]] = {}
        for s_tree, t_tree in visual_tree_edges:
            to = _tree_to_original.get(t_tree, t_tree) if _tree_to_original is not None else t_tree
            incoming_by_target.setdefault(int(to), []).append((int(s_tree), int(t_tree)))

        n_substituted = 0
        n_undrawable = 0

        for ci in range(n_cells):
            obs_i = int(obs_indices[ci])
            if obs_states[obs_i] != 'Transitioning':
                continue
            src_orig = _resolve_simplified_label(str(obs_sources[obs_i]), cl_to_idx, name_to_cl_ids)
            tgt_orig = _resolve_simplified_label(str(obs_targets[obs_i]), cl_to_idx, name_to_cl_ids)
            if src_orig is None or tgt_orig is None:
                n_undrawable += 1
                continue
            candidates = edges_by_orig.get((int(src_orig), int(tgt_orig)), [])
            substituted = False
            if not candidates:
                # This cell's own link is not in the drawing — a strict tree had
                # to drop it. Draw it against the branch that IS drawn into the
                # same target.
                candidates = incoming_by_target.get(int(tgt_orig), [])
                if not candidates:
                    n_undrawable += 1
                    continue
                substituted = True

            if len(candidates) == 1:
                s_tree, t_tree = candidates[0]
            else:
                pred = int(predictions[ci])
                s_tree, t_tree = next(
                    (e for e in candidates if pred in e),
                    candidates[0],
                )
            if substituted:
                n_substituted += 1
            transition_state[ci] = True
            transition_source_idx[ci] = s_tree
            transition_target_idx[ci] = t_tree
            transition_t[ci] = float(obs_positions[obs_i])

        if n_substituted:
            _simplify_logger.info(
                "%d transitioning cell(s) drawn against a substituted parent: "
                "their ontology link is not in this layout's tree. Distance "
                "from the target type is preserved; the branch they sit on is "
                "not the one named in simplified_trajectory_source.%s",
                n_substituted,
                " Use layout_mode='horizontal' to keep every ontology link."
                if is_polar else "",
            )
        if n_undrawable:
            _simplify_logger.warning(
                "%d transitioning cell(s) could not be drawn on any branch and "
                "are shown in their type's cloud as if stable.", n_undrawable,
            )

    # ------------------------------------------------------------------
    # Step 10 — Place cells via the shared LatentBarycentricPlacer
    # ------------------------------------------------------------------
    # Routes through the same pixel-space sizing path the renderers were
    # calibrated for. Transitioning cells are placed on the placer's
    # pre-sampled Bezier curves (matching the visible edges) instead of the
    # straight Cartesian chord the previous inline block used.
    #
    # Both layout engines populate ``node_widths`` in compute_layout; the
    # placer uses them to scale transition jitter to the local branch
    # width. The per-backend conversion mirrors
    # ``HierarchicalTrajectoryAnalyzer.visualize``:
    #   HTML:             pass-through (already in display pixels).
    #   PDF radial:       1.5 × pt_to_px — convert mpl linewidth points → px
    #                     and absorb the radial renderer's 1.5× internal scale.
    #   PDF horizontal:   compute_adaptive_pdf_ribbon_scale() — the renderer
    #                     sizes ribbons proportionally to node spacing.
    _placer_node_widths: Optional[Dict[int, float]] = None
    if getattr(layout_engine, 'node_widths', None):
        if _output_format == 'pdf':
            if is_polar:
                pt_to_px = config.pdf_dpi / 72.0
                _w_scale = 1.5 * pt_to_px
            else:
                _placer_pg = _build_placer_pixel_geometry(
                    config, node_positions, active_indices, is_polar, _output_format,
                )
                _node_spacing_px = config.y_scale * _placer_pg.px_per_data_y()
                _w_scale = compute_adaptive_pdf_ribbon_scale(_node_spacing_px)
        else:
            _w_scale = 1.0
        _placer_node_widths = {
            n: float(w) * _w_scale for n, w in layout_engine.node_widths.items()
        }

    cell_positions, node_counts, cloud_extents, is_transitioning = _place_simplified_cells(
        config=config,
        node_positions=node_positions,
        node_vectors=node_vectors,
        predictions=predictions,
        cell_vectors=cell_vectors,
        visual_tree_edges=visual_tree_edges,
        active_indices=active_indices,
        node_counts=node_counts,
        is_polar=is_polar,
        output_format=_output_format,
        transition_state=transition_state,
        transition_source_idx=transition_source_idx,
        transition_target_idx=transition_target_idx,
        transition_t=transition_t,
        node_widths=_placer_node_widths,
    )

    # ------------------------------------------------------------------
    # Step 11 — Cell colors
    # ------------------------------------------------------------------
    cell_colors = color_mapper.get_colors(predictions)

    # ------------------------------------------------------------------
    # Step 12 — Build builder arguments
    # ------------------------------------------------------------------
    is_atypical = np.zeros(n_cells, dtype=bool)
    anchor_nodes = predictions.copy()
    noise_mask = np.zeros(n_cells, dtype=bool)

    # node_cell_counts and cloud_label_counts mirror the placer-refreshed
    # counts so cloud labels match the population the placer used to size
    # each cloud.
    node_cell_counts: Dict[int, int] = dict(node_counts)
    cloud_label_counts: Dict[int, int] = dict(node_counts)
    max_count = max(node_cell_counts.values()) if node_cell_counts else 1

    # formatted_node_names already computed above
    atypical_cluster_infos: List = []
    inferred_path_weights: Dict = {}

    # ------------------------------------------------------------------
    # Step 13 — Dispatch to builder
    # ------------------------------------------------------------------
    from .trajectory_render import (
        _build_matplotlib_figure,
        _build_matplotlib_radial_figure,
        _build_plotly_figure,
    )

    if _output_format == 'pdf':
        if is_polar:
            _build_matplotlib_radial_figure(
                _config_holder,
                cell_positions=cell_positions,
                predictions=predictions,
                is_transitioning=is_transitioning,
                is_atypical=is_atypical,
                anchor_nodes=anchor_nodes,
                node_positions=node_positions,
                active_indices=active_indices,
                node_names=node_names,
                formatted_node_names=formatted_node_names,
                visual_tree_edges=visual_tree_edges,
                canonical_tree_edges=canonical_tree_edges,
                layout_engine=layout_engine,
                color_mapper=color_mapper,
                cell_colors=cell_colors,
                subgraph=subgraph,
                node_cell_counts=node_cell_counts,
                max_count=max_count,
                output_path=output_file,
                atypical_cluster_infos=atypical_cluster_infos,
                inferred_path_weights=inferred_path_weights,
                cloud_label_counts=cloud_label_counts,
                noise_mask=noise_mask,
                cloud_extents=cloud_extents,
                highlight_data=None,
                highlight_color_mapper=None,
            )
        else:
            _build_matplotlib_figure(
                _config_holder,
                cell_positions=cell_positions,
                predictions=predictions,
                is_transitioning=is_transitioning,
                is_atypical=is_atypical,
                anchor_nodes=anchor_nodes,
                node_positions=node_positions,
                active_indices=active_indices,
                node_names=node_names,
                formatted_node_names=formatted_node_names,
                visual_tree_edges=visual_tree_edges,
                canonical_tree_edges=canonical_tree_edges,
                color_mapper=color_mapper,
                cell_colors=cell_colors,
                subgraph=subgraph,
                node_cell_counts=node_cell_counts,
                max_count=max_count,
                output_path=output_file,
                atypical_cluster_infos=atypical_cluster_infos,
                inferred_path_weights=inferred_path_weights,
                cloud_label_counts=cloud_label_counts,
                noise_mask=noise_mask,
                cloud_extents=cloud_extents,
                highlight_data=None,
                highlight_color_mapper=None,
            )
    else:
        _build_plotly_figure(
            _config_holder,
            cell_positions=cell_positions,
            predictions=predictions,
            is_transitioning=is_transitioning,
            is_atypical=is_atypical,
            anchor_nodes=anchor_nodes,
            node_positions=node_positions,
            active_indices=active_indices,
            node_names=node_names,
            formatted_node_names=formatted_node_names,
            visual_tree_edges=visual_tree_edges,
            canonical_tree_edges=canonical_tree_edges,
            color_mapper=color_mapper,
            cell_colors=cell_colors,
            subgraph=subgraph,
            node_cell_counts=node_cell_counts,
            max_count=max_count,
            output_path=output_file,
            atypical_cluster_infos=atypical_cluster_infos,
            is_polar=is_polar,
            layout_engine=layout_engine if is_polar else None,
            inferred_path_weights=inferred_path_weights,
            cloud_label_counts=cloud_label_counts,
            noise_mask=noise_mask,
            cloud_extents=cloud_extents,
            highlight_data=None,
            highlight_color_mapper=None,
        )


# ---------------------------------------------------------------------------
# Shared trajectory rendering helpers
# ---------------------------------------------------------------------------


def format_node_names(
    node_ids: List[str],
    id_to_name_map: Dict[str, str],
    label_format: str,
    line_separator: str = ' ',
) -> List[str]:
    """
    Format node names based on label_format preference.

    Args:
        node_ids: List of ontology IDs (e.g., ['CL:0000128', 'CL:0000084'])
        id_to_name_map: Dictionary mapping IDs to names (from predictor)
        label_format: 'id', 'name', or 'both'
        line_separator: Separator for 'both' format. Use '\\n' for matplotlib,
                      '<br>' for Plotly/HTML, or ' ' for single-line inline.

    Returns:
        List of formatted node names
    """
    formatted_names = []

    for node_id in node_ids:
        if label_format == 'id':
            formatted_names.append(node_id)
        elif label_format == 'name':
            # Get name from mapping, fallback to ID if not available
            name = id_to_name_map.get(node_id, '')
            formatted_names.append(name if name else node_id)
        else:  # 'both'
            # Get name from mapping
            name = id_to_name_map.get(node_id, '')
            if name:
                # Two-line format: name on first line, ID on second
                formatted_names.append(f"{name}{line_separator}({node_id})")
            else:
                # Fallback to ID only if name not available
                formatted_names.append(node_id)

    return formatted_names


def compute_node_depths_longest_path(
    adjacency_matrix: np.ndarray,
) -> Dict[int, int]:
    """
    Compute ontology depth for each node using topological sort.

    Depth is defined as the longest path from any root node to the given node.
    Root nodes (no incoming edges) have depth 0.

    Args:
        adjacency_matrix: [n_nodes, n_nodes] adjacency matrix where
            entry [i, j] = 1 indicates edge from node i to node j.

    Returns:
        node_depths: Dictionary mapping node index to depth (integer).
    """
    from collections import deque

    n_nodes = adjacency_matrix.shape[0]
    node_depths: Dict[int, int] = {}

    # Build graph for topological traversal
    # In our adjacency format, adjacency_matrix[child, parent] = 1
    # means child -> parent (child points to parent)
    # So in_degree for a node is the number of children pointing to it

    # For depth computation, we want to traverse from roots (no parents) to leaves
    # A node's parent is where it points to: if adj[node, parent] = 1, then parent is node's parent

    # Compute parents for each node
    parents: Dict[int, list] = {i: [] for i in range(n_nodes)}
    children: Dict[int, list] = {i: [] for i in range(n_nodes)}

    for i in range(n_nodes):
        for j in range(n_nodes):
            if adjacency_matrix[i, j] > 0:
                # Edge from i to j means j is parent of i
                parents[i].append(j)
                children[j].append(i)

    # Find roots (nodes with no parents)
    roots = [i for i in range(n_nodes) if len(parents[i]) == 0]

    # Initialize all depths to 0
    for i in range(n_nodes):
        node_depths[i] = 0

    # For each node, depth = max(parent depths) + 1
    # Process in topological order (BFS from roots)
    visited: set = set()
    queue = deque(roots)

    for root in roots:
        node_depths[root] = 0
        visited.add(root)

    while queue:
        node = queue.popleft()
        current_depth = node_depths[node]

        for child in children[node]:
            # Child's depth is at least parent's depth + 1
            new_depth = current_depth + 1
            if new_depth > node_depths[child]:
                node_depths[child] = new_depth

            # Add child to queue if all its parents have been processed
            parent_depths_known = all(p in visited for p in parents[child])
            if parent_depths_known and child not in visited:
                visited.add(child)
                queue.append(child)

    # Handle any unvisited nodes (disconnected components)
    for i in range(n_nodes):
        if i not in visited:
            node_depths[i] = 0

    return node_depths


def compute_trajectory_placement_context(
    config: 'TrajectoryAnalysisConfig',
    predictor,
    adata,
    viz_data: Dict[str, Any],
    use_grit: bool = True,
    cache: Optional[VisualizationCache] = None,
) -> Dict[str, Any]:
    """Compute the shared inputs needed to build the full trajectory placer.

    Encapsulates inferred-path discovery, direction filtering,
    active-subgraph construction, and GRIT-refined anchor resolution
    used by both ``visualize()`` and ``simplify_ontology_tree()``. With the same
    inputs and runtime configuration, both paths use the same placement rules.

    Args:
        config: TrajectoryAnalysisConfig instance.
        predictor: HECTOR instance.
        adata: AnnData object that has been processed by ``predict()``.
        viz_data: Output of ``predictor.get_visualization_vectors(adata)``.
        use_grit: When True, pass GRIT-refined predictions as the
            placer's ``prior_anchors``. When False, the caller should
            use ``None`` for ``prior_anchors`` and let the placer fall
            back to argmax-of-weights.
        cache: Optional VisualizationCache for reusing precomputed ontology
            distances and embedding similarity across calls. When None,
            computations are performed without caching.

    Returns:
        Dict with keys: adjacency_matrix, augmented_adjacency,
        active_indices, tree_edges, lateral_directions,
        inferred_path_weights, inferred_path_weights_unfiltered,
        node_depths_dag, prediction_indices, cell_embeddings,
        node_embeddings, node_names, keep_mask, node_counts.
    """
    from .trajectory_support import (
        _build_canonical_trajectory_context,
        _build_visual_edges,
    )

    cell_vectors = viz_data['cell_vectors']
    node_vectors = viz_data['node_vectors']
    predictions = viz_data['predictions']
    node_names = viz_data['node_names']

    # Build adjacency: prefer the model's pure ontology adjacency unless
    # the caller's config supplied an explicit OBO file.
    n_nodes = len(node_names)
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    if config.ontology_path:
        if not os.path.exists(config.ontology_path):
            raise FileNotFoundError(
                f"Ontology file not found: {config.ontology_path}"
            )
        g_obo = SimpleOBOParser.parse(config.ontology_path)
        adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for parent_str, child_str in g_obo.edges():
            if parent_str in name_to_idx and child_str in name_to_idx:
                adjacency_matrix[name_to_idx[child_str],
                                 name_to_idx[parent_str]] = 1.0
    else:
        pure_adj = viz_data.get('pure_adjacency_matrix')
        if pure_adj is None:
            raise ValueError(
                "Model checkpoint does not contain 'pure_adjacency_matrix'."
            )
        adjacency_matrix = pure_adj

    # Active subgraph + min_cells_number pruning, shared with visualize().
    canonical_context = _build_canonical_trajectory_context(
        adjacency_matrix=adjacency_matrix,
        node_names=node_names,
        predicted_indices=predictions,
        min_cells_number=config.min_cells_number,
    )
    active_indices = list(canonical_context["active_indices"])
    node_counts = dict(canonical_context["node_counts"])
    keep_mask = np.asarray(canonical_context["keep_mask"], dtype=bool)

    # GRIT-refined anchors restricted to surviving cells (mirror
    # visualize()'s `refined_predictions` exactly).
    if use_grit:
        prediction_indices = predictions[keep_mask].astype(np.int64)
    else:
        prediction_indices = None

    # ------------------------------------------------------------------
    # Soft / inferred path discovery — identical to visualize().
    # ------------------------------------------------------------------
    node_depths_dag = compute_node_depths_longest_path(adjacency_matrix)
    inferred_path_weights: Dict[Tuple[int, int], float] = {}
    lateral_directions: Dict[Tuple[int, int], Tuple[int, int]] = {}
    augmented_adjacency = adjacency_matrix

    gat_embeddings = viz_data.get('gat_embeddings')
    ontology_adj = viz_data.get('ontology_adj')
    id_to_name_map = getattr(predictor, 'id_to_name_map', {})

    # Inferred-path discovery weights come from the INFERRED_PATH_* constants above
    # (SoftEdgeConfig defaults) — the single tuning surface.
    if (
        config.enable_inferred_paths
        and gat_embeddings is not None
        and ontology_adj is not None
    ):
        inferred_path_config = SoftEdgeConfig(enable_inferred_paths=True)
        try:
            augmented_adjacency, inferred_path_weights, lateral_directions = (
                discover_inferred_paths(
                    pure_ontology_adj=adjacency_matrix,
                    ontology_adj=ontology_adj,
                    node_embeddings=gat_embeddings,
                    config=inferred_path_config,
                    cache=cache,
                )
            )
            # Direction filter — semantic + embedding-gradient + depth rules.
            direction_filtered, _dir_counters = _apply_direction_filter(
                inferred_path_weights=inferred_path_weights,
                lateral_directions=lateral_directions,
                node_depths_dag=node_depths_dag,
                node_names=node_names,
                id_to_name_map=id_to_name_map,
            )
            inferred_path_weights = direction_filtered
        except Exception as exc:  # pragma: no cover — defensive
            _logging.getLogger(__name__).warning(
                "Inferred-path discovery failed (%s); proceeding with "
                "canonical edges only.", exc,
            )
            inferred_path_weights = {}
            lateral_directions = {}
            augmented_adjacency = adjacency_matrix

    # Build the placer's edge list to match what visualize() passes:
    # canonical (parent, child) edges from the active subgraph plus
    # any inferred-path edges that connect active nodes.
    # `canonical_context['tree_edges']` is the canonical edge list
    # used by visualize() (via the layout engine in horizontal mode).
    canonical_tree_edges = list(canonical_context['tree_edges'])
    tree_edges = _build_visual_edges(
        canonical_edges=canonical_tree_edges,
        inferred_path_weights=inferred_path_weights,
        active_indices=active_indices,
    )

    # Filter `inferred_path_weights` exactly the way `visualize()`
    # does (trajectory.py:1355-1360): remove any inferred edges that
    # overlap a canonical edge (in either direction).  The placer
    # uses the filtered set to decide which edges are "soft / lateral"
    # — if a canonical edge slipped into the inferred set, the
    # placer's edge_direction would treat it as an inferred edge and
    # consult lateral_directions instead of geometry, which can flip
    # its (src, tgt) orientation.
    canonical_edges_set: Set[Tuple[int, int]] = set(canonical_tree_edges)
    canonical_edges_set.update(
        (v, u) for u, v in canonical_tree_edges
    )
    placer_inferred_path_weights = {
        edge: weight
        for edge, weight in inferred_path_weights.items()
        if edge not in canonical_edges_set
        and (edge[1], edge[0]) not in canonical_edges_set
    }

    return {
        'adjacency_matrix': adjacency_matrix,
        'augmented_adjacency': augmented_adjacency,
        'active_indices': active_indices,
        'tree_edges': tree_edges,
        'lateral_directions': lateral_directions,
        'inferred_path_weights': placer_inferred_path_weights,
        'inferred_path_weights_unfiltered': inferred_path_weights,
        'node_depths_dag': node_depths_dag,
        'prediction_indices': prediction_indices,
        'cell_embeddings': cell_vectors[keep_mask],
        'node_embeddings': node_vectors,
        'node_names': list(node_names),
        'keep_mask': keep_mask,
        'node_counts': node_counts,
    }
