"""Trajectory support components for HECTOR.

This module holds the large internal building blocks used by the public
trajectory API: geometry helpers, overlap resolution, atypical-cell
placement, and clustering.
"""

from __future__ import annotations

import colorsys
import base64
import inspect
import io
import numpy as np
import os
import pandas as pd
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# Visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# SciPy for dendrogram layout
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.sparse import csr_matrix

# NetworkX for graph operations
import networkx as nx

# HECTOR Elastic River imports
from .trajectory_ontology import (
    discover_inferred_paths,
    compute_elastic_depths,
    get_edge_style,
    SoftEdgeConfig,
    ElasticLayoutConfig,
)


# =============================================================================
# Hierarchical component-filter defaults
# =============================================================================
# Function signatures and plot fallbacks reference these shared constants.

#: Axiom 1 — Mass Principle: minimum component weight.
_DEFAULT_MIN_COMPONENT_WEIGHT: float = 0.035

#: Axiom 2 — Cohesion Principle: maximum component standard deviation.
_DEFAULT_MAX_COMPONENT_STD: float = 0.08

#: Axiom 3 — Relevance Principle: mean threshold for divergence ("upper") gate.
_DEFAULT_MIN_COMPONENT_MEAN: float = 0.618


# =============================================================================
# Pie Chart Overlap Resolution Constants
# =============================================================================
# Shared sizing constants (layout- and backend-independent).
PIE_CHART_EXTERIOR_MARGIN_FRACTION = 0.02  # 2% of plot height above tree for exterior pies
PIE_CHART_OVERLAP_MARGIN = 0.1             # spacing between pies (fraction of combined radii)

# Pixel-space sizing constants (target display pixels, backend- and
# layout-independent). Consumed by the renderers via ``PixelGeometry``
# conversions in trajectory_render.py.
PIE_RADIUS_MAX_PX = 28.0      # largest cluster pie
PIE_RADIUS_MIN_PX = 12.0      # smallest cluster pie
CLOUD_BASE_RADIUS_PX = 14.0   # 1σ jitter radius at scale_factor=1.0
RIBBON_BASE_HW_PX = 2.2       # canonical ribbon half-width at edge_width=1.5
SOFT_ARROW_LENGTH_PX = 10.0   # soft-edge arrow shaft length
SOFT_ARROW_SPACING_PX = 45.0  # gap between consecutive soft-edge arrows

# Backend-specific visual-calibration multipliers applied on top of the
# base pixel constants above. They absorb residual differences between
# CSS pixels (HTML @ ~96 DPI) and matplotlib display pixels (PDF @ 400
# DPI typically viewed at 100% zoom), plus per-glyph calibration.
PIE_RADIUS_HTML_SCALE = 1.0
PIE_RADIUS_PDF_SCALE = 5.0           # PDFs at 100% zoom need markedly larger pies
CLOUD_RADIUS_HTML_SCALE = 0.8        # HTML clouds 20% smaller for tighter reading
CLOUD_RADIUS_PDF_SCALE = 3         # PDF clouds 300% bigger for publication scale
HIGHLIGHT_MARKER_PDF_SCALE = 0.3     # PDF highlight scatter dots half size
HIGHLIGHT_MARKER_HTML_SCALE = 1.0

def compute_adaptive_pdf_ribbon_scale(
    node_spacing_px: float,
    target_fraction: float = 0.25,
    min_scale: float = 0.3,
    max_scale: float = 8.0,
) -> float:
    """Compute PDF ribbon width scale so ribbons stay proportional to node spacing."""
    ribbon_full_width_px = 2.0 * RIBBON_BASE_HW_PX
    scale = target_fraction * node_spacing_px / ribbon_full_width_px
    return max(min_scale, min(max_scale, scale))


def compute_data_diagonal(node_positions, active_indices):
    """Return the diagonal span of the active node positions."""
    from .trajectory_render import compute_data_diagonal as _compute_data_diagonal

    return _compute_data_diagonal(node_positions, active_indices)


def compute_pie_radius_px(cell_count: int, max_count: int) -> float:
    """Map a cluster cell count to its pixel-space pie radius.

    Uses a square-root count ratio so the radius remains between
    ``PIE_RADIUS_MIN_PX`` and ``PIE_RADIUS_MAX_PX``.
    """
    if max_count <= 0:
        ratio = 1.0
    else:
        ratio = max(0.0, min(1.0, float(cell_count) / float(max_count)))
    return PIE_RADIUS_MIN_PX + (PIE_RADIUS_MAX_PX - PIE_RADIUS_MIN_PX) * np.sqrt(ratio)


def compute_count_scaled_cloud_scale(
    count: int,
    max_count: int,
    scale_cloud_by_count: bool,
    cloud_scale_exponent: float,
    cloud_size_multiplier: float,
    cloud_size_min: float,
    cloud_size_max: float,
) -> float:
    """Return the placement radius multiplier for one stable cloud.

    Semantic:
    - ``scale_cloud_by_count=False`` -> returns ``cloud_size_multiplier`` as-is
      (no count-based scaling).
    - Otherwise, the multiplier is **linearly interpolated between
      ``cloud_size_min`` and ``cloud_size_max``** by a ``sqrt(ratio)`` curve.
      Smallest group (ratio→0) gets ``cloud_size_min``; largest group
      (ratio→1, i.e. ``count == max_count``) gets ``cloud_size_max``.
      ``cloud_scale_exponent`` shapes the curve. With exponent 1, the varying
      portion follows ``sqrt(count / max_count)``; the nonzero minimum means
      total cloud area is not strictly proportional to count.
    - ``cloud_size_multiplier`` is applied as a global post-multiplier so
      users can rescale everything uniformly without touching min/max.
    """
    if not scale_cloud_by_count or max_count <= 0:
        return float(cloud_size_multiplier)

    ratio = max(0.0, min(1.0, float(count) / float(max_count)))
    # shaped ∈ [0, 1]: 0 at count=0 (or below threshold), 1 at count=max_count
    shaped = float(np.sqrt(ratio ** (1.0 / float(cloud_scale_exponent))))
    scale = float(cloud_size_min) + (float(cloud_size_max) - float(cloud_size_min)) * shaped
    return float(cloud_size_multiplier) * scale


# =============================================================================
# Geometry Helpers (Shared between edge drawing and cell placement)
# =============================================================================

def calculate_bezier_point(t: float, start: Tuple[float, float], end: Tuple[float, float], curvature: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate point P(t) on a cubic Bezier curve for HORIZONTAL tree layout.
    Also returns the normal vector for perpendicular jittering.
    
    Uses step-curve control points: horizontal extension then vertical drop.
    This matches the River Delta layout style.
    
    Args:
        t: Parameter from 0 (start) to 1 (end)
        start: (x, y) position of start node
        end: (x, y) position of end node
        curvature: Not used in horizontal mode, kept for API compatibility
        
    Returns:
        (point, normal): The curve point and the perpendicular normal vector
    """
    p0 = np.array(start)
    p3 = np.array(end)
    
    # Use control points that make the curve horizontal at both endpoints.
    # 
    # Logic:
    # 1. Start Node: P0 = (x_start, y_start)
    # 2. End Node:   P3 = (x_end, y_end)
    # 3. Control Point 1: P1 = (mid_x, y_start)  <-- Y locked to Start (dy/dx = 0)
    # 4. Control Point 2: P2 = (mid_x, y_end)    <-- Y locked to End (dy/dx = 0)
    #
    # Vertical curvature is concentrated between hierarchy levels.
    
    mid_x = (p0[0] + p3[0]) / 2
    
    p1 = np.array([mid_x, p0[1]])  # Horizontal extension from start (Tangent is horizontal)
    p2 = np.array([mid_x, p3[1]])  # Vertical drop/rise to end (Tangent is horizontal)
    
    # Cubic Bezier Formula: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
    # Note: We calculate this manually for efficiency
    
    one_minus_t = 1 - t
    point = (
        one_minus_t**3 * p0 + 
        3 * one_minus_t**2 * t * p1 + 
        3 * one_minus_t * t**2 * p2 + 
        t**3 * p3
    )
    
    # Calculate Tangent & Normal for perpendicular jitter
    # We use analytical derivative for better precision than finite difference
    # B'(t) = 3(1-t)²(P1-P0) + 6(1-t)t(P2-P1) + 3t²(P3-P2)
    tangent = (
        3 * one_minus_t**2 * (p1 - p0) +
        6 * one_minus_t * t * (p2 - p1) +
        3 * t**2 * (p3 - p2)
    )
    
    len_tan = np.linalg.norm(tangent)
    
    if len_tan < 1e-9:
        normal = np.array([0.0, 1.0])  # Fallback: vertical normal
    else:
        unit_tan = tangent / len_tan
        normal = np.array([-unit_tan[1], unit_tan[0]])  # Rotate 90°
        
    return point, normal


def calculate_radial_bezier_curve(
    start_theta: float,
    start_r: float,
    end_theta: float,
    end_r: float,
    num_points: int = 30,
    curvature_scale: float = 1.0,
) -> np.ndarray:
    """Calculate Bezier curve points for radial tree layout.

    Uses Cartesian-space cubic Bezier with control points projecting
    outward along each node's radial angle, then converts back to polar.

    Args:
        start_theta: Starting angle (radians).
        start_r: Starting radius.
        end_theta: Ending angle (radians).
        end_r: Ending radius.
        num_points: Number of points to generate along curve.
        curvature_scale: Multiplier on the default curvature factor (0.384 * dist).
            Values < 1.0 flatten the curve; 1.0 preserves the original shape.

    Returns:
        np.ndarray: Array of shape (num_points, 2) with (theta, r) coordinates.
    """
    start_x = start_r * np.cos(start_theta)
    start_y = start_r * np.sin(start_theta)
    end_x = end_r * np.cos(end_theta)
    end_y = end_r * np.sin(end_theta)

    start_pt = np.array([start_x, start_y])
    end_pt = np.array([end_x, end_y])
    dist = np.linalg.norm(end_pt - start_pt)
    curvature_factor = 0.384 * dist * curvature_scale

    cp1_x = start_x + np.cos(start_theta) * curvature_factor
    cp1_y = start_y + np.sin(start_theta) * curvature_factor
    cp2_x = end_x - np.cos(end_theta) * curvature_factor
    cp2_y = end_y - np.sin(end_theta) * curvature_factor

    t = np.linspace(0, 1, num_points)
    curve_x = (1-t)**3*start_x + 3*(1-t)**2*t*cp1_x + 3*(1-t)*t**2*cp2_x + t**3*end_x
    curve_y = (1-t)**3*start_y + 3*(1-t)**2*t*cp1_y + 3*(1-t)*t**2*cp2_y + t**3*end_y

    curve_r = np.sqrt(curve_x**2 + curve_y**2)
    curve_theta = np.arctan2(curve_y, curve_x)

    return np.column_stack([curve_theta, curve_r])


def circular_arc_from_chord_cartesian(
    A: Tuple[float, float], 
    B: Tuple[float, float], 
    curvature: float = 0.3
) -> Tuple[np.ndarray, float, float, float]:
    """
    Compute center, radius, and angles for a perfect circular arc through A and B.
    
    This is adapted from arc.py for use in radial inferred path visualization.
    The arc passes through both endpoints with curvature controlling the bulge.
    
    The curvature parameter controls the arc's bulge:
    - curvature = 0: Straight line (degenerate arc)
    - curvature > 0: Arc bulges to the left (counter-clockwise) of the chord
    - curvature < 0: Arc bulges to the right (clockwise) of the chord
    
    The magnitude of curvature controls how much the arc bulges. The sagitta
    (perpendicular distance from chord midpoint to arc midpoint) is:
        sagitta = curvature * chord_length
    
    Args:
        A: Start point (x, y) in Cartesian coordinates
        B: End point (x, y) in Cartesian coordinates  
        curvature: Controls arc bulge. 0 = straight line, >0 = bulge left (CCW), <0 = bulge right (CW)
        
    Returns:
        Tuple of (center, radius, start_angle, end_angle):
        - center: np.ndarray [x, y] of circle center
        - radius: float, radius of the circle
        - start_angle: float, angle from center to point A (radians)
        - end_angle: float, angle from center to point B (radians)
        
    Raises:
        ValueError: If A and B are identical (cannot define arc)
        
    Example:
        >>> A = (0.0, 0.0)
        >>> B = (1.0, 0.0)
        >>> center, radius, a0, a1 = circular_arc_from_chord_cartesian(A, B, curvature=0.5)
        >>> # Arc bulges upward (left of chord direction) with sagitta = 0.5 * 1.0 = 0.5
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    
    # Safety check: identical points cannot define an arc
    if np.allclose(A, B, atol=1e-9):
        raise ValueError("Start and end points A and B cannot be identical.")
    
    # Chord vector and length
    chord_vec = B - A
    chord_length = np.linalg.norm(chord_vec)
    
    # Handle very close points (numerical stability)
    if chord_length < 1e-6:
        raise ValueError(f"Points A and B are too close (distance={chord_length:.2e}). Cannot compute stable arc.")
    
    # Chord midpoint
    midpoint = (A + B) / 2.0
    
    # Perpendicular unit vector (rotate chord 90° counter-clockwise)
    # For chord vector (dx, dy), perpendicular is (-dy, dx)
    perp_unit = np.array([-chord_vec[1], chord_vec[0]]) / chord_length
    
    # Sagitta: perpendicular offset from chord midpoint to arc midpoint
    # Positive curvature -> offset in perp_unit direction (left/CCW)
    # Negative curvature -> offset opposite to perp_unit (right/CW)
    sagitta = curvature * chord_length
    
    # Circle center: offset from chord midpoint by sagitta
    center = midpoint + perp_unit * sagitta
    
    # Perfect circle radius: distance from center to either endpoint
    radius = np.linalg.norm(A - center)
    
    # Start and end angles in circle coordinates
    start_angle = np.arctan2(A[1] - center[1], A[0] - center[0])
    end_angle = np.arctan2(B[1] - center[1], B[0] - center[0])
    
    # Ensure correct angular direction based on curvature sign
    # Positive curvature -> counter-clockwise arc (end_angle > start_angle)
    # Negative curvature -> clockwise arc (end_angle < start_angle)
    if curvature > 0:
        # Counter-clockwise: ensure end_angle > start_angle
        if end_angle < start_angle:
            end_angle += 2 * np.pi
    else:
        # Clockwise: ensure end_angle < start_angle
        if end_angle > start_angle:
            start_angle += 2 * np.pi
    
    return center, radius, start_angle, end_angle


def generate_circular_arc_points(
    A: Tuple[float, float],
    B: Tuple[float, float],
    curvature: float = 0.3,
    num_points: int = 21
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate points and tangent normals along a circular arc from A to B.
    
    Used for both cell placement (edge_paths) and glyph rendering.
    Returns points along the arc and perpendicular normal vectors at each point.
    
    Args:
        A: Start point (x, y) in Cartesian coordinates
        B: End point (x, y) in Cartesian coordinates
        curvature: Arc curvature parameter (see circular_arc_from_chord_cartesian)
        num_points: Number of points to generate along arc
        
    Returns:
        Tuple of (points, normals):
        - points: np.ndarray of shape (num_points, 2) with (x, y) coordinates
        - normals: np.ndarray of shape (num_points, 2) with perpendicular normal vectors
        
    Example:
        >>> A = (0.0, 0.0)
        >>> B = (1.0, 0.0)
        >>> points, normals = generate_circular_arc_points(A, B, curvature=0.3, num_points=11)
        >>> points.shape
        (11, 2)
        >>> normals.shape
        (11, 2)
    """
    # Handle degenerate case: zero curvature -> straight line
    if abs(curvature) < 1e-9:
        # Generate linear interpolation
        t = np.linspace(0, 1, num_points)
        A_arr = np.asarray(A, dtype=float)
        B_arr = np.asarray(B, dtype=float)
        
        points = A_arr[np.newaxis, :] + t[:, np.newaxis] * (B_arr - A_arr)[np.newaxis, :]
        
        # Normal is perpendicular to line direction
        line_vec = B_arr - A_arr
        line_length = np.linalg.norm(line_vec)
        
        if line_length < 1e-9:
            # Degenerate: A == B
            normal = np.array([0.0, 1.0])
        else:
            # Perpendicular: rotate 90° counter-clockwise
            normal = np.array([-line_vec[1], line_vec[0]]) / line_length
        
        normals = np.tile(normal, (num_points, 1))
        return points, normals
    
    # Compute circular arc geometry
    try:
        center, radius, start_angle, end_angle = circular_arc_from_chord_cartesian(A, B, curvature)
    except ValueError as e:
        # Fallback to straight line if arc computation fails
        A_arr = np.asarray(A, dtype=float)
        B_arr = np.asarray(B, dtype=float)
        t = np.linspace(0, 1, num_points)
        points = A_arr[np.newaxis, :] + t[:, np.newaxis] * (B_arr - A_arr)[np.newaxis, :]
        
        line_vec = B_arr - A_arr
        line_length = np.linalg.norm(line_vec)
        if line_length < 1e-9:
            normal = np.array([0.0, 1.0])
        else:
            normal = np.array([-line_vec[1], line_vec[0]]) / line_length
        normals = np.tile(normal, (num_points, 1))
        return points, normals
    
    # Generate points along the arc
    angles = np.linspace(start_angle, end_angle, num_points)
    
    # Compute arc points in Cartesian coordinates
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    points = np.column_stack([xs, ys])
    
    # Compute tangent normals at each point
    # Use finite differences to compute tangent vectors from arc points
    tangents = np.zeros_like(points)
    
    # Forward difference for first point
    tangents[0] = points[1] - points[0]
    
    # Central difference for middle points
    for i in range(1, len(points) - 1):
        tangents[i] = points[i+1] - points[i-1]
    
    # Backward difference for last point
    tangents[-1] = points[-1] - points[-2]
    
    # Normalize tangent vectors
    tangent_lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent_lengths[tangent_lengths == 0] = 1.0  # Avoid division by zero
    tangents = tangents / tangent_lengths
    
    # Normal is perpendicular to tangent (rotate tangent 90° CCW)
    # If tangent is (tx, ty), normal is (-ty, tx)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    
    return points, normals


# =============================================================================
# Visual Edge Builder (Soft Edge Integration)
# =============================================================================

def _build_visual_edges(
    canonical_edges: List[Tuple[int, int]],
    inferred_path_weights: Dict[Tuple[int, int], float],
    active_indices: List[int]
) -> List[Tuple[int, int]]:
    """
    Build visual edge list combining canonical and inferred paths.
    
    This function merges the canonical edges from the pure ontology DAG with
    inferred paths (data-driven transitions) for visualization. Only inferred paths
    that connect currently active nodes are included.
    
    Args:
        canonical_edges: Edges from pure ontology DAG (parent -> child)
        inferred_path_weights: Inferred paths with weights {(src, tgt): weight}
        active_indices: Currently active node indices in the visualization
    
    Returns:
        Combined edge list for visualization, containing all canonical edges
        plus inferred paths that connect active nodes (no duplicates)
    
    Example:
        >>> canonical = [(0, 1), (1, 2)]
        >>> soft_weights = {(0, 2): 0.8, (1, 3): 0.7}
        >>> active = [0, 1, 2]
        >>> _build_visual_edges(canonical, soft_weights, active)
        [(0, 1), (1, 2), (0, 2)]  # (1, 3) excluded since 3 not active
    """
    active_set = set(active_indices)
    visual_edges = list(canonical_edges)
    
    # Add inferred paths that connect active nodes
    for (src, tgt), weight in inferred_path_weights.items():
        if src in active_set and tgt in active_set:
            edge = (src, tgt)
            if edge not in visual_edges:
                visual_edges.append(edge)
    
    return visual_edges


class PieChartOverlapResolver:
    """Convenience wrapper for resolving pie chart overlaps with coordinate conversion."""

    @staticmethod
    def resolve_for_renderer(
        clusters: List['AtypicalClusterInfo'],
        node_positions: Dict[int, Tuple[float, float]],
        is_polar: bool,
        margin: float = PIE_CHART_OVERLAP_MARGIN,
        pixel_geometry: Optional[Any] = None,
    ) -> None:
        """
        Resolve pie chart overlaps for a renderer.

        Handles coordinate conversion automatically based on layout mode.
        Collision resolution runs in Cartesian coordinates; when
        ``pixel_geometry`` is provided, the Cartesian coordinates are
            pixel-space so pies are isotropic circles of ``cluster.radius_px``
        regardless of axis anisotropy. Modifies cluster positions and
        ``is_exterior`` flags in-place.

        Args:
            clusters: ``AtypicalClusterInfo`` objects. Pixel-space mode uses
                ``.radius_px``; data-space mode uses ``.radius``.
            node_positions: Dict mapping node_idx to (coord0, coord1).
            is_polar: True for radial layout, False for horizontal layout.
            margin: Minimum spacing between pies as fraction of combined radii.
            pixel_geometry: Optional ``PixelGeometry`` (defined in
                ``trajectory_render``). When supplied, the resolver runs in
                pixel space using each cluster's ``radius_px``. Left ``None``
                to use data-space radii, including radial layouts whose axes
                are already isotropic.
        """
        if not clusters:
            return

        # Extract positions and radii from clusters
        positions = np.array([c.position for c in clusters], dtype=float)
        if pixel_geometry is not None:
            radii = np.array(
                [getattr(c, 'radius_px', 0.0) or 0.0 for c in clusters],
                dtype=float,
            )
        else:
            radii = np.array([c.radius for c in clusters], dtype=float)

        # Extract barycenters (or use positions if not set)
        barycenters = np.array([
            c.barycenter if c.barycenter is not None else c.position
            for c in clusters
        ], dtype=float)

        # Convert node positions to array
        if len(node_positions) > 0:
            node_pos_array = np.array(list(node_positions.values()), dtype=float)
        else:
            node_pos_array = np.array([]).reshape(0, 2)

        # Convert everything to Cartesian for calculations
        if is_polar:
            # Convert node positions (theta, r) to (x, y)
            if len(node_pos_array) > 0:
                theta = node_pos_array[:, 0]
                r = node_pos_array[:, 1]
                node_pos_cartesian = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            else:
                node_pos_cartesian = node_pos_array

            # Convert cluster positions to Cartesian
            cluster_theta = positions[:, 0]
            cluster_r = positions[:, 1]
            positions_cartesian = np.column_stack([
                cluster_r * np.cos(cluster_theta),
                cluster_r * np.sin(cluster_theta)
            ])

            # Convert barycenters to Cartesian
            bary_theta = barycenters[:, 0]
            bary_r = barycenters[:, 1]
            barycenters_cartesian = np.column_stack([
                bary_r * np.cos(bary_theta),
                bary_r * np.sin(bary_theta)
            ])
        else:
            node_pos_cartesian = node_pos_array
            positions_cartesian = positions
            barycenters_cartesian = barycenters

        # Store Cartesian barycenters in clusters (for anchor lines).
        # Barycenters stay in data coordinates — the renderer uses them to
        # draw leader lines back to the original tree position.
        for i, cluster in enumerate(clusters):
            cluster.barycenter = tuple(barycenters_cartesian[i])

        if pixel_geometry is not None:
            # Transform Cartesian positions into pixel coordinates so the
            # force-directed resolver operates on isotropic circles.
            def _to_px(xy: np.ndarray) -> np.ndarray:
                if len(xy) == 0:
                    return xy
                return np.column_stack([
                    (xy[:, 0] - pixel_geometry.x_min) * pixel_geometry.px_per_data_x(),
                    (xy[:, 1] - pixel_geometry.y_min) * pixel_geometry.px_per_data_y(),
                ])

            def _from_px(xy_px: np.ndarray) -> np.ndarray:
                if len(xy_px) == 0:
                    return xy_px
                return np.column_stack([
                    xy_px[:, 0] * pixel_geometry.data_per_px_x() + pixel_geometry.x_min,
                    xy_px[:, 1] * pixel_geometry.data_per_px_y() + pixel_geometry.y_min,
                ])

            positions_for_resolver = _to_px(positions_cartesian)
            nodes_for_resolver = _to_px(node_pos_cartesian)
        else:
            positions_for_resolver = positions_cartesian
            nodes_for_resolver = node_pos_cartesian

        # Call core overlap resolution (Cartesian; pixel coords when geom is supplied)
        resolved_positions, is_exterior = resolve_pie_chart_overlaps(
            positions_for_resolver,
            radii,
            nodes_for_resolver,
            margin
        )

        # Unwind pixel-space transform before the polar conversion, so the
        # downstream steps see the same coordinate system they started with.
        if pixel_geometry is not None:
            resolved_positions = _from_px(resolved_positions)

        # Convert resolved positions back to polar if needed
        if is_polar:
            resolved_r = np.sqrt(resolved_positions[:, 0]**2 + resolved_positions[:, 1]**2)
            resolved_theta = np.arctan2(resolved_positions[:, 1], resolved_positions[:, 0])
            resolved_positions_final = np.column_stack([resolved_theta, resolved_r])
        else:
            resolved_positions_final = resolved_positions

        # Update clusters in-place
        for i, cluster in enumerate(clusters):
            cluster.position = tuple(resolved_positions_final[i])
            cluster.is_exterior = bool(is_exterior[i])


def resolve_pie_chart_overlaps(
    positions: np.ndarray,
    radii: np.ndarray,
    node_positions: np.ndarray,
    margin: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduce pie-chart overlaps using force-directed simulation.
    
    All inputs must be in Cartesian coordinates (x, y), and radii must use the
    same data units as the positions.
    
    Args:
        positions: (n_pies, 2) array of pie positions in Cartesian coordinates
        radii: (n_pies,) array of pie radii in data units.
        node_positions: (n_nodes, 2) array of tree node positions in Cartesian coordinates
        margin: Minimum spacing between pies as fraction of combined radii (default: 0.2)
        
    Returns:
        Tuple of (updated_positions, is_exterior_flags):
        - updated_positions: (n_pies, 2) array of resolved positions
        - is_exterior_flags: (n_pies,) boolean array, True if pie is exterior
    
    Raises:
        ValueError: If array shapes don't match or if radii/margin are invalid
    """
    # Input validation
    if len(positions) == 0:
        return positions.copy(), np.array([], dtype=bool)
    
    positions = np.asarray(positions, dtype=float)
    radii = np.asarray(radii, dtype=float)
    node_positions = np.asarray(node_positions, dtype=float)
    
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must have shape (n_pies, 2), got {positions.shape}")
    
    if radii.ndim != 1 or len(radii) != len(positions):
        raise ValueError(f"radii must have shape ({len(positions)},), got {radii.shape}")
    
    if len(node_positions) > 0 and (node_positions.ndim != 2 or node_positions.shape[1] != 2):
        raise ValueError(f"node_positions must have shape (n_nodes, 2), got {node_positions.shape}")
    
    if np.any(radii <= 0):
        raise ValueError("All radii must be positive")
    
    if margin < 0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    
    n_pies = len(positions)
    
    # Compute plot bounds from node positions
    if len(node_positions) > 0:
        plot_x_min, plot_x_max = np.min(node_positions[:, 0]), np.max(node_positions[:, 0])
        plot_y_min, plot_y_max = np.min(node_positions[:, 1]), np.max(node_positions[:, 1])
    else:
        plot_x_min, plot_x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
        plot_y_min, plot_y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
    
    plot_width = max(plot_x_max - plot_x_min, 1e-6)
    plot_height = max(plot_y_max - plot_y_min, 1e-6)
    
    # Store original positions as barycenters (for gravity toward original location)
    barycenters = positions.copy()
    updated_positions = positions.copy()
    is_exterior = np.zeros(n_pies, dtype=bool)
    
    # Clamp initial positions to be within plot bounds (with small margin)
    clamp_margin = max(plot_width, plot_height) * 0.1
    for i in range(n_pies):
        updated_positions[i, 0] = np.clip(updated_positions[i, 0], 
                                          plot_x_min - clamp_margin, 
                                          plot_x_max + clamp_margin)
        updated_positions[i, 1] = np.clip(updated_positions[i, 1], 
                                          plot_y_min - clamp_margin, 
                                          plot_y_max + clamp_margin)
    
    # Force simulation parameters
    gravity_strength = 0.05
    repulsion_strength = 1.5
    damping = 0.7
    max_iterations = 200
    convergence_threshold = 1e-5
    
    avg_radius = np.mean(radii)
    velocities = np.zeros_like(updated_positions)
    
    # Force-directed simulation
    for iteration in range(max_iterations):
        forces = np.zeros_like(updated_positions)
        
        for i in range(n_pies):
            # Gravity toward original barycenter (weak)
            displacement = barycenters[i] - updated_positions[i]
            if np.linalg.norm(displacement) > avg_radius * 3:
                forces[i] += gravity_strength * displacement
            
            # Repulsion from tree nodes
            for j in range(len(node_positions)):
                diff = updated_positions[i] - node_positions[j]
                distance = np.linalg.norm(diff)
                if distance < 1e-6:
                    angle = (i * 11 + j * 17) * 0.5
                    diff = np.array([np.cos(angle), np.sin(angle)])
                    distance = 1e-6
                
                clearance = radii[i] * (1 + margin)
                if distance < clearance:
                    overlap = clearance - distance
                    force_multiplier = 1.0 + (overlap / clearance) * 2.0
                    forces[i] += repulsion_strength * force_multiplier * overlap * (diff / distance)
            
            # Repulsion from other pies
            for j in range(n_pies):
                if i != j:
                    diff = updated_positions[i] - updated_positions[j]
                    distance = np.linalg.norm(diff)
                    if distance < 1e-6:
                        angle = (i * 7 + j * 13) * 0.5
                        diff = np.array([np.cos(angle), np.sin(angle)])
                        distance = 1e-6
                    
                    min_separation = (radii[i] + radii[j]) * (1 + margin)
                    if distance < min_separation:
                        overlap = min_separation - distance
                        force_multiplier = 1.0 + (overlap / min_separation) * 3.0
                        forces[i] += repulsion_strength * force_multiplier * overlap * (diff / distance)
        
        # Update velocities and positions
        velocities = damping * velocities + forces
        updated_positions += velocities

        # Hard-clamp positions strictly within plot bounds after each step.
        # Pies must stay inside the tree area — any overhang would cause
        # matplotlib/Plotly to auto-expand the axes, compressing the tree.
        # If a pie genuinely can't fit inside, the post-simulation is_exterior
        # check will flag it for placement in the designated exterior zone.
        for i in range(n_pies):
            updated_positions[i, 0] = np.clip(
                updated_positions[i, 0],
                plot_x_min,
                plot_x_max
            )
            updated_positions[i, 1] = np.clip(
                updated_positions[i, 1],
                plot_y_min,
                plot_y_max
            )

        # Convergence check
        if np.max(np.linalg.norm(forces, axis=1)) < convergence_threshold:
            break
    
    # Exterior detection: mark pies as exterior if they have severe overlaps
    # that couldn't be resolved by the force simulation, OR if they're outside plot bounds
    max_radius = np.max(radii)
    
    for i in range(n_pies):
        px, py = updated_positions[i]
        
        # Check if outside plot bounds (with small margin for edge cases)
        bound_margin = max_radius * 0.5
        outside_bounds = (px < plot_x_min - bound_margin or 
                         px > plot_x_max + bound_margin or
                         py < plot_y_min - bound_margin or 
                         py > plot_y_max + bound_margin)
        # Check for severe remaining overlaps with nodes (more than 50% overlap)
        severe_node_overlap = False
        for j in range(len(node_positions)):
            dist = np.linalg.norm(updated_positions[i] - node_positions[j])
            if dist < radii[i] * 0.5:  # More than 50% overlap with node
                severe_node_overlap = True
                break
        
        # Check for severe remaining overlaps with other pies (more than 50% overlap)
        severe_pie_overlap = False
        for j in range(n_pies):
            if i != j:
                dist = np.linalg.norm(updated_positions[i] - updated_positions[j])
                min_sep = (radii[i] + radii[j]) * 0.5  # 50% of combined radii
                if dist < min_sep:
                    severe_pie_overlap = True
                    break
        
        # Mark as exterior if outside bounds OR severe overlap that couldn't be resolved
        is_exterior[i] = outside_bounds or severe_node_overlap or severe_pie_overlap
    
    # Exterior placement: place exterior pies ABOVE the tree (higher Y values)
    exterior_indices = np.where(is_exterior)[0]
    
    if len(exterior_indices) > 0:
        # Sort by original x-coordinate for consistent ordering
        exterior_sorted = sorted(exterior_indices, key=lambda i: barycenters[i, 0])
        
        # Place ABOVE the tree with a small margin
        max_ext_radius = np.max(radii[exterior_indices])
        exterior_margin = plot_height * PIE_CHART_EXTERIOR_MARGIN_FRACTION
        y_exterior = plot_y_max + exterior_margin + max_ext_radius
        
        n_exterior = len(exterior_sorted)
        if n_exterior == 1:
            x_positions = [(plot_x_min + plot_x_max) / 2]
        else:
            # Compute total width needed with proper spacing
            total_spacing = sum(
                (radii[exterior_sorted[k]] + radii[exterior_sorted[k+1]]) * (1 + margin)
                for k in range(n_exterior - 1)
            )
            edge_margin = max_ext_radius * (1 + margin)
            total_width = total_spacing + 2 * edge_margin
            actual_width = min(plot_width * 0.9, total_width)  # Don't exceed 90% of plot width
            
            x_start = (plot_x_min + plot_x_max) / 2 - actual_width / 2 + edge_margin
            x_positions = [x_start]
            for k in range(n_exterior - 1):
                sep = (radii[exterior_sorted[k]] + radii[exterior_sorted[k+1]]) * (1 + margin)
                x_positions.append(x_positions[-1] + sep)
        
        for idx, pie_i in enumerate(exterior_sorted):
            updated_positions[pie_i] = np.array([x_positions[idx], y_exterior])
    
    return updated_positions, is_exterior





@dataclass
class AtypicalClustererConfig:
    """Configuration for atypical-cell clustering (HDBSCAN + anchor-based placement).

    Atypical-cell clusters are virtual hubs that group atypical (low-adherence)
    cells into distinct populations and place them in meaningful locations on
    the tree based on their affinity to known ontology nodes.
    """
    
    # Clustering parameters
    min_atypical_cluster_size: int = 30      # HDBSCAN min_cluster_size parameter
    min_atypical_for_clustering: int = 20  # Skip clustering if fewer atypical cells
    
    # Placement parameters
    atypical_anchor_count: int = 3           # Number of anchor nodes per atypical-cluster hub
    
    # Visualization parameters
    show_atypical_anchor_lines: bool = True  # Draw dashed lines to anchor nodes
    atypical_hub_color: str = 'rgba(128, 0, 128, 0.6)'  # Purple, semi-transparent
    atypical_hub_size: int = 10              # Size of atypical-cluster hub marker
    atypical_cloud_jitter: float = 1.0       # Jitter multiplier for cell positions around hub (reduced for tighter clouds)
    
    # Scaling for large clusters
    atypical_cloud_scale_threshold: int = 100  # Scale cloud radius above this cell count
    atypical_cloud_scale_factor: float = 0.5   # Radius ~ count^factor for large clusters
    # Pie Chart visualization is accessed directly from self.config (TrajectoryAnalysisConfig)

    def __post_init__(self):
        """Run validate() automatically at construction time."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.min_atypical_cluster_size < 5:
            raise ValueError(f"min_atypical_cluster_size must be >= 5 (HDBSCAN minimum), got {self.min_atypical_cluster_size}")
        
        if self.atypical_anchor_count < 1:
            raise ValueError(f"atypical_anchor_count must be >= 1, got {self.atypical_anchor_count}")
        
        if self.atypical_cloud_jitter <= 0:
            raise ValueError(f"atypical_cloud_jitter must be > 0, got {self.atypical_cloud_jitter}")
        
        if self.min_atypical_for_clustering < 1:
            raise ValueError(f"min_atypical_for_clustering must be >= 1, got {self.min_atypical_for_clustering}")


@dataclass
class AtypicalClusterInfo:
    """Information about a single cluster of atypical cells."""
    
    cluster_id: int                       # Cluster ID (0, 1, 2, ...)
    centroid_vector: np.ndarray           # 768-dim latent centroid of cluster
    cell_indices: np.ndarray              # Global indices of cells in this cluster
    anchor_nodes: List[int]               # Top-K anchor node indices (global)
    anchor_weights: np.ndarray            # Affinity weights for each anchor
    position: Tuple[float, float]         # 2D position on tree map
    cell_count: int                       # Number of cells in cluster
    cell_type_counts: Dict[int, int] = None  # {node_idx: count} - actual cells per type for pie chart
    radius: float = 0.0                   # Data-space y-half-extent of the rendered pie ellipse
    radius_px: float = 0.0                # Pixel-space pie radius (authoritative under pixel-space sizing)
    is_exterior: bool = False             # True if pie was pushed outside the tree bounding box
    barycenter: Tuple[float, float] = None  # Original anchor barycenter (for leader line when exterior)


@dataclass
class HighlightData:
    """Highlight overlay data passed to figure builders."""
    positions: np.ndarray          # (N, 2) cell positions (already cloud-jittered)
    categories: List[str]          # Category label per cell
    color_mapper: 'HighlightColorMapper'  # Shared color mapper




# =============================================================================
# Class-Relative Adherence Scoring Functions
# (Relocated from experimental_adherence.py)
# =============================================================================


def sigmoid_rescale_z_scores(z_scores: np.ndarray) -> np.ndarray:
    """Map z-scores to a bounded 0-1 range with a sigmoid."""

    z_scores = np.asarray(z_scores, dtype=np.float64)
    clipped = np.clip(z_scores, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def compute_lower_tail_scale(scores: np.ndarray, center: float) -> float:
    """Estimate spread from the lower half of the score distribution only."""

    scores = np.asarray(scores, dtype=np.float64)
    lower_scores = scores[scores <= center]
    if len(lower_scores) == 0:
        return np.nan

    lower_distances = center - lower_scores
    mad = float(np.median(lower_distances))
    return 1.4826 * mad


def align_scores_by_median_iqr(
    source_scores: np.ndarray,
    target_scores: np.ndarray,
    fit_mask: np.ndarray,
    apply_mask: np.ndarray,
    iqr_floor: float = 1e-4,
) -> Dict[str, object]:
    """Match source scores to the target score scale using median and IQR."""

    source_scores = np.asarray(source_scores, dtype=np.float64)
    target_scores = np.asarray(target_scores, dtype=np.float64)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    apply_mask = np.asarray(apply_mask, dtype=bool)

    if len(source_scores) != len(target_scores):
        raise ValueError("source_scores and target_scores must align.")
    if len(source_scores) != len(fit_mask):
        raise ValueError("source_scores and fit_mask must align.")
    if len(source_scores) != len(apply_mask):
        raise ValueError("source_scores and apply_mask must align.")

    aligned_scores = np.full(len(source_scores), np.nan, dtype=np.float64)
    fit_count = int(np.sum(fit_mask))

    if fit_count == 0:
        return {
            "aligned_scores": aligned_scores,
            "fit_count": fit_count,
            "alignment_method": "no_fit_cells",
            "source_median": np.nan,
            "source_iqr": np.nan,
            "target_median": np.nan,
            "target_iqr": np.nan,
            "scale_ratio": np.nan,
        }

    fit_source = source_scores[fit_mask]
    fit_target = target_scores[fit_mask]

    source_q25, source_median, source_q75 = np.quantile(fit_source, [0.25, 0.5, 0.75])
    target_q25, target_median, target_q75 = np.quantile(fit_target, [0.25, 0.5, 0.75])

    source_iqr = float(source_q75 - source_q25)
    target_iqr = float(target_q75 - target_q25)

    # When the source spread is nearly degenerate, a location-only correction is
    # safer than exploding the scale ratio.
    if source_iqr <= iqr_floor:
        aligned_values = source_scores[apply_mask] + (target_median - source_median)
        scale_ratio = np.nan
        alignment_method = "median_shift_only"
    elif target_iqr <= iqr_floor:
        aligned_values = np.full(np.sum(apply_mask), target_median, dtype=np.float64)
        scale_ratio = 0.0
        alignment_method = "target_iqr_degenerate"
    else:
        scale_ratio = target_iqr / source_iqr
        aligned_values = target_median + (
            (source_scores[apply_mask] - source_median) * scale_ratio
        )
        alignment_method = "median_iqr_match"

    aligned_scores[apply_mask] = np.clip(aligned_values, 0.0, 1.0)

    return {
        "aligned_scores": aligned_scores,
        "fit_count": fit_count,
        "alignment_method": alignment_method,
        "source_median": float(source_median),
        "source_iqr": source_iqr,
        "target_median": float(target_median),
        "target_iqr": target_iqr,
        "scale_ratio": float(scale_ratio) if np.isfinite(scale_ratio) else np.nan,
    }


def estimate_iterative_lower_tail_reference(
    class_scores: np.ndarray,
    min_class_size: int = 30,
    min_retained_size: int = 15,
    scale_floor: float = 1e-4,
    trim_z_threshold: float = -2.5,
) -> Dict[str, object]:
    """Estimate a robust within-class reference with one trim pass."""

    class_scores = np.asarray(class_scores, dtype=np.float64)
    n_query_cells = len(class_scores)

    if n_query_cells < min_class_size:
        return {
            "valid_self_reference": False,
            "fallback_reason": "class_too_small",
            "n_query_cells": n_query_cells,
            "n_retained_cells": n_query_cells,
            "robust_center": np.nan,
            "robust_lower_tail_scale": np.nan,
            "retained_mask": np.zeros(n_query_cells, dtype=bool),
            "provisional_z_scores": np.full(n_query_cells, np.nan, dtype=np.float64),
        }

    provisional_center = float(np.median(class_scores))
    provisional_scale = compute_lower_tail_scale(class_scores, provisional_center)

    if not np.isfinite(provisional_scale) or provisional_scale <= scale_floor:
        return {
            "valid_self_reference": False,
            "fallback_reason": "degenerate_class_scale",
            "n_query_cells": n_query_cells,
            "n_retained_cells": n_query_cells,
            "robust_center": np.nan,
            "robust_lower_tail_scale": np.nan,
            "retained_mask": np.zeros(n_query_cells, dtype=bool),
            "provisional_z_scores": np.full(n_query_cells, np.nan, dtype=np.float64),
        }

    provisional_z_scores = (class_scores - provisional_center) / provisional_scale
    retained_mask = provisional_z_scores >= trim_z_threshold
    n_retained_cells = int(np.sum(retained_mask))

    if n_retained_cells < min_retained_size:
        return {
            "valid_self_reference": False,
            "fallback_reason": "class_too_small_after_trim",
            "n_query_cells": n_query_cells,
            "n_retained_cells": n_retained_cells,
            "robust_center": np.nan,
            "robust_lower_tail_scale": np.nan,
            "retained_mask": retained_mask,
            "provisional_z_scores": provisional_z_scores,
        }

    retained_scores = class_scores[retained_mask]
    final_center = float(np.median(retained_scores))
    final_scale = compute_lower_tail_scale(retained_scores, final_center)

    if not np.isfinite(final_scale) or final_scale <= scale_floor:
        return {
            "valid_self_reference": False,
            "fallback_reason": "degenerate_class_scale",
            "n_query_cells": n_query_cells,
            "n_retained_cells": n_retained_cells,
            "robust_center": np.nan,
            "robust_lower_tail_scale": np.nan,
            "retained_mask": retained_mask,
            "provisional_z_scores": provisional_z_scores,
        }

    return {
        "valid_self_reference": True,
        "fallback_reason": "",
        "n_query_cells": n_query_cells,
        "n_retained_cells": n_retained_cells,
        "robust_center": final_center,
        "robust_lower_tail_scale": final_scale,
        "retained_mask": retained_mask,
        "provisional_z_scores": provisional_z_scores,
    }


def build_self_reference_stats(
    original_adherence_scores: np.ndarray,
    predicted_classes: np.ndarray,
    class_names: list[str] | None = None,
    min_class_size: int = 30,
    min_retained_size: int = 15,
    scale_floor: float = 1e-4,
    trim_z_threshold: float = -2.5,
) -> pd.DataFrame:
    """Build per-class robust reference statistics from original adherence."""

    original_adherence_scores = np.asarray(original_adherence_scores, dtype=np.float64)
    predicted_classes = np.asarray(predicted_classes, dtype=np.int64)

    if len(original_adherence_scores) != len(predicted_classes):
        raise ValueError("original_adherence_scores and predicted_classes must align.")

    rows = []
    for class_label in np.unique(predicted_classes):
        class_mask = predicted_classes == class_label
        class_scores = original_adherence_scores[class_mask]
        reference = estimate_iterative_lower_tail_reference(
            class_scores,
            min_class_size=min_class_size,
            min_retained_size=min_retained_size,
            scale_floor=scale_floor,
            trim_z_threshold=trim_z_threshold,
        )

        if class_names is not None and 0 <= int(class_label) < len(class_names):
            class_name = class_names[int(class_label)]
        else:
            class_name = ""

        rows.append(
            {
                "class_label": int(class_label),
                "class_name": class_name,
                "n_query_cells": int(reference["n_query_cells"]),
                "n_retained_cells": int(reference["n_retained_cells"]),
                "robust_center": float(reference["robust_center"])
                if np.isfinite(reference["robust_center"])
                else np.nan,
                "robust_lower_tail_scale": float(reference["robust_lower_tail_scale"])
                if np.isfinite(reference["robust_lower_tail_scale"])
                else np.nan,
                "valid_self_reference": bool(reference["valid_self_reference"]),
                "fallback_reason": str(reference["fallback_reason"]),
            }
        )

    return pd.DataFrame(rows).sort_values("class_label").reset_index(drop=True)


def compute_class_relative_adherence(
    original_adherence_scores: np.ndarray,
    predicted_classes: np.ndarray,
    class_reference_stats: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Compute class-relative scores from the robust self-reference table."""

    original_adherence_scores = np.asarray(original_adherence_scores, dtype=np.float64)
    predicted_classes = np.asarray(predicted_classes, dtype=np.int64)

    if len(original_adherence_scores) != len(predicted_classes):
        raise ValueError("original_adherence_scores and predicted_classes must align.")

    n_cells = len(original_adherence_scores)
    class_relative_scores = np.full(n_cells, np.nan, dtype=np.float64)
    z_scores = np.full(n_cells, np.nan, dtype=np.float64)
    class_centers = np.full(n_cells, np.nan, dtype=np.float64)
    class_scales = np.full(n_cells, np.nan, dtype=np.float64)
    fallback_reasons = np.full(n_cells, "", dtype=object)
    used_class_relative = np.zeros(n_cells, dtype=bool)

    class_reference_lookup = class_reference_stats.set_index("class_label")

    for class_label in np.unique(predicted_classes):
        class_mask = predicted_classes == class_label
        class_scores = original_adherence_scores[class_mask]

        if class_label not in class_reference_lookup.index:
            fallback_reasons[class_mask] = "class_too_small"
            continue

        class_row = class_reference_lookup.loc[class_label]
        if not bool(class_row["valid_self_reference"]):
            fallback_reasons[class_mask] = str(class_row["fallback_reason"])
            continue

        class_center = float(class_row["robust_center"])
        class_scale = float(class_row["robust_lower_tail_scale"])

        class_z_scores = (class_scores - class_center) / class_scale
        class_relative_scores[class_mask] = sigmoid_rescale_z_scores(class_z_scores)
        z_scores[class_mask] = class_z_scores
        class_centers[class_mask] = class_center
        class_scales[class_mask] = class_scale
        used_class_relative[class_mask] = True

    return {
        "class_relative_scores": class_relative_scores,
        "z_scores": z_scores,
        "class_centers": class_centers,
        "class_scales": class_scales,
        "fallback_reasons": fallback_reasons,
        "used_class_relative": used_class_relative,
    }


def compute_adjusted_adherence_scores(
    original_adherence_scores: np.ndarray,
    class_relative_scores: np.ndarray,
    original_otsu_threshold: float,
    blend_weight: float = 0.5,
) -> Dict[str, object]:
    """Blend original adherence with aligned class-relative adherence.

    Every cell with a valid class-relative score is blended. The
    ``original_otsu_threshold`` defines the ``guard_mask`` diagnostic and the
    alignment population: median and IQR are estimated from valid cells above
    the threshold, then the resulting transform is applied to all valid cells.
    """

    original_adherence_scores = np.asarray(original_adherence_scores, dtype=np.float64)
    class_relative_scores = np.asarray(class_relative_scores, dtype=np.float64)

    if len(original_adherence_scores) != len(class_relative_scores):
        raise ValueError("original_adherence_scores and class_relative_scores must align.")

    adjusted_scores = original_adherence_scores.copy()

    valid_class_relative = np.isfinite(class_relative_scores)
    guard_mask = original_adherence_scores <= float(original_otsu_threshold)
    alignment_fit_mask = valid_class_relative & (~guard_mask)
    blended_mask = valid_class_relative

    alignment_results = align_scores_by_median_iqr(
        source_scores=class_relative_scores,
        target_scores=original_adherence_scores,
        fit_mask=alignment_fit_mask,
        apply_mask=valid_class_relative,
    )
    aligned_class_relative_scores = alignment_results["aligned_scores"]

    class_relative_scores_used = np.full_like(adjusted_scores, np.nan)
    class_relative_scores_used[blended_mask] = aligned_class_relative_scores[blended_mask]

    adjusted_scores[blended_mask] = (
        blend_weight * original_adherence_scores[blended_mask]
        + (1.0 - blend_weight) * aligned_class_relative_scores[blended_mask]
    )
    adjusted_scores = np.clip(adjusted_scores, 0.0, 1.0)

    return {
        "adjusted_scores": adjusted_scores,
        "valid_class_relative": valid_class_relative,
        "guard_mask": guard_mask,
        "blended_mask": blended_mask,
        "class_relative_scores_used": class_relative_scores_used,
        "aligned_class_relative_scores": aligned_class_relative_scores,
        "alignment_fit_mask": alignment_fit_mask,
        "alignment_fit_count": alignment_results["fit_count"],
        "alignment_method": alignment_results["alignment_method"],
        "alignment_source_median": alignment_results["source_median"],
        "alignment_source_iqr": alignment_results["source_iqr"],
        "alignment_target_median": alignment_results["target_median"],
        "alignment_target_iqr": alignment_results["target_iqr"],
        "alignment_scale_ratio": alignment_results["scale_ratio"],
    }


# Backward compatibility alias
combine_original_and_class_relative_scores = compute_adjusted_adherence_scores


def summarize_class_relative_results(
    class_reference_stats: pd.DataFrame,
    predicted_classes: np.ndarray,
    original_is_atypical: np.ndarray,
    combined_is_atypical: np.ndarray,
    fallback_reasons: np.ndarray,
) -> pd.DataFrame:
    """Summarize per-class support, fallback, and atypical-rate changes."""

    predicted_classes = np.asarray(predicted_classes, dtype=np.int64)
    original_is_atypical = np.asarray(original_is_atypical, dtype=bool)
    combined_is_atypical = np.asarray(combined_is_atypical, dtype=bool)
    fallback_reasons = np.asarray(fallback_reasons, dtype=object)

    query_summary = pd.DataFrame(
        {
            "class_label": predicted_classes,
            "is_atypical_original": original_is_atypical,
            "is_atypical_combined": combined_is_atypical,
            "changed_to_known": original_is_atypical & (~combined_is_atypical),
            "changed_to_atypical": (~original_is_atypical) & combined_is_atypical,
            "fallback_reason": fallback_reasons,
        }
    )

    class_summary = query_summary.groupby("class_label", as_index=False).agg(
        n_query_cells_observed=("class_label", "size"),
        atypical_rate_original=("is_atypical_original", "mean"),
        atypical_rate_combined=("is_atypical_combined", "mean"),
        n_changed_to_known=("changed_to_known", "sum"),
        n_changed_to_atypical=("changed_to_atypical", "sum"),
    )

    for fallback_reason in [
        "class_too_small",
        "class_too_small_after_trim",
        "degenerate_class_scale",
    ]:
        reason_counts = (
            query_summary.loc[query_summary["fallback_reason"] == fallback_reason]
            .groupby("class_label")
            .size()
            .rename(f"n_{fallback_reason}")
            .reset_index()
        )
        class_summary = class_summary.merge(reason_counts, on="class_label", how="left")

    summary_table = class_reference_stats.merge(class_summary, on="class_label", how="outer")

    if "n_query_cells" in summary_table.columns and "n_query_cells_observed" in summary_table.columns:
        summary_table["n_query_cells"] = (
            summary_table["n_query_cells_observed"]
            .fillna(summary_table["n_query_cells"])
        )
        summary_table = summary_table.drop(columns=["n_query_cells_observed"])
    elif "n_query_cells_observed" in summary_table.columns:
        summary_table = summary_table.rename(
            columns={"n_query_cells_observed": "n_query_cells"}
        )

    for column_name in [
        "n_query_cells",
        "n_changed_to_known",
        "n_changed_to_atypical",
        "n_class_too_small",
        "n_class_too_small_after_trim",
        "n_degenerate_class_scale",
    ]:
        if column_name in summary_table.columns:
            summary_table[column_name] = summary_table[column_name].fillna(0).astype(int)

    summary_table["atypical_rate_delta"] = (
        summary_table["atypical_rate_combined"] - summary_table["atypical_rate_original"]
    )
    summary_table["atypical_rate_original"] = summary_table["atypical_rate_original"].fillna(0.0)
    summary_table["atypical_rate_combined"] = summary_table["atypical_rate_combined"].fillna(0.0)
    summary_table["atypical_rate_delta"] = summary_table["atypical_rate_delta"].fillna(0.0)
    summary_table["valid_self_reference"] = summary_table["valid_self_reference"].fillna(False)
    summary_table["class_name"] = summary_table["class_name"].fillna("")

    return summary_table.sort_values(
        ["valid_self_reference", "n_query_cells", "class_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


# =============================================================================
# Shared Threshold Utilities
# =============================================================================



# =============================================================================
# Shared Canonical Atypical Evaluation Helpers
# =============================================================================

def _calculate_otsu_threshold_for_scores(scores: np.ndarray) -> float:
    """Calculate the lower Multi-Otsu threshold for adherence scores.

    Falls back to KMeans when skimage is unavailable or when the score
    distribution has too few distinct values for multi-Otsu.
    """
    try:
        from skimage.filters import threshold_multiotsu

        thresholds = threshold_multiotsu(scores, classes=3)
        return float(thresholds[0])
    except (ImportError, ValueError):
        from sklearn.cluster import KMeans

        n_clusters = min(3, len(np.unique(scores)))
        if n_clusters < 2:
            return float(scores.min())
        X = scores.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        kmeans.fit(X)
        centers = sorted(kmeans.cluster_centers_.flatten())
        return float((centers[0] + centers[1]) / 2.0)


def _pearson_to(C: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Pearson correlation of each row of C to the vector ref (across columns)."""
    Cc = C - C.mean(1, keepdims=True)
    r = ref - ref.mean()
    return (Cc @ r) / (np.linalg.norm(Cc, axis=1) * (np.linalg.norm(r) + 1e-12) + 1e-12)


def inferred_path_blob_verdict(cell_vectors, predictions, is_inferred_path, is_transitioning,
                               parent_nodes, child_nodes, node_u, node_v,
                               blob_ratio: float = 1.5, min_anchor: int = 50) -> str:
    """Per-cell identity-spread verdict for the inferred edge (node_u, node_v).

    Poles are the mean embedding of cells *predicted* as each endpoint and *not on an inferred path*
    (excluding inferred-path cells so the edge's own transitioning cells never define its poles — no
    circularity). An edge whose identity spread is not materially larger (< ``blob_ratio``x) than a single
    homogeneous type's within-type spread is a 'blob'. Direction-agnostic.
    Returns 'blob' | 'spread' | 'unreliable'. All per-cell arrays are row-aligned (kept cells).
    """
    anchor_u = (predictions == node_u) & (~is_inferred_path)
    anchor_v = (predictions == node_v) & (~is_inferred_path)
    if int(anchor_u.sum()) < min_anchor or int(anchor_v.sum()) < min_anchor:
        return "unreliable"
    cU = cell_vectors[anchor_u].mean(0)
    cV = cell_vectors[anchor_v].mean(0)
    dU = np.median(_pearson_to(cell_vectors[anchor_u], cV) - _pearson_to(cell_vectors[anchor_u], cU))
    dV = np.median(_pearson_to(cell_vectors[anchor_v], cV) - _pearson_to(cell_vectors[anchor_v], cU))
    to_t = lambda m: (_pearson_to(cell_vectors[m], cV) - _pearson_to(cell_vectors[m], cU) - dU) / (dV - dU + 1e-12)
    edge = is_transitioning & (
        ((parent_nodes == node_u) & (child_nodes == node_v)) |
        ((parent_nodes == node_v) & (child_nodes == node_u))
    )
    if int(edge.sum()) == 0:
        return "unreliable"
    iqr = lambda x: float(np.subtract(*np.percentile(x, [75, 25])))
    edge_iqr = iqr(to_t(edge))
    null_iqr = float(np.median([iqr(to_t(anchor_u)), iqr(to_t(anchor_v))]))
    return "blob" if edge_iqr / (null_iqr + 1e-12) < blob_ratio else "spread"


def prune_blob_inferred_paths(edge_weights, cell_vectors, predictions, is_inferred_path,
                              is_transitioning, parent_nodes, child_nodes,
                              blob_ratio: float = 1.5):
    """Apply Gate 1 to a dict of inferred-path weights, dropping the 'blob' edges.

    ``edge_weights`` maps ``(node_u, node_v) -> weight`` in whatever node-id space the caller uses — tree
    indices inside ``visualize``; CL-id strings in the obs-space acceptance test. All per-cell arrays are
    row-aligned to the kept cells. Returns ``(kept_weights, verdicts)``: ``kept_weights`` is a new dict with
    the blob edges removed; ``verdicts`` maps every edge to 'blob' | 'spread' | 'unreliable'.
    The calculation is direction-agnostic.
    """
    kept = dict(edge_weights)
    verdicts = {}
    for (u, v) in list(edge_weights.keys()):
        verdict = inferred_path_blob_verdict(
            cell_vectors, predictions, is_inferred_path, is_transitioning,
            parent_nodes, child_nodes, u, v, blob_ratio=blob_ratio)
        verdicts[(u, v)] = verdict
        if verdict == "blob":
            del kept[(u, v)]
    return kept, verdicts


def _filter_bgmm_components(
    bgmm,
    *,
    min_weight: float = _DEFAULT_MIN_COMPONENT_WEIGHT,
    max_std: float = _DEFAULT_MAX_COMPONENT_STD,
    mean_threshold: float = _DEFAULT_MIN_COMPONENT_MEAN,
    axiom_3_direction: str = "upper",
    scoring_axis: int = 0,
) -> Tuple[np.ndarray, int]:
    """Apply three component-selection criteria to a fitted BGMM.

    Each criterion excludes components that do not meet the requested mass,
    cohesion, and relevance thresholds:

    Axiom 1 — Mass (``weight >= min_weight``):
        A component below the configured weight (default ``0.035``) is excluded.

    Axiom 2 — Cohesion (``std <= max_std``):
        A component above the configured standard deviation (default ``0.08``)
        is excluded.

    Axiom 3 — Relevance (mean vs ``mean_threshold``):
        - ``"upper"`` (divergence): ``mean >= mean_threshold``; a divergence
          below 0.618 (default) is excluded by the configured upper gate.
        - ``"lower"`` (adherence): ``mean <= mean_threshold``; low adherence
          marks atypical cells, the Otsu-based threshold being the boundary.

    Cascade Rule:
        Once at least one valid anchor component exists, any remaining
        component that passes Axiom 2 and whose mean is more aberrant than the
        anchor's is also included. This prevents a confirmed signal at one
        score level causing a smaller but more aberrant cluster to be rejected
        on weight alone.

    All three axioms are evaluated as a single vectorised boolean mask; the
    cascade extends the mask in one pass when an anchor exists.

    Args:
        bgmm: A fitted ``BayesianGaussianMixture`` instance.
        min_weight: Minimum component weight for Axiom 1. Default ``0.035``.
        max_std: Maximum component std along ``scoring_axis`` for Axiom 2.
            Default ``0.08``.
        mean_threshold: Threshold for the Axiom 3 mean comparison.
            Default ``0.618``.
        axiom_3_direction: ``"upper"`` (mean >= threshold) or ``"lower"``
            (mean <= threshold). Defaults to ``"upper"``.
        scoring_axis: Axis index for mean/std extraction in multi-dimensional
            fits. Default ``0``.

    Returns:
        A ``(valid_indices, target_index)`` tuple where:

        - ``valid_indices``: ``int64`` array of component indices that pass
          all three axioms. May be empty (length zero) when no component
          qualifies, indicating that this criterion flags no cells.
        - ``target_index``: Index of the target component — the valid
          component with the highest mean for ``"upper"`` or the lowest mean
          for ``"lower"``. Returns ``-1`` when ``valid_indices`` is empty.

    Raises:
        ValueError: If ``axiom_3_direction`` is not ``"upper"`` or ``"lower"``.
        ValueError: If ``bgmm.covariance_type`` is not one of ``"full"``,
            ``"diag"``, ``"spherical"``, or ``"tied"``.
    """
    if axiom_3_direction not in ("upper", "lower"):
        raise ValueError(
            f"axiom_3_direction must be 'upper' or 'lower', got {axiom_3_direction!r}"
        )

    weights = bgmm.weights_          # shape (n_components,)
    means = bgmm.means_              # shape (n_components,) or (n_components, n_features)
    covariances = bgmm.covariances_  # shape depends on covariance_type
    cov_type = bgmm.covariance_type

    # --- Extract per-component mean along scoring_axis ---
    if means.ndim == 1:
        # 1-D fit: means is already (n_components,)
        component_means = means
    else:
        # Multi-dimensional fit: means is (n_components, n_features)
        component_means = means[:, scoring_axis]

    # --- Extract per-component std along scoring_axis (covariance-type dispatch) ---
    if cov_type == "full":
        # covariances_ shape: (n_components, n_features, n_features)
        variances = covariances[:, scoring_axis, scoring_axis]
        component_stds = np.sqrt(variances)
    elif cov_type == "diag":
        # covariances_ shape: (n_components, n_features)
        component_stds = np.sqrt(covariances[:, scoring_axis])
    elif cov_type == "spherical":
        # covariances_ shape: (n_components,) — single scalar variance per component
        component_stds = np.sqrt(covariances)
    elif cov_type == "tied":
        # covariances_ shape: (n_features, n_features) — one shared matrix
        shared_std = np.sqrt(covariances[scoring_axis, scoring_axis])
        component_stds = np.full(len(weights), shared_std)
    else:
        raise ValueError(
            f"Unrecognised covariance_type {cov_type!r}. "
            "Expected one of 'full', 'diag', 'spherical', 'tied'."
        )

    # --- Build three boolean masks and AND them in a single vectorised operation ---
    mask_axiom1 = weights >= min_weight                    # Mass Principle
    mask_axiom2 = component_stds <= max_std               # Cohesion Principle
    if axiom_3_direction == "upper":
        mask_axiom3 = component_means >= mean_threshold   # Relevance Principle (upper)
    else:
        mask_axiom3 = component_means <= mean_threshold   # Relevance Principle (lower)

    combined_mask = mask_axiom1 & mask_axiom2 & mask_axiom3
    valid_indices = np.where(combined_mask)[0].astype(np.int64)

    # --- Cascade: absorb cohesive components beyond the anchor ---
    # Once a valid anchor component is found, any component that is cohesive
    # (passes Axiom 2) and sits further into the aberrant tail than the anchor
    # is almost certainly part of the same biological signal — just a smaller
    # sub-population.  Rejecting it on weight alone would be inconsistent:
    # "the region at μ=0.69 is dangerous but the tighter cluster at μ=0.83
    # is fine" makes no biological sense.
    #
    # The cascade relaxes Axiom 1 (weight) for these tail components while
    # still requiring Axiom 2 (cohesion) to filter out diffuse noise.
    # Axiom 3 is inherently satisfied since "more aberrant than anchor" implies
    # the component is already past the threshold.
    if len(valid_indices) > 0:
        valid_means = component_means[valid_indices]
        if axiom_3_direction == "upper":
            anchor_mean = float(np.max(valid_means))
            cascade_mask = mask_axiom2 & (component_means >= anchor_mean)
        else:
            anchor_mean = float(np.min(valid_means))
            cascade_mask = mask_axiom2 & (component_means <= anchor_mean)
        combined_mask = combined_mask | cascade_mask
        valid_indices = np.where(combined_mask)[0].astype(np.int64)

    # --- Compute target_index ---
    if len(valid_indices) == 0:
        target_index = -1
    else:
        valid_means = component_means[valid_indices]
        if axiom_3_direction == "upper":
            target_index = int(valid_indices[np.argmax(valid_means)])
        else:
            target_index = int(valid_indices[np.argmin(valid_means)])

    return valid_indices, target_index


def _build_child_parent_graph(
    adjacency_matrix: np.ndarray,
    node_names: List[str],
) -> nx.DiGraph:
    """Build the child→parent ontology graph used by pruning logic."""
    graph = nx.DiGraph()
    for idx, name in enumerate(node_names):
        graph.add_node(idx, name=name)

    rows, cols = np.where(adjacency_matrix > 0)
    for child_idx, parent_idx in zip(rows, cols):
        graph.add_edge(int(child_idx), int(parent_idx))
    return graph


def _get_relevant_node_indices_from_graph(
    graph: nx.DiGraph,
    node_names: List[str],
    predicted_indices: np.ndarray,
    include_ancestors: bool = True,
    include_siblings: bool = False,
    siblings_rule: str = "leaves_only",
) -> List[int]:
    """Select the active ontology nodes using the same rules as trajectory."""
    n_nodes = len(node_names)
    active_nodes = {
        idx for idx in np.unique(predicted_indices).tolist() if 0 <= idx < n_nodes
    }
    leaf_nodes = list(set(predicted_indices.tolist()) & set(graph.nodes()))

    roots = [n for n in graph.nodes() if graph.out_degree(n) == 0]
    if not roots:
        cell_root_indices = [
            i for i, name in enumerate(node_names) if "CL:0000000" in name
        ]
        if cell_root_indices:
            roots = cell_root_indices
        else:
            degrees = dict(graph.in_degree())
            roots = [max(degrees, key=degrees.get)]

    if include_ancestors:
        ancestors: Set[int] = set()
        for leaf in leaf_nodes:
            for root in roots:
                if leaf == root:
                    continue
                try:
                    path = nx.shortest_path(graph, source=leaf, target=root)
                    ancestors.update(path)
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    pass
        if not ancestors and leaf_nodes:
            for leaf in leaf_nodes:
                ancestors.update(graph.successors(leaf))
        active_nodes.update(ancestors)

    if include_siblings and siblings_rule == "leaves_only" and len(active_nodes) < 200:
        for node_idx in leaf_nodes:
            try:
                parents = list(graph.successors(node_idx))
                for parent in parents:
                    if parent in roots:
                        continue
                    siblings = list(graph.predecessors(parent))
                    if len(siblings) > 5:
                        siblings = siblings[:5]
                    active_nodes.update(siblings)
            except Exception:
                pass

    return sorted(idx for idx in active_nodes if 0 <= idx < n_nodes)


def _build_projected_active_graph(
    adjacency_matrix: np.ndarray,
    active_indices: List[int],
) -> nx.DiGraph:
    """Project the active ontology nodes onto the reduced parent→child DAG."""
    graph_full = nx.DiGraph()
    rows, cols = np.where(adjacency_matrix > 0)
    for child_idx, parent_idx in zip(rows, cols):
        graph_full.add_edge(int(parent_idx), int(child_idx))

    for idx in range(adjacency_matrix.shape[0]):
        if idx not in graph_full:
            graph_full.add_node(idx)

    graph = nx.DiGraph()
    graph.add_nodes_from(active_indices)
    sorted_active = sorted(active_indices)

    for src in sorted_active:
        for dst in sorted_active:
            if src == dst:
                continue
            try:
                has_path_forward = nx.has_path(graph_full, src, dst)
                has_path_reverse = nx.has_path(graph_full, dst, src)
            except nx.NetworkXError:
                continue
            if has_path_forward and not has_path_reverse:
                graph.add_edge(src, dst)

    try:
        graph = nx.transitive_reduction(graph)
    except Exception:
        pass
    return graph


def _finalize_trajectory_context(
    *,
    graph: nx.DiGraph,
    adjacency_matrix: np.ndarray,
    predicted_indices: np.ndarray,
    active_indices: List[int],
    node_counts: Dict[int, int],
    nodes_below_threshold: Dict[int, int],
) -> Dict[str, Any]:
    """Build one trajectory scoring context from a chosen active node set."""
    active_indices = [int(node) for node in active_indices]
    active_set = set(active_indices)
    keep_mask = np.array(
        [int(pred) in active_set for pred in predicted_indices],
        dtype=bool,
    )
    projected_graph = _build_projected_active_graph(adjacency_matrix, active_indices)

    return {
        "graph": graph,
        "active_indices": active_indices,
        "active_set": active_set,
        "keep_mask": keep_mask,
        "node_counts": dict(node_counts),
        "nodes_below_threshold": dict(nodes_below_threshold),
        "projected_graph": projected_graph,
        "tree_edges": list(projected_graph.edges()),
    }


def _build_trajectory_scoring_contexts(
    adjacency_matrix: np.ndarray,
    node_names: List[str],
    predicted_indices: np.ndarray,
    min_cells_number: int,
) -> Dict[str, Any]:
    """Build the pre-pruning and canonical trajectory scoring contexts."""
    graph = _build_child_parent_graph(adjacency_matrix, node_names)
    predicted_indices = np.asarray(predicted_indices, dtype=np.int64)
    pre_pruning_active_indices = _get_relevant_node_indices_from_graph(
        graph,
        node_names,
        predicted_indices=predicted_indices,
        include_ancestors=True,
        include_siblings=False,
        siblings_rule="leaves_only",
    )

    unique_nodes, counts = np.unique(predicted_indices, return_counts=True)
    node_counts = {int(node): int(count) for node, count in zip(unique_nodes, counts)}
    nodes_below_threshold = {
        int(node): int(count)
        for node, count in node_counts.items()
        if int(count) < int(min_cells_number)
    }
    pre_pruning_context = _finalize_trajectory_context(
        graph=graph,
        adjacency_matrix=adjacency_matrix,
        predicted_indices=predicted_indices,
        active_indices=pre_pruning_active_indices,
        node_counts=node_counts,
        nodes_below_threshold=nodes_below_threshold,
    )

    active_indices = list(pre_pruning_active_indices)
    iteration = 0
    while True:
        iteration += 1
        current_graph = graph.subgraph(active_indices)
        condensation_graph = nx.condensation(current_graph)
        leaf_components = [
            component_id
            for component_id in condensation_graph.nodes()
            if condensation_graph.in_degree(component_id) == 0
        ]

        components_to_remove = []
        for component_id in leaf_components:
            members = condensation_graph.nodes[component_id]["members"]
            component_total = sum(node_counts.get(member, 0) for member in members)
            if component_total < min_cells_number:
                components_to_remove.append(component_id)

        if not components_to_remove:
            break

        nodes_to_remove: List[int] = []
        for component_id in components_to_remove:
            nodes_to_remove.extend(condensation_graph.nodes[component_id]["members"])

        to_remove_set = set(nodes_to_remove)
        active_indices = [node for node in active_indices if node not in to_remove_set]
        if not active_indices or not nodes_to_remove or iteration > 50:
            break

    collapsed_count = 0
    while True:
        current_graph = graph.subgraph(active_indices)
        to_collapse: List[int] = []
        for node_idx in active_indices:
            in_degree = current_graph.in_degree(node_idx)
            out_degree = current_graph.out_degree(node_idx)
            cell_count = node_counts.get(node_idx, 0)
            if in_degree == 1 and out_degree >= 1 and cell_count == 0:
                to_collapse.append(node_idx)

        if not to_collapse:
            break

        active_indices = [node for node in active_indices if node not in to_collapse]
        collapsed_count += len(to_collapse)
        if collapsed_count > 100:
            break

    canonical_context = _finalize_trajectory_context(
        graph=graph,
        adjacency_matrix=adjacency_matrix,
        predicted_indices=predicted_indices,
        active_indices=active_indices,
        node_counts=node_counts,
        nodes_below_threshold=nodes_below_threshold,
    )

    return {
        "canonical": canonical_context,
        "pre_pruning_active": pre_pruning_context,
    }


def _build_canonical_trajectory_context(
    adjacency_matrix: np.ndarray,
    node_names: List[str],
    predicted_indices: np.ndarray,
    min_cells_number: int,
) -> Dict[str, Any]:
    """Build the pruned canonical trajectory context shared by predictor and trajectory."""
    contexts = _build_trajectory_scoring_contexts(
        adjacency_matrix=adjacency_matrix,
        node_names=node_names,
        predicted_indices=predicted_indices,
        min_cells_number=min_cells_number,
    )
    return contexts["canonical"]
 

def _compute_similarity_weights(
    cell_embeddings: np.ndarray,
    node_embeddings: np.ndarray,
    active_indices: List[int],
    temperature: float = 0.1,
    top_k_neighbors: int = 15,
) -> Dict[str, Any]:
    """Compute active-node similarities and softmax weights for trajectory scoring."""
    active_indices = list(active_indices)
    active_embeddings = np.asarray(node_embeddings[active_indices], dtype=np.float64)
    active_norms = np.linalg.norm(active_embeddings, axis=1, keepdims=True)
    active_norms[active_norms == 0] = 1e-10
    active_normalized = active_embeddings / active_norms

    cell_embeddings = np.asarray(cell_embeddings, dtype=np.float64)
    cell_norms = np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
    cell_norms[cell_norms == 0] = 1e-10
    cell_normalized = cell_embeddings / cell_norms

    similarities = cell_normalized @ active_normalized.T
    unmasked_similarities = similarities.copy()

    if 0 < top_k_neighbors < similarities.shape[1]:
        top_k_indices = np.argpartition(similarities, -top_k_neighbors, axis=1)[:, -top_k_neighbors:]
        masked_sims = np.full_like(similarities, -np.inf)
        row_indices = np.arange(similarities.shape[0])[:, None]
        masked_sims[row_indices, top_k_indices] = similarities[row_indices, top_k_indices]
        similarities = masked_sims

    weights = _softmax_similarity_rows(similarities, temperature)

    return {
        "active_indices": active_indices,
        "similarities": similarities,
        "unmasked_similarities": unmasked_similarities,
        "weights": weights,
    }


def _softmax_similarity_rows(
    similarities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Convert similarity rows into normalized probabilities."""

    similarities = np.asarray(similarities, dtype=np.float64)
    if similarities.ndim != 2:
        raise ValueError("similarities must be a 2D array.")
    if similarities.shape[1] == 0:
        return np.zeros_like(similarities, dtype=np.float64)

    shifted = similarities / max(float(temperature), 1e-12)
    row_max = np.max(shifted, axis=1, keepdims=True)
    row_max[~np.isfinite(row_max)] = 0.0

    stable = np.where(np.isfinite(shifted), shifted - row_max, -np.inf)
    exp_sims = np.exp(stable)
    exp_sims[~np.isfinite(stable)] = 0.0

    row_sums = np.sum(exp_sims, axis=1, keepdims=True)
    weights = np.divide(
        exp_sims,
        row_sums,
        out=np.zeros_like(exp_sims, dtype=np.float64),
        where=row_sums > 0,
    )

    zero_rows = np.squeeze(row_sums <= 0, axis=1)
    if np.any(zero_rows):
        weights[zero_rows] = 1.0 / similarities.shape[1]
    return weights


def _normalize_nonnegative_rows(values: np.ndarray) -> np.ndarray:
    """Normalize a non-negative row matrix into probabilities."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array.")
    if values.shape[1] == 0:
        return np.zeros_like(values, dtype=np.float64)

    clipped = np.clip(values, 0.0, None)
    row_sums = np.sum(clipped, axis=1, keepdims=True)
    weights = np.divide(
        clipped,
        row_sums,
        out=np.zeros_like(clipped, dtype=np.float64),
        where=row_sums > 0,
    )

    zero_rows = np.squeeze(row_sums <= 0, axis=1)
    if np.any(zero_rows):
        weights[zero_rows] = 1.0 / values.shape[1]
    return weights


def _compute_js_divergence_rows(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Return row-wise Jensen-Shannon divergence in base-2 units."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("left and right distributions must share the same shape.")
    if left.ndim != 2:
        raise ValueError("left and right distributions must be 2D arrays.")
    if left.shape[1] == 0:
        return np.zeros(left.shape[0], dtype=np.float64)

    left = _normalize_nonnegative_rows(left)
    right = _normalize_nonnegative_rows(right)

    epsilon = 1e-12
    left = np.clip(left, epsilon, None)
    right = np.clip(right, epsilon, None)
    left /= np.sum(left, axis=1, keepdims=True)
    right /= np.sum(right, axis=1, keepdims=True)

    midpoint = 0.5 * (left + right)
    js_divergence = 0.5 * np.sum(
        left * (np.log2(left) - np.log2(midpoint)),
        axis=1,
    ) + 0.5 * np.sum(
        right * (np.log2(right) - np.log2(midpoint)),
        axis=1,
    )
    return np.clip(js_divergence, 0.0, 1.0)


def _compute_jsd_scores(
    cell_embeddings: np.ndarray,
    node_embeddings: np.ndarray,
    active_indices: List[int],
    pre_grit_scores: np.ndarray,
    temperature: float = 0.1,
    top_k_neighbors: int = 15,
) -> np.ndarray:
    """Per-cell Jensen-Shannon divergence between a geometry-based neighbour
    similarity distribution and the normalized pre-GRIT prediction
    distribution over active ontology nodes.

    Used by ``HECTOR.evaluate_cells`` as the JSD axis of the 2-D BGMM hook
    on ``(JSD, adherence)``.
    """
    sim_result = _compute_similarity_weights(
        cell_embeddings,
        node_embeddings,
        active_indices,
        temperature,
        top_k_neighbors,
    )
    geometry_dist = _softmax_similarity_rows(
        sim_result["unmasked_similarities"], temperature,
    )
    prediction_dist = _normalize_nonnegative_rows(
        pre_grit_scores[:, active_indices]
    )
    return _compute_js_divergence_rows(geometry_dist, prediction_dist)


def _compute_edge_and_node_adherence(
    weights: np.ndarray,
    active_indices: List[int],
    tree_edges: List[Tuple[int, int]],
) -> Dict[str, Any]:
    """Compute best-node and best-edge adherence terms for each cell."""
    idx_to_local = {idx: i for i, idx in enumerate(active_indices)}
    edge_u: List[int] = []
    edge_v: List[int] = []
    for src, dst in tree_edges:
        if src in idx_to_local and dst in idx_to_local:
            edge_u.append(idx_to_local[src])
            edge_v.append(idx_to_local[dst])

    if edge_u:
        edge_u_array = np.asarray(edge_u, dtype=np.int64)
        edge_v_array = np.asarray(edge_v, dtype=np.int64)
        all_edge_scores = weights[:, edge_u_array] + weights[:, edge_v_array]
        best_edge_scores = np.max(all_edge_scores, axis=1)
        best_edge_indices = np.argmax(all_edge_scores, axis=1)
    else:
        edge_u_array = np.array([], dtype=np.int64)
        edge_v_array = np.array([], dtype=np.int64)
        best_edge_scores = np.zeros(weights.shape[0], dtype=np.float64)
        best_edge_indices = np.zeros(weights.shape[0], dtype=np.int64)

    best_node_scores = np.max(weights, axis=1)
    adherence_scores = np.maximum(best_edge_scores, best_node_scores).astype(np.float64)

    return {
        "edge_u": edge_u_array,
        "edge_v": edge_v_array,
        "best_edge_scores": best_edge_scores.astype(np.float64),
        "best_edge_indices": best_edge_indices,
        "best_node_scores": best_node_scores.astype(np.float64),
        "adherence_scores": adherence_scores,
    }


def _compute_normalized_entropy_from_similarities(
    unmasked_similarities: np.ndarray,
    temperature: float,
    n_active_nodes: int,
) -> np.ndarray:
    """Compute normalized Shannon entropy from the unmasked active-node similarities."""
    if n_active_nodes <= 1:
        return np.zeros(unmasked_similarities.shape[0], dtype=np.float64)

    shifted = unmasked_similarities / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    exp_sims = np.exp(shifted)
    probs = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)
    log_k = np.log(n_active_nodes)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1) / log_k
    return entropy.astype(np.float64)


def _score_cells_on_active_graph(
    cell_embeddings: np.ndarray,
    node_embeddings: np.ndarray,
    active_indices: List[int],
    tree_edges: List[Tuple[int, int]],
    *,
    temperature: float = 0.1,
    top_k_neighbors: int = 15,
    predictive_scores: Optional[np.ndarray] = None,
    predicted_classes: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    class_relative_adherence: bool = False,
    class_relative_blend_weight: float = 0.4,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the shared trajectory scorer on one active ontology graph.

    Computes similarity weights, per-cell adherence (with optional
    class-relative blending), normalized entropy, and per-cell
    Jensen-Shannon divergence against pre-GRIT prediction scores. No
    atypical classification is performed — callers that need an
    atypical mask should consume ``adata.obs["is_atypical"]`` written
    by ``HECTOR.evaluate_cells``.

    Args:
        cell_embeddings: ``(n_cells, d)`` cell embedding matrix.
        node_embeddings: ``(n_nodes, d)`` node embedding matrix.
        active_indices: Indices of active ontology nodes.
        tree_edges: List of ``(parent, child)`` edge tuples.
        temperature: Softmax temperature for similarity weighting.
        top_k_neighbors: Number of nearest graph neighbours.
        predictive_scores: Optional pre-GRIT score matrix for divergence.
        predicted_classes: Integer class indices for class-relative adjustment.
        class_names: Ordered list of class name strings.
        class_relative_adherence: Whether to apply class-relative adherence.
        class_relative_blend_weight: Blend weight for class-relative adjustment.
        log_fn: Optional logging callback (currently unused; reserved).
    """
    n_cells = int(np.asarray(cell_embeddings).shape[0])
    active_indices = list(active_indices)

    if n_cells == 0:
        empty_float = np.zeros(0, dtype=np.float64)
        empty_weights = np.zeros((0, len(active_indices)), dtype=np.float64)
        return {
            "active_indices": active_indices,
            "similarities": empty_weights.copy(),
            "unmasked_similarities": empty_weights.copy(),
            "weights": empty_weights,
            "edge_u": np.array([], dtype=np.int64),
            "edge_v": np.array([], dtype=np.int64),
            "best_edge_scores": empty_float.copy(),
            "best_edge_indices": np.zeros(0, dtype=np.int64),
            "best_node_scores": empty_float.copy(),
            "adherence_scores": empty_float.copy(),
            "original_adherence_scores": empty_float.copy(),
            "class_relative_scores": np.full(0, np.nan, dtype=np.float64),
            "entropy": empty_float.copy(),
            "predictive_divergence": empty_float.copy(),
        }

    if not active_indices:
        empty_float = np.zeros(n_cells, dtype=np.float64)
        empty_weights = np.zeros((n_cells, 0), dtype=np.float64)
        return {
            "active_indices": active_indices,
            "similarities": empty_weights.copy(),
            "unmasked_similarities": empty_weights.copy(),
            "weights": empty_weights,
            "edge_u": np.array([], dtype=np.int64),
            "edge_v": np.array([], dtype=np.int64),
            "best_edge_scores": empty_float.copy(),
            "best_edge_indices": np.zeros(n_cells, dtype=np.int64),
            "best_node_scores": empty_float.copy(),
            "adherence_scores": empty_float.copy(),
            "original_adherence_scores": empty_float.copy(),
            "class_relative_scores": np.full(n_cells, np.nan, dtype=np.float64),
            "entropy": empty_float.copy(),
            "predictive_divergence": empty_float.copy(),
        }

    similarity_result = _compute_similarity_weights(
        cell_embeddings=cell_embeddings,
        node_embeddings=node_embeddings,
        active_indices=active_indices,
        temperature=temperature,
        top_k_neighbors=top_k_neighbors,
    )
    predictive_divergence = np.zeros(n_cells, dtype=np.float64)
    if predictive_scores is not None:
        active_predictive_scores = np.asarray(
            predictive_scores[:, active_indices], dtype=np.float64
        )
        predictive_divergence = _compute_js_divergence_rows(
            _softmax_similarity_rows(
                similarity_result["unmasked_similarities"],
                temperature,
            ),
            _normalize_nonnegative_rows(active_predictive_scores),
        )
    adherence_result = _compute_edge_and_node_adherence(
        weights=similarity_result["weights"],
        active_indices=active_indices,
        tree_edges=tree_edges,
    )
    entropy = _compute_normalized_entropy_from_similarities(
        similarity_result["unmasked_similarities"],
        temperature=temperature,
        n_active_nodes=len(active_indices),
    )

    original_adherence_scores = np.asarray(
        adherence_result["adherence_scores"], dtype=np.float64
    )
    adherence_scores = original_adherence_scores.copy()
    class_relative_scores = np.full(n_cells, np.nan, dtype=np.float64)
    if (
        class_relative_adherence
        and predicted_classes is not None
        and len(np.unique(predicted_classes)) >= 2
    ):
        otsu_threshold = _calculate_otsu_threshold_for_scores(original_adherence_scores)
        reference_stats = build_self_reference_stats(
            original_adherence_scores,
            predicted_classes,
            class_names,
        )
        class_relative = compute_class_relative_adherence(
            original_adherence_scores,
            predicted_classes,
            reference_stats,
        )
        class_relative_scores = np.asarray(
            class_relative["class_relative_scores"], dtype=np.float64
        )
        adjusted = compute_adjusted_adherence_scores(
            original_adherence_scores,
            class_relative_scores,
            otsu_threshold,
            class_relative_blend_weight,
        )
        adherence_scores = adjusted["adjusted_scores"].astype(np.float64)

    return {
        **similarity_result,
        **adherence_result,
        "adherence_scores": adherence_scores,
        "original_adherence_scores": original_adherence_scores,
        "class_relative_scores": class_relative_scores,
        "entropy": entropy,
        "predictive_divergence": predictive_divergence,
    }


def _compute_jsd_knn_std(
    cell_embeddings: np.ndarray,
    jsd_values: np.ndarray,
    *,
    k: int = 30,
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """Standard deviation of ``jsd_values`` among each cell's k nearest
    neighbours in ``cell_embeddings`` (Euclidean).

    Captures cells that live in a neighborhood of heterogeneous ontology
    confusion, orthogonal to the cell's own JSD. The kNN is computed fresh here
    because the SNN graph used downstream in ``evaluate_cells`` is built
    after feature computation and uses a different parameterisation.
    """
    from sklearn.neighbors import NearestNeighbors

    cell_embeddings = np.asarray(cell_embeddings, dtype=np.float32)
    jsd_values = np.asarray(jsd_values, dtype=np.float64)
    n_cells = cell_embeddings.shape[0]
    if n_cells == 0:
        return np.zeros(0, dtype=np.float64)
    if n_cells == 1:
        return np.zeros(1, dtype=np.float64)
    k_eff = int(min(k, n_cells - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric='euclidean',
                          n_jobs=n_jobs if n_jobs is not None else -1)
    nn.fit(cell_embeddings)
    _, idx = nn.kneighbors(cell_embeddings)
    idx = idx[:, 1:]  # drop self
    return np.nanstd(jsd_values[idx], axis=1)


def _compute_adherence_scores(
    cell_embeddings: np.ndarray,
    node_embeddings: np.ndarray,
    active_indices: List[int],
    tree_edges: List[Tuple[int, int]],
    *,
    temperature: float = 0.1,
    top_k_neighbors: int = 15,
    predicted_classes: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    class_relative_adherence: bool = True,
    class_relative_blend_weight: float = 0.4,
) -> np.ndarray:
    """Per-cell trajectory adherence (optionally class-relative blended).

    Thin scoring-only helper symmetric to ``_compute_jsd_scores``. Used
    by ``HECTOR.evaluate_cells`` as the adherence axis of the 2-D BGMM
    hook on ``(JSD, adherence)``, and by the visualisation fallback
    when atypical detection is disabled.
    """
    result = _score_cells_on_active_graph(
        cell_embeddings=cell_embeddings,
        node_embeddings=node_embeddings,
        active_indices=active_indices,
        tree_edges=tree_edges,
        temperature=temperature,
        top_k_neighbors=top_k_neighbors,
        predicted_classes=predicted_classes,
        class_names=class_names,
        class_relative_adherence=class_relative_adherence,
        class_relative_blend_weight=class_relative_blend_weight,
    )
    return np.asarray(result["adherence_scores"], dtype=np.float64)


# =============================================================================
# Latent Barycentric Placer (Optimal Transport)
# =============================================================================

class LatentBarycentricPlacer:
    """
    Place cells around ontology nodes and along transition edges.

    Stable and transitioning positions are derived from node affinities and
    supplied trajectory state. Atypical classification is external to this
    class and is provided through the placement inputs.
    """
    
    def __init__(
        self,
        node_positions: Dict[int, Tuple[float, float]],
        node_embeddings: np.ndarray,
        tree_edges: List[Tuple[int, int]],
        temperature: float = 0.1,
        transition_stream_width: float = 0.08,
        top_k_neighbors: int = 15,
        stable_source_zone: Optional[float] = None,
        stable_target_zone: Optional[float] = None,
        node_cell_counts: Optional[Dict[int, int]] = None,
        scale_cloud_by_count: bool = True,
        cloud_scale_exponent: float = 3,
        cloud_size_multiplier: float = 1.0,  # Global post-multiplier on the lerp result
        cloud_size_min: float = 0.2,  # Lerp lower bound: smallest-count (or below-threshold) node
        cloud_size_max: float = 1.2,  # Lerp upper bound: largest-count node
        y_scale: float = 0.5,
        is_polar: bool = False,  # True for radial tree layout
        is_pdf_output: bool = False,  # True for PDF rendering
        inferred_path_weights: Optional[Dict[Tuple[int, int], float]] = None,  # Inferred path weights for straight line paths
        # Class-relative adherence parameters
        class_relative_adherence: bool = False,
        class_relative_blend_weight: float = 0.4,
        # Lateral direction metadata from discover_inferred_paths.
        lateral_directions: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
        # Pixel-space sizing: the renderer constructs a PixelGeometry from
        # its axis limits and passes it here so cloud jitter is sized in
        # display pixels regardless of the layout's data aspect.
        pixel_geometry: Optional[Any] = None,
        # Per-node rendered branch widths in pixels, populated by both
        # radial and horizontal layout engines. When supplied,
        # transitioning-cell jitter scales to a fraction of the local
        # branch width so cells visibly fan out within the branch's
        # footprint instead of stacking on its centerline.
        node_widths: Optional[Dict[int, float]] = None,
        # Per-edge Bezier curvature scales from RadialTreeLayoutEngine
        # (neighbor-aware curvature reduction).  Keys are (parent, child).
        edge_curvature_scales: Optional[Dict[Tuple[int, int], float]] = None,
    ):
        self.node_positions = node_positions
        self.node_embeddings = node_embeddings
        self.tree_edges = tree_edges
        self.temperature = temperature
        self.top_k_neighbors = top_k_neighbors
        # ``None`` is the auto-detect sentinel; resolved inside place_cells
        # once relative_position has been computed for every cell.
        self.stable_source_zone = stable_source_zone
        self.stable_target_zone = stable_target_zone
        self.last_auto_zone_info: Optional[Dict[str, Any]] = None
        self.y_scale = y_scale
        self.is_polar = is_polar
        self.is_pdf_output = is_pdf_output
        self.inferred_path_weights = inferred_path_weights or {}
        self.lateral_directions = lateral_directions or {}
        self.class_relative_adherence = class_relative_adherence
        self.class_relative_blend_weight = class_relative_blend_weight
        self.node_widths = node_widths
        self.edge_curvature_scales = edge_curvature_scales or {}

        self.node_cell_counts = node_cell_counts or {}
        self.scale_cloud_by_count = scale_cloud_by_count
        self.cloud_scale_exponent = cloud_scale_exponent
        self.cloud_size_multiplier = cloud_size_multiplier
        self.cloud_size_min = cloud_size_min
        self.cloud_size_max = cloud_size_max
        self.max_node_count = max(self.node_cell_counts.values()) if self.node_cell_counts else 1

        self.active_indices = list(node_positions.keys())
        self.pixel_geometry = pixel_geometry

        # Data diagonal remains available for legacy callers / diagnostics,
        # but is no longer the authoritative sizing reference for clouds
        # under pixel-space sizing.
        self.data_diag = compute_data_diagonal(node_positions, self.active_indices)

        # Transition-stream width stays in data units. Use pixel geometry
        # (y-axis) when available so the stream width reads as the same
        # number of pixels regardless of layout aspect; fall back to the
        # previous diagonal-scaled constant when geometry is unavailable.
        if self.pixel_geometry is not None:
            transition_stream_width_data = (
                transition_stream_width
                * 15.0  # legacy ~15 px stream width at default data_diag=25
                * self.pixel_geometry.data_per_px_y()
            )
            self.transition_stream_width = transition_stream_width_data
        else:
            diag_scale = min(self.data_diag / 25.0, 1.0)
            self.transition_stream_width = transition_stream_width * diag_scale
        self.idx_to_local = {idx: i for i, idx in enumerate(self.active_indices)}
        self.node_pos_array = np.array([node_positions[idx] for idx in self.active_indices])
        self.active_embeddings = node_embeddings[self.active_indices]
        
        self.active_norms = np.linalg.norm(self.active_embeddings, axis=1)
        self.active_norms[self.active_norms == 0] = 1e-10
        
        self.neighbors = {idx: set() for idx in self.active_indices}
        for u, v in tree_edges:
            if u in self.neighbors and v in self.neighbors:
                self.neighbors[u].add(v)
                self.neighbors[v].add(u)
        self.edge_direction = {} 
        for u, v in tree_edges:
            if u in node_positions and v in node_positions:
                # Check if this is a lateral inferred path edge with known cell-evidence direction.
                # lateral_directions maps (src, tgt) → (parent, child) for both forward and reverse
                # lookup keys, so either (u, v) or (v, u) will hit the canonical direction.
                lateral_dir = self.lateral_directions.get((u, v)) or self.lateral_directions.get((v, u))
                is_inferred = (u, v) in self.inferred_path_weights or (v, u) in self.inferred_path_weights

                if is_inferred and lateral_dir is not None:
                    # Lateral inferred path: use cell-evidence direction from lateral_directions
                    self.edge_direction[(u, v)] = lateral_dir
                    self.edge_direction[(v, u)] = lateral_dir
                else:
                    # Canonical edge or vertical inferred path: fall back to geometry-based direction.
                    # In polar mode, use r (radius) instead of x for direction.
                    # In Cartesian mode, use x coordinate.
                    if is_polar:
                        r_u = node_positions[u][1]  # (theta, r)[1] = r
                        r_v = node_positions[v][1]
                    else:
                        r_u = node_positions[u][0]  # (x, y)[0] = x
                        r_v = node_positions[v][0]

                    if r_u < r_v:
                        self.edge_direction[(u, v)] = (u, v)
                        self.edge_direction[(v, u)] = (u, v)
                    elif r_u > r_v:
                        self.edge_direction[(u, v)] = (v, u)
                        self.edge_direction[(v, u)] = (v, u)
                    else:
                        # Geometry is ambiguous (same coordinate). For inferred paths,
                        # use embedding gradient to determine canonical direction.
                        # For canonical edges with same coordinate, fall back to index order.
                        if is_inferred:
                            # Compute embedding-gradient direction: node with higher L2 norm
                            # or higher projection along the mean vertical gradient is the target.
                            norm_u = float(np.linalg.norm(node_embeddings[u]))
                            norm_v = float(np.linalg.norm(node_embeddings[v]))
                            if norm_v >= norm_u:
                                canonical = (u, v)
                            else:
                                canonical = (v, u)
                            self.edge_direction[(u, v)] = canonical
                            self.edge_direction[(v, u)] = canonical
                        else:
                            # Canonical edge: fall back to index order
                            self.edge_direction[(u, v)] = (u, v)
                            self.edge_direction[(v, u)] = (u, v)
        
        self.edge_paths = {}
        if not is_polar:
            # Cartesian mode: use Bezier curves for canonical edges, straight lines for inferred paths
            for u, v in tree_edges:
                if u not in node_positions or v not in node_positions: continue
                start = node_positions[u]
                end = node_positions[v]
                
                # Check if this edge is a inferred path (check both directions)
                is_soft = (u, v) in self.inferred_path_weights or (v, u) in self.inferred_path_weights
                
                if is_soft:
                    # Circular arc path using existing geometry helper
                    # Dynamic curvature: Proportional to distance (Constant Arc Height)
                    dist = np.linalg.norm(np.array(start) - np.array(end))
                    dynamic_curvature = max(0.05 * dist, 0.4) if dist > 0 else 0.4

                    arc_points, arc_normals = generate_circular_arc_points(
                        A=start,
                        B=end,
                        curvature=dynamic_curvature,
                        num_points=21
                    )
                    # In Cartesian mode, paths are always in Cartesian (no conversion needed)
                    self.edge_paths[(u, v)] = (arc_points, arc_normals, False)
                    self.edge_paths[(v, u)] = (arc_points[::-1], arc_normals[::-1], False)
                else:
                    # Bezier curve (existing behavior for canonical edges)
                    points, normals = [], []
                    for t in np.linspace(0, 1, 21):
                        pt, n = calculate_bezier_point(t, start, end)
                        points.append(pt)
                        normals.append(n)
                    self.edge_paths[(u, v)] = (np.array(points), np.array(normals), False)
                    self.edge_paths[(v, u)] = (np.array(points[::-1]), np.array(normals[::-1]), False)
        else:
            # Polar mode: use linear interpolation in polar coordinates  
            for u, v in tree_edges:
                if u not in node_positions or v not in node_positions: continue
                theta_u, r_u = node_positions[u]
                theta_v, r_v = node_positions[v]
                
                # Initialize points and normals for this edge
                points = []
                normals = []
                is_cartesian = False  # Track if this edge uses Cartesian coordinates
                
                # Check if this edge is a inferred path (check both directions)
                is_soft = (u, v) in self.inferred_path_weights or (v, u) in self.inferred_path_weights
                
                # Detect root node (very small radius)
                is_root_edge = (r_u < 0.1)
                
                if is_soft:
                    # Inferred paths: Use circular arc in Cartesian space
                    # Convert polar to Cartesian
                    x_u = r_u * np.cos(theta_u)
                    y_u = r_u * np.sin(theta_u)
                    x_v = r_v * np.cos(theta_v)
                    y_v = r_v * np.sin(theta_v)
                    
                    A = (x_u, y_u)
                    B = (x_v, y_v)
                    
                    # Generate circular arc points in Cartesian space
                    # Dynamic curvature: Proportional to distance (Constant Arc Height)
                    dist = np.hypot(x_u - x_v, y_u - y_v)
                    dynamic_curvature = max(0.05 * dist, 0.4) if dist > 0 else 0.4

                    arc_points, arc_normals = generate_circular_arc_points(
                        A, B, curvature=dynamic_curvature, num_points=21
                    )
                    
                    # Store Cartesian arc points directly
                    # Mark as Cartesian so place_on_edge knows to convert to polar
                    is_cartesian = True
                    for pt, normal in zip(arc_points, arc_normals):
                        points.append(pt)  # Store as Cartesian (x, y)
                        normals.append(normal)  # Store Cartesian normal
                elif is_root_edge:
                    # Root edges: Constant theta (straight radial line), r from 0.2 to r_v
                    # Matches RadialTreeLayoutEngine.get_radial_edges logic
                    for t in np.linspace(0, 1, 21):
                        theta = theta_v  # Constant theta = child theta
                        r = (1 - t) * 0.2 + t * r_v  # Start from r=0.2 offset
                        points.append(np.array([theta, r]))
                        # Normal is tangential (angular direction)
                        normals.append(np.array([1.0, 0.0]))
                else:
                    # Normal canonical edges: Use Bezier curve (consistent with visual edges)
                    # CRITICAL: Apply theta wrapping to match get_radial_edges() logic
                    # Without this, edges crossing ±π boundary create different Bezier curves
                    theta_start = theta_u
                    theta_end = theta_v
                    diff = theta_end - theta_start
                    if diff > np.pi:
                        theta_end -= 2 * np.pi
                    elif diff < -np.pi:
                        theta_end += 2 * np.pi

                    _cs = self.edge_curvature_scales.get((u, v), 1.0)
                    curve_pts = calculate_radial_bezier_curve(
                        theta_start, r_u, theta_end, r_v, num_points=21,
                        curvature_scale=_cs,
                    )
                    # Convert each (theta, r) to Cartesian (x, y), compute true
                    # local tangents via finite differences, and derive a
                    # perpendicular normal at each sample. Storing the path as
                    # Cartesian (is_cartesian=True) routes place_on_edge
                    # through the existing Cartesian-normal jitter branch — no
                    # more pure-angular jitter that ignores the actual path
                    # tangent.
                    xy = np.array([(r * np.cos(t), r * np.sin(t))
                                   for t, r in curve_pts])
                    diffs = np.diff(xy, axis=0)
                    # Pad endpoints so tangents has len == len(xy).
                    diffs = np.vstack([diffs[:1], diffs, diffs[-1:]])
                    # Average left/right neighbor diffs to get a smooth tangent
                    # at interior points; endpoints reuse the boundary diff.
                    avg_diffs = 0.5 * (diffs[:-1] + diffs[1:])
                    norms_mag = np.linalg.norm(avg_diffs, axis=1, keepdims=True)
                    norms_mag = np.clip(norms_mag, 1e-12, None)
                    tangents = avg_diffs / norms_mag
                    # Rotate tangent 90° in 2D to get the perpendicular normal.
                    cart_normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
                    is_cartesian = True
                    for pt, n in zip(xy, cart_normals):
                        points.append(pt)
                        normals.append(n)

                # Store path with explicit Cartesian flag
                self.edge_paths[(u, v)] = (np.array(points), np.array(normals), is_cartesian)
                self.edge_paths[(v, u)] = (np.array(points[::-1]), np.array(normals[::-1]), is_cartesian)

    def compute_raw_transitions(
        self,
        cell_embeddings: np.ndarray,
        prior_anchors: Optional[np.ndarray] = None,
        show_transitions: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Score cells against this placer's tree and compute each cell's
        position along its best edge.

        Returns each cell's best-edge endpoints (``parent_nodes``,
        ``child_nodes``) and two fractional positions along that edge, both
        oriented from topological source (0) to target (1). Does NOT
        auto-detect zones, classify cells as Stable/Transitioning, or place
        them in 2D — those are :meth:`place_cells`'s responsibilities.

        Both are linear rescalings of the cosine-similarity difference::

            gap = similarity(cell, target node) - similarity(cell, source node)

        and differ only in the divisor. ``0.5`` means equally similar to both
        nodes; neither is re-centred.

        * ``raw_t`` (``relative_position``) divides by the 99th percentile of
          ``abs(gap)`` on that edge, then clips to ``[0, 1]``, mapping each
          edge's own spread onto the full range. The zone detection and the
          drawing both use it. The divisor is measured from the cells, so the
          same cell scored on a subset can come back with a different number.
        * ``absolute_t`` (``absolute_position``) divides by the straight-line
          distance between the two node prototypes. It is comparable across
          inputs that use the same checkpoint, features, and preprocessing. It is bounded in ``[0, 1]`` by
          geometry, and it occupies only the middle of that range because a
          cell would have to sit exactly on a prototype to reach either end.

        ``temperature`` affects upstream anchor, edge, and adherence scoring,
        but is not part of either position formula.

        Use this when only the trajectory positions are needed — e.g., for
        zone calibration against the full ontology, or for downstream
        classification logic that bypasses the placer's own labelling.

        Args:
            cell_embeddings: ``(n_cells, d)`` cell latent vectors.
            prior_anchors: Optional ``(n_cells,)`` global anchor indices
                (e.g., GRIT-refined). When supplied, used for anchor
                selection in place of argmax-of-weights; cells whose
                supplied anchor is not in the active subgraph fall back
                to embedding similarity.
            show_transitions: If ``False``, every cell is treated as
                stable: ``parent_nodes[i] = anchor_nodes[i]``,
                ``child_nodes[i] = -1``, ``raw_t[i] = 0``,
                ``absolute_t[i] = NaN``. Matches place_cells's
                raw-prediction-mode semantics.

        Returns:
            Dict with keys:
              - ``"raw_t"``: ``(n_cells,)`` float in [0, 1] — the per-edge-rescaled position (``relative_position``). 0.0 for cells that got no edge.
              - ``"absolute_t"``: ``(n_cells,)`` float in [0, 1] — the model-fixed position (``absolute_position``). NaN for cells that got no edge.
              - ``"parent_nodes"``: ``(n_cells,)`` int — source-of-best-edge global index.
              - ``"child_nodes"``: ``(n_cells,)`` int — target-of-best-edge global index, -1 where no edge.
              - ``"anchor_nodes"``: ``(n_cells,)`` int — argmax / prior anchor global index.
              - ``"weights"``: ``(n_cells, n_active)`` float — per-cell anchor weights.
              - ``"edge_processed"``: ``(n_cells,)`` bool — True for cells that received a best edge.
              - ``"anchor_off_edge"``: ``(n_cells,)`` bool — anchor not at either endpoint of the best edge.
              - ``"prediction_trajectory_conflict"``: ``(n_cells,)`` object — 'None' or 'off-trajectory' (this method tags only off-trajectory; ``place_cells`` may add 'wrong_end' after classification).
        """
        n_cells = cell_embeddings.shape[0]

        anchor_nodes = np.zeros(n_cells, dtype=int)
        parent_nodes = np.zeros(n_cells, dtype=int)
        child_nodes = np.full(n_cells, -1, dtype=int)
        raw_transition_scores = np.zeros(n_cells, dtype=float)
        # NaN, not 0.0: a cell with no edge has nothing to be a fraction
        # along, and 0.0 would read as "sitting on the source node".
        absolute_transition_scores = np.full(n_cells, np.nan, dtype=float)
        edge_processed = np.zeros(n_cells, dtype=bool)
        anchor_off_edge_arr = np.zeros(n_cells, dtype=bool)
        prediction_trajectory_conflict = np.full(n_cells, 'None', dtype=object)

        if len(self.active_indices) == 0:
            return {
                "raw_t": raw_transition_scores,
                "absolute_t": absolute_transition_scores,
                "parent_nodes": parent_nodes,
                "child_nodes": child_nodes,
                "anchor_nodes": anchor_nodes,
                "weights": np.zeros((n_cells, 0), dtype=float),
                "edge_processed": edge_processed,
                "anchor_off_edge": anchor_off_edge_arr,
                "prediction_trajectory_conflict": prediction_trajectory_conflict,
            }

        base_score_result = _score_cells_on_active_graph(
            cell_embeddings=cell_embeddings,
            node_embeddings=self.node_embeddings,
            active_indices=self.active_indices,
            tree_edges=self.tree_edges,
            temperature=self.temperature,
            top_k_neighbors=self.top_k_neighbors,
            class_relative_adherence=False,
        )
        weights = base_score_result["weights"]
        edge_u = base_score_result["edge_u"]
        edge_v = base_score_result["edge_v"]
        best_edge_indices = base_score_result["best_edge_indices"]

        # Determine anchor nodes
        if prior_anchors is not None:
            active_lookup = {uid: i for i, uid in enumerate(self.active_indices)}
            best_node_local_indices = np.zeros(n_cells, dtype=int)
            for i in range(n_cells):
                grit_label = prior_anchors[i]
                if grit_label in active_lookup:
                    best_node_local_indices[i] = active_lookup[grit_label]
                else:
                    best_node_local_indices[i] = np.argmax(weights[i])
        else:
            best_node_local_indices = np.argmax(weights, axis=1)

        # The per-edge scale needs every cell on that edge, so this cannot be
        # one per-cell pass: assign edges, measure the gaps, then rescale.

        # --- Phase A: per-cell edge assignment + off-trajectory diagnostic ---
        # ``src_local`` / ``tgt_local`` are ``parent_nodes`` / ``child_nodes``
        # in the active subgraph's index space, which is how the similarity
        # matrix and the node embeddings below are indexed.
        src_local = np.full(n_cells, -1, dtype=np.int64)
        tgt_local = np.full(n_cells, -1, dtype=np.int64)

        for i in range(n_cells):
            anchor_local = best_node_local_indices[i]
            anchor_global = self.active_indices[anchor_local]
            anchor_nodes[i] = anchor_global

            if len(edge_u) == 0 or not show_transitions:
                parent_nodes[i] = anchor_global
                continue

            edge_idx = best_edge_indices[i]
            u_local = edge_u[edge_idx]
            v_local = edge_v[edge_idx]

            u_global = self.active_indices[u_local]
            v_global = self.active_indices[v_local]
            lateral_dir = (
                self.lateral_directions.get((u_global, v_global))
                or self.lateral_directions.get((v_global, u_global))
            )
            if lateral_dir is not None:
                src, tgt = lateral_dir
            elif (u_global, v_global) in self.edge_direction:
                src, tgt = self.edge_direction[(u_global, v_global)]
            else:
                if self.node_positions[u_global][0] < self.node_positions[v_global][0]:
                    src, tgt = u_global, v_global
                else:
                    src, tgt = v_global, u_global

            edge_processed[i] = True
            parent_nodes[i] = src
            child_nodes[i] = tgt
            if v_global == tgt:
                src_local[i], tgt_local[i] = u_local, v_local
            else:
                src_local[i], tgt_local[i] = v_local, u_local

            anchor_off_edge = anchor_global != src and anchor_global != tgt
            anchor_off_edge_arr[i] = anchor_off_edge
            if anchor_off_edge:
                prediction_trajectory_conflict[i] = 'off-trajectory'

        rows = np.flatnonzero(edge_processed)
        if rows.size == 0:
            return {
                "raw_t": raw_transition_scores,
                "absolute_t": absolute_transition_scores,
                "parent_nodes": parent_nodes,
                "child_nodes": child_nodes,
                "anchor_nodes": anchor_nodes,
                "weights": weights,
                "edge_processed": edge_processed,
                "anchor_off_edge": anchor_off_edge_arr,
                "prediction_trajectory_conflict": prediction_trajectory_conflict,
            }

        # --- Phase B: the gap, and the position the model fixes -------------
        # ``unmasked_similarities`` is plain cosine — _compute_similarity_weights
        # builds it before any top-k masking or softmax — and is already
        # computed above, so the gap costs two gathers.
        similarities = np.asarray(
            base_score_result["unmasked_similarities"], dtype=np.float64
        )
        src_rows = src_local[rows]
        tgt_rows = tgt_local[rows]
        gap = similarities[rows, tgt_rows] - similarities[rows, src_rows]

        # Straight-line distance between the two node prototypes on the unit
        # sphere. Dividing by it bounds the result in [0, 1] by geometry.
        # Dividing by ``1 - cos`` instead — which would put 0 and 1 exactly
        # on the nodes — was measured and puts up to 98% of an edge's cells
        # outside [0, 1], where clipping ties them all to 1.0.
        active_nodes = np.asarray(
            self.node_embeddings[self.active_indices], dtype=np.float64
        )
        node_norms = np.linalg.norm(active_nodes, axis=1, keepdims=True)
        node_norms[node_norms == 0] = 1e-10
        active_nodes = active_nodes / node_norms
        node_cosine = np.einsum(
            "ij,ij->i", active_nodes[src_rows], active_nodes[tgt_rows]
        )
        node_distance = np.sqrt(np.maximum(2.0 * (1.0 - node_cosine), 1e-12))
        absolute_transition_scores[rows] = (gap / node_distance + 1.0) / 2.0

        # --- Phase C: the per-edge scale the dataset fixes ------------------
        # One scale per oriented (source, target) pair — grouping on the pair
        # rather than the best-edge index stays correct if the same endpoints
        # appear twice in ``tree_edges``. ``.ravel()`` because NumPy 2 returns
        # the inverse shaped ``(n, 1)`` when ``axis=0`` is given.
        _, edge_group = np.unique(
            np.stack([src_rows, tgt_rows], axis=1), axis=0, return_inverse=True
        )
        edge_group = np.asarray(edge_group).ravel()
        for group in range(edge_group.max() + 1):
            on_this_edge = edge_group == group
            scale = max(float(np.percentile(np.abs(gap[on_this_edge]), 99.0)), 1e-12)
            raw_transition_scores[rows[on_this_edge]] = np.clip(
                (gap[on_this_edge] / scale + 1.0) / 2.0, 0.0, 1.0
            )

        return {
            "raw_t": raw_transition_scores,
            "absolute_t": absolute_transition_scores,
            "parent_nodes": parent_nodes,
            "child_nodes": child_nodes,
            "anchor_nodes": anchor_nodes,
            "weights": weights,
            "edge_processed": edge_processed,
            "anchor_off_edge": anchor_off_edge_arr,
            "prediction_trajectory_conflict": prediction_trajectory_conflict,
        }

    def place_cells(
        self,
        cell_embeddings: np.ndarray,
        external_atypical_mask: np.ndarray,
        external_adherence_scores: np.ndarray,
        prior_anchors: Optional[np.ndarray] = None,
        show_transitions: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Place cells on the tree using barycentric positioning.

        Atypical classification is authored by ``HECTOR.evaluate_cells``;
        callers must supply the resulting ``is_atypical`` mask and
        ``adherence_score`` column via ``external_atypical_mask`` and
        ``external_adherence_scores``. This method performs no
        classification of its own.

        Args:
            cell_embeddings: ``(n_cells, d)`` cell latent vectors.
            external_atypical_mask: ``(n_cells,)`` boolean mask of cells
                flagged atypical (typically ``adata.obs["is_atypical"]``).
            external_adherence_scores: ``(n_cells,)`` per-cell trajectory
                adherence scores (typically
                ``adata.obs["adherence_score"]``).
            prior_anchors: ``(n_cells,)`` optional GRIT-refined class
                indices (global indices). When provided, used for anchor
                selection; otherwise embedding similarity drives the
                anchor choice.
            show_transitions: If ``False``, all cells placed as stable
                (raw prediction mode).

        Returns:
            Tuple of ``(positions, is_transitioning, parent_nodes,
            child_nodes, transition_scores, raw_transition_scores,
            anchor_nodes, adherence_scores, is_atypical,
            absolute_transition_scores)``.

            Three are positions along the best edge, source (0) to target
            (1); see :meth:`compute_raw_transitions` for how they are built.

            * ``raw_transition_scores`` (``relative_position``) — what the
              classification below reads and what the drawing uses.
            * ``absolute_transition_scores`` (``absolute_position``) —
              comparable for inputs using the same checkpoint, features, and
              preprocessing; NaN where there is no edge.
              Passed straight through for the caller to write to obs.
            * ``transition_scores`` (``transition_position``) — the
              transition band stretched back out to span ``[0, 1]``, and
              ``0.0`` for stable cells, meaning "in the stable pool".

            Cells inside the transition band
            ``(stable_source_zone, 1 - stable_target_zone)`` are tagged
            transitioning; cells in ``[0, stable_source_zone]`` are
            stable at the source; cells in ``[1 - stable_target_zone, 1]``
            are stable at the target. The two zones are independent so
            the source-side and target-side stable bands can be sized
            asymmetrically.

            Either zone may be passed as ``None`` to the constructor; in
            that case ``place_cells`` runs :func:`_auto_detect_stable_zones`
            on the computed ``raw_transition_scores`` and resolves the
            ``None`` side(s) to the detected value. The resolved zones
            are then used for classification and stored on
            ``self.stable_source_zone`` / ``self.stable_target_zone``;
            full diagnostics (mode, surviving components) live on
            ``self.last_auto_zone_info``.

            ``parent_nodes`` and ``child_nodes`` describe the topological
            best edge for every cell that went through the best-edge
            analysis (transitioning *and* stable). When no edges exist
            or ``show_transitions=False``, every cell is forced stable
            and ``parent_nodes`` falls back to the anchor with
            ``child_nodes = -1``.

            The placer also records anchor/best-edge disagreement on
            ``self.last_prediction_trajectory_conflict`` (a string array
            with values ``'None'`` / ``'off-trajectory'`` /
            ``'wrong_end'``) so callers can write it into ``adata.obs``.
        """
        n_cells = cell_embeddings.shape[0]

        # Outputs
        positions = np.zeros((n_cells, 2))
        is_transitioning = np.zeros(n_cells, dtype=bool)
        transition_scores = np.zeros(n_cells, dtype=float)
        adherence_scores = np.asarray(external_adherence_scores, dtype=np.float64).copy()
        is_atypical = np.asarray(external_atypical_mask, dtype=bool).copy()

        # Snapshot which sides arrived as ``None`` (auto-detect sentinel)
        # so we still know after we eagerly default the live attributes
        # to the fallback below. Auto-detection only runs in the main loop
        # path (show_transitions and active graph non-empty); this default
        # keeps ``self.stable_source_zone`` / ``self.stable_target_zone``
        # concrete on every early-return path so callers writing them
        # into ``adata.uns`` always see a number.
        _user_src_zone = self.stable_source_zone
        _user_tgt_zone = self.stable_target_zone
        if self.stable_source_zone is None:
            self.stable_source_zone = 0.1
        if self.stable_target_zone is None:
            self.stable_target_zone = 0.1

        if adherence_scores.shape[0] != n_cells or is_atypical.shape[0] != n_cells:
            raise ValueError(
                "external_atypical_mask and external_adherence_scores must "
                f"be length {n_cells}; got {is_atypical.shape[0]} and "
                f"{adherence_scores.shape[0]}."
            )

        # =====================================================================
        # TWO-PASS PLACEMENT
        # Pass 1: Classify all cells (stable vs transitioning) without placing.
        #   Pass 1a (compute_raw_transitions): score cells, find best edge,
        #            compute raw_t per cell.
        #   Auto-detect: resolve any ``None`` zone(s) from raw_transition_scores.
        #   Pass 1b: apply the (now-resolved) zones to classify each cell as
        #            transitioning vs stable and tag ``wrong_end`` conflicts.
        # Pass 2: Recompute cloud counts from stable-only, then place cells.
        # This ensures cloud sizes match the actual stable cell population.
        # =====================================================================

        # --- Pass 1a: delegated to compute_raw_transitions ---
        raw_result = self.compute_raw_transitions(
            cell_embeddings,
            prior_anchors=prior_anchors,
            show_transitions=show_transitions,
        )
        raw_transition_scores = raw_result["raw_t"]
        absolute_transition_scores = raw_result["absolute_t"]
        parent_nodes = raw_result["parent_nodes"]
        child_nodes = raw_result["child_nodes"]
        anchor_nodes = raw_result["anchor_nodes"]
        weights = raw_result["weights"]
        edge_processed = raw_result["edge_processed"]
        anchor_off_edge_arr = raw_result["anchor_off_edge"]
        prediction_trajectory_conflict = raw_result["prediction_trajectory_conflict"]

        if len(self.active_indices) == 0:
            self.last_prediction_trajectory_conflict = prediction_trajectory_conflict
            return positions, is_transitioning, parent_nodes, child_nodes, transition_scores, raw_transition_scores, anchor_nodes, adherence_scores, is_atypical, absolute_transition_scores

        # Temporary storage for placement t-values (direction-corrected)
        _placement_t = np.zeros(n_cells, dtype=float)

        # --- Auto-detect any originally-None zone(s) ---
        if show_transitions and (_user_src_zone is None or _user_tgt_zone is None):
            if edge_processed.any():
                info = _auto_detect_stable_zones(raw_transition_scores[edge_processed])
            else:
                info = {
                    "stable_source_zone": 0.1,
                    "stable_target_zone": 0.1,
                    "mode": "fallback",
                    "components": [],
                    "n_fit_cells": 0,
                    "reason": "no cells reached the position computation",
                }
            info["user_supplied_source"] = _user_src_zone is not None
            info["user_supplied_target"] = _user_tgt_zone is not None
            if _user_src_zone is None:
                self.stable_source_zone = info["stable_source_zone"]
            if _user_tgt_zone is None:
                self.stable_target_zone = info["stable_target_zone"]
            print(
                f"[HECTOR] auto-detected stable_source_zone="
                f"{self.stable_source_zone:.4f}, stable_target_zone="
                f"{self.stable_target_zone:.4f}  (mode={info['mode']})"
            )
            self.last_auto_zone_info = info

        # Sum-constraint guard: warn and fall back if the band would collapse.
        if self.stable_source_zone + self.stable_target_zone >= 1.0:
            warnings.warn(
                f"stable_source_zone ({self.stable_source_zone:.4f}) + "
                f"stable_target_zone ({self.stable_target_zone:.4f}) >= 1.0; "
                "falling back to 0.1 / 0.1.",
                RuntimeWarning,
            )
            self.stable_source_zone = 0.1
            self.stable_target_zone = 0.1
            self.last_auto_zone_info = {
                "stable_source_zone": 0.1,
                "stable_target_zone": 0.1,
                "mode": "fallback",
                "components": [],
                "n_fit_cells": int(edge_processed.sum()),
                "reason": "src + tgt >= 1.0",
                "user_supplied_source": _user_src_zone is not None,
                "user_supplied_target": _user_tgt_zone is not None,
            }

        # --- Pass 1b: classification + wrong_end diagnostic ------------------
        for i in range(n_cells):
            if not edge_processed[i]:
                continue
            t_raw = raw_transition_scores[i]
            src_node = int(parent_nodes[i])
            tgt_node = int(child_nodes[i])
            anchor_global = int(anchor_nodes[i])
            anchor_off_edge = bool(anchor_off_edge_arr[i])

            # Transition check. After the source→target reorientation
            # in Pass 1a, t_raw always points from source (0) to target
            # (1), so the asymmetric source/target zones apply directly.
            if self.stable_source_zone < t_raw < (1.0 - self.stable_target_zone):
                is_transitioning[i] = True
                t_viz = np.clip(
                    (t_raw - self.stable_source_zone)
                    / (1.0 - self.stable_source_zone - self.stable_target_zone),
                    0.0, 1.0,
                )
                transition_scores[i] = t_viz
                # Drawn from the un-stretched position: the stretched value
                # would always push the extreme cells onto the two nodes
                # whatever the cut-offs are, hiding where the band starts.
                _placement_t[i] = t_raw
            else:
                is_transitioning[i] = False
                # Wrong-end is stable-only: anchor is on the edge but
                # at the opposite endpoint from where t_raw places
                # the cell.
                if not anchor_off_edge:
                    if t_raw <= self.stable_source_zone and anchor_global == tgt_node:
                        prediction_trajectory_conflict[i] = 'wrong_end'
                    elif t_raw >= 1.0 - self.stable_target_zone and anchor_global == src_node:
                        prediction_trajectory_conflict[i] = 'wrong_end'

        # Surface diagnostic conflict labels for the trajectory writer.
        self.last_prediction_trajectory_conflict = prediction_trajectory_conflict

        # --- Between passes: recompute cloud counts from stable-only cells ---
        stable_counts: Dict[int, int] = {}
        for i in range(n_cells):
            if not is_transitioning[i] and not is_atypical[i]:
                node = int(anchor_nodes[i])
                stable_counts[node] = stable_counts.get(node, 0) + 1
        self.node_cell_counts = stable_counts
        self.max_node_count = max(stable_counts.values()) if stable_counts else 1

        # --- Pass 2: Placement (using corrected cloud counts) ---
        for i in range(n_cells):
            anchor_global = int(anchor_nodes[i])

            if is_transitioning[i]:
                # Place on edge path (parent/child/t already set in pass 1)
                positions[i] = self.place_on_edge(
                    parent_nodes[i], child_nodes[i],
                    _placement_t[i],
                )
            else:
                positions[i] = self._place_stable(weights[i], anchor_global)

        return positions, is_transitioning, parent_nodes, child_nodes, transition_scores, raw_transition_scores, anchor_nodes, adherence_scores, is_atypical, absolute_transition_scores

    def compute_cloud_extents(self) -> Dict[int, float]:
        """Compute per-node cloud extent in data-space y-units.

        Under pixel-space sizing the cloud is a circle of
        ``scale_factor * CLOUD_BASE_RADIUS_PX * backend_scale`` pixels;
        the extent returned here is the data-space y half-axis of that
        circle, which is what ribbon terminus blending and contour-window
        bounds consume.
        """
        backend_scale = self._cloud_backend_scale()
        extents: Dict[int, float] = {}
        for node_idx in self.active_indices:
            scale_factor = self._cloud_scale_factor_for_node(node_idx)
            r_px = CLOUD_BASE_RADIUS_PX * scale_factor * backend_scale
            extents[node_idx] = self._cloud_extent_data_y(r_px)
        return extents

    def _cloud_backend_scale(self) -> float:
        """Backend-specific cloud radius multiplier (HTML vs PDF)."""
        return CLOUD_RADIUS_PDF_SCALE if self.is_pdf_output else CLOUD_RADIUS_HTML_SCALE

    def _cloud_scale_factor_for_node(self, node_idx: int) -> float:
        """Return the dimensionless cloud-size multiplier for a node.

        Delegates to ``compute_count_scaled_cloud_scale`` which **lerps**
        the multiplier between ``cloud_size_min`` (smallest-count anchor)
        and ``cloud_size_max`` (largest-count anchor) along a
        ``sqrt(ratio)`` curve. The nonzero minimum prevents strict
        area-to-count proportionality. Applies to both polar (radial) and
        cartesian (horizontal) layouts — same formula in both branches.
        """
        if self.scale_cloud_by_count and self.max_node_count > 0:
            count = self.node_cell_counts.get(node_idx, 1)
            return compute_count_scaled_cloud_scale(
                count=count,
                max_count=self.max_node_count,
                scale_cloud_by_count=self.scale_cloud_by_count,
                cloud_scale_exponent=self.cloud_scale_exponent,
                cloud_size_multiplier=self.cloud_size_multiplier,
                cloud_size_min=self.cloud_size_min,
                cloud_size_max=self.cloud_size_max,
            )
        return 1.0

    def _cloud_extent_data_y(self, r_px: float) -> float:
        """Translate a pixel-space cloud radius to a data-space y-extent.

        Uses ``pixel_geometry`` when available; otherwise falls back to a
        simple data-diagonal-derived scale.
        """
        if self.pixel_geometry is not None:
            return r_px * self.pixel_geometry.data_per_px_y()
        # Without pixel geometry, approximate the conversion from the layout
        # diagonal.
        return r_px * (np.sqrt(min(self.data_diag, 25.0)) * 0.01)

    def _place_stable(self, weights: np.ndarray, anchor: int) -> np.ndarray:
        """Place cell in stable position around anchor node (with cloud jitter).

        Under pixel-space sizing, jitter is drawn from an isotropic pixel
        ellipse of radius ``CLOUD_BASE_RADIUS_PX * scale_factor`` and then
        converted to data units per axis — so the cloud renders as a
        circle on screen regardless of axis anisotropy.
        """
        center = np.array(self.node_positions[anchor])
        scale_factor = self._cloud_scale_factor_for_node(anchor)
        r_px = CLOUD_BASE_RADIUS_PX * scale_factor * self._cloud_backend_scale()

        if self.is_polar:
            # POLAR MODE: Jitter in (theta, r) coordinates. Radial layouts
            # use an isotropic pixel geometry (scaleanchor='y'), so the
            # pixel radius maps into data-space via a single conversion.
            theta_center, r_center = center
            base_jitter_data = self._cloud_extent_data_y(r_px) * 0.35  # 0.35 preserved
            angular_jitter = base_jitter_data / max(r_center, 1.0)
            radial_jitter = base_jitter_data
            delta_theta = np.random.normal(0, angular_jitter)
            delta_r = np.random.normal(0, radial_jitter)
            return np.array([theta_center + delta_theta, r_center + delta_r])

        # CARTESIAN MODE: PCA-aligned elliptical noise computed in pixel
        # space, then converted back to data space using per-axis scale
        # factors. This produces a round cloud on screen regardless of
        # horizontal-layout anisotropy.
        neighbors = list(self.neighbors.get(anchor, []))
        if self.pixel_geometry is not None:
            nbrs_px = []
            for n in neighbors:
                if n in self.node_positions:
                    dxy = np.array(self.node_positions[n]) - center
                    nbrs_px.append(np.array([
                        dxy[0] * self.pixel_geometry.px_per_data_x(),
                        dxy[1] * self.pixel_geometry.px_per_data_y(),
                    ]))
        else:
            nbrs_px = []
            for n in neighbors:
                if n in self.node_positions:
                    nbrs_px.append(np.array(self.node_positions[n]) - center)

        angle_px = 0.0
        if len(nbrs_px) > 0:
            nbrs_arr = np.array(nbrs_px)
            if len(nbrs_arr) == 1:
                angle_px = np.arctan2(nbrs_arr[0][1], nbrs_arr[0][0])
            else:
                cov = np.cov(nbrs_arr.T)
                if cov.ndim >= 2 and not np.all(cov == 0):
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    principal = eigvecs[:, np.argmax(eigvals)]
                    angle_px = np.arctan2(principal[1], principal[0])

        # Isotropic pixel-space jitter: std = r_px on both axes. The 1.2
        # multiplier preserved from the legacy cartesian branch so visual
        # cloud size matches historical output at the calibrated
        # CLOUD_BASE_RADIUS_PX.
        r_long_px = np.random.normal(0, r_px * 1.2)
        r_short_px = np.random.normal(0, r_px * 1.2)
        x_px = r_long_px * np.cos(angle_px) - r_short_px * np.sin(angle_px)
        y_px = r_long_px * np.sin(angle_px) + r_short_px * np.cos(angle_px)

        if self.pixel_geometry is not None:
            dx = x_px * self.pixel_geometry.data_per_px_x()
            dy = y_px * self.pixel_geometry.data_per_px_y()
        else:
            # Test fallback: treat pixel displacements as data displacements
            # scaled by the sqrt(data_diag) reference used elsewhere.
            data_scale = np.sqrt(min(self.data_diag, 25.0)) * 0.01
            dx = x_px * data_scale
            dy = y_px * data_scale
        return center + np.array([dx, dy])

    def place_on_edge(self, source: int, target: int, t: float) -> np.ndarray:
        """Place a cell at parameter ``t`` along the edge (source, target).

        Handles both polar and Cartesian coordinate systems:
        - In polar mode with canonical edges: path points are (theta, r)
        - In polar mode with inferred paths: path points are Cartesian (x, y), converted to polar at end
        - In Cartesian mode: path points are (x, y)

        Args:
            source: Source node index (must match an edge passed to the constructor).
            target: Target node index.
            t: Fractional position in [0, 1] from source to target.

        Returns:
            Position in layout-native coordinates (polar for radial layouts,
            Cartesian for horizontal). Falls back to linear interpolation if
            the (source, target) edge was not pre-sampled.
        """
        edge_key = (source, target)
        edge_data = self.edge_paths.get(edge_key, None)
        
        # Handle both old 2-tuple and new 3-tuple format
        if edge_data is None:
            pts, norms, is_cartesian_path = None, None, False
        elif len(edge_data) == 3:
            pts, norms, is_cartesian_path = edge_data
        else:
            # Legacy 2-tuple format (for compatibility)
            pts, norms = edge_data
            is_cartesian_path = False
        
        if pts is None:
            # Fallback: linear interpolation with jitter
            p1, p2 = np.array(self.node_positions[source]), np.array(self.node_positions[target])
            jitter = np.random.normal(0, self.transition_stream_width * 0.3, 2)
            if self.is_polar:
                # In polar, jitter[0] is angular, jitter[1] is radial
                jitter[1] *= 0.5  # Less radial jitter
            return (1 - t) * p1 + t * p2 + jitter
        
        n_segments = len(pts) - 1
        seg_idx = min(int(t * n_segments), n_segments - 1)
        t_global_start = seg_idx / n_segments
        t_global_end = (seg_idx + 1) / n_segments
        t_local = np.clip((t - t_global_start) / (t_global_end - t_global_start + 1e-10), 0, 1)
        
        pos = (1 - t_local) * pts[seg_idx] + t_local * pts[seg_idx + 1]
        normal = (1 - t_local) * norms[seg_idx] + t_local * norms[seg_idx + 1]

        edge_taper = t / 0.1 if t < 0.1 else ((1.0 - t) / 0.1 if t > 0.9 else 1.0)
        width_factor = 0.5 * (0.5 + 0.5 * edge_taper)

        # Per-call jitter sigma. Both layout engines (TreeLayoutEngine and
        # RadialTreeLayoutEngine) populate ``node_widths``, so this is the
        # active path in production.
        #
        # Sigma tracks the local branch width, so a wide proximal trunk
        # carries a wider stream than a thin distal twig, and is floored by
        # the configured ``transition_stream_width`` — leaf branches are
        # narrower than the cell markers drawn on them, so a purely
        # proportional sigma is invisible there. The branch width is a shape
        # term, not a clip: cells are meant to fan past the branch outline,
        # which is what makes the stream readable as a stream.
        #
        # Turning ``transition_stream_width`` up past the branch term widens
        # every stream uniformly; turning it down lets branch width dominate.
        # The bare fallback below only fires for unit tests that construct the
        # placer without ``pixel_geometry`` or without ``node_widths``.
        jitter_sigma = self.transition_stream_width  # test-only fallback
        if (self.node_widths is not None
                and self.pixel_geometry is not None
                and source in self.node_widths
                and target in self.node_widths):
            backend_scale = 0.5
            w_src_px = float(self.node_widths[source])
            w_tgt_px = float(self.node_widths[target])
            width_at_t_px = (1.0 - t) * w_src_px + t * w_tgt_px
            branch_sigma = (
                1.5 * width_at_t_px * backend_scale
                * self.pixel_geometry.data_per_px_y()
            )
            jitter_sigma = max(branch_sigma, self.transition_stream_width)

        if self.is_polar:
            if is_cartesian_path:
                # Path is in Cartesian (canonical Bezier or inferred arc).
                # Apply jitter perpendicular to the local Cartesian tangent,
                # then convert back to polar.
                jitter_cartesian = normal * np.random.randn() * jitter_sigma * width_factor
                pos_jittered = pos + jitter_cartesian
                x, y = pos_jittered[0], pos_jittered[1]
                theta = np.arctan2(y, x)
                r = np.sqrt(x**2 + y**2)
                return np.array([theta, r])
            else:
                # Polar path (root edges only — pure radial line, angular
                # jitter is geometrically correct here).
                angular_jitter = np.random.randn() * jitter_sigma * width_factor
                return pos + np.array([angular_jitter / max(pos[1], 1.0), 0])
        else:
            jitter_vec = normal * np.random.randn() * jitter_sigma * width_factor
            return pos + jitter_vec


# =============================================================================
# Atypical-cell clusterer (HDBSCAN + anchor-based placement)
# =============================================================================

class AtypicalClusterer:
    """
    Clusters atypical cells into distinct populations and places them as
    virtual anchor hubs on the tree map based on anchor affinity.
    
    The resulting groups and anchor affinities support visualization and
    downstream inspection; they do not by themselves establish a novel state.
    """
    
    def __init__(
        self,
        node_positions: Dict[int, Tuple[float, float]],
        node_embeddings: np.ndarray,
        active_indices: List[int],
        config: 'AtypicalClustererConfig',
        cell_size: float = 1.0,
        layout_x_scale: float = 4.0,  # Pass this to know safe limits
        is_polar: bool = False  # True for radial tree layout
    ):
        """
        Initialize the atypical-cell clusterer.
        
        Args:
            node_positions: {node_idx: (x, y)} positions of ontology nodes
            node_embeddings: [n_nodes, 768] latent vectors for all nodes
            active_indices: List of active node indices in visualization
            config: AtypicalClustererConfig with clustering parameters
            cell_size: Base cell size for jitter scaling
            is_polar: True for radial layout (positions are theta, r)
        """
        self.node_positions = node_positions
        self.node_embeddings = node_embeddings
        self.active_indices = active_indices
        self.config = config
        self.cell_size = cell_size
        self.layout_x_scale = layout_x_scale
        self.is_polar = is_polar
        
        # Precompute normalized embeddings for active nodes
        self.active_embeddings = node_embeddings[active_indices]
        self.active_norms = np.linalg.norm(self.active_embeddings, axis=1, keepdims=True)
        self.active_norms[self.active_norms == 0] = 1e-10
        self.active_normalized = self.active_embeddings / self.active_norms
        
        # Compute tree center for fallback positioning
        if node_positions:
            all_x = [pos[0] for pos in node_positions.values()]
            all_y = [pos[1] for pos in node_positions.values()]
            self.tree_center = (np.mean(all_x), np.mean(all_y))
        else:
            self.tree_center = (0.0, 0.0)
    
    def cluster_atypical(
        self,
        cell_vectors: np.ndarray,
        atypical_mask: np.ndarray,
        cell_anchor_nodes: np.ndarray = None  # Per-cell anchor assignments for pie chart
    ) -> Tuple[List['AtypicalClusterInfo'], np.ndarray]:
        """
        Cluster atypical cells using HDBSCAN.
        
        Args:
            cell_vectors: [n_cells, 768] latent vectors for all cells
            atypical_mask: [n_cells] boolean mask of atypical cells
            cell_anchor_nodes: [n_cells] anchor node index for each cell (for pie chart)
            
        Returns:
            Tuple of (clusters, unclustered_indices):
            - clusters: List of AtypicalClusterInfo objects, one per cluster
            - unclustered_indices: NumPy integer array of atypical cell indices not
              assigned to any atypical cluster by HDBSCAN
        """
        atypical_indices = np.where(atypical_mask)[0]
        n_atypical = len(atypical_indices)
        
        if n_atypical == 0:
            return ([], np.array([], dtype=int))
        
        # Warn if >50% cells are atypical (may indicate model issues)
        total_cells = len(cell_vectors)
        atypical_ratio = n_atypical / total_cells if total_cells > 0 else 0
        if atypical_ratio > 0.5:
            print(f"  > WARNING: {atypical_ratio:.1%} of cells are atypical. This may indicate model issues or a dataset with many novel cell types.")
        
        atypical_vectors = cell_vectors[atypical_indices]
        
        # Helper function to compute cell type counts for a cluster
        def compute_cell_type_counts(cluster_cell_indices: np.ndarray) -> Dict[int, int]:
            """Count actual cell types in cluster based on anchor node assignments."""
            if cell_anchor_nodes is None:
                return {}
            counts = {}
            for cell_idx in cluster_cell_indices:
                cell_type = int(cell_anchor_nodes[cell_idx])
                counts[cell_type] = counts.get(cell_type, 0) + 1
            return counts
        
        # Skip clustering if too few cells
        if n_atypical < self.config.min_atypical_for_clustering:
            # Treat all as single cluster
            centroid = np.mean(atypical_vectors, axis=0)
            anchor_nodes, anchor_weights = self._compute_anchor_affinity(centroid)
            position = self._compute_barycenter(anchor_nodes, anchor_weights)
            cell_type_counts = compute_cell_type_counts(atypical_indices)
            
            clusters = [AtypicalClusterInfo(
                cluster_id=0,
                centroid_vector=centroid,
                cell_indices=atypical_indices,
                anchor_nodes=anchor_nodes,
                anchor_weights=anchor_weights,
                position=position,
                cell_count=n_atypical,
                cell_type_counts=cell_type_counts,
                barycenter=position
            )]
            return (clusters, np.array([], dtype=int))
        
        # Run HDBSCAN clustering with GPU acceleration (cuML) if available
        # Fallback: sklearn HDBSCAN with PCA acceleration (CPU)
        labels = None
        clustering_backend = None
        
        # 1. Try cuML GPU-accelerated HDBSCAN (NVIDIA RAPIDS)
        try:
            from cuml.cluster import HDBSCAN as CumlHDBSCAN
            import cupy as cp
            
            # Convert to cupy array for GPU processing (float64 prevents numerical instability across GPUs)
            atypical_vectors_gpu = cp.asarray(atypical_vectors, dtype=cp.float64)
            
            clusterer = CumlHDBSCAN(
                min_cluster_size=self.config.min_atypical_cluster_size,
                min_samples=max(7, self.config.min_atypical_cluster_size // 4),
                metric='euclidean'
            )
            labels_gpu = clusterer.fit_predict(atypical_vectors_gpu)
            
            # Convert labels back to numpy
            labels = cp.asnumpy(labels_gpu)
            clustering_backend = "cuML (GPU)"
        except ImportError:
            pass  # cuML not available, try next option
        except Exception as e:
            # cuML available but failed (e.g., no GPU, CUDA error)
            print(f"  > cuML HDBSCAN failed: {e}, falling back to CPU")
        
        # 2. Fallback to sklearn HDBSCAN with PCA acceleration (CPU).
        #    PCA approximates the distance structure in fewer dimensions.
        if labels is None:
            try:
                from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
                from sklearn.decomposition import PCA
                print(f"  GPU-accelerated clustering failed, using CPU instead ...")
                # Auto-detect PCA components for 95% explained variance
                # Cap at min(n_samples-1, n_features) for numerical stability
                max_components = min(n_atypical - 1, atypical_vectors.shape[1])
                pca = PCA(n_components=0.95, svd_solver='full', random_state=42)
                reduced_vectors = pca.fit_transform(atypical_vectors)
                n_components = reduced_vectors.shape[1]                
                clusterer = SklearnHDBSCAN(
                    min_cluster_size=self.config.min_atypical_cluster_size,
                    min_samples=max(7, self.config.min_atypical_cluster_size // 4),
                    n_jobs=-1,  # Use all CPU cores
                    copy=True
                )
                labels = clusterer.fit_predict(reduced_vectors)
                clustering_backend = f"sklearn HDBSCAN (CPU, PCA {n_components}D)"
            except ImportError:
                pass  # sklearn not available
        
        # 3. Final fallback: treat all as single cluster
        if labels is None:
            print("  > Warning: HDBSCAN not available (tried cuML, sklearn), treating all atypical as single cluster")
            centroid = np.mean(atypical_vectors, axis=0)
            anchor_nodes, anchor_weights = self._compute_anchor_affinity(centroid)
            position = self._compute_barycenter(anchor_nodes, anchor_weights)
            
            cell_type_counts = compute_cell_type_counts(atypical_indices)
            
            clusters = [AtypicalClusterInfo(
                cluster_id=0,
                centroid_vector=centroid,
                cell_indices=atypical_indices,
                anchor_nodes=anchor_nodes,
                anchor_weights=anchor_weights,
                position=position,
                cell_count=n_atypical,
                cell_type_counts=cell_type_counts,
                barycenter=position
            )]
            return (clusters, np.array([], dtype=int))
        
        # Collect atypical cells that HDBSCAN leaves unclustered (label == -1).
        unclustered_global_indices = atypical_indices[labels == -1]

        # Build AtypicalClusterInfo for each real cluster (exclude unclustered cells).
        atypical_cluster_infos = []
        unique_labels = sorted(set(labels))
        
        for cluster_id, label in enumerate(unique_labels):
            if label < 0:
                continue
                
            cluster_mask = labels == label
            cluster_cell_indices = atypical_indices[cluster_mask]
            cluster_vectors = atypical_vectors[cluster_mask]
            
            # Compute centroid
            centroid = np.mean(cluster_vectors, axis=0)
            
            # Compute anchor affinity
            anchor_nodes, anchor_weights = self._compute_anchor_affinity(centroid)
            
            # Compute initial position (barycenter only — overlap resolved later)
            position = self._compute_barycenter(anchor_nodes, anchor_weights)
            
            # Compute actual cell type counts for pie chart
            cell_type_counts = compute_cell_type_counts(cluster_cell_indices)
            
            atypical_cluster_infos.append(AtypicalClusterInfo(
                cluster_id=cluster_id,
                centroid_vector=centroid,
                cell_indices=cluster_cell_indices,
                anchor_nodes=anchor_nodes,
                anchor_weights=anchor_weights,
                position=position,
                cell_count=len(cluster_cell_indices),
                cell_type_counts=cell_type_counts,
                barycenter=position
            ))
        
        print(
            f"  > Atypical clustering: {n_atypical} cells → "
            f"{len(atypical_cluster_infos)} clusters "
            f"({len(unclustered_global_indices)} unclustered atypical)"
        )

        return (atypical_cluster_infos, unclustered_global_indices)
    
    def _compute_anchor_affinity(self, centroid: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """
        Compute cosine similarity between centroid and active nodes,
        return top-K anchors.
        
        Args:
            centroid: [768] latent vector of cluster centroid
            
        Returns:
            (anchor_nodes, anchor_weights): Top-K anchor indices and their weights
        """
        # Normalize centroid
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            centroid_norm = 1e-10
        centroid_normalized = centroid / centroid_norm
        
        # Cosine similarity with all active nodes
        similarities = centroid_normalized @ self.active_normalized.T
        
        # Get top-K anchors
        k = min(self.config.atypical_anchor_count, len(self.active_indices))
        top_k_local = np.argsort(similarities)[-k:][::-1]  # Descending order
        
        anchor_nodes = [self.active_indices[i] for i in top_k_local]
        anchor_weights = similarities[top_k_local]
        
        # Normalize weights to sum to 1 (for barycentric placement)
        weight_sum = np.sum(np.maximum(anchor_weights, 0))
        if weight_sum > 0:
            anchor_weights = np.maximum(anchor_weights, 0) / weight_sum
        else:
            anchor_weights = np.ones(k) / k  # Equal weights if all negative
        
        return anchor_nodes, anchor_weights
    
    # -------------------------------------------------------------------------
    # Pie Chart Position Resolution
    # -------------------------------------------------------------------------
    def _compute_barycenter(
        self,
        anchor_nodes: List[int],
        anchor_weights: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute weighted average of anchor node positions in CARTESIAN coordinates.
        
        Always returns (x, y) Cartesian coordinates, regardless of layout mode.
        For polar layouts, converts anchor positions to Cartesian first.
        """
        if self.is_polar:
            # Convert polar (theta, r) to Cartesian (x, y), then compute barycenter
            xs, ys, weights = [], [], []
            for node_idx, weight in zip(anchor_nodes, anchor_weights):
                if node_idx in self.node_positions and weight > 0:
                    theta, r = self.node_positions[node_idx]
                    xs.append(r * np.cos(theta))
                    ys.append(r * np.sin(theta))
                    weights.append(weight)
            if not xs:
                return (0.0, 0.0)
            xs = np.array(xs)
            ys = np.array(ys)
            weights = np.array(weights)
            w_sum = np.sum(weights)
            if w_sum > 0:
                weights = weights / w_sum
            else:
                weights = np.ones(len(weights)) / len(weights)
            cart_x = np.sum(weights * xs)
            cart_y = np.sum(weights * ys)
            return (cart_x, cart_y)
        else:
            x_sum, y_sum, w_sum = 0.0, 0.0, 0.0
            for node_idx, weight in zip(anchor_nodes, anchor_weights):
                if node_idx in self.node_positions and weight > 0:
                    pos = self.node_positions[node_idx]
                    x_sum += pos[0] * weight
                    y_sum += pos[1] * weight
                    w_sum += weight
            if w_sum <= 0:
                return (0.0, 0.0)
            return (x_sum / w_sum, y_sum / w_sum)
    



    def place_atypical_cells(

        self,
        cell_positions: np.ndarray,
        atypical_cluster_infos: List['AtypicalClusterInfo']
    ) -> np.ndarray:
        """
        Update cell positions with CLAMPED size limits.
        
        In polar mode, positions are (theta, r). Jitter is applied as:
        - Angular jitter scaled by 1/r (so arc-length is uniform)
        - Radial jitter applied directly
        """
        updated_positions = cell_positions.copy()
        
        # Scale the jitter standard deviation relative to layer spacing.
        max_jitter_radius = self.layout_x_scale * 0.15
        
        for cluster in atypical_cluster_infos:
            hub_coord0, hub_coord1 = cluster.position
            n_cells = cluster.cell_count
            
            # Base scale
            base_jitter = self.cell_size * self.config.atypical_cloud_jitter
            
            # Scale for large clusters
            if n_cells > self.config.atypical_cloud_scale_threshold:
                scale = (n_cells / self.config.atypical_cloud_scale_threshold) ** self.config.atypical_cloud_scale_factor
                jitter_sigma = base_jitter * scale
            else:
                jitter_sigma = base_jitter
            
            # CLAMP: Ensure 3-sigma (99% of cells) fits within max radius
            # If sigma is too big, cap it.
            if jitter_sigma * 3 > max_jitter_radius:
                jitter_sigma = max_jitter_radius / 3.0
            
            # Generate Gaussian cloud positions
            if self.is_polar:
                # POLAR MODE: position is (theta, r)
                # Angular jitter must be scaled by 1/r so that the arc-length
                # displacement is uniform regardless of radius.
                hub_theta, hub_r = hub_coord0, hub_coord1
                for cell_idx in cluster.cell_indices:
                    angular_jitter = np.random.normal(0, jitter_sigma) / max(hub_r, 1.0)
                    radial_jitter = np.random.normal(0, jitter_sigma * 0.5)
                    updated_positions[cell_idx] = [hub_theta + angular_jitter, hub_r + radial_jitter]
            else:
                # CARTESIAN MODE: position is (x, y)
                for cell_idx in cluster.cell_indices:
                    jx = np.random.normal(0, jitter_sigma)
                    jy = np.random.normal(0, jitter_sigma)
                    updated_positions[cell_idx] = [hub_coord0 + jx, hub_coord1 + jy]
        
        return updated_positions




# =============================================================================
# SNN graph + Leiden community detection over the HECTOR latent space.
# =============================================================================
#
# These helpers implement the Phase 1 / Phase 2 backbone that the canonical
# ``HECTOR.evaluate_cells`` pipeline uses:
#   Phase 1: Build a Jaccard-weighted Shared Nearest Neighbor (SNN) graph in
#            the HECTOR latent space (``adata.obsm['X_hector']``).
#   Phase 2: Run Leiden community detection on the FULL SNN graph.
#
# Community labelling (Phase 4 of ``evaluate_cells``) lives in
# ``_label_communities_by_enrichment_fdr`` below, paired with
# ``_fit_bgmm_aberrant_1d`` for the 1-D BGMM seed.


# --- Phase 1 — SNN graph in X_hector space ----------------------------------


def _compute_jaccard_snn_from_knn(
    knn_indices: np.ndarray,
    *,
    min_shared_fraction: float = 0.1,
    chunk_size: int = 4096,
) -> csr_matrix:
    """Build a Jaccard-weighted SNN sparse adjacency from k-NN indices.

    Given each cell's ``k`` nearest neighbours (non-self), the SNN weight
    between cell ``i`` and cell ``j`` is ``|N(i) ∩ N(j)| / |N(i) ∪ N(j)|``,
    computed over the symmetric k-NN adjacency (either direction).  Edges
    with weight below ``min_shared_fraction`` are pruned.  The result is a
    symmetric sparse CSR matrix with zero diagonal.

    Args:
        knn_indices: ``(n_cells, k)`` integer matrix of non-self k-NN indices.
        min_shared_fraction: Minimum Jaccard weight to keep an edge.  Default
            ``0.1``.
        chunk_size: Number of rows processed per scipy sparse multiply.  Keeps
            peak memory bounded on large datasets.

    Returns:
        Sparse CSR matrix of shape ``(n_cells, n_cells)`` with float32 Jaccard
        weights on surviving edges.
    """
    knn_indices = np.asarray(knn_indices, dtype=np.int32)
    n_cells, k = knn_indices.shape
    if n_cells == 0 or k == 0:
        return csr_matrix((n_cells, n_cells), dtype=np.float32)

    rows = np.repeat(np.arange(n_cells, dtype=np.int32), k)
    cols = knn_indices.reshape(-1)
    data = np.ones(n_cells * k, dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))
    A_sym = A.maximum(A.T)
    A_sym.setdiag(0)
    A_sym.eliminate_zeros()
    A_sym = A_sym.tocsr()

    degree = np.asarray(A_sym.sum(axis=1)).ravel().astype(np.float32)

    out_rows: List[np.ndarray] = []
    out_cols: List[np.ndarray] = []
    out_data: List[np.ndarray] = []

    chunk_size = max(1, int(chunk_size))
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        chunk_block = A_sym[start:end]
        # shared[i, j] = number of common neighbours between cell (start+i) and j.
        # Do NOT mask with chunk_block — pairs that share neighbours but are not
        # direct k-NN of each other are valid SNN edges and must not be zeroed out.
        shared = (chunk_block @ A_sym).tocoo()
        if shared.nnz == 0:
            continue
        u_idx = shared.row.astype(np.int64) + start
        v_idx = shared.col.astype(np.int64)
        intersect = shared.data.astype(np.float32)
        union = degree[u_idx] + degree[v_idx] - intersect
        jaccard = np.divide(
            intersect,
            union,
            out=np.zeros_like(intersect, dtype=np.float32),
            where=union > 0,
        )
        keep = jaccard >= float(min_shared_fraction)
        if not np.any(keep):
            continue
        out_rows.append(u_idx[keep].astype(np.int32))
        out_cols.append(v_idx[keep].astype(np.int32))
        out_data.append(jaccard[keep])

    if not out_rows:
        return csr_matrix((n_cells, n_cells), dtype=np.float32)

    rows = np.concatenate(out_rows)
    cols = np.concatenate(out_cols)
    data = np.concatenate(out_data)
    snn = csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float32)
    snn.setdiag(0)
    snn.eliminate_zeros()
    return snn


def _classify_gpu_exception(exc: BaseException) -> str:
    """Bucket a GPU-pipeline exception into a diagnosable category.

    Returns one of ``'OOM'``, ``'not installed'``, or ``'runtime error'``.
    OOM detection matches both ``MemoryError`` and runtime exceptions
    whose string repr contains ``bad_alloc`` / ``out_of_memory`` / ``out
    of memory`` — the three phrasings RMM / cupy / cugraph use.
    """
    if isinstance(exc, ImportError):
        return "not installed"
    if isinstance(exc, MemoryError):
        return "OOM"
    text = f"{type(exc).__name__}: {exc}".lower()
    if ("bad_alloc" in text) or ("out_of_memory" in text) or ("out of memory" in text):
        return "OOM"
    return "runtime error"


def _build_snn_graph_v2(
    cell_embeddings: np.ndarray,
    *,
    n_neighbors: int = 30,
    min_shared_fraction: float = 0.1,
    backend: str = "auto",
    cuml_knn_fn: Optional[Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray, int]]] = None,
    random_state: int = 42,
    pynndescent_n_jobs: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Build a Jaccard-weighted SNN graph on ``cell_embeddings``.

    Dual-backend helper that mirrors the GPU-first / CPU-fallback pattern used
    by :meth:`HECTOR.project`.  ``backend='auto'`` first tries cuML (via the
    ``cuml_knn_fn`` callable provided by the caller); on any ``ImportError``
    or runtime failure it falls back to ``pynndescent``.  ``backend='gpu'``
    forces cuML (raising if unavailable) and ``backend='cpu'`` forces
    pynndescent.

    Args:
        cell_embeddings: ``(n_cells, d)`` float embedding matrix, typically
            ``adata.obsm['X_hector']``.
        n_neighbors: Number of non-self nearest neighbours to use for SNN
            reweighting.  Default ``30`` matches common single-cell SNN
            defaults.
        min_shared_fraction: Minimum Jaccard weight for an SNN edge to be
            retained.  Default ``0.1``.
        backend: ``'auto' | 'gpu' | 'cpu'``.
        cuml_knn_fn: Callable returning ``(knn_indices, knn_distances,
            effective_k)`` given ``(embeddings, n_neighbors)``.  Typically
            ``HECTOR._run_cuml_knn`` bound to the predictor instance.
        random_state: Seed for the pynndescent CPU k-NN.  Ignored by the
            cuML GPU path; the GPU branch applies a
            ``(rounded_distance, index)`` lexsort for stable ordering of the
            returned candidates. This does not guarantee identical candidate
            sets across backends or GPU runs.
        pynndescent_n_jobs: Thread count for the pynndescent CPU k-NN.
            ``None`` (default) uses pynndescent's own default, which is
            multi-threaded. PyNNDescent's multi-threaded mode is **not**
            bit-reproducible even with a fixed ``random_state``; pass ``1``
            for strict same-seed determinism at the cost of speed.
        log_fn: Optional logging callback.

    Returns:
        Dictionary with keys:
          - ``snn_adj``: sparse CSR Jaccard-weighted adjacency.
          - ``knn_indices``: the raw k-NN index matrix actually used.
          - ``backend_used``: ``'gpu'`` or ``'cpu'`` — the actual backend.
          - ``n_neighbors_effective``: clamped neighbor count.
    """
    cell_embeddings = np.asarray(cell_embeddings)
    if cell_embeddings.ndim != 2:
        raise ValueError("cell_embeddings must be a 2D array")
    n_cells = int(cell_embeddings.shape[0])
    if n_cells == 0:
        return {
            "snn_adj": csr_matrix((0, 0), dtype=np.float32),
            "knn_indices": np.zeros((0, 0), dtype=np.int32),
            "backend_used": "cpu",
            "n_neighbors_effective": 0,
        }

    backend = (backend or "auto").lower()
    if backend not in {"auto", "gpu", "cpu"}:
        raise ValueError(f"backend must be one of 'auto', 'gpu', 'cpu' (got {backend!r})")

    knn_indices: Optional[np.ndarray] = None
    backend_used: Optional[str] = None
    effective_k = int(n_neighbors)

    if backend in {"auto", "gpu"}:
        if cuml_knn_fn is None:
            if backend == "gpu":
                raise RuntimeError(
                    "backend='gpu' requires cuml_knn_fn to be provided"
                )
        else:
            try:
                knn_indices_gpu, knn_distances_gpu, effective_k_gpu = cuml_knn_fn(
                    cell_embeddings, n_neighbors
                )
                knn_indices = np.asarray(knn_indices_gpu, dtype=np.int32)
                knn_distances = np.asarray(knn_distances_gpu, dtype=np.float32)
                effective_k = int(effective_k_gpu)
                # Stable tie-break for the candidates returned by cuML: round
                # distances, then lexsort each row by distance and index.
                rounded = np.round(knn_distances, 6).astype(np.float32)
                order = np.lexsort((knn_indices, rounded), axis=1)
                row_idx = np.arange(knn_indices.shape[0])[:, None]
                knn_indices = knn_indices[row_idx, order]
                backend_used = "gpu"
                if log_fn is not None:
                    log_fn(
                        f"  [SNN] cuML k-NN ok (k={effective_k}, "
                        f"n_cells={n_cells})"
                    )
            except (ImportError, RuntimeError, MemoryError) as exc:
                if backend == "gpu":
                    raise
                reason = _classify_gpu_exception(exc)
                if log_fn is not None:
                    log_fn(
                        f"  [SNN] cuML k-NN unavailable (reason: {reason}; "
                        f"{type(exc).__name__}: {exc}); falling back to pynndescent"
                    )

    if knn_indices is None:
        try:
            from pynndescent import NNDescent
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pynndescent is required for the atypical pipeline CPU "
                "backend. Install with: pip install pynndescent"
            ) from exc

        target_k = min(max(int(n_neighbors), 1), max(n_cells - 1, 1))
        if log_fn is not None:
            log_fn(
                f"  [SNN] pynndescent k-NN (k={target_k}, n_cells={n_cells})"
            )
        nnd_kwargs: Dict[str, Any] = dict(
            n_neighbors=target_k + 1,
            metric="euclidean",
            random_state=int(random_state),
        )
        if pynndescent_n_jobs is not None:
            nnd_kwargs["n_jobs"] = int(pynndescent_n_jobs)
        index = NNDescent(
            np.asarray(cell_embeddings, dtype=np.float32),
            **nnd_kwargs,
        )
        all_indices, _ = index.neighbor_graph
        all_indices = np.asarray(all_indices, dtype=np.int32)
        trimmed = np.empty((n_cells, target_k), dtype=np.int32)
        for row in range(n_cells):
            row_idx = all_indices[row]
            row_idx = row_idx[row_idx != row]
            if row_idx.size < target_k:
                pad_needed = target_k - row_idx.size
                fill = np.full(pad_needed, row, dtype=np.int32)
                row_idx = np.concatenate([row_idx, fill])
            trimmed[row] = row_idx[:target_k]
        knn_indices = trimmed
        effective_k = target_k
        backend_used = "cpu"

    if log_fn is not None:
        log_fn(
            f"  [SNN] computing Jaccard SNN weights "
            f"(min_shared_fraction={min_shared_fraction})"
        )
    snn_adj = _compute_jaccard_snn_from_knn(
        knn_indices,
        min_shared_fraction=min_shared_fraction,
    )

    return {
        "snn_adj": snn_adj,
        "knn_indices": knn_indices,
        "backend_used": backend_used or "cpu",
        "n_neighbors_effective": int(effective_k),
    }


# Leiden community detection on the full SNN graph


def _detect_snn_communities_v2(
    snn_adj,
    *,
    backend: str = "auto",
    resolution: float = 1.0,
    random_seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run Leiden community detection on the FULL Jaccard-SNN graph.

    Leiden runs on the complete surviving SNN graph. Classification of the
    resulting communities is performed by the downstream labeling gate.

    GPU path (``backend='auto'`` / ``'gpu'``) is gated on a live VRAM probe:
    the estimated graph footprint (COO + Leiden scratch) is compared against
    the free cupy pool and cugraph is skipped when the fit is too tight to
    land without OOM.  On ``'auto'``, a VRAM-skip falls through to leidenalg
    silently; on ``'gpu'`` it raises.

    Reproducibility: cugraph 23.10+ accepts ``random_state`` on
    ``cugraph.leiden`` and we forward ``random_seed`` when the running
    cugraph build supports it (feature-detected via ``inspect.signature``).
    The GPU path is therefore *seedable but not bit-reproducible* — float32
    reduction order on the GPU still permits a small number of boundary
    cells to flip across runs, but substantially fewer than with an
    unseeded call.  Use ``backend='cpu'`` (``leidenalg``) when strict
    bit-identical reproducibility is required.

    Args:
        snn_adj: Sparse CSR Jaccard-weighted SNN adjacency over all cells.
        backend: ``'auto' | 'gpu' | 'cpu'``.  GPU uses ``cugraph.leiden``;
            CPU uses ``leidenalg`` with igraph.
        resolution: Leiden resolution parameter.
        random_seed: RNG seed for reproducibility.  Forwarded to
            ``cugraph.leiden`` when the build supports ``random_state``
            (cugraph 23.10+); always honored by ``leidenalg`` on the CPU
            backend.
        log_fn: Optional logging callback.

    Returns:
        ``(community_labels, community_size_per_cell)`` — both length
        ``n_cells``.  ``community_labels`` is ``int64``, ``-1`` for any cell
        that is isolated in the SNN graph (no surviving edges).
        ``community_size_per_cell`` is ``int64`` giving the count of cells
        in the same community (``1`` for isolates).
    """
    snn_adj = snn_adj.tocsr() if hasattr(snn_adj, "tocsr") else snn_adj
    n_cells = int(snn_adj.shape[0])
    community_labels = np.full(n_cells, -1, dtype=np.int64)
    community_size_per_cell = np.zeros(n_cells, dtype=np.int64)

    if n_cells == 0:
        return community_labels, community_size_per_cell

    snn_sym = snn_adj.maximum(snn_adj.T)
    snn_sym.setdiag(0)
    snn_sym.eliminate_zeros()
    snn_coo = snn_sym.tocoo()

    if snn_coo.nnz == 0:
        if log_fn is not None:
            log_fn(
                "  [Leiden] SNN graph has zero edges — every cell is its "
                "own singleton community"
            )
        community_labels = np.arange(n_cells, dtype=np.int64)
        community_size_per_cell[:] = 1
        return community_labels, community_size_per_cell

    backend = (backend or "auto").lower()
    if backend not in {"auto", "gpu", "cpu"}:
        raise ValueError(f"backend must be one of 'auto', 'gpu', 'cpu' (got {backend!r})")

    labels: Optional[np.ndarray] = None

    if backend in {"auto", "gpu"}:
        # --- VRAM gate ----------------------------------------------------
        # Estimate cugraph's device footprint: COO src+dst+weight
        # (4+4+4 bytes per edge) plus Leiden scratch (empirically 3-4x
        # the graph).  Probe the live cupy free pool and skip cugraph
        # when the fit is too tight to land without OOM.
        skip_reason: Optional[str] = None
        try:
            from .predictor_support import _available_vram_bytes
            free_bytes = _available_vram_bytes(allocator="rmm")   # RMM/cupy pool
            if free_bytes is not None:
                nnz = int(snn_coo.nnz)
                overhead = float(os.environ.get("HECTOR_CUGRAPH_LEIDEN_OVERHEAD", 3.5))
                needed = int(nnz * 12 * overhead)  # 4+4+4 bytes/edge × overhead
                if log_fn is not None:
                    log_fn(
                        f"  [Leiden] cugraph VRAM probe: free={free_bytes / 1e9:.2f} GB, "
                        f"need~{needed / 1e9:.2f} GB (nnz={nnz:,})"
                    )
                if needed > int(0.85 * free_bytes):
                    skip_reason = (
                        f"graph too large for free VRAM "
                        f"(need ~{needed / 1e9:.2f} GB, have "
                        f"{free_bytes / 1e9:.2f} GB)"
                    )
        except Exception:
            # No cupy or probe failed — don't block cugraph, just try it.
            pass

        if skip_reason is not None:
            if backend == "gpu":
                raise MemoryError(f"cugraph.leiden skipped: {skip_reason}")
            if log_fn is not None:
                log_fn(
                    f"  [Leiden] cugraph skipped ({skip_reason}); "
                    f"using leidenalg"
                )
        else:
            try:
                import cudf  # type: ignore
                import cugraph  # type: ignore

                edge_df = cudf.DataFrame(
                    {
                        "src": snn_coo.row.astype(np.int32),
                        "dst": snn_coo.col.astype(np.int32),
                        "weight": snn_coo.data.astype(np.float32),
                    }
                )
                g = cugraph.Graph()
                g.from_cudf_edgelist(
                    edge_df,
                    source="src",
                    destination="dst",
                    edge_attr="weight",
                    renumber=False,
                )
                # Forward the seed and tighten convergence on cugraph builds
                # that support the kwargs (random_state added in cugraph
                # 23.10).  Older builds silently ignore — we feature-detect
                # via the live signature so the package keeps working with
                # mismatched RAPIDS versions.
                _leiden_kw = set(inspect.signature(cugraph.leiden).parameters)
                _leiden_kwargs: Dict[str, Any] = {"resolution": float(resolution)}
                if "random_state" in _leiden_kw:
                    _leiden_kwargs["random_state"] = int(random_seed)
                if "max_iter" in _leiden_kw:
                    _leiden_kwargs["max_iter"] = 200
                parts_df, _ = cugraph.leiden(g, **_leiden_kwargs)
                parts_sorted = parts_df.sort_values("vertex")
                vtx = parts_sorted["vertex"].to_numpy().astype(np.int64)
                part = parts_sorted["partition"].to_numpy().astype(np.int64)
                tmp = np.full(n_cells, -1, dtype=np.int64)
                tmp[vtx] = part
                labels = tmp
                if log_fn is not None:
                    log_fn(
                        f"  [Leiden] cugraph Leiden ok "
                        f"(n_cells={n_cells}, "
                        f"n_communities={len(np.unique(labels[labels >= 0]))})"
                    )
            except (ImportError, RuntimeError, MemoryError) as exc:
                if backend == "gpu":
                    raise
                reason = _classify_gpu_exception(exc)
                if log_fn is not None:
                    log_fn(
                        f"  [Leiden] cugraph unavailable (reason: {reason}; "
                        f"{type(exc).__name__}: {exc}); falling back to leidenalg"
                    )

    if labels is None:
        try:
            import igraph as ig
            import leidenalg as la
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "igraph and leidenalg are required for the CPU Leiden "
                "backend. Install with: pip install python-igraph leidenalg"
            ) from exc

        edges = list(zip(snn_coo.row.tolist(), snn_coo.col.tolist()))
        weights = snn_coo.data.astype(float).tolist()
        g = ig.Graph(n=int(n_cells), edges=edges, directed=False)
        g.es["weight"] = weights
        partition = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(resolution),
            seed=int(random_seed),
        )
        labels = np.asarray(partition.membership, dtype=np.int64)
        if log_fn is not None:
            log_fn(
                f"  [Leiden] leidenalg Leiden ok "
                f"(n_cells={n_cells}, "
                f"n_communities={len(np.unique(labels))})"
            )

    # Mark cells with no SNN edges as "-1" (isolate); still count them as size 1.
    degree = np.asarray((snn_sym != 0).sum(axis=1)).ravel().astype(np.int64)
    isolate_mask = degree == 0
    if isolate_mask.any():
        labels[isolate_mask] = -1

    community_labels = labels.astype(np.int64)

    # Community sizes (ignoring -1 for counting, but set size=1 for isolates).
    connected_mask = community_labels >= 0
    if connected_mask.any():
        uniq, counts = np.unique(community_labels[connected_mask], return_counts=True)
        size_lookup = dict(zip(uniq.tolist(), counts.tolist()))
        for c in range(n_cells):
            lbl = int(community_labels[c])
            if lbl < 0:
                community_size_per_cell[c] = 1
            else:
                community_size_per_cell[c] = int(size_lookup.get(lbl, 0))
    else:
        community_size_per_cell[:] = 1

    return community_labels, community_size_per_cell


# =============================================================================
# Phase 1.5 — Optional ontology-aware SNN edge pruning.
# =============================================================================


def _build_co_graph_from_directed_adj(
    pure_adjacency: np.ndarray,
    node_names: List[str],
) -> Dict[str, List[str]]:
    """Build a child→parent ``co_graph`` dict from a directed adjacency.

    The HECTOR predictor exposes two adjacency matrices via
    ``_build_visualization_state``:

      * ``adjacency_matrix`` — the GAT's working adjacency, which is
        symmetric + transitive + carries self-loops. Not suitable for
        LCA-depth computation: every node ends up with at least one
        outgoing edge so the BFS-from-roots step finds no roots.
      * ``pure_adjacency_matrix`` — a clean directed adjacency where
        ``adj[child, parent] = 1``. This is what the ontology-aware
        prune step needs.

    Args:
        pure_adjacency: ``(n, n)`` directed adjacency where
            ``adj[i, j] > 0`` means node ``i`` is a child of node ``j``.
        node_names: ``n``-length list of ontology IDs indexing
            ``pure_adjacency``.

    Returns:
        ``{child_name: [parent_names, ...]}`` dict suitable for
        :func:`_build_lca_distance_table`. Self-loops are dropped.
    """
    co_graph: Dict[str, List[str]] = {}
    rows, cols = np.where(pure_adjacency > 0)
    for r, c in zip(rows, cols):
        if r == c:
            continue
        child_name = node_names[int(r)]
        parent_name = node_names[int(c)]
        co_graph.setdefault(child_name, []).append(parent_name)
    return co_graph


def _build_lca_distance_table(
    co_graph: Dict[str, List[str]],
    node_names: List[str],
    unique_class_indices: np.ndarray,
    *,
    level_array: Optional[np.ndarray] = None,
    min_lca_level: int = 0,
) -> np.ndarray:
    """Precompute pairwise leaf-anchored LCA distances for the unique classes.

    For two predicted classes A and B, the leaf-anchored distance is::

        d(A, B) = min over admissible common ancestors L of
                  max(steps_from_A_to_L, steps_from_B_to_L)

    where steps are along child→parent edges of the ontology DAG.
    ``d`` is small when A and B share a common ancestor close to *both*
    leaves (closely related cell types) and large when they only meet
    high in the tree (distantly related). Same-class pairs have d=0.

    The metric is **leaf-anchored**, not root-anchored: ``d=2`` means
    the same biological closeness anywhere in the tree, regardless of
    how deep the relevant branch happens to sit.

    Optional **branch barrier**: when both ``level_array`` and
    ``min_lca_level >= 1`` are supplied, common ancestors with
    ``level_array[ancestor] < min_lca_level`` are excluded from the
    minimisation. Pairs whose only common ancestors fall below the
    barrier are reported as ``-1`` (no admissible LCA), so a downstream
    consumer can refuse to bridge across separate branches. With the
    defaults (``level_array=None`` or ``min_lca_level=0``), the branch barrier
    is disabled. Used by
    :func:`_lca_routed_build_g5_partitions` to enforce the
    "shallow classes stay separate" principle.

    Args:
        co_graph: ``{child_name: [parent_names, ...]}`` ontology
            adjacency. Should be a clean directed DAG; see
            :func:`_build_co_graph_from_directed_adj` for converting a
            ``pure_adjacency_matrix`` into this form.
        node_names: List indexable by class index; entry ``i`` is the
            ontology ID corresponding to class ``i``.
        unique_class_indices: 1-D int array of distinct predicted
            classes appearing in this dataset (e.g.,
            ``np.unique(predictions)``). Indices are into
            ``node_names``.
        level_array: optional ``(N,)`` int array giving each node's
            depth from the ontology root (``0`` = root, ``1`` = direct
            child of root, …). Required to activate the branch barrier.
        min_lca_level: minimum ``level_array[ancestor]`` for an
            ancestor to count as an admissible LCA. ``0`` (default)
            disables the barrier. ``2`` is the recommended setting for
            HECTOR's full ontology — it excludes the root and the 24
            super-generic level-1 nodes (e.g. ``eukaryotic cell``,
            ``hematopoietic cell``).

    Returns:
        ``(U, U)`` symmetric int32 matrix indexed by *position* in
        ``unique_class_indices``. Diagonal is 0. Pairs whose admissible
        ancestor set is empty (disconnected ontology, or all common
        ancestors filtered out by the branch barrier) are recorded as
        ``-1`` (the prune helper maps this to factor 0; the G5 builder
        maps it to ``+inf`` to refuse cross-branch merges).
    """
    from collections import deque

    n_nodes = len(node_names)
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    # Build the child -> parent DiGraph.
    G = nx.DiGraph()
    for i in range(n_nodes):
        G.add_node(i, name=node_names[i])
    for child_name, parent_names in co_graph.items():
        if child_name not in name_to_idx:
            continue
        child_idx = name_to_idx[child_name]
        for parent_name in parent_names:
            if parent_name in name_to_idx:
                parent_idx = name_to_idx[parent_name]
                G.add_edge(child_idx, parent_idx)

    unique = np.asarray(unique_class_indices, dtype=np.int64)
    U = int(unique.shape[0])

    # Precompute ancestors-with-distance-from-the-class for every queried
    # class ONCE. ``ancestors_per[a][n] = shortest path length from class
    # ``a`` to ancestor ``n`` along child→parent edges. Same-node has
    # distance 0.
    ancestors_per: List[Dict[int, int]] = []
    for a in unique:
        seen: Dict[int, int] = {int(a): 0}
        q: deque = deque([(int(a), 0)])
        while q:
            cur, d_cur = q.popleft()
            for parent in G.successors(cur):
                new_d = d_cur + 1
                if parent not in seen or new_d < seen[parent]:
                    seen[parent] = new_d
                    q.append((parent, new_d))
        ancestors_per.append(seen)

    # Branch-barrier admissibility: build a per-class ancestor view that
    # has already had below-threshold nodes filtered out. Avoids the
    # filter cost on the inner pairwise loop.
    barrier_active = (
        level_array is not None and int(min_lca_level) >= 1
    )
    if barrier_active:
        lvl = np.asarray(level_array)
        if lvl.shape[0] != n_nodes:
            raise ValueError(
                "_build_lca_distance_table: level_array length "
                f"{lvl.shape[0]} != n_nodes {n_nodes}"
            )
        thr = int(min_lca_level)
        admissible_per: List[Dict[int, int]] = [
            {n: d for n, d in anc.items() if int(lvl[n]) >= thr}
            for anc in ancestors_per
        ]
    else:
        admissible_per = ancestors_per

    # Pairwise leaf-anchored distance over admissible ancestors. ``-1``
    # sentinel marks "no admissible LCA" (disconnected ontology OR
    # branch-barrier excluded all common ancestors).
    table = np.full((U, U), -1, dtype=np.int32)
    for i in range(U):
        table[i, i] = 0
    for a in range(U):
        anc_a = admissible_per[a]
        anc_a_keys = anc_a.keys()
        for b in range(a + 1, U):
            anc_b = admissible_per[b]
            common = anc_a_keys & anc_b.keys()
            if not common:
                continue
            d = min(max(anc_a[node], anc_b[node]) for node in common)
            table[a, b] = d
            table[b, a] = d

    return table


def _prune_snn_by_ontology_v1(
    snn_adj,
    predictions: np.ndarray,
    co_graph: Dict[str, List[str]],
    node_names: List[str],
    *,
    lineage_levels: int,
    soft_power: float = 1.0,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Downweight or cut SNN edges that bridge ontologically distant classes.

    Slots between :func:`_build_snn_graph_v2` and
    :func:`_detect_snn_communities_v2` to prevent Leiden from forming
    communities that span unrelated lineages. Edge weights are scaled
    by the **leaf-anchored** LCA distance of the two cells' predicted
    classes::

        d(A, B) = min over common ancestors L of
                  max(steps_from_A_to_L, steps_from_B_to_L)

    Smaller ``d`` = closely related cell types; larger ``d`` = leaves
    that only meet high in the tree. Same-class pairs have ``d=0``.
    Being leaf-anchored, ``d=K`` means the same biological closeness
    anywhere in the tree regardless of branch depth.

    Two knobs:

      * ``lineage_levels`` (int): edges with ``d > lineage_levels`` are
        cross-lineage and zeroed; edges with ``d <= lineage_levels`` are
        kept (optionally attenuated by ``soft_power``). Larger = more
        permissive.
      * ``soft_power`` (float, default ``1.0``): exponent on the
        normalized in-band distance. ``0`` = pure hard cutoff (in-band
        edges keep full weight), ``1`` = linear taper, larger = shrink
        mid-distance in-band edges more.

    Per-edge factor::

        K = lineage_levels
        if d < 0 or d > K:        factor = 0
        elif soft_power == 0:     factor = 1
        else:                     factor = ((K + 1 - d) / (K + 1)) ** soft_power

    At K=4, soft_power=1: d=0 -> 1.0, d=1 -> 0.8, ... d=4 -> 0.2, d>=5 -> 0.

    Args:
        snn_adj: Symmetric Jaccard-weighted SNN graph as a sparse CSR
            matrix from :func:`_build_snn_graph_v2`.
        predictions: ``(n_cells,)`` int array of predicted class
            indices into ``node_names``.
        co_graph: ``{child_name: [parent_names]}`` ontology adjacency.
        node_names: List indexable by predicted class index; entry
            ``i`` is the ontology ID for class ``i``.
        lineage_levels: ``K``, the leaf-anchored distance cutoff.  Must
            be a non-negative int.  The caller is responsible for
            handling ``None`` (feature off — do not call this function
            in that case).
        soft_power: Exponent on the normalized in-band distance.  Must
            be ``>= 0``.  Default ``1.0`` (linear taper).  ``0.0`` is
            a pure hard cutoff.
        log_fn: Optional log callback for a one-line summary.

    Returns:
        ``(pruned_adj, prune_info)``. ``pruned_adj`` is a CSR matrix
        with the same shape and dtype as ``snn_adj`` and structural
        zeros eliminated. ``prune_info`` contains:

          * ``lineage_levels``
          * ``soft_power``
          * ``n_unique_classes``
          * ``n_edges_before`` (counted as upper-triangle pairs)
          * ``n_edges_after``
          * ``frac_zeroed``
          * ``mean_weight_before``
          * ``mean_weight_after``
          * ``lineage_distance_histogram`` (``{distance: count}`` over
            upper triangle; ``-1`` key means no LCA / disconnected)
    """
    if not isinstance(lineage_levels, (int, np.integer)) or lineage_levels < 0:
        raise ValueError(
            "_prune_snn_by_ontology_v1: lineage_levels must be a "
            f"non-negative integer, got {lineage_levels!r}"
        )
    if soft_power < 0:
        raise ValueError(
            "_prune_snn_by_ontology_v1: soft_power must be >= 0, "
            f"got {soft_power!r}"
        )

    predictions = np.asarray(predictions, dtype=np.int64)
    if predictions.ndim != 1:
        raise ValueError(
            "_prune_snn_by_ontology_v1: predictions must be 1-D, "
            f"got shape {predictions.shape}"
        )
    if int(predictions.max(initial=-1)) >= len(node_names):
        raise ValueError(
            "_prune_snn_by_ontology_v1: predictions contain index "
            f"{int(predictions.max())} but node_names has only "
            f"{len(node_names)} entries."
        )

    unique_classes, inverse = np.unique(predictions, return_inverse=True)
    n_unique = int(unique_classes.shape[0])

    # Edge case: only one unique class — nothing to prune against.
    if n_unique <= 1:
        if log_fn is not None:
            log_fn(
                f"  [SNN-prune] skipped (n_unique_classes={n_unique}); "
                "graph unchanged."
            )
        coo = snn_adj.tocoo()
        upper = coo.row < coo.col
        n_upper = int(upper.sum())
        mean_w = float(coo.data[upper].mean()) if n_upper else 0.0
        prune_info = {
            "lineage_levels": int(lineage_levels),
            "soft_power": float(soft_power),
            "n_unique_classes": n_unique,
            "n_edges_before": n_upper,
            "n_edges_after": n_upper,
            "frac_zeroed": 0.0,
            "mean_weight_before": mean_w,
            "mean_weight_after": mean_w,
            "lineage_distance_histogram": {},
            "skipped_reason": "single_class",
        }
        return snn_adj, prune_info

    distance_table = _build_lca_distance_table(
        co_graph, node_names, unique_classes
    )

    if n_unique > 1000 and log_fn is not None:
        log_fn(
            f"  [SNN-prune] note: n_unique_classes={n_unique} — LCA "
            "precompute may take a few seconds."
        )

    coo = snn_adj.tocoo()
    row = coo.row
    col = coo.col
    data = coo.data.astype(np.float64, copy=False)

    edge_d = distance_table[inverse[row], inverse[col]]
    K = int(lineage_levels)

    # In-band: 0 <= d <= K (and d != -1, which means no LCA).
    in_band = (edge_d >= 0) & (edge_d <= K)

    if soft_power == 0.0:
        # Pure hard cutoff: in-band edges keep full weight, out-of-band zero.
        factor = in_band.astype(np.float64)
    else:
        denom = float(K + 1)
        # ``(K + 1 - d) / (K + 1)`` for in-band; 0 elsewhere.
        raw = np.where(
            in_band,
            (denom - edge_d.astype(np.float64)) / denom,
            0.0,
        )
        if soft_power == 1.0:
            factor = raw
        else:
            factor = np.power(raw, float(soft_power))

    new_data = data * factor

    # Histogram + summary stats reported on the upper triangle only.
    upper_mask = row < col
    n_edges_before = int(upper_mask.sum())
    upper_depths = edge_d[upper_mask]
    upper_data_before = data[upper_mask]
    upper_data_after = new_data[upper_mask]

    n_edges_after = int(np.count_nonzero(upper_data_after))
    if n_edges_before:
        mean_before = float(upper_data_before.mean())
        mean_after = float(upper_data_after.mean())
        frac_zeroed = float(
            np.count_nonzero(upper_data_after == 0) / n_edges_before
        )
    else:
        mean_before = 0.0
        mean_after = 0.0
        frac_zeroed = 0.0

    hist_d, hist_c = np.unique(upper_depths, return_counts=True)
    lineage_distance_histogram = {
        int(d): int(c) for d, c in zip(hist_d, hist_c)
    }

    pruned = csr_matrix(
        (new_data.astype(coo.data.dtype, copy=False), (row, col)),
        shape=snn_adj.shape,
        dtype=snn_adj.dtype,
    )
    # Drop structural zeros so Leiden backends (especially cugraph,
    # which traverses the CSR structure rather than checking values)
    # don't treat zeroed entries as live edges of weight 0.
    pruned.eliminate_zeros()

    prune_info = {
        "lineage_levels": int(lineage_levels),
        "soft_power": float(soft_power),
        "n_unique_classes": n_unique,
        "n_edges_before": n_edges_before,
        "n_edges_after": n_edges_after,
        "frac_zeroed": frac_zeroed,
        "mean_weight_before": mean_before,
        "mean_weight_after": mean_after,
        "lineage_distance_histogram": lineage_distance_histogram,
    }

    if log_fn is not None:
        log_fn(
            f"  [SNN-prune] n_unique_classes={n_unique} "
            f"edges {n_edges_before} -> {n_edges_after} "
            f"(frac_zeroed={frac_zeroed:.3f}, "
            f"mean_w {mean_before:.3f} -> {mean_after:.3f})"
        )

    return pruned, prune_info


# =============================================================================
# LCA-routed dispatch helpers (used by HECTOR.evaluate_cells).
# =============================================================================


def _all_ancestors_in_co_graph(
    start: str,
    co_graph: Dict[str, List[str]],
) -> Set[str]:
    """Transitive closure of ancestors via ``co_graph`` (child → [parents]).

    DFS upward from ``start`` along parent edges. Returns ``start`` itself
    plus every node reachable by repeatedly following parent links.
    Useful for computing the lowest common ancestor of a set of leaves
    by intersecting per-leaf ancestor sets.
    """
    seen: Set[str] = {start}
    stack = [start]
    while stack:
        n = stack.pop()
        for p in co_graph.get(n, []):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def _build_child_map(
    co_graph: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Reverse the co_graph: ``parent → [children]``.

    The ontology DAG is stored as ``child → [parents]`` (the standard CL
    representation). Some routines (e.g., LCA selection) need to walk
    downward; this helper materialises the reverse map once.
    """
    child_map: Dict[str, List[str]] = defaultdict(list)
    for child, parents in co_graph.items():
        for p in parents:
            child_map[p].append(child)
    return dict(child_map)


def _lca_routed_predicted_classes_lca(
    co_graph: Dict[str, List[str]],
    node_names: List[str],
    predictions_full: np.ndarray,
) -> Dict[str, Any]:
    """Lowest common ancestor(s) of all unique predicted classes.

    The LCA is the most-specific ontology node that is an ancestor of
    every predicted class. Computed as: intersect the ancestor sets of
    each unique predicted class, then keep elements that have *no*
    descendant in the intersection (= the deepest common ancestors).

    The routing decision for ``HECTOR.evaluate_cells`` rests on
    ``lca_is_root_only``: if all predicted classes only share the
    ontology root as a common ancestor, the dataset spans multiple
    biological lineages → multi-lineage path. Otherwise the predicted
    classes are nested under a specific cell type → single-lineage path.

    Args:
        co_graph: ``{child_name: [parent_names]}`` ontology adjacency
            (the same ``full_co_graph`` produced by
            :func:`_build_co_graph_from_directed_adj`).
        node_names: List indexable by class index; entry ``i`` is the
            ontology ID for class ``i``.
        predictions_full: ``(n_cells,)`` int array of full-ontology
            predicted-class indices (typically
            ``core_result.top_indices[:, 0]`` from a top-1 prediction).

    Returns:
        Dict with keys

        * ``n_classes``: number of unique predicted classes.
        * ``n_common_ancestors``: size of the common-ancestor set.
        * ``common``: set of ontology IDs that are ancestors of *every*
          predicted class.
        * ``lca``: subset of ``common`` whose elements have no
          descendant in ``common`` — the most-specific common ancestors.
        * ``roots``: set of ontology IDs that have no parents in
          ``co_graph`` (typically a single CL root such as
          ``CL:0000000``).
        * ``lca_is_root_only``: ``True`` iff ``lca == roots`` (i.e., the
          predicted classes only meet at the top of the ontology).
    """
    classes = np.unique(np.asarray(predictions_full, dtype=np.int64))
    if classes.size == 0:
        return {
            "n_classes": 0,
            "n_common_ancestors": 0,
            "common": set(),
            "lca": set(),
            "roots": {n for n in node_names if not co_graph.get(n)},
            "lca_is_root_only": False,
        }

    class_names = [node_names[int(i)] for i in classes]
    common: Set[str] = set.intersection(
        *(_all_ancestors_in_co_graph(n, co_graph) for n in class_names)
    )

    child_map = _build_child_map(co_graph)
    lca: Set[str] = set()
    for n in common:
        dominated = False
        stack = list(child_map.get(n, []))
        seen: Set[str] = set()
        while stack:
            d = stack.pop()
            if d in seen:
                continue
            seen.add(d)
            if d in common:
                dominated = True
                break
            stack.extend(child_map.get(d, []))
        if not dominated:
            lca.add(n)

    roots: Set[str] = {n for n in node_names if not co_graph.get(n)}
    return {
        "n_classes": int(classes.size),
        "n_common_ancestors": int(len(common)),
        "common": common,
        "lca": lca,
        "roots": roots,
        "lca_is_root_only": (lca == roots),
    }


def _lca_routed_scan_leiden_by_ari(
    snn_pruned,
    predictions: np.ndarray,
    *,
    resolution_range: Tuple[float, float] = (0.5, 2.0),
    resolution_step: float = 0.1,
    backend: str = "auto",
    random_seed: int = 42,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Scan Leiden resolutions on a pruned SNN and pick the ARI-best one.

    Runs :func:`_detect_snn_communities_v2` once per candidate resolution
    in the inclusive range ``[resolution_range[0], resolution_range[1]]``
    with step ``resolution_step``, scoring each clustering by adjusted
    Rand index against the input ``predictions`` array. Selects the
    resolution whose Leiden labels best agree with the predicted-class
    structure — the "auto-resolution Leiden" branch (G1) of the
    LCA-routed dispatch.

    Leiden receives ``random_seed`` at each resolution. Backend and numerical
    differences can still affect the selected partition; cache the result when
    this scan is used inside repeated computations.

    Args:
        snn_pruned: lineage-pruned Jaccard SNN adjacency (the output of
            :func:`_prune_snn_by_ontology_v1`).
        predictions: ``(n_cells,)`` int array of predicted-class
            indices used as the ARI reference clustering.
        resolution_range: inclusive ``(lo, hi)`` bounds for the
            resolution scan.
        resolution_step: stride between consecutive resolutions.
        backend: forwarded to :func:`_detect_snn_communities_v2`.
        random_seed: forwarded to Leiden.
        log_fn: optional log callback for a one-line summary.

    Returns:
        Dict with keys

        * ``best_resolution``: ``float``.
        * ``best_ari``: ``float`` ARI of the chosen clustering.
        * ``community_labels``: ``(n_cells,)`` int64 Leiden labels at
          the chosen resolution.
        * ``community_size_per_cell``: ``(n_cells,)`` int64.
        * ``ari_scores``: ``Dict[float, float]`` mapping each tested
          resolution (rounded to 10 decimals) to its ARI score.
    """
    from sklearn.metrics import adjusted_rand_score

    lo, hi = resolution_range
    if not (lo > 0 and hi > lo):
        raise ValueError(
            "_lca_routed_scan_leiden_by_ari: resolution_range must satisfy "
            f"0 < lo < hi (got {resolution_range!r})"
        )
    if not (resolution_step > 0):
        raise ValueError(
            "_lca_routed_scan_leiden_by_ari: resolution_step must be > 0 "
            f"(got {resolution_step!r})"
        )

    predictions = np.asarray(predictions)
    resolutions = np.arange(
        float(lo), float(hi) + float(resolution_step) / 2.0, float(resolution_step)
    )
    scores: Dict[float, float] = {}
    best_score = -np.inf
    best_res: Optional[float] = None
    best_labels: Optional[np.ndarray] = None
    best_sizes: Optional[np.ndarray] = None

    for r in resolutions:
        r_val = float(round(r, 10))
        labels_r, sizes_r = _detect_snn_communities_v2(
            snn_pruned,
            backend=backend,
            resolution=r_val,
            random_seed=random_seed,
            log_fn=None,  # avoid 16 noisy log lines
        )
        score = float(adjusted_rand_score(predictions, labels_r))
        scores[r_val] = score
        if score > best_score:
            best_score = score
            best_res = r_val
            best_labels = labels_r
            best_sizes = sizes_r

    assert best_labels is not None and best_sizes is not None and best_res is not None

    if log_fn is not None:
        n_comms = int(len(np.unique(best_labels[best_labels >= 0])))
        log_fn(
            f"  [LCA-G1] auto-resolution scan: best res={best_res:.2f} "
            f"(ARI={best_score:.3f}, {n_comms} communities); "
            f"scanned {len(scores)} resolutions"
        )

    return {
        "best_resolution": float(best_res),
        "best_ari": float(best_score),
        "community_labels": best_labels,
        "community_size_per_cell": best_sizes,
        "ari_scores": scores,
    }


def _lca_routed_build_g5_partitions(
    co_graph: Dict[str, List[str]],
    node_names_full: List[str],
    predictions_full: np.ndarray,
    *,
    d_cuts: Tuple[float, ...] = (0.5, 1.5, 2.5, 3.5, 4.5),
    level_array: Optional[np.ndarray] = None,
    min_lca_level: int = 0,
) -> List[Dict[str, Any]]:
    """Build the multi-resolution ontology rollup partitions for G5.

    Cluster the unique predicted classes by leaf-anchored LCA distance
    via average-link hierarchical clustering, then cut the dendrogram at
    each value in ``d_cuts`` to produce a stack of nested coarser
    partitions. Cells inherit the cluster id of their predicted class.

    Used by the multi-lineage dispatch branch: each partition is fed
    through :func:`_label_communities_by_enrichment_fdr` and the
    per-cell atypical masks are unioned across depths.

    Optional **branch barrier** (``level_array`` + ``min_lca_level``):
    pairs whose only common ancestors sit at level < ``min_lca_level``
    receive a large finite sentinel in the linkage input. At the configured
    finite cut values, classes whose
    LCAs fall below the barrier remain in separate clusters at every
    finite ``d_cut`` — enforcing "shallow classes stay separate even if
    only one small branch". With ``level_array=None`` or
    ``min_lca_level=0``, the branch barrier is disabled.

    Args:
        co_graph: ``{child_name: [parent_names]}`` ontology adjacency.
        node_names_full: list indexable by class index — entry ``i`` is
            the ontology ID for class ``i``.
        predictions_full: ``(n_cells,)`` full-ontology predicted-class
            indices.
        d_cuts: cluster-distance thresholds used to extract nested
            partitions. Defaults to the calibrated grid
            ``(0.5, 1.5, 2.5, 3.5, 4.5)`` — fine to coarse.
        level_array: optional ``(N,)`` int array giving each
            ontology node's depth from the root (``0`` = root).
            Required to activate the branch barrier.
        min_lca_level: minimum admissible LCA level for a class pair to
            be allowed to merge. ``0`` (default) disables the barrier.
            ``2`` is recommended for HECTOR's full ontology
            (excludes root and 24 super-generic level-1 nodes).

    Returns:
        List of dicts, one per ``d_cut``, each with

        * ``d_cut``: the threshold used.
        * ``n_communities``: number of clusters at that cut.
        * ``community_labels``: ``(n_cells,)`` int64 (0-indexed).
        * ``community_size_per_cell``: ``(n_cells,)`` int64.
    """
    from scipy.cluster.hierarchy import fcluster
    # ``linkage`` and ``squareform`` are imported at module top.

    if not d_cuts:
        raise ValueError(
            "_lca_routed_build_g5_partitions: d_cuts must be non-empty "
            f"(got {d_cuts!r})"
        )
    if any(d <= 0 for d in d_cuts):
        raise ValueError(
            "_lca_routed_build_g5_partitions: d_cuts must all be > 0 "
            f"(got {d_cuts!r})"
        )

    predictions_full = np.asarray(predictions_full, dtype=np.int64)
    unique_classes, inverse = np.unique(predictions_full, return_inverse=True)

    if unique_classes.size <= 1:
        # Degenerate — only one predicted class. Every depth collapses to a
        # single community spanning all cells.
        n_cells = int(predictions_full.size)
        labels = np.zeros(n_cells, dtype=np.int64)
        sizes = np.full(n_cells, n_cells, dtype=np.int64)
        return [
            {
                "d_cut": float(d_cut),
                "n_communities": 1,
                "community_labels": labels,
                "community_size_per_cell": sizes,
            }
            for d_cut in d_cuts
        ]

    dist_table = _build_lca_distance_table(
        co_graph, node_names_full, unique_classes,
        level_array=level_array, min_lca_level=int(min_lca_level),
    )
    # Replace -1 (no admissible LCA — disconnected ontology OR branch
    # barrier excluded all common ancestors) with a large finite sentinel
    # that is strictly above any configured d_cut. ``scipy.linkage``
    # rejects np.inf, but a sentinel two orders of magnitude above
    # max(d_cuts) ensures any cross-barrier merge in the dendrogram
    # happens at a distance well beyond every fcluster cut, so
    # cross-branch pairs remain in separate clusters.
    max_finite = float(dist_table[dist_table >= 0].max()) if (
        np.any(dist_table >= 0)
    ) else 0.0
    barrier_sentinel = max(
        100.0 * float(max(d_cuts)),
        100.0 * (max_finite + 1.0),
    )
    clean = np.where(
        dist_table < 0, barrier_sentinel, dist_table
    ).astype(np.float64)
    np.fill_diagonal(clean, 0.0)
    # Symmetrise numerically (LCA distance is symmetric in principle but
    # may carry tiny numerical asymmetries from the int32 table).
    clean = (clean + clean.T) / 2.0
    cond = squareform(clean, checks=False)
    Z = linkage(cond, method="average")

    partitions: List[Dict[str, Any]] = []
    for d_cut in d_cuts:
        cluster_per_class = fcluster(Z, t=float(d_cut), criterion="distance")
        cell_labels = (cluster_per_class[inverse] - 1).astype(np.int64)
        sizes_by_label = np.bincount(cell_labels)
        community_size_per_cell = sizes_by_label[cell_labels].astype(np.int64)
        partitions.append({
            "d_cut": float(d_cut),
            "n_communities": int(cluster_per_class.max()),
            "community_labels": cell_labels,
            "community_size_per_cell": community_size_per_cell,
        })

    return partitions


# =============================================================================
# Community-majority halt: BGMM aberrant detection + Leiden aggregation.
# =============================================================================


def _fit_bgmm_aberrant_1d(
    abnormality: np.ndarray,
    *,
    n_components: int,
    min_weight: float,
    max_std: float,
    train_subsample: int,
    threshold: float,
    random_state: int,
    weight_concentration_prior: float,
    max_iter: int,
    covariance_type: str,
    reg_covar: float,
) -> Dict[str, Any]:
    """Fit a 1-D Dirichlet-process BGMM on per-cell abnormality and select seeds.

    Cells flagged as ``is_aberrant`` are those whose hard-assigned BGMM
    component passes the configured right-tail selection criteria.

    Seed selection: after axiom 1 (mass, ``weight >= min_weight``) and
    axiom 2 (cohesion, ``std <= max_std``) gate out junk components,
    take all surviving components whose mean is at or above
    ``threshold``. Halts (``bgmm_halted=True``, ``is_aberrant`` all
    False) when zero components qualify — no real atypical tail.

    NaN cells get ``bgmm_component = -1`` and ``is_aberrant = False``,
    flowing harmlessly through the downstream community gate.

    All hyperparameters are required keyword-only arguments: a pure
    parametric primitive with no domain defaults. Callers are
    atypical-cell detection (``predictor.py``) and stable-zone
    auto-detection (``_auto_detect_stable_zones`` below).

    Args:
        abnormality: ``(n_cells,)`` per-cell 1-D score (e.g. abnormality
            in ``[0, 1]`` for the atypical caller, or
            ``relative_position`` for the zone-detection caller).
            NaN entries are excluded from the fit.
        n_components: BGMM upper bound on components. The Dirichlet
            process prior shrinks unused components.
        min_weight: Axiom 1 minimum component weight.
        max_std: Axiom 2 maximum component standard deviation.
        train_subsample: Cap on points fed to the fit (predict still
            runs on every valid cell). ``0`` disables.
        threshold: Per-component mean cutoff. All cohesive components
            with ``mean >= threshold`` become seeds; halt when none
            qualify.
        random_state: RNG seed for the BGMM fit and subsampling.
        weight_concentration_prior: Dirichlet-process concentration
            parameter passed to ``BayesianGaussianMixture``.
        max_iter: Maximum EM iterations.
        covariance_type: One of sklearn's covariance types
            (``"diag"`` keeps ``covariances_`` shaped
            ``(n_components, 1)`` for 1-D inputs).
        reg_covar: Regularisation added to the diagonal of the
            covariance matrices to keep them positive.

    Returns:
        Dict with:

        * ``is_aberrant``: ``(n_cells,)`` bool array.
        * ``bgmm_component``: ``(n_cells,)`` int64 hard component
          assignment (``-1`` for NaN cells).
        * ``aberrant_component_indices``: sorted list of seed component
          indices.
        * ``bgmm``: fitted ``BayesianGaussianMixture`` or ``None`` (when
          the fit was skipped because ``n_valid < 50``).
        * ``component_report``: list of per-component dicts with
          ``component``, ``weight``, ``mean``, ``std``, ``axiom1_mass``,
          ``axiom2_cohesion``, ``is_seed``, ``passes_all_axioms``.
        * ``n_fit_cells``: number of cells fed to the fit.
        * ``bgmm_halted``: ``True`` when no component qualifies as a
          seed.
        * ``threshold``: echo of the input.
    """
    abnormality = np.asarray(abnormality, dtype=np.float64)
    n_cells = int(abnormality.size)
    is_aberrant = np.zeros(n_cells, dtype=bool)
    bgmm_component = np.full(n_cells, -1, dtype=np.int64)

    valid = np.isfinite(abnormality)
    X_all = abnormality[valid].reshape(-1, 1)
    n_valid = int(X_all.shape[0])

    empty_report: List[Dict[str, Any]] = []

    if n_valid < 50:
        return {
            "is_aberrant": is_aberrant,
            "bgmm_component": bgmm_component,
            "aberrant_component_indices": [],
            "bgmm": None,
            "component_report": empty_report,
            "n_fit_cells": n_valid,
            "bgmm_halted": True,
            "threshold": float(threshold),
        }

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import BayesianGaussianMixture

    if train_subsample > 0 and n_valid > train_subsample:
        rng_local = np.random.default_rng(int(random_state))
        train_idx_local = rng_local.choice(
            n_valid, size=int(train_subsample), replace=False
        )
        X_fit = X_all[train_idx_local]
    else:
        X_fit = X_all

    # ``n_init=1`` (sklearn default) preserves the prototype's component
    # spread on the abnormality histogram. The previous ``n_init=10``
    # picked the highest-ELBO init out of 10 restarts, which on this data
    # collapsed the fit to ~3 consolidated components — small components
    # are the easiest to absorb into adjacent large ones at higher ELBO.
    # All BGMM hyperparameters are now caller-supplied so this primitive
    # carries no domain bias; pick ``covariance_type="diag"`` to keep
    # ``covariances_`` shaped ``(n_components, 1)`` rather than
    # ``(n_components, 1, 1)`` (required by the component_report builder
    # below) for 1-D inputs.
    bgmm = BayesianGaussianMixture(
        n_components=int(n_components),
        covariance_type=str(covariance_type),
        weight_concentration_prior=float(weight_concentration_prior),
        weight_concentration_prior_type="dirichlet_process",
        random_state=int(random_state),
        max_iter=int(max_iter),
        reg_covar=float(reg_covar),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Best performing initialization did not converge", category=ConvergenceWarning)
        bgmm.fit(X_fit)

    weights = bgmm.weights_
    means = bgmm.means_[:, 0]
    stds = np.sqrt(bgmm.covariances_[:, 0])

    axiom1 = weights >= float(min_weight)
    axiom2 = stds <= float(max_std)
    cohesive = axiom1 & axiom2
    cohesive_idx = np.where(cohesive)[0]

    if cohesive_idx.size == 0:
        seed_indices: List[int] = []
        bgmm_halted = True
    else:
        # Take all cohesive components with mean >= threshold.
        keep_mask = means[cohesive_idx] >= float(threshold)
        seed_indices = sorted(int(c) for c in cohesive_idx[keep_mask])
        bgmm_halted = len(seed_indices) == 0

    labels_valid = bgmm.predict(X_all)
    bgmm_component[valid] = labels_valid.astype(np.int64)

    if seed_indices:
        seed_arr = np.asarray(seed_indices, dtype=np.int64)
        is_aberrant_valid = np.isin(labels_valid, seed_arr)
        is_aberrant_full = np.zeros(n_cells, dtype=bool)
        is_aberrant_full[valid] = is_aberrant_valid
        is_aberrant = is_aberrant_full

    component_report: List[Dict[str, Any]] = []
    seed_set = set(seed_indices)
    for k in range(int(weights.size)):
        component_report.append({
            "component": int(k),
            "weight": float(weights[k]),
            "mean": float(means[k]),
            "std": float(stds[k]),
            "axiom1_mass": bool(axiom1[k]),
            "axiom2_cohesion": bool(axiom2[k]),
            "is_seed": bool(k in seed_set),
            "passes_all_axioms": bool(k in seed_set),
        })
    component_report.sort(key=lambda r: r["weight"], reverse=True)

    return {
        "is_aberrant": is_aberrant,
        "bgmm_component": bgmm_component,
        "aberrant_component_indices": list(seed_indices),
        "bgmm": bgmm,
        "component_report": component_report,
        "n_fit_cells": int(X_fit.shape[0]),
        "bgmm_halted": bool(bgmm_halted),
        "threshold": float(threshold),
    }


def _merge_overlapping_zone_components(
    means: np.ndarray, weights: np.ndarray, sigmas: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Merge adjacent components whose means lie within ``max(σ_i, σ_{i+1})``
    of each other.

    Such pairs are BGMM oversplits — two components describing one mode.
    The mixture density between them has no local minimum (the narrower
    component sits inside the broader one's 1σ envelope), so treating
    them as separate peaks for trough-picking is wrong. Merging combines
    them into a single component with weighted-mean μ, summed weight,
    and pooled σ (within-component variance plus the centroid-spread
    term so the result reflects the actual width of the merged density).

    Args:
        means / weights / sigmas: 1-D arrays sorted by ``means`` in
            ascending order.

    Returns:
        ``(merged_means, merged_weights, merged_sigmas, merged_pairs)``
        where ``merged_pairs`` is a list of ``(i, j)`` index pairs from
        the *input* arrays that were folded together, for diagnostic
        display. Empty list when no merge fired.
    """
    if len(means) <= 1:
        return means.copy(), weights.copy(), sigmas.copy(), []
    out_m: List[float] = [float(means[0])]
    out_w: List[float] = [float(weights[0])]
    out_s: List[float] = [float(sigmas[0])]
    last_in_idx: List[int] = [0]
    merged_pairs: List[Tuple[int, int]] = []
    for i in range(1, len(means)):
        prev_m, prev_w, prev_s = out_m[-1], out_w[-1], out_s[-1]
        cur_m = float(means[i])
        cur_w = float(weights[i])
        cur_s = float(sigmas[i])
        if abs(cur_m - prev_m) < max(prev_s, cur_s):
            new_w = prev_w + cur_w
            new_m = (prev_m * prev_w + cur_m * cur_w) / new_w
            new_var = (
                prev_w * (prev_s ** 2 + (prev_m - new_m) ** 2)
                + cur_w * (cur_s ** 2 + (cur_m - new_m) ** 2)
            ) / new_w
            out_m[-1] = new_m
            out_w[-1] = new_w
            out_s[-1] = float(np.sqrt(new_var))
            merged_pairs.append((last_in_idx[-1], i))
            last_in_idx[-1] = i
        else:
            out_m.append(cur_m)
            out_w.append(cur_w)
            out_s.append(cur_s)
            last_in_idx.append(i)
    return (
        np.asarray(out_m, dtype=np.float64),
        np.asarray(out_w, dtype=np.float64),
        np.asarray(out_s, dtype=np.float64),
        merged_pairs,
    )


def _drop_spike_bins(
    t: np.ndarray, bin_width: float = 0.005, max_bin_mass: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Pre-fit cell-level mask: drop cells in any 0.005-wide bin holding
    ≥ 5% of total mass, with ±1 bin dilation.

    The spike mask is dilated by one bin on each side via 1-D convolution
    to absorb the spike's noisy floating-point footprint around the
    centre bin.

    Args:
        t: ``(n_cells,)`` position values in ``[0, 1]``.
        bin_width: Width of detection bins. ``0.005`` puts ≈200 bins
            across ``[0, 1]``.
        max_bin_mass: Per-bin mass threshold. A bin holding more than
            ``max_bin_mass`` of the total cells is flagged as a spike.

    Returns:
        ``(t_kept, t_dropped)`` — the cells outside any flagged bin and
        the cells inside.
    """
    edges = np.arange(0.0, 1.0 + bin_width, bin_width)
    counts, _ = np.histogram(t, bins=edges)
    spike = counts > max_bin_mass * len(t)
    if not spike.any():
        return t, t[:0]
    spike = np.convolve(spike.astype(int), np.ones(3, dtype=int), mode="same") > 0
    idx = np.clip((t / bin_width).astype(int), 0, len(counts) - 1)
    drop = spike[idx]
    return t[~drop], t[drop]


def _drop_shoulder_components(
    means: np.ndarray, weights: np.ndarray, sigmas: np.ndarray, n_grid: int = 2001
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drop components whose means are not local maxima of the mixture
    density.

    A BGMM component fitted to a slowly declining tail can sit at a *shoulder*
    of the overall mixture density rather than a local peak: the density is
    monotonic through its mean. Treating such a component as a peak
    fools downstream trough-picking: the "first peak walking outward,
    then take the trough after it" rule walks through a pseudo-peak and
    lands on a non-trough.

    The check is purely topological: evaluate the mixture density on a
    fine grid, find its true local maxima, keep only the components
    whose means coincide with one of those maxima (within
    ``max(σ/2, 2·dx)``). No weight or σ thresholds — a component is
    "real" iff the smooth density it contributes to actually has a bump
    where its mean sits.

    Args:
        means / weights / sigmas: 1-D arrays sorted by ``means``
            ascending.
        n_grid: Number of grid points used to evaluate the mixture
            density on ``[0, 1]``.

    Returns:
        ``(means_kept, weights_kept, sigmas_kept, drop_mask)`` —
        ``drop_mask`` is on the input length, ``True`` for components
        flagged as shoulders.
    """
    n_in = len(means)
    if n_in <= 1:
        return means.copy(), weights.copy(), sigmas.copy(), np.zeros(n_in, dtype=bool)
    grid = np.linspace(0.0, 1.0, n_grid)
    dx = grid[1] - grid[0]
    f = np.zeros_like(grid)
    for m, w, s in zip(means, weights, sigmas):
        f += w / (s * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((grid - m) / s) ** 2)
    is_max = np.zeros(n_grid, dtype=bool)
    is_max[1:-1] = (f[1:-1] > f[:-2]) & (f[1:-1] > f[2:])
    max_locs = grid[is_max]
    keep = np.zeros(n_in, dtype=bool)
    for i, (m, s) in enumerate(zip(means, sigmas)):
        tol = max(0.5 * s, 2 * dx)
        if max_locs.size and np.min(np.abs(max_locs - m)) <= tol:
            keep[i] = True
    drop_mask = ~keep
    return means[keep], weights[keep], sigmas[keep], drop_mask


def _pick_zones_from_components(
    means: np.ndarray, weights: np.ndarray, centre: float = 0.5
) -> Tuple[float, float, str, str]:
    """Pick ``src`` and ``tgt`` from a set of (already shoulder-filtered)
    BGMM components.

    Source side (walk left from ``centre``): ``src`` is the midpoint of
    the two leftmost-of-centre peaks (i.e. the trough between the two
    rightmost peaks on the left side). Falls back to
    ``0.1`` if fewer than two left peaks survive.

    Target side (walk right from ``centre``): two robustness preprocessing
    passes plus the rising-shoulder walk:

      * **Tail-drop** — iteratively strip rightmost peaks whose weight is
        below ``0.20 × max(right_weights)``. Catches post-mask residuals
        like a tiny w≈0.026 bump that would otherwise hijack trough
        placement.
      * **Dominance margin** — the rising-shoulder walk uses ``rw[i] >=
        1.10 × rw[i+1]`` (instead of ``>=``) to declare a real peak,
        giving a 10% margin above BGMM weight-estimation noise so
        near-equal adjacent weights don't trigger a false stop.

    The trough is the midpoint between the first real peak and its
    appropriate adjacent neighbour (the right neighbour for an interior
    real peak, the left neighbour when the real peak is the rightmost
    survivor). Falls back to ``0.1`` if zero or one
    right peaks remain.

    Status strings are emitted as ``"ok"`` or ``"fallback (...)"``.

    Args:
        means / weights: 1-D arrays sorted by ``means`` ascending. Should
            already be shoulder-filtered.
        centre: Boundary between source and target halves; defaults to
            ``0.5``.

    Returns:
        ``(src, tgt, src_status, tgt_status)``.
    """
    order = np.argsort(means)
    means_sorted = means[order]
    weights_sorted = weights[order]
    left = means_sorted[means_sorted < centre]
    right_mask = means_sorted > centre
    right = means_sorted[right_mask].copy()
    rw = weights_sorted[right_mask].copy()

    # --- source side ---
    if len(left) >= 2:
        src = float((left[-1] + left[-2]) / 2.0)
        src_status = "ok"
    else:
        src = 0.1
        src_status = f"fallback (left peaks={len(left)})"

    # --- target side: tail-drop, then rising-shoulder walk with margin ---
    while len(right) >= 2 and rw[-1] < 0.20 * rw.max():
        right = right[:-1]
        rw = rw[:-1]

    if len(right) == 0:
        tgt = 0.1
        tgt_status = "fallback (no right peaks)"
    else:
        real_idx = len(right) - 1
        for i in range(len(right) - 1):
            if rw[i] >= 1.10 * rw[i + 1]:
                real_idx = i
                break
        if real_idx < len(right) - 1:
            one_minus_tgt = float((right[real_idx] + right[real_idx + 1]) / 2.0)
            tgt = 1.0 - one_minus_tgt
            tgt_status = "ok"
        elif real_idx >= 1:
            one_minus_tgt = float((right[real_idx - 1] + right[real_idx]) / 2.0)
            tgt = 1.0 - one_minus_tgt
            tgt_status = "ok"
        else:
            tgt = 0.1
            tgt_status = "fallback (single right peak)"

    return src, tgt, src_status, tgt_status


def _auto_detect_stable_zones(t_raw: np.ndarray) -> Dict[str, Any]:
    """Detect ``(stable_source_zone, stable_target_zone)`` from a vector
    of ``relative_position`` values.

    Every cell from every edge is pooled into one histogram and the two
    numbers apply to all of them. Pooling works because
    ``relative_position`` already maps each edge's own spread onto the
    full ``[0, 1]``.

    Pipeline:

    1. Fit a 1-D BGMM via :func:`_fit_bgmm_aberrant_1d` with selection
       neutralised (we want only the BGMM fit) and ``n_components=15``,
       enough budget to resolve fine target-plateau structure.
    2. Filter components to real peaks via a structural compound rule:
       drop fat valleys with ``σ > 0.10`` and drop low-weight
       ``w < 0.05`` components below ``μ < 0.90`` while protecting
       small sub-peaks above the target plateau.
    3. Merge adjacent components whose means lie within
       ``max(σ_i, σ_{i+1})`` of each other (BGMM oversplits of a single
       mode) via :func:`_merge_overlapping_zone_components`.
    4. Drop *shoulder* components — those whose means are not local
       maxima of the mixture density — via
       :func:`_drop_shoulder_components`. A component fitted to a slow
       declining tail isn't a real peak and shouldn't drive trough
       placement.
    5. Pick the two trough boundaries via
       :func:`_pick_zones_from_components`:

       * Source side — the trough between the two leftmost surviving
         peaks.
       * Target side — tail-drop residuals (rightmost peaks below
         ``0.20 × max(right_weights)``), then walk right with a 10%
         dominance margin (``rw[i] >= 1.10 × rw[i+1]``) to find the
         first real peak; pivot to the appropriate adjacent trough.

    The fit uses every finite input cell so terminal populations remain part of
    the fitted distribution.

    Each side falls back to ``0.1`` independently when its own structure
    is degenerate. Mode is ``"fallback"`` only when both sides fall back;
    otherwise ``"auto"``. Both sides are reset to the fallback when the
    safety check ``src + tgt >= 1.0`` trips. Some fallback paths emit a
    ``RuntimeWarning`` with diagnostic context.

    Args:
        t_raw: ``(n_cells,)`` ``relative_position`` in ``[0, 1]``.
            NaN entries are dropped.

    Returns:
        Dict with:

        * ``stable_source_zone`` (float),
        * ``stable_target_zone`` (float),
        * ``mode`` (``"auto"`` if at least one side computed,
          ``"fallback"`` only when both fell back),
        * ``src_status`` / ``tgt_status`` (``"ok"`` or
          ``"fallback (...)"`` per side),
        * ``components`` — list of ``{"mean", "sigma", "weight"}``
          dicts for the post-merge surviving peaks (empty on early
          fallback),
        * ``merged_pairs`` — list of ``(i, j)`` filtered-component
          index pairs that were folded together (empty if merge
          didn't fire),
        * ``n_fit_cells`` (int),
        * ``reason`` (str, only when ``mode == "fallback"``).
    """
    t_raw = np.asarray(t_raw, dtype=np.float64)
    finite = np.isfinite(t_raw)
    t_filtered = t_raw[finite & (t_raw >= 0.0) & (t_raw <= 1.0)]

    fallback = {
        "stable_source_zone": 0.1,
        "stable_target_zone": 0.1,
        "mode": "fallback",
        "src_status": "fallback",
        "tgt_status": "fallback",
        "components": [],
        "merged_pairs": [],
        "n_fit_cells": int(t_filtered.size),
    }

    if t_filtered.size < 50:
        fallback["reason"] = "too few cells"
        return fallback

    # Selection-neutralised: ``min_weight=0`` and ``max_std=inf`` keep
    # every component cohesive; ``threshold=999`` ensures no seed mass
    # passes and the seed mask comes back empty (we do not use it).
    # ``train_subsample=0`` fits on every kept cell — subsampling drifts
    # the small source-bump components enough to move src by several
    # percent on real datasets, so we pay the ~30 s fit cost for parity
    # with the validated standalone algorithm. ``n_components=15`` (vs.
    # 10) gives EM enough budget to resolve fine target-plateau structure
    # alongside the tall end-of-edge peak.
    bgmm_result = _fit_bgmm_aberrant_1d(
        t_filtered,
        n_components=15,
        min_weight=0.0,
        max_std=float("inf"),
        train_subsample=0,
        threshold=999.0,
        random_state=0,
        weight_concentration_prior=1e-2,
        max_iter=500,
        covariance_type="diag",
        reg_covar=1e-6,
    )
    bgmm = bgmm_result["bgmm"]
    if bgmm is None:
        fallback["reason"] = "bgmm fit skipped"
        return fallback

    means = bgmm.means_[:, 0]
    weights = bgmm.weights_
    sigmas = np.sqrt(bgmm.covariances_[:, 0])
    # Structural compound filter:
    #   * weight floor at 0.05 with a high-μ protection so legitimate
    #     small sub-peaks above the target plateau (μ ≥ 0.90) survive
    #   * fat-valley cap at σ ≤ 0.10
    # No narrow-spike clause: the tall pile at the end of an edge is a
    # real population, and the point is that it gets its own component
    # for the target cut-off to be placed against.
    keep = (
        ((weights >= 0.05) | (means >= 0.90))
        & (sigmas <= 0.10)
    )
    means = means[keep]
    weights = weights[keep]
    sigmas = sigmas[keep]
    order = np.argsort(means)
    means = means[order]
    weights = weights[order]
    sigmas = sigmas[order]

    # Merge BGMM oversplits — adjacent pairs with overlapping σ envelopes
    # describe a single mode and should be fed to pick_zones as one peak.
    merged_means, merged_weights, merged_sigmas, merged_pairs = (
        _merge_overlapping_zone_components(means, weights, sigmas)
    )

    # ``components`` reports the post-merge surviving peaks (preserves
    # the documented contract). The shoulder filter below produces a
    # smaller set that drives trough-picking but isn't reported here.
    components = [
        {"mean": float(m), "sigma": float(s), "weight": float(w)}
        for m, s, w in zip(merged_means, merged_sigmas, merged_weights)
    ]

    # Shoulder filter — drop components whose means are not local maxima
    # of the mixture density (BGMM components fitted to slow declining
    # tails contribute a bump-less Gaussian; their means are not real
    # peaks for trough-picking purposes).
    final_means, final_weights, _final_sigmas, _shoulder_mask = (
        _drop_shoulder_components(merged_means, merged_weights, merged_sigmas)
    )

    src, tgt, src_status, tgt_status = _pick_zones_from_components(
        final_means, final_weights, centre=0.5
    )

    # Counts of the post-shoulder set that drove pick_zones, used in
    # the both-sides-fallback warning text below.
    left_means = final_means[final_means < 0.5]
    right_means = final_means[final_means > 0.5]

    src_ok = src_status == "ok"
    tgt_ok = tgt_status == "ok"

    # If neither side could be picked, emit the same warning shape as
    # before (the test suite matches "left-side and"). Mode = fallback.
    if not src_ok and not tgt_ok:
        fallback["components"] = components
        fallback["merged_pairs"] = merged_pairs
        fallback["src_status"] = src_status
        fallback["tgt_status"] = tgt_status
        fallback["reason"] = "not enough peaks per side"
        return fallback

    # Sanity check: only meaningful when both sides computed.
    if src_ok and tgt_ok:
        if (
            not (0.0 < src < 1.0)
            or not (0.0 < tgt < 1.0)
            or (src + tgt >= 1.0)
        ):
            fallback["components"] = components
            fallback["merged_pairs"] = merged_pairs
            fallback["src_status"] = "fallback (degenerate)"
            fallback["tgt_status"] = "fallback (degenerate)"
            fallback["reason"] = "degenerate (range/sum)"
            return fallback

    return {
        "stable_source_zone": src,
        "stable_target_zone": tgt,
        "mode": "auto",
        "src_status": src_status,
        "tgt_status": tgt_status,
        "components": components,
        "merged_pairs": merged_pairs,
        "n_fit_cells": int(t_filtered.size),
    }


def _fit_bgmm_aberrant_2d(
    jsd: np.ndarray,
    adherence: np.ndarray,
    *,
    n_components: int = 10,
    min_weight: float = 0.01,
    max_std: float = 0.20,
    train_subsample: int = 10000,
    jsd_mean_threshold: float = 0.55,
    adh_mean_threshold: float = 0.30,
    random_state: int = 42,
    stability_repeats: int = 0,
) -> Dict[str, Any]:
    """Fit a 2-D BGMM on ``(JSD, adherence)`` and label aberrant cells.

    A cell is ``is_aberrant`` iff its hard-assigned BGMM component passes
    the three-axiom Hierarchical Component Filter on **either** axis:

    * axis 0 (JSD), direction ``"upper"``, ``mean >= jsd_mean_threshold``;
    * axis 1 (adherence), direction ``"lower"``, ``mean <= adh_mean_threshold``.

    Axioms 1 (mass) and 2 (cohesion) are applied per-axis to filter out
    sub-threshold and diffuse components before the mean test. A component
    is aberrant iff it appears in the **union** of the two valid sets.

    The union is deliberate: this step is tuned for recall, and the
    community-enrichment gate in
    :func:`_label_communities_by_enrichment_fdr` supplies specificity —
    scattered single-axis hits cannot form a significant community.

    Thresholds default to fixed absolute values (``jsd >= 0.55`` and
    ``adh <= 0.30``). They directly affect whether components qualify and
    therefore whether the detector halts without flagging cells.

    Args:
        jsd: ``(n_cells,)`` JSD score. NaN entries are ignored in the fit
            and labelled ``is_aberrant = False``.
        adherence: ``(n_cells,)`` adherence score. Same NaN treatment as
            ``jsd``.
        n_components: BGMM upper bound on components. The Dirichlet-process
            prior shrinks unused components toward zero weight, so the
            effective ``K`` is learned.
        min_weight: Axiom 1 minimum component weight.
        max_std: Axiom 2 maximum per-axis standard deviation.
        train_subsample: Cap on the number of points fed to the fit
            (predict still runs on every valid cell). ``0`` disables.
        jsd_mean_threshold: Axiom 3 threshold for axis 0 (``"upper"``).
        adh_mean_threshold: Axiom 3 threshold for axis 1 (``"lower"``).
        random_state: RNG seed for the BGMM fit and subsampling.
        stability_repeats: If > 0, re-fit the BGMM on ``stability_repeats``
            additional seeds (``random_state + 1`` … ``random_state +
            stability_repeats``) and return a diagnostic ``stability`` dict
            summarising agreement with the primary fit. The primary
            ``is_aberrant`` mask is unchanged — repeats are diagnostic only
            and do not vote.  This measures BGMM fit variance only; it does
            not cover run-to-run jitter in the SNN kNN or Leiden steps.

    Returns:
        Dictionary with

        * ``is_aberrant``: ``(n_cells,)`` bool array.
        * ``aberrant_component_indices``: sorted list of aberrant BGMM
          component indices (union of the two axiom filters).
        * ``bgmm``: the fitted ``BayesianGaussianMixture`` or ``None``
          when the fit was skipped (too few valid cells).
        * ``jsd_threshold`` / ``adh_threshold``: the thresholds actually
          used (the inputs, preserved for the manifest).
        * ``component_report``: list of per-component dicts describing
          weight, per-axis mean/std, and individual axiom-pass flags.
        * ``n_fit_cells``: number of cells actually fed to the fit.
        * ``stability``: present iff ``stability_repeats > 0``; dict with
          ``n_repeats``, ``is_aberrant_agreement`` (mean Jaccard of repeat
          vs. primary masks), ``is_aberrant_agreement_min`` (worst repeat),
          ``n_aberrant_components_mean``, and
          ``n_aberrant_components_std``.
    """
    jsd = np.asarray(jsd, dtype=np.float64)
    adherence = np.asarray(adherence, dtype=np.float64)
    if jsd.shape != adherence.shape:
        raise ValueError(
            "jsd and adherence must share shape; "
            f"got {jsd.shape} vs {adherence.shape}"
        )

    n_cells = int(jsd.size)
    is_aberrant = np.zeros(n_cells, dtype=bool)

    valid = np.isfinite(jsd) & np.isfinite(adherence)
    X_all = np.column_stack([jsd[valid], adherence[valid]])
    n_valid = int(X_all.shape[0])

    empty_report: List[Dict[str, Any]] = []

    if n_valid < 50:
        return {
            "is_aberrant": is_aberrant,
            "aberrant_component_indices": [],
            "bgmm": None,
            "jsd_threshold": float(jsd_mean_threshold),
            "adh_threshold": float(adh_mean_threshold),
            "component_report": empty_report,
            "n_fit_cells": n_valid,
        }

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import BayesianGaussianMixture

    def _fit_one(seed: int) -> Tuple[Any, np.ndarray, List[int], int]:
        """Fit the BGMM with ``seed`` and return ``(bgmm, is_aberrant, aberrant, n_fit)``."""
        if train_subsample > 0 and n_valid > train_subsample:
            rng_local = np.random.default_rng(seed)
            train_idx_local = rng_local.choice(
                n_valid, size=train_subsample, replace=False
            )
            X_fit = X_all[train_idx_local]
        else:
            X_fit = X_all

        model = BayesianGaussianMixture(
            n_components=int(n_components),
            covariance_type="diag",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.4,
            mean_precision_prior=1e-2,
            random_state=int(seed),
            n_init=10,
            max_iter=1500,
            tol=1e-4,
            reg_covar=1e-5,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Best performing initialization did not converge", category=ConvergenceWarning)
            model.fit(X_fit)

        valid_jsd_local, _ = _filter_bgmm_components(
            model,
            min_weight=float(min_weight),
            max_std=float(max_std),
            mean_threshold=float(jsd_mean_threshold),
            axiom_3_direction="upper",
            scoring_axis=0,
        )
        valid_adh_local, _ = _filter_bgmm_components(
            model,
            min_weight=float(min_weight),
            max_std=float(max_std),
            mean_threshold=float(adh_mean_threshold),
            axiom_3_direction="lower",
            scoring_axis=1,
        )
        aberrant_local = sorted(
            {int(i) for i in valid_jsd_local.tolist()}
            | {int(i) for i in valid_adh_local.tolist()}
        )

        labels_local = model.predict(X_all)
        if aberrant_local:
            mask_valid = np.isin(
                labels_local, np.array(aberrant_local, dtype=np.int64)
            )
        else:
            mask_valid = np.zeros(n_valid, dtype=bool)
        mask_full = np.zeros(n_cells, dtype=bool)
        mask_full[valid] = mask_valid
        return model, mask_full, aberrant_local, int(X_fit.shape[0])

    bgmm, is_aberrant, aberrant, n_fit_cells_primary = _fit_one(int(random_state))

    weights = bgmm.weights_
    means = bgmm.means_                     # (K, 2)
    stds = np.sqrt(bgmm.covariances_)       # diag: (K, 2)
    component_report: List[Dict[str, Any]] = []
    for k in range(len(weights)):
        w = float(weights[k])
        m_jsd = float(means[k, 0])
        m_adh = float(means[k, 1])
        s_jsd = float(stds[k, 0])
        s_adh = float(stds[k, 1])
        component_report.append({
            "component": int(k),
            "weight": w,
            "mean_jsd": m_jsd,
            "mean_adh": m_adh,
            "std_jsd": s_jsd,
            "std_adh": s_adh,
            "axiom1_mass": bool(w >= float(min_weight)),
            "axiom2_cohesion_jsd": bool(s_jsd <= float(max_std)),
            "axiom2_cohesion_adh": bool(s_adh <= float(max_std)),
            "axiom3_jsd_upper": bool(m_jsd >= float(jsd_mean_threshold)),
            "axiom3_adh_lower": bool(m_adh <= float(adh_mean_threshold)),
            "passes_all_axioms": bool(k in aberrant),
        })
    component_report.sort(key=lambda r: r["weight"], reverse=True)

    result: Dict[str, Any] = {
        "is_aberrant": is_aberrant,
        "aberrant_component_indices": aberrant,
        "bgmm": bgmm,
        "jsd_threshold": float(jsd_mean_threshold),
        "adh_threshold": float(adh_mean_threshold),
        "component_report": component_report,
        "n_fit_cells": int(n_fit_cells_primary),
    }

    if int(stability_repeats) > 0:
        primary_count = int(is_aberrant.sum())
        jaccards: List[float] = []
        n_aberrant_counts: List[int] = [len(aberrant)]
        for repeat_idx in range(1, int(stability_repeats) + 1):
            _, repeat_mask, repeat_aberrant, _ = _fit_one(
                int(random_state) + repeat_idx
            )
            repeat_count = int(repeat_mask.sum())
            inter = int(np.logical_and(is_aberrant, repeat_mask).sum())
            union = primary_count + repeat_count - inter
            jaccards.append(1.0 if union == 0 else inter / union)
            n_aberrant_counts.append(len(repeat_aberrant))
        counts_arr = np.asarray(n_aberrant_counts, dtype=np.float64)
        result["stability"] = {
            "n_repeats": int(stability_repeats),
            "is_aberrant_agreement": float(np.mean(jaccards)) if jaccards else 1.0,
            "is_aberrant_agreement_min": float(np.min(jaccards)) if jaccards else 1.0,
            "n_aberrant_components_mean": float(counts_arr.mean()),
            "n_aberrant_components_std": float(counts_arr.std(ddof=0)),
        }

    return result


def _label_communities_by_enrichment_fdr(
    community_labels: np.ndarray,
    community_size_per_cell: np.ndarray,
    is_aberrant: np.ndarray,
    adherence: Optional[np.ndarray] = None,
    jsd: Optional[np.ndarray] = None,
    *,
    fdr_alpha: float = 0.08,
    effect_floor: float = 0.10,
    min_community_size: int = 50,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Hypergeometric + BH-FDR enrichment gate over Leiden communities.

    For each Leiden community c with size N_c and BGMM-aberrant count k_c
    (with K total aberrants across N total cells), computes the one-sided
    hypergeometric p-value of seeing >= k_c aberrants under random
    scattering: ``p_c = hypergeom.sf(k_c - 1, N, K, N_c)``.

    A community is ``suspicious`` iff:

    1. Size + label gate: ``size >= min_community_size`` AND ``label >= 0``
       (isolates excluded) AND K > 0.
    2. BH-FDR: ``q_c < fdr_alpha`` after Benjamini-Hochberg correction
       across all eligible communities.
    3. Effect-size floor: ``fraction_aberrant_c >= floor_threshold``,
       where ``floor_threshold = min(baseline + effect_floor,
       1.5 * baseline)`` — the lower of an absolute zone-floor and a
       relative-lift floor (``lift_multiplier=1.5`` hardcoded).

    The dataset is ``halted`` — i.e. all per-cell labels are ``False`` —
    iff no community is suspicious (this includes the ``K == 0``
    BGMM-halted case).

    Args:
        community_labels: ``(n_cells,)`` Leiden labels; ``-1`` for isolates.
        community_size_per_cell: ``(n_cells,)`` per-cell community size.
        is_aberrant: ``(n_cells,)`` bool mask produced by
            :func:`_fit_bgmm_aberrant_1d`.
        adherence: ``(n_cells,)`` adherence; used only for the community
            stats diagnostic (NaNs ignored).
        jsd: ``(n_cells,)`` JSD; used only for the community stats
            diagnostic (NaNs ignored).
        fdr_alpha: BH-FDR level (default 0.08).
        effect_floor: Absolute lift required above the dataset baseline
            aberrant rate (default 0.10).
        min_community_size: Minimum size for a community to be eligible.
        log_fn: Optional logging callback.

    Returns:
        Dictionary with

        * ``is_atypical``: ``(n_cells,)`` bool — broadcast of
          ``suspicious`` over cell-to-community membership.
        * ``is_suspicious_community``: same array (kept for API clarity).
        * ``community_fraction_aberrant``: ``(n_cells,)`` float, per-cell
          view of the community's ``fraction_aberrant``. Size-0
          communities carry ``NaN``.
        * ``community_stats``: list of per-community dicts with ``label``,
          ``size``, ``fraction_aberrant``, ``n_aberrant``,
          ``mean_adherence``, ``mean_jsd``, ``hypergeom_p``,
          ``hypergeom_q``, ``hypergeom_baseline_diff``,
          ``enrichment_floor_threshold``, ``suspicious``. Ineligible
          communities (too small, isolate, or ``K == 0``) carry
          ``hypergeom_p = hypergeom_q = NaN``.
        * ``halted``: ``True`` iff zero suspicious communities.
        * ``n_suspicious_communities``: int.
    """
    # Lazy imports — statsmodels lives in the [atypical] extra so we
    # avoid hauling it in at hector import time.
    from scipy.stats import hypergeom
    from statsmodels.stats.multitest import multipletests

    # Hardcoded mechanism details — part of the gate's calibration but not
    # exposed as separate knobs.
    use_relative_lift = True
    lift_multiplier = 1.5

    community_labels = np.asarray(community_labels, dtype=np.int64)
    community_size_per_cell = np.asarray(community_size_per_cell, dtype=np.int64)
    is_aberrant = np.asarray(is_aberrant, dtype=bool)
    n_cells = int(community_labels.size)
    # Per-community adherence/JSD means are diagnostic only. Callers that
    # don't have these signals (e.g. evaluate_cells_hybrid, which uses a
    # 1-D abnormality score) pass None and the means are filled with NaN.
    adherence_arr = (
        None if adherence is None
        else np.asarray(adherence, dtype=np.float64)
    )
    jsd_arr = (
        None if jsd is None
        else np.asarray(jsd, dtype=np.float64)
    )

    if not (0.0 < float(fdr_alpha) <= 1.0):
        raise ValueError(
            f"fdr_alpha must be in (0, 1]; got {fdr_alpha!r}"
        )
    if float(effect_floor) < 0.0:
        raise ValueError(
            f"effect_floor must be non-negative; got {effect_floor!r}"
        )

    K = int(is_aberrant.sum())
    N = int(n_cells)
    baseline_rate = float(K) / float(N) if N > 0 else 0.0

    # Effect-size floor: lower of absolute and relative.
    abs_threshold = baseline_rate + float(effect_floor)
    if use_relative_lift:
        rel_threshold = float(lift_multiplier) * baseline_rate
        floor_threshold = float(min(abs_threshold, rel_threshold))
    else:
        floor_threshold = float(abs_threshold)

    is_atypical = np.zeros(n_cells, dtype=bool)
    community_frac_per_cell = np.full(n_cells, np.nan, dtype=np.float64)

    # Pass 1: per-community basic stats + hypergeom p (NaN where ineligible).
    rows: List[Dict[str, Any]] = []
    masks_per_row: List[np.ndarray] = []
    for lbl in np.unique(community_labels):
        lbl_int = int(lbl)
        mask = community_labels == lbl_int
        size = int(mask.sum())
        if size == 0:
            continue
        k = int(is_aberrant[mask].sum())
        frac = float(k) / float(size)
        community_frac_per_cell[mask] = frac

        if adherence_arr is None:
            mean_adh = float("nan")
        else:
            adh_slice = adherence_arr[mask]
            adh_finite = adh_slice[np.isfinite(adh_slice)]
            mean_adh = (
                float(np.mean(adh_finite))
                if adh_finite.size else float("nan")
            )
        if jsd_arr is None:
            mean_jsd = float("nan")
        else:
            jsd_slice = jsd_arr[mask]
            jsd_finite = jsd_slice[np.isfinite(jsd_slice)]
            mean_jsd = (
                float(np.mean(jsd_finite))
                if jsd_finite.size else float("nan")
            )

        # Eligibility: real (non-isolate) community above the size gate
        # AND K > 0. With K = 0, hypergeom.sf returns 1.0 — emitting NaN
        # makes the no-test state distinguishable from "tested but not
        # significant" downstream.
        eligible = (
            lbl_int >= 0
            and size >= int(min_community_size)
            and K > 0
        )
        if eligible:
            p = float(hypergeom.sf(k - 1, N, K, size))
        else:
            p = float("nan")

        rows.append({
            "label": lbl_int,
            "size": size,
            "fraction_aberrant": frac,
            "n_aberrant": k,
            "mean_adherence": mean_adh,
            "mean_jsd": mean_jsd,
            "hypergeom_p": p,
            "hypergeom_q": float("nan"),
            "hypergeom_baseline_diff": frac - baseline_rate,
            "enrichment_floor_threshold": floor_threshold,
            "_eligible": eligible,
            "suspicious": False,
        })
        masks_per_row.append(mask)

    # Pass 2: BH-FDR across eligible rows. NaN < x evaluates False, so
    # ineligible rows can never be marked suspicious in pass 3.
    eligible_idx = [i for i, r in enumerate(rows) if r["_eligible"]]
    if eligible_idx:
        p_eligible = np.asarray(
            [rows[i]["hypergeom_p"] for i in eligible_idx], dtype=np.float64,
        )
        _, q_eligible, _, _ = multipletests(
            p_eligible, alpha=float(fdr_alpha), method="fdr_bh",
        )
        for j, i in enumerate(eligible_idx):
            rows[i]["hypergeom_q"] = float(q_eligible[j])

    # Pass 3: suspicious = eligible AND q < alpha AND fraction >= floor.
    n_suspicious = 0
    for i, row in enumerate(rows):
        if not row["_eligible"]:
            continue
        if (
            row["hypergeom_q"] < float(fdr_alpha)
            and row["fraction_aberrant"] >= floor_threshold
        ):
            row["suspicious"] = True
            is_atypical[masks_per_row[i]] = True
            n_suspicious += 1

    # Strip the internal eligibility flag from the public stats.
    community_stats = [
        {k_: v for k_, v in row.items() if k_ != "_eligible"}
        for row in rows
    ]

    halted = n_suspicious == 0
    return {
        "is_atypical": is_atypical,
        "is_suspicious_community": is_atypical.copy(),
        "community_fraction_aberrant": community_frac_per_cell,
        "community_stats": community_stats,
        "halted": bool(halted),
        "n_suspicious_communities": int(n_suspicious),
    }


def _per_class_fairness_mask(
    predictions: np.ndarray,
    is_aberrant: np.ndarray,
    *,
    fdr_alpha: float,
    log_fn: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """Per-class admissibility mask for inheriting bucket-level atypical.

    For each unique predicted class, run a left-tail hypergeometric test
    of the class's BGMM-aberrant count against the dataset baseline:
    ``p_c = hypergeom.cdf(k_c, N, K, n_c)`` — the probability of seeing
    AT MOST ``k_c`` aberrant cells given ``n_c`` draws from a pool of
    ``N`` cells with ``K`` aberrant total. Apply BH-FDR across the
    unique classes. A class with adjusted ``q_c < fdr_alpha`` is
    statistically *below baseline* and gets marked NOT-admissible — its
    cells are protected from inheriting any bucket-level atypical
    verdict (G1 or G5).

    This is the symmetric complement of
    :func:`_label_communities_by_enrichment_fdr` (right-tail, per
    community) — same hypergeom + BH machinery, opposite tail, applied
    over predicted classes instead of over Leiden / G5 buckets.
    Always-on (no opt-out parameter); for symmetry the test reuses the
    same ``fdr_alpha`` as the community gate.

    Edge cases handled by the math itself:
      * ``K == 0`` (BGMM halted): every ``p_c = 1`` → no class
        protected → mask is all True. Caller's BGMM-halt short-circuit
        zeros ``is_atypical`` anyway.
      * Single unique class: degenerate test, ``p = 1`` → not
        protected.
      * Tiny classes are protected only when their adjusted lower-tail
        probability passes the same FDR threshold.

    Args:
        predictions: ``(n_cells,)`` int per-cell predicted-class index
            (any consistent index space; only used for grouping).
        is_aberrant: ``(n_cells,)`` bool BGMM-seed mask.
        fdr_alpha: BH-FDR level — same value used by the community gate.
        log_fn: optional callback for a one-line summary.

    Returns:
        ``(n_cells,)`` bool mask. ``True`` = cell admissible (its class
        is NOT statistically below-baseline aberrant); ``False`` =
        cell protected from atypical inheritance.
    """
    from scipy.stats import hypergeom
    from statsmodels.stats.multitest import multipletests

    predictions = np.asarray(predictions)
    is_aberrant = np.asarray(is_aberrant, dtype=bool)
    N = int(is_aberrant.size)
    K = int(is_aberrant.sum())

    if K == 0 or N == 0:
        return np.ones(N, dtype=bool)

    unique_classes, inverse = np.unique(predictions, return_inverse=True)
    n_per_class = np.bincount(inverse, minlength=unique_classes.size)
    k_per_class = np.bincount(
        inverse, weights=is_aberrant.astype(np.float64),
        minlength=unique_classes.size,
    ).astype(np.int64)

    # Left-tail p: P(X <= k | hypergeom(N, K, n_c)).
    p_per_class = hypergeom.cdf(k_per_class, N, K, n_per_class)
    p_per_class = np.asarray(p_per_class, dtype=np.float64)

    # BH-FDR across the unique classes.
    if unique_classes.size > 1:
        _, q_per_class, _, _ = multipletests(
            p_per_class, alpha=float(fdr_alpha), method="fdr_bh",
        )
    else:
        q_per_class = p_per_class

    protected_classes = q_per_class < float(fdr_alpha)
    admissible_classes = ~protected_classes
    cell_admissible = admissible_classes[inverse]
    return cell_admissible
