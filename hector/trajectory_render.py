"""Rendering backends and geometry helpers for HECTOR trajectories."""

from __future__ import annotations

import colorsys
import base64
import io
import logging
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    
    # Ensure text is exported as actual text (editable) rather than paths
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.sparse import csr_matrix

import networkx as nx

from .trajectory_ontology import (
    discover_inferred_paths,
    compute_elastic_depths,
    get_edge_style,
    SoftEdgeConfig,
    ElasticLayoutConfig,
)
from .trajectory_support import (
    PIE_CHART_EXTERIOR_MARGIN_FRACTION,
    PIE_CHART_OVERLAP_MARGIN,
    PIE_RADIUS_MAX_PX,
    PIE_RADIUS_MIN_PX,
    PIE_RADIUS_HTML_SCALE,
    PIE_RADIUS_PDF_SCALE,
    CLOUD_BASE_RADIUS_PX,
    RIBBON_BASE_HW_PX,
    compute_adaptive_pdf_ribbon_scale,
    SOFT_ARROW_LENGTH_PX,
    SOFT_ARROW_SPACING_PX,
    HIGHLIGHT_MARKER_PDF_SCALE,
    PieChartOverlapResolver,
    AtypicalClustererConfig,
    calculate_bezier_point,
    compute_pie_radius_px,
    generate_circular_arc_points,
)


CELLXGENE_BASE_URL = "https://cellxgene.cziscience.com/cellguide/"
# Visible gap between adjacent shaded clade wedges. Each wedge is inset by
# half of this on both sides, so one full gap is spent per wedge.
RADIAL_GAP = np.radians(1.0)
RADIAL_CENTER_RADIUS = 0.1


def _hex_to_rgb(color: str) -> Tuple[float, float, float]:
    """Convert a #RRGGBB color to normalized RGB."""
    color = color.lstrip('#')
    if len(color) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {color!r}")
    return tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    """Convert normalized RGB values back to #RRGGBB."""
    clipped = [max(0, min(255, round(channel * 255))) for channel in rgb]
    return '#' + ''.join(f'{value:02x}' for value in clipped)


def boost_saturation(hex_color: str, boost: float = 0.18) -> str:
    """Increase HSL saturation by `boost`, clamping at 1.0."""
    r, g, b = int(hex_color[1:3], 16)/255, int(hex_color[3:5], 16)/255, int(hex_color[5:7], 16)/255
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = min(1.0, s + boost)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return f'#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}'


def compute_data_diagonal(node_positions, active_indices):
    """Compute the bounding-box diagonal of the node layout.

    Returns ``sqrt(x_range² + y_range²)`` with a minimum of 1.0. Layouts with
    fewer than two available points use a fallback value of 10.0.
    """
    pts = [node_positions[idx] for idx in active_indices if idx in node_positions]
    if len(pts) < 2:
        return 10.0  # sensible fallback
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)
    diag = np.sqrt(x_range**2 + y_range**2)
    return max(diag, 1.0)  # floor to avoid near-zero


def cloud_blend_hw(t, base_end_hw, cloud_extent, blend_start=0.75):
    """Smoothly widen ribbon terminus to match cloud extent."""
    if t < blend_start:
        return base_end_hw
    progress = (t - blend_start) / (1.0 - blend_start)  # 0→1
    eased = progress ** 3  # cubic ease-in
    target_hw = cloud_extent * 0.8  # derive max from cloud size
    return base_end_hw + eased * (target_hw - base_end_hw)


def apply_ribbon_neck(half_widths, ts, neck_min=0.15, neck_power=0.4):
    """Apply a neck envelope: pinch ribbon at both ends, swell in the middle.

    Uses sin(π*t)^neck_power blended with neck_min so the ribbon never
    fully collapses at the endpoints.

    Parameters
    ----------
    half_widths : np.ndarray  — base half-widths (already tapered)
    ts          : np.ndarray  — parameter values in [0, 1]
    neck_min    : float       — minimum width fraction at endpoints (0=fully pinched)
    neck_power  : float       — controls how quickly the ribbon opens up (lower=faster)
    """
    envelope = neck_min + (1.0 - neck_min) * np.power(np.sin(np.pi * ts), neck_power)
    return half_widths * envelope


def build_ribbon_polygon(points, normals, half_widths):
    """Build ribbon polygon from curve sample points, normals, and half-widths.

    For each sample index *i* the upper and lower edges are computed as::

        upper[i] = point[i] + half_width[i] * normal[i]
        lower[i] = point[i] - half_width[i] * normal[i]

    The closed polygon concatenates the upper edge (forward), the lower edge
    (reversed), and closes back to the first upper point.

    Parameters
    ----------
    points : array-like of shape (N, 2)
        Curve sample positions (x, y).
    normals : array-like of shape (N, 2)
        Unit normal vectors at each sample point.
    half_widths : array-like of shape (N,)
        Half-width of the ribbon at each sample point.

    Returns
    -------
    tuple of (upper_x, upper_y, lower_x, lower_y, poly_x, poly_y)
        All elements are 1-D numpy arrays.  ``poly_x`` / ``poly_y`` form a
        closed polygon suitable for ``go.Scatter(fill='toself')`` or
        ``matplotlib.patches.Polygon``.
    """
    pts = np.asarray(points, dtype=float)
    nrm = np.asarray(normals, dtype=float)
    hw = np.asarray(half_widths, dtype=float)

    # Offset upper and lower edges
    offsets = hw[:, np.newaxis] * nrm          # (N, 2)
    upper = pts + offsets
    lower = pts - offsets

    upper_x = upper[:, 0]
    upper_y = upper[:, 1]
    lower_x = lower[:, 0]
    lower_y = lower[:, 1]

    # Closed polygon: upper forward + lower reversed + close
    poly_x = np.concatenate([upper_x, lower_x[::-1], upper_x[:1]])
    poly_y = np.concatenate([upper_y, lower_y[::-1], upper_y[:1]])

    return upper_x, upper_y, lower_x, lower_y, poly_x, poly_y


def format_cell_count_label(count: int, transition_count: Optional[int] = None) -> str:
    """Format cell count as 'n=X' with optional transition suffix.

    Args:
        count: Non-negative integer stable cell count.
        transition_count: If provided and > 0, appended as '(+Y)' to show
            cells rendered on edges rather than in the cloud.

    Returns:
        Formatted string, e.g. "n=1,234" or "n=14 (+39)".
    """
    label = f"n={count:,}"
    if transition_count is not None and transition_count > 0:
        label += f" (+{transition_count:,})"
    return label


from dataclasses import dataclass


@dataclass(frozen=True)
class PixelGeometry:
    """Affine between data space and the rendered pixel canvas.

    A glyph authored with a pixel radius ``R`` renders at visually consistent
    size by constructing a data-space ellipse with semi-axes
    ``(R * data_per_px_x, R * data_per_px_y)``. This removes the need for
    ``set_aspect('equal')`` / ``scaleanchor`` on anisotropic layouts.

    Assumes rectilinear axes that fully occupy the pixel box (excluding
    margins). For matplotlib, ``axes_width_px`` and ``axes_height_px`` are
    derived from ``(pdf_figsize, pdf_dpi, margins_inches)``; for Plotly,
    from ``(html_plot_width, html_plot_height, margins_px)``.
    """
    axes_width_px: float
    axes_height_px: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def x_span(self) -> float:
        return float(self.x_max - self.x_min)

    @property
    def y_span(self) -> float:
        return float(self.y_max - self.y_min)

    def data_per_px_x(self) -> float:
        return self.x_span / max(self.axes_width_px, 1.0)

    def data_per_px_y(self) -> float:
        return self.y_span / max(self.axes_height_px, 1.0)

    def px_per_data_x(self) -> float:
        return self.axes_width_px / max(self.x_span, 1e-12)

    def px_per_data_y(self) -> float:
        return self.axes_height_px / max(self.y_span, 1e-12)

    def px_aspect(self) -> float:
        """Ratio px_per_data_x / px_per_data_y. 1.0 iff axes are visually isotropic."""
        return self.px_per_data_x() / self.px_per_data_y()

    def px_to_data_ellipse(self, radius_px: float) -> Tuple[float, float]:
        """A pixel-space circle of ``radius_px`` is this ellipse in data space."""
        return (
            float(radius_px) * self.data_per_px_x(),
            float(radius_px) * self.data_per_px_y(),
        )

    def px_to_data_x(self, dx_px: float) -> float:
        return float(dx_px) * self.data_per_px_x()

    def px_to_data_y(self, dy_px: float) -> float:
        return float(dy_px) * self.data_per_px_y()

    def data_to_px_xy(self, dx: float, dy: float) -> Tuple[float, float]:
        return (float(dx) * self.px_per_data_x(), float(dy) * self.px_per_data_y())

    def pie_wedge_polygon(
        self,
        cx: float,
        cy: float,
        radius_px: float,
        theta_a_rad: float,
        theta_b_rad: float,
        n: int = 40,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Closed pie-wedge polygon (xs, ys) in data space for a pixel-circle sector.

        Returned arrays start and end at ``(cx, cy)`` so the polygon is closed.
        """
        rx, ry = self.px_to_data_ellipse(radius_px)
        thetas = np.linspace(theta_a_rad, theta_b_rad, int(max(n, 2)))
        xs = np.concatenate([[cx], cx + rx * np.cos(thetas), [cx]])
        ys = np.concatenate([[cy], cy + ry * np.sin(thetas), [cy]])
        return xs, ys


def pixel_geometry_from_plotly(
    html_plot_width: int,
    html_plot_height: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    margins: Tuple[int, int, int, int] = (20, 20, 120, 120),
) -> PixelGeometry:
    """Build a ``PixelGeometry`` from Plotly layout constants.

    Args:
        html_plot_width: Full canvas width in CSS pixels.
        html_plot_height: Full canvas height in CSS pixels.
        x_range: Tuple ``(x_min, x_max)`` already set on the Plotly xaxis.
        y_range: Tuple ``(y_min, y_max)`` already set on the Plotly yaxis.
        margins: ``(left, right, top, bottom)`` in pixels. The default matches
            the horizontal HTML renderer.

    Returns:
        A ``PixelGeometry`` whose ``axes_width_px`` / ``axes_height_px`` equal
        the plot area after subtracting margins.
    """
    l, r, t, b = margins
    w_px = max(1.0, float(html_plot_width - l - r))
    h_px = max(1.0, float(html_plot_height - t - b))
    return PixelGeometry(
        axes_width_px=w_px,
        axes_height_px=h_px,
        x_min=float(x_range[0]),
        x_max=float(x_range[1]),
        y_min=float(y_range[0]),
        y_max=float(y_range[1]),
    )


def pixel_geometry_from_matplotlib(
    figsize_inches: Tuple[float, float],
    dpi: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    margins_inches: Tuple[float, float, float, float] = (0.9, 0.6, 1.2, 1.2),
) -> PixelGeometry:
    """Build a ``PixelGeometry`` from matplotlib figure constants.

    Uses closed-form figsize × dpi arithmetic so callers don't need a live
    axes handle; this keeps the pixel-geometry available before the final
    ``savefig`` pass.

    Args:
        figsize_inches: ``(width_in, height_in)`` from ``pdf_figsize``.
        dpi: Matplotlib figure DPI (``pdf_dpi``).
        x_range: Tuple ``(x_min, x_max)`` already set via ``ax.set_xlim``.
        y_range: Tuple ``(y_min, y_max)`` already set via ``ax.set_ylim``.
        margins_inches: ``(left, right, top, bottom)`` in inches reserved
            around the axes box for titles, labels, and legends. Defaults
            chosen to match the current horizontal PDF layout.

    Returns:
        ``PixelGeometry`` with axes_width_px / axes_height_px measured in
        display pixels at the given DPI.
    """
    fig_w_in, fig_h_in = float(figsize_inches[0]), float(figsize_inches[1])
    l_in, r_in, t_in, b_in = margins_inches
    w_in = max(0.1, fig_w_in - l_in - r_in)
    h_in = max(0.1, fig_h_in - t_in - b_in)
    return PixelGeometry(
        axes_width_px=w_in * float(dpi),
        axes_height_px=h_in * float(dpi),
        x_min=float(x_range[0]),
        x_max=float(x_range[1]),
        y_min=float(y_range[0]),
        y_max=float(y_range[1]),
    )


def generate_cellxgene_url(node_name: str, base_url: str = CELLXGENE_BASE_URL) -> Optional[str]:
    """
    Generate a CellxGene URL for a given cell ontology term.
    
    Args:
        node_name: Cell ontology term (e.g., 'CL:0000630')
        base_url: Base URL for CellxGene cellguide
        
    Returns:
        URL string (e.g., 'https://cellxgene.cziscience.com/cellguide/CL_0000630')
        or None if not a CL term
        
    Example:
        >>> generate_cellxgene_url('CL:0000630')
        'https://cellxgene.cziscience.com/cellguide/CL_0000630'
    """
    import re
    
    # Match CL:XXXXXXX pattern (Cell Ontology terms)
    match = re.search(r'(CL:\d+)', node_name)
    if match:
        cl_term = match.group(1)
        # Replace : with _ for the URL format
        cl_url_term = cl_term.replace(':', '_')
        return f"{base_url}{cl_url_term}"
    return None


def _compute_radial_legend_anchor(
    layout_engine: 'RadialTreeLayoutEngine',
    outer_r: float
) -> Tuple[float, float, str]:
    """
    Compute the figure-space anchor used for radial legends.

    This keeps the PDF and HTML radial renderers on the same placement logic:
    prefer the open 270-360 degree quadrant, otherwise fall back to bottom-center.
    """
    if layout_engine.is_safe_for_legend():
        legend_angle = np.radians(300)
        plot_diagonal = outer_r * np.sqrt(2)
        legend_distance = 0.05 * plot_diagonal

        legend_x_data = legend_distance * np.cos(legend_angle)
        legend_y_data = legend_distance * np.sin(legend_angle)

        legend_x_fig = 0.5 + (legend_x_data / outer_r) * 0.4
        legend_y_fig = 0.5 + (legend_y_data / outer_r) * 0.4
        return legend_x_fig, legend_y_fig, 'upper_left'

    return 0.5, -0.05, 'lower_center'


def _compute_radial_lineage_legend_anchor_y(
    plot_range: float,
    node_positions: Dict[int, Tuple[float, float]],
    active_indices: List[int],
    cell_positions: Optional[np.ndarray] = None,
    atypical_cluster_infos: Optional[List[Any]] = None,
    gap_fraction_of_plot_range: float = 0.05,
) -> float:
    """Anchor the radial lineage legend just below the occupied tree area."""
    if plot_range <= 0:
        return 0.02

    content_min_y = None

    for idx in active_indices:
        if idx not in node_positions:
            continue
        theta, radius = node_positions[idx]
        node_y = float(radius * np.sin(theta))
        content_min_y = node_y if content_min_y is None else min(content_min_y, node_y)

    if cell_positions is not None and len(cell_positions) > 0:
        cell_y = np.asarray(cell_positions[:, 1], dtype=float) * np.sin(
            np.asarray(cell_positions[:, 0], dtype=float)
        )
        if cell_y.size > 0:
            cell_min_y = float(np.min(cell_y))
            content_min_y = cell_min_y if content_min_y is None else min(content_min_y, cell_min_y)

    for cluster in atypical_cluster_infos or []:
        if getattr(cluster, 'position', None) is None:
            continue
        theta, radius = cluster.position
        cluster_radius = float(getattr(cluster, 'radius', 0.0) or 0.0)
        cluster_min_y = float(radius * np.sin(theta)) - cluster_radius
        content_min_y = cluster_min_y if content_min_y is None else min(content_min_y, cluster_min_y)

    if content_min_y is None:
        return 0.02

    gap_data_units = plot_range * gap_fraction_of_plot_range
    legend_top_y_data = content_min_y - gap_data_units
    legend_top_y_fig = (legend_top_y_data + plot_range) / (2 * plot_range)

    return float(np.clip(legend_top_y_fig, 0.02, 0.95))


def _resolve_radial_html_branch_mode(config) -> str:
    """Resolve radial HTML branch rendering mode from config."""
    configured_mode = str(getattr(config, 'radial_html_branch_mode', 'auto')).lower()
    if configured_mode not in {'auto', 'raster', 'vector'}:
        configured_mode = 'auto'

    if configured_mode == 'auto':
        return 'raster'
    return configured_mode


def _contract_stable_scatter_positions(
    stable_positions: np.ndarray,
    node_center: Tuple[float, float],
    scale: float,
    is_polar: bool = False,
) -> np.ndarray:
    """Scale stable scatter offsets around their anchor when contours are skipped.

    Values below 1 contract offsets, 1 leaves them unchanged, and values above
    1 expand them.
    """
    if len(stable_positions) == 0 or np.isclose(scale, 1.0):
        return stable_positions

    contracted_positions = np.asarray(stable_positions, dtype=float).copy()

    if is_polar:
        center_theta, center_r = node_center
        theta_delta = contracted_positions[:, 0] - center_theta
        theta_delta = (theta_delta + np.pi) % (2 * np.pi) - np.pi
        contracted_positions[:, 0] = np.mod(center_theta + theta_delta * scale, 2 * np.pi)
        contracted_positions[:, 1] = center_r + (contracted_positions[:, 1] - center_r) * scale
        return contracted_positions

    center_xy = np.asarray(node_center, dtype=float)
    return center_xy + (contracted_positions - center_xy) * scale


def _get_node_stable_scatter_positions(
    config,
    node_positions: Dict[int, Tuple[float, float]],
    node_idx: int,
    stable_positions: np.ndarray,
    is_polar: bool = False,
) -> np.ndarray:
    """Return stable scatter positions after applying the configured contraction."""
    return _contract_stable_scatter_positions(
        stable_positions,
        node_positions[node_idx],
        config.stable_scatter_jitter_scale,
        is_polar=is_polar,
    )


def _compute_cartesian_contour_bounds(
    node_idx: int,
    node_center: Tuple[float, float],
    stable_positions: np.ndarray,
    cloud_extents: Optional[Dict[int, float]] = None,
    padding_fraction: float = 0.25,
    min_half_span: float = 0.08,
    extent_cap_multiplier: float = 1.35,
) -> Tuple[float, float, float, float]:
    """Bound a Cartesian KDE window to the stable cloud footprint."""
    center = np.asarray(node_center, dtype=float)
    offsets = np.asarray(stable_positions, dtype=float) - center

    half_span_x = float(np.max(np.abs(offsets[:, 0]))) if len(offsets) else 0.0
    half_span_y = float(np.max(np.abs(offsets[:, 1]))) if len(offsets) else 0.0

    target_half_span_x = half_span_x + max(half_span_x * padding_fraction, min_half_span)
    target_half_span_y = half_span_y + max(half_span_y * padding_fraction, min_half_span)

    if cloud_extents and node_idx in cloud_extents:
        extent_cap = float(cloud_extents[node_idx]) * extent_cap_multiplier
        target_half_span_x = min(target_half_span_x, max(half_span_x, extent_cap))
        target_half_span_y = min(target_half_span_y, max(half_span_y, extent_cap))

    return (
        float(center[0] - target_half_span_x),
        float(center[0] + target_half_span_x),
        float(center[1] - target_half_span_y),
        float(center[1] + target_half_span_y),
    )


def _compute_polar_contour_bounds(
    node_idx: int,
    node_center: Tuple[float, float],
    stable_positions: np.ndarray,
    cloud_extents: Optional[Dict[int, float]] = None,
    padding_fraction: float = 0.25,
    min_theta_half_span: float = 0.02,
    min_r_half_span: float = 0.08,
    extent_cap_multiplier: float = 1.35,
) -> Tuple[float, float, float, float]:
    """Bound a polar KDE window to the stable cloud footprint.

    Under pixel-space sizing the cloud is isotropic in pixel space, so the
    theta/r caps collapse to the same base jitter amount.
    """
    center_theta, center_r = node_center
    positions = np.asarray(stable_positions, dtype=float)

    theta_offsets = (positions[:, 0] - center_theta + np.pi) % (2 * np.pi) - np.pi
    r_offsets = positions[:, 1] - center_r

    half_theta = float(np.max(np.abs(theta_offsets))) if len(theta_offsets) else 0.0
    half_r = float(np.max(np.abs(r_offsets))) if len(r_offsets) else 0.0

    target_half_theta = half_theta + max(half_theta * padding_fraction, min_theta_half_span)
    target_half_r = half_r + max(half_r * padding_fraction, min_r_half_span)

    if cloud_extents and node_idx in cloud_extents:
        extent = float(cloud_extents[node_idx])
        base_jitter = extent * 0.35
        cap_theta = (base_jitter / max(center_r, 1.0)) * extent_cap_multiplier
        cap_r = base_jitter * extent_cap_multiplier
        target_half_theta = min(target_half_theta, max(half_theta, cap_theta))
        target_half_r = min(target_half_r, max(half_r, cap_r))

    return (
        float(center_theta - target_half_theta),
        float(center_theta + target_half_theta),
        float(max(0.0, center_r - target_half_r)),
        float(center_r + target_half_r),
    )


def _render_radial_inferred_path_glyph_data(
    node_positions: Dict[int, Tuple[float, float]],
    inferred_path_weights: Dict[Tuple[int, int], float],
    glyph_spacing_data: float = 0.3,
) -> List[Dict[str, Any]]:
    """Precompute radial soft-edge glyph geometry for vector or raster rendering.

    ``glyph_spacing_data`` is the arrow-to-arrow spacing along each arc in
    data units. Callers with a ``PixelGeometry`` should pass
    ``SOFT_ARROW_SPACING_PX * geom.data_per_px_y()`` so glyph density is
    constant in display pixels regardless of canvas scale.
    """
    glyph_data: List[Dict[str, Any]] = []

    def to_cartesian(theta, r):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    for (src, tgt), weight in inferred_path_weights.items():
        if src not in node_positions or tgt not in node_positions:
            continue

        src_theta, src_r = node_positions[src]
        tgt_theta, tgt_r = node_positions[tgt]

        src_x, src_y = to_cartesian(src_theta, src_r)
        tgt_x, tgt_y = to_cartesian(tgt_theta, tgt_r)

        style = get_edge_style('soft', weight)
        edge_payload: Dict[str, Any] = {
            'style': style,
            'src_xy': (src_x, src_y),
            'tgt_xy': (tgt_x, tgt_y),
        }

        try:
            dist = np.hypot(src_x - tgt_x, src_y - tgt_y)
            dynamic_curvature = max(0.05 * dist, 0.4) if dist > 0 else 0.4

            arc_points, arc_normals = generate_circular_arc_points(
                (src_x, src_y),
                (tgt_x, tgt_y),
                curvature=dynamic_curvature,
                num_points=50
            )

            curve_x = arc_points[:, 0]
            curve_y = arc_points[:, 1]

            segment_lengths = np.sqrt(np.diff(curve_x) ** 2 + np.diff(curve_y) ** 2)
            arc_length = np.sum(segment_lengths)

            n_glyphs = max(2, int(arc_length / glyph_spacing_data))
            indices = np.linspace(0, len(curve_x) - 1, n_glyphs).astype(int)

            arrow_x = []
            arrow_y = []
            arrow_u = []
            arrow_v = []

            for idx in indices:
                x, y = curve_x[idx], curve_y[idx]

                if idx < len(curve_x) - 1:
                    dx = curve_x[idx + 1] - curve_x[idx]
                    dy = curve_y[idx + 1] - curve_y[idx]
                else:
                    dx = curve_x[idx] - curve_x[idx - 1]
                    dy = curve_y[idx] - curve_y[idx - 1]

                length = np.sqrt(dx ** 2 + dy ** 2)
                if length > 1e-9:
                    dx = dx / length
                    dy = dy / length

                arrow_x.append(x)
                arrow_y.append(y)
                arrow_u.append(dx)
                arrow_v.append(dy)

            edge_payload['arrow_x'] = np.array(arrow_x)
            edge_payload['arrow_y'] = np.array(arrow_y)
            edge_payload['arrow_u'] = np.array(arrow_u)
            edge_payload['arrow_v'] = np.array(arrow_v)
            edge_payload['is_fallback_line'] = False
        except (ValueError, Exception):
            edge_payload['is_fallback_line'] = True

        glyph_data.append(edge_payload)

    return glyph_data


def _add_plotly_radial_vector_static_layers(
    fig: 'go.Figure',
    layout_engine: 'RadialTreeLayoutEngine',
    node_positions: Dict[int, Tuple[float, float]],
    edges_data: List[Dict[str, Any]],
    inner_r: float,
    outer_r: float,
    gap: float,
    inferred_path_weights: Dict[Tuple[int, int], float]
) -> None:
    """Add the static radial tree layers as Plotly traces."""
    def to_cartesian(theta, r):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    for theta_start, theta_width, bg_color in layout_engine.get_clade_background_sectors():
        start_angle = theta_start + (gap / 2)
        end_angle = (theta_start + theta_width) - (gap / 2)

        if end_angle <= start_angle:
            continue

        theta_range = np.linspace(start_angle, end_angle, 100)
        inner_x, inner_y = to_cartesian(theta_range, inner_r)
        outer_x, outer_y = to_cartesian(theta_range[::-1], outer_r)

        wedge_x = np.concatenate([inner_x, outer_x, [inner_x[0]]])
        wedge_y = np.concatenate([inner_y, outer_y, [inner_y[0]]])

        fig.add_trace(go.Scatter(
            x=wedge_x,
            y=wedge_y,
            fill='toself',
            fillcolor=bg_color,
            opacity=0.5,
            mode='lines',
            line=dict(width=0, color='rgba(0,0,0,0)'),
            hoverinfo='skip',
            showlegend=False
        ))

    for edge in edges_data:
        for seg in edge['segments']:
            start_theta, start_r = seg['start']
            end_theta, end_r = seg['end']

            start_x, start_y = to_cartesian(start_theta, start_r)
            end_x, end_y = to_cartesian(end_theta, end_r)

            fig.add_trace(go.Scatter(
                x=[start_x, end_x],
                y=[start_y, end_y],
                mode='lines',
                line=dict(color=seg['color'], width=seg['width'] * 0.8),  # radial HTML: shrink taper widths by 20%
                hoverinfo='skip',
                showlegend=False
            ))

    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        marker=dict(size=20, color='#212121'),
        hoverinfo='skip',
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode='markers',
        marker=dict(size=8, color='#ffffff'),
        hoverinfo='skip',
        showlegend=False
    ))

    inferred_path_glyphs = _render_radial_inferred_path_glyph_data(node_positions, inferred_path_weights)
    if inferred_path_glyphs:
        import plotly.figure_factory as ff

        for glyph in inferred_path_glyphs:
            style = glyph['style']
            if glyph['is_fallback_line']:
                src_x, src_y = glyph['src_xy']
                tgt_x, tgt_y = glyph['tgt_xy']
                fig.add_trace(go.Scatter(
                    x=[src_x, tgt_x],
                    y=[src_y, tgt_y],
                    mode='lines',
                    line=dict(color=style['line_color'], width=style['line_width'], dash='dot'),
                    opacity=style['opacity'],
                    hoverinfo='skip',
                    showlegend=False
                ))
                continue

            if len(glyph['arrow_x']) == 0:
                continue

            quiver_fig = ff.create_quiver(
                x=glyph['arrow_x'],
                y=glyph['arrow_y'],
                u=glyph['arrow_u'],
                v=glyph['arrow_v'],
                scale=0.2,
                arrow_scale=0.6,
                line=dict(color=style['line_color'], width=2),
                name='',
                showlegend=False
            )

            for trace in quiver_fig.data:
                trace.opacity = style['opacity']
                trace.showlegend = False
                trace.hoverinfo = 'skip'
                fig.add_trace(trace)


def _render_radial_static_layer_png(
    layout_engine: 'RadialTreeLayoutEngine',
    node_positions: Dict[int, Tuple[float, float]],
    edges_data: List[Dict[str, Any]],
    inferred_path_weights: Dict[Tuple[int, int], float],
    inner_r: float,
    outer_r: float,
    gap: float,
    plot_range: float,
    canvas_px: int = 3200
) -> str:
    """Render static radial layers to a transparent PNG data URI."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def to_cartesian(theta, r):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    fig = Figure(figsize=(8, 8), dpi=canvas_px / 8, facecolor=(1, 1, 1, 0))
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor((1, 1, 1, 0))
    ax.set_xlim(-plot_range, plot_range)
    ax.set_ylim(-plot_range, plot_range)
    ax.set_aspect('equal')
    ax.axis('off')

    for theta_start, theta_width, bg_color in layout_engine.get_clade_background_sectors():
        start_angle = theta_start + (gap / 2)
        end_angle = (theta_start + theta_width) - (gap / 2)
        if end_angle <= start_angle:
            continue

        theta_range = np.linspace(start_angle, end_angle, 100)
        inner_x, inner_y = to_cartesian(theta_range, inner_r)
        outer_x, outer_y = to_cartesian(theta_range[::-1], outer_r)

        wedge_x = np.concatenate([inner_x, outer_x, [inner_x[0]]])
        wedge_y = np.concatenate([inner_y, outer_y, [inner_y[0]]])
        polygon = mpatches.Polygon(
            np.column_stack([wedge_x, wedge_y]),
            closed=True,
            facecolor=bg_color,
            edgecolor='none',
            alpha=0.5,
            zorder=0,
        )
        ax.add_patch(polygon)

    segments_coords = []
    segment_colors = []
    segment_widths = []
    for edge in edges_data:
        for seg in edge['segments']:
            start_theta, start_r = seg['start']
            end_theta, end_r = seg['end']
            start_x, start_y = to_cartesian(start_theta, start_r)
            end_x, end_y = to_cartesian(end_theta, end_r)
            segments_coords.append([(start_x, start_y), (end_x, end_y)])
            segment_colors.append(seg['color'])
            segment_widths.append(seg['width'])
    if segments_coords:
        line_collection = LineCollection(
            segments_coords,
            colors=segment_colors,
            linewidths=segment_widths,
            capstyle='round',
            zorder=1,
        )
        ax.add_collection(line_collection)

    inferred_path_glyphs = _render_radial_inferred_path_glyph_data(node_positions, inferred_path_weights)
    for glyph in inferred_path_glyphs:
        style = glyph['style']
        if glyph['is_fallback_line']:
            src_x, src_y = glyph['src_xy']
            tgt_x, tgt_y = glyph['tgt_xy']
            ax.plot(
                [src_x, tgt_x],
                [src_y, tgt_y],
                color=style['line_color'],
                linewidth=style['line_width'],
                linestyle=':',
                alpha=style['opacity'],
                zorder=2,
            )
            continue

        if len(glyph['arrow_x']) == 0:
            continue

        arrow_scale = 0.18
        ax.quiver(
            glyph['arrow_x'],
            glyph['arrow_y'],
            glyph['arrow_u'] * arrow_scale,
            glyph['arrow_v'] * arrow_scale,
            angles='xy',
            scale_units='xy',
            scale=1.0,
            color=style['line_color'],
            alpha=style['opacity'],
            width=0.0016,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=3.8,
            minlength=0,
            zorder=2,
        )

    outer_pin_radius = plot_range / 50.0
    inner_pin_radius = plot_range / 100.0
    ax.add_patch(mpatches.Circle((0, 0), outer_pin_radius, facecolor='#212121', edgecolor='none', zorder=3))
    ax.add_patch(mpatches.Circle((0, 0), inner_pin_radius, facecolor='#ffffff', edgecolor='none', zorder=4))

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=canvas_px / 8, transparent=True)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode('ascii')
    return f"data:image/png;base64,{encoded}"


def _add_radial_static_underlay(fig: 'go.Figure', data_uri: str, plot_range: float) -> None:
    """Attach a raster underlay aligned to the radial Plotly data axes."""
    fig.add_layout_image(
        dict(
            source=data_uri,
            xref='x',
            yref='y',
            x=-plot_range,
            y=plot_range,
            sizex=2 * plot_range,
            sizey=2 * plot_range,
            xanchor='left',
            yanchor='top',
            sizing='stretch',
            layer='below',
            opacity=1.0,
        )
    )


def _get_plotly_legend_color(color_value: Any, fallback: str = '#666666') -> str:
    """Extract a representative color for a Plotly legend proxy item."""
    if color_value is None:
        return fallback

    if isinstance(color_value, np.ndarray):
        if color_value.size == 0:
            return fallback
        color_value = color_value.tolist()

    if isinstance(color_value, (list, tuple)):
        for item in color_value:
            color = _get_plotly_legend_color(item, fallback=None)
            if color is not None:
                return color
        return fallback

    if isinstance(color_value, str):
        stripped = color_value.strip()
        return stripped if stripped else fallback

    return fallback


def _collect_plotly_legend_entries(fig: 'go.Figure') -> List[Dict[str, str]]:
    """Collect visible legend items from Plotly traces for custom legend rendering."""
    legend_entries: List[Dict[str, str]] = []

    for trace in fig.data:
        if not getattr(trace, 'showlegend', False):
            continue

        label = getattr(trace, 'name', None)
        if not label:
            continue

        mode = getattr(trace, 'mode', '') or ''
        marker = getattr(trace, 'marker', None)
        line = getattr(trace, 'line', None)

        symbol = 'circle'
        color = '#666666'

        if 'markers' in mode and marker is not None:
            marker_symbol = getattr(marker, 'symbol', 'circle') or 'circle'
            if isinstance(marker_symbol, (list, tuple, np.ndarray)):
                marker_symbol = marker_symbol[0]
            symbol = 'diamond' if 'diamond' in str(marker_symbol) else 'circle'
            color = _get_plotly_legend_color(getattr(marker, 'color', None), fallback=color)
        elif 'lines' in mode and line is not None:
            symbol = 'line'
            color = _get_plotly_legend_color(getattr(line, 'color', None), fallback=color)
        elif line is not None:
            symbol = 'line'
            color = _get_plotly_legend_color(getattr(line, 'color', None), fallback=color)

        label_text = str(label)
        if label_text.startswith('Highlight: '):
            label_text = label_text[len('Highlight: '):]

        legend_entries.append({
            'label': label_text,
            'color': color,
            'symbol': symbol,
        })

    return legend_entries


def _add_plotly_radial_custom_legend(
    fig: 'go.Figure',
    legend_entries: List[Dict[str, str]],
    anchor_x: float,
    anchor_y: float,
    anchor_mode: str,
    max_cols: int = 7
) -> None:
    """
    Draw a fixed Plotly legend block in paper coordinates.

    Plotly's built-in legend auto-wraps aggressively when many long labels are
    present. For the radial highlight case we need a PDF-like multi-column block
    anchored from the same figure-space point.
    """
    if not legend_entries:
        return

    n_cols = min(len(legend_entries), max_cols)
    n_rows = int(np.ceil(len(legend_entries) / n_cols))

    font_size = 7
    char_width = 0.0027
    marker_w = 0.008
    marker_h = 0.008
    marker_gap = 0.004
    col_gap = 0.010
    row_h = 0.016
    pad_x = 0.007
    pad_y = 0.007

    column_widths: List[float] = []
    for col_idx in range(n_cols):
        column_items = legend_entries[col_idx::n_cols]
        max_label_len = max(len(item['label']) for item in column_items)
        text_w = max_label_len * char_width
        column_widths.append(marker_w + marker_gap + text_w)

    total_width = (2 * pad_x) + sum(column_widths) + col_gap * max(0, n_cols - 1)
    total_height = (2 * pad_y) + (n_rows * row_h)

    if anchor_mode == 'upper_left':
        x0 = anchor_x
        y1 = anchor_y
        y0 = y1 - total_height
    else:
        x0 = anchor_x - (total_width / 2)
        y0 = anchor_y
        y1 = y0 + total_height

    # Keep the legend inside the fixed Plotly canvas while preserving the same
    # anchor logic as closely as possible.
    x0 = min(max(x0, 0.01), 0.99 - total_width)
    y0 = min(max(y0, 0.01), 0.99 - total_height)
    y1 = y0 + total_height

    fig.add_shape(
        type='rect',
        xref='paper',
        yref='paper',
        x0=x0,
        y0=y0,
        x1=x0 + total_width,
        y1=y1,
        line=dict(color='rgba(0,0,0,0.6)', width=1),
        fillcolor='rgba(255,255,255,0.95)',
        layer='above',
    )

    current_col_x = x0 + pad_x
    for col_idx, col_width in enumerate(column_widths):
        for row_idx in range(n_rows):
            item_idx = row_idx * n_cols + col_idx
            if item_idx >= len(legend_entries):
                continue

            item = legend_entries[item_idx]
            item_y = y1 - pad_y - ((row_idx + 0.5) * row_h)
            marker_y0 = item_y - (marker_h / 2)
            marker_y1 = item_y + (marker_h / 2)

            if item['symbol'] == 'line':
                fig.add_shape(
                    type='line',
                    xref='paper',
                    yref='paper',
                    x0=current_col_x,
                    y0=item_y,
                    x1=current_col_x + marker_w,
                    y1=item_y,
                    line=dict(color=item['color'], width=2),
                    layer='above',
                )
            elif item['symbol'] == 'diamond':
                mid_x = current_col_x + (marker_w / 2)
                mid_y = item_y
                path = (
                    f"M {mid_x},{marker_y1} "
                    f"L {current_col_x + marker_w},{mid_y} "
                    f"L {mid_x},{marker_y0} "
                    f"L {current_col_x},{mid_y} Z"
                )
                fig.add_shape(
                    type='path',
                    xref='paper',
                    yref='paper',
                    path=path,
                    line=dict(color=item['color'], width=1),
                    fillcolor=item['color'],
                    layer='above',
                )
            else:
                fig.add_shape(
                    type='circle',
                    xref='paper',
                    yref='paper',
                    x0=current_col_x,
                    y0=marker_y0,
                    x1=current_col_x + marker_w,
                    y1=marker_y1,
                    line=dict(color=item['color'], width=1),
                    fillcolor=item['color'],
                    layer='above',
                )

            fig.add_annotation(
                x=current_col_x + marker_w + marker_gap,
                y=item_y,
                xref='paper',
                yref='paper',
                text=item['label'],
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                align='left',
                font=dict(size=font_size, color='black'),
            )

        current_col_x += col_width + col_gap


# =============================================================================
# Main Visualizer
# =============================================================================

def _build_plotly_figure(
    self,
    cell_positions: np.ndarray,
    predictions: np.ndarray,
    is_transitioning: np.ndarray,
    is_atypical: np.ndarray,
    anchor_nodes: np.ndarray,
    node_positions: Dict[int, Tuple[float, float]],
    active_indices: List[int],
    node_names: List[str],
    formatted_node_names: List[str],
    visual_tree_edges: List[Tuple[int, int]],
    canonical_tree_edges: List[Tuple[int, int]],
    color_mapper,  # AdaptiveLineageColorMapper or TreeColorMapper
    cell_colors: List[str],
    subgraph: 'AdaptiveOntologySubgraph',
    node_cell_counts: Dict[int, int],
    max_count: int,
    output_path: str,
    atypical_cluster_infos: Optional[List['AtypicalClusterInfo']] = None,
    is_polar: bool = False,  # Radial tree mode
    layout_engine: Optional['RadialTreeLayoutEngine'] = None,  # Radial colors
    inferred_path_weights: Optional[Dict[Tuple[int, int], float]] = None,
    cloud_label_counts: Optional[Dict[int, int]] = None,  # Per-node cell counts for cloud labels
    cloud_transition_counts: Optional[Dict[int, int]] = None,  # Per-node transitioning cell counts
    noise_mask: Optional[np.ndarray] = None,  # Mask for atypical cells left unclustered
    cloud_extents: Optional[Dict[int, float]] = None,  # Per-node cloud extent for ribbon terminus blending
    highlight_data: Optional['HighlightData'] = None,
    highlight_color_mapper: Optional['HighlightColorMapper'] = None
) -> go.Figure:
    """Build interactive Plotly figure with CellxGene links and atypical-cluster hubs."""

    # For radial mode, delegate to dedicated polar Plotly builder
    if is_polar:
        return _build_plotly_radial_figure(self, 
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
            canonical_tree_edges=canonical_tree_edges,  # Pass canonical edges for styling
            layout_engine=layout_engine,
            color_mapper=color_mapper,  # Pass for individual node colors
            cell_colors=cell_colors,
            subgraph=subgraph,
            node_cell_counts=node_cell_counts,
            max_count=max_count,
            output_path=output_path,
            atypical_cluster_infos=atypical_cluster_infos,
            inferred_path_weights=inferred_path_weights,  # Pass filtered inferred paths
            cloud_label_counts=cloud_label_counts,
            cloud_transition_counts=cloud_transition_counts,
            noise_mask=noise_mask,  # Thread unclustered atypical mask to radial builder
            highlight_data=highlight_data,
            highlight_color_mapper=highlight_color_mapper
        )

    atypical_cluster_infos = atypical_cluster_infos or []
    clusterer_config = AtypicalClustererConfig()
    inferred_path_weights = inferred_path_weights or {}
    cloud_extents = cloud_extents or {}

    # --- Axis range + PixelGeometry (computed once, consumed by pies and layout) ---
    # Range is derived from tree node positions plus 8%/5% padding.
    # PixelGeometry then converts pixel-space
    # glyph sizes (pies, later clouds/ribbons) into the correct anisotropic
    # data-space ellipses, without relying on axis aspect locking.
    _html_node_x = [node_positions[idx][0] for idx in active_indices]
    _html_node_y = [node_positions[idx][1] for idx in active_indices]
    _html_x_pad = (max(_html_node_x) - min(_html_node_x)) * 0.08
    _html_y_pad = (max(_html_node_y) - min(_html_node_y)) * 0.05
    _html_x_range = [min(_html_node_x) - _html_x_pad, max(_html_node_x) + _html_x_pad]
    _html_y_range = [min(_html_node_y) - _html_y_pad, max(_html_node_y) + _html_y_pad]
    # Margins MUST match update_layout.margin below exactly so the plot
    # area pixels here equal what Plotly renders. With autoexpand=False,
    # Plotly honours these margins strictly.
    _html_margins = (20, 20, 120, 200)  # (left, right, top, bottom)
    geom = pixel_geometry_from_plotly(
        html_plot_width=self.config.html_plot_width,
        html_plot_height=self.config.html_plot_height,
        x_range=(_html_x_range[0], _html_x_range[1]),
        y_range=(_html_y_range[0], _html_y_range[1]),
        margins=_html_margins,
    )

    # Ribbon half-width is anchored in display pixels. Converted to data
    # units via `geom.data_per_px_y()` so visual ribbon thickness stays
    # constant across tree size, canvas size, elastic mode, and backend.
    ribbon_base_hw = RIBBON_BASE_HW_PX * geom.data_per_px_y()

    fig = go.Figure()

    # Layer 0: Edges - Draw FIRST (bottom layer)
    # =================================================================
    # Separate canonical edges from inferred paths for different styling
    # Canonical edges: tapered ribbon polygons with node color
    # Inferred paths: gray arrows indicating direction (no lines)

    ribbon_traces = []  # Collect ribbon traces for canonical edges
    inferred_paths_data = []  # List of (edge_x, edge_y, weight, arc_points, arc_normals) tuples

    # Build set of canonical edges for priority checking (both directions)
    canonical_edge_set = set()
    for (i, j) in canonical_tree_edges:
        canonical_edge_set.add((i, j))
        canonical_edge_set.add((j, i))  # Add reverse for undirected lookup

    # Build set of inferred path pairs for quick lookup (both directions)
    inferred_path_set = set()
    for (i, j) in inferred_path_weights.keys():
        inferred_path_set.add((i, j))
        inferred_path_set.add((j, i))  # Add reverse for undirected lookup

    for u, v in visual_tree_edges:
        if u not in node_positions or v not in node_positions:
            continue

        start = node_positions[u]
        end = node_positions[v]

        # Canonical edges always use canonical styling.
        is_canonical = (u, v) in canonical_edge_set or (v, u) in canonical_edge_set

        if is_canonical:
            # Canonical edge - render as tapered ribbon polygon
            num_samples = 50
            ts = np.linspace(0, 1, num_samples)
            points = []
            normals = []
            for t in ts:
                pt, n = calculate_bezier_point(t, start, end)
                points.append(pt)
                normals.append(n)

            # Compute tapered half-widths
            start_hw = ribbon_base_hw * (self.config.edge_width / 1.5)
            end_hw = start_hw * self.config.ribbon_taper_ratio

            half_widths = np.array([
                start_hw + t_val * (end_hw - start_hw) for t_val in ts
            ])

            # Apply cloud blend terminus widening if enabled and target has cloud
            if self.config.cloud_blend_enabled and v in cloud_extents:
                ce = cloud_extents[v]
                half_widths = np.array([
                    cloud_blend_hw(t_val, hw, ce)
                    for t_val, hw in zip(ts, half_widths)
                ])

            # Neck envelope: pinch at endpoints, swell in the middle
            half_widths = apply_ribbon_neck(half_widths, ts)

            points_arr = np.array(points)
            normals_arr = np.array(normals)

            # Get child node color and boost saturation
            node_color = color_mapper.get_node_color(v)
            boosted_color = boost_saturation(node_color)

            # Parse hex color to RGB components
            rc = int(boosted_color[1:3], 16)
            gc = int(boosted_color[3:5], 16)
            bc = int(boosted_color[5:7], 16)

            # Render three concentric sub-ribbons (outer→inner) for opacity gradient
            for frac, opacity_mult in [(1.0, 0.2), (0.66, 0.6), (0.33, 1.0)]:
                scaled_hw = half_widths * frac
                _, _, _, _, poly_x, poly_y = build_ribbon_polygon(
                    points_arr, normals_arr, scaled_hw
                )
                fill_opacity = self.config.ribbon_opacity * opacity_mult
                fillcolor = f'rgba({rc},{gc},{bc},{fill_opacity:.3f})'

                ribbon_traces.append(go.Scatter(
                    x=poly_x.tolist(),
                    y=poly_y.tolist(),
                    fill='toself',
                    fillcolor=fillcolor,
                    line=dict(width=0),
                    hoverinfo='skip',
                    showlegend=False
                ))
        else:
            # Not canonical - check if it's a inferred path
            is_inferred_path = (u, v) in inferred_path_set or (v, u) in inferred_path_set

            if is_inferred_path:
                # Circular arc for inferred paths
                # Dynamic curvature: Proportional to distance (Constant Arc Height)
                dist = np.linalg.norm(np.array(start) - np.array(end))
                dynamic_curvature = max(0.05 * dist, 0.4) if dist > 0 else 0.4

                arc_points, arc_normals = generate_circular_arc_points(
                    A=start,
                    B=end,
                    curvature=dynamic_curvature,
                    num_points=21
                )
                edge_x = arc_points[:, 0].tolist()
                edge_y = arc_points[:, 1].tolist()
                weight = inferred_path_weights.get((u, v), inferred_path_weights.get((v, u), 0.5))
                inferred_paths_data.append((edge_x, edge_y, weight, arc_points, arc_normals))
            else:
                # Fallback: treat as canonical with Bezier curve (ribbon rendering)
                num_samples = 50
                ts = np.linspace(0, 1, num_samples)
                points = []
                normals = []
                for t in ts:
                    pt, n = calculate_bezier_point(t, start, end)
                    points.append(pt)
                    normals.append(n)

                start_hw = ribbon_base_hw * (self.config.edge_width / 1.5)
                end_hw = start_hw * self.config.ribbon_taper_ratio
                half_widths = np.array([
                    start_hw + t_val * (end_hw - start_hw) for t_val in ts
                ])

                if self.config.cloud_blend_enabled and v in cloud_extents:
                    ce = cloud_extents[v]
                    half_widths = np.array([
                        cloud_blend_hw(t_val, hw, ce)
                        for t_val, hw in zip(ts, half_widths)
                    ])

                # Neck envelope: pinch at endpoints, swell in the middle
                half_widths = apply_ribbon_neck(half_widths, ts)

                points_arr = np.array(points)
                normals_arr = np.array(normals)

                node_color = color_mapper.get_node_color(v)
                boosted_color = boost_saturation(node_color)
                rc = int(boosted_color[1:3], 16)
                gc = int(boosted_color[3:5], 16)
                bc = int(boosted_color[5:7], 16)

                for frac, opacity_mult in [(1.0, 0.2), (0.66, 0.6), (0.33, 1.0)]:
                    scaled_hw = half_widths * frac
                    _, _, _, _, poly_x, poly_y = build_ribbon_polygon(
                        points_arr, normals_arr, scaled_hw
                    )
                    fill_opacity = self.config.ribbon_opacity * opacity_mult
                    fillcolor = f'rgba({rc},{gc},{bc},{fill_opacity:.3f})'

                    ribbon_traces.append(go.Scatter(
                        x=poly_x.tolist(),
                        y=poly_y.tolist(),
                        fill='toself',
                        fillcolor=fillcolor,
                        line=dict(width=0),
                        hoverinfo='skip',
                        showlegend=False
                    ))

    # Draw canonical edge ribbons (z-order below transitioning cell scatter layer)
    for trace in ribbon_traces:
        fig.add_trace(trace)

    # NOTE: Inferred paths are drawn LATER (after cell clouds/transitioning cells)
    # to ensure correct layer ordering: inferred paths above cells but below labels
    # The inferred_paths_data list is populated above and used in Layer 3.5 below

    # Layer 1: Assigned Transitioning cells - Draw BEFORE clouds
    # =================================================================
    html_cell_size_multiplier = 3  # Horizontal layout: Increased for better visibility in HTML

    # Without separate atypical clusters, retain atypical-transitioning cells in
    # the transitioning layer.
    if self.config.atypical_clusters:
        # Full mode: Only show clean transitioning cells (atypical shown separately)
        assigned_transition_mask = is_transitioning & ~is_atypical
    else:
        # Merged mode: Show ALL transitioning cells (merge atypical into transitioning)
        assigned_transition_mask = is_transitioning

    transition_positions = cell_positions[assigned_transition_mask]
    transition_colors = [cell_colors[i] for i in range(len(cell_colors)) if assigned_transition_mask[i]]
    transition_texts = [formatted_node_names[int(predictions[i])] for i in range(len(predictions)) if assigned_transition_mask[i]]

    if len(transition_positions) > 0:
        fig.add_trace(go.Scattergl(
            x=transition_positions[:, 0], 
            y=transition_positions[:, 1],
            mode='markers',
            marker=dict(
                size=self.config.cell_size * html_cell_size_multiplier, #HTML Horizontal layout
                color=transition_colors,
                opacity=self.config.cell_opacity,
                line=dict(width=0.3, color='white')
            ),
            text=transition_texts,
            name='Transitioning Cells',
        ))

    # Layer 1a: Unclustered atypical cells at their original tree positions.
    # =================================================================
    if self.config.atypical_clusters and noise_mask is not None and np.any(noise_mask):
        noise_count = int(np.sum(noise_mask))
        noise_positions = cell_positions[noise_mask]
        fig.add_trace(go.Scatter(
            x=noise_positions[:, 0].tolist(),
            y=noise_positions[:, 1].tolist(),
            mode='markers',
            marker=dict(size=2, color='grey', opacity=0.3),
            hoverinfo='skip',
            name=f'Unclustered Atypical ({noise_count})',
            showlegend=True
        ))

    # Layer 2: Stable Density Contours - Draw BEFORE atypical cells so they appear underneath
    # =================================================================
    # Only stable cells that are NOT atypical
    stable_mask = ~is_transitioning & ~is_atypical
    stable_positions = cell_positions[stable_mask]
    stable_anchors = anchor_nodes[stable_mask]

    unique_stable_nodes = np.unique(stable_anchors)

    if len(stable_positions) > 0:
        if self.config.display_contours:
            from scipy.stats import gaussian_kde

        for node_idx in unique_stable_nodes:
            node_mask = stable_anchors == node_idx
            node_stable_positions = stable_positions[node_mask]
            if len(node_stable_positions) == 0:
                continue

            use_contour = (
                self.config.display_contours
                and len(node_stable_positions) >= self.config.contour_min_cells
            )

            if not use_contour:
                node_color = color_mapper.get_node_color(node_idx)
                scatter_positions = _get_node_stable_scatter_positions(
                    self.config,
                    node_positions,
                    node_idx,
                    node_stable_positions,
                )
                fig.add_trace(go.Scattergl(
                    x=scatter_positions[:, 0],
                    y=scatter_positions[:, 1],
                    mode='markers',
                    marker=dict(
                        size=self.config.cell_size * html_cell_size_multiplier,
                        color=node_color,
                        opacity=0.8,
                        line=dict(width=0.3, color='white')
                    ),
                    name=f'Stable ({formatted_node_names[node_idx]})',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                continue

            node_color = color_mapper.get_node_color(node_idx)
            if node_color.startswith('#'):
                r = int(node_color[1:3], 16)
                g = int(node_color[3:5], 16)
                b = int(node_color[5:7], 16)
            else:
                r, g, b = 100, 150, 220

            try:
                x = node_stable_positions[:, 0]
                y = node_stable_positions[:, 1]

                xmin, xmax, ymin, ymax = _compute_cartesian_contour_bounds(
                    node_idx=node_idx,
                    node_center=node_positions[node_idx],
                    stable_positions=node_stable_positions,
                    cloud_extents=cloud_extents,
                )

                # Create grid for KDE evaluation
                grid_size = 80  # Higher resolution for smoother contours
                xx, yy = np.meshgrid(
                    np.linspace(xmin, xmax, grid_size),
                    np.linspace(ymin, ymax, grid_size)
                )

                positions = np.vstack([xx.ravel(), yy.ravel()])
                xy_train = np.vstack([x, y])

                # Use config bandwidth if specified, otherwise auto (Scott's rule)
                bw_method = self.config.contour_bandwidth if self.config.contour_bandwidth is not None else 'scott'
                kernel = gaussian_kde(xy_train, bw_method=bw_method)
                z = kernel(positions).reshape(xx.shape)

                # Subtract grid-edge baseline so the rectangular grid perimeter
                # never carries non-zero density — eliminates the square outline
                # that appears when the KDE tail spills past the cloud's tight
                # bounding rectangle.
                z_edge = float(max(
                    z[0, :].max(), z[-1, :].max(),
                    z[:, 0].max(), z[:, -1].max(),
                ))
                z_max = float(z.max())
                if z_max <= 0 or z_max <= z_edge:
                    continue
                z = np.clip(z - z_edge, 0.0, None) / (z_max - z_edge)

                # Colorscale: start from higher threshold to eliminate square shadow
                # Values below 0.08 will be fully transparent
                contour_colorscale = [
                    [0.0,  f'rgba({r},{g},{b},0.0)'],
                    [0.08, f'rgba({r},{g},{b},0.0)'],
                    [0.15, f'rgba({r},{g},{b},0.10)'],
                    [0.25, f'rgba({r},{g},{b},0.20)'],
                    [0.40, f'rgba({r},{g},{b},0.35)'],
                    [0.55, f'rgba({r},{g},{b},0.50)'],
                    [0.75, f'rgba({r},{g},{b},0.65)'],
                    [1.0,  f'rgba({r},{g},{b},0.85)'],
                ]

                fig.add_trace(go.Contour(
                    x=np.linspace(xmin, xmax, grid_size),
                    y=np.linspace(ymin, ymax, grid_size),
                    z=z,
                    colorscale=contour_colorscale,
                    showscale=False,
                    ncontours=30,
                    line_smoothing=1.3,
                    contours=dict(coloring='fill', showlines=False),
                    hoverinfo='skip',
                    opacity=1.0,
                    name=f'Stable: {formatted_node_names[node_idx]}',
                    showlegend=False
                ))

            except Exception:
                scatter_positions = _get_node_stable_scatter_positions(
                    self.config,
                    node_positions,
                    node_idx,
                    node_stable_positions,
                )
                fig.add_trace(go.Scattergl(
                    x=scatter_positions[:, 0],
                    y=scatter_positions[:, 1],
                    mode='markers',
                    marker=dict(
                        size=self.config.cell_size * html_cell_size_multiplier,
                        color=node_color,
                        opacity=0.8,
                        line=dict(width=0.3, color='white')
                    ),
                    name=f'Stable ({formatted_node_names[node_idx]})',
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # Layer 3: Atypical cells as Pie Charts (showing actual cell type composition)
    # =================================================================
    # Pie charts show the composition of atypical cell clusters.
    # Radii are computed relative to the data diagonal for consistent sizing.
    
    if atypical_cluster_infos and self.config.show_pie_charts and self.config.enable_atypical_detection:
        # Pixel-space pie radii: every pie is a circle of `radius_px` pixels
        # regardless of axis anisotropy. The renderer converts that pixel
        # radius to a data-space ellipse at draw time using `geom`.
        max_count = max(c.cell_count for c in atypical_cluster_infos)

        for cluster in atypical_cluster_infos:
            cluster.radius_px = (
                compute_pie_radius_px(cluster.cell_count, max_count) * PIE_RADIUS_HTML_SCALE
            )
            # Data-space y-extent — used by label offsets and any legacy
            # consumer that still reads cluster.radius. Picking the y-axis
            # conversion matches the existing label-offset convention
            # (hub_y + radius + constant).
            cluster.radius = geom.px_to_data_y(cluster.radius_px)

        # Resolve overlaps in pixel space so collisions are isotropic
        # (pies are pixel-space circles regardless of data anisotropy).
        try:
            PieChartOverlapResolver.resolve_for_renderer(
                clusters=atypical_cluster_infos,
                node_positions=node_positions,
                is_polar=False,
                pixel_geometry=geom,
            )
        except Exception as _pie_exc:
            logger.warning("Pie-chart overlap resolution failed: %s", _pie_exc)

        # Draw pie charts at resolved positions
        for cluster in atypical_cluster_infos:
            hub_x, hub_y = cluster.position

            # Get cell type counts for pie chart slices
            if cluster.cell_type_counts:
                sorted_types = sorted(cluster.cell_type_counts.items(), key=lambda x: x[1], reverse=True)
                cell_types = [ct[0] for ct in sorted_types]
                counts = [ct[1] for ct in sorted_types]
            else:
                cell_types = cluster.anchor_nodes
                counts = [1] * len(cell_types)

            total = sum(counts)
            if total == 0:
                continue

            # Compute angles for pie slices
            proportions = [c / total for c in counts]
            angles = np.array([0.0] + list(np.cumsum(proportions))) * 360

            # Draw each pie slice as a data-space ellipse of pixel-circle
            # equivalent — looks round on screen regardless of axis aspect.
            for i, cell_type in enumerate(cell_types):
                theta1, theta2 = angles[i], angles[i + 1]
                if theta2 - theta1 < 1:
                    continue

                n_points = max(10, int((theta2 - theta1) / 5))
                wedge_xs, wedge_ys = geom.pie_wedge_polygon(
                    hub_x, hub_y, cluster.radius_px,
                    np.radians(theta1), np.radians(theta2),
                    n=n_points,
                )
                wedge_x = wedge_xs.tolist()
                wedge_y = wedge_ys.tolist()

                slice_label = formatted_node_names[cell_type] if cell_type < len(formatted_node_names) else f"Type {cell_type}"
                if highlight_color_mapper is not None:
                    try:
                        slice_color = highlight_color_mapper.get_color(slice_label)
                    except KeyError:
                        slice_color = color_mapper.get_node_color(cell_type)
                else:
                    slice_color = color_mapper.get_node_color(cell_type)

                hover_text = (f"<b>{slice_label}</b><br>"
                             f"Count: {counts[i]}<br>"
                             f"Proportion: {proportions[i]:.1%}<br>"
                             f"<i>Atypical_{cluster.cluster_id} (n={cluster.cell_count})</i>")

                fig.add_trace(go.Scatter(
                    x=wedge_x,
                    y=wedge_y,
                    mode='lines',
                    fill='toself',
                    fillcolor=slice_color,
                    line=dict(color=self.config.pie_chart_edge_color, width=self.config.pie_chart_edge_width),
                    hovertext=hover_text,
                    hoverinfo='text',
                    showlegend=False,
                    name=f'Pie: {slice_label}',
                    visible=True
                ))

            # Label sits above the pie — offset by the ellipse's vertical
            # extent (cluster.radius already carries the y-component) plus a
            # small fixed pixel gap converted to data units.
            _label_gap_data_y = geom.px_to_data_y(4.0)
            fig.add_trace(go.Scatter(
                x=[hub_x],
                y=[hub_y + cluster.radius + _label_gap_data_y],
                mode='text',
                text=[f"Atypical_{cluster.cluster_id}<br>(n={cluster.cell_count})"],
                textposition='top center',
                textfont=dict(size=9, color='#000080'),
                hoverinfo='skip',
                showlegend=False,
                visible=True
            ))
    else:
        # Scatter fallback when pie charts are disabled.
        if (highlight_data is None
                and self.config.atypical_scatter
                and self.config.enable_atypical_detection
                and np.any(is_atypical)):
            atypical_positions = cell_positions[is_atypical]
            atypical_texts = [f"Atypical (anchor: {formatted_node_names[int(anchor_nodes[i])]})" 
                               for i in range(len(predictions)) if is_atypical[i]]

            fig.add_trace(go.Scattergl(
                x=atypical_positions[:, 0] if len(atypical_positions) > 0 else [],
                y=atypical_positions[:, 1] if len(atypical_positions) > 0 else [],
                mode='markers',
                marker=dict(
                    size=self.config.cell_size * 3,
                    color='red',
                    opacity=0.6,
                    line=dict(width=0.5, color='darkred')
                ),
                text=atypical_texts if len(atypical_texts) > 0 else [],
                name='Atypical Cells',
                visible=True
            ))

    # Layer 4: Atypical-cluster hub anchor lines (dashed lines to anchor nodes) - ON TOP of clouds
    # =================================================================
    if atypical_cluster_infos and clusterer_config.show_atypical_anchor_lines:
        for cluster in atypical_cluster_infos:
            hub_x, hub_y = cluster.position
            for anchor_idx, weight in zip(cluster.anchor_nodes, cluster.anchor_weights):
                if anchor_idx in node_positions:
                    anchor_x, anchor_y = node_positions[anchor_idx]
                    # Opacity proportional to weight
                    line_opacity = max(0.3, min(0.7, weight))

                    fig.add_trace(go.Scatter(
                        x=[hub_x, anchor_x],
                        y=[hub_y, anchor_y],
                        mode='lines',
                        line=dict(
                            color=f'rgba(147, 112, 219, {line_opacity})',  # Light purple with variable opacity
                            width=1.5,
                            dash='dash'
                        ),
                        hoverinfo='skip',
                        name='AtypicalAnchorLine',  # Tag for visibility control
                        showlegend=False,
                        visible=True  # Controlled by Atypical Cells visibility toggle
                    ))
            
            # Exterior leader line: connect exterior pie back to its barycenter
            if cluster.is_exterior and cluster.barycenter is not None:
                bary_x, bary_y = cluster.barycenter
                fig.add_trace(go.Scatter(
                    x=[hub_x, bary_x],
                    y=[hub_y, bary_y],
                    mode='lines',
                    line=dict(
                        color='rgba(150, 150, 150, 0.5)',
                        width=1.0,
                        dash='dot'
                    ),
                    hoverinfo='skip',
                    name='ExteriorLeaderLine',
                    showlegend=False,
                    visible=True
                ))

    # Layer 4.5: Soft Edges - Draw AFTER cells/clouds but BEFORE node labels
    # =================================================================
    # Inferred paths are rendered here (not with canonical edges) to ensure they
    # appear above cell clouds and transitioning cells but below node labels.
    # Design: Thin ribbon polygons with directional arrow glyphs on top.
    if inferred_paths_data:
        # --- Soft Ribbon Rendering (below arrows) ---
        # Soft ribbons use 40-50% of canonical width, half opacity, no saturation boost
        canonical_start_hw = ribbon_base_hw * (self.config.edge_width / 1.5)
        soft_ribbon_hw = 0.45 * canonical_start_hw  # 45% of canonical width
        soft_fill_opacity = self.config.ribbon_opacity * 0.5

        for edge_x, edge_y, weight, arc_pts, arc_nrm in inferred_paths_data:
            style = get_edge_style('soft', weight)
            soft_color = style.get('color', style.get('line_color', '#AAAAAA'))

            # Parse hex color to RGB
            if soft_color.startswith('#') and len(soft_color) >= 7:
                sr = int(soft_color[1:3], 16)
                sg = int(soft_color[3:5], 16)
                sb = int(soft_color[5:7], 16)
            else:
                sr, sg, sb = 170, 170, 170  # fallback gray

            # Constant half-width (no taper) for soft ribbons
            n_pts = len(arc_pts)
            soft_hws = np.full(n_pts, soft_ribbon_hw)

            # Build ribbon polygon from arc points and normals
            _, _, _, _, poly_x, poly_y = build_ribbon_polygon(
                arc_pts, arc_nrm, soft_hws
            )

            fillcolor = f'rgba({sr},{sg},{sb},{soft_fill_opacity:.3f})'
            fig.add_trace(go.Scatter(
                x=poly_x.tolist(),
                y=poly_y.tolist(),
                fill='toself',
                fillcolor=fillcolor,
                line=dict(width=0),
                hoverinfo='skip',
                showlegend=False
            ))

        # Arrow glyphs removed — inferred paths are distinguished by ribbon opacity alone

    # --- Highlight Scatter Overlay ---
    # Positions come from `_place_stable`, which already scales cloud area
    # by per-node cell count via `compute_count_scaled_cloud_scale` — so
    # highlights on sparse nodes are already tight. No extra contraction.
    if highlight_data is not None:
        for cat, color in highlight_data.color_mapper.get_legend_items():
            cat_mask = [c == cat for c in highlight_data.categories]
            cat_positions = highlight_data.positions[np.array(cat_mask)]
            if len(cat_positions) > 0:
                fig.add_trace(go.Scattergl(
                    x=cat_positions[:, 0].tolist(),
                    y=cat_positions[:, 1].tolist(),
                    mode='markers',
                    marker=dict(
                        size=self.config.cell_size * 2,
                        color=color,
                        opacity=0.8,
                    ),
                    name=f'Highlight: {cat}',
                    showlegend=True,
                    visible=True,
                ))

    # Layer 5: Nodes
    all_node_x = [node_positions[idx][0] for idx in active_indices]
    all_node_y = [node_positions[idx][1] for idx in active_indices]
    all_node_colors = [color_mapper.get_node_color(idx) for idx in active_indices]
    all_node_text = [formatted_node_names[idx] for idx in active_indices]

    # Generate CellxGene URLs for each node
    node_urls = []
    hover_templates = []
    for idx in active_indices:
        name = formatted_node_names[idx]
        # Use original ID for URL generation (not formatted name)
        original_id = node_names[idx]
        url = generate_cellxgene_url(original_id)
        if url:
            node_urls.append(url)
            hover_templates.append(
                f'<b>{name}</b><br>'
                f'<a href="{url}" target="_blank" style="color: #1890ff;">View on CellxGene →</a>'
                '<extra></extra>'
            )
        else:
            node_urls.append("")
            hover_templates.append(f'<b>{name}</b><extra></extra>')

    # Trace: All Node Markers (Clickable)
    fig.add_trace(go.Scatter(
        x=all_node_x, y=all_node_y,
        mode='markers',
        marker=dict(
            size=self.config.node_size * 3.0, #HTML horizontal nodes size
            color=all_node_colors, 
            symbol='circle', 
            line=dict(width=2, color='white')
        ),
        text=all_node_text,
        name='Ontology Nodes',
        customdata=node_urls,
        hovertemplate='<b>%{text}</b><br><span style="font-size:10px; color:#666">Click to view on CellxGene</span><extra></extra>',
        showlegend=False,
    ))

    # --- Node Highlight Overlay (colored rings) ---
    if highlight_color_mapper is not None:
        for idx in active_indices:
            node_name = formatted_node_names[idx]
            try:
                ring_color = highlight_color_mapper.get_color(node_name)
            except KeyError:
                continue  # Node not in highlight categories, skip
            nx_pos, ny_pos = node_positions[idx]
            fig.add_trace(go.Scatter(
                x=[nx_pos], y=[ny_pos],
                mode='markers',
                marker=dict(
                    size=self.config.node_size * 5.0,
                    color='rgba(0,0,0,0)',  # Transparent fill
                    line=dict(width=3, color=ring_color),  # Colored ring
                ),
                name=f'NodeHL: {node_name}',
                showlegend=False,
                hoverinfo='skip',
            ))

    # =================================================================
    # Layer 6: Labels (Clickable links to CellxGene)
    # =================================================================
    # Generate label data for all active nodes
    label_x, label_y, label_text, label_urls = [], [], [], []
    for idx in active_indices:
        # Show labels for ALL active nodes
        label_x.append(node_positions[idx][0])
        label_y.append(node_positions[idx][1] + 0.06)  # Name label ABOVE node
        name = formatted_node_names[idx]
        # Use the full formatted label (respects label_format preference)
        label_text.append(f"<b>{name}</b>")
    # Labels (Text Only)
    fig.add_trace(go.Scatter(
        x=label_x, y=label_y,
        mode='text',
        text=label_text,
        textposition='top center',
        textfont=dict(size=self.config.label_font_size*1.5, #HTML horizontal nodes label size
        color='black', family='sans-serif'),
        name='Labels',
        hoverinfo='skip',
        showlegend=False,
    ))

    # =================================================================
    # Layer 6b: Count Labels (below node) — "n=X" cell count labels
    # =================================================================
    if self.config.show_cloud_cell_counts and cloud_label_counts:
        count_label_x, count_label_y, count_label_text = [], [], []
        for idx in active_indices:
            count = cloud_label_counts.get(idx)
            if count is None:
                continue
            count_label_x.append(node_positions[idx][0])
            count_label_y.append(node_positions[idx][1] - 0.06)  # Count label BELOW node
            trans = cloud_transition_counts.get(idx) if cloud_transition_counts else None
            count_label_text.append(format_cell_count_label(count, transition_count=trans))
        if count_label_x:
            fig.add_trace(go.Scatter(
                x=count_label_x, y=count_label_y,
                mode='text',
                text=count_label_text,
                textposition='bottom center',
                textfont=dict(
                    size=self.config.label_font_size * 1.5,
                    color='black',
                    family='sans-serif'
                ),
                name='Count Labels',
                hoverinfo='skip',
                showlegend=False
            ))

    # =================================================================
    # Lineage Color Legend: One entry per major branch
    # =================================================================
    for label, color in color_mapper.get_legend_items(formatted_names=formatted_node_names):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color, symbol='circle'),
            name=label,
            showlegend=True,
            hoverinfo='skip'
        ))

    # =================================================================
    # Edge Legend: Add legend entries for canonical and inferred paths
    # =================================================================
    # Add dummy traces for edge type legend (invisible points with legend entries)

    # Canonical edge legend entry (solid black line)
    canonical_style = get_edge_style('canonical')
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(
            color=canonical_style['line_color'],
            width=canonical_style['line_width'] * 2,  # Thicker for visibility in legend
            dash=canonical_style['line_dash']
        ),
        name='Canonical Edge (Ontology)',
        showlegend=True,
        hoverinfo='skip'
    ))

    # Inferred path legend entry (arrow markers) - only show if inferred paths exist
    if inferred_path_weights:
        soft_style = get_edge_style('soft', weight=0.7)  # Use mid-range weight for legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                symbol='arrow-right',
                size=12,
                color=soft_style['line_color'],
                angle=0
            ),
            opacity=soft_style['opacity'],
            name='Soft Edge (Data-driven)',
            showlegend=True,
            hoverinfo='skip'
        ))

    # Build Updatemenus for toggle buttons
    updatemenus = []

    # Collect highlight trace indices (scatter + node overlays)
    highlight_trace_indices = []
    for i, trace in enumerate(fig.data):
        if trace.name and (trace.name.startswith('Highlight:') or trace.name.startswith('NodeHL:')):
            highlight_trace_indices.append(i)

    # Collect pie chart trace indices
    pie_trace_indices = [i for i, t in enumerate(fig.data) if t.name and t.name.startswith('Pie:')]

    # Collect pie-related traces (labels and anchor lines)
    pie_label_indices = []
    anchor_line_indices = []
    for i, trace in enumerate(fig.data):
        # Pie chart labels contain "Atypical_"
        if (trace.textfont is not None and 
            hasattr(trace, 'text') and 
            trace.text and 
            any('Atypical_' in str(t) for t in (trace.text if isinstance(trace.text, list) else [trace.text]))):
            pie_label_indices.append(i)
        # Anchor lines connecting atypical clusters to ontology nodes (including exterior leader lines)
        elif trace.name in ('AtypicalAnchorLine', 'ExteriorLeaderLine'):
            anchor_line_indices.append(i)

    # Add "Highlight" toggle button
    # Use trace-specific updates to avoid interfering with other buttons
    if highlight_trace_indices:
        updatemenus.append(dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[{"visible": 'legendonly'}, highlight_trace_indices],
                    label="Hide Highlight",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": True}, highlight_trace_indices],
                    label="Show Highlight",
                    method="restyle"
                )
            ]),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.0,
            xanchor="right",
            y=1.05,
            yanchor="bottom",
            bgcolor="#eeeeee",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11)
        ))

    # Add "Pie Charts" toggle button (includes labels and anchor lines)
    # Use trace-specific updates to avoid interfering with other buttons
    if pie_trace_indices and self.config.atypical_clusters:
        # Combine all pie-related indices
        all_pie_indices = pie_trace_indices + pie_label_indices + anchor_line_indices
        
        updatemenus.append(dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[{"visible": 'legendonly'}, all_pie_indices],
                    label="Hide Pie Charts",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": True}, all_pie_indices],
                    label="Show Pie Charts",
                    method="restyle"
                )
            ]),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.0,
            xanchor="right",
            y=1.10,
            yanchor="bottom",
            bgcolor="#eeeeee",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11)
        ))

    # Layout with Toggle Button for Atypical Cells
    # Build subtitle with optional unclustered-atypical count
    subtitle_parts = [f"Atypical: {np.sum(is_atypical)} cells ({100*np.sum(is_atypical)/len(is_atypical):.1f}%)"]
    if self.config.atypical_clusters and noise_mask is not None and np.any(noise_mask):
        subtitle_parts.append(f"Unclustered atypical: {int(np.sum(noise_mask))} cells")
    subtitle_text = " | ".join(subtitle_parts)

    # Axis ranges were precomputed near the top of this function so
    # pies (and later clouds/ribbons) can size themselves against the same
    # final viewport via `geom`. Reuse them here.
    fig.update_layout(
        title=dict(
            text=f"{self.config.title}<br><sub>{subtitle_text}</sub>", 
            font=dict(size=20)
        ),
        # Bottom horizontal legend — let Plotly size entries from the true text
        # width so long biological names wrap onto new rows instead of
        # overflowing into neighbouring legend items.
        showlegend=True,
        legend=dict(
            orientation='h',
            x=0.5,
            xanchor='center',
            y=-0.05,
            yanchor='top',
            font=dict(size=11),
            itemsizing='constant',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1,
            tracegroupgap=0,        # no gap between trace groups
        ),
        template='plotly_white',
        # No scaleanchor — pies, clouds, and ribbons are sized in pixel
        # space via `PixelGeometry`, so circles render correctly at the
        # natural anisotropic data→pixel mapping. `autoexpand=False` is
        # critical: with a large horizontal legend, Plotly's default
        # autoexpand would silently shrink the plot area vertically and
        # make the pies render as squashed ellipses (actual
        # data_per_px_y < declared data_per_px_y). Locking margins keeps
        # the plot area deterministic and matches PixelGeometry's math.
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='',
                   range=_html_x_range, autorange=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='',
                   range=_html_y_range, autorange=False),
        margin=dict(l=20, r=20, t=120, b=200, autoexpand=False),
        clickmode='event',
        updatemenus=updatemenus,
        width=self.config.html_plot_width,
        height=self.config.html_plot_height
    )

    # Save with custom JavaScript for clickable CellxGene links
    # Save HTML

    # Custom JavaScript with pointer cursor styling for labels
    click_script = """
    <style>
    /* Make label text show pointer cursor on hover */
    .plotly .textpoint { cursor: pointer !important; }
    .plotly text { cursor: pointer !important; }
    </style>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var plot = document.querySelector('.plotly-graph-div');
        if (plot && plot.on) {
            plot.on('plotly_click', function(data) {
                var point = data.points[0];
                // Only respond to clicks on the "Ontology Nodes" trace
                if (point && point.data.name === 'Ontology Nodes') {
                    var url = point.customdata;
                    if (url && url.startsWith('http')) {
                        window.open(url, '_blank');
                    }
                }
            });
        } else {
            console.warn('Plotly plot not found or not initialized.');
        }
    });
    </script>
    """

    # Write HTML with click handler
    # Use 'cdn' to keep file size small and avoid browser parsing issues with massive bundled JS
    html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
    # Inject click script before closing body tag
    html_content = html_content.replace('</body>', click_script + '</body>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ Saved: {output_path}")    
    return fig


def _build_plotly_radial_figure(
    self,
    cell_positions: np.ndarray,
    predictions: np.ndarray,
    is_transitioning: np.ndarray,
    is_atypical: np.ndarray,
    anchor_nodes: np.ndarray,
    node_positions: Dict[int, Tuple[float, float]],
    active_indices: List[int],
    node_names: List[str],
    formatted_node_names: List[str],
    visual_tree_edges: List[Tuple[int, int]],
    canonical_tree_edges: List[Tuple[int, int]],  # Canonical edges for styling
    layout_engine: 'RadialTreeLayoutEngine',
    color_mapper,  # AdaptiveLineageColorMapper or TreeColorMapper
    cell_colors: List[str],
    subgraph: 'AdaptiveOntologySubgraph',
    node_cell_counts: Dict[int, int],
    max_count: int,
    output_path: str,
    atypical_cluster_infos: Optional[List['AtypicalClusterInfo']] = None,
    inferred_path_weights: Optional[Dict[Tuple[int, int], float]] = None,  # Inferred path weights
    cloud_label_counts: Optional[Dict[int, int]] = None,  # Per-node cell counts for cloud labels
    cloud_transition_counts: Optional[Dict[int, int]] = None,  # Per-node transitioning cell counts
    noise_mask: Optional[np.ndarray] = None,  # Mask for atypical cells left unclustered
    highlight_data: Optional['HighlightData'] = None,
    highlight_color_mapper: Optional['HighlightColorMapper'] = None
) -> go.Figure:
    """Build interactive Plotly figure with radial layout (using Cartesian coordinates)."""
    import plotly.graph_objects as go

    atypical_cluster_infos = atypical_cluster_infos or []
    inferred_path_weights = inferred_path_weights or {}

    # Build set of canonical edges for priority checking
    canonical_edge_set = set()
    for (i, j) in canonical_tree_edges:
        canonical_edge_set.add((i, j))
        canonical_edge_set.add((j, i))

    fig = go.Figure()

    # Helper: convert polar (theta, r) to Cartesian (x, y)
    def to_cartesian(theta, r):
        """Convert polar coordinates to Cartesian."""
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    # Get layout parameters
    inner_r = RADIAL_CENTER_RADIUS
    outer_r = self.config.radial_max_radius
    gap = RADIAL_GAP
    plot_range = outer_r * 1.15
    # PixelGeometry for the radial HTML canvas. Radial keeps scaleanchor='y'
    # so data aspect is 1:1 and pie/cloud sizes computed through `geom`
    # render as circles without further correction.
    _radial_html_margins = (10, 10, 40, 60)  # matches update_layout margin dict below
    geom = pixel_geometry_from_plotly(
        html_plot_width=self.config.html_plot_width,
        html_plot_height=self.config.html_plot_height,
        x_range=(-plot_range, plot_range),
        y_range=(-plot_range, plot_range),
        margins=_radial_html_margins,
    )
    # Radial HTML uses xaxis.scaleanchor='y' with equal data ranges, so
    # Plotly letterboxes the wider axis to keep data aspect 1:1 on screen.
    # The raw margins (1580x1500 at a 1600x1600 canvas) are anisotropic —
    # if we keep those in the geom, pie_wedge_polygon produces data-space
    # ellipses (rx < ry) that render as horizontally-squashed pies. Force
    # isotropy here by using the shorter side for both dims.
    _iso_side = min(geom.axes_width_px, geom.axes_height_px)
    geom = PixelGeometry(
        axes_width_px=_iso_side,
        axes_height_px=_iso_side,
        x_min=-plot_range,
        x_max=plot_range,
        y_min=-plot_range,
        y_max=plot_range,
    )
    edges_data = layout_engine.get_radial_edges(node_positions)
    radial_branch_mode = _resolve_radial_html_branch_mode(self.config)
    radial_static_underlay_uri = None

    if radial_branch_mode == 'raster':
        if MATPLOTLIB_AVAILABLE:
            try:
                radial_static_underlay_uri = _render_radial_static_layer_png(
                    layout_engine=layout_engine,
                    node_positions=node_positions,
                    edges_data=edges_data,
                    inferred_path_weights=inferred_path_weights,
                    inner_r=inner_r,
                    outer_r=outer_r,
                    gap=gap,
                    plot_range=plot_range,
                )
            except Exception as exc:
                print(f"[INFO] Radial HTML raster underlay failed ({exc}); falling back to vector branches.")
                radial_branch_mode = 'vector'
        else:
            print("[INFO] Matplotlib unavailable for radial HTML raster underlay; falling back to vector branches.")
            radial_branch_mode = 'vector'

    if radial_branch_mode == 'vector':
        _add_plotly_radial_vector_static_layers(
            fig=fig,
            layout_engine=layout_engine,
            node_positions=node_positions,
            edges_data=edges_data,
            inner_r=inner_r,
            outer_r=outer_r,
            gap=gap,
            inferred_path_weights=inferred_path_weights,
        )


    # =================================================================
    # Layer 2a: Transitioning Cells
    # =================================================================
    cell_thetas = cell_positions[:, 0]
    cell_rs = cell_positions[:, 1]
    html_cell_size_multiplier = 2  # Radial layout: Smaller multiplier for compact radial display
    # Radial HTML only: expand each stable cell's polar offset from its
    # anchor node by this factor so the cell cloud's spatial footprint is
    # ~20% larger than in the horizontal layout. Marker size stays fixed
    # — we're scaling the cluster extent, not the dot diameter.
    radial_cloud_position_scale = 1.2

    # Without separate atypical clusters, retain atypical-transitioning cells in
    # the transitioning layer.
    if self.config.atypical_clusters:
        # Full mode: Only show clean transitioning cells (atypical shown separately)
        assigned_transition_mask = is_transitioning & ~is_atypical
    else:
        # Merged mode: Show ALL transitioning cells (merge atypical into transitioning)
        assigned_transition_mask = is_transitioning

    if np.any(assigned_transition_mask):
        trans_thetas = cell_thetas[assigned_transition_mask]
        trans_rs = cell_rs[assigned_transition_mask]
        trans_x, trans_y = to_cartesian(trans_thetas, trans_rs)
        trans_colors = [cell_colors[i] for i in range(len(cell_colors)) if assigned_transition_mask[i]]

        fig.add_trace(go.Scattergl(
            x=trans_x,
            y=trans_y,
            mode='markers',
            marker=dict(
                size=self.config.cell_size * html_cell_size_multiplier * 1.5, ##HTML Radial Layout
                color=trans_colors,
                opacity=0.7,
                line=dict(width=0.2, color='white')
            ),
            name='Transitioning',
            showlegend=True
        ))

    # =================================================================
    # Layer 2a: Unclustered atypical cells at their original tree positions.
    # =================================================================
    if self.config.atypical_clusters and noise_mask is not None and np.any(noise_mask):
        noise_count = int(np.sum(noise_mask))
        noise_positions = cell_positions[noise_mask]
        noise_thetas = noise_positions[:, 0]
        noise_rs = noise_positions[:, 1]
        noise_x, noise_y = to_cartesian(noise_thetas, noise_rs)

        fig.add_trace(go.Scattergl(
            x=noise_x,
            y=noise_y,
            mode='markers',
            marker=dict(size=2, color='grey', opacity=0.3),
            hoverinfo='skip',
            name=f'Unclustered Atypical ({noise_count})',
            showlegend=True
        ))

    # =================================================================
    # Layer 2b: Stable Cells (Density Clouds)
    # =================================================================
    stable_mask = ~is_transitioning & ~is_atypical
    if np.any(stable_mask):
        stable_positions = cell_positions[stable_mask]
        stable_anchors = anchor_nodes[stable_mask]

        unique_stable_nodes = np.unique(stable_anchors)

        for node_idx in unique_stable_nodes:
            if node_idx not in active_indices:
                continue

            node_mask = stable_anchors == node_idx
            node_positions_polar = stable_positions[node_mask]

            if len(node_positions_polar) == 0:
                continue

            node_color = color_mapper.get_node_color(node_idx)
            use_cloud_effect = (
                self.config.display_contours
                and len(node_positions_polar) >= self.config.contour_min_cells
            )

            # Expand each cell's offset from its anchor to grow the cloud's
            # spatial footprint. Radial HTML only; horizontal is untouched.
            cloud_positions_polar = _contract_stable_scatter_positions(
                node_positions_polar,
                node_positions[node_idx],
                radial_cloud_position_scale,
                is_polar=True,
            )

            if use_cloud_effect:
                node_x, node_y = to_cartesian(cloud_positions_polar[:, 0], cloud_positions_polar[:, 1])
                fig.add_trace(go.Scattergl(
                    x=node_x,
                    y=node_y,
                    mode='markers',
                    marker=dict(
                        size=self.config.cell_size * html_cell_size_multiplier,
                        color=node_color,
                        opacity=0.15,
                        line=dict(width=0)
                    ),
                    name=f'Stable ({formatted_node_names[node_idx]})',
                    showlegend=False,
                    hoverinfo='skip'
                ))

                fig.add_trace(go.Scattergl(
                    x=node_x,
                    y=node_y,
                    mode='markers',
                    marker=dict(
                        size=self.config.cell_size * html_cell_size_multiplier,
                        color=node_color,
                        opacity=0.4,
                        line=dict(width=0)
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            else:
                scatter_positions = _get_node_stable_scatter_positions(
                    self.config,
                    node_positions,
                    node_idx,
                    cloud_positions_polar,
                    is_polar=True,
                )
                scatter_x, scatter_y = to_cartesian(scatter_positions[:, 0], scatter_positions[:, 1])
                fig.add_trace(go.Scattergl(
                    x=scatter_x,
                    y=scatter_y,
                    mode='markers',
                    marker=dict(
                        size=self.config.cell_size * html_cell_size_multiplier,
                        color=node_color,
                        opacity=0.8,
                        line=dict(width=0.3, color='white')
                    ),
                    name=f'Stable ({formatted_node_names[node_idx]})',
                    showlegend=False,
                    hoverinfo='skip'
                ))

        if not self.config.display_contours:
            fig.add_trace(go.Scattergl(
                x=[], y=[],
                mode='markers',
                marker=dict(
                    size=self.config.cell_size * html_cell_size_multiplier,
                    color='#666666',
                    opacity=0.8,
                    line=dict(width=0.3, color='white')
                ),
                name='Stable',
                showlegend=True,
                hoverinfo='skip'
            ))

    # =================================================================
    # Layer 3: Pie Charts (Atypical Cells)
    # =================================================================
    # Pie charts show the composition of atypical cell clusters.
    # Radii are computed relative to the data diagonal for consistent sizing.
    clusterer_config = AtypicalClustererConfig()

    if atypical_cluster_infos and self.config.show_pie_charts and self.config.enable_atypical_detection:
        # Pixel-space pie radii. Radial HTML keeps scaleanchor='y', so the
        # pixel geometry is effectively isotropic and pies render as circles.
        # Radial HTML only: bump pie radius by 20% over the shared HTML scale.
        radial_pie_html_radius_scale = 1.2
        max_count = max(c.cell_count for c in atypical_cluster_infos)
        for cluster in atypical_cluster_infos:
            cluster.radius_px = (
                compute_pie_radius_px(cluster.cell_count, max_count)
                * PIE_RADIUS_HTML_SCALE
                * radial_pie_html_radius_scale
            )
            cluster.radius = geom.px_to_data_y(cluster.radius_px)

        # Resolve overlaps in pixel space (handles polar→Cartesian conversion internally)
        PieChartOverlapResolver.resolve_for_renderer(
            clusters=atypical_cluster_infos,
            node_positions=node_positions,
            is_polar=True,
            pixel_geometry=geom,
        )

        # Draw pies at resolved positions
        for cluster in atypical_cluster_infos:
            # Position is in polar coordinates (theta, r) - convert to Cartesian for drawing
            theta, r = cluster.position
            hub_x = r * np.cos(theta)
            hub_y = r * np.sin(theta)

            # Get cell type counts for pie slices
            if cluster.cell_type_counts:
                sorted_types = sorted(cluster.cell_type_counts.items(),
                                     key=lambda x: x[1], reverse=True)
                cell_types = [ct[0] for ct in sorted_types]
                counts = [ct[1] for ct in sorted_types]
            else:
                cell_types = cluster.anchor_nodes
                counts = [1] * len(cell_types)

            total = sum(counts)
            if total == 0:
                continue

            # Compute angles for pie slices
            proportions = [c / total for c in counts]
            angles = np.array([0.0] + list(np.cumsum(proportions))) * 2 * np.pi

            # Draw each pie slice as a pixel-circle → data-space ellipse
            for i, cell_type in enumerate(cell_types):
                theta1, theta2 = angles[i], angles[i + 1]

                if theta2 - theta1 < 0.02:  # Skip tiny slices
                    continue

                slice_label = formatted_node_names[cell_type] if cell_type < len(formatted_node_names) else f"Type {cell_type}"
                if highlight_color_mapper is not None:
                    try:
                        slice_color = highlight_color_mapper.get_color(slice_label)
                    except KeyError:
                        slice_color = color_mapper.get_node_color(cell_type)
                else:
                    slice_color = color_mapper.get_node_color(cell_type)

                n_points = max(10, int((theta2 - theta1) * 20 / np.pi))
                wedge_xs, wedge_ys = geom.pie_wedge_polygon(
                    hub_x, hub_y, cluster.radius_px, theta1, theta2, n=n_points,
                )
                wedge_x = wedge_xs.tolist()
                wedge_y = wedge_ys.tolist()

                hover_text = (f"<b>{slice_label}</b><br>"
                             f"Count: {counts[i]}<br>"
                             f"Proportion: {proportions[i]:.1%}<br>"
                             f"<i>Atypical_{cluster.cluster_id} (n={cluster.cell_count})</i>")

                fig.add_trace(go.Scatter(
                    x=wedge_x,
                    y=wedge_y,
                    mode='lines',
                    fill='toself',
                    fillcolor=slice_color,
                    line=dict(
                        color=self.config.pie_chart_edge_color,
                        width=self.config.pie_chart_edge_width
                    ),
                    hovertext=hover_text,
                    hoverinfo='text',
                    showlegend=False,
                    name=f'Pie: {slice_label}'
                ))

            # Add label above pie — fixed 4 px gap above the ellipse top
            _label_gap_data_y = geom.px_to_data_y(4.0)
            fig.add_trace(go.Scatter(
                x=[hub_x],
                y=[hub_y + cluster.radius + _label_gap_data_y],
                mode='text',
                text=[f"Amb_{cluster.cluster_id}<br>(n={cluster.cell_count})"],
                textposition='top center',
                textfont=dict(size=9, color='#000080'),
                hoverinfo='skip',
                showlegend=False
            ))

    else:
        # Fallback: red scatter for atypical cells when pie charts disabled
        # Skip if highlight_data is present — the highlight overlay handles coloring
        # Skip if atypical_scatter=False — user explicitly wants atypical cells merged
        if (highlight_data is None
                and self.config.atypical_scatter
                and self.config.enable_atypical_detection
                and np.any(is_atypical)):
            atyp_thetas = cell_thetas[is_atypical]
            atyp_rs = cell_rs[is_atypical]
            atyp_x, atyp_y = to_cartesian(atyp_thetas, atyp_rs)

            fig.add_trace(go.Scattergl(
                x=atyp_x,
                y=atyp_y,
                mode='markers',
                marker=dict(
                    size=self.config.cell_size * html_cell_size_multiplier*2, ##HTML Radial layout
                    color='red',
                    opacity=0.7,
                    line=dict(width=0.5, color='darkred')
                ),
                name='Atypical',
                showlegend=True
            ))

    # =================================================================
    # Layer 3b: Atypical-cluster hub anchor lines (dashed lines to anchor nodes)
    # =================================================================
    if atypical_cluster_infos and clusterer_config.show_atypical_anchor_lines:
        for cluster in atypical_cluster_infos:
            # Position is in polar coordinates (theta, r) - convert to Cartesian for drawing
            theta, r = cluster.position
            hub_x = r * np.cos(theta)
            hub_y = r * np.sin(theta)
            for anchor_idx, weight in zip(cluster.anchor_nodes, cluster.anchor_weights):
                if anchor_idx in node_positions:
                    anchor_theta, anchor_r = node_positions[anchor_idx]
                    anchor_x, anchor_y = to_cartesian(anchor_theta, anchor_r)
                    line_opacity = max(0.3, min(0.7, weight))

                    fig.add_trace(go.Scatter(
                        x=[hub_x, anchor_x],
                        y=[hub_y, anchor_y],
                        mode='lines',
                        line=dict(
                            color=f'rgba(147, 112, 219, {line_opacity})',
                            width=1.5,
                            dash='dash'
                        ),
                        hoverinfo='skip',
                        name='AtypicalAnchorLine',
                        showlegend=False,
                        visible=True
                    ))
            
    # --- Highlight Scatter Overlay ---
    # Positions already carry cloud-jitter scaled by per-node cell count,
    # so sparse-anchor highlights are naturally tight. No extra contraction.
    if highlight_data is not None:
        for cat, color in highlight_data.color_mapper.get_legend_items():
            cat_mask = [c == cat for c in highlight_data.categories]
            cat_positions = highlight_data.positions[np.array(cat_mask)]
            if len(cat_positions) > 0:
                # Convert polar to Cartesian
                hl_thetas = cat_positions[:, 0]
                hl_rs = cat_positions[:, 1]
                hl_x, hl_y = to_cartesian(hl_thetas, hl_rs)
                fig.add_trace(go.Scattergl(
                    x=hl_x,
                    y=hl_y,
                    mode='markers',
                    marker=dict(
                        size=self.config.cell_size * 2,
                        color=color,
                        opacity=0.8,
                    ),
                    name=f'Highlight: {cat}',
                    showlegend=True,
                    visible=True,
                ))

    # =================================================================
    # Layer 4: Nodes (Ontology Nodes)
    # =================================================================
    node_xs = []
    node_ys = []
    node_colors_list = []
    node_texts = []
    node_urls = []

    for idx in active_indices:
        theta, r = node_positions[idx]
        x, y = to_cartesian(theta, r)
        node_xs.append(x)
        node_ys.append(y)
        node_colors_list.append(color_mapper.get_node_color(idx))

        name = formatted_node_names[idx]
        count = node_cell_counts.get(idx, 0)
        # Use full formatted name for hover text (respects label_format preference)
        node_texts.append(f"{name}<br>n={count}")

        # Use original ID for URL generation (not formatted name)
        original_id = node_names[idx]
        url = generate_cellxgene_url(original_id)
        node_urls.append(url)

    # Nodes as markers only (labels via annotations with fishbone rotation)
    fig.add_trace(go.Scatter(
        x=node_xs,
        y=node_ys,
        mode='markers',
        marker=dict(
            size=self.config.node_size * 1.5,
            color=node_colors_list,
            symbol='diamond',
            line=dict(width=1.5, color='white')
        ),
        hovertext=node_texts,
        hoverinfo='text',
        customdata=node_urls,
        name='Ontology Nodes',
        showlegend=True
    ))

    # --- Node Highlight Overlay (colored rings) ---
    if highlight_color_mapper is not None:
        for idx in active_indices:
            node_name = formatted_node_names[idx]
            try:
                ring_color = highlight_color_mapper.get_color(node_name)
            except KeyError:
                continue
            n_theta, n_r = node_positions[idx]
            n_x, n_y = to_cartesian(np.array([n_theta]), np.array([n_r]))
            fig.add_trace(go.Scatter(
                x=n_x.tolist(), y=n_y.tolist(),
                mode='markers',
                marker=dict(
                    size=self.config.node_size * 2.5,
                    color='rgba(0,0,0,0)',
                    line=dict(width=3, color=ring_color),
                ),
                name=f'NodeHL: {node_name}',
                showlegend=False,
                hoverinfo='skip',
            ))

    # Add labels using node-centered fishbone approach (matching PDF)
    # Labels positioned along ±25° fishbone directions from each node

    # Helper function: compute dynamic label distance based on node depth
    def compute_label_offset(node_r: float) -> float:
        """
        Compute label distance from node.

        Inner nodes receive a larger inverse-radius offset than outer nodes.
        """
        base_offset = 0.1
        inverse_scale = 0.5 / (node_r + 2.0)
        return base_offset + inverse_scale

    # Fishbone tilt configuration (matching PDF)
    fishbone_tilt_deg = 25  # degrees from radial

    for i, idx in enumerate(active_indices):
        theta, r = node_positions[idx]
        deg_angle = np.degrees(theta) % 360
        label_text = formatted_node_names[idx]

        # Node-centered circular sector approach
        label_distance = compute_label_offset(r)

        # Node position in Cartesian
        node_x = r * np.cos(theta)
        node_y = r * np.sin(theta)

        # Fishbone direction for name label (+25°)
        fishbone_angle_name = theta + np.radians(fishbone_tilt_deg)

        # Calculate offset in the fishbone direction (in Cartesian screen coords)
        # ax/ay are in pixels when using showarrow, so convert label_distance to pixels
        # Approximate: 100 pixels per data unit (adjust based on plot scale)
        pixels_per_unit = 80  
        # Reduce label distance by half for tighter spacing
        ax_offset = (label_distance * 0.5) * np.cos(fishbone_angle_name) * pixels_per_unit
        ay_offset = -(label_distance * 0.5) * np.sin(fishbone_angle_name) * pixels_per_unit  # Negative for screen coords

        # Text rotation and anchoring
        if 90 < deg_angle < 270:
            text_angle = deg_angle + 180 + fishbone_tilt_deg
            # Left side: text extends left, anchor at right end (closest to node)
            xanchor = 'right'
        else:
            text_angle = deg_angle + fishbone_tilt_deg
            # Right side: text extends right, anchor at left end (closest to node)
            xanchor = 'left'

        fig.add_annotation(
            x=node_x,
            y=node_y,
            text=label_text,
            textangle=-text_angle,
            xanchor=xanchor,
            yanchor='middle',
            showarrow=True,
            arrowhead=0,
            arrowwidth=0.1,  # Minimum allowed value
            arrowcolor='rgba(0,0,0,0)',
            ax=ax_offset,
            ay=ay_offset,
            font=dict(size=self.config.label_font_size * 1.1, color='black'),
            bgcolor='rgba(255,255,255,0.7)'
        )

    # =================================================================
    # Layer 4b: Count Labels (node-centered fishbone)
    # =================================================================
    if self.config.show_cloud_cell_counts and cloud_label_counts:
        for i, idx in enumerate(active_indices):
            count = cloud_label_counts.get(idx)
            if count is None:
                continue

            theta, r = node_positions[idx]
            deg_angle = np.degrees(theta) % 360

            # Node-centered circular sector approach
            label_distance = compute_label_offset(r)

            # Node position in Cartesian
            node_x = r * np.cos(theta)
            node_y = r * np.sin(theta)


            # Fishbone direction for count label (-25°)
            fishbone_angle_count = theta - np.radians(fishbone_tilt_deg)

            # Calculate offset in the fishbone direction
            pixels_per_unit = 80
            # Reduce label distance by half for tighter spacing
            ax_offset = (label_distance * 0.5) * np.cos(fishbone_angle_count) * pixels_per_unit
            ay_offset = -(label_distance * 0.5) * np.sin(fishbone_angle_count) * pixels_per_unit

            # Text rotation and anchoring
            if 90 < deg_angle < 270:
                text_angle = deg_angle + 180 - fishbone_tilt_deg
                # Left side: text extends left, anchor at right end (closest to node)
                xanchor = 'right'
            else:
                text_angle = deg_angle - fishbone_tilt_deg
                # Right side: text extends right, anchor at left end (closest to node)
                xanchor = 'left'

            trans = cloud_transition_counts.get(idx) if cloud_transition_counts else None
            fig.add_annotation(
                x=node_x,
                y=node_y,
                text=format_cell_count_label(count, transition_count=trans),
                textangle=-text_angle,
                xanchor=xanchor,
                yanchor='middle',
                showarrow=True,
                arrowhead=0,
                arrowwidth=0.1,  # Minimum allowed value
                arrowcolor='rgba(0,0,0,0)',
                ax=ax_offset,
                ay=ay_offset,
                font=dict(size=self.config.label_font_size * 1.1, color='black'),
                bgcolor='rgba(255,255,255,0.7)'
            )

    # =================================================================
    # Layout Configuration (Cartesian with locked aspect ratio)
    # =================================================================
    # Build toggle buttons
    updatemenus = []

    # Collect highlight trace indices (scatter + node overlays)
    highlight_trace_indices = []
    for i, trace in enumerate(fig.data):
        if trace.name and (trace.name.startswith('Highlight:') or trace.name.startswith('NodeHL:')):
            highlight_trace_indices.append(i)

    # Collect pie chart trace indices
    pie_trace_indices = [i for i, t in enumerate(fig.data) if t.name and t.name.startswith('Pie:')]

    # Collect pie-related traces (labels and anchor lines)
    pie_label_indices = []
    anchor_line_indices = []
    for i, trace in enumerate(fig.data):
        # Pie chart labels contain "Amb_"
        if (trace.textfont is not None and 
            hasattr(trace, 'text') and 
            trace.text and 
            any('Amb_' in str(t) for t in (trace.text if isinstance(trace.text, list) else [trace.text]))):
            pie_label_indices.append(i)
        # Anchor lines connecting atypical clusters to ontology nodes (including exterior leader lines)
        elif trace.name in ('AtypicalAnchorLine', 'ExteriorLeaderLine'):
            anchor_line_indices.append(i)

    # Add "Highlight" toggle button
    # Use trace-specific updates to avoid interfering with other buttons
    if highlight_trace_indices:
        updatemenus.append(dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[{"visible": 'legendonly'}, highlight_trace_indices],
                    label="Hide Highlight",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": True}, highlight_trace_indices],
                    label="Show Highlight",
                    method="restyle"
                )
            ]),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.0,
            xanchor="right",
            y=1.05,
            yanchor="bottom",
            bgcolor="#eeeeee",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11)
        ))

    # Add "Pie Charts" toggle button (includes labels and anchor lines)
    # Use trace-specific updates to avoid interfering with other buttons
    if pie_trace_indices and self.config.atypical_clusters:
        # Combine all pie-related indices
        all_pie_indices = pie_trace_indices + pie_label_indices + anchor_line_indices
        
        updatemenus.append(dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[{"visible": 'legendonly'}, all_pie_indices],
                    label="Hide Pie Charts",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": True}, all_pie_indices],
                    label="Show Pie Charts",
                    method="restyle"
                )
            ]),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.0,
            xanchor="right",
            y=1.10,
            yanchor="bottom",
            bgcolor="#eeeeee",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11)
        ))

    # Build title with optional unclustered-atypical count
    title_text = self.config.title
    if self.config.atypical_clusters and noise_mask is not None and np.any(noise_mask):
        noise_count = int(np.sum(noise_mask))
        title_text = f"{title_text}<br><sub>Unclustered atypical: {noise_count} cells</sub>"

    legend_entries = _collect_plotly_legend_entries(fig)
    legend_x_fig, legend_y_fig, legend_anchor_mode = _compute_radial_legend_anchor(
        layout_engine,
        outer_r
    )

    # --- Unified custom legend ---------------------------------------------
    # Radial HTML used to render two legends: Plotly's native legend (state
    # entries scraped from showlegend=True traces: Transitioning, Atypical,
    # Unclustered Atypical, Stable, Ontology Nodes) *plus* the custom
    # annotation-based lineage legend. Depending on which state traces were
    # present, the two legends overlapped (looked like one) or stacked
    # (looked like two). Merge every entry into one custom legend and
    # suppress Plotly's native legend so the output is consistent across
    # atypical_clusters / atypical_scatter / highlight combinations.
    lineage_legend_entries = [
        {'label': label, 'color': color, 'symbol': 'circle'}
        for label, color in color_mapper.get_legend_items(
            formatted_names=formatted_node_names
        )
    ]

    # Deduplicate by (label, color) — avoids double-listing the same cell
    # type when it appears both as a state entry and a lineage entry.
    combined_legend_entries: List[Dict[str, str]] = []
    _seen_labels = set()
    for entry in list(legend_entries) + lineage_legend_entries:
        key = (entry['label'], entry['color'])
        if key in _seen_labels:
            continue
        _seen_labels.add(key)
        combined_legend_entries.append(entry)

    if combined_legend_entries:
        legend_anchor_x = 0.72
        legend_anchor_y = _compute_radial_lineage_legend_anchor_y(
            plot_range=plot_range,
            node_positions=node_positions,
            active_indices=active_indices,
            cell_positions=cell_positions,
            atypical_cluster_infos=atypical_cluster_infos,
        )
        _add_plotly_radial_custom_legend(
            fig,
            combined_legend_entries,
            legend_anchor_x,
            legend_anchor_y,
            'upper_left',
            max_cols=7,
        )

    legend_config = None
    use_custom_legend = True

    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            font=dict(size=14)
        ),
        xaxis=dict(
            visible=False,
            range=[-plot_range, plot_range],
            scaleanchor='y',  # Lock aspect ratio
            scaleratio=1
        ),
        yaxis=dict(
            visible=False,
            range=[-plot_range, plot_range],
            scaleanchor="x",
            scaleratio=1,
        ),
        showlegend=False,
        legend=legend_config,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=10, r=10, t=40, b=60),
        updatemenus=updatemenus if updatemenus else None,
        width=1600,
        height=1600
    )

    if radial_static_underlay_uri is not None:
        _add_radial_static_underlay(fig, radial_static_underlay_uri, plot_range)

    # =================================================================
    # Save HTML with click handler
    # =================================================================
    click_script = """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        var plot = document.querySelector('.js-plotly-plot');
        if (plot) {
            plot.on('plotly_click', function(data) {
                var point = data.points[0];
                if (point && point.customdata) {
                    window.open(point.customdata, '_blank');
                }
            });
        }
    });
    </script>
    """

    # CSS to remove margins
    margin_css = """
    <style>
    html, body {
        margin: 0;
        padding: 0;
        overflow: auto;
    }
    .js-plotly-plot, .plotly {
        margin: 0 auto;
        display: block;
    }
    </style>
    """

    html_content = fig.to_html(include_plotlyjs='cdn', full_html=True)
    # Inject CSS and script
    html_content = html_content.replace('</head>', margin_css + '</head>')
    html_content = html_content.replace('</body>', click_script + '</body>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ Saved radial HTML: {output_path}")
    return fig


def _build_matplotlib_figure(
    self,
    cell_positions: np.ndarray,
    predictions: np.ndarray,
    is_transitioning: np.ndarray,
    is_atypical: np.ndarray,
    anchor_nodes: np.ndarray,
    node_positions: Dict[int, Tuple[float, float]],
    active_indices: List[int],
    node_names: List[str],
    formatted_node_names: List[str],
    visual_tree_edges: List[Tuple[int, int]],
    canonical_tree_edges: List[Tuple[int, int]],  # Canonical edges for styling
    color_mapper,  # AdaptiveLineageColorMapper or TreeColorMapper
    cell_colors: List[str],
    subgraph: 'AdaptiveOntologySubgraph',
    node_cell_counts: Dict[int, int],
    max_count: int,
    output_path: str,
    atypical_cluster_infos: Optional[List['AtypicalClusterInfo']] = None,
    inferred_path_weights: Optional[Dict[Tuple[int, int], float]] = None,
    cloud_label_counts: Optional[Dict[int, int]] = None,
    cloud_transition_counts: Optional[Dict[int, int]] = None,
    noise_mask: Optional[np.ndarray] = None,  # Mask for atypical cells left unclustered
    cloud_extents: Optional[Dict[int, float]] = None,  # Per-node cloud extent for ribbon terminus blending
    highlight_data: Optional['HighlightData'] = None,
    highlight_color_mapper: Optional['HighlightColorMapper'] = None
):
    """Build static matplotlib figure for PDF export with atypical-cluster hubs."""

    atypical_cluster_infos = atypical_cluster_infos or []
    clusterer_config = AtypicalClustererConfig()
    inferred_path_weights = inferred_path_weights or {}
    cloud_extents = cloud_extents or {}

    # Build set of canonical edges for priority checking
    canonical_edge_set = set()
    for (i, j) in canonical_tree_edges:
        canonical_edge_set.add((i, j))
        canonical_edge_set.add((j, i))

    # Build set of inferred paths for quick lookup
    inferred_path_set = set()
    for (i, j) in inferred_path_weights.keys():
        inferred_path_set.add((i, j))
        inferred_path_set.add((j, i))

    # Build PDF

    # --- Nature/Science Style Setup ---
    # 1. Use clean sans-serif fonts (Arial/Helvetica)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.5

    fig, ax = plt.subplots(figsize=self.config.pdf_figsize)

    # --- Axis range + PixelGeometry (computed once, consumed by pies and layout) ---
    # Matches the 8%/5% padding used historically below. Hoisting the
    # computation here lets pixel-space glyph sizing reference the final
    # viewport through `geom`.
    _all_x = [node_positions[idx][0] for idx in active_indices]
    _all_y = [node_positions[idx][1] for idx in active_indices]
    _x_pad = (max(_all_x) - min(_all_x)) * 0.08
    _y_pad = (max(_all_y) - min(_all_y)) * 0.05
    _pdf_x_range = (min(_all_x) - _x_pad, max(_all_x) + _x_pad)
    _pdf_y_range = (min(_all_y) - _y_pad, max(_all_y) + _y_pad)
    _pdf_margins_in = (0.9, 0.6, 1.2, 1.2)  # (left, right, top, bottom) inches
    geom = pixel_geometry_from_matplotlib(
        figsize_inches=self.config.pdf_figsize,
        dpi=self.config.pdf_dpi,
        x_range=_pdf_x_range,
        y_range=_pdf_y_range,
        margins_inches=_pdf_margins_in,
    )

    # Ribbon half-width in display pixels, converted to data-space y-units
    # via `geom`. Consistent across tree size, canvas size, elastic mode.
    ribbon_base_hw = RIBBON_BASE_HW_PX * geom.data_per_px_y()

    # Adaptive PDF ribbon scale: keeps ribbon width proportional to node
    # spacing so large DAGs don't saturate the plot.
    _node_spacing_px = self.config.y_scale * geom.px_per_data_y()
    _pdf_ribbon_scale = compute_adaptive_pdf_ribbon_scale(_node_spacing_px)

    # Layer 0: Canonical Edges (Ribbon Polygons) - Draw first
    # =================================================================
    from matplotlib.patches import Polygon as MplPolygon

    # Collect inferred paths for later rendering (Layer 2.5) to ensure correct z-order
    inferred_paths_data = []  # List of (edge_x, edge_y, weight, arc_points, arc_normals) for inferred paths

    for u, v in visual_tree_edges:
        if u not in node_positions or v not in node_positions:
            continue

        start = node_positions[u]
        end = node_positions[v]

        # Canonical edges use canonical styling.
        is_canonical = (u, v) in canonical_edge_set or (v, u) in canonical_edge_set

        if is_canonical:
            # Canonical edge - render as tapered ribbon polygon
            num_samples = 50
            ts = np.linspace(0, 1, num_samples)
            points = []
            normals = []
            for t in ts:
                pt, n = calculate_bezier_point(t, start, end)
                points.append(pt)
                normals.append(n)

            # Compute tapered half-widths
            start_hw = ribbon_base_hw * (self.config.edge_width / 1.5) * _pdf_ribbon_scale
            end_hw = start_hw * self.config.ribbon_taper_ratio

            half_widths = np.array([
                start_hw + t_val * (end_hw - start_hw) for t_val in ts
            ])

            # Apply cloud blend terminus widening if enabled and target has cloud
            if self.config.cloud_blend_enabled and v in cloud_extents:
                ce = cloud_extents[v]
                half_widths = np.array([
                    cloud_blend_hw(t_val, hw, ce)
                    for t_val, hw in zip(ts, half_widths)
                ])

            # Neck envelope: pinch at endpoints, swell in the middle
            half_widths = apply_ribbon_neck(half_widths, ts)

            points_arr = np.array(points)
            normals_arr = np.array(normals)

            # Get child node color and boost saturation
            node_color = color_mapper.get_node_color(v)
            boosted_color = boost_saturation(node_color)

            # Parse hex color to RGB tuple
            rc = int(boosted_color[1:3], 16) / 255
            gc = int(boosted_color[3:5], 16) / 255
            bc = int(boosted_color[5:7], 16) / 255

            # Render three concentric sub-ribbons (outer→inner) for opacity gradient
            for frac, opacity_mult in [(1.0, 0.2), (0.66, 0.6), (0.33, 1.0)]:
                scaled_hw = half_widths * frac
                _, _, _, _, poly_x, poly_y = build_ribbon_polygon(
                    points_arr, normals_arr, scaled_hw
                )
                fill_alpha = self.config.ribbon_opacity * opacity_mult
                verts = list(zip(poly_x, poly_y))
                patch = MplPolygon(verts, closed=True,
                                   facecolor=(rc, gc, bc, fill_alpha),
                                   edgecolor='none', zorder=0)
                ax.add_patch(patch)
        else:
            # Not canonical - check if it's a inferred path
            is_soft = (u, v) in inferred_path_set or (v, u) in inferred_path_set

            if is_soft:
                # Circular arc for inferred paths
                # Dynamic curvature: Proportional to distance (Constant Arc Height)
                dist = np.linalg.norm(np.array(start) - np.array(end))
                dynamic_curvature = max(0.05 * dist, 0.4) if dist > 0 else 0.4

                arc_points, arc_normals = generate_circular_arc_points(
                    A=start,
                    B=end,
                    curvature=dynamic_curvature,
                    num_points=21
                )
                edge_x = arc_points[:, 0].tolist()
                edge_y = arc_points[:, 1].tolist()
                weight = inferred_path_weights.get((u, v), inferred_path_weights.get((v, u), 0.5))
                # Store for later rendering (Layer 2.5) to ensure correct z-order
                inferred_paths_data.append((edge_x, edge_y, weight, arc_points, arc_normals))
            else:
                # Fallback: treat as canonical with Bezier curve (ribbon rendering)
                num_samples = 50
                ts = np.linspace(0, 1, num_samples)
                points = []
                normals = []
                for t in ts:
                    pt, n = calculate_bezier_point(t, start, end)
                    points.append(pt)
                    normals.append(n)

                start_hw = ribbon_base_hw * (self.config.edge_width / 1.5) * _pdf_ribbon_scale
                end_hw = start_hw * self.config.ribbon_taper_ratio
                half_widths = np.array([
                    start_hw + t_val * (end_hw - start_hw) for t_val in ts
                ])

                if self.config.cloud_blend_enabled and v in cloud_extents:
                    ce = cloud_extents[v]
                    half_widths = np.array([
                        cloud_blend_hw(t_val, hw, ce)
                        for t_val, hw in zip(ts, half_widths)
                    ])

                # Neck envelope: pinch at endpoints, swell in the middle
                half_widths = apply_ribbon_neck(half_widths, ts)

                points_arr = np.array(points)
                normals_arr = np.array(normals)

                node_color = color_mapper.get_node_color(v)
                boosted_color = boost_saturation(node_color)
                rc = int(boosted_color[1:3], 16) / 255
                gc = int(boosted_color[3:5], 16) / 255
                bc = int(boosted_color[5:7], 16) / 255

                for frac, opacity_mult in [(1.0, 0.2), (0.66, 0.6), (0.33, 1.0)]:
                    scaled_hw = half_widths * frac
                    _, _, _, _, poly_x, poly_y = build_ribbon_polygon(
                        points_arr, normals_arr, scaled_hw
                    )
                    fill_alpha = self.config.ribbon_opacity * opacity_mult
                    verts = list(zip(poly_x, poly_y))
                    patch = MplPolygon(verts, closed=True,
                                       facecolor=(rc, gc, bc, fill_alpha),
                                       edgecolor='none', zorder=0)
                    ax.add_patch(patch)

    # Layer 1: Assigned Transitioning cells - Draw BEFORE clouds
    # =================================================================
    # Without separate atypical clusters, render all transitioning cells.
    if self.config.atypical_clusters:
        assigned_transition_mask = is_transitioning & ~is_atypical
    else:
        assigned_transition_mask = is_transitioning

    transition_positions = cell_positions[assigned_transition_mask]
    transition_colors = [cell_colors[i] for i in range(len(cell_colors)) if assigned_transition_mask[i]]

    if len(transition_positions) > 0:
        ax.scatter(
            transition_positions[:, 0], transition_positions[:, 1],
            c=transition_colors,
            s=self.config.cell_size * 7,
            alpha=0.8,
            linewidths=0.3,
            edgecolors='white',
            zorder=1,
            rasterized=True,
            label='Transitioning'
        )

    # Layer 1a: Unclustered atypical cells at their original tree positions.
    # =================================================================
    if self.config.atypical_clusters and noise_mask is not None and np.any(noise_mask):
        noise_positions = cell_positions[noise_mask]
        ax.scatter(noise_positions[:, 0], noise_positions[:, 1],
                  s=1, c='grey', alpha=0.3, zorder=1, rasterized=True)

    # Layer 1b: Atypical cells as Pie Charts (showing actual cell type composition)
    # =================================================================
    # Pie charts at each atypical-cluster hub show its cell-type composition.
    # Radii are computed relative to the data diagonal for consistent sizing.

    if atypical_cluster_infos and self.config.show_pie_charts and self.config.enable_atypical_detection:
        from matplotlib.patches import Polygon
        import matplotlib.patheffects as PathEffects

        # Pixel-space pie radii — isotropic circles in display pixels become
        # the correct data-space ellipses via `geom.pie_wedge_polygon`,
        # regardless of horizontal layout anisotropy. PDF uses
        # PIE_RADIUS_PDF_SCALE to match on-page visual scale.
        max_count = max(c.cell_count for c in atypical_cluster_infos)
        for cluster in atypical_cluster_infos:
            cluster.radius_px = (
                compute_pie_radius_px(cluster.cell_count, max_count) * PIE_RADIUS_PDF_SCALE
            )
            cluster.radius = geom.px_to_data_y(cluster.radius_px)

        # Resolve overlaps in pixel space so collisions are isotropic
        try:
            PieChartOverlapResolver.resolve_for_renderer(
                clusters=atypical_cluster_infos,
                node_positions=node_positions,
                is_polar=False,
                pixel_geometry=geom,
            )
        except Exception as _pie_exc:
            logger.warning("Pie-chart overlap resolution failed: %s", _pie_exc)

        # Draw pie charts at resolved positions
        for cluster in atypical_cluster_infos:
            hub_x, hub_y = cluster.position

            # Get cell type counts for pie chart slices
            if cluster.cell_type_counts:
                sorted_types = sorted(cluster.cell_type_counts.items(),
                                     key=lambda x: x[1], reverse=True)
                cell_types = [ct[0] for ct in sorted_types]
                counts = [ct[1] for ct in sorted_types]
            else:
                cell_types = cluster.anchor_nodes
                counts = [1] * len(cell_types)

            total = sum(counts)
            if total == 0:
                continue

            # Compute angles for pie slices
            proportions = [c / total for c in counts]
            angles = np.array([0.0] + list(np.cumsum(proportions))) * 360

            # Draw each pie slice as a data-space ellipse matching a
            # pixel-space circle of `cluster.radius_px`.
            for i, cell_type in enumerate(cell_types):
                theta1_deg, theta2_deg = angles[i], angles[i + 1]
                if theta2_deg - theta1_deg < 1:
                    continue

                slice_label = formatted_node_names[cell_type] if cell_type < len(formatted_node_names) else f"Type {cell_type}"
                if highlight_color_mapper is not None:
                    try:
                        slice_color = highlight_color_mapper.get_color(slice_label)
                    except KeyError:
                        slice_color = color_mapper.get_node_color(cell_type)
                else:
                    slice_color = color_mapper.get_node_color(cell_type)

                n_points = max(20, int((theta2_deg - theta1_deg) / 3))
                wedge_xs, wedge_ys = geom.pie_wedge_polygon(
                    hub_x, hub_y, cluster.radius_px,
                    np.radians(theta1_deg), np.radians(theta2_deg),
                    n=n_points,
                )
                vertices = list(zip(wedge_xs.tolist(), wedge_ys.tolist()))

                wedge_poly = Polygon(
                    vertices,
                    closed=True,
                    facecolor=slice_color,
                    edgecolor=self.config.pie_chart_edge_color,
                    linewidth=self.config.pie_chart_edge_width,
                    zorder=5
                )
                ax.add_patch(wedge_poly)

            # Label sits above the pie — y-axis gap is the ellipse's y
            # half-extent plus a small fixed pixel gap.
            _label_gap_data_y = geom.px_to_data_y(2.0)
            ax.annotate(
                f"Atypical_{cluster.cluster_id}\n(n={cluster.cell_count})",
                (hub_x, hub_y + cluster.radius + _label_gap_data_y),
                fontsize=self.config.label_font_size,
                color='#000080',
                fontweight='bold',
                ha='center',
                va='bottom',
                zorder=35
            )

    else:
        # Scatter fallback when pie charts are disabled.
        # Skip if highlight_data is present — the highlight overlay handles coloring
        # Skip if atypical_scatter=False — user explicitly wants atypical cells merged
        if (highlight_data is None
                and self.config.atypical_scatter
                and self.config.enable_atypical_detection):
            atypical_positions = cell_positions[is_atypical]

            if len(atypical_positions) > 0:
                ax.scatter(
                    atypical_positions[:, 0], atypical_positions[:, 1],
                    c='red',  # DISTINCT RED
                    s=self.config.cell_size * 1.5,  # Slightly larger
                    alpha=0.7,
                    linewidths=0.5,
                    edgecolors='darkred',
                    zorder=25,  # ON TOP of clouds, nodes, edges
                    rasterized=True,
                    label=f'Atypical ({len(atypical_positions)})'
                )

        # Layer 1d: Atypical-cluster hub markers (only when not using pie charts)
        if atypical_cluster_infos:
            import matplotlib.patheffects as PathEffects
            for cluster in atypical_cluster_infos:
                hub_x, hub_y = cluster.position

                # Draw dashed circle marker
                circle = plt.Circle(
                    (hub_x, hub_y),
                    radius=clusterer_config.atypical_hub_size * 0.1,
                    fill=False,
                    color='purple',
                    linestyle='--',
                    linewidth=2,
                    zorder=28
                )
                ax.add_patch(circle)

                # Add label
                ax.annotate(
                    f"Atypical_{cluster.cluster_id}\n(n={cluster.cell_count})",
                    (hub_x, hub_y),
                    xytext=(0, clusterer_config.atypical_hub_size ),
                    textcoords='offset points',
                    fontsize=self.config.label_font_size,
                    color='#000080',
                    fontweight='bold',
                    ha='center',
                    va='bottom',
                    zorder=30
                )

    # Layer 1c: Atypical-cluster hub anchor lines (dashed lines to anchor nodes)
    # =================================================================
    # NOTE: Drawn with high zorder so they appear ON TOP of clouds
    if atypical_cluster_infos and clusterer_config.show_atypical_anchor_lines:
        for cluster in atypical_cluster_infos:
            hub_x, hub_y = cluster.position
            for anchor_idx, weight in zip(cluster.anchor_nodes, cluster.anchor_weights):
                if anchor_idx in node_positions:
                    anchor_x, anchor_y = node_positions[anchor_idx]
                    # Opacity proportional to weight
                    line_alpha = max(0.2, min(0.6, weight))

                    ax.plot(
                        [hub_x, anchor_x], [hub_y, anchor_y],
                        color='purple',
                        linestyle='--',
                        linewidth=1.0,
                        alpha=line_alpha,
                        zorder=21  # ON TOP of clouds
                    )

    # Layer 2: Stable Density Contours - Draw AFTER transitioning cells so edges are visible
    # =================================================================
    # Only stable cells that are NOT atypical
    stable_mask = ~is_transitioning & ~is_atypical
    stable_positions = cell_positions[stable_mask]
    stable_anchors = anchor_nodes[stable_mask]

    if len(stable_positions) > 0:
        from scipy.stats import gaussian_kde
        from matplotlib.colors import LinearSegmentedColormap, to_rgba

        unique_stable_nodes = np.unique(stable_anchors)

        for node_idx in unique_stable_nodes:
            node_mask = stable_anchors == node_idx
            node_stable_positions = stable_positions[node_mask]

            # Use scatter whenever contours are disabled or the node falls below the contour threshold.
            use_contour = (
                self.config.display_contours
                and len(node_stable_positions) >= self.config.contour_min_cells
            )
            if not use_contour:
                node_color = color_mapper.get_node_color(node_idx)
                scatter_positions = _get_node_stable_scatter_positions(
                    self.config,
                    node_positions,
                    node_idx,
                    node_stable_positions,
                )
                ax.scatter(
                    scatter_positions[:, 0], scatter_positions[:, 1],
                    color=node_color, s=self.config.cell_size * 3, alpha=0.7, 
                    edgecolors='none', zorder=2
                )
                continue

            try:
                x = node_stable_positions[:, 0]
                y = node_stable_positions[:, 1]

                xmin, xmax, ymin, ymax = _compute_cartesian_contour_bounds(
                    node_idx=node_idx,
                    node_center=node_positions[node_idx],
                    stable_positions=node_stable_positions,
                    cloud_extents=cloud_extents,
                )

                xx, yy = np.meshgrid(
                    np.linspace(xmin, xmax, 80),
                    np.linspace(ymin, ymax, 80)
                )

                positions = np.vstack([xx.ravel(), yy.ravel()])
                xy_train = np.vstack([x, y])
                # Use config bandwidth if specified, otherwise auto (Scott's rule)
                bw_method = self.config.contour_bandwidth if self.config.contour_bandwidth is not None else 'scott'

                kernel = gaussian_kde(xy_train, bw_method=bw_method)
                z = kernel(positions).reshape(xx.shape)

                # Subtract grid-edge baseline before normalizing so the
                # rectangular grid perimeter has z=0 everywhere — kills the
                # faint square outline of the KDE bounding box.
                z_edge = float(max(
                    z[0, :].max(), z[-1, :].max(),
                    z[:, 0].max(), z[:, -1].max(),
                ))
                z_max = float(z.max())
                if z_max <= 0 or z_max <= z_edge:
                    continue
                z = np.clip(z - z_edge, 0.0, None) / (z_max - z_edge)

                # Soft gradient colormap: controls opacity gradient
                # First value: edge opacity (0.0 = transparent edge)
                # Second value: center opacity (higher = more solid center)
                node_color = color_mapper.get_node_color(node_idx)
                rgba = to_rgba(node_color)

                cmap_colors = [
                    (rgba[0], rgba[1], rgba[2], 0.0),   # Edge: fully transparent
                    (rgba[0], rgba[1], rgba[2], 0.95)    # Center: semi-opaque (was 0.6)
                ]
                cmap = LinearSegmentedColormap.from_list(f"node_{node_idx}", cmap_colors)

                # Render soft contour
                # levels: lower start = more gradual fade at edges
                cs = ax.contourf(
                    xx, yy, z, 
                    levels=np.linspace(0.05, 1.0, 30),  # Start at 0.1 (was 0.2), more levels
                    cmap=cmap, 
                    zorder=2
                )
                # Note: Rasterization not applied to contours (not supported in matplotlib 3.8+)

            except Exception:
                pass

    # =================================================================
    # Layer 2.5: Soft Edges - Draw AFTER cells/clouds but BEFORE nodes
    # =================================================================
    # Inferred paths are rendered here (not with canonical edges in Layer 0) to ensure they
    # appear above cell clouds and transitioning cells but below node labels.
    # zorder ~5 places them above cells (zorder 1-2) but below nodes (zorder 10).
    # Design: Thin ribbon polygons with directional arrow glyphs on top.
    if inferred_paths_data:
        # --- Soft Ribbon Rendering (below arrows) ---
        # Soft ribbons use 40-50% of canonical width, half opacity, no saturation boost
        canonical_start_hw = ribbon_base_hw * (self.config.edge_width / 1.5) * _pdf_ribbon_scale
        soft_ribbon_hw = 0.45 * canonical_start_hw  # 45% of canonical width
        soft_fill_opacity = self.config.ribbon_opacity * 0.5

        for edge_x, edge_y, weight, arc_pts, arc_nrm in inferred_paths_data:
            style = get_edge_style('soft', weight)
            soft_color = style.get('color', style.get('line_color', '#AAAAAA'))

            # Parse hex color to RGB tuple (no saturation boost for soft ribbons)
            if soft_color.startswith('#') and len(soft_color) >= 7:
                sr = int(soft_color[1:3], 16) / 255
                sg = int(soft_color[3:5], 16) / 255
                sb = int(soft_color[5:7], 16) / 255
            else:
                sr, sg, sb = 170 / 255, 170 / 255, 170 / 255  # fallback gray

            # Constant half-width (no taper) for soft ribbons
            n_pts = len(arc_pts)
            soft_hws = np.full(n_pts, soft_ribbon_hw)

            # Build ribbon polygon from arc points and normals
            _, _, _, _, poly_x, poly_y = build_ribbon_polygon(
                arc_pts, arc_nrm, soft_hws
            )

            # Render as MplPolygon — above canonical ribbons (zorder=0) but below cells
            verts = list(zip(poly_x, poly_y))
            patch = MplPolygon(verts, closed=True,
                               facecolor=(sr, sg, sb, soft_fill_opacity),
                               edgecolor='none', zorder=3)
            ax.add_patch(patch)

        # Arrow glyphs removed — inferred paths are distinguished by ribbon opacity alone

    # --- Highlight Scatter Overlay ---
    # Cloud-jitter positions already scale with per-node cell count.
    if highlight_data is not None:
        colors = [highlight_data.color_mapper.get_color(c) for c in highlight_data.categories]
        ax.scatter(
            highlight_data.positions[:, 0],
            highlight_data.positions[:, 1],
            c=colors,
            s=self.config.cell_size * 4 * HIGHLIGHT_MARKER_PDF_SCALE,
            alpha=0.8,
            zorder=10,
            rasterized=True,
        )
        # Add legend entries
        for cat, color in highlight_data.color_mapper.get_legend_items():
            ax.scatter([], [], c=color, s=20, label=cat)
        ax.legend(loc='upper left', fontsize=6, framealpha=0.8, ncol=2)

    # =================================================================
    # Layer 3: Nodes (Simple diamonds)
    # =================================================================
    all_node_x = [node_positions[idx][0] for idx in active_indices]
    all_node_y = [node_positions[idx][1] for idx in active_indices]
    all_node_colors = [color_mapper.get_node_color(idx) for idx in active_indices]

    ax.scatter(
        all_node_x, all_node_y,
        c=all_node_colors,
        s=self.config.node_size * 8,
        marker='o',
        edgecolors='white',
        linewidths=1.5,
        zorder=10
    )

    # --- Node Highlight Overlay (colored rings) ---
    if highlight_color_mapper is not None:
        for idx in active_indices:
            node_name = formatted_node_names[idx]
            try:
                ring_color = highlight_color_mapper.get_color(node_name)
            except KeyError:
                continue
            nx_pos, ny_pos = node_positions[idx]
            ax.scatter(
                [nx_pos], [ny_pos],
                s=self.config.node_size * 32,
                facecolors='none',
                edgecolors=ring_color,
                linewidths=2,
                zorder=11,
            )

    # Layer 4: Labels (Professional Typography)
    # =================================================================
    import matplotlib.patheffects as PathEffects

    for idx in active_indices:
        x, y = node_positions[idx]
        # Use the full formatted label (respects label_format preference)
        label = formatted_node_names[idx]

        ax.annotate(
            label,
            xy=(x, y),
            xytext=(0, 0.4 * self.config.label_font_size),  # Proportional to font size, smaller offset
            textcoords='offset points',
            fontsize=self.config.label_font_size*1.0, # horizontal layout PDF
            fontweight='bold',
            ha='center',
            va='bottom',
            zorder=20
        )

    # Layer 4b: Count Labels (below node, symmetric to name labels above)
    # =================================================================
    if self.config.show_cloud_cell_counts and cloud_label_counts:
        for idx in active_indices:
            count = cloud_label_counts.get(idx)
            if count is None:
                continue
            x, y = node_positions[idx]
            trans = cloud_transition_counts.get(idx) if cloud_transition_counts else None
            ax.annotate(
                format_cell_count_label(count, transition_count=trans),
                xy=(x, y),
                xytext=(0, -0.4 * self.config.label_font_size),  # Proportional to font size, smaller offset
                textcoords='offset points',
                fontsize=self.config.label_font_size * 1.0,
                ha='center',
                va='top',
                zorder=20
            )

    # Styling
    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    # Apply the axis limits that were computed at the top of this function.
    # Hoisting the computation lets pixel-space glyph sizing use the same
    # final viewport via `geom`; without this, matplotlib auto-expands to
    # include any exterior pie charts or anchor lines, compressing the tree
    # into a small portion of the canvas.
    ax.set_xlim(_pdf_x_range[0], _pdf_x_range[1])
    ax.set_ylim(_pdf_y_range[0], _pdf_y_range[1])

    # Build title with optional unclustered-atypical count
    mpl_title = self.config.title
    if self.config.atypical_clusters and noise_mask is not None and np.any(noise_mask):
        noise_count = int(np.sum(noise_mask))
        mpl_title = f"{mpl_title}\nUnclustered atypical: {noise_count} cells"

    ax.set_title(mpl_title, fontsize=10, fontweight='bold', pad=20)

    # Bottom horizontal legend — collect entries from node scatter and highlight traces.
    # ncol is set to a wide value so entries flow in rows; matplotlib wraps automatically.
    legend_handles = []
    legend_labels = []

    # Node color legend: one entry per lineage branch (from color mapper).
    # Uses get_legend_items() to ensure the legend shows the true branch
    # progenitor name, not a randomly-encountered downstream leaf.
    for label, color in color_mapper.get_legend_items(formatted_names=formatted_node_names):
        handle = plt.Line2D(
            [0], [0],
            marker='o', color='w',
            markerfacecolor=color,
            markersize=6,
            label=label
        )
        legend_handles.append(handle)
        legend_labels.append(label)

    # Atypical scatter entry (when atypical_scatter=True, no highlight, no pie).
    # The scatter's label= is ignored by the explicit handles= below, so we
    # must add it manually here.
    if (highlight_data is None
            and self.config.atypical_scatter
            and self.config.enable_atypical_detection
            and np.any(is_atypical)):
        atypical_count = int(np.sum(is_atypical))
        handle = plt.Line2D(
            [0], [0],
            marker='o', color='w',
            markerfacecolor='red',
            markersize=6,
            label=f'Atypical ({atypical_count})'
        )
        legend_handles.append(handle)
        legend_labels.append(f'Atypical ({atypical_count})')

    # Highlight entries (if present)
    if highlight_data is not None:
        for cat, color in highlight_data.color_mapper.get_legend_items():
            handle = plt.Line2D(
                [0], [0],
                marker='o', color='w',
                markerfacecolor=color,
                markersize=6,
                label=cat
            )
            legend_handles.append(handle)
            legend_labels.append(cat)

    if legend_handles:
        # Pick ncol so the legend row width stays within pdf_figsize[0].
        # With a fixed ncol (e.g. 16) and long cell-type names, the legend
        # can overflow the axes width; bbox_inches='tight' then expands the
        # saved PDF horizontally, and viewers' fit-to-page shrinks the whole
        # figure, visually compressing the tree.
        legend_fontsize = self.config.label_font_size * 0.7
        fig_w_inch = float(self.config.pdf_figsize[0])
        # Rough column width estimate: ~0.5 * fontsize_pt per character at
        # 1pt = 1/72 inch, plus ~0.35 inch for handle + padding per column.
        max_label_chars = max((len(lbl) for lbl in legend_labels), default=10)
        char_w_inch = legend_fontsize / 72.0 * 0.5
        per_col_inch = max_label_chars * char_w_inch + 0.35
        max_cols = max(1, int(fig_w_inch / per_col_inch))
        max_cols = min(max_cols, len(legend_handles))
        ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=max_cols,
            fontsize=legend_fontsize,
            framealpha=0.85,
            edgecolor='#cccccc',
            fancybox=False,
            handlelength=1.0,
            handletextpad=0.4,
            columnspacing=0.6,
        )

    # No set_aspect('equal') — pies, clouds, and ribbons are drawn via
    # `PixelGeometry` so they render correctly at the natural anisotropic
    # data→pixel mapping. Aspect locking here would collapse the axes box
    # on tall trees.

    # Tight layout with extra bottom margin for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Save PDF
    fig.savefig(
        output_path,
        format='pdf',
        dpi=self.config.pdf_dpi,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none'
    )
    plt.close(fig)

    print(f"✅ Saved: {output_path}")

    return None  # No Plotly figure to return for PDF


def _build_matplotlib_radial_figure(
        self,
        cell_positions: np.ndarray,
        predictions: np.ndarray,
        is_transitioning: np.ndarray,
        is_atypical: np.ndarray,
        anchor_nodes: np.ndarray,
        node_positions: Dict[int, Tuple[float, float]],
        active_indices: List[int],
        node_names: List[str],
        formatted_node_names: List[str],
        visual_tree_edges: List[Tuple[int, int]],
        canonical_tree_edges: List[Tuple[int, int]],  # Canonical edges for styling
        layout_engine: 'RadialTreeLayoutEngine',
        color_mapper,  # AdaptiveLineageColorMapper or TreeColorMapper
        cell_colors: List[str],
        subgraph: 'AdaptiveOntologySubgraph',
        node_cell_counts: Dict[int, int],
        max_count: int,
        output_path: str,
        atypical_cluster_infos: Optional[List['AtypicalClusterInfo']] = None,
        inferred_path_weights: Optional[Dict[Tuple[int, int], float]] = None,  # Inferred path weights
        cloud_label_counts: Optional[Dict[int, int]] = None,  # Per-node cell counts for cloud labels
        cloud_transition_counts: Optional[Dict[int, int]] = None,  # Per-node transitioning cell counts
        noise_mask: Optional[np.ndarray] = None,  # Mask for atypical cells left unclustered
        cloud_extents: Optional[Dict[int, float]] = None,  # Per-node cloud extent for contour/ribbon fidelity
        highlight_data: Optional['HighlightData'] = None,
        highlight_color_mapper: Optional['HighlightColorMapper'] = None
    ):
        """Build radial tree matplotlib figure for PDF export (TooManyCells style)."""
        from matplotlib.collections import LineCollection
        import matplotlib.colors as mcolors
        import matplotlib.patheffects as PathEffects
        from scipy.stats import gaussian_kde
        from matplotlib.colors import LinearSegmentedColormap, to_rgba

        atypical_cluster_infos = atypical_cluster_infos or []
        clusterer_config = AtypicalClustererConfig()
        inferred_path_weights = inferred_path_weights or {}
        cloud_extents = cloud_extents or {}

        # Build set of canonical edges for priority checking
        canonical_edge_set = set()
        for (i, j) in canonical_tree_edges:
            canonical_edge_set.add((i, j))
            canonical_edge_set.add((j, i))

        if not output_path.lower().endswith('.pdf'):
            print(f"  [WARNING] output_path '{output_path}' does not end with .pdf")

        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['font.size'] = 8

        # Increase figure size significantly to make room for labels
        fig = plt.figure(figsize=(20, 20)) 
        ax = fig.add_subplot(111, projection='polar')
        ax.set_axis_off()


        # =================================================================
        # Layer 0: Background Clade Wedges
        # =================================================================
        gap = RADIAL_GAP
        inner_r = RADIAL_CENTER_RADIUS
        outer_r = self.config.radial_max_radius
        # Plot range matches the Plotly radial path (outer_r * 1.15) so the
        # pixel geometry used for pie sizing/overlap resolution stays
        # consistent between the PDF and HTML radial renderers.
        plot_range = outer_r * 1.15

        for theta_start, theta_width, bg_color in layout_engine.get_clade_background_sectors(
                visible_nodes=set(active_indices)):
            start_angle = theta_start + (gap / 2)
            end_angle = (theta_start + theta_width) - (gap / 2)
            if end_angle > start_angle:
                theta_range = np.linspace(start_angle, end_angle, 100)
                ax.fill_between(theta_range, inner_r, outer_r,
                               color=bg_color, alpha=0.5, zorder=0, lw=0)

        # =================================================================
        # Layer 1: Canonical Edges (Bezier curves from layout engine)
        # =================================================================
        edges_data = layout_engine.get_radial_edges(node_positions)
        segments_coords = []
        segment_colors = []
        segment_widths = []

        _PDF_RADIAL_WIDTH_SCALE = 1.5  # PDF branches render thinner than HTML; compensate here

        for edge in edges_data:
            for seg in edge['segments']:
                start_theta, start_r = seg['start']
                end_theta, end_r = seg['end']
                segments_coords.append([(start_theta, start_r), (end_theta, end_r)])
                segment_colors.append(seg['color'])
                segment_widths.append(seg['width'] * _PDF_RADIAL_WIDTH_SCALE)

        if segments_coords:
            lc = LineCollection(segments_coords, colors=segment_colors, 
                               linewidths=segment_widths, capstyle='round')
            ax.add_collection(lc)

        # =================================================================
        # Layer 2: Cells (Scatter & Contours)
        # =================================================================
        cell_thetas = cell_positions[:, 0]
        cell_rs = cell_positions[:, 1]

        # A. Transitioning cells
        # Without separate atypical clusters, render all transitioning cells.
        if self.config.atypical_clusters:
            assigned_transition_mask = is_transitioning & ~is_atypical
        else:
            assigned_transition_mask = is_transitioning

        if np.any(assigned_transition_mask):
            trans_thetas = cell_thetas[assigned_transition_mask]
            trans_rs = cell_rs[assigned_transition_mask]
            trans_colors = [cell_colors[i] for i in range(len(cell_colors)) if assigned_transition_mask[i]]

            ax.scatter(trans_thetas, trans_rs, c=trans_colors, s=self.config.cell_size * 10,
                      alpha=0.7, linewidths=0.2, edgecolors='white', zorder=2, rasterized=True)

        # A. Unclustered atypical cells at their original tree positions.
        if self.config.atypical_clusters and noise_mask is not None and np.any(noise_mask):
            noise_positions = cell_positions[noise_mask]
            ax.scatter(noise_positions[:, 0], noise_positions[:, 1],
                      s=1, c='grey', alpha=0.3, zorder=1, rasterized=True)

        # B. Stable Cells (KDE CONTOURS)
        stable_mask = ~is_transitioning & ~is_atypical
        # Radial PDF only: expand each cell's offset from its anchor by
        # this factor so the cell cloud's spatial footprint renders ~5x
        # larger than the horizontal layout. Horizontal PDF uses its own
        # render function and is untouched.
        #
        # NOTE: expansion happens in *Cartesian* space (around the node's
        # x,y) and then converts back to (theta, r). Scaling directly in
        # polar space is wrong for nodes whose center_theta is near 0 or
        # 2π: a small negative wrapped theta delta (e.g. -0.02) multiplied
        # by 5 lands at theta ≈ -0.1, which `_contract_stable_scatter_positions`
        # then mods into theta ≈ 2π - 0.1. matplotlib's polar contourf
        # trains gaussian_kde on raw (theta, r) values, so the two halves
        # of the cloud end up 6.2 radians apart in the KDE's Euclidean
        # metric — producing a smear on the opposite side of the canvas.
        radial_pdf_cloud_position_scale = 5.0
        if np.any(stable_mask):
            stable_positions = cell_positions[stable_mask]
            stable_anchors = anchor_nodes[stable_mask]
            unique_stable_nodes = np.unique(stable_anchors)

            for node_idx in unique_stable_nodes:
                if node_idx not in active_indices: continue

                node_mask = stable_anchors == node_idx
                node_stable_positions = stable_positions[node_mask]

                # Cartesian-space expansion: scale Δ(x,y) from the node's
                # Cartesian center, then convert back to (theta, r). No 2π
                # wrap, no ambiguity at theta=0.
                if len(node_stable_positions):
                    _c_theta, _c_r = node_positions[node_idx]
                    _cx = _c_r * np.cos(_c_theta)
                    _cy = _c_r * np.sin(_c_theta)
                    _x = node_stable_positions[:, 1] * np.cos(node_stable_positions[:, 0])
                    _y = node_stable_positions[:, 1] * np.sin(node_stable_positions[:, 0])
                    _x_sc = _cx + (_x - _cx) * radial_pdf_cloud_position_scale
                    _y_sc = _cy + (_y - _cy) * radial_pdf_cloud_position_scale
                    _r_sc = np.hypot(_x_sc, _y_sc)
                    _theta_sc = np.arctan2(_y_sc, _x_sc)
                    node_stable_positions = np.column_stack([_theta_sc, _r_sc])

                use_contour = (
                    self.config.display_contours
                    and len(node_stable_positions) >= self.config.contour_min_cells
                )
                if not use_contour:
                    node_color = color_mapper.get_node_color(node_idx)
                    scatter_positions = _get_node_stable_scatter_positions(
                        self.config,
                        node_positions,
                        node_idx,
                        node_stable_positions,
                        is_polar=True,
                    )
                    ax.scatter(
                        scatter_positions[:, 0], scatter_positions[:, 1],
                        color=node_color, s=self.config.cell_size * 2, alpha=0.6,
                        edgecolors='none', zorder=2, rasterized=True
                    )
                    continue

                try:
                    theta = node_stable_positions[:, 0]
                    r = node_stable_positions[:, 1]

                    theta_min, theta_max, r_min, r_max = _compute_polar_contour_bounds(
                        node_idx=node_idx,
                        node_center=node_positions[node_idx],
                        stable_positions=node_stable_positions,
                        cloud_extents=cloud_extents,
                    )

                    theta_grid, r_grid = np.meshgrid(
                        np.linspace(theta_min, theta_max, 60),
                        np.linspace(r_min, r_max, 60)
                    )

                    positions = np.vstack([theta_grid.ravel(), r_grid.ravel()])
                    xy_train = np.vstack([theta, r])
                    # Use config bandwidth if specified, otherwise auto (Scott's rule)
                    bw_method = self.config.contour_bandwidth if self.config.contour_bandwidth is not None else 'scott'

                    kernel = gaussian_kde(xy_train, bw_method=bw_method)
                    z = kernel(positions).reshape(theta_grid.shape)

                    # Subtract grid-edge baseline before normalizing so the
                    # rectangular (theta, r) grid perimeter has z=0 — eliminates
                    # the square outline visible when the KDE tail extends past
                    # the cloud's tight bounding box.
                    z_edge = float(max(
                        z[0, :].max(), z[-1, :].max(),
                        z[:, 0].max(), z[:, -1].max(),
                    ))
                    z_max = float(z.max())
                    if z_max <= 0 or z_max <= z_edge:
                        continue
                    z = np.clip(z - z_edge, 0.0, None) / (z_max - z_edge)

                    # Soft Gradient Colormap: Solid core, fading only at edges
                    # Stops: 0.0 (edge) → 0.3 (fade zone) → 1.0 (solid center)
                    node_color = color_mapper.get_node_color(node_idx)
                    rgba = to_rgba(node_color)
                    cmap_colors = [
                        (0.1, (rgba[0], rgba[1], rgba[2], 0.0)),   # Edge: fully transparent
                        (0.5, (rgba[0], rgba[1], rgba[2], 0.7)),   # Fade zone: semi-opaque
                        (1.0, (rgba[0], rgba[1], rgba[2], 1.0))    # Center: fully opaque
                    ]
                    cmap = LinearSegmentedColormap.from_list(
                        f"node_{node_idx}",
                        [c[1] for c in cmap_colors],
                        N=256
                    )

                    # Render soft contour for above-threshold stable clouds only.
                    ax.contourf(theta_grid, r_grid, z, levels=np.linspace(0.05, 1.0, 25), cmap=cmap, zorder=2)

                except Exception:
                    node_color = color_mapper.get_node_color(node_idx)
                    scatter_positions = _get_node_stable_scatter_positions(
                        self.config,
                        node_positions,
                        node_idx,
                        node_stable_positions,
                        is_polar=True,
                    )
                    ax.scatter(
                        scatter_positions[:, 0],
                        scatter_positions[:, 1],
                        c=node_color,
                        s=self.config.cell_size * 2,
                        alpha=0.5,
                        zorder=2,
                        rasterized=True,
                    )

        # --- Highlight Scatter Overlay ---
        if highlight_data is not None:
            colors = [highlight_data.color_mapper.get_color(c) for c in highlight_data.categories]
            ax.scatter(
                highlight_data.positions[:, 0],
                highlight_data.positions[:, 1],
                c=colors,
                s=self.config.cell_size * 4 * HIGHLIGHT_MARKER_PDF_SCALE,
                alpha=0.8,
                zorder=10,
                rasterized=True,
            )

        # --- Legend: lineage branch colors (always) + highlights (when present) ---
        # Add lineage branch legend entries from color mapper
        for label, color in color_mapper.get_legend_items(formatted_names=formatted_node_names):
            ax.scatter([], [], c=color, s=20, marker='o', label=label)

        # Add highlight legend entries (if present)
        if highlight_data is not None:
            for cat, color in highlight_data.color_mapper.get_legend_items():
                ax.scatter([], [], c=color, s=20, label=cat)

        # Render legend if there are any entries
        legend_artists = ax.get_legend_handles_labels()
        if legend_artists[0]:
            legend_x_fig, legend_y_fig, legend_anchor_mode = _compute_radial_legend_anchor(
                layout_engine,
                outer_r
            )

            total_entries = len(legend_artists[0])
            legend_labels_text = legend_artists[1]
            # Pick ncol so the legend row width fits inside the figure. A
            # fixed cap (e.g. 7) lets long anchor-cell-type names overflow
            # the figure right edge; bbox_inches='tight' then expands the
            # saved PDF width and the viewer's fit-to-page compresses the
            # tree. Estimate per-column inch width from label length and
            # size ncol to the available horizontal space.
            legend_fontsize = 6
            fig_w_inch = float(fig.get_size_inches()[0])
            max_label_chars = max((len(lbl) for lbl in legend_labels_text), default=10)
            char_w_inch = legend_fontsize / 72.0 * 0.5
            per_col_inch = max_label_chars * char_w_inch + 0.35
            if legend_anchor_mode == 'upper_left':
                # Legend expands rightward from legend_x_fig → available
                # width is the distance from the anchor to the right edge,
                # minus a small safety margin.
                available_inch = max(1.0, (1.0 - legend_x_fig) * fig_w_inch - 0.2)
            else:
                # lower_center expands symmetrically; allow ~90% of width.
                available_inch = max(1.0, 0.9 * fig_w_inch)
            max_cols = max(1, int(available_inch / per_col_inch))
            max_cols = min(max_cols, total_entries)

            if legend_anchor_mode == 'upper_left':
                ax.legend(loc='upper left', fontsize=legend_fontsize, framealpha=0.95,
                          ncol=max_cols,
                          bbox_to_anchor=(legend_x_fig, legend_y_fig),
                          bbox_transform=fig.transFigure,
                          title='Legend', title_fontsize=7,
                          edgecolor='black', fancybox=False)
            else:
                ax.legend(loc='lower center', fontsize=legend_fontsize, framealpha=0.8,
                          ncol=max_cols,
                          bbox_to_anchor=(legend_x_fig, legend_y_fig))

        # =================================================================
        # Layer 3: Nodes
        # =================================================================
        for idx in active_indices:
            theta, r = node_positions[idx]
            node_color = color_mapper.get_node_color(idx)
            # Root nodes (r < 0.2) use smaller size (reduced by 2 sizes)
            node_marker_size = self.config.node_size * 10 if r < 0.2 else self.config.node_size * 12
            ax.scatter(theta, r, c=node_color, s=node_marker_size,
                      marker='o', edgecolors='white', linewidths=1.0, zorder=10)

        # --- Node Highlight Overlay (colored rings) ---
        if highlight_color_mapper is not None:
            for idx in active_indices:
                node_name = formatted_node_names[idx]
                try:
                    ring_color = highlight_color_mapper.get_color(node_name)
                except KeyError:
                    continue
                n_theta, n_r = node_positions[idx]
                ax.scatter(
                    [n_theta], [n_r],
                    s=self.config.node_size * 8,
                    facecolors='none',
                    edgecolors=ring_color,
                    linewidths=2,
                    zorder=11,
                )

        # =================================================================
        # Layer 4: Inset Pie Charts (BEFORE labels so labels render on top)
        # =================================================================
        # Pie charts show the composition of atypical cell clusters.
        # Radii are computed relative to the data diagonal for consistent sizing.

        if atypical_cluster_infos and self.config.show_pie_charts and self.config.enable_atypical_detection:
            from matplotlib.patches import Wedge
            from matplotlib.offsetbox import DrawingArea, AnnotationBbox

            # Pixel-space pie radii. PDF radial uses a polar axes with isotropic
            # aspect, so the `geom` we build below is only used to set
            # cluster.radius (for the overlap resolver's data-space fallback
            # path on polar layouts). DrawingArea sizes come directly from
            # radius_px × (dpi / 96) so pie diameters stay at the target
            # pixel size regardless of figure size.
            max_count = max(c.cell_count for c in atypical_cluster_infos)
            # Radial PDF figure is 20x20 inches; polar axes fill the figure.
            # Treat the full canvas as the axes box for pixel-geometry purposes.
            _radial_pdf_fig_inches = self.config.radial_pdf_figsize if hasattr(self.config, 'radial_pdf_figsize') else (20.0, 20.0)
            _radial_geom = pixel_geometry_from_matplotlib(
                figsize_inches=_radial_pdf_fig_inches,
                dpi=self.config.pdf_dpi,
                x_range=(-plot_range, plot_range),
                y_range=(-plot_range, plot_range),
                margins_inches=(0.0, 0.0, 0.0, 0.0),
            )
            # Radial PDF only: shrink pie radius to 50% of the shared PDF scale.
            radial_pie_pdf_radius_scale = 0.35
            for cluster in atypical_cluster_infos:
                cluster.radius_px = (
                    compute_pie_radius_px(cluster.cell_count, max_count)
                    * PIE_RADIUS_PDF_SCALE
                    * radial_pie_pdf_radius_scale
                )
                cluster.radius = _radial_geom.px_to_data_y(cluster.radius_px)

            # Resolve overlaps in pixel space (polar→Cartesian handled internally)
            PieChartOverlapResolver.resolve_for_renderer(
                clusters=atypical_cluster_infos,
                node_positions=node_positions,
                is_polar=True,
                pixel_geometry=_radial_geom,
            )

            # Draw pie charts
            for cluster in atypical_cluster_infos:
                # Position is in polar coordinates (theta, r) after overlap resolution
                hub_theta, hub_r = cluster.position

                # DrawingArea is sized directly in display pixels. Convert
                # `cluster.radius_px` (CSS-display pixels) to matplotlib
                # display pixels at the figure's DPI.
                pie_size_inches = (cluster.radius_px * 2) / 96.0  # CSS px → inches (96 CSS px per inch)

                if cluster.cell_type_counts:
                    sorted_types = sorted(cluster.cell_type_counts.items(), key=lambda x: x[1], reverse=True)
                    counts = [x[1] for x in sorted_types]
                    if highlight_color_mapper is not None:
                        colors = []
                        for x in sorted_types:
                            ct_label = formatted_node_names[x[0]] if x[0] < len(formatted_node_names) else f"Type {x[0]}"
                            try:
                                colors.append(highlight_color_mapper.get_color(ct_label))
                            except KeyError:
                                colors.append(color_mapper.get_node_color(x[0]))
                    else:
                        colors = [color_mapper.get_node_color(x[0]) for x in sorted_types]
                else:
                    counts = [1] * len(cluster.anchor_nodes)
                    if highlight_color_mapper is not None:
                        colors = []
                        for n in cluster.anchor_nodes:
                            ct_label = formatted_node_names[n] if n < len(formatted_node_names) else f"Type {n}"
                            try:
                                colors.append(highlight_color_mapper.get_color(ct_label))
                            except KeyError:
                                colors.append(color_mapper.get_node_color(n))
                    else:
                        colors = [color_mapper.get_node_color(n) for n in cluster.anchor_nodes]

                total = sum(counts)
                if total == 0:
                    continue

                # Build pie chart in a DrawingArea (pixel-based, resolution-independent)
                dpi = fig.get_dpi()
                da_size = int(pie_size_inches * dpi)
                da = DrawingArea(da_size, da_size, 0, 0)

                # Draw wedges centered in the DrawingArea
                cx, cy = da_size / 2, da_size / 2
                radius = da_size / 2 - 2

                proportions = [c / total for c in counts]
                angles = np.array([0.0] + list(np.cumsum(proportions))) * 360

                for i in range(len(counts)):
                    theta1_deg, theta2_deg = angles[i], angles[i + 1]
                    if theta2_deg - theta1_deg < 0.5:
                        continue
                    wedge = Wedge(
                        (cx, cy), radius, theta1_deg, theta2_deg,
                        facecolor=colors[i],
                        edgecolor='white',
                        linewidth=0.5
                    )
                    da.add_artist(wedge)

                # Anchor the DrawingArea to the polar data point
                ab = AnnotationBbox(
                    da, (hub_theta, hub_r),
                    xycoords='data',
                    frameon=False,
                    pad=0,
                    zorder=15
                )
                ax.add_artist(ab)

                # Add label above pie chart
                ax.annotate(
                    f"Amb_{cluster.cluster_id}\n(n={cluster.cell_count})",
                    xy=(hub_theta, hub_r),
                    xytext=(0, da_size / 2 + 8),
                    textcoords='offset points',
                    fontsize=6,
                    color='#000080',
                    ha='center',
                    va='bottom',
                    zorder=25
                )

        # =================================================================
        # Layer 1c: Atypical-cluster hub anchor lines (polar coordinates)
        # =================================================================
        # NOTE: Drawn AFTER pie charts so lines visually connect to pie chart centers.
        # High zorder ensures they appear ON TOP of clouds and pie charts.
        if atypical_cluster_infos and clusterer_config.show_atypical_anchor_lines:
            for cluster in atypical_cluster_infos:
                # Position is in polar coordinates (theta, r) after overlap resolution
                hub_theta, hub_r = cluster.position
                for anchor_idx, weight in zip(cluster.anchor_nodes, cluster.anchor_weights):
                    if anchor_idx in node_positions:
                        anchor_theta, anchor_r = node_positions[anchor_idx]
                        line_alpha = max(0.2, min(0.6, weight))
                        ax.plot(
                            [hub_theta, anchor_theta], [hub_r, anchor_r],
                            color='purple', linestyle='--', linewidth=1.0,
                            alpha=line_alpha, zorder=21
                        )
                
        # =================================================================
        # Layer 4.5: Soft Edges (Arrows indicating direction) - Above all layers but below labels
        # =================================================================
        # Inferred paths are rendered as glyph arcs (repeated '>' characters along circular arc).
        # High zorder ensures they appear above all other elements but below node labels.
        if inferred_path_weights:
            for (src, tgt), weight in inferred_path_weights.items():
                if src not in node_positions or tgt not in node_positions:
                    continue

                src_theta, src_r = node_positions[src]
                tgt_theta, tgt_r = node_positions[tgt]

                # Get edge style
                style = get_edge_style('soft', weight)

                # Convert polar to Cartesian for arc computation
                src_x = src_r * np.cos(src_theta)
                src_y = src_r * np.sin(src_theta)
                tgt_x = tgt_r * np.cos(tgt_theta)
                tgt_y = tgt_r * np.sin(tgt_theta)

                A = (src_x, src_y)
                B = (tgt_x, tgt_y)

                try:
                    # Generate circular arc points using unified geometry function
                    # Dynamic curvature: Proportional to distance (Constant Arc Height)
                    dist = np.hypot(src_x - tgt_x, src_y - tgt_y)
                    dynamic_curvature = max(0.05 * dist, 0.4) if dist > 0 else 0.4

                    arc_points, arc_normals = generate_circular_arc_points(
                        A, B, curvature=dynamic_curvature, num_points=50
                    )

                    # Arc points are in Cartesian (x, y)
                    curve_x = arc_points[:, 0]
                    curve_y = arc_points[:, 1]

                    # Convert arc points back to polar for ax.text() placement
                    curve_r = np.sqrt(curve_x**2 + curve_y**2)
                    curve_theta = np.arctan2(curve_y, curve_x)

                    # Calculate arc length for glyph spacing
                    d_x = np.diff(curve_x)
                    d_y = np.diff(curve_y)
                    segment_lengths = np.sqrt(d_x**2 + d_y**2)
                    arc_length = np.sum(segment_lengths)

                    # Place glyphs along arc with consistent spacing
                    glyph_spacing = 0.2  # Spacing in data units (matching HTML)
                    n_glyphs = max(2, int(arc_length / glyph_spacing))
                    indices = np.linspace(0, len(curve_theta)-1, n_glyphs).astype(int)

                    for idx in indices:
                        theta, r = curve_theta[idx], curve_r[idx]

                        # Calculate tangent angle from Cartesian arc geometry
                        # Tangent is perpendicular to normal
                        normal = arc_normals[idx]
                        # Tangent is 90° rotation of normal: if normal is (nx, ny), tangent is (ny, -nx)
                        tangent_x = normal[1]
                        tangent_y = -normal[0]

                        # Angle in Cartesian plane (screen coordinates relative to center)
                        tangent_angle_deg = np.degrees(np.arctan2(tangent_y, tangent_x))

                        # In Matplotlib polar plots, ax.text(theta, r, ...) places text at polar coords
                        # BUT rotation is specified in degrees relative to the horizontal screen axis
                        # So the Cartesian tangent angle is exactly what we need

                        ax.text(theta, r, '>',
                               fontsize=10,
                               fontweight='bold',  # Make it bold for thickness
                               color=style['line_color'],
                               alpha=style.get('opacity', 0.5),
                               ha='center', va='center',
                               rotation=tangent_angle_deg,
                               rotation_mode='anchor',
                               zorder=15)
                except (ValueError, Exception) as e:
                    # Fallback to straight line if arc computation fails
                    ax.plot([src_theta, tgt_theta], [src_r, tgt_r],
                           color=style['line_color'], linewidth=style['line_width'],
                           linestyle=':', alpha=style['opacity'], zorder=15)


        # =================================================================
        # Layer 5: Labels (AFTER pie charts so they appear on top)
        # =================================================================
        # Strategy: Sort labels by angle. If close, stack them radially.

        # 1. Collect all label candidates
        labels_to_plot = []
        for idx in active_indices:
            theta, r = node_positions[idx]
            # Use the full formatted label (respects label_format preference)
            label_text = formatted_node_names[idx]
            labels_to_plot.append({
                'theta': theta,
                'r': r,
                'text': label_text,
                'orig_r': r,
                'is_root': r < 0.2,
                'node_idx': idx  # Store node index for count label lookup
            })

        # 2. Sort by Theta for linear sweep collision detection
        labels_to_plot.sort(key=lambda x: x['theta'])

        # 3. Collision Loop (Simple 1-Pass Nudge)
        # We check the previous K neighbors. If they are close angularly, we push the current one out.
        angular_threshold = np.radians(3.0) # ~3 degrees collision zone (slightly wider)
        radius_step = 0.1 # Smaller step for compact labels

        for i in range(len(labels_to_plot)):
            current = labels_to_plot[i]
            if current['is_root']: continue 

            # Check previous few labels for overlaps
            # (In a circle, we should technically check wrap-around too, but linear sort is usually enough for local density)
            max_r_in_zone = current['r']
            collision_detected = False

            # Look back at neighbors
            for j in range(max(0, i-5), i):
                neighbor = labels_to_plot[j]

                # Check angular distance
                diff = abs(current['theta'] - neighbor['theta'])
                if diff > np.pi: diff = 2*np.pi - diff

                if diff < angular_threshold:
                    # Overlap detected! Track the max radius used in this cluster
                    if neighbor['r'] >= max_r_in_zone:
                        max_r_in_zone = neighbor['r']
                        collision_detected = True

            # If collision, place this label strictly OUTSIDE the cluster
            if collision_detected:
                current['r'] = max_r_in_zone + radius_step

        # 4. Render Final Labels using node-centered fishbone approach
        # Labels positioned along ±25° fishbone directions from each node

        # Helper function: compute dynamic label distance based on node depth
        # Inner nodes (small r) need larger spacing to prevent overlap
        def compute_label_offset(node_r: float) -> float:
            """
            Compute label distance from node.

            Inner nodes receive a larger inverse-radius offset than outer nodes.
            """
            base_offset = 0.1
            inverse_scale = 0.5 / (node_r + 2.0)  # Decreases as r increases
            return base_offset + inverse_scale

        # Fishbone tilt configuration
        fishbone_tilt_deg = 25  # degrees from radial

        for item in labels_to_plot:
            theta, label = item['theta'], item['text']
            node_r = item['orig_r']

            if item['is_root']:
                 # Position root label below the center node using annotation with offset
                 ax.annotate(label, xy=(0, 0), xytext=(0, -15), textcoords='offset points',
                            fontsize=self.config.label_font_size*1.5, fontweight='bold', 
                            ha='center', va='top', zorder=22)
            else:
                 deg_angle = np.degrees(theta) % 360

                 # Node-centered circular sector approach:
                 # Labels positioned along fishbone radii (±25° from node's radial)
                 # at equal distance 'd' from the node
                 label_distance = compute_label_offset(node_r)

                 # Convert node position to Cartesian
                 node_x = node_r * np.cos(theta)
                 node_y = node_r * np.sin(theta)

                 # Name label: along theta + 25° fishbone direction
                 fishbone_angle_name = theta + np.radians(fishbone_tilt_deg)
                 name_x = node_x + label_distance * np.cos(fishbone_angle_name)
                 name_y = node_y + label_distance * np.sin(fishbone_angle_name)

                 # Convert back to polar
                 name_r = np.sqrt(name_x**2 + name_y**2)
                 name_theta = np.arctan2(name_y, name_x)

                 # Label rotation and alignment
                 if 90 < deg_angle < 270:
                    rotation = deg_angle + 180 + fishbone_tilt_deg
                    ha_align = 'right'
                 else:
                    rotation = deg_angle + fishbone_tilt_deg
                    ha_align = 'left'

                 ax.text(name_theta, name_r, label,
                             fontsize=self.config.label_font_size,
                             fontweight='bold',
                             ha=ha_align, va='center',
                             rotation=rotation, rotation_mode='anchor',
                             zorder=20)

        # =================================================================
        # Layer 5b: Count Labels (opposite side of name labels — fishbone)
        # =================================================================
        # Fishbone structure: name label tilts +25° from radial, count label
        # tilts -25° (opposite direction). Angular nudge in the opposite
        # direction from the name label separates them at the converging point.
        if self.config.show_cloud_cell_counts and cloud_label_counts:
            for item in labels_to_plot:
                theta = item['theta']
                node_r = item['orig_r']
                node_idx = item['node_idx']

                if item['is_root']:
                    continue

                count = cloud_label_counts.get(node_idx)
                if count is None:
                    continue

                deg_angle = np.degrees(theta) % 360

                # Node-centered circular sector approach:
                # Count label positioned along theta - 25° fishbone direction
                label_distance = compute_label_offset(node_r)

                # Convert node position to Cartesian
                node_x = node_r * np.cos(theta)
                node_y = node_r * np.sin(theta)

                # Count label: along theta - 25° fishbone direction
                fishbone_angle_count = theta - np.radians(fishbone_tilt_deg)
                count_x = node_x + label_distance * np.cos(fishbone_angle_count)
                count_y = node_y + label_distance * np.sin(fishbone_angle_count)

                # Convert back to polar
                count_r = np.sqrt(count_x**2 + count_y**2)
                count_theta = np.arctan2(count_y, count_x)

                # Label rotation and alignment
                if 90 < deg_angle < 270:
                    rotation = deg_angle + 180 - fishbone_tilt_deg
                    ha_align = 'right'
                else:
                    rotation = deg_angle - fishbone_tilt_deg
                    ha_align = 'left'

                trans = cloud_transition_counts.get(node_idx) if cloud_transition_counts else None
                ax.text(count_theta, count_r, format_cell_count_label(count, transition_count=trans),
                        fontsize=self.config.label_font_size,
                        ha=ha_align, va='center',
                        rotation=rotation, rotation_mode='anchor',
                        zorder=20)

        # Center Pin & Save
        ax.scatter(0, 0, s=400, color='#212121', marker='o', zorder=20)
        ax.scatter(0, 0, s=100, color='#ffffff', marker='o', zorder=21)
        ax.set_ylim(0, outer_r * 1.2)

        # Build title with optional unclustered-atypical count
        radial_title = self.config.title
        if self.config.atypical_clusters and noise_mask is not None and np.any(noise_mask):
            noise_count = int(np.sum(noise_mask))
            radial_title = f"{radial_title}\nUnclustered atypical: {noise_count} cells"

        fig.suptitle(radial_title, fontsize=12, fontweight='bold', y=1.01)

        fig.savefig(output_path, format='pdf', dpi=self.config.pdf_dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved radial tree: {output_path}")
        return None


# ---------------------------------------------------------------------------
# Atypical-cell diagnostic plotters
# ---------------------------------------------------------------------------
def _draw_suspicious_community_bars(
    ax,
    community_stats: List[Dict[str, Any]],
) -> None:
    """Render the per-suspicious-community stacked aberrant/non-aberrant bars.

    Used by the routed atypical diagnostic grid. Reads ``community_stats`` produced by
    :func:`_label_communities_by_enrichment_fdr` (each entry must carry
    ``label``, ``size``, ``n_aberrant``, ``suspicious``).
    """
    susp_rows = [r for r in community_stats if r.get("suspicious")]
    if susp_rows:
        susp_rows = sorted(
            susp_rows,
            key=lambda r: int(r.get("n_aberrant", 0)),
            reverse=False,  # largest on top when plotted bottom-up
        )
        ids = [int(r["label"]) for r in susp_rows]
        n_aber = np.array(
            [int(r["n_aberrant"]) for r in susp_rows], dtype=np.int64
        )
        sizes = np.array(
            [int(r["size"]) for r in susp_rows], dtype=np.int64
        )
        n_non = sizes - n_aber
        ys = np.arange(len(susp_rows))
        ax.barh(
            ys, n_aber, color="#e74c3c", edgecolor="none",
            label="aberrant",
        )
        ax.barh(
            ys, n_non, left=n_aber, color="#bdc3c7", edgecolor="none",
            label="non-aberrant",
        )
        for y, na, sz in zip(ys, n_aber, sizes):
            pct = (float(na) / float(sz) * 100.0) if sz > 0 else 0.0
            ax.text(
                float(sz), float(y),
                f"  {int(na)}/{int(sz)} = {pct:.0f}%",
                va="center", ha="left", fontsize=7, color="#2c3e50",
            )
        ax.set_yticks(ys)
        ax.set_yticklabels([str(i) for i in ids], fontsize=8)
        ax.set_xlim(0, float(sizes.max()) * 1.25)
        # Cap bar thickness when there are only a few groups by padding the
        # y-axis to a minimum slot count; bars stay anchored to the bottom.
        min_slots = 6
        n_bars = len(susp_rows)
        ax.set_ylim(-0.6, max(n_bars - 0.4, min_slots - 0.4))
        ax.legend(
            loc="lower right", fontsize=8,
            framealpha=0.0, edgecolor="none",
        )
    else:
        ax.text(
            0.5, 0.5, "No suspicious communities",
            ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="#7f8c8d",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("Cells")
    ax.set_ylabel("Community id")
    ax.set_title("(c) Aberrant cell composition of suspicious communities")


def _draw_size_vs_fraction_panel(
    ax,
    community_stats: List[Dict[str, Any]],
    *,
    min_community_size: int,
) -> None:
    """Render the community size vs fraction_aberrant scatter (panel d).

    Used by the routed atypical diagnostic grid. Each ``community_stats`` entry
    must carry ``label``, ``size``, ``fraction_aberrant``,
    ``enrichment_floor_threshold``, and ``suspicious``.

    The horizontal "atypical zone" line is drawn at the gate's
    ``enrichment_floor_threshold`` (the lower of ``baseline + effect_floor``
    and ``1.5 * baseline`` — see :func:`_label_communities_by_enrichment_fdr`).
    """
    if community_stats:
        sizes = np.array(
            [int(r["size"]) for r in community_stats], dtype=np.int64
        )
        fracs = np.array(
            [float(r["fraction_aberrant"]) for r in community_stats],
            dtype=np.float64,
        )
        labels_arr = np.array(
            [int(r["label"]) for r in community_stats], dtype=np.int64
        )
        suspicious_flags = np.array(
            [bool(r.get("suspicious")) for r in community_stats], dtype=bool
        )
        # All rows share the same floor; pull from the first row.
        floor_threshold = float(
            community_stats[0].get(
                "enrichment_floor_threshold", float("nan")
            )
        )
        if np.any(~suspicious_flags):
            ax.scatter(
                sizes[~suspicious_flags], fracs[~suspicious_flags],
                s=20, c="#7f8c8d", alpha=0.8,
                label=f"non-suspicious (n={int((~suspicious_flags).sum())})",
                edgecolors="none", zorder=2,
            )
        if np.any(suspicious_flags):
            ax.scatter(
                sizes[suspicious_flags], fracs[suspicious_flags],
                s=30, c="#e74c3c", alpha=0.9,
                label=f"suspicious (n={int(suspicious_flags.sum())})",
                edgecolors="none", zorder=3,
            )
        ax.axvline(
            float(min_community_size), color="#2c3e50",
            linestyle="--", linewidth=1.0,
            label=f"min_community_size = {int(min_community_size)}",
        )
        if np.isfinite(floor_threshold):
            ax.axhline(
                floor_threshold, color="#8e44ad",
                linestyle="--", linewidth=1.0,
                label=f"floor_threshold = {floor_threshold:.3f}",
            )
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.0)
        # Shade the top-right "atypical zone" using current xlim for width.
        if np.isfinite(floor_threshold):
            x_lo, x_hi = ax.get_xlim()
            zone_x = max(float(min_community_size), x_lo)
            ax.add_patch(mpatches.Rectangle(
                (zone_x, floor_threshold),
                x_hi - zone_x,
                1.0 - floor_threshold,
                facecolor="#e74c3c", alpha=0.08,
                edgecolor="none", zorder=0,
            ))
        # Annotate suspicious communities with their label id.
        for sz, fr, lbl in zip(
            sizes[suspicious_flags],
            fracs[suspicious_flags],
            labels_arr[suspicious_flags],
        ):
            ax.annotate(
                str(int(lbl)), xy=(float(sz), float(fr)),
                xytext=(3, 3), textcoords="offset points",
                fontsize=7, color="#7b241c", alpha=0.9,
            )
        ax.legend(
            loc="upper left", fontsize=8,
            framealpha=0.0, edgecolor="none",
        )
    else:
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Community size (log)")
    ax.set_ylabel("fraction_aberrant")
    ax.set_title("(d) Community size vs fraction_aberrant")




def _plot_atypical_lca_routed_diagnostic_grid(
    *,
    abnormality: np.ndarray,
    is_aberrant: np.ndarray,
    bgmm,
    component_report: List[Dict[str, Any]],
    aberrant_component_indices: List[int],
    threshold: float,
    bgmm_halted: bool,
    predicted_class_names: np.ndarray,
    community_stats: List[Dict[str, Any]],
    min_community_size: int,
    halted: bool,
    route: str,
    lca: List[str],
    lca_is_root_only: bool,
    g1_resolution: float,
    g1_ari_scores: Dict[float, float],
    g5_per_depth_summary: List[Dict[str, Any]],
    output_path: str,
    boxplot_top_k: int = 30,
    boxplot_min_class_size: int = 20,
) -> None:
    """Render the LCA-routed diagnostic grid (3x2).

    Panels (a)–(d) reflect the upstream BGMM seed selection plus G1's chosen
    Leiden partition (the partition that maximised ARI to the predicted classes).
    Panels (e) and (f) surface the routed signals exposed by
    :meth:`HECTOR.evaluate_cells`:

      * (e) **G1 ARI-vs-resolution scan** — line plot of every Leiden
        resolution tested (``g1_ari_scores``) with the chosen
        ``g1_resolution`` highlighted as a vertical line. Diagnoses
        whether the auto-selected resolution sits at a clear ARI peak or
        on a flat plateau.
      * (f) **G5 multi-resolution rollup** — twin horizontal bars per
        ``d_cut`` showing ``n_atypical`` and ``n_suspicious_communities``.
        Only populated on the multi-lineage route; single-lineage runs
        render a placeholder explaining that G5 did not run.

    The supertitle records the route taken (``single_lineage`` vs
    ``multi_lineage``), the LCA node IDs (truncated for display), and
    the halt label so the routing decision is visible at a glance.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    from scipy.stats import norm as scipy_norm

    abn = np.asarray(abnormality, dtype=np.float64)
    aberrant = np.asarray(is_aberrant, dtype=bool)
    finite = np.isfinite(abn)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]
    ax20, ax21 = axes[2, 0], axes[2, 1]

    # --- (a) 1-D BGMM on abnormality --------------------------------------
    if np.any(finite):
        bins = np.linspace(0.0, 1.0, 61)
        ax00.hist(
            abn[finite], bins=bins, density=True,
            color="#bfd3e6", alpha=0.55, edgecolor="none",
            label=f"abnormality (n={int(finite.sum())})",
        )
        if bgmm is not None and component_report:
            x_grid = np.linspace(0.0, 1.0, 400)
            cmap = plt.get_cmap("tab10")
            seed_set = set(int(i) for i in aberrant_component_indices)
            non_seed_drawn = 0
            total_pdf = np.zeros_like(x_grid)
            for rep in component_report:
                if not (rep.get("axiom1_mass") and rep.get("axiom2_cohesion")):
                    continue
                mu = float(rep["mean"])
                sigma = float(rep["std"])
                w = float(rep["weight"])
                if sigma < 1e-8:
                    continue
                pdf = w * scipy_norm.pdf(x_grid, mu, sigma)
                total_pdf = total_pdf + pdf
                k = int(rep["component"])
                if k in seed_set:
                    ax00.plot(
                        x_grid, pdf,
                        color="#e74c3c", linewidth=2.0, alpha=0.95,
                        label=(
                            f"seed comp {k}: "
                            f"mu={mu:.3f}, sigma={sigma:.3f}, w={w:.3f}"
                        ),
                    )
                else:
                    ax00.plot(
                        x_grid, pdf,
                        color=cmap(non_seed_drawn % 10),
                        linewidth=1.0, alpha=0.7,
                        label=(
                            f"comp {k}: "
                            f"mu={mu:.3f}, sigma={sigma:.3f}, w={w:.3f}"
                        ),
                    )
                    non_seed_drawn += 1
            if np.any(total_pdf > 0):
                ax00.plot(
                    x_grid, total_pdf,
                    color="#2c3e50", linewidth=1.0, linestyle="--",
                    alpha=0.7, label="mixture (cohesive)",
                )
        ax00.axvline(
            float(threshold),
            color="#27ae60", linestyle=":", linewidth=1.4,
            label=f"threshold = {float(threshold):.2f}",
        )
        ax00.axvspan(
            float(threshold), 1.0,
            color="#e74c3c", alpha=0.05,
        )
        ax00.set_xlim(0.0, 1.0)
        ax00.set_xlabel("abnormality")
        ax00.set_ylabel("density")
        ax00.legend(
            loc="upper right", fontsize=7,
            framealpha=0.0, edgecolor="none",
        )
    else:
        ax00.text(
            0.5, 0.5, "No qualifying cells",
            ha="center", va="center",
            transform=ax00.transAxes, fontsize=12, color="#7f8c8d",
        )
        ax00.set_xticks([])
        ax00.set_yticks([])
    ax00.set_title(
        f"(a) 1-D BGMM on abnormality "
        f"(bgmm_halted={bool(bgmm_halted)})",
        fontsize=11,
    )

    # --- (b) abnormality boxplot across top-K classes ---------------------
    classes = np.asarray(predicted_class_names)
    finite_class = finite & (classes != None)  # noqa: E711
    if np.any(finite_class):
        cls_finite = classes[finite_class]
        abn_finite = abn[finite_class]
        unique_cls, counts = np.unique(cls_finite, return_counts=True)
        keep = counts >= int(boxplot_min_class_size)
        unique_cls = unique_cls[keep]
        if unique_cls.size > 0:
            medians = np.array([
                float(np.median(abn_finite[cls_finite == c]))
                for c in unique_cls
            ], dtype=np.float64)
            order = np.argsort(-medians)[:int(boxplot_top_k)]
            sel_classes = unique_cls[order]
            data = [
                abn_finite[cls_finite == c] for c in sel_classes
            ]
            labels = [str(c) for c in sel_classes]
            bp = ax01.boxplot(
                data,
                labels=labels,
                vert=False,
                showfliers=False,
                patch_artist=True,
            )
            cmap = plt.get_cmap("tab20")
            for k, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(cmap(k % 20))
                patch.set_alpha(0.6)
            ax01.axvline(
                float(threshold),
                color="#27ae60", linestyle=":", linewidth=1.2,
                label=f"threshold = {float(threshold):.2f}",
            )
            ax01.invert_yaxis()
            ax01.set_xlabel("abnormality")
            ax01.tick_params(axis="y", labelsize=7)
            ax01.set_xlim(0.0, 1.0)
            ax01.legend(
                loc="lower right", fontsize=7,
                framealpha=0.0, edgecolor="none",
            )
            ax01.set_title(
                f"(b) abnormality by class (top {len(sel_classes)} "
                f"by median, min_size={int(boxplot_min_class_size)})",
                fontsize=11,
            )
        else:
            ax01.text(
                0.5, 0.5,
                f"No class has >= {int(boxplot_min_class_size)} cells",
                ha="center", va="center",
                transform=ax01.transAxes, fontsize=12, color="#7f8c8d",
            )
            ax01.set_xticks([])
            ax01.set_yticks([])
            ax01.set_title("(b) abnormality by class", fontsize=11)
    else:
        ax01.text(
            0.5, 0.5, "No qualifying cells",
            ha="center", va="center",
            transform=ax01.transAxes, fontsize=12, color="#7f8c8d",
        )
        ax01.set_xticks([])
        ax01.set_yticks([])
        ax01.set_title("(b) abnormality by class", fontsize=11)

    # --- (c) suspicious community bars (G1 partition) --------------------
    _draw_suspicious_community_bars(ax10, community_stats)

    # --- (d) community size vs fraction_aberrant (G1 partition) ----------
    _draw_size_vs_fraction_panel(
        ax11,
        community_stats,
        min_community_size=int(min_community_size),
    )

    # --- (e) G1 ARI scan -------------------------------------------------
    if g1_ari_scores:
        items = sorted(g1_ari_scores.items(), key=lambda kv: float(kv[0]))
        xs = np.array([float(k) for k, _ in items], dtype=np.float64)
        ys = np.array([float(v) for _, v in items], dtype=np.float64)
        ax20.plot(
            xs, ys, marker="o", markersize=4,
            color="#2c3e50", linewidth=1.2, alpha=0.9,
            label="ARI vs predicted classes",
        )
        if np.isfinite(g1_resolution):
            ax20.axvline(
                float(g1_resolution),
                color="#e74c3c", linestyle="--", linewidth=1.4,
                label=f"chosen resolution = {float(g1_resolution):.2f}",
            )
            best_ari = float(g1_ari_scores.get(float(g1_resolution), float("nan")))
            if np.isfinite(best_ari):
                ax20.scatter(
                    [float(g1_resolution)], [best_ari],
                    s=60, color="#e74c3c", zorder=3,
                    edgecolors="white", linewidths=1.2,
                    label=f"best ARI = {best_ari:.3f}",
                )
        ax20.set_xlabel("Leiden resolution")
        ax20.set_ylabel("ARI to predicted classes")
        ax20.set_ylim(min(0.0, float(np.nanmin(ys)) - 0.02),
                      max(1.0, float(np.nanmax(ys)) + 0.02))
        ax20.legend(
            loc="lower right", fontsize=8,
            framealpha=0.0, edgecolor="none",
        )
    else:
        ax20.text(
            0.5, 0.5, "No ARI scan recorded",
            ha="center", va="center",
            transform=ax20.transAxes, fontsize=12, color="#7f8c8d",
        )
        ax20.set_xticks([])
        ax20.set_yticks([])
    ax20.set_title("(e) G1 auto-resolution Leiden ARI scan", fontsize=11)

    # --- (f) G5 multi-resolution rollup ----------------------------------
    if not lca_is_root_only:
        ax21.text(
            0.5, 0.5,
            "Single-lineage route — G5 not run",
            ha="center", va="center",
            transform=ax21.transAxes, fontsize=12, color="#7f8c8d",
        )
        ax21.set_xticks([])
        ax21.set_yticks([])
    elif g5_per_depth_summary:
        depths = np.array(
            [float(d["d_cut"]) for d in g5_per_depth_summary],
            dtype=np.float64,
        )
        n_atyp = np.array(
            [int(d.get("n_atypical", 0)) for d in g5_per_depth_summary],
            dtype=np.int64,
        )
        n_susp = np.array(
            [int(d.get("n_suspicious_communities", 0))
             for d in g5_per_depth_summary],
            dtype=np.int64,
        )
        n_comms = np.array(
            [int(d.get("n_communities", 0)) for d in g5_per_depth_summary],
            dtype=np.int64,
        )
        ys = np.arange(len(depths))
        bar_h = 0.38
        ax21.barh(
            ys - bar_h / 2.0, n_atyp, height=bar_h,
            color="#e74c3c", edgecolor="none",
            label="n_atypical",
        )
        ax21.barh(
            ys + bar_h / 2.0, n_susp, height=bar_h,
            color="#3498db", edgecolor="none",
            label="n_suspicious_communities",
        )
        for y, na, ns, nc in zip(ys, n_atyp, n_susp, n_comms):
            ax21.text(
                float(max(na, ns)) * 1.02, float(y),
                f"  {int(nc)} communities",
                va="center", ha="left",
                fontsize=7, color="#2c3e50",
            )
        ax21.set_yticks(ys)
        ax21.set_yticklabels([f"d={d:.2f}" for d in depths], fontsize=8)
        ax21.invert_yaxis()
        ax21.set_xlabel("Count")
        ax21.legend(
            loc="lower right", fontsize=8,
            framealpha=0.0, edgecolor="none",
        )
    else:
        ax21.text(
            0.5, 0.5, "Multi-lineage route — no G5 depth summary recorded",
            ha="center", va="center",
            transform=ax21.transAxes, fontsize=12, color="#7f8c8d",
        )
        ax21.set_xticks([])
        ax21.set_yticks([])
    ax21.set_title("(f) G5 multi-resolution rollup (per-depth)", fontsize=11)

    halt_label = (
        "BGMM_HALTED" if bgmm_halted
        else ("HALTED" if halted else "non-halted")
    )
    lca_display = sorted(str(n) for n in (lca or []))
    lca_str = (
        ",".join(lca_display[:3])
        + ("…" if len(lca_display) > 3 else "")
    ) if lca_display else "∅"
    fig.suptitle(
        "HECTOR LCA-routed atypical detection — "
        f"[{str(route)}] LCA={lca_str} ({len(lca_display)} node(s))  "
        f"[{halt_label}]  aberrant={int(aberrant.sum())}",
        fontsize=13,
    )
    fig.subplots_adjust(top=0.95, hspace=0.40, wspace=0.25)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Standalone diagnostic — density surface of an obs label
# =============================================================================
#
# General-purpose visualization utility: render a Gaussian-smoothed 2-D density
# grid over any pair of numeric ``adata.obs`` columns, and overlay cells that
# match a user-supplied label value. Independent of the atypical-detection
# pipeline — works with any categorical obs column and any two numeric obs
# columns.
# =============================================================================


def plot_obs_label_density(
    adata: Any,
    obs_column: str,
    label_value: Any,
    *,
    x_column: Union[str, "np.ndarray"] = "adherence_score",
    y_column: Union[str, "np.ndarray"] = "jsd",
    output_path: str,
    hist_bins: int = 256,
    blur_sigma: float = 2.0,
    max_background_cells: int = 200_000,
    elev: float = 30.0,
    azim: float = -60.0,
    highlight_color: str = "red",
    background_color: str = "lightgrey",
    title: Optional[str] = None,
    plot_mode: str = "3d",
    highlight_mode: str = "both",
) -> None:
    """Density-surface diagnostic: highlight ``obs[obs_column] == label_value``.

    Builds a 2-D histogram density grid over ``(adata.obs[x_column],
    adata.obs[y_column])`` (smoothed with a Gaussian blur) and overlays two
    scatter layers:

    * Background cells (``lightgrey``), optionally downsampled to
      ``max_background_cells`` for rendering speed.
    * Highlighted cells matching ``obs[obs_column] == label_value`` (or
      ``obs[obs_column].isin(label_value)`` if a list/tuple/set is supplied).

    Rendering style is controlled by ``plot_mode``:

    * ``"3d"`` (default) — density surface raised as a 3-D mountain; cells
      rendered at their local density height.  ``.html`` output uses
      Plotly ``Surface`` + ``Scatter3d``; other extensions fall back to
      ``matplotlib`` (``mpl_toolkits.mplot3d``).
    * ``"2d"`` — density surface shown as a flat heatmap background; cells
      rendered at their ``(x, y)`` coordinates.  No Z axis.  ``.html`` output
      uses Plotly ``Heatmap`` + ``Scatter``.

    This function is independent of the atypical-detection pipeline — any
    pair of numeric obs columns and any categorical obs column will work.
    The defaults ``x_column='adherence_score'`` and ``y_column='jsd'`` match
    the columns written by :meth:`HECTOR.evaluate_cells`, but can be
    overridden with any other numeric obs columns.

    Args:
        adata: ``AnnData`` with ``obs_column`` in ``adata.obs``.
        obs_column: Column name in ``adata.obs`` to pull the highlight label
            from.
        label_value: Scalar value (or list/tuple/set of values) to highlight.
            Cells with ``adata.obs[obs_column] == label_value`` are plotted
            in ``highlight_color``.
        x_column: Column name in ``adata.obs`` **or** a pre-built numeric
            array (e.g. ``adata.obsm['X_umap'][:, 0]``) used for the X axis.
            Default ``'adherence_score'``.
        y_column: Column name in ``adata.obs`` **or** a pre-built numeric
            array (e.g. ``adata.obsm['X_umap'][:, 1]``) used for the Y axis.
            Default ``'jsd'``.
        output_path: File path for the rendered plot.  ``.html`` extensions
            trigger the interactive Plotly path; all other extensions use the
            static Matplotlib path.
        hist_bins: Histogram bins per axis for the density grid.
        blur_sigma: Gaussian blur sigma (in bins) applied to the 2-D histogram.
        max_background_cells: Downsample background cells to at most this many
            for the scatter overlay.  Highlighted cells are never downsampled.
        elev: Elevation angle for the static 3-D Matplotlib path; ignored in
            two-dimensional mode.
        azim: Azimuth angle for the static 3-D Matplotlib path; ignored in
            two-dimensional mode.
        highlight_color: Color of the highlighted overlay.
        background_color: Color of the background-cell overlay.
        title: Optional plot title.  If omitted, a sensible default is used.
        plot_mode: ``'3d'`` (default) renders a 3-D density mountain;
            ``'2d'`` renders a flat density heatmap without a Z axis.
        highlight_mode: How to render highlighted cells. ``'both'``
            (default) draws a smoothed red density layer under small red
            dots (enrichment stripes + per-cell detail); ``'density'``
            omits the dots; ``'points'`` draws only translucent dots.
    """
    if plot_mode not in ("2d", "3d"):
        raise ValueError(
            f"plot_mode must be '2d' or '3d'; got {plot_mode!r}"
        )
    if highlight_mode not in ("points", "density", "both"):
        raise ValueError(
            f"highlight_mode must be 'points', 'density', or 'both'; "
            f"got {highlight_mode!r}"
        )

    import pandas as _pd

    if obs_column not in adata.obs.columns:
        raise KeyError(
            f"obs column {obs_column!r} not found in adata.obs "
            f"(available: {list(adata.obs.columns)[:8]}{'...' if adata.obs.shape[1] > 8 else ''})"
        )

    # x_column / y_column may be either a string obs column name or a
    # pre-built numeric array (e.g. adata.obsm['X_umap'][:, 0]).
    def _resolve_axis(col, label: str) -> "np.ndarray":
        if isinstance(col, str):
            if col not in adata.obs.columns:
                raise KeyError(
                    f"obs column {col!r} not found in adata.obs; cannot build "
                    f"the density surface.  Pass x_column / y_column as a "
                    f"column name string or a numeric array."
                )
            return np.asarray(
                _pd.to_numeric(adata.obs[col], errors="coerce"),
                dtype=np.float64,
            )
        # Array-like path — convert directly.
        arr = np.asarray(col, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != adata.n_obs:
            raise ValueError(
                f"{label} array must be 1-D with length adata.n_obs "
                f"({adata.n_obs}); got shape {arr.shape}."
            )
        return arr

    x = _resolve_axis(x_column, "x_column")
    y = _resolve_axis(y_column, "y_column")

    # Derive human-readable axis labels from the column spec.
    x_label = x_column if isinstance(x_column, str) else "x"
    y_label = y_column if isinstance(y_column, str) else "y"
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        raise ValueError(
            f"All values in ({x_column}, {y_column}) are non-finite; nothing "
            f"to plot."
        )

    obs_vals = adata.obs[obs_column].to_numpy()
    if isinstance(label_value, (list, tuple, set, np.ndarray)):
        label_set = set(label_value) if not isinstance(label_value, set) else label_value
        highlight_mask = np.array(
            [v in label_set for v in obs_vals.tolist()], dtype=bool
        )
        label_desc = f"{obs_column} in {sorted(map(str, list(label_set)))[:6]}" + (
            " ..." if len(label_set) > 6 else ""
        )
    else:
        highlight_mask = (obs_vals == label_value)
        if not highlight_mask.any():
            highlight_mask = np.array(
                [str(v) == str(label_value) for v in obs_vals.tolist()],
                dtype=bool,
            )
        label_desc = f"{obs_column} == {label_value!r}"

    highlight_mask = highlight_mask & finite
    background_mask = (~highlight_mask) & finite

    x_fin = x[finite]
    y_fin = y[finite]
    n_bins = int(hist_bins)
    hist, x_edges, y_edges = np.histogram2d(x_fin, y_fin, bins=n_bins)
    try:
        from scipy.ndimage import gaussian_filter as _gauss
        smoothed = _gauss(hist, sigma=float(blur_sigma), mode="nearest")
    except Exception:
        smoothed = hist.astype(np.float64)
    smoothed = smoothed.astype(np.float64)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    # ==== Use a square plotting domain so density patterns are not stretched
    x_min = float(x_edges[0])
    x_max = float(x_edges[-1])
    y_min = float(y_edges[0])
    y_max = float(y_edges[-1])
    x_span = x_max - x_min
    y_span = y_max - y_min
    square_span = max(x_span, y_span)
    if not np.isfinite(square_span) or square_span <= 0:
        square_span = 1.0

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    x_plot_limits = (x_mid - 0.5 * square_span, x_mid + 0.5 * square_span)
    y_plot_limits = (y_mid - 0.5 * square_span, y_mid + 0.5 * square_span)

    x_bin_all = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins - 1)
    y_bin_all = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins - 1)
    density_at_cell = smoothed[x_bin_all, y_bin_all]

    # Highlight-only density grid — reused by 2D/3D, plotly/matplotlib.
    hi_smoothed = None
    if highlight_mode in ("density", "both") and np.any(highlight_mask):
        hi_hist, _, _ = np.histogram2d(
            x[highlight_mask], y[highlight_mask],
            bins=[x_edges, y_edges],
        )
        try:
            from scipy.ndimage import gaussian_filter as _gauss
            hi_smoothed = _gauss(
                hi_hist, sigma=float(blur_sigma), mode="nearest",
            ).astype(np.float64)
        except Exception:
            hi_smoothed = hi_hist.astype(np.float64)

    # Place the highlighted density just above the overall density surface to
    # avoid z-fighting while preserving the height of the underlying surface.
    hi_surface_z = None
    hi_surface_max = None
    if hi_smoothed is not None and np.any(hi_smoothed > 0):
        hi_surface_max = float(np.max(hi_smoothed))
        surface_lift = max(float(np.max(smoothed)) * 0.005, 1e-12)
        hi_surface_z = np.where(
            hi_smoothed > 0,
            smoothed + surface_lift,
            np.nan,
        )

    output_suffix = os.path.splitext(str(output_path))[1].lower()
    wants_html = output_suffix in (".html", ".htm")

    plot_title = title or f"Density surface — highlight: {label_desc}"
    n_hi = int(highlight_mask.sum())
    n_bg = int(background_mask.sum())
    square_canvas_px = 900
    square_static_inches = 9

    if wants_html:
        try:
            import plotly.graph_objects as _go
            bg_idx = np.flatnonzero(background_mask)
            if bg_idx.size > int(max_background_cells):
                rng = np.random.default_rng(42)
                bg_idx = rng.choice(bg_idx, size=int(max_background_cells),
                                    replace=False)
            fig = _go.Figure()

            if plot_mode == "2d":
                fig.add_trace(
                    _go.Scatter(
                        x=x[bg_idx], y=y[bg_idx],
                        mode="markers",
                        name=f"Other (n={n_bg})",
                        marker=dict(size=2, color=background_color, opacity=0.85),
                    )
                )
                if (
                    n_hi > 0
                    and highlight_mode in ("density", "both")
                    and hi_smoothed is not None
                ):
                    hi_z = hi_smoothed.T.copy()
                    hi_z[hi_z <= 0] = None
                    fig.add_trace(
                        _go.Heatmap(
                            x=x_centers, y=y_centers, z=hi_z,
                            colorscale="Reds",
                            opacity=0.55,
                            showscale=False,
                            name=f"{label_desc} (density)",
                        )
                    )
                if n_hi > 0 and highlight_mode in ("points", "both"):
                    hi_idx = np.flatnonzero(highlight_mask)
                    hi_size = 2 if highlight_mode == "points" else 1.5
                    hi_opac = 0.45 if highlight_mode == "points" else 0.25
                    fig.add_trace(
                        _go.Scatter(
                            x=x[hi_idx], y=y[hi_idx],
                            mode="markers",
                            name=f"{label_desc} (n={n_hi})",
                            marker=dict(
                                size=hi_size,
                                color=highlight_color,
                                opacity=hi_opac,
                            ),
                        )
                    )
                fig.update_layout(
                    title=plot_title,
                    width=square_canvas_px,
                    height=square_canvas_px,
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis=dict(
                        showgrid=True, gridcolor="#e8e8e8",
                        zeroline=False, showline=True,
                        linecolor="#888888",
                        range=list(x_plot_limits),
                        constrain="domain",
                    ),
                    yaxis=dict(
                        showgrid=True, gridcolor="#e8e8e8",
                        zeroline=False, showline=True,
                        linecolor="#888888",
                        range=list(y_plot_limits),
                        scaleanchor="x",
                        scaleratio=1,
                        constrain="domain",
                    ),
                    legend=dict(
                        x=0.02, y=0.98,
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="#cccccc", borderwidth=0.5,
                    ),
                )
            else:
                X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
                fig.add_trace(
                    _go.Surface(
                        x=X, y=Y, z=smoothed,
                        colorscale="Blues",
                        opacity=0.4,
                        showscale=False,
                        name="Density",
                    )
                )
                if hi_surface_z is not None:
                    fig.add_trace(
                        _go.Surface(
                            x=X,
                            y=Y,
                            z=hi_surface_z,
                            surfacecolor=hi_smoothed,
                            colorscale="Reds",
                            cmin=0.0,
                            cmax=hi_surface_max,
                            opacity=0.65,
                            showscale=False,
                            name=f"{label_desc} (density)",
                        )
                    )
                fig.add_trace(
                    _go.Scatter3d(
                        x=x[bg_idx], y=y[bg_idx], z=density_at_cell[bg_idx],
                        mode="markers",
                        name=f"Other (n={n_bg})",
                        marker=dict(size=1.5, color=background_color, opacity=0.15),
                    )
                )
                if n_hi > 0 and highlight_mode in ("points", "both"):
                    hi_idx = np.flatnonzero(highlight_mask)
                    hi_size = 1.0 if highlight_mode == "points" else 0.8
                    hi_opac = 0.5 if highlight_mode == "points" else 0.35
                    fig.add_trace(
                        _go.Scatter3d(
                            x=x[hi_idx], y=y[hi_idx],
                            z=density_at_cell[hi_idx],
                            mode="markers",
                            name=f"{label_desc} (n={n_hi})",
                            marker=dict(
                                size=hi_size,
                                color=highlight_color,
                                opacity=hi_opac,
                            ),
                        )
                    )
                fig.update_layout(
                    title=plot_title,
                    width=square_canvas_px,
                    height=square_canvas_px,
                    scene=dict(
                        xaxis=dict(title=x_label, range=list(x_plot_limits)),
                        yaxis=dict(title=y_label, range=list(y_plot_limits)),
                        zaxis=dict(title="Density"),
                        aspectmode="manual",
                        aspectratio=dict(x=1, y=1, z=0.65),
                    ),
                    paper_bgcolor="white",
                    legend=dict(
                        x=0.02, y=0.98,
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="#cccccc", borderwidth=0.5,
                    ),
                )

            fig.write_html(output_path)
            return
        except ImportError as exc:
            raise ImportError(
                "plotly is required for .html output from "
                "plot_obs_label_density"
            ) from exc

    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for the static plot; install with "
            "`pip install matplotlib` or pass an output path ending in .html "
            "with plotly installed."
        )

    bg_idx = np.flatnonzero(background_mask)
    if bg_idx.size > int(max_background_cells):
        rng = np.random.default_rng(42)
        bg_idx = rng.choice(bg_idx, size=int(max_background_cells),
                            replace=False)

    if plot_mode == "2d":
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots(
            figsize=(square_static_inches, square_static_inches),
        )
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Background cells — plain dot scatter (no density heatmap); keep
        # the raw point cloud visible so the reader sees the full
        # distribution of non-highlighted cells.
        ax.scatter(
            x[bg_idx], y[bg_idx],
            c=background_color, s=4, alpha=0.85,
            edgecolors="none", zorder=1,
        )

        # Highlight density — Reds, alpha-masked.
        if (
            n_hi > 0
            and highlight_mode in ("density", "both")
            and hi_smoothed is not None
        ):
            hi_max = float(hi_smoothed.max()) if hi_smoothed.max() > 0 else 1.0
            hi_alpha_arr = (np.clip(hi_smoothed / hi_max, 0.0, 1.0) ** 0.55) * 0.7
            ax.imshow(
                hi_smoothed.T, origin="lower",
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                aspect="auto", cmap="Reds",
                alpha=hi_alpha_arr.T, zorder=2,
            )

        # Highlight dots — mode-dependent.
        if n_hi > 0 and highlight_mode in ("points", "both"):
            hi_idx = np.flatnonzero(highlight_mask)
            hi_s = 1.5 if highlight_mode == "points" else 1.0
            hi_a = 0.25 if highlight_mode == "points" else 0.15
            ax.scatter(
                x[hi_idx], y[hi_idx],
                c=highlight_color, s=hi_s, alpha=hi_a,
                edgecolors="none", zorder=4,
            )

        # Chart chrome.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")
        ax.grid(True, color="#e8e8e8", linewidth=0.5, alpha=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#333333", labelsize=9)
        ax.set_xlabel(x_label, fontsize=10, color="#333333")
        ax.set_ylabel(y_label, fontsize=10, color="#333333")
        ax.set_xlim(x_plot_limits)
        ax.set_ylim(y_plot_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.set_title(
            plot_title, fontsize=13, loc="left", pad=18, color="#222222",
        )
        ax.text(
            0.0, 1.01,
            f"highlighted = {n_hi:,}    |    other = {n_bg:,}",
            transform=ax.transAxes,
            fontsize=9, color="#666666", style="italic",
            ha="left", va="bottom",
        )

        legend_items = [
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=background_color, markeredgecolor="none",
                   markersize=6, label=f"Other (n={n_bg:,})"),
        ]
        if n_hi > 0:
            legend_items.append(
                Line2D([0], [0], marker="o", linestyle="none",
                       markerfacecolor=highlight_color, markeredgecolor="none",
                       markersize=6,
                       label=f"{label_desc} (n={n_hi:,})"),
            )
        ax.legend(
            handles=legend_items, loc="upper right",
            frameon=False, fontsize=9,
        )

        fig.tight_layout()
        fig.savefig(
            output_path, dpi=200, facecolor="white",
        )
        plt.close(fig)
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from matplotlib.lines import Line2D
        X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
        fig = plt.figure(figsize=(square_static_inches, square_static_inches))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(projection="3d")
        ax.set_facecolor("white")
        ax.plot_surface(
            X, Y, smoothed,
            cmap="Blues",
            alpha=0.30,
            linewidth=0,
            antialiased=True,
        )
        if hi_surface_z is not None:
            hi_normalized = np.clip(hi_smoothed / hi_surface_max, 0.0, 1.0)
            hi_facecolors = plt.get_cmap("Reds")(hi_normalized)
            hi_facecolors[..., 3] = (hi_normalized ** 0.55) * 0.75
            ax.plot_surface(
                X,
                Y,
                hi_surface_z,
                facecolors=hi_facecolors,
                linewidth=0,
                antialiased=True,
                shade=False,
            )
        ax.scatter(
            x[bg_idx], y[bg_idx], density_at_cell[bg_idx],
            c=background_color, s=1.5, alpha=0.06,
            depthshade=False,
            edgecolors="none",
        )
        if n_hi > 0 and highlight_mode in ("points", "both"):
            hi_idx = np.flatnonzero(highlight_mask)
            hi_s = 1.0 if highlight_mode == "points" else 0.7
            hi_a = 0.35 if highlight_mode == "points" else 0.20
            ax.scatter(
                x[hi_idx], y[hi_idx], density_at_cell[hi_idx],
                c=highlight_color, s=hi_s, alpha=hi_a,
                depthshade=False, edgecolors="none",
            )
        ax.set_xlabel(x_label, fontsize=10, color="#333333")
        ax.set_ylabel(y_label, fontsize=10, color="#333333")
        ax.set_zlabel("Density", fontsize=10, color="#333333")
        ax.set_xlim(x_plot_limits)
        ax.set_ylim(y_plot_limits)
        ax.set_box_aspect((1, 1, 0.65))
        ax.view_init(elev=float(elev), azim=float(azim))
        ax.set_title(
            plot_title, fontsize=13, loc="left", pad=14, color="#222222",
        )
        legend_items = [
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=background_color, markeredgecolor="none",
                   markersize=6, label=f"Other (n={n_bg:,})"),
        ]
        if n_hi > 0:
            legend_items.append(
                Line2D([0], [0], marker="o", linestyle="none",
                       markerfacecolor=highlight_color, markeredgecolor="none",
                       markersize=6,
                       label=f"{label_desc} (n={n_hi:,})"),
            )
        ax.legend(
            handles=legend_items, loc="upper right",
            frameon=False, fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(
            output_path, dpi=200, facecolor="white",
        )
        plt.close(fig)
