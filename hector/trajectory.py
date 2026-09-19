"""Public trajectory analysis API for HECTOR.

This module provides trajectory visualization, comparison, pseudotime, and
gene-trend analysis APIs. Layout engines, color mappers, graph utilities, and
the simplification subsystem live in
``hector.trajectory_ontology``; rendering backends live in
``hector.trajectory_render``.
"""

from __future__ import annotations

import json
import logging
import numpy as np
import os
import warnings
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from .trajectory_ontology import (
    TrajectoryAnalysisConfig,
    ElasticLayoutConfig,
    SoftEdgeConfig,
    compute_elastic_depths,
    discover_inferred_paths,
    SimpleOBOParser,
    CanonicalOrderComputer,
    AdaptiveOntologySubgraph,
    TreeLayoutEngine,
    RadialTreeLayoutEngine,
    DAGToTreeConverter,
    _apply_direction_filter,
    _create_projected_lineage_color_mapper,
    HighlightColorMapper,
    compute_cloud_cell_counts,
    compute_cloud_transition_counts,
    _build_placer_pixel_geometry,
    format_node_names,
    compute_node_depths_longest_path,
    VisualizationCache,
    PLOTLY_AVAILABLE,
    MATPLOTLIB_AVAILABLE,
)
from .trajectory_support import (
    HighlightData,
    LatentBarycentricPlacer,
    AtypicalClusterer,
    AtypicalClustererConfig,
    _build_canonical_trajectory_context,
    _build_visual_edges,
    _compute_adherence_scores,
    compute_adaptive_pdf_ribbon_scale,
)

logger = logging.getLogger(__name__)

TRAJECTORY_STATE_COLUMN = 'trajectory_state'

class HierarchicalTrajectoryAnalyzer:
    """
    Main orchestrator for the Trajectory Analysis and Visualization.
    
    This is the recommended tool for showing cells in the context
    of the ontology hierarchy and identifying transitioning vs stable cells.
    
    Supports two output formats:
    - 'html': Interactive Plotly visualization with clickable CellxGene links
    - 'pdf': Static matplotlib visualization for publication-quality figures
    """
    
    def __init__(self, config: Optional[TrajectoryAnalysisConfig] = None):
        self.config = config or TrajectoryAnalysisConfig()
        self._viz_cache = VisualizationCache()
        self._last_inferred_path_gate1_verdicts = {}
    
    def _compute_trajectory_placement_context(
        self,
        predictor,
        adata,
        viz_data: Dict[str, Any],
        use_grit: bool = True,
    ) -> Dict[str, Any]:
        """Compute the shared inputs needed to build the full trajectory placer.

        Encapsulates inferred-path discovery, direction filtering,
        active-subgraph construction, and GRIT-refined anchor resolution
        for trajectory placement. The returned arrays use the same placement
        conventions as :meth:`visualize`.

        Args:
            predictor: HECTOR instance.
            adata: AnnData object that has been processed by ``predict()``.
            viz_data: Output of ``predictor.get_visualization_vectors(adata)``.
            use_grit: When True, pass GRIT-refined predictions as the
                placer's ``prior_anchors``. When False, the caller should
                use ``None`` for ``prior_anchors`` and let the placer fall
                back to argmax-of-weights.

        Returns:
            Dictionary containing canonical and inferred edges, filtered cell
            and node arrays, masks, counts, and optional prior anchors used to
            construct a trajectory placer.
        """
        cell_vectors = viz_data['cell_vectors']
        node_vectors = viz_data['node_vectors']
        predictions = viz_data['predictions']
        node_names = viz_data['node_names']

        # Build adjacency: prefer the model's pure ontology adjacency unless
        # the caller's config supplied an explicit OBO file.
        n_nodes = len(node_names)
        name_to_idx = {name: i for i, name in enumerate(node_names)}
        if self.config.ontology_path:
            if not os.path.exists(self.config.ontology_path):
                raise FileNotFoundError(
                    f"Ontology file not found: {self.config.ontology_path}"
                )
            g_obo = SimpleOBOParser.parse(self.config.ontology_path)
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
            min_cells_number=self.config.min_cells_number,
        )
        active_indices = list(canonical_context["active_indices"])
        node_counts = dict(canonical_context["node_counts"])
        keep_mask = np.asarray(canonical_context["keep_mask"], dtype=bool)

        # Restrict GRIT-refined anchors to cells retained by canonical pruning.
        if use_grit:
            prediction_indices = predictions[keep_mask].astype(np.int64)
        else:
            prediction_indices = None

        # ------------------------------------------------------------------
        # Soft/inferred path discovery uses the visualization configuration.
        # ------------------------------------------------------------------
        node_depths_dag = compute_node_depths_longest_path(adjacency_matrix)
        inferred_path_weights: Dict[Tuple[int, int], float] = {}
        lateral_directions: Dict[Tuple[int, int], Tuple[int, int]] = {}
        augmented_adjacency = adjacency_matrix

        gat_embeddings = viz_data.get('gat_embeddings')
        ontology_adj = viz_data.get('ontology_adj')
        id_to_name_map = getattr(predictor, 'id_to_name_map', {})

        # Inferred-path discovery uses the ``SoftEdgeConfig`` defaults.
        if (
            self.config.enable_inferred_paths
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
                        cache=self._viz_cache,
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
        # ``canonical_context['tree_edges']`` supplies the canonical edge list.
        canonical_tree_edges = list(canonical_context['tree_edges'])
        tree_edges = _build_visual_edges(
            canonical_edges=canonical_tree_edges,
            inferred_path_weights=inferred_path_weights,
            active_indices=active_indices,
        )

        # Remove inferred edges that overlap a canonical edge in either direction.
        # The placer
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

    def visualize(
        self,
        predictor,  # HECTOR
        adata,  # AnnData
        output_file: str = "HECTOR_Tree_Visualization.html",
        # GRIT controls for Unity Mode
        use_grit: bool = True,
        grit_refinement_percentile: Optional[float] = None,
        label_format: str = 'id',
        force_recompute: bool = False,
    ) -> go.Figure:
        """
        Create the hierarchical tree visualization with Unity Mode support.
        
        When use_grit=True (Unity Mode), the visualization uses GRIT-refined
        predictions from the predictor to ensure cells appear exactly where
        the prediction report says they belong.
        
        Inferred paths are direction-filtered and deduplicated against canonical
        ontology edges before rendering.
        
        Args:
            predictor: HECTOR instance
            adata: AnnData object
            output_file: Path to save output. Format is inferred from extension:
                - '.html' → interactive Plotly visualization
                - '.pdf' → static matplotlib figure
            use_grit: Enable GRIT refinement in predictor (default: True)
            grit_refinement_percentile: Controls which cells get GRIT refinement
                - None (default): Auto-detect using Multi-Otsu thresholding
                - 0-100: Manual override - refine top X% low-confidence prediction
            label_format: Format for node labels (default: 'id')
                - 'id': Show only ontology IDs (e.g., 'CL:0000128')
                - 'name': Show only cell type names (e.g., 'monocyte')
                - 'both': Show both (e.g., 'monocyte (CL:0000128)')
            force_recompute: If True, bypass all cached embeddings and predictions
                and re-run the full pipeline from scratch (default: False)
            
        Returns:
            Plotly figure for HTML output or Matplotlib figure for PDF output.
        """
        # Infer output format from file extension
        _ext = os.path.splitext(output_file)[1].lower()
        if _ext == '.pdf':
            _output_format = 'pdf'
        elif _ext in ('.html', '.htm'):
            _output_format = 'html'
        else:
            raise ValueError(
                f"Unsupported output file extension '{_ext}'. "
                f"Use '.html' for interactive Plotly or '.pdf' for static matplotlib."
            )
        
        # Validate required dependencies for chosen format
        if _output_format == 'html':
            if not PLOTLY_AVAILABLE:
                raise ImportError("Plotly required for HTML output. Install with: pip install plotly")
        elif _output_format == 'pdf':
            if not MATPLOTLIB_AVAILABLE:
                raise ImportError("Matplotlib required for PDF output. Install with: pip install matplotlib")
        
        unity_mode = "On" if use_grit else "Off"
        print("\nHECTOR Tree Visualization")
        print(f"Unity Mode: GRIT={unity_mode}")
        print(f"Label Format: {label_format}")
        print("="*40)
        
        # Materialize a view before local annotation writes. Direct callers should
        # pass a materialized AnnData when they need writes on their own object.
        if adata.is_view:
            print("[INFO] adata is a view — materializing copy for annotation safety")
            adata = adata.copy()

        # Reset per-call Gate 1 state so a reused analyzer / re-run adata never carries stale verdicts.
        self._last_inferred_path_gate1_verdicts = {}
        adata.uns.pop('hector_inferred_path_gate1', None)

        # Step 1: Extract vectors (with GRIT if enabled)
        viz_data = predictor.get_visualization_vectors(
            adata,
            use_grit=use_grit,
            grit_refinement_percentile=grit_refinement_percentile,
            force_recompute=force_recompute
        )
        
        cell_vectors = viz_data['cell_vectors']
        node_vectors = viz_data['node_vectors']
        predictions = viz_data['predictions']
        node_names = viz_data['node_names']
        
        # Get ID-to-name mapping from predictor (if available)
        id_to_name_map = getattr(predictor, 'id_to_name_map', {})
        
        # Determine line separator based on output format
        # PDF (matplotlib) uses '\n', HTML (Plotly) uses '<br>'
        line_sep = '\n' if _output_format == 'pdf' else '<br>'
        
        # Create formatted node names based on label_format
        formatted_node_names = format_node_names(node_names, id_to_name_map, label_format, line_sep)
        formatted_original_node_names = list(formatted_node_names)
        
        unique_preds = np.unique(predictions)
        
        # Step 2: Build the direct ontology graph from the checkpoint or an OBO file.
        name_to_idx = {name: i for i, name in enumerate(node_names)}
        n_nodes = len(node_names)
        
        if self.config.ontology_path:
            # User explicitly provided cl.obo file - use it
            if not os.path.exists(self.config.ontology_path):
                raise FileNotFoundError(f"Ontology file not found: {self.config.ontology_path}")
            
            G_obo = SimpleOBOParser.parse(self.config.ontology_path)
            
            # Convert OBO graph to adjacency matrix
            adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)
            for parent_str, child_str in G_obo.edges():
                if parent_str in name_to_idx and child_str in name_to_idx:
                    p_idx = name_to_idx[parent_str]
                    c_idx = name_to_idx[child_str]
                    adjacency_matrix[c_idx, p_idx] = 1.0
        else:
            # Use the checkpoint's direct, unweighted ontology adjacency.
            pure_adj = viz_data.get('pure_adjacency_matrix')
            if pure_adj is None:
                raise ValueError(
                    "Model checkpoint does not contain 'pure_adjacency_matrix'. "
                    "This model was trained before this feature was added. "
                    "Please provide ontology_path='cl.obo' or re-train with updated code."
                )
            adjacency_matrix = pure_adj
        
        subgraph = AdaptiveOntologySubgraph(
            adjacency_matrix=adjacency_matrix,
            node_names=node_names,
            co_graph=None
        )
        
        canonical_order = CanonicalOrderComputer(subgraph.nx_graph, subgraph.node_names)
        
        canonical_context = _build_canonical_trajectory_context(
            adjacency_matrix=adjacency_matrix,
            node_names=node_names,
            predicted_indices=predictions,
            min_cells_number=self.config.min_cells_number,
        )
        active_indices = list(canonical_context["active_indices"])
        node_counts = dict(canonical_context["node_counts"])
        self._nodes_below_threshold = dict(canonical_context["nodes_below_threshold"])

        print(
            f"[INFO] Pruning cell types below {self.config.min_cells_number} cells: "
            f"{len(self._nodes_below_threshold)} cell types removed."
        )

        keep_mask = np.asarray(canonical_context["keep_mask"], dtype=bool)

        cell_vectors = cell_vectors[keep_mask]
        predictions = predictions[keep_mask]

        # Unity Mode: Refined predictions
        refined_predictions = viz_data['predictions'][keep_mask] if use_grit else None

        # =====================================================================
        # Soft-edge discovery and elastic layout
        # =====================================================================
        # Candidate links augment the direct ontology graph for visualization;
        # elastic edge length reflects the configured structural/embedding blend.
        
        # Initialize inferred path tracking
        inferred_path_weights = {}
        lateral_directions = {}
        augmented_adjacency = adjacency_matrix  # Default: use original adjacency
        elastic_depths = None  # Default: use rigid integer depths
        
        # Get GAT embeddings for inferred path discovery and elastic layout
        gat_embeddings = viz_data.get('gat_embeddings')
        ontology_adj = viz_data.get('ontology_adj')
        
        # Compute node depths from adjacency matrix (needed for inferred path filtering and layout)
        node_depths_dag = compute_node_depths_longest_path(adjacency_matrix)
        
        # === Soft Edge Discovery ===
        # Candidate-link weights use ``SoftEdgeConfig`` defaults.
        if self.config.enable_inferred_paths and gat_embeddings is not None and ontology_adj is not None:
            print("\n[INFO] Discovering inferred paths...")

            inferred_path_config = SoftEdgeConfig(enable_inferred_paths=True)
            
            # Identify high-weight candidate links.
            try:
                augmented_adjacency, inferred_path_weights, lateral_directions = discover_inferred_paths(
                    pure_ontology_adj=adjacency_matrix,
                    ontology_adj=ontology_adj,
                    node_embeddings=gat_embeddings,
                    config=inferred_path_config,
                    cache=self._viz_cache,
                )                
                # ===================================================================
                # STAGE 1: DIRECTION FILTERING (Primitive → Differentiated)
                # ===================================================================
                # Inferred paths must respect differentiation direction. Defaults:
                #   - Cross-level pairs: low depth → high depth.
                #   - Lateral pairs (same depth): embedding-gradient canonical direction.
                # A semantic name-based override reverses the default
                # when one side's name matches a developmental-precursor pattern
                # (e.g. "immature ", "early ", "pre-", "plasmablast"). See
                # ``_apply_direction_filter`` and ``_semantic_direction`` for details.

                direction_filtered, _dir_counters = _apply_direction_filter(
                    inferred_path_weights=inferred_path_weights,
                    lateral_directions=lateral_directions,
                    node_depths_dag=node_depths_dag,
                    node_names=node_names,
                    id_to_name_map=id_to_name_map,
                )
                inferred_path_weights = direction_filtered
                
            except Exception as e:
                print(f"  > WARNING: Inferred path discovery failed: {e}")
                print(f"  > Falling back to pure ontology adjacency")
                augmented_adjacency = adjacency_matrix
                inferred_path_weights = {}
                lateral_directions = {}
        elif self.config.enable_inferred_paths:
            print("\n[INFO] Inferred paths disabled: Missing gat_embeddings or ontology_adj in checkpoint")
        
        # === Elastic Layout Computation ===
        # Use the direct ontology DAG for layout to avoid cycles from candidate links.
        if self.config.enable_elastic_layout and gat_embeddings is not None:
            print("[INFO] Computing elastic depths...")
            print("  > Using PURE ontology DAG for layout pipeline (no inferred paths)")
            
            # Create elastic layout config from trajectory analysis config
            elastic_config = ElasticLayoutConfig(
                elasticity_factor=self.config.elasticity_factor,
                max_deviation=self.config.elastic_max_deviation,
                min_layer_gap=self.config.elastic_min_layer_gap,
                min_node_dist=self.config.elastic_min_node_dist,
                enable_elastic_layout=True
            )
            
            # Compute node depths from PURE adjacency matrix (topological sort)
            # Using pure DAG ensures no cycles that would break topological sort
            node_depths = compute_node_depths_longest_path(adjacency_matrix)  # PURE DAG
            
            try:
                elastic_depths = compute_elastic_depths(
                    adjacency_matrix=adjacency_matrix,  # direct ontology DAG
                    node_embeddings=gat_embeddings,
                    node_depths=node_depths,
                    config=elastic_config
                )
                if elastic_depths:
                    depth_values = [elastic_depths[i] for i in active_indices if i in elastic_depths]
                    if depth_values:
                        min_depth, max_depth = min(depth_values), max(depth_values)
                        print(f"  > Elastic depth range: [{min_depth:.2f}, {max_depth:.2f}]")
                        if max_depth > 50:
                            print(f"  > WARNING: Elastic depth max ({max_depth:.2f}) > 50. Consider adjusting thresholds.")
            except Exception as e:
                print(f"  > WARNING: Elastic depth computation failed: {e}")
                print(f"  > Falling back to rigid ontology depths")
                elastic_depths = None
        elif self.config.enable_elastic_layout:
            print("[INFO] Elastic layout disabled: Missing gat_embeddings in checkpoint")
        
        print("="*40)
        print("[INFO] Starting Trajectory Analysis...")
        # =====================================================================
        # STATISTICAL CONSISTENCY PASS (The "Canonical DAG" Check)
        # =====================================================================
        # Perform adherence analysis on the CANONICAL DAG (unique nodes) first.
        # This bases atypical status on the configured adherence calculation,
        # not on the diluted probabilities of the visual tree (Radial Mode).
                
        # Diagnose unexpected cycles before layout.
        _debug_G = nx.DiGraph()
        _debug_rows, _debug_cols = np.where(adjacency_matrix > 0)
        for _r, _c in zip(_debug_rows, _debug_cols):
            _debug_G.add_edge(int(_c), int(_r))  # Parent -> Child flow
        _debug_cycles = list(nx.simple_cycles(_debug_G))
        if _debug_cycles:
            print(f"  > WARNING: adjacency_matrix has {len(_debug_cycles)} cycles!")
            print(f"  > First cycle: {_debug_cycles[0][:5]}...")  # Show first 5 nodes
        else:
            pass
        
        # 2. Source the canonical per-cell atypical mask and adherence
        # from the routed evaluate_cells pipeline (adata.obs), which is the
        # single source of truth for atypical classification. When
        # enable_atypical_detection=False, evaluate_cells is skipped and a fallback
        # adherence is computed locally for layout purposes; the mask is
        # all-False in that case.
        _cr_predictions = refined_predictions if refined_predictions is not None else predictions
        if self.config.enable_atypical_detection:
            if "is_atypical" not in adata.obs.columns or "adherence_score" not in adata.obs.columns:
                raise RuntimeError(
                    "enable_atypical_detection=True requires adata.obs['is_atypical'] "
                    "and adata.obs['adherence_score'] written by HECTOR.evaluate_cells. "
                    "Run create_trajectory_analysis which invokes evaluate_cells, or "
                    "call predictor.evaluate_cells(adata) before visualize()."
                )
            canonical_is_atypical = np.asarray(
                adata.obs["is_atypical"].values, dtype=bool,
            )[keep_mask]
            canonical_adherence = np.asarray(
                adata.obs["adherence_score"].values, dtype=np.float64,
            )[keep_mask]
        else:
            canonical_is_atypical = np.zeros(int(np.sum(keep_mask)), dtype=bool)
            canonical_adherence = _compute_adherence_scores(
                cell_embeddings=cell_vectors,
                node_embeddings=node_vectors,
                active_indices=list(canonical_context["active_indices"]),
                tree_edges=list(canonical_context["tree_edges"]),
                temperature=self.config.temperature,
                top_k_neighbors=self.config.top_k_neighbors,
                predicted_classes=_cr_predictions,
                class_names=list(node_names),
                class_relative_adherence=self.config.class_relative_adherence,
                class_relative_blend_weight=self.config.class_relative_blend_weight,
            )

        # Step 3: Compute tree layout
        # Choose layout engine based on layout_mode config
        is_polar = (self.config.layout_mode == 'radial')
        

        # Layout uses the direct ontology DAG.
        # The layout pipeline uses the pure ontology DAG to avoid cycles from inferred paths
        # This ensures topological sort and causality enforcement work correctly
        print("[INFO] Layout pipeline: Using PURE ontology DAG")
        layout_adjacency = adjacency_matrix
        
        if is_polar:
            # === DAG to Tree Conversion for Clean Radial Layout ===
            # Convert the DAG to a strict tree by replicating nodes with multiple parents.
            # This ensures each lineage stays in its own angular sector.
            print("Converting DAG to Strict Tree for Radial Layout...")
            
            converter = DAGToTreeConverter(layout_adjacency, node_names, node_vectors)
            tree_data = converter.convert(active_indices, predictions, cell_vectors,
                                          min_cells_number=self.config.min_cells_number)
            
            # Map source DAG indices to their tree-clone indices.
            _debug_original_to_tree_map = {}
            for new_idx, old_idx in enumerate(tree_data['original_indices']):
                if old_idx not in _debug_original_to_tree_map:
                    _debug_original_to_tree_map[old_idx] = []
                _debug_original_to_tree_map[old_idx].append(new_idx)
            
            # Overwrite data with expanded tree versions
            layout_adjacency = tree_data['tree_adj']
            node_names = tree_data['tree_names']
            node_vectors = tree_data['tree_vectors']
            predictions = tree_data['tree_predictions']
            active_indices = tree_data['active_indices']

            # Apply cell mask: drop cells whose tree node was pruned by min_cells_number.
            # All cell-parallel arrays must stay in sync with predictions.
            _tree_cell_mask = tree_data['cell_keep_mask']
            cell_vectors = cell_vectors[_tree_cell_mask]
            canonical_is_atypical = canonical_is_atypical[_tree_cell_mask]
            canonical_adherence = canonical_adherence[_tree_cell_mask]
            if refined_predictions is not None:
                refined_predictions = refined_predictions[_tree_cell_mask]

            # Update keep_mask to reflect the additional cells dropped by the tree converter.
            # keep_mask is in adata space; _tree_cell_mask is in post-SCC-pruned space.
            # We fold _tree_cell_mask back into keep_mask so kept_indices stays correct
            # for the adata tagging block below.
            _keep_positions = np.where(keep_mask)[0]          # adata indices of SCC-kept cells
            _dropped = _keep_positions[~_tree_cell_mask]       # adata indices now also dropped
            keep_mask = keep_mask.copy()
            keep_mask[_dropped] = False
            
            # Update formatted names for the new tree structure
            line_sep = '\n' if _output_format == 'pdf' else '<br>'
            formatted_node_names = format_node_names(node_names, id_to_name_map, label_format, line_sep)
            
            # Remap candidate-link weights to tree indices.
            # The inferred paths were discovered on DAG indices and direction-filtered.
            # After tree conversion, we need to remap them to the new tree indices
            # while preserving the direction constraint.
            # 
            # IMPORTANT: We only keep inferred paths where BOTH endpoints exist in the tree.
            # Many inferred paths will be dropped because:
            # 1. Source/target node was not in active_indices (pruned before tree conversion)
            # 2. Source/target node was filtered during tree conversion (no cells assigned)
            if inferred_path_weights:
                remapped_inferred_path_weights = {}
                
                # Build the tree-index to DAG-index mapping.
                tree_to_original = {new_idx: old_idx for new_idx, old_idx in enumerate(tree_data['original_indices'])}
                
                # Compute tree depths for direction checks.
                # Note: tree_data['tree_adj'] uses same convention as adjacency_matrix
                tree_node_depths = compute_node_depths_longest_path(tree_data['tree_adj'])
                
                for (old_src, old_tgt), weight in inferred_path_weights.items():
                    # Check if both endpoints exist in the tree
                    # _debug_original_to_tree_map contains: {original_DAG_idx: [tree_idx1, tree_idx2, ...]}
                    if old_src in _debug_original_to_tree_map and old_tgt in _debug_original_to_tree_map:
                        # For each combination of new indices, create a inferred path
                        new_src_list = _debug_original_to_tree_map[old_src]
                        new_tgt_list = _debug_original_to_tree_map[old_tgt]
                        for new_src in new_src_list:
                            for new_tgt in new_tgt_list:
                                # Only add if they're different nodes
                                if new_src != new_tgt:
                                    # Preserve the direction constraint after cloning.
                                    # (DAG-to-tree conversion can change depths)
                                    src_depth_tree = tree_node_depths.get(new_src, 0)
                                    tgt_depth_tree = tree_node_depths.get(new_tgt, 0)
                                    
                                    if src_depth_tree <= tgt_depth_tree:
                                        # Direction OK: primitive → differentiated (or lateral)
                                        remapped_inferred_path_weights[(new_src, new_tgt)] = weight
                
                inferred_path_weights = remapped_inferred_path_weights

                # --- Remap lateral_directions to tree index space ---
                # lateral_directions maps DAG-index pairs to canonical (src, tgt) tuples.
                # After tree conversion each DAG node may expand into one or more tree
                # clone nodes, so we must produce a new dict keyed by tree-index pairs.
                #
                # For each lateral DAG edge (dag_src, dag_tgt) → canonical (dag_parent, dag_child),
                # enumerate all tree clone combinations (tree_src, tree_tgt) and store the
                # corresponding canonical (tree_parent, tree_child) pair.  Both the forward
                # and reverse lookup keys are stored so downstream code can look up either
                # direction, matching the convention used in trajectory_ontology.py.
                if lateral_directions:
                    remapped_lateral_directions: Dict[Tuple[int, int], Tuple[int, int]] = {}
                    # Iterate over unique canonical entries only (avoid double-processing
                    # forward/reverse duplicates stored by discover_inferred_paths).
                    seen_dag_pairs: set = set()
                    for (dag_src, dag_tgt), (dag_parent, dag_child) in lateral_directions.items():
                        canonical_dag_key = (min(dag_src, dag_tgt), max(dag_src, dag_tgt))
                        if canonical_dag_key in seen_dag_pairs:
                            continue
                        seen_dag_pairs.add(canonical_dag_key)

                        if dag_src not in _debug_original_to_tree_map or dag_tgt not in _debug_original_to_tree_map:
                            continue  # At least one endpoint was pruned from the tree

                        tree_src_list = _debug_original_to_tree_map[dag_src]
                        tree_tgt_list = _debug_original_to_tree_map[dag_tgt]

                        # Determine which DAG node is parent vs child in the canonical direction
                        # so we can map to the correct tree clone indices.
                        for tree_src in tree_src_list:
                            for tree_tgt in tree_tgt_list:
                                if tree_src == tree_tgt:
                                    continue
                                # Map canonical (dag_parent, dag_child) → (tree_parent, tree_child)
                                # dag_parent == dag_src  →  tree_parent == tree_src
                                # dag_parent == dag_tgt  →  tree_parent == tree_tgt
                                if dag_parent == dag_src:
                                    tree_canonical = (tree_src, tree_tgt)
                                else:
                                    tree_canonical = (tree_tgt, tree_src)

                                # Store both lookup directions (forward and reverse)
                                remapped_lateral_directions[(tree_src, tree_tgt)] = tree_canonical
                                remapped_lateral_directions[(tree_tgt, tree_src)] = tree_canonical

                    lateral_directions = remapped_lateral_directions
                # --- End lateral_directions remapping ---

                # Store tree_to_original mapping for Stage 3 deduplication
                _tree_to_original = tree_to_original
            else:
                # No inferred paths, but still need empty mapping for consistency
                _tree_to_original = {new_idx: old_idx for new_idx, old_idx in enumerate(tree_data['original_indices'])}
            
            # Remap elastic depths to tree indices.
            # The tree conversion changed all node indices. We must map the 
            # calculated elastic depths (DAG indices) to the new Tree indices.
            if elastic_depths is not None:
                new_elastic_depths = {}
                # tree_data['original_indices'] maps New_Tree_Index -> Old_DAG_Index
                for new_idx, old_idx in enumerate(tree_data['original_indices']):
                    if old_idx in elastic_depths:
                        # Assign each DAG node's depth to all tree clones.
                        new_elastic_depths[new_idx] = elastic_depths[old_idx]
                
                elastic_depths = new_elastic_depths
                print(f"  > Remapped elastic depths for {len(elastic_depths)} tree nodes")
            # --------------------------------------------------------------
            
            # Update GRIT prior anchors to tree indices.
            # The converter has already remapped cells to the correct new node clones
            # (stored in tree_data['tree_predictions']). We must apply this remapping
            # to refined_predictions so GRIT anchors match the layout nodes.
            # Prior anchors must use tree indices after clone expansion.
            if use_grit and refined_predictions is not None:
                refined_predictions = tree_data['tree_predictions']
            # ---------------------------------------------------------------------------------
            
            # === End DAG to Tree Conversion ===
            
            layout_engine = RadialTreeLayoutEngine(layout_adjacency, node_names, self.config)
            layout_engine.canonical_order = canonical_order
            layout_engine.original_indices = list(tree_data['original_indices'])
        else:
            layout_engine = TreeLayoutEngine(layout_adjacency, node_names, self.config)
            layout_engine.canonical_order = canonical_order
            # Horizontal mode: no DAG-to-Tree conversion, indices are unchanged
            layout_engine.original_indices = None
            _tree_to_original = None
        
        layout_engine.set_embeddings(node_vectors)
        
        # Attach elastic depths to layout engine if available
        if elastic_depths is not None:
            layout_engine.elastic_depths = elastic_depths
        
        # =============================================================================
        # IMPORTANT: Internal vs Display Naming Convention
        # =============================================================================
        # - node_names: CL IDs (e.g., "CL:0000128") - USE FOR ALL INTERNAL LOGIC
        # - formatted_node_names: Display labels (e.g., "monocyte") - USE FOR DISPLAY ONLY
        #
        # Internal logic includes:
        #   - Color mapping (AdaptiveLineageColorMapper)
        #   - Index lookups (name_to_idx dictionaries)
        #   - Graph operations (NetworkX node identifiers)
        #   - URL generation (CellxGene links)
        #
        # Display logic includes:
        #   - Node labels on the tree
        #   - Hover text
        #   - Legend entries
        #   - Trajectory source/target labels
        # =============================================================================
        
        layout_engine.set_embeddings(node_vectors)
        
        # Attach elastic depths to layout engine if available
        if elastic_depths is not None:
            layout_engine.elastic_depths = elastic_depths

        pure_adj = viz_data.get('pure_adjacency_matrix')
        if pure_adj is None:
            raise RuntimeError(
                "Model does not contain 'pure_adjacency_matrix'.\n"
                "This is required for adaptive lineage coloring.\n"
                "Solution: Provide ontology_path='cl.obo' in TrajectoryAnalysisConfig"
            )

        color_mapper = None

        # =============================================================================
        # RADIAL ADAPTIVE COLORING: project full-ontology lineages onto the tree
        # before layout so branch ribbons can use the same connected branch colors.
        # =============================================================================
        if is_polar:
            if tree_data is None:
                raise RuntimeError("Radial layout expected tree_data after DAG-to-tree conversion.")

            try:
                tree_visible_edges: List[Tuple[int, int]] = []
                rows, cols = np.where(tree_data['tree_adj'] > 0)
                for child_idx, parent_idx in zip(rows, cols):
                    tree_visible_edges.append((int(parent_idx), int(child_idx)))

                color_mapper = _create_projected_lineage_color_mapper(
                    pure_adjacency=pure_adj,
                    original_node_names=viz_data['node_names'],
                    active_source_indices=[int(p) for p in viz_data['predictions']],
                    visible_node_names=node_names,
                    visible_edges=tree_visible_edges,
                    color_palette=self.config.adaptive_color_palette,
                    visible_to_source={
                        new_idx: int(old_idx)
                        for new_idx, old_idx in enumerate(tree_data['original_indices'])
                    },
                    legend_formatted_names=formatted_original_node_names,
                )
                layout_engine.set_color_mapper(color_mapper)
            except Exception as e:
                error_msg = (
                    f"Adaptive lineage coloring failed: {str(e)}\n"
                    f"This visualization requires 'pure_adjacency_matrix' in the model.\n"
                    f"Please provide ontology_path='cl.obo' in config and try again."
                )
                raise RuntimeError(error_msg) from e
        
        # Compute layout with PURE DAG
        if is_polar:
            node_positions = layout_engine.compute_layout(
                active_indices,
                augmented_adjacency=layout_adjacency,
            )
        else:
            node_positions = layout_engine.compute_layout(
                active_indices,
                augmented_adjacency=layout_adjacency,
            )
        
        # Precompute edge curvature scales for radial mode (before placer)
        if is_polar:
            layout_engine.compute_edge_curvature_scales(node_positions)

        # Get canonical tree edges from layout engine (pure DAG edges)
        canonical_tree_edges = layout_engine.tree_edges
        
        # === VISUAL PIPELINE: Augmented adjacency for rendering ===
        # Combine canonical edges with inferred paths for visualization
        # Inferred paths are rendered as dashed gray lines, cells can flow along them
        print("[INFO] Visual pipeline: Building visual edges")
        visual_tree_edges = _build_visual_edges(
            canonical_edges=canonical_tree_edges,
            inferred_path_weights=inferred_path_weights,
            active_indices=active_indices
        )
        
        # Store inferred path weights for rendering
        layout_engine.inferred_path_weights = inferred_path_weights
        
        if not is_polar:
            try:
                color_mapper = _create_projected_lineage_color_mapper(
                    pure_adjacency=pure_adj,
                    original_node_names=viz_data['node_names'],
                    active_source_indices=[int(p) for p in viz_data['predictions']],
                    visible_node_names=node_names,
                    visible_edges=canonical_tree_edges,
                    visible_node_indices=active_indices,
                    color_palette=self.config.adaptive_color_palette,
                    legend_formatted_names=formatted_original_node_names,
                )
            except Exception as e:
                error_msg = (
                    f"Adaptive lineage coloring failed: {str(e)}\n"
                    f"This visualization requires 'pure_adjacency_matrix' in the model.\n"
                    f"Please provide ontology_path='cl.obo' in config and try again."
                )
                raise RuntimeError(error_msg) from e
        
        cell_colors = color_mapper.get_colors(predictions)
        
        # Step 3.5: Compute per-node cell counts for proportional cloud scaling
        node_cell_counts = {}
        for pred in predictions:
            node_cell_counts[pred] = node_cell_counts.get(pred, 0) + 1
        
        # Step 4: Place cells (Trajectory Analysis - Barycentric Method).
        # Under pixel-space sizing, the placer builds cell jitter from a
        # PixelGeometry that matches the renderer's final viewport, so
        # clouds render as pixel-space circles regardless of axis aspect.
        # Pre-filter inferred paths to exclude any that overlap with canonical edges
        # This ensures canonical edges always use Bezier curves, never straight lines
        canonical_edges_set = set(canonical_tree_edges)
        canonical_edges_set.update((v, u) for u, v in canonical_tree_edges)  # Both directions
        placer_inferred_path_weights = {
            edge: weight for edge, weight in inferred_path_weights.items()
            if edge not in canonical_edges_set and (edge[1], edge[0]) not in canonical_edges_set
        }

        _placer_pixel_geometry = _build_placer_pixel_geometry(
            self.config, node_positions, active_indices, is_polar, _output_format,
        )

        # Per-node rendered branch widths, populated by both layout
        # engines after compute_layout. The placer uses them to scale
        # transitioning-cell jitter to ~40% of the local branch width so
        # cells visibly fan out within the branch's footprint.
        #
        # Unit conversion to the active backend's pixels:
        #   HTML (plotly):           pass-through. Engines populate node_widths
        #                            in HTML display pixels (the values plotly
        #                            interprets directly as line.width or
        #                            polygon offsets).
        #   PDF radial (matplotlib): LineCollection.linewidths is in POINTS and
        #                            the renderer applies a 1.5× internal scale.
        #                            Convert points → pixels at PDF DPI.
        #   PDF horizontal:          polygon-based; the renderer uses
        #                            compute_adaptive_pdf_ribbon_scale() to size
        #                            ribbons proportionally. Mirror that here.
        if hasattr(layout_engine, 'node_widths') and layout_engine.node_widths:
            if _output_format == 'pdf':
                if is_polar:
                    pt_to_px = self.config.pdf_dpi / 72.0
                    _w_scale = 1.5 * pt_to_px
                else:
                    _node_spacing_px = self.config.y_scale * _placer_pixel_geometry.px_per_data_y()
                    _w_scale = compute_adaptive_pdf_ribbon_scale(_node_spacing_px)
            else:
                _w_scale = 1.0
            _placer_node_widths = {
                n: float(w) * _w_scale for n, w in layout_engine.node_widths.items()
            }
        else:
            _placer_node_widths = None

        cell_placer = LatentBarycentricPlacer(
            node_positions=node_positions,
            node_embeddings=node_vectors,
            tree_edges=visual_tree_edges,
            temperature=self.config.temperature,
            transition_stream_width=self.config.transition_stream_width,
            top_k_neighbors=self.config.top_k_neighbors,
            stable_source_zone=self.config.stable_source_zone,
            stable_target_zone=self.config.stable_target_zone,
            node_cell_counts=node_cell_counts,
            scale_cloud_by_count=self.config.scale_cloud_by_count,
            cloud_scale_exponent=self.config.cloud_scale_exponent,
            cloud_size_multiplier=self.config.cloud_size_multiplier,
            cloud_size_min=self.config.cloud_size_min,
            cloud_size_max=self.config.cloud_size_max,
            y_scale=self.config.y_scale,
            is_polar=is_polar,
            is_pdf_output=(_output_format == 'pdf'),
            inferred_path_weights=placer_inferred_path_weights,  # Pre-filtered: excludes canonical edges
            lateral_directions=lateral_directions,
            pixel_geometry=_placer_pixel_geometry,
            node_widths=_placer_node_widths,
            edge_curvature_scales=getattr(layout_engine, '_edge_curvature_scales', None),
        )
        
        # Unity Mode: Pass GRIT-refined predictions as prior_anchors
        # These guide placement for valid cells while atypical cells still go to atypical-cluster clouds
        # refined_predictions ALREADY defined above
        #
        # NOTE: `refined_predictions` comes from `viz_data['predictions']`
        # (GRIT-refined class indices, per `get_visualization_vectors()`'s
        # contract).  The simplify path's `prediction_indices` is sourced
        # from the same `core_result.top_indices` array, so the two anchor
        # arrays are identical by construction — only the variable names
        # differ.
        (cell_positions, is_transitioning, parent_nodes, child_nodes,
         transition_scores, raw_transition_scores, anchor_nodes,
         adherence_scores, is_atypical,
         absolute_transition_scores) = cell_placer.place_cells(
            cell_vectors,
            external_atypical_mask=canonical_is_atypical,
            external_adherence_scores=canonical_adherence,
            prior_anchors=refined_predictions,
            show_transitions=self.config.show_transitions,
        )
        # last_prediction_trajectory_conflict is set by place_cells; fall
        # back to all-'None' for stubbed/mocked placers in tests.
        prediction_trajectory_conflict = getattr(
            cell_placer,
            'last_prediction_trajectory_conflict',
            np.full(len(is_transitioning), 'None', dtype=object),
        )

        # Collect placement-derived cloud extents for ribbon blending and
        # contour bounding whenever either feature is in use.
        if self.config.cloud_blend_enabled or self.config.display_contours:
            cloud_extents = cell_placer.compute_cloud_extents()
        else:
            cloud_extents: Dict[int, float] = {}
        
        # Track atypical cells that were not assigned to any atypical cluster.
        unclustered_atypical_mask = np.zeros(len(predictions), dtype=bool)
        
        # Save original cell positions before atypical clustering may reposition them
        # This is needed for highlight overlay: cells should appear at their original
        # tree positions (on edges/nodes), not at pie chart hub positions.
        original_cell_positions = cell_positions.copy()
        
        # Step 4.5: Atypical-cell clustering
        atypical_cluster_infos = []
        
        # Cluster atypical cells only when they are rendered separately.
        # When atypical_clusters=False, atypical cells should be merged into stable/transitioning
        # and should NOT be clustered (which creates pie charts)
        if (self.config.enable_atypical_clustering and 
            self.config.enable_atypical_detection and 
            self.config.atypical_clusters and
            np.sum(is_atypical) > 0):
            print(f"  Running atypical-cell clustering on {np.sum(is_atypical)} atypical cells...")
            
            # Create AtypicalClustererConfig for clustering
            clusterer_config = AtypicalClustererConfig(
                atypical_anchor_count=self.config.atypical_anchor_count
            )
            
            clusterer = AtypicalClusterer(
                node_positions=node_positions,
                node_embeddings=node_vectors,
                active_indices=active_indices,
                config=clusterer_config,
                cell_size=self.config.cell_size,
                layout_x_scale=self.config.x_scale,
                is_polar=is_polar
            )
            
            # Cluster ALL atypical cells (both stable and transitioning)
            # Rationale: Atypical cells are uncertain, so separating them is overly precise
            if np.sum(is_atypical) > 0:
                print(f"  Clustering {np.sum(is_atypical)} atypical cells...")
                
                # Cluster all atypical cells (pass anchor_nodes for pie chart cell type counting)
                atypical_cluster_infos, unclustered_indices = clusterer.cluster_atypical(
                    cell_vectors,
                    is_atypical,
                    anchor_nodes,
                )

                if len(unclustered_indices) > 0:
                    unclustered_atypical_mask[unclustered_indices] = True

                # Update cell positions to cluster around atypical-cluster hubs
                if atypical_cluster_infos:
                    cell_positions = clusterer.place_atypical_cells(cell_positions, atypical_cluster_infos)
            else:
                print(f"  No atypical cells to cluster")
        
        # --- Highlight Data Preparation ---
        highlight_data = None
        highlight_color_mapper = None

        if self.config.highlight_obs is not None:
            # Custom obs column highlighting (takes priority over atypical_scatter)
            col = self.config.highlight_obs
            if col not in adata.obs.columns:
                print(f"  [WARNING] highlight_obs='{col}' not found in adata.obs. Skipping highlight.")
            else:
                obs_values = adata.obs[col].values[keep_mask]
                if self.config.highlight_groups is not None:
                    mask = np.isin(obs_values, self.config.highlight_groups)
                else:
                    mask = np.ones(len(obs_values), dtype=bool)
                cats = [str(v) for v in obs_values[mask]]
                unique_cats = sorted(set(cats))
                highlight_color_mapper = HighlightColorMapper(unique_cats, self.config.highlight_palette)
                highlight_data = HighlightData(
                    positions=cell_positions[mask],
                    categories=cats,
                    color_mapper=highlight_color_mapper,
                )

        elif self.config.atypical_scatter:
            # Atypical cell highlighting by anchor cell type
            # Use ORIGINAL positions (before atypical clustering repositioned them)
            # so cells appear at their natural tree positions, not at pie chart hubs.
            atyp_mask = is_atypical
            if np.any(atyp_mask):
                cats = [formatted_node_names[int(anchor_nodes[i])] for i in range(len(predictions)) if atyp_mask[i]]
                unique_cats = sorted(set(cats))
                highlight_color_mapper = HighlightColorMapper(unique_cats, self.config.highlight_palette)
                highlight_data = HighlightData(
                    positions=original_cell_positions[atyp_mask],
                    categories=cats,
                    color_mapper=highlight_color_mapper,
                )

        # --- Pie-chart-only HighlightColorMapper ---
        if highlight_color_mapper is None and self.config.atypical_clusters and atypical_cluster_infos:
            # Collect all cell type names from atypical clusters
            pie_cats = set()
            for cluster in atypical_cluster_infos:
                if cluster.cell_type_counts:
                    for ct_idx in cluster.cell_type_counts.keys():
                        pie_cats.add(formatted_node_names[ct_idx] if ct_idx < len(formatted_node_names) else f"Type {ct_idx}")
            if pie_cats:
                highlight_color_mapper = HighlightColorMapper(sorted(pie_cats), self.config.highlight_palette)

        # Compute per-node cell counts after atypical clustering.
        # Atypical cells stay excluded from the stable cloud regardless of cluster assignment.
        cloud_label_counts = compute_cloud_cell_counts(
            active_indices=active_indices,
            anchor_nodes=anchor_nodes,
            is_transitioning=is_transitioning,
            is_atypical=is_atypical,
            enable_atypical_detection=self.config.enable_atypical_detection
        )

        if self.config.show_transitions:
            cloud_transition_counts = compute_cloud_transition_counts(
                active_indices=active_indices,
                anchor_nodes=anchor_nodes,
                is_transitioning=is_transitioning,
            )
        else:
            cloud_transition_counts = None

        # Step 4.1: Tag Cells in AnnData (always enabled)
        full_status = np.full(len(adata), 'Pruned', dtype=object)
        full_source = np.full(len(adata), '', dtype=object)  # Parent node (tree direction)
        full_target = np.full(len(adata), '', dtype=object)  # Child node (tree direction)
        full_transition_position = np.full(len(adata), np.nan, dtype=float)  # Transition band, stretched
        full_relative_position = np.full(len(adata), np.nan, dtype=float)  # Rescaled per edge (dataset-fixed)
        full_absolute_position = np.full(len(adata), np.nan, dtype=float)  # Rescaled by node distance (model-fixed)
        full_conflict = np.full(len(adata), 'None', dtype=object)  # Anchor/best-edge disagreement

        # `keep_mask` maps from viz_data indices to kept cell indices
        if len(viz_data['predictions']) == len(adata):
            kept_indices = np.where(keep_mask)[0]

            # Initialize status based on transitioning state
            # This will be refined for atypical cells below (if detected and shown separately)
            full_status[kept_indices] = np.where(
                is_transitioning, 'Transitioning', 'Stable'
            )

            # Apply atypical labeling based on two-level control:
            # Level 1: enable_atypical_detection (controls BGMM classification)
            # Level 2: atypical_clusters (controls visualization of detected atypical cells)
            if self.config.enable_atypical_detection and self.config.atypical_clusters:
                cluster_id_by_viz_index = {}
                for cluster in atypical_cluster_infos:
                    for viz_idx in cluster.cell_indices:
                        cluster_id_by_viz_index[int(viz_idx)] = int(cluster.cluster_id)

                # Mode: Full - show atypical cells as clustered or unclustered groups.
                for i, kept_idx in enumerate(kept_indices):
                    if not is_atypical[i]:
                        continue

                    if i in cluster_id_by_viz_index:
                        full_status[kept_idx] = f'Atypical_{cluster_id_by_viz_index[i]}'
                    else:
                        full_status[kept_idx] = 'Atypical-Unclustered'

            # Build node-index → canonical CL ID lookup for stable, filterable tags.
            # - In horizontal mode: node_names[idx] is already the CL ID.
            # - In radial mode: node_names holds tree-clone names (e.g. 'CL:0002038_clone_1');
            #   we resolve back through _tree_to_original → original DAG index →
            #   viz_data['node_names'] to get the real CL ID.
            _original_cl_ids = viz_data['node_names']  # Always the canonical CL ID list
            if is_polar and _tree_to_original is not None:
                def _node_to_clid(tree_idx):
                    orig_idx = _tree_to_original.get(tree_idx, tree_idx)
                    if 0 <= orig_idx < len(_original_cl_ids):
                        return _original_cl_ids[orig_idx]
                    return node_names[tree_idx] if tree_idx < len(node_names) else ''
            else:
                def _node_to_clid(idx):
                    if 0 <= idx < len(node_names):
                        return node_names[idx]  # horizontal: node_names == CL IDs
                    return ''

            _canonical_edges = set(canonical_tree_edges)

            full_is_inferred_path = np.full(len(adata), False, dtype=bool)

            # Format trajectory endpoints with the same formatter that wrote
            # `hector_prediction`, so all three columns share an identical
            # surface string for the same node (joinable, value_counts groups
            # cleanly). Use plot-independent formatting — no <br> / \n.
            _has_format_prediction = predictor is not None and hasattr(
                predictor, '_format_prediction'
            )

            def _format_obs_label(cl_id: str) -> str:
                if not cl_id:
                    return ''
                if _has_format_prediction:
                    return predictor._format_prediction(cl_id, label_format)
                return cl_id

            # Assign parent/child nodes (formatted to match label_format) for
            # all kept cells. parent_nodes / child_nodes describe the topological
            # best edge for every cell that went through the best-edge analysis
            # (transitioning *and* stable). For the early-exit case (no edges or
            # show_transitions=False), parent_nodes falls back to the anchor and
            # child_nodes is -1 — we leave trajectory_target empty in that case.
            for i, kept_idx in enumerate(kept_indices):
                parent_node_idx = parent_nodes[i]
                child_node_idx = child_nodes[i]

                full_source[kept_idx] = _format_obs_label(_node_to_clid(parent_node_idx))
                if child_node_idx >= 0:
                    full_target[kept_idx] = _format_obs_label(_node_to_clid(child_node_idx))
                else:
                    full_target[kept_idx] = ''
                full_relative_position[kept_idx] = raw_transition_scores[i]
                full_absolute_position[kept_idx] = absolute_transition_scores[i]
                full_conflict[kept_idx] = prediction_trajectory_conflict[i]

                if is_transitioning[i]:
                    full_transition_position[kept_idx] = transition_scores[i]
                    # Flag inferred paths (soft edges)
                    is_canon = (parent_node_idx, child_node_idx) in _canonical_edges or \
                               (child_node_idx, parent_node_idx) in _canonical_edges
                    full_is_inferred_path[kept_idx] = not is_canon
                else:
                    full_transition_position[kept_idx] = 0.0

            # Assign to adata — view already materialized at method entry
            # (no need to copy here; the early copy ensures in-place writes work)
            adata.obs[TRAJECTORY_STATE_COLUMN] = full_status
            adata.obs['trajectory_source'] = full_source  # Parent node (upstream)
            adata.obs['trajectory_target'] = full_target  # Child node (downstream)
            # Three positions along the same edge, source (0) to target (1),
            # all linear rescalings of the same similarity difference.
            adata.obs['absolute_position'] = full_absolute_position  # Scaled by the distance between the two node prototypes, which the model fixes — the column to compare between datasets. NaN where there is no edge.
            adata.obs['relative_position'] = full_relative_position  # Scaled by that edge's own spread, so every edge uses the full [0, 1]. Cells in (stable_source_zone, 1 - stable_target_zone) are Transitioning, the end bands are Stable. Zones default to None → BGMM auto-detection; resolved values in adata.uns['hector_trajectory']['auto_zones'].
            adata.obs['transition_position'] = full_transition_position  # The transition band of relative_position stretched back out to [0, 1]; 0.0 for Stable cells, meaning "in the stable pool".
            if self.config.enable_inferred_paths:
                # Written when shortcut links are enabled; the column records
                # which placements use those links.
                adata.obs['is_inferred_path'] = full_is_inferred_path
            adata.obs['prediction_trajectory_conflict'] = full_conflict  # Diagnostic flag: 'None' / 'off-trajectory' (anchor not on best edge) / 'wrong_end' (stable cell at opposite endpoint from anchor).

            # The pruning threshold this run used. Recorded because
            # create_pseudotime_analysis has to split the ontology with the same
            # one to get the same tree, and has no other way to recover it.
            # Written unconditionally, unlike the zones below: it is known before
            # anything is placed.
            adata.uns.setdefault("hector_trajectory", {})["min_cells_number"] = int(
                self.config.min_cells_number
            )

            # Persist the resolved stable zones (and BGMM diagnostics
            # when auto-detected) so callers can recover what the placer
            # actually used regardless of whether values were user-
            # supplied or auto-detected. Skip when ``place_cells`` did
            # not resolve concrete values, because there is nothing meaningful
            # to record in that case.
            _src = cell_placer.stable_source_zone
            _tgt = cell_placer.stable_target_zone
            if _src is not None and _tgt is not None:
                _auto_info = cell_placer.last_auto_zone_info
                _components = _auto_info.get("components") if _auto_info else None
                adata.uns.setdefault("hector_trajectory", {})["auto_zones"] = {
                    "stable_source_zone": float(_src),
                    "stable_target_zone": float(_tgt),
                    "mode": _auto_info["mode"] if _auto_info else "user",
                    # Per-side, because ``mode`` reads "auto" as soon as ONE
                    # side is measured — without these a half-fallen-back run
                    # is indistinguishable from a fully measured one.
                    "src_status": (
                        _auto_info.get("src_status", "unknown")
                        if _auto_info else "user"
                    ),
                    "tgt_status": (
                        _auto_info.get("tgt_status", "unknown")
                        if _auto_info else "user"
                    ),
                    "components": json.dumps(_components) if _components is not None else "null",
                    "user_supplied_source": (
                        _auto_info.get("user_supplied_source", False)
                        if _auto_info else True
                    ),
                    "user_supplied_target": (
                        _auto_info.get("user_supplied_target", False)
                        if _auto_info else True
                    ),
                }

            # When canonical atypical detection ran, adata.obs['is_atypical'] and
            # adata.obs['adherence_score'] are already authoritatively
            # written by HECTOR.evaluate_cells. When detection is disabled we
            # still expose the layout-time adherence in obs, but we do NOT
            # write 'is_atypical' (no detection ran, so the column would be
            # vacuously False and misleading).
            if not self.config.enable_atypical_detection:
                full_adherence = np.full(len(adata), np.nan, dtype=float)
                full_adherence[kept_indices] = adherence_scores
                adata.obs['adherence_score'] = full_adherence

            # Add atypical-cluster metadata for atypical cells (only when atypical clusters are enabled)
            if self.config.atypical_clusters:
                full_atypical_cluster_id = np.full(len(adata), -1, dtype=int)
                full_atypical_anchor_nodes = np.full(len(adata), '', dtype=object)
                full_atypical_anchor_weights = np.full(len(adata), '', dtype=object)

                if atypical_cluster_infos:
                    for cluster in atypical_cluster_infos:
                        # Map cluster cell indices (in viz space) to adata indices
                        for viz_idx in cluster.cell_indices:
                            if viz_idx < len(kept_indices):
                                adata_idx = kept_indices[viz_idx]
                                full_atypical_cluster_id[adata_idx] = cluster.cluster_id
                                anchor_names = [formatted_node_names[idx] for idx in cluster.anchor_nodes]
                                full_atypical_anchor_nodes[adata_idx] = ','.join(anchor_names)
                                full_atypical_anchor_weights[adata_idx] = ','.join(
                                    [f'{w:.4f}' for w in cluster.anchor_weights]
                                )

                adata.obs['atypical_cluster_id'] = full_atypical_cluster_id
                adata.obs['atypical_anchor_nodes'] = full_atypical_anchor_nodes
                adata.obs['atypical_anchor_weights'] = full_atypical_anchor_weights

            clustered_atypical_mask = np.zeros(len(predictions), dtype=bool)
            if atypical_cluster_infos:
                clustered_indices = np.concatenate(
                    [np.asarray(cluster.cell_indices, dtype=int) for cluster in atypical_cluster_infos]
                )
                clustered_atypical_mask[clustered_indices] = True

            transitioning_typical_count = int(np.sum(is_transitioning & ~is_atypical))
            stable_typical_count = int(np.sum(~is_transitioning & ~is_atypical))
            clustered_atypical_count = int(np.sum(clustered_atypical_mask & is_atypical))
            unclustered_atypical_count = int(
                np.sum(is_atypical & ~clustered_atypical_mask)
            )
            print(
                "  > Tagged "
                f"{transitioning_typical_count} Transitioning Typical, "
                f"{stable_typical_count} Stable Typical, "
                f"{clustered_atypical_count} Clustered Atypical, and "
                f"{unclustered_atypical_count} Unclustered Atypical cells."
            )

        else:
            print("  WARNING: Prediction count mismatch with adata. Skipping tagging.")




        
        # =============================================================================
        # IMPORTANT: Internal vs Display Naming Convention
        # =============================================================================
        # - node_names: CL IDs (e.g., "CL:0000128") - USE FOR ALL INTERNAL LOGIC
        # - formatted_node_names: Display labels (e.g., "monocyte") - USE FOR DISPLAY ONLY
        #
        # Internal logic includes:
        #   - Color mapping (AdaptiveLineageColorMapper)
        #   - Index lookups (name_to_idx dictionaries)
        #   - Graph operations (NetworkX node identifiers)
        #   - URL generation (CellxGene links)
        #
        # Display logic includes:
        #   - Node labels on the tree
        #   - Hover text
        #   - Legend entries
        #   - Trajectory source/target labels
        # =============================================================================
        
        # Step 5: Compute colors (choose mapper based on configuration)
        
        # Step 6: Use cloud_label_counts as the single source of truth for node cell counts.
        # cloud_label_counts reflects stable-only cells (excluding transitioning and atypical)
        # which matches what's actually rendered in the cloud (transitioning cells are plotted
        # as separate dots on edges, not in the cloud).
        node_cell_counts = dict(cloud_label_counts)
        max_count = max(node_cell_counts.values()) if node_cell_counts else 1
        
        # Step 6.5: Compute edges with cells assigned (for inferred path filtering)
        # Only show inferred paths that have transitioning cells on them
        # Also track cell count per edge for minimum threshold
        edges_with_cells = set()
        edge_cell_counts = {}  # Track how many cells on each edge (unordered key)
        directional_cell_counts = {}  # Track per-direction cell counts: (parent, child) -> int
        edge_type_counts = {}  # Track canonical vs inferred path cells
        n_transitioning = np.sum(is_transitioning)
        
        # Build canonical edge set for classification
        canonical_edges_lookup = set()
        for (src, tgt) in canonical_tree_edges:
            canonical_edges_lookup.add((src, tgt))
            canonical_edges_lookup.add((tgt, src))
        
        # Populate edges_with_cells and edge_cell_counts by analyzing transitioning cells
        for i in range(len(is_transitioning)):
            if is_transitioning[i]:
                parent = parent_nodes[i]
                child = child_nodes[i]
                edge = (parent, child)
                edges_with_cells.add(edge)
                edges_with_cells.add((child, parent))  # Add reverse direction
                
                # Track cell count (unordered for threshold check)
                edge_key = (min(parent, child), max(parent, child))
                edge_cell_counts[edge_key] = edge_cell_counts.get(edge_key, 0) + 1
                
                # Track per-direction count for lateral inferred path intelligence
                directional_cell_counts[edge] = directional_cell_counts.get(edge, 0) + 1
                
                # Track edge type (canonical vs soft)
                is_canonical = edge in canonical_edges_lookup or (child, parent) in canonical_edges_lookup
                edge_type = 'canonical' if is_canonical else 'soft'
                if edge_key not in edge_type_counts:
                    edge_type_counts[edge_key] = {'canonical': 0, 'soft': 0}
                edge_type_counts[edge_key][edge_type] += 1
        
        # Minimum cell threshold for showing inferred paths (from config)
        min_cells_for_inferred_path = self.config.inferred_path_min_cells
        
        # Filter inferred_path_weights to only include edges with enough cells
        # For radial mode, deduplicate based on original DAG indices
        # Exclude candidate links that overlap canonical edges.
        filtered_inferred_path_weights = {}
        n_not_in_edges_with_cells = 0
        n_below_threshold = 0
        n_reciprocal_removed = 0
        n_dag_duplicates_removed = 0
        n_canonical_overlap_removed = 0
        
        # Build set to track which edges we've already added (to avoid reciprocals)
        added_edges = set()
        
        # Build canonical edge set for overlap checking
        # For radial mode, we need to check based on ORIGINAL DAG indices
        canonical_edges_set = set()
        canonical_dag_edges_set = set()  # For radial mode: canonical edges in DAG index space
        for (src, tgt) in canonical_tree_edges:
            canonical_edges_set.add((src, tgt))
            canonical_edges_set.add((tgt, src))
            # For radial mode: also track canonical edges in DAG index space
            if is_polar and _tree_to_original is not None:
                orig_src = _tree_to_original.get(src, src)
                orig_tgt = _tree_to_original.get(tgt, tgt)
                canonical_dag_edges_set.add((orig_src, orig_tgt))
                canonical_dag_edges_set.add((orig_tgt, orig_src))
        
        # For radial mode deduplication: track best tree edge per DAG edge
        # Key: (orig_src, orig_tgt) tuple of original DAG indices
        # Value: (tree_src, tree_tgt, weight, cell_count)
        dag_edge_best = {}
        
        # Pre-compute depths for filtering
        # Use the appropriate depth dictionary depending on layout mode
        if is_polar and _tree_to_original is not None:
            # Use DAG depths for direction checking in Radial mode logic
            check_depths = node_depths_dag
        else:
            # Use standard layout depths for Horizontal mode
            check_depths = layout_engine.node_depths if hasattr(layout_engine, 'node_depths') else node_depths_dag

        for (src, tgt), weight in inferred_path_weights.items():
            # Canonical edges take precedence over overlapping candidate links.
            if is_polar and _tree_to_original is not None:
                # Radial mode: check in DAG index space
                orig_src = _tree_to_original.get(src, src)
                orig_tgt = _tree_to_original.get(tgt, tgt)
                if (orig_src, orig_tgt) in canonical_dag_edges_set or (orig_tgt, orig_src) in canonical_dag_edges_set:
                    n_canonical_overlap_removed += 1
                    continue
            else:
                # Horizontal mode: check in tree index space
                if (src, tgt) in canonical_edges_set or (tgt, src) in canonical_edges_set:
                    n_canonical_overlap_removed += 1
                    continue
            
            # Check if cells flow along this edge (in either direction)
            # Note: edges_with_cells uses TREE LAYOUT direction (based on X/radius position)
            # Inferred paths use BIOLOGICAL direction (based on ontology depth)
            # These may differ, so we check both directions
            cells_same_dir = (src, tgt) in edges_with_cells
            cells_reverse_dir = (tgt, src) in edges_with_cells
            
            if cells_same_dir or cells_reverse_dir:
                # Check cell count threshold
                edge_key = (min(src, tgt), max(src, tgt))
                cell_count = edge_cell_counts.get(edge_key, 0)
                if cell_count >= min_cells_for_inferred_path:
                    # Get depths to determine if this is a lateral transition
                    # For radial mode, map tree indices back to DAG indices for depth lookup
                    if is_polar and _tree_to_original is not None:
                        d_src = check_depths.get(orig_src, 0)
                        d_tgt = check_depths.get(orig_tgt, 0)
                    else:
                        d_src = check_depths.get(src, 0)
                        d_tgt = check_depths.get(tgt, 0)
                    
                    is_lateral = (d_src == d_tgt)
                    
                    # DECISION MATRIX:
                    # 1. Lateral Transition (Same Depth): Use lateral_directions as
                    #    single source of truth for direction enforcement.
                    # 2. Vertical Transition (Diff Depth): DEDUPLICATE.
                    #    Enforce flow from Shallow -> Deep.
                    if is_lateral:
                        _canonical = lateral_directions.get((src, tgt)) or lateral_directions.get((tgt, src))
                        if is_polar and _tree_to_original is not None:
                            # For Radial: use lateral_directions as single source of truth
                            # for direction gating, then apply clone deduplication tie-breaking.
                            # Preserve the selected direction and weight.
                            _radial_canonical = lateral_directions.get((src, tgt)) or lateral_directions.get((tgt, src))
                            _radial_keep = (src, tgt) == _radial_canonical if _radial_canonical is not None else True
                            if _radial_keep:
                                dag_key = (orig_src, orig_tgt)
                                if dag_key not in dag_edge_best:
                                    dag_edge_best[dag_key] = (src, tgt, weight, cell_count)
                                else:
                                    existing_count = dag_edge_best[dag_key][3]
                                    if cell_count > existing_count:
                                        # Higher cell count wins outright
                                        n_dag_duplicates_removed += 1
                                        dag_edge_best[dag_key] = (src, tgt, weight, cell_count)
                                    elif cell_count == existing_count:
                                        # Tie: use embedding direction score as secondary tiebreaker.
                                        # A clone whose (src, tgt) matches the canonical embedding
                                        # direction scores 1; the reverse scores 0.
                                        _lat_canonical = lateral_directions.get((src, tgt)) or lateral_directions.get((tgt, src))
                                        _new_dir_score = 1 if (_lat_canonical is not None and (src, tgt) == _lat_canonical) else 0
                                        _ex_src, _ex_tgt = dag_edge_best[dag_key][0], dag_edge_best[dag_key][1]
                                        _ex_dir_score = 1 if (_lat_canonical is not None and (_ex_src, _ex_tgt) == _lat_canonical) else 0
                                        if _new_dir_score > _ex_dir_score:
                                            n_dag_duplicates_removed += 1
                                            dag_edge_best[dag_key] = (src, tgt, weight, cell_count)
                        else:
                            # For Horizontal: use lateral_directions as single source of truth
                            # Use the resolved lateral direction.
                            if _canonical is not None:
                                # Canonical direction found: keep only if (src, tgt) matches
                                if (src, tgt) == _canonical:
                                    filtered_inferred_path_weights[(src, tgt)] = weight
                            else:
                                # No lateral_directions entry (fallback): keep as-is
                                filtered_inferred_path_weights[(src, tgt)] = weight
                            # We do NOT add to 'added_edges' so reciprocal is evaluated independently
                    else:
                        # Vertical Transition: Standard Deduplication (One-way only)
                        if (tgt, src) not in added_edges and (src, tgt) not in added_edges:
                            if is_polar and _tree_to_original is not None:
                                dag_key = tuple(sorted((orig_src, orig_tgt)))  # Sort to force unique
                                if dag_key not in dag_edge_best or cell_count > dag_edge_best[dag_key][3]:
                                    if dag_key in dag_edge_best:
                                        n_dag_duplicates_removed += 1
                                    dag_edge_best[dag_key] = (src, tgt, weight, cell_count)
                            else:
                                filtered_inferred_path_weights[(src, tgt)] = weight
                                added_edges.add((src, tgt))
                        else:
                            n_reciprocal_removed += 1
                else:
                    n_below_threshold += 1
            else:
                n_not_in_edges_with_cells += 1
        
        # For radial mode: convert dag_edge_best to filtered_inferred_path_weights
        if is_polar and _tree_to_original is not None:
            for dag_key, (tree_src, tree_tgt, weight, cell_count) in dag_edge_best.items():
                filtered_inferred_path_weights[(tree_src, tree_tgt)] = weight
                added_edges.add((tree_src, tree_tgt))

        # === Gate 1: drop blob inferred paths (per-cell identity spread) ===
        # Guard on the SAME condition that defined full_is_inferred_path / kept_indices earlier
        # (the `len(viz_data['predictions']) == len(adata)` obs-tagging block), so those locals are
        # guaranteed to exist here. cell_vectors / predictions / parent_nodes / child_nodes /
        # is_transitioning are the kept (tree-masked) cells; full_is_inferred_path[kept_indices] is
        # row-aligned to them because keep_mask was folded with the tree cell-mask upstream.
        if filtered_inferred_path_weights and len(viz_data['predictions']) == len(adata):
            from .trajectory_support import prune_blob_inferred_paths
            _is_inf_kept = full_is_inferred_path[kept_indices]
            filtered_inferred_path_weights, _gate1_verdicts = prune_blob_inferred_paths(
                filtered_inferred_path_weights, cell_vectors, predictions, _is_inf_kept,
                is_transitioning, parent_nodes, child_nodes,
                blob_ratio=self.config.inferred_path_gate1_blob_ratio,
            )

            # Map tree-node ids -> CL ids so verdicts are stable and inspectable after the run.
            def _gate1_clid(_idx):
                if is_polar and _tree_to_original is not None:
                    _orig = _tree_to_original.get(_idx, _idx)
                    return viz_data['node_names'][_orig] if 0 <= _orig < len(viz_data['node_names']) else str(_idx)
                return node_names[_idx] if _idx < len(node_names) else str(_idx)

            self._last_inferred_path_gate1_verdicts = {
                (_gate1_clid(_u), _gate1_clid(_v)): _verd for (_u, _v), _verd in _gate1_verdicts.items()
            }
            # Also stash on adata (create_trajectory_analysis returns None + mutates adata in place) so
            # Flat string keys preserve these decisions through h5ad round-trips.
            adata.uns['hector_inferred_path_gate1'] = {
                f"{_s}->{_t}": _verd for (_s, _t), _verd in self._last_inferred_path_gate1_verdicts.items()
            }

        # Build set of all inferred paths (both directions) for filtering visual_tree_edges
        all_inferred_paths_set = set()
        for (src, tgt) in inferred_path_weights.keys():
            all_inferred_paths_set.add((src, tgt))
            all_inferred_paths_set.add((tgt, src))
        
        # Build set of canonical edges (both directions) for priority checking
        canonical_edges_set = set()
        for (src, tgt) in canonical_tree_edges:
            canonical_edges_set.add((src, tgt))
            canonical_edges_set.add((tgt, src))
        
        # Build set of filtered inferred paths (those with enough cells)
        filtered_inferred_paths_set = set()
        for (src, tgt) in filtered_inferred_path_weights.keys():
            filtered_inferred_paths_set.add((src, tgt))
            filtered_inferred_paths_set.add((tgt, src))
        
        # Filter visual_tree_edges to remove inferred paths without enough cells
        # Canonical edges are always retained.
        # Keep: canonical edges (always) + inferred paths with >= MIN_CELLS_FOR_INFERRED_PATH cells
        filtered_visual_tree_edges = []
        n_canonical_kept = 0
        n_soft_kept = 0
        kept_inferred_paths = []  # Track which inferred paths are kept for logging
        for (u, v) in visual_tree_edges:
            # Retain canonical edges before applying inferred-path gates.
            is_canonical = (u, v) in canonical_edges_set or (v, u) in canonical_edges_set
            
            if is_canonical:
                # Canonical edge - ALWAYS keep, regardless of inferred path status
                filtered_visual_tree_edges.append((u, v))
                n_canonical_kept += 1
            else:
                # Not canonical - check if it's a inferred path with enough cells
                is_soft = (u, v) in all_inferred_paths_set or (v, u) in all_inferred_paths_set
                if is_soft:
                    # Only keep inferred path if it exists in EXACT direction in filtered_inferred_path_weights
                    # This ensures correct differentiation direction (low depth -> high depth)
                    if (u, v) in filtered_inferred_path_weights:
                        filtered_visual_tree_edges.append((u, v))
                        n_soft_kept += 1
                        kept_inferred_paths.append((u, v))
                    elif (v, u) in filtered_inferred_path_weights:
                        # Normalize to the correct direction from filtered_inferred_path_weights
                        filtered_visual_tree_edges.append((v, u))
                        n_soft_kept += 1
                        kept_inferred_paths.append((v, u))
                else:
                    # Not canonical, not soft - this shouldn't happen, but keep it for safety
                    filtered_visual_tree_edges.append((u, v))
                    n_canonical_kept += 1
        
        n_soft_with_cells = len(filtered_inferred_path_weights)
        n_soft_total = len(inferred_path_weights)
        if n_soft_total > 0:
            if kept_inferred_paths:
                print(f"  > Final inferred trajectories ({n_soft_kept} trajectories):")
                for (u, v) in kept_inferred_paths:
                    u_name = formatted_node_names[u] if u < len(formatted_node_names) else f"Node_{u}"
                    v_name = formatted_node_names[v] if v < len(formatted_node_names) else f"Node_{v}"
                    cell_count = edge_cell_counts.get((min(u, v), max(u, v)), 0)
                    u_name_clean=u_name.replace("\n", " ").replace("<br>", " ")
                    v_name_clean=v_name.replace("\n", " ").replace("<br>", " ")
                    print(f"      {u_name_clean} -> {v_name_clean}: {cell_count} cells")
        
        # Reassign cells whose inferred path was filtered out.
        # Cells placed on inferred paths that were filtered out should NOT appear as transitioning
        # They should be re-positioned to their anchor node (treated as stable)
        #
        # Canonical edges are undirected for cell reassignment: add both (u,v) and (v,u).
        # Inferred path edges are directed: only the exact winning direction (u,v) is added.
        # Adding the reverse for inferred paths would allow cells on the losing direction
        # to remain visible without a surviving rendered path.
        filtered_edges_set = set()
        for (u, v) in filtered_visual_tree_edges:
            filtered_edges_set.add((u, v))
            # Only add the reverse for canonical edges (undirected); inferred paths keep exact direction.
            if (u, v) not in filtered_inferred_path_weights:
                filtered_edges_set.add((v, u))
        
        n_cells_reassigned = 0
        _resync_rows = []  # (local_i, anchor_node_idx) for the adata.obs resync below
        for i in range(len(predictions)):
            if is_transitioning[i]:
                # Check if this cell's edge is in the filtered set
                cell_edge = (parent_nodes[i], child_nodes[i])
                cell_edge_reverse = (child_nodes[i], parent_nodes[i])

                if cell_edge not in filtered_edges_set and cell_edge_reverse not in filtered_edges_set:
                    # This cell is on a filtered-out edge - re-assign to anchor node
                    anchor = anchor_nodes[i]
                    if anchor in node_positions:
                        cell_positions[i] = np.array(node_positions[anchor])
                        is_transitioning[i] = False
                        n_cells_reassigned += 1
                        _resync_rows.append((i, anchor))

        # === Resync adata.obs for cells reassigned off filtered-out inferred paths ===
        # The obs trajectory columns were written earlier (before the min-cells filter and Gate 1
        # ran), so without this a cell whose inferred edge was dropped (a Gate 1 'blob' or a
        # below-min-cells edge) would still read as Transitioning / is_inferred_path=True / a target
        # on the dropped edge — contradicting both the rendered figure and
        # adata.uns['hector_inferred_path_gate1']. These cells now render as Stable at their anchor;
        # make obs agree. Guarded by the same condition that defined the full_* arrays / kept_indices.
        if _resync_rows and len(viz_data['predictions']) == len(adata):
            for (i, anchor) in _resync_rows:
                kidx = kept_indices[i]
                full_is_inferred_path[kidx] = False
                full_target[kidx] = ''
                # Into the stable pool, matching the label below. The other
                # two positions are left alone — they still record where the
                # cell sat on its best edge, even though it is not drawn.
                full_transition_position[kidx] = 0.0
                full_source[kidx] = _format_obs_label(_node_to_clid(anchor))
                # Downgrade only a plain 'Transitioning' label to 'Stable'; keep Atypical_* labels.
                if full_status[kidx] == 'Transitioning':
                    full_status[kidx] = 'Stable'
            adata.obs[TRAJECTORY_STATE_COLUMN] = full_status
            adata.obs['trajectory_source'] = full_source
            adata.obs['trajectory_target'] = full_target
            adata.obs['transition_position'] = full_transition_position
            adata.obs['is_inferred_path'] = full_is_inferred_path
        
        # Summary output
        print(f"  Cells: {len(predictions)} | Nodes: {len(active_indices)} | Edges: {len(filtered_visual_tree_edges)} (inferred paths: {n_soft_with_cells})")
        print("Plot generating......")
        
        # Report predictions that do not have a node in the active layout.
        pred_unique = set(np.unique(predictions))
        active_set = set(active_indices)
        invalid_preds = pred_unique - active_set
        
        if invalid_preds:
            below_threshold = set(getattr(self, '_nodes_below_threshold', {}))
            logger.warning(
                "%d predicted node(s) are absent from the active layout "
                "(min_cells_number=%d, below_threshold=%d, sample=%s).",
                len(invalid_preds),
                self.config.min_cells_number,
                len(invalid_preds & below_threshold),
                sorted(invalid_preds)[:5],
            )

        # Import renderers lazily so trajectory_render can import shared helpers
        # from this module without creating a package import cycle.
        from .trajectory_render import (
            _build_matplotlib_figure,
            _build_matplotlib_radial_figure,
            _build_plotly_figure,
        )
        
        # Step 7: Build figure
        if _output_format == 'pdf':
            # Build matplotlib figure for PDF export
            if is_polar:
                return _build_matplotlib_radial_figure(self, 
                    cell_positions=cell_positions,
                    predictions=predictions,
                    is_transitioning=is_transitioning,
                    is_atypical=is_atypical,
                    anchor_nodes=anchor_nodes,
                    node_positions=node_positions,
                    active_indices=active_indices,
                    node_names=node_names,
                    formatted_node_names=formatted_node_names,
                    visual_tree_edges=filtered_visual_tree_edges,
                    canonical_tree_edges=canonical_tree_edges,  # Pass canonical edges for styling
                    layout_engine=layout_engine,  # Pass for edge color/width info
                    color_mapper=color_mapper,  # Pass for individual node colors
                    cell_colors=cell_colors,
                    subgraph=subgraph,
                    node_cell_counts=node_cell_counts,
                    max_count=max_count,
                    output_path=output_file,
                    atypical_cluster_infos=atypical_cluster_infos,
                    inferred_path_weights=filtered_inferred_path_weights,  # Pass filtered inferred paths
                    cloud_label_counts=cloud_label_counts,
                    cloud_transition_counts=cloud_transition_counts,
                    noise_mask=unclustered_atypical_mask,  # Pass unclustered atypical cells for rendering
                    cloud_extents=cloud_extents,
                    highlight_data=highlight_data,
                    highlight_color_mapper=highlight_color_mapper
                )
            else:
                return _build_matplotlib_figure(self, 
                    cell_positions=cell_positions,
                    predictions=predictions,
                    is_transitioning=is_transitioning,
                    is_atypical=is_atypical,
                    anchor_nodes=anchor_nodes,
                    node_positions=node_positions,
                    active_indices=active_indices,
                    node_names=node_names,
                    formatted_node_names=formatted_node_names,
                    visual_tree_edges=filtered_visual_tree_edges,
                    canonical_tree_edges=canonical_tree_edges,
                    color_mapper=color_mapper,
                    cell_colors=cell_colors,
                    subgraph=subgraph,
                    node_cell_counts=node_cell_counts,
                    max_count=max_count,
                    output_path=output_file,
                    atypical_cluster_infos=atypical_cluster_infos,
                    inferred_path_weights=filtered_inferred_path_weights,  # Pass filtered inferred paths
                    cloud_label_counts=cloud_label_counts,
                    cloud_transition_counts=cloud_transition_counts,
                    noise_mask=unclustered_atypical_mask,  # Pass unclustered atypical cells for rendering
                    cloud_extents=cloud_extents,  # Per-node cloud extent for ribbon terminus blending
                    highlight_data=highlight_data,
                    highlight_color_mapper=highlight_color_mapper
                )
        else:
            # Build Plotly figure for interactive HTML
            # Note: For now, radial mode uses same Plotly with polar data
            return _build_plotly_figure(self, 
                cell_positions=cell_positions,
                predictions=predictions,
                is_transitioning=is_transitioning,
                is_atypical=is_atypical,
                anchor_nodes=anchor_nodes,
                node_positions=node_positions,
                active_indices=active_indices,
                node_names=node_names,
                formatted_node_names=formatted_node_names,
                visual_tree_edges=filtered_visual_tree_edges,
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
                inferred_path_weights=filtered_inferred_path_weights,  # Pass filtered inferred paths (only those with cells)
                cloud_label_counts=cloud_label_counts,
                cloud_transition_counts=cloud_transition_counts,
                noise_mask=unclustered_atypical_mask,  # Pass unclustered atypical cells for rendering
                cloud_extents=cloud_extents,  # Per-node cloud extent for ribbon terminus blending
                highlight_data=highlight_data,
                highlight_color_mapper=highlight_color_mapper
            )
    


def create_trajectory_analysis(
    predictor,
    adata,
    output_file: str = "HECTOR_Trajectory_Analysis.html",
    use_grit: bool = True,
    grit_refinement_percentile: Optional[float] = None,
    label_format: str = 'id',
    force_recompute: bool = False,
    **kwargs
) -> None:
    """Create a trajectory analysis and write its visualization.

    The function reuses compatible cached prediction and atypical-cell results.
    Atypical-cell detection is disabled by default; setting
    ``enable_atypical_detection=True`` runs :meth:`HECTOR.evaluate_cells` when
    its cache is missing, incompatible, or explicitly bypassed.

    Args:
        predictor: HECTOR instance
        adata: AnnData object
        output_file: Path to save output. Format is inferred from file extension:
            - '.html' → interactive Plotly visualization
            - '.pdf' → static matplotlib figure
        use_grit: Use GRIT-refined predictions as placement anchors.
        grit_refinement_percentile: Controls which cells get GRIT refinement
            - None (default): Auto-detect using Multi-Otsu thresholding
              (finds natural boundary in entropy distribution)
            - 0-100: Manual override - refine top X% low-confidence prediction
              (e.g., 10 = refine cells with entropy in top 10%)
        label_format: Format for node labels (default: 'id')
            - 'id': Show only ontology IDs (e.g., 'CL:0000128')
            - 'name': Show only cell type names (e.g., 'monocyte')
            - 'both': Show both (e.g., 'monocyte (CL:0000128)')
            Note: Falls back to ID if name is not available in checkpoint
        force_recompute: If True, bypass all cached embeddings and predictions
            in the AnnData object and re-run the full pipeline (encoding,
            prediction, visualization). Default is False, which allows the
            pipeline to skip steps whose results are already cached.
        **kwargs: Additional :class:`TrajectoryAnalysisConfig` parameters,
            including:
            - transition_stream_width: float (default 0.5) - minimum lateral spread
              of transitioning cells on edges; branch width governs thick branches
            - y_scale: float (default 1.0) - controls vertical spacing in the layout
            - top_k_neighbors: int (default 15) - limits competition to top K nodes
            - stable_source_zone / stable_target_zone: float or None (default None) - width of the
              Stable bands at the source (position <= source) and target (position >= 1 - target) ends;
              cells between them are Transitioning. ``None`` auto-detects via a BGMM fit on
              ``relative_position`` (resolved values in ``adata.uns['hector_trajectory']['auto_zones']``).
              The two are independent (asymmetric bands allowed); numeric values must satisfy
              ``source + target < 1.0``.
            - stable_scatter_jitter_scale: float (default 0.25) - contracts stable scatter fallback points toward the node center when contours are not used. Also applied to highlight scatter per anchor node.
            - min_cells_number: int (default 20) - minimum absolute cell count required per node
            - atypical_anchor_count: int (default 3) - number of anchor nodes for atypical-cluster hub positioning
            
            - enable_atypical_detection: bool (default False) - If True, HECTOR.evaluate_cells
              populates adata.obs['is_atypical'] and adata.obs['adherence_score']; the
              visualization consumes those directly. Off by default: it is a
              separate, expensive pipeline, so trajectory rendering does not run
              it unless asked.
            - atypical_clusters: bool (default False) - If True and detection enabled, cluster atypical
              cells via atypical-cell clustering and render each cluster as a pie chart.
            - show_pie_charts: bool (default True) - show pie charts instead of dot clouds
            - pie_chart_edge_color: str (default 'white') - edge color between pie slices
            - pie_chart_edge_width: float (default 0.5) - edge width between pie slices
            
            - save_diagnostic_plot: bool (default True) - save the routed evaluate_cells
              diagnostic grid as a PNG alongside the main output (filename:
              ``<output_file_stem>_diagnostic_plot.png``).
    
    Side Effects:
        Writes the requested HTML or PDF and annotates ``adata.obs`` with
        ``trajectory_state``, edge endpoints, placement coordinates, and the
        prediction/trajectory conflict flag. When atypical detection or
        clustering is enabled, it also stores the corresponding adherence,
        atypical, cluster, and anchor annotations. Cells omitted by minimum-count
        pruning receive the ``'Pruned'`` trajectory state.

    Returns:
        None.
    """
    args_copy = kwargs.copy()
    if 'add_cellxgene_links' in args_copy: del args_copy['add_cellxgene_links']
    # output_format is no longer a config field — strip it from kwargs for backward compat
    args_copy.pop('output_format', None)
    
    # Validate label_format
    valid_formats = ['id', 'name', 'both']
    if label_format not in valid_formats:
        raise ValueError(f"Invalid label_format '{label_format}'. Must be one of: {valid_formats}")
    
    config = TrajectoryAnalysisConfig(**args_copy)
    analyzer = HierarchicalTrajectoryAnalyzer(config)
    
    # === Materialize views before any workflow writes ===
    adata_was_view = adata.is_view
    if adata_was_view:
        adata_materialized = adata.copy()
    else:
        adata_materialized = adata

    if config.enable_atypical_detection and predictor is not None:
        # Compute diagnostic output path for the routed evaluate_cells
        # diagnostic grid.
        _diagnostic_path = None
        if config.save_diagnostic_plot:
            _base = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
            _diagnostic_path = f"{_base}_diagnostic_plot.png"

        # evaluate_cells() handles its own caching via the atypical manifest
        # fingerprint (adata.uns['hector_atypical_manifest']); callers need
        # only pass force_recompute through and evaluate_cells will decide
        # whether to reuse cached results or recompute.
        predictor.evaluate_cells(
            adata_materialized,
            min_cells_number=config.min_cells_number,
            top_k_neighbors=config.top_k_neighbors,
            temperature=config.temperature,
            class_relative_adherence=config.class_relative_adherence,
            class_relative_blend_weight=config.class_relative_blend_weight,
            force_recompute=force_recompute,
            diagnostic_output_path=_diagnostic_path,
        )

    if predictor is not None and hasattr(predictor, '_ensure_public_annotation'):
        predictor._ensure_public_annotation(
            adata_materialized,
            use_grit=use_grit,
            grit_refinement_percentile=grit_refinement_percentile,
            force_recompute=force_recompute,
            prefix='hector_',
            label_format=label_format,
        )

    fig = analyzer.visualize(
        predictor,
        adata_materialized,
        output_file,
        use_grit=use_grit,
        grit_refinement_percentile=grit_refinement_percentile,
        label_format=label_format,
        force_recompute=force_recompute
    )

    # Stash warm cache for downstream simplify_ontology_tree() calls
    adata_materialized._hector_viz_cache = analyzer._viz_cache

    # === Persist workflow outputs back to caller's adata for views ===
    if adata_was_view:
        adata.obs = adata.obs.copy()

        # obsm: X_hector
        if 'X_hector' in adata_materialized.obsm:
            adata.obsm['X_hector'] = adata_materialized.obsm['X_hector']

        if 'hector_cell_variance' in adata_materialized.obs.columns:
            adata.obs['hector_cell_variance'] = adata_materialized.obs[
                'hector_cell_variance'
            ].values

        if predictor is not None and hasattr(predictor, '_get_embedding_manifest'):
            embedding_manifest = predictor._get_embedding_manifest(adata_materialized)
            if embedding_manifest is not None:
                adata.uns = dict(adata.uns)
                adata.uns['hector_embedding_manifest'] = embedding_manifest

        if predictor is not None and hasattr(predictor, '_get_prediction_core_manifest'):
            prediction_core_manifest = predictor._get_prediction_core_manifest(
                adata_materialized
            )
            if prediction_core_manifest is not None:
                adata.uns = dict(adata.uns)
                adata.uns['hector_prediction_core_manifest'] = (
                    prediction_core_manifest
                )

        if predictor is not None and hasattr(predictor, '_resolve_annotation_reference'):
            annotation_ref = predictor._resolve_annotation_reference(
                adata_materialized,
                require_score=False,
                allow_fallback=False,
            )
            if annotation_ref is not None:
                for col_name in (
                    annotation_ref.get('prediction_col'),
                    annotation_ref.get('score_col'),
                    annotation_ref.get('cell_variance_col'),
                ):
                    if isinstance(col_name, str) and col_name in adata_materialized.obs.columns:
                        adata.obs[col_name] = adata_materialized.obs[col_name].values
                manifest = predictor._get_annotation_manifest(adata_materialized)
                if manifest is not None:
                    adata.uns = dict(adata.uns)
                    adata.uns['hector_annotation_manifest'] = manifest
        if predictor is not None and hasattr(predictor, '_get_atypical_manifest'):
            atypical_manifest = predictor._get_atypical_manifest(adata_materialized)
            if atypical_manifest is not None:
                adata.uns = dict(adata.uns)
                adata.uns['hector_atypical_manifest'] = atypical_manifest

        if 'is_low_quality' in adata_materialized.obs.columns:
            adata.obs['is_low_quality'] = adata_materialized.obs['is_low_quality'].values

        # Propagate trajectory columns written during visualization.
        trajectory_columns = [
            TRAJECTORY_STATE_COLUMN,
            'trajectory_source',
            'trajectory_target',
            'absolute_position',
            'relative_position',
            'transition_position',
            'is_inferred_path',
            'prediction_trajectory_conflict',
            'adherence_score',
            'entropy',
            'is_atypical',
            'predictive_divergence',
            'is_divergent',
            'divergence_probability',
            'divergence_component',
            'atypical_cluster_id',
            'atypical_anchor_nodes',
            'atypical_anchor_weights',
        ]
        for col in trajectory_columns:
            if col in adata_materialized.obs.columns:
                adata.obs[col] = adata_materialized.obs[col].values
    
    return None


def create_trajectory_comparison(
    predictor,
    adata,
    groupby: str,
    groups: Optional[List[str]] = None,
    output_prefix: str = "comparison",
    use_grit: bool = True,
    grit_refinement_percentile: Optional[float] = None,
    label_format: str = 'id',
    force_recompute: bool = False,
    **kwargs,
) -> None:
    """Create trajectory visualizations comparing groups within a single dataset.

    Generates one output file per group. All plots share the same layout
    scaffold so that cell types appear at the same position in every plot.
    Cell types present in some groups but absent from others leave a visible
    gap, making differences easy to spot.

    Supports both radial and horizontal layouts (controlled via
    ``layout_mode`` in kwargs, default 'radial').

    The function works by:
    1. Running predictions once on the full dataset.
    2. Building the ontology scaffold from the *union* of all per-group cell types.
    3. Computing a shared layout (radial tree or horizontal DAG).
    4. Rendering each group's plot by masking the full-dataset arrays.

    Args:
        predictor: HECTOR instance.
        adata: AnnData object containing all cells.
        groupby: Column name in ``adata.obs`` to split by. Each unique value
            (or each value in ``groups``) becomes one output plot.
        groups: Subset of values from the ``groupby`` column to compare.
            ``None`` (default) uses all unique values, sorted.
        output_prefix: File prefix for the PDF outputs. Output files are named
            ``{output_prefix}_{group}.pdf``. If the prefix ends in ``.html``
            or ``.htm``, a warning is emitted and the suffix is replaced with
            ``.pdf``.
        use_grit: Enable GRIT refinement (default True).
        grit_refinement_percentile: GRIT percentile override (default auto).
        label_format: Node label format — 'id', 'name', or 'both'.
        force_recompute: Bypass all caches and recompute from scratch.
        **kwargs: Additional config parameters forwarded to
            ``TrajectoryAnalysisConfig`` (same as ``create_trajectory_analysis``).

    Returns:
        None. PDF files are written to disk. Prediction annotations may be
        updated in ``adata.obs``; per-group trajectory placement columns are
        not written.
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if groupby not in adata.obs.columns:
        raise KeyError(
            f"Column '{groupby}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    valid_formats = ['id', 'name', 'both']
    if label_format not in valid_formats:
        raise ValueError(f"Invalid label_format '{label_format}'. Must be one of: {valid_formats}")

    col_values = adata.obs[groupby]
    if groups is None:
        group_labels = sorted(col_values.unique(), key=str)
    else:
        missing = [g for g in groups if g not in col_values.values]
        if missing:
            raise ValueError(
                f"Groups not found in adata.obs['{groupby}']: {missing}"
            )
        group_labels = list(groups)

    if len(group_labels) == 0:
        raise ValueError(f"No groups found in adata.obs['{groupby}']")

    # Comparison plots currently use the shared static Matplotlib scaffold.
    _ext = os.path.splitext(output_prefix)[1].lower()
    if _ext == '.pdf':
        output_ext = '.pdf'
        output_stem = os.path.splitext(output_prefix)[0]
    elif _ext in ('.html', '.htm'):
        warnings.warn(
            "create_trajectory_comparison supports PDF output only; "
            f"replacing {_ext} with .pdf.",
            RuntimeWarning,
            stacklevel=2,
        )
        output_ext = '.pdf'
        output_stem = os.path.splitext(output_prefix)[0]
    elif _ext:
        raise ValueError(
            "create_trajectory_comparison supports PDF output only; "
            f"got output_prefix={output_prefix!r}."
        )
    else:
        output_ext = '.pdf'
        output_stem = output_prefix

    args_copy = kwargs.copy()
    args_copy.pop('add_cellxgene_links', None)
    args_copy.pop('output_format', None)
    config = TrajectoryAnalysisConfig(**args_copy)
    is_polar = (config.layout_mode == 'radial')

    print("\nHECTOR Trajectory Comparison")
    print(f"Groups ({groupby}): {group_labels}")
    print("="*40)

    # ------------------------------------------------------------------
    # Phase A: Predict once on full dataset
    # ------------------------------------------------------------------
    print("\n[Full dataset] Running predictions...")

    if config.enable_atypical_detection and predictor is not None:
        _diagnostic_path = None
        if config.save_diagnostic_plot:
            _diagnostic_path = f"{output_stem}_diagnostic_plot.png"
        predictor.evaluate_cells(
            adata,
            min_cells_number=config.min_cells_number,
            top_k_neighbors=config.top_k_neighbors,
            temperature=config.temperature,
            class_relative_adherence=config.class_relative_adherence,
            class_relative_blend_weight=config.class_relative_blend_weight,
            force_recompute=force_recompute,
            diagnostic_output_path=_diagnostic_path,
        )

    if predictor is not None and hasattr(predictor, '_ensure_public_annotation'):
        predictor._ensure_public_annotation(
            adata,
            use_grit=use_grit,
            grit_refinement_percentile=grit_refinement_percentile,
            force_recompute=force_recompute,
            prefix='hector_',
            label_format=label_format,
        )

    viz_data = predictor.get_visualization_vectors(
        adata,
        use_grit=use_grit,
        grit_refinement_percentile=grit_refinement_percentile,
        force_recompute=force_recompute,
    )

    all_predictions = viz_data['predictions']
    all_cell_vectors = viz_data['cell_vectors']

    pure_adj = viz_data.get('pure_adjacency_matrix')
    if pure_adj is None:
        raise ValueError(
            "Model checkpoint does not contain 'pure_adjacency_matrix'. "
            "Please provide ontology_path='cl.obo' or re-train with updated code."
        )
    shared_adjacency = pure_adj
    shared_node_names = viz_data['node_names']

    if "is_atypical" in adata.obs.columns:
        all_is_atypical = np.asarray(adata.obs["is_atypical"].values, dtype=bool)
        all_adherence = np.asarray(adata.obs["adherence_score"].values, dtype=np.float64)
    else:
        all_is_atypical = np.zeros(adata.shape[0], dtype=bool)
        all_adherence = np.ones(adata.shape[0], dtype=np.float64)

    group_masks: Dict[str, np.ndarray] = {}
    for label in group_labels:
        group_masks[label] = np.asarray(col_values == label, dtype=bool)

    # ------------------------------------------------------------------
    # Phase B: Union active indices (per-group, then union)
    # ------------------------------------------------------------------
    print("\n[Union] Building shared ontology subgraph...")

    subgraph = AdaptiveOntologySubgraph(
        adjacency_matrix=shared_adjacency,
        node_names=shared_node_names,
        co_graph=None,
    )
    canonical_order = CanonicalOrderComputer(subgraph.nx_graph, subgraph.node_names)

    union_active: Set[int] = set()
    for label in group_labels:
        group_pred = all_predictions[group_masks[label]]
        ctx = _build_canonical_trajectory_context(
            adjacency_matrix=shared_adjacency,
            node_names=shared_node_names,
            predicted_indices=group_pred,
            min_cells_number=config.min_cells_number,
        )
        union_active.update(ctx['active_indices'])

    union_active_indices = sorted(union_active)
    print(f"  Union cell types: {len(union_active_indices)}")

    # ------------------------------------------------------------------
    # Phase C: Build shared scaffold (union tree/DAG + layout)
    # ------------------------------------------------------------------
    print("[Union] Building shared scaffold...")

    from collections import defaultdict
    from sklearn.metrics.pairwise import cosine_similarity
    from .trajectory_render import _build_matplotlib_radial_figure, _build_matplotlib_figure

    node_vectors = viz_data.get('node_vectors')
    id_to_name_map = getattr(predictor, 'id_to_name_map', {})
    line_sep = '\n'
    formatted_original_names = format_node_names(
        shared_node_names, id_to_name_map, label_format, line_sep,
    )

    if is_polar:
        # --- Radial: DAG → Tree conversion for clean angular layout ---
        # The converter clones every multi-parent cell type, one clone per
        # root-to-node path, but each cell can only be assigned to one clone.
        # Prune the clones that end up under-populated across the WHOLE dataset:
        # they would otherwise be allocated an angular sector in the shared
        # layout and never drawn, leaving blank wedges between clades. Pruning
        # on the pooled counts is safe — a clone that clears the threshold in
        # any single group necessarily clears it across all groups too.
        converter = DAGToTreeConverter(shared_adjacency, shared_node_names, node_vectors)
        union_tree_data = converter.convert(
            union_active_indices, all_predictions, all_cell_vectors,
            min_cells_number=config.min_cells_number,
        )

        scaffold_adj = union_tree_data['tree_adj']
        scaffold_names = union_tree_data['tree_names']
        scaffold_active = union_tree_data['active_indices']
        union_original_indices = union_tree_data['original_indices']

        orig_to_new_map: Dict[int, List[int]] = defaultdict(list)
        for tree_idx, dag_idx in enumerate(union_original_indices):
            orig_to_new_map[dag_idx].append(tree_idx)

        n_scaffold_nodes = len(scaffold_names)
        tree_node_parent = [-1] * n_scaffold_nodes
        rows, cols = np.where(scaffold_adj > 0)
        for child_idx, parent_idx in zip(rows, cols):
            tree_node_parent[child_idx] = parent_idx

        scaffold_vectors = np.array([
            node_vectors[union_original_indices[i]] for i in range(n_scaffold_nodes)
        ])
        parent_vectors = np.zeros_like(scaffold_vectors)
        for i in range(n_scaffold_nodes):
            p = tree_node_parent[i]
            parent_vectors[i] = scaffold_vectors[p] if p != -1 else scaffold_vectors[i]

        scaffold_edges: List[Tuple[int, int]] = []
        for child_idx, parent_idx in zip(*np.where(scaffold_adj > 0)):
            scaffold_edges.append((int(parent_idx), int(child_idx)))

        color_mapper = _create_projected_lineage_color_mapper(
            pure_adjacency=shared_adjacency,
            original_node_names=shared_node_names,
            active_source_indices=[int(p) for p in all_predictions],
            visible_node_names=scaffold_names,
            visible_edges=scaffold_edges,
            color_palette=config.adaptive_color_palette,
            visible_to_source={
                i: int(union_original_indices[i])
                for i in range(len(union_original_indices))
            },
            legend_formatted_names=formatted_original_names,
        )

        layout_engine = RadialTreeLayoutEngine(scaffold_adj, scaffold_names, config)
        layout_engine.canonical_order = canonical_order
        layout_engine.original_indices = list(union_original_indices)
        layout_engine.set_embeddings(node_vectors)
        layout_engine.set_color_mapper(color_mapper)

        union_node_positions = layout_engine.compute_layout(
            scaffold_active, augmented_adjacency=scaffold_adj,
        )
        layout_engine.compute_edge_curvature_scales(union_node_positions)
        union_canonical_edges = layout_engine.tree_edges

    else:
        # --- Horizontal: operate directly on the DAG (no tree conversion) ---
        scaffold_adj = shared_adjacency
        scaffold_names = shared_node_names
        scaffold_active = union_active_indices
        scaffold_vectors = node_vectors

        layout_engine = TreeLayoutEngine(scaffold_adj, scaffold_names, config)
        layout_engine.canonical_order = canonical_order
        layout_engine.original_indices = None
        layout_engine.set_embeddings(node_vectors)

        union_node_positions = layout_engine.compute_layout(
            scaffold_active, augmented_adjacency=scaffold_adj,
        )
        union_canonical_edges = layout_engine.tree_edges

        color_mapper = _create_projected_lineage_color_mapper(
            pure_adjacency=shared_adjacency,
            original_node_names=shared_node_names,
            active_source_indices=[int(p) for p in all_predictions],
            visible_node_names=scaffold_names,
            visible_edges=union_canonical_edges,
            visible_node_indices=scaffold_active,
            color_palette=config.adaptive_color_palette,
            legend_formatted_names=formatted_original_names,
        )

    formatted_union_names = format_node_names(
        scaffold_names, id_to_name_map, label_format, line_sep,
    )

    print(f"  Union {'tree' if is_polar else 'DAG'}: {len(scaffold_names)} nodes, "
          f"{len(union_canonical_edges)} edges")

    # ------------------------------------------------------------------
    # Phase D: Per-group rendering on shared scaffold
    # ------------------------------------------------------------------
    for label in group_labels:
        output_file = f"{output_stem}_{label}{output_ext}"
        print(f"\n[{label}] Rendering → {output_file}")

        mask = group_masks[label]
        predictions_dag = all_predictions[mask]
        cell_vectors_sample = all_cell_vectors[mask]

        # D1: keep_mask (same SCC pruning as visualize)
        sample_ctx = _build_canonical_trajectory_context(
            adjacency_matrix=shared_adjacency,
            node_names=shared_node_names,
            predicted_indices=predictions_dag,
            min_cells_number=config.min_cells_number,
        )
        keep_mask = np.asarray(sample_ctx['keep_mask'], dtype=bool)
        predictions_dag = predictions_dag[keep_mask]
        cell_vectors_sample = cell_vectors_sample[keep_mask]

        canonical_is_atypical = all_is_atypical[mask][keep_mask]
        canonical_adherence = all_adherence[mask][keep_mask]

        # D2: Remap cell predictions to scaffold space
        if is_polar:
            # -1 marks "no counterpart in the scaffold". Starting from zeros
            # would silently pile those cells onto tree node 0, the root.
            tree_predictions = np.full(len(predictions_dag), -1, dtype=int)
            for orig_pred in np.unique(predictions_dag):
                if orig_pred not in orig_to_new_map:
                    continue
                possible_targets = orig_to_new_map[orig_pred]
                pred_mask = (predictions_dag == orig_pred)
                if len(possible_targets) == 1:
                    tree_predictions[pred_mask] = possible_targets[0]
                else:
                    cells = cell_vectors_sample[pred_mask]
                    candidate_parent_vecs = parent_vectors[possible_targets]
                    sims = cosine_similarity(cells, candidate_parent_vecs)
                    best_local = np.argmax(sims, axis=1)
                    best_global = np.array(possible_targets)[best_local]
                    tree_predictions[pred_mask] = best_global

            # Drop cells whose cell type has no scaffold node (its clones were
            # all pruned as under-populated). Every cell-parallel array must
            # stay in sync.
            mapped_mask = tree_predictions >= 0
            if not mapped_mask.all():
                tree_predictions = tree_predictions[mapped_mask]
                predictions_dag = predictions_dag[mapped_mask]
                cell_vectors_sample = cell_vectors_sample[mapped_mask]
                canonical_is_atypical = canonical_is_atypical[mapped_mask]
                canonical_adherence = canonical_adherence[mapped_mask]
        else:
            tree_predictions = predictions_dag.copy()

        # D3: Visibility mask (min_cells_number threshold + ancestor propagation)
        node_counts: Dict[int, int] = {}
        for p in tree_predictions:
            node_counts[p] = node_counts.get(p, 0) + 1

        nodes_with_cells: Set[int] = {
            n for n, c in node_counts.items() if c >= config.min_cells_number
        }
        visible_nodes: Set[int] = set(nodes_with_cells)

        if is_polar:
            def _mark_ancestors(node: int, visible: Set[int]):
                p = tree_node_parent[node]
                if p != -1 and p not in visible:
                    visible.add(p)
                    _mark_ancestors(p, visible)
        else:
            _scaffold_active_set = set(scaffold_active)
            def _mark_ancestors(node: int, visible: Set[int]):
                parents = np.where(scaffold_adj[node] > 0)[0]
                for p in parents:
                    pi = int(p)
                    if pi not in visible and pi in _scaffold_active_set:
                        visible.add(pi)
                        _mark_ancestors(pi, visible)

        for n in list(nodes_with_cells):
            _mark_ancestors(n, visible_nodes)

        visible_active = sorted(visible_nodes)
        visible_edges = [
            (u, v) for u, v in union_canonical_edges
            if u in visible_nodes and v in visible_nodes
        ]

        cell_visible_mask = np.array([
            p in nodes_with_cells for p in tree_predictions
        ])
        tree_predictions = tree_predictions[cell_visible_mask]
        cell_vectors_sample = cell_vectors_sample[cell_visible_mask]
        canonical_is_atypical = canonical_is_atypical[cell_visible_mask]
        canonical_adherence = canonical_adherence[cell_visible_mask]

        print(f"  Visible: {len(visible_active)}/{len(scaffold_active)} nodes, "
              f"{len(visible_edges)}/{len(union_canonical_edges)} edges, "
              f"{len(tree_predictions)} cells")

        # D4: Cell colors from the shared color mapper
        cell_colors = color_mapper.get_colors(tree_predictions)

        # D5: Cell placement
        node_cell_counts_sample: Dict[int, int] = {}
        for p in tree_predictions:
            node_cell_counts_sample[p] = node_cell_counts_sample.get(p, 0) + 1

        _placer_pixel_geometry = _build_placer_pixel_geometry(
            config, union_node_positions, visible_active, is_polar, 'pdf',
        )

        # Convert the engine's per-node branch widths into the placer's pixel
        # units, which is what scales transitioning-cell jitter to the local
        # branch width. Same rule as HierarchicalTrajectoryAnalyzer.visualize:
        #   PDF radial:     1.5 x pt_to_px — LineCollection.linewidths are in
        #                   points and the radial renderer applies its own 1.5x
        #                   scale (_PDF_RADIAL_WIDTH_SCALE).
        #   PDF horizontal: compute_adaptive_pdf_ribbon_scale() — that renderer
        #                   sizes its polygon ribbons against node spacing.
        # Mixing the two (adaptive scale x pt_to_px) overstates radial branch
        # widths ~5x, fanning cells well outside the branch they belong to.
        if hasattr(layout_engine, 'node_widths') and layout_engine.node_widths:
            if is_polar:
                pt_to_px = config.pdf_dpi / 72.0
                _w_scale = 1.5 * pt_to_px
            else:
                _node_spacing_px = config.y_scale * _placer_pixel_geometry.px_per_data_y()
                _w_scale = compute_adaptive_pdf_ribbon_scale(_node_spacing_px)
            _placer_node_widths = {
                n: float(w) * _w_scale
                for n, w in layout_engine.node_widths.items()
            }
        else:
            _placer_node_widths = None

        cell_placer = LatentBarycentricPlacer(
            node_positions=union_node_positions,
            node_embeddings=scaffold_vectors,
            tree_edges=visible_edges,
            temperature=config.temperature,
            transition_stream_width=config.transition_stream_width,
            top_k_neighbors=config.top_k_neighbors,
            stable_source_zone=config.stable_source_zone,
            stable_target_zone=config.stable_target_zone,
            node_cell_counts=node_cell_counts_sample,
            scale_cloud_by_count=config.scale_cloud_by_count,
            cloud_scale_exponent=config.cloud_scale_exponent,
            cloud_size_multiplier=config.cloud_size_multiplier,
            cloud_size_min=config.cloud_size_min,
            cloud_size_max=config.cloud_size_max,
            y_scale=config.y_scale,
            is_polar=is_polar,
            is_pdf_output=True,
            inferred_path_weights={},
            lateral_directions={},
            pixel_geometry=_placer_pixel_geometry,
            node_widths=_placer_node_widths,
            edge_curvature_scales=getattr(
                layout_engine, '_edge_curvature_scales', None,
            ),
        )

        (cell_positions, is_transitioning, _parent_nodes, _child_nodes,
         _transition_scores, _raw_transition_scores, anchor_nodes,
         _adherence_scores, is_atypical,
         _absolute_transition_scores) = cell_placer.place_cells(
            cell_vectors_sample,
            external_atypical_mask=canonical_is_atypical,
            external_adherence_scores=canonical_adherence,
            prior_anchors=tree_predictions,
            show_transitions=config.show_transitions,
        )

        # D6: Render
        cloud_label_counts = compute_cloud_cell_counts(
            active_indices=visible_active,
            anchor_nodes=anchor_nodes,
            is_transitioning=is_transitioning,
            is_atypical=is_atypical,
            enable_atypical_detection=config.enable_atypical_detection,
        )
        if config.show_transitions:
            cloud_transition_counts = compute_cloud_transition_counts(
                active_indices=visible_active,
                anchor_nodes=anchor_nodes,
                is_transitioning=is_transitioning,
            )
        else:
            cloud_transition_counts = None

        node_cell_counts_display = dict(cloud_label_counts)
        max_count = max(node_cell_counts_display.values()) if node_cell_counts_display else 1

        saved_tree_edges = layout_engine.tree_edges
        layout_engine.tree_edges = visible_edges

        analyzer = HierarchicalTrajectoryAnalyzer(config)
        if is_polar:
            _build_matplotlib_radial_figure(
                analyzer,
                cell_positions=cell_positions,
                predictions=tree_predictions,
                is_transitioning=is_transitioning,
                is_atypical=is_atypical,
                anchor_nodes=anchor_nodes,
                node_positions=union_node_positions,
                active_indices=visible_active,
                node_names=scaffold_names,
                formatted_node_names=formatted_union_names,
                visual_tree_edges=visible_edges,
                canonical_tree_edges=visible_edges,
                layout_engine=layout_engine,
                color_mapper=color_mapper,
                cell_colors=cell_colors,
                subgraph=subgraph,
                node_cell_counts=node_cell_counts_display,
                max_count=max_count,
                output_path=output_file,
                atypical_cluster_infos=[],
                inferred_path_weights={},
                cloud_label_counts=cloud_label_counts,
                cloud_transition_counts=cloud_transition_counts,
                noise_mask=np.zeros(len(tree_predictions), dtype=bool),
                cloud_extents={},
                highlight_data=None,
                highlight_color_mapper=None,
            )
        else:
            _build_matplotlib_figure(
                analyzer,
                cell_positions=cell_positions,
                predictions=tree_predictions,
                is_transitioning=is_transitioning,
                is_atypical=is_atypical,
                anchor_nodes=anchor_nodes,
                node_positions=union_node_positions,
                active_indices=visible_active,
                node_names=scaffold_names,
                formatted_node_names=formatted_union_names,
                visual_tree_edges=visible_edges,
                canonical_tree_edges=visible_edges,
                color_mapper=color_mapper,
                cell_colors=cell_colors,
                subgraph=subgraph,
                node_cell_counts=node_cell_counts_display,
                max_count=max_count,
                output_path=output_file,
                atypical_cluster_infos=[],
                inferred_path_weights={},
                cloud_label_counts=cloud_label_counts,
                cloud_transition_counts=cloud_transition_counts,
                noise_mask=np.zeros(len(tree_predictions), dtype=bool),
                cloud_extents={},
                highlight_data=None,
                highlight_color_mapper=None,
            )

        layout_engine.tree_edges = saved_tree_edges

    print(f"\n[Done] {len(group_labels)} comparison plots written.")
    return None


# =============================================================================
# Public API: create_pseudotime_analysis()
# =============================================================================
#
# A pseudotime is normally one number per cell, the distance from a root along a
# graph, and that number cannot say which branch a cell is on: two cells equally
# far from the root get the same value however different they are. What is built
# here never puts one number on every cell at once. It works out which lineage
# each cell belongs to first, then ranks a cell only against the cells of that
# same lineage, so two cells on different branches are never given a value that
# invites comparing them.
#
# HECTOR already puts each cell on an edge between two ontology terms and records
# how far along that edge it sits, but the position is local -- 0.5 on one edge
# and 0.5 on another are not the same thing. Everything below exists to make
# those local positions comparable, by putting the terms themselves in order
# first.


def _load_developmental_order():
    """The ontology's statements about which of two cell types comes later.

    A model checkpoint carries the is-a hierarchy and nothing else. is-a says what
    kind of cell something is, which is not the same as which of two cells comes
    later, and in blood the two disagree outright: the ontology files basophilic,
    polychromatophilic and orthochromatic erythroblast as three siblings under
    erythroblast, so is-a states no order between the three stages a maturing red
    cell passes through. The ontology does state it, through relations the
    checkpoint does not carry, and this table is those relations distilled.

    The table is keyed by ontology identifier so term-name changes do not affect
    lookup.

    Returns:
        Dict mapping a later term's identifier to the set of identifiers the
        ontology places before it.
    """
    from importlib import resources

    path = resources.files("hector").joinpath(
        "resources", "developmental_order", "cl_developmental_order.tsv"
    )
    earlier_of: Dict[str, Set[str]] = {}
    with resources.as_file(path) as p:
        with open(p, encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split("\t")
            earlier_at, later_at = header.index("earlier"), header.index("later")
            for line in handle:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                earlier_of.setdefault(parts[later_at], set()).add(parts[earlier_at])
    return earlier_of


def _ancestor_lookup(adjacency: np.ndarray, n_nodes: int):
    """A function giving every is-a ancestor of a node index, memoised.

    ``adjacency`` is the pure is-a matrix, ``adj[child, parent] = 1``. Built on
    ``_build_child_to_parent_ontology_graph``, whose ``successors()`` are parents.
    """
    from .trajectory_ontology import _build_child_to_parent_ontology_graph

    graph = _build_child_to_parent_ontology_graph(
        adjacency_matrix=adjacency, n_nodes=n_nodes
    )
    cache: Dict[int, Set[int]] = {}

    def ancestors(node: int) -> Set[int]:
        if node not in cache:
            seen, stack = set(), [node]
            while stack:
                current = stack.pop()
                for parent in graph.successors(current):
                    if parent not in seen:
                        seen.add(parent)
                        stack.append(parent)
            cache[node] = seen
        return cache[node]

    return ancestors


def _build_order_constraints(occupied, ancestors_of, earlier_ids, id_of):
    """The ontology's statements about order, as edges from the earlier term to the later.

    Two relations, each used only for the pairs it speaks about. is-a contributes
    wherever one occupied term is an ancestor of another, the descendant being the
    later one. The developmental relations contribute what is-a cannot -- they are
    the only ones that chain basophilic to polychromatophilic to orthochromatic
    erythroblast, which is-a files as siblings.

    This is not a count of is-a steps from the root. A relation compares only the
    pairs it describes and leaves unrelated pairs unordered.

    Args:
        occupied: Term names the cells actually occupy.
        ancestors_of: Callable giving a term name's is-a ancestor names.
        earlier_ids: Mapping from a term identifier to the identifiers placed
            before it, from :func:`_load_developmental_order`.
        id_of: Mapping from a term name to its ontology identifier.

    Returns:
        A ``networkx.DiGraph`` over the occupied terms, each edge carrying a
        ``rel`` attribute of ``'is-a'`` or ``'develops'``.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(occupied)
    occupied = set(occupied)
    by_id = {id_of[name]: name for name in occupied if name in id_of}
    for later in occupied:
        for earlier in ancestors_of(later) & occupied:
            graph.add_edge(earlier, later, rel="is-a")
        for earlier_id in earlier_ids.get(id_of.get(later, ""), ()):  # noqa: B038
            earlier = by_id.get(earlier_id)
            if earlier is not None and earlier != later and not graph.has_edge(earlier, later):
                graph.add_edge(earlier, later, rel="develops")
    return graph


def _walk_terms(graph, score):
    """Put the terms in an order that breaks none of the ontology's statements.

    Terms come out one at a time. A term becomes available once everything the
    ontology places before it has already come out, and among those available the
    one with the lowest score goes first. So the ontology is never overridden and
    the score decides only the pairs the ontology left open.

    Because a term is emitted only after its predecessors, the resulting order
    satisfies every statement in the acyclic constraint graph.

    Args:
        graph: The constraint graph from :func:`_build_order_constraints`.
        score: Mapping from term name to the number breaking ties, lowest first.

    Returns:
        The term names, in order.

    Raises:
        ValueError: If the statements contain a cycle, so no order satisfies them.
    """
    import heapq

    indegree = {node: graph.in_degree(node) for node in graph}
    ready = [(score.get(node, 0.0), node) for node in graph if indegree[node] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        _, node = heapq.heappop(ready)
        order.append(node)
        for nxt in graph.successors(node):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(ready, (score.get(nxt, 0.0), nxt))
    if len(order) != graph.number_of_nodes():
        stuck = sorted(set(graph.nodes) - set(order))
        raise ValueError(
            "The ontology's order statements contain a cycle, so no ordering of the "
            f"cell types satisfies them. {len(stuck)} terms could not be placed, "
            f"starting with: {stuck[:5]}. This indicates a contradictory ontology "
            "release; report it with the model name and these term names."
        )
    return order


def _compose_positions(source, target, position, place):
    """One value per cell: the source term's place, plus the way to the target's.

        value = place(source) + position x (place(target) - place(source))

    A cell sitting at its destination reads that term's place whichever edge
    carried it there, which is what makes cells on different edges comparable.

    The two numbers being combined are of different kinds and knowingly so:
    ``position`` is a fraction of a distance the model measured, while the
    difference between two places is a gap between two positions in the walk. What
    comes out interleaves cells from different edges into one order. It is not a
    distance along anything, a single cell's value is not a quantity, and the gap
    between two cells' values is not an amount.

    Args:
        source: Per-cell source term names.
        target: Per-cell target term names.
        position: Per-cell position along the edge, 0 at the source and 1 at the
            target; ``NaN`` where the cell has no edge.
        place: Mapping from term name to its place in the walk, 0 to 1.

    Returns:
        A float array, ``NaN`` where the cell has no edge or an unplaced term.
    """
    start = np.array([place.get(s, np.nan) for s in source], dtype=float)
    end = np.array([place.get(t, np.nan) for t in target], dtype=float)
    return start + np.asarray(position, dtype=float) * (end - start)


def _rank_within_lineage(value, lineage):
    """The percentile rank of each cell among the cells of its own lineage.

    Nothing is ever compared between lineages: two cells of the same value in
    different lineages are each at the same place in their own order, not at the
    same moment. One number covering every cell in the dataset cannot express
    that difference -- it reads the two as equal -- which is why the ranking is
    done inside a lineage and never across lineages.

    Args:
        value: The composed value per cell.
        lineage: The lineage per cell; ``None`` where the cell has none.

    Returns:
        A float array in ``[0, 1]``, ``NaN`` where the cell has no lineage or no
        value.
    """
    import pandas as pd

    frame = pd.DataFrame({"value": np.asarray(value, dtype=float), "lineage": list(lineage)})
    ranked = frame.groupby("lineage", observed=True)["value"].rank(pct=True)
    return ranked.where(frame["value"].notna()).to_numpy()


def _most_specific_common_term(terms, ancestors_of):
    """The most specific ontology term sitting at or above every one of ``terms``.

    Used to name a lineage the user merged. The name is always a term the ontology
    supplies -- never one written by hand, and never the first term the user
    happened to type, which would let typing order decide a name.

    A broad answer is not a failure. It says the members share nothing narrower,
    which is a finding about the group rather than a fault in the naming.

    Returns:
        ``(name, tied)``, where ``tied`` lists the other equally specific
        candidates when the ontology offers no single answer. The name is chosen
        deterministically among ties so a run is reproducible; ``tied`` is
        non-empty so the caller can say so.
    """
    terms = list(terms)
    if not terms:
        return None, []
    shared = None
    for term in terms:
        above = set(ancestors_of(term)) | {term}
        shared = above if shared is None else (shared & above)
    if not shared:
        return None, []
    deepest = [
        candidate
        for candidate in shared
        if not any(candidate in ancestors_of(other) for other in shared if other != candidate)
    ]
    if not deepest:
        deepest = sorted(shared)
    deepest = sorted(deepest)
    return deepest[0], deepest[1:]


def _apply_lineage_list(entries, of_lineage, ancestors_of, descendants_of):
    """Merge or split the lineages the automatic rule found, as the user asked.

    The list patches only what it names: a lineage the list does not mention keeps
    its terms and its name, so a short list cannot quietly strand most of the data.

    One rule covers both operations. A name resolves to a set of occupied terms --
    the lineage it leads if it leads one, otherwise every occupied term at or below
    it in the ontology. An entry is the union of its names, joined by ``+``, and
    becomes one lineage; the terms it claims leave whatever lineage held them. So
    naming a term deeper than a lineage's own splits that term out, and joining two
    names merges them, with no separate code for either.

    Args:
        entries: The user's list, each entry one or more term names joined by ``+``.
        of_lineage: Term name to lineage name, from the automatic rule. Not modified.
        ancestors_of: Callable giving a term name's is-a ancestor names.
        descendants_of: Callable giving the occupied terms at or below a term name.

    Returns:
        ``(of_lineage, moved, notes)`` -- the new mapping, the number of terms the
        list moved, and a line for each entry whose lineage came back under a name
        other than the one written, which is the only outcome a reader cannot see
        in the lineage listing itself. Entries that got the name they asked for
        produce no line.

    Raises:
        ValueError: If an entry names nothing the cells occupy, or if two entries
            claim the same term, which only the list's author can resolve.
    """
    leaders = {name for name in of_lineage.values() if name}
    members: Dict[str, List[str]] = {}
    for leader in leaders:
        members[leader] = [t for t, lin in of_lineage.items() if lin == leader]

    claimed: Dict[str, str] = {}
    resolved: List[Tuple[str, List[str]]] = []
    for entry in entries:
        names = [part.strip() for part in str(entry).split("+") if part.strip()]
        if not names:
            raise ValueError(f"Empty entry in the lineage list: {entry!r}")
        terms: Set[str] = set()
        for name in names:
            if name in members:
                terms.update(members[name])
            else:
                below = descendants_of(name)
                if not below:
                    raise ValueError(
                        f"'{name}' in the lineage list is neither a lineage the run "
                        "found nor a term with anything below it in this dataset. "
                        "Run without a list first and read the printed lineages; an "
                        "entry may name any ontology term at or above the cells you "
                        "want grouped."
                    )
                terms.update(below)
        for term in terms:
            if term in claimed and claimed[term] != entry:
                raise ValueError(
                    f"'{term}' is claimed by two entries in the lineage list, "
                    f"{claimed[term]!r} and {entry!r}. Only the list's author can say "
                    "which should hold it; narrow one of the two."
                )
            claimed[term] = entry
        resolved.append((entry, sorted(terms)))

    of_lineage = dict(of_lineage)
    notes, moved = [], 0
    for entry, terms in resolved:
        name, tied = _most_specific_common_term(terms, ancestors_of)
        if name is None:
            raise ValueError(
                f"The terms named by {entry!r} have no common term above them in the "
                "ontology, so the lineage cannot be named. Split the entry."
            )
        changed = sum(1 for term in terms if of_lineage.get(term) != name)
        moved += changed
        for term in terms:
            of_lineage[term] = name
        # Only a name the user would not recognise is worth saying. When the entry
        # is already the ontology's answer, the lineage listing carries the same
        # counts under the name the user typed, and repeating them says nothing.
        if name != entry:
            note = (
                f"{entry!r} was named '{name}' -- the most specific term above "
                "every cell type it holds."
            )
            if tied:
                note += f" (Tied with {', '.join(tied)}; '{name}' taken.)"
            notes.append(note)
        elif tied:
            notes.append(
                f"{entry!r} tied with {', '.join(tied)} as the most specific term "
                f"above every cell type it holds; '{name}' taken."
            )
    return of_lineage, moved, notes


def _canonical_names(labels, predictor, index):
    """Turn labels in any of the three label formats into plain cell type names.

    ``create_trajectory_analysis`` writes ``trajectory_source`` and
    ``trajectory_target`` through the caller's ``label_format``, which defaults to
    ``'id'`` -- so the columns may hold ``'CL:0000128'``, ``'monocyte'`` or
    ``'monocyte (CL:0000128)'`` depending on how it was called. Everything here
    works in names, so all three are accepted and reduced to one.

    Args:
        labels: The raw label strings.
        predictor: HECTOR instance, for ``_parse_ontology_id`` and the name map.
        index: Mapping from a known cell type name to its row in the vocabulary.

    Returns:
        A string array of names, empty where the label names nothing the model knows
        (including the empty label a cell gets when it never reached the best-edge
        analysis).
    """
    resolved = {}
    for label in set(map(str, labels)):
        if not label or label in ("nan", "None"):
            resolved[label] = ""
            continue
        if label in index:
            resolved[label] = label
            continue
        ontology_id = predictor._parse_ontology_id(label)
        name = predictor.id_to_name_map.get(ontology_id, "") if ontology_id else ""
        resolved[label] = name if name in index else ""
    return np.array([resolved[str(label)] for label in labels], dtype=object)


def _publish_term_table(adata, table) -> None:
    """Leave the term-to-lineage table on the object.

    A column-wise dict of plain lists, not a list of records: ``write_h5ad`` cannot
    serialise a list of dicts, so the table is stored as a mapping of columns to
    plain lists.
    """
    adata.uns["hector_pseudotime_lineages"] = {
        column: table[column].tolist() for column in table.columns
    }


def _detected_gene_score(adata, embedding, source, target, n_neighbors):
    """One number per term: how many genes its cells express, smoothed and negated.

    Nothing else is taken from the matrix -- not which genes, not how much of each.
    A progenitor holds a broad, uncommitted transcriptome and a maturing cell shuts
    genes off, which is the observation the original CytoTRACE was built on. One
    cell's count is noisy from dropout, so it is averaged over that cell's nearest
    neighbours in HECTOR's own embedding: the space the value lives in is where two
    cells being close should mean their values are comparable.

    Whether the matrix holds counts or log-normalised values makes no difference,
    since only whether an entry is nonzero is read.

    This number decides only the pairs the ontology leaves open; the ontology
    remains the ordering framework.

    It does read sequencing depth as well as maturity. Where a cell genuinely holds
    less RNA at the end of its lineage the confound and the signal are the same
    thing; elsewhere it is exposure.

    Returns:
        ``(score, smoothed)`` -- a dict from term name to its score, lowest coming
        first in the walk, and the per-cell smoothed count the scores were built
        from. The per-cell array is returned rather than discarded because
        ``create_gene_trend_analysis`` has to hold this exact quantity fixed: it is
        the ruler this function breaks ties with, and a gene that merely tracks it
        would otherwise look like biology. Recomputing something close to it would
        not do -- the smoothing is part of what the ordering used.
    """
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors

    matrix = getattr(adata, "X", None)
    if matrix is None:
        raise RuntimeError(
            "adata.X is empty, so the number of genes each cell expresses cannot be "
            "counted. That count is what breaks the ties the ontology leaves open."
        )
    # Strictly greater than zero, never the count of stored entries: a sparse matrix
    # may hold explicit zeros, and a scaled one holds negatives, both of which would
    # be counted as expression.
    detected = np.asarray((matrix > 0).sum(axis=1)).ravel().astype(float)

    k = int(min(max(n_neighbors, 1), len(detected)))
    finder = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    _, near = finder.fit(embedding).kneighbors(embedding)
    smoothed = detected[near].mean(axis=1)

    # A term means the same thing wherever it appears, so every cell sent to it
    # counts towards its score whatever lineage the cell ends up in; pooling both
    # ends of the edge steadies the mean.
    at_term = pd.concat(
        [
            pd.DataFrame({"term": np.asarray(source), "value": smoothed}),
            pd.DataFrame({"term": np.asarray(target), "value": smoothed}),
        ]
    )
    return (-at_term.groupby("term")["value"].mean()).to_dict(), smoothed


def _find_lineages(predictor, adata, occupied, embedding, ancestors_of, min_cells_number):
    """Work out the lineages from the ontology, with no biology typed in.

    The ontology uses one level for two different kinds of term. Some name what a
    cell will become; others say only how far along it is, and both sit as children
    of the same parent. So any single horizontal cut takes both kinds and the
    how-far-along ones swallow the early stages of every fate. Four steps instead,
    none of them a threshold:

    1. Split the ontology into a tree. ``DAGToTreeConverter`` duplicates every term
       with several parents, one copy per branch, and puts each cell on the copy
       whose parent its embedding sits closest to -- so the cells decide which
       branch a shared term belongs on.
    2. Find the fates on that tree. The projected lineage mapper marks a node only
       when every visible descendant belongs to one family, and deliberately leaves
       a node with descendants in several families unmarked. That is the
       how-far-along test done structurally: a shared progenitor cannot be claimed
       by one fate.
    3. Give each term the most specific fate above it **in the original ontology,
       not the tree**. The tree is the right thing for finding the fates and the
       wrong thing for membership, because splitting a term puts one copy on a
       branch and starves the other.
    4. Where two fates tie, neither sitting above the other, take the tree's
       placement -- the cells already chose.

    A term with no fate above it is not given one. What such a term does get is
    its own name, so that it
    stands as a group of its own that the ``lineages`` argument can name and fold
    into a real lineage. It is marked as having no fate, and that mark, not the
    name, is what keeps its cells out of the ranking and out of the colours.

    Returns:
        ``(of_lineage, basis, has_fate)`` -- term name to lineage name, term name
        to a short phrase saying how it was decided, and term name to whether a
        fate was found for it. The last is what code should read; the phrase is
        for a person.
    """
    from .trajectory_ontology import (
        DAGToTreeConverter,
        _create_projected_lineage_color_mapper,
    )

    names = list(predictor.full_class_names)
    index = {name: i for i, name in enumerate(names)}
    adjacency = np.asarray(predictor.full_pure_ontology_adj)
    vectors = np.asarray(
        predictor._build_visualization_state(use_full_ontology=True)["node_vectors"]
    )

    # The manifest names the prediction column when the object still carries one;
    # a plain `hector_prediction` is accepted so an object rebuilt from saved
    # columns still works. Only the prediction is needed, not its score.
    reference = predictor._resolve_annotation_reference(adata, require_score=False)
    prediction_col = (reference or {}).get("prediction_col")
    if not prediction_col or prediction_col not in adata.obs.columns:
        prediction_col = "hector_prediction"
    if prediction_col not in adata.obs.columns:
        raise RuntimeError(
            "No cell type prediction column on this object. "
            "create_trajectory_analysis writes 'hector_prediction'; run it first."
        )
    # The prediction carries whatever label_format the caller used, exactly as the
    # trajectory columns do, so it is reduced to names the same way.
    predicted = _canonical_names(adata.obs[prediction_col].to_numpy(), predictor, index)
    on_panel = np.array([bool(p) for p in predicted], dtype=bool)
    predictions = np.array([index[p] for p in predicted[on_panel]], dtype=int)
    if not on_panel.any():
        raise RuntimeError(
            f"No value in adata.obs['{prediction_col}'] is a cell type this model "
            "knows, in any label format, so the ontology cannot be split into a tree."
        )

    active = sorted({index[t] for t in occupied} | set(np.unique(predictions).tolist()))
    tree = DAGToTreeConverter(
        adjacency_matrix=adjacency, node_names=names, node_vectors=vectors
    ).convert(
        active_indices=active,
        cell_predictions=predictions,
        cell_vectors=np.asarray(embedding)[on_panel],
        min_cells_number=min_cells_number,
    )

    tree_names = list(tree["tree_names"])
    tree_adj = np.asarray(tree["tree_adj"])
    original = tree.get("original_indices")
    edges = [(int(p), int(c)) for c, p in zip(*np.where(tree_adj > 0))]
    mapper = _create_projected_lineage_color_mapper(
        pure_adjacency=adjacency,
        original_node_names=names,
        active_source_indices=active,
        visible_node_names=tree_names,
        visible_edges=edges,
        visible_node_indices=list(range(len(tree_names))),
        visible_to_source={i: int(original[i]) for i in range(len(tree_names))},
    )

    keys = mapper.node_lineage_keys
    placed: Dict[str, Dict[str, int]] = {}
    for i, name in enumerate(tree_names):
        key = keys.get(i)
        if key:
            placed.setdefault(name, {})
            placed[name][key] = placed[name].get(key, 0) + 1
    fates = [f for f in sorted({k for k in keys.values() if k}) if f in index]

    # has_fate is set in the same breath as basis everywhere below, so the mark a
    # reader sees and the mark the code acts on cannot come apart.
    of_lineage, basis, has_fate = {}, {}, {}
    for term in occupied:
        above = ancestors_of(term)
        over = [f for f in fates if f == term or f in above]
        if not over:
            of_lineage[term], basis[term] = term, "no fate above it"
            has_fate[term] = False
            continue
        if len(over) == 1:
            of_lineage[term], basis[term] = over[0], "the only fate above it"
            has_fate[term] = True
            continue
        deepest = [f for f in over if not any(f in ancestors_of(o) for o in over if o != f)]
        if len(deepest) == 1:
            of_lineage[term], basis[term] = deepest[0], "the most specific fate above it"
            has_fate[term] = True
            continue
        votes = placed.get(term, {})
        chosen = None
        for fate, _ in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0])):
            if fate in deepest:
                chosen = fate
                break
        if chosen is not None:
            of_lineage[term], basis[term] = chosen, "tied; settled by the split tree"
            has_fate[term] = True
        else:
            # Neither candidate sits above the other and the cells did not settle
            # it, so the ontology has not said which fate this is. Its own name,
            # like the no-fate case, and the same mark.
            of_lineage[term] = term
            basis[term] = f"tied between {' and '.join(sorted(deepest))}"
            has_fate[term] = False
    return of_lineage, basis, has_fate


def create_pseudotime_analysis(
    predictor,
    adata,
    lineages: Optional[List[str]] = None,
    min_cells_number: Optional[int] = None,
    n_neighbors: int = 30,
    force_recompute: bool = False,
) -> None:
    """Order the cells within each lineage, using the ontology to order the cell types.

    ``create_trajectory_analysis`` puts each cell on an edge between two ontology
    terms and records how far along that edge it sits, but that position is local:
    0.5 on one edge and 0.5 on another are not the same thing. This function makes
    those positions comparable by putting the terms themselves in order first, then
    reading each cell's position between its own two terms.

    The order comes from the ontology wherever the ontology speaks -- an is-a
    ancestor comes before its descendant, and the developmental relations state the
    rest -- and from a count of detected genes for the pairs it says nothing about.
    The lineages are worked out from the ontology too, so nothing about the tissue
    is typed in. Cells are ranked only against the cells of their own lineage.

    **The result is an order, not an amount.** A value of 0.5 means half the cells
    of that lineage come before this one. It does not mean half way along the
    lineage, the gap between two cells' values is not an amount of anything, and a
    single cell's value is not a quantity to quote. Values from different lineages
    are never comparable.

    Args:
        predictor: HECTOR instance, the same one used for the trajectory analysis.
        adata: AnnData already processed by ``create_trajectory_analysis``.
        lineages: Optional list of entries merging or splitting the lineages the
            run works out for itself. Each entry is one or more ontology term
            names joined by ``' + '``; the entry becomes one lineage, named by the
            most specific ontology term above all of its members. A name resolves
            to the lineage it leads if it leads one, otherwise to every occupied
            term at or below it -- so naming a deeper term splits it out, and
            joining two names merges them. **The list patches only what it names**:
            lineages it does not mention keep their terms and their names. Run
            without it first and read the printed lineages. This is also how a
            cell type with no fate above it is put to use: name it beside a
            lineage, as in ``'myeloid cell + monocyte'``, and its cells join that
            lineage and are ranked with it. Naming one on its own does nothing,
            since it already stands as a group of its own.
        min_cells_number: Minimum cells for an ontology node to survive the tree
            split. Left at ``None`` it is read back from what
            ``create_trajectory_analysis`` recorded, which is what makes the two
            steps agree; a value passed here overrides that. A different value
            gives a different tree, so override only deliberately.
        n_neighbors: Cells averaged over when smoothing the detected-gene count.
        force_recompute: Recompute HECTOR's cell embedding even if one is cached.

    Returns:
        None. Writes onto ``adata``:

        - ``adata.obs['hector_lineage']`` -- the group a cell belongs to. Usually a
          lineage. Where the ontology found no fate above the cell's own cell type,
          it is that cell type's own name instead, so those cells are a group that
          can be named and joined rather than a blank. ``NaN`` means one thing only:
          the cell was never placed on an edge, because its cell type held fewer
          than ``min_cells_number`` cells and was pruned before the trajectory was
          built.
        - ``adata.obs['hector_pseudotime']`` -- the cell's rank within its own
          lineage, 0 to 1. ``NaN`` where the cell has no lineage, and also where its
          group is a cell type with no fate above it: such a cell has not committed
          to a fate, so there is nothing for it to be far along in.
        - ``adata.obs['hector_detected_genes']`` -- how many genes each cell
          expresses, averaged over that cell's nearest neighbours in HECTOR's own
          embedding. This is the ruler the run breaks ties with wherever the
          ontology states no order, and it is written down so that
          ``create_gene_trend_analysis`` can hold the same quantity fixed. A gene
          that merely tracks it would otherwise look like a gene that changes
          along the lineage.
        - ``adata.uns['hector_pseudotime_lineages']`` -- one record per occupied
          cell type: its lineage, whether a fate was found for it, how that was
          decided, and its place in the order. ``has_fate`` is what code should
          read; the phrase in ``basis`` is for a person.

    Raises:
        RuntimeError: If the trajectory analysis has not been run.
        ValueError: If the ontology's statements contain a cycle, or the lineage
            list is ambiguous.
    """
    import pandas as pd

    needed = ["trajectory_source", "trajectory_target", "absolute_position"]
    missing = [c for c in needed if c not in adata.obs.columns]
    if missing:
        raise RuntimeError(
            f"create_pseudotime_analysis requires {missing}, written by "
            "create_trajectory_analysis. Run create_trajectory_analysis(predictor, "
            "adata, ...) first, then call this with the same predictor."
        )

    # Shares X and deep-copies only the annotations, the same way
    # compute_cell_embeddings makes a view writable.
    predictor._ensure_cache_writable(adata)

    # The tree split has to use the threshold the trajectory run used, or it
    # yields a different tree and so different lineages. Read back rather than
    # retyped, because a value retyped wrongly fails silently.
    # Only the two cases worth a word are said out loud. The ordinary one --
    # reading the trajectory run's own value back -- is silent.
    recorded_threshold = adata.uns.get("hector_trajectory", {}).get("min_cells_number")
    if min_cells_number is None:
        min_cells_number = 20 if recorded_threshold is None else int(recorded_threshold)
        if recorded_threshold is None:
            print(
                f"\n[Pseudotime] Minimum cells for a cell type to survive the tree "
                f"split: {min_cells_number}, the default -- the trajectory run "
                "recorded none. Rerun create_trajectory_analysis, or pass the value "
                "it used."
            )
    else:
        min_cells_number = int(min_cells_number)
        if recorded_threshold is not None and int(recorded_threshold) != min_cells_number:
            print(
                f"\n[Pseudotime] Minimum cells for a cell type to survive the tree "
                f"split: {min_cells_number}, as passed -- overriding the "
                f"{int(recorded_threshold)} the trajectory run used, which gives a "
                "different tree."
            )

    names = list(predictor.full_class_names)
    index = {name: i for i, name in enumerate(names)}
    identifiers = list(predictor.celltype_gat.full_classes)
    id_of = {name: str(identifiers[i]) for name, i in index.items() if i < len(identifiers)}

    source = _canonical_names(adata.obs["trajectory_source"].to_numpy(), predictor, index)
    target = _canonical_names(adata.obs["trajectory_target"].to_numpy(), predictor, index)
    position = pd.to_numeric(adata.obs["absolute_position"], errors="coerce").to_numpy()

    occupied = sorted({t for t in set(source) | set(target) if t})
    if not occupied:
        raise RuntimeError(
            "No cell type in adata.obs['trajectory_source'] or ['trajectory_target'] "
            "is in this model's vocabulary, in any label format. The predictor and "
            "the AnnData appear to come from different models."
        )

    ancestor_indices = _ancestor_lookup(np.asarray(predictor.full_pure_ontology_adj), len(names))

    def ancestors_of(term: str) -> Set[str]:
        return {names[i] for i in ancestor_indices(index[term])} if term in index else set()

    graph = _build_order_constraints(occupied, ancestors_of, _load_developmental_order(), id_of)
    print(f"\n[Pseudotime] {len(occupied)} cell types occupied.")

    if "X_hector" not in adata.obsm or force_recompute:
        predictor.compute_cell_embeddings(adata, force_recompute=force_recompute)
    embedding = np.asarray(adata.obsm["X_hector"], dtype=np.float64)

    score, detected_per_cell = _detected_gene_score(
        adata, embedding, source, target, n_neighbors
    )
    unusable = sorted(t for t in occupied if not np.isfinite(score.get(t, 0.0)))
    if unusable:
        raise RuntimeError(
            f"The detected-gene count is not finite for {len(unusable)} cell types, "
            f"starting with {unusable[:3]}. The tie-break cannot order them. Check "
            "adata.X and adata.obsm['X_hector'] for missing values."
        )
    order = _walk_terms(graph, score)
    place = {term: i / (len(order) - 1) for i, term in enumerate(order)} if len(order) > 1 \
        else {order[0]: 0.0}

    value = _compose_positions(source, target, position, place)

    of_lineage, basis, has_fate = _find_lineages(
        predictor, adata, occupied, embedding, ancestors_of, min_cells_number
    )
    notes = []
    if lineages:
        occupied_set = set(occupied)
        # A user may write a name, an identifier, or "name (identifier)", the same
        # three forms the trajectory columns can hold.
        entries = [
            " + ".join(
                _canonical_names([part.strip()], predictor, index)[0] or part.strip()
                for part in str(entry).split("+")
                if part.strip()
            )
            for entry in lineages
        ]

        def descendants_of(name: str) -> Set[str]:
            return {t for t in occupied_set if t == name or name in ancestors_of(t)}

        before = dict(of_lineage)
        of_lineage, _, notes = _apply_lineage_list(
            entries, of_lineage, ancestors_of, descendants_of
        )
        for term, lineage in of_lineage.items():
            if before.get(term) != lineage:
                basis[term] = "moved by the lineage list"
                has_fate[term] = True

    # A lineage is a fate if any of its terms is. That is what makes joining a
    # shared term work: 'myeloid cell + monocyte' is named after the shared term,
    # so its own row never changes and only monocyte's terms move -- and those
    # moved terms are what say the lineage has been spoken for.
    fated = {of_lineage[t] for t in occupied if has_fate.get(t)}

    per_cell = np.array([of_lineage.get(t) if t else None for t in target], dtype=object)
    # Ranked only inside a lineage that is a fate. A shared term holds its own
    # cells so they can be named and folded into one, but until they are there is
    # nothing for them to be far along in.
    ranked = np.array([lin if lin in fated else None for lin in per_cell], dtype=object)
    # Assigned as raw arrays, not index-aligned Series: obs_names are duplicated in
    # plenty of concatenated atlases and an aligned assignment would fail there.
    adata.obs["hector_lineage"] = pd.Categorical(
        [x if x else None for x in per_cell],
        categories=sorted({x for x in per_cell if x}),
    )
    adata.obs["hector_pseudotime"] = _rank_within_lineage(value, ranked)
    # The ruler this run broke ties with, kept so that a later stage can hold the
    # same quantity fixed rather than a recomputed approximation of it.
    adata.obs["hector_detected_genes"] = np.asarray(detected_per_cell, dtype=float)

    table = pd.DataFrame(
        {
            "term": occupied,
            "lineage": [of_lineage.get(t) or "" for t in occupied],
            "has_fate": [bool(has_fate.get(t)) for t in occupied],
            "basis": [basis.get(t, "") for t in occupied],
            "place": [place.get(t, np.nan) for t in occupied],
        }
    ).sort_values(["lineage", "place"])
    _publish_term_table(adata, table)

    # Three groups, and reporting any two of them as one would say something
    # untrue. A lineage is a fate the ontology found. A shared cell type is the
    # ontology declining to commit a cell that has not committed. A pruned cell
    # never reached the best-edge analysis at all, and its number moves with
    # min_cells_number rather than with anything about lineages.
    counts = pd.Series(per_cell).value_counts(dropna=True)
    fates = [(name, n) for name, n in counts.items() if name in fated]
    shared = [(name, n) for name, n in counts.items() if name not in fated]

    print(f"\n  {len(fates)} lineages, each ranked 0-1 on its own:")
    for name, n_cells in fates:
        n_terms = sum(1 for t in occupied if of_lineage.get(t) == name)
        print(f"    {str(name)[:52]:54s} {n_terms:>3} cell types  {n_cells:>8,} cells")

    if shared:
        import textwrap

        def say(text: str, indent: str = "  ") -> None:
            print(textwrap.fill(text, width=78, initial_indent=indent,
                                subsequent_indent=indent))

        def portion(group) -> str:
            held = sum(c for _, c in group)
            return f"{held:,} cells ({100 * held / len(per_cell):.1f}%)"

        def listing(group, keep: int = 10) -> str:
            """The names, capped -- and saying so, since a silent cut reads as all."""
            shown = ", ".join(f"{str(n)[:52]} ({c:,} cells)" for n, c in group[:keep])
            if len(group) > keep:
                shown += (f", and {len(group) - keep} more (see "
                          "adata.uns['hector_pseudotime_lineages'])")
            return shown

        # Two different reasons a cell type is left out, and one sentence cannot
        # state both without saying something false of one of them: either none of
        # the lineages sits above it, or several do and nothing chose between them.
        outside = [(n, c) for n, c in shared
                   if not str(basis.get(n, "")).startswith("tied")]
        undecided = [(n, c) for n, c in shared
                     if str(basis.get(n, "")).startswith("tied")]

        if outside:
            one = len(outside) == 1
            head = (f"{str(outside[0][0])[:52]}: {portion(outside)}" if one
                    else f"{len(outside)} cell types, {portion(outside)}")
            print()
            say(f"{head}, no pseudotime. "
                + ("It sits" if one else "Each sits")
                + f" under none of the {len(fates)} lineage names listed above, so "
                "there is no lineage to rank "
                + ("it" if one else "them")
                + " within; hector_lineage keeps "
                + ("its own name." if one else "their own names."))
            if not one:
                say(listing(outside), indent="    ")

        if undecided:
            # Which lineages tied differs term by term, so it goes in the table
            # rather than into a paragraph repeated once per cell type.
            one = len(undecided) == 1
            head = (f"{str(undecided[0][0])[:52]}: {portion(undecided)}" if one
                    else f"{len(undecided)} cell types, {portion(undecided)}")
            print()
            say(f"{head}, no pseudotime. Two or more of the {len(fates)} lineages sit "
                "above "
                + ("it" if one else "each")
                + ", none inside another, and "
                + ("its" if one else "their")
                + " cells did not settle which "
                + ("it belongs" if one else "they belong")
                + " to; which lineages tied is in "
                "adata.uns['hector_pseudotime_lineages']['basis'].")
            if not one:
                say(listing(undecided), indent="    ")

    never_placed = int(sum(1 for t in target if not t))
    if never_placed:
        print(
            f"\n  {never_placed:,} cells ({100 * never_placed / len(per_cell):.1f}%) "
            f"were never placed on an edge: their cell type\n  held fewer than "
            f"{min_cells_number} cells and was pruned before the trajectory was built."
        )
    if notes:
        import textwrap

        print()
        for note in notes:
            print(textwrap.fill(note, width=78, initial_indent="  ",
                                subsequent_indent="  "))
    print(
        "\n  adata.obs['hector_pseudotime'] is a rank within a lineage, not a "
        "distance.\n  Values from different lineages are not comparable."
    )
    return None


# =============================================================================
# Public API: plot_pseudotime()
# =============================================================================
#
# A rank that is only defined within a lineage cannot be drawn the way a single
# number covering every cell is. Two cells of the same value in different
# lineages are each at the same place in their own order, not at the same
# moment, so no drawing of this may put two lineages on one colour scale. Two
# layouts honour that, and the number of colours that stay apart on paper is
# what picks between them.

PSEUDOTIME_MAP_WIDTH_MM = 120       # one and a half columns; a single map, drawn to be read
PSEUDOTIME_PANELS_WIDTH_MM = 180    # full page width
PSEUDOTIME_FADE_FLOOR = 0.18        # how much colour the palest cell keeps, on the map
# The three greys a panel uses, light to dark: the rest of the dataset lying
# behind the panel's own lineage, then the cells discarded before the trajectory
# was built, then the cells whose cell type has no fate above it. They must run
# in that order -- a cell drawn lighter than the ground it sits on reads as a
# hole rather than a mark -- and the steps between them are modest because the
# panel's title carries both counts, so nothing has to be told apart by eye.
PSEUDOTIME_PANEL_BACK = "#e7e7e4"
PSEUDOTIME_PANEL_PRUNED = "#d8d8d8"
# One hue, light to dark. A rank is a magnitude, so the colour has to carry
# an order; a multi-hue scale would invite reading the hues as categories.
PSEUDOTIME_RANK_STOPS = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#256abf",
                         "#184f95", "#0d366b"]
# How big a cell is drawn, in points squared, when the caller says nothing. The
# map holds every cell at once so its dots are small; a panel holds one lineage
# against a grey ground, so it can afford bigger ones. The panel's ground keeps
# its share of whatever size the caller asks for, so raising the dots does not
# turn the grey into a solid block behind them.
PSEUDOTIME_MAP_DOT = 0.35
PSEUDOTIME_PANEL_DOT = 1.6
PSEUDOTIME_PANEL_BACK_DOT = 0.5
PSEUDOTIME_HUES = 8                 # what _lineage_hues() comes to; it checks
PSEUDOTIME_LABEL_SHARE = 0.01       # a lineage is named on the map if it holds this much
PSEUDOTIME_LINE_THINNEST = 0.15     # where the direction is barely determined
PSEUDOTIME_LINE_THICKEST = 1.00     # and where it is as determined as anywhere
PSEUDOTIME_NO_FATE = "no fate above their cell type"
PSEUDOTIME_PRUNED = "cell type too rare to keep"


class _IllustratorSafe:
    """Fonts and clipping set for the length of one drawing, then put back.

    Two things decide whether a PDF can be opened and edited afterwards. The
    font has to be one Illustrator resolves without substituting, written as
    TrueType -- ``pdf.fonttype`` 3 hands over each glyph as a drawing procedure
    rather than as text, so a label arrives as outlines. And Matplotlib clips
    nearly every artist to its axes rectangle, which Illustrator turns into a
    clipping mask to release by hand, one per artist.

    Both are settings a library has no business leaving behind in a caller's
    session, so they are put back on the way out. The clip interception has to
    be in place while the figure is built rather than at save time: a clip is
    attached as each artist is added, so a repair afterwards only sees whatever
    happens to be on the figure at that moment.

    Two kinds of clip are kept deliberately. One whose path is not a rectangle
    is doing real work -- the wedge bounding a polar axes, a coastline -- and
    dropping it would break the drawing. And an artist holding no data keeps
    its clip: unclipped, its extent falls back to a marker-sized box at the
    display origin, which drags the whole figure's extent down with it.
    """

    SETTINGS = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,
    }

    def __enter__(self):
        import matplotlib as mpl
        from matplotlib.artist import Artist
        from matplotlib.patches import Rectangle

        self._rc = mpl.rc_context(self.SETTINGS)
        self._rc.__enter__()
        self._clip_box, self._clip_path = Artist.set_clip_box, Artist.set_clip_path
        set_box, set_path = self._clip_box, self._clip_path

        def draws_nothing(artist):
            for getter in ("get_xdata", "get_offsets"):
                read = getattr(artist, getter, None)
                if read is not None:
                    try:
                        return len(read()) == 0
                    except Exception:
                        return False
            return False

        def set_clip_box(artist, clipbox):
            return set_box(artist, clipbox if draws_nothing(artist) else None)

        def set_clip_path(artist, path, transform=None):
            if draws_nothing(artist):
                return set_path(artist, path, transform)
            if transform is None and isinstance(path, Rectangle):
                return set_path(artist, None)
            return set_path(artist, path, transform)

        Artist.set_clip_box = set_clip_box
        Artist.set_clip_path = set_clip_path
        return self

    def __exit__(self, *exc_info):
        from matplotlib.artist import Artist

        Artist.set_clip_box = self._clip_box
        Artist.set_clip_path = self._clip_path
        self._rc.__exit__(*exc_info)
        return False


def _lineage_hues():
    """The colours a lineage may take on the map, and the grey for a cell without
    one.

    All from the palette the rest of the package draws in, so a lineage here is
    in the same family of colours as the same lineage in a tree HECTOR draws.
    Two of the palette's first ten are left out, and both reasons matter: the
    grey, because that is what a cell with no lineage is drawn in and so cannot
    also mean a lineage, and the yellow, because it disappears once it is mixed
    with white at the start of a lineage. That leaves eight, which is the number
    the layout rule turns on.

    They are handed out in a different order from the one the palette lists them
    in. The largest lineages take the first colours, and the palette's first four
    hold both of its blues; a navy at the start of its own lineage and a blue at
    the end of another are the same colour on paper, so the blues are separated
    here and the four most different go first.
    """
    from .trajectory_ontology import ADAPTIVE_PALETTE, NEUTRAL_GRAY

    keep = [c for c in ADAPTIVE_PALETTE[:10]
            if c.upper() not in {"#EFC000", "#7F7F7F"}]      # the yellow, the grey
    if len(keep) != PSEUDOTIME_HUES:
        raise RuntimeError(
            f"The shared palette now yields {len(keep)} usable colours, not the "
            f"{PSEUDOTIME_HUES} this drawing is written around. That number decides "
            "which layout a dataset gets, so set PSEUDOTIME_HUES to match and say so."
        )
    red, blue, teal, navy, orange, purple, pink, brown = keep
    return [red, blue, orange, teal, purple, pink, navy, brown], NEUTRAL_GRAY


def _rank_ramp():
    """The colours a panel fades its own lineage through.

    One hue, light to dark, and most of the lightness there is: from a light blue
    to a dark navy. Fading a single hue towards white instead can only reach that
    hue's own lightness, which squeezes the second half of every lineage into the
    last part of the range.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("hector_pseudotime", PSEUDOTIME_RANK_STOPS)


def _fade(colour, amount, floor=PSEUDOTIME_FADE_FLOOR):
    """A colour mixed with white by ``amount``, 0 palest and 1 full strength.

    Mixed rather than made see-through. Opacity adds up where dots overlap, so
    in the crowded parts of a map a pile of faint cells reads as solid; mixing
    with white leaves every dot carrying its own value however many sit under it.

    ``floor`` is how much colour the palest cell keeps, and it has to answer to
    what the cell is drawn on. On the white of the map a low floor is legible;
    on the grey the panels lay the rest of the dataset in, the same floor is
    paler than the background and the start of a lineage disappears.
    """
    import matplotlib.colors as mcolors

    rgb = np.asarray(mcolors.to_rgb(colour))
    strength = floor + (1 - floor) * np.asarray(amount, dtype=float)
    faded = 1 - (1 - rgb) * strength[:, None]
    return np.column_stack([faded, np.ones(len(strength))])


def _read_pseudotime(adata, lineage_key, pseudotime_key, basis):
    """What the drawing needs, and which groups are lineages rather than shared
    cell types.

    The two groups that carry no rank are told apart here and never merged. A
    cell whose group is a cell type with no fate above it was placed and has not
    committed; a cell with no group at all was never placed on an edge, because
    its cell type was pruned. Describing either as the other is a false caption.
    """
    import pandas as pd

    for key in (lineage_key, pseudotime_key):
        if key not in adata.obs.columns:
            raise KeyError(
                f"adata.obs['{key}'] is missing. Run create_pseudotime_analysis on "
                "this object first."
            )
    if basis not in adata.obsm:
        raise KeyError(
            f"adata.obsm['{basis}'] is missing, so there is nowhere to draw the "
            "cells. Compute a UMAP, or name another set of coordinates."
        )

    group = pd.Series(adata.obs[lineage_key]).astype(object).to_numpy()
    group = np.array([None if (g is None or pd.isna(g)) else str(g) for g in group],
                     dtype=object)
    rank = pd.to_numeric(adata.obs[pseudotime_key], errors="coerce").to_numpy()
    xy = np.asarray(adata.obsm[basis], dtype=float)[:, :2]

    # Which groups are fates, from the table create_pseudotime_analysis leaves.
    # A lineage is one if any of its cell types is, the same rule the ranking
    # uses. Without the table every group is taken to be a lineage, which is the
    # best an older object can support.
    table = adata.uns.get("hector_pseudotime_lineages")
    fated = None
    if isinstance(table, dict) and {"lineage", "has_fate"} <= set(table):
        fated = {str(name) for name, has in zip(table["lineage"], table["has_fate"])
                 if has}

    on_map = np.isfinite(xy).all(axis=1)
    if not on_map.all():
        print(f"  {int((~on_map).sum()):,} cells have no coordinates and are not drawn")
    return xy[on_map], group[on_map], rank[on_map], fated, on_map


def _by_size(group, keep):
    """The groups in ``keep``, largest first, ties broken by name so that a rerun
    draws the same picture."""
    import pandas as pd

    counts = pd.Series([g for g in group if g in keep]).value_counts()
    return sorted(counts.index, key=lambda n: (-counts[n], n)), counts


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _corner_marker(ax, dx=0.11, dy=0.11, x0=0.01, y0=0.01, gap=0.038, size=6):
    """Two arrows in place of axes, the usual way of showing a UMAP: its axes
    carry no units, so ticks would be numbers that mean nothing. ``gap`` is how
    far a name sits from its arrow, larger on a small axes drawn only to hold the
    marker or the name lands on the arrow."""
    from matplotlib.patches import FancyArrowPatch

    for ddx, ddy, tx, ty, text, rotation in [
        (dx, 0, dx / 2, -gap, "UMAP 1", 0),
        (0, dy, -gap, dy / 2, "UMAP 2", 90),
    ]:
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x0 + ddx, y0 + ddy), transform=ax.transAxes,
            arrowstyle="-|>", mutation_scale=4, linewidth=0.7, color="black",
            clip_on=False))
        ax.text(x0 + tx, y0 + ty, text, transform=ax.transAxes, fontsize=size,
                ha="center", va="center", rotation=rotation, clip_on=False)


def _draw_strength_key(ax, n=120):
    """A grey ramp faded the way the cells are, to say what colour strength
    means. Grey because the ramp stands for every lineage's hue at once and must
    not look like one of them."""
    from matplotlib.patches import Rectangle

    for i, value in enumerate(np.linspace(0, 1, n)):
        grey = 1 - 0.85 * (PSEUDOTIME_FADE_FLOOR + (1 - PSEUDOTIME_FADE_FLOOR) * value)
        ax.add_patch(Rectangle((i / n, 0), 1.02 / n, 1, transform=ax.transAxes,
                               linewidth=0, facecolor=(grey, grey, grey)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _neighbour_index(adata, neighbors_basis, n_neighbors, on_map):
    """Each cell's nearest others, found in HECTOR's own embedding rather than on
    the map. The map is the picture; neighbours found on it would make the lines
    a creature of the drawing they are drawn on.

    Restricted to the cells being drawn before the search rather than after, so
    that the numbers it returns index the same rows the drawing holds. Slicing
    the result instead would keep positions counted against the whole object and
    quietly point every arrow at the wrong cell.
    """
    from sklearn.neighbors import NearestNeighbors

    if neighbors_basis not in adata.obsm:
        raise KeyError(
            f"The lines need adata.obsm['{neighbors_basis}'], HECTOR's own cell "
            "embedding. create_pseudotime_analysis computes it, or pass lines=False."
        )
    space = np.asarray(adata.obsm[neighbors_basis], dtype=np.float32)[on_map]
    k = int(min(max(n_neighbors, 1), len(space) - 1))
    print(f"  finding {k} neighbours for each of {space.shape[0]:,} cells in "
          f"{space.shape[1]} dimensions, a few minutes on a large dataset", flush=True)
    finder = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric="cosine")
    return finder.fit(space).kneighbors(space, return_distance=False)[:, 1:]


def _cell_arrows(xy, rank, group, ranked, idx):
    """One arrow per cell, pointing the way the rank goes up.

    The average, over a cell's neighbours, of the unit step towards that
    neighbour weighted by how much further along it is. The weight goes through
    a hyperbolic tangent against the typical difference among neighbours, which
    bounds it, so a neighbour far ahead counts for more than a close one but not
    without limit.

    Only pairs inside one lineage contribute. A rank in one lineage against a
    rank in another is not a difference in how far along anything, and without
    the restriction the field will draw an arrow from one branch to another that
    looks just as convincing as the real ones.
    """
    step = xy[idx] - xy[:, None, :]
    length = np.linalg.norm(step, axis=2, keepdims=True)
    unit = np.divide(step, length, out=np.zeros_like(step), where=length > 0)

    ahead = np.nan_to_num(rank[idx] - rank[:, None])
    scale = np.median(np.abs(ahead))
    weight = np.tanh(ahead / scale) if scale > 0 else np.sign(ahead)

    same = (group[:, None] == group[idx]) & ranked[:, None]
    print(f"  {same.mean():.1%} of neighbour pairs are inside one lineage")
    weight = np.where(same, weight, 0.0)
    return (weight[:, :, None] * unit).sum(axis=1) / np.maximum(
        same.sum(axis=1, keepdims=True), 1
    )


def _arrows_to_grid(xy, arrows, n=42, smooth=1.0):
    """Average the per-cell arrows onto a square grid, blur, and drop the squares
    too thinly populated to say anything. How thin is measured against the number
    of cells, because a fixed count would empty the field on a small dataset."""
    from scipy.ndimage import gaussian_filter

    min_cells = max(5, round(len(xy) / 10000))
    print(f"  a square must hold {min_cells} cells once blurred to carry a line")

    xe = np.linspace(xy[:, 0].min(), xy[:, 0].max(), n + 1)
    ye = np.linspace(xy[:, 1].min(), xy[:, 1].max(), n + 1)
    count, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye])
    total = [np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye], weights=arrows[:, d])[0]
             for d in (0, 1)]

    # Blur the sums and the counts together, so squares beside empty map are not
    # dragged towards zero by their neighbours' emptiness.
    blurred = gaussian_filter(count, smooth)
    field = [gaussian_filter(t, smooth) / np.where(blurred > 0, blurred, 1)
             for t in total]
    # Drop a square if it is thinly populated once blurred, and also if it holds
    # no cells at all -- the blur alone would let lines wander across the gaps
    # between the arms of the map.
    thin = (blurred < min_cells) | (count == 0)
    u, v = (np.where(thin, np.nan, f).T for f in field)   # streamplot wants y, x
    return xe[:-1] + np.diff(xe) / 2, ye[:-1] + np.diff(ye) / 2, u, v


def _draw_lines(ax, xc, yc, u, v):
    """Lines following the arrows, one dark colour, thicker where the direction is
    better determined.

    **Width is the only thing they say beyond which way they run**, and what it
    says is how much a cell's neighbours agree, not how fast the rank changes.
    The arrows are averaged unit steps, so they cancel where neighbours point
    every which way and survive where they agree. Steepness would be the wrong
    quantity to put here: a rank is a percentile within a lineage, so a small
    lineage still runs 0 to 1 and its rank changes fast across the map only
    because it is small.

    Colour is wrong for the same reason and worse. A line is traced onward from
    square to square and crosses between lineages, and a rank in one lineage is
    not a rank in another; the map already spends its colour on which lineage a
    cell belongs to and how far along it sits.

    They are not RNA velocity, which measures a rate per cell from unspliced and
    spliced counts. These are the slope of a value that was assigned: where that
    value is wrong the lines are wrong in the same way and look exactly as
    convincing.
    """
    import matplotlib.patheffects as pe

    strength = np.hypot(u, v)
    top = np.nanpercentile(strength, 98)
    # The dropped squares carry no strength, but the width array still has to be
    # all finite or the PDF writer refuses it.
    share = np.clip(np.nan_to_num(strength) / (top if top > 0 else 1), 0, 1)
    width = PSEUDOTIME_LINE_THINNEST + share * (
        PSEUDOTIME_LINE_THICKEST - PSEUDOTIME_LINE_THINNEST
    )
    drawn = ax.streamplot(xc, yc, u, v, density=1.0, color="#1A1A1A",
                          linewidth=width, arrowsize=0.55, arrowstyle="-|>",
                          minlength=0.08)
    # One stroke width covers the whole collection, so it is set wide enough to
    # leave white on both sides of the thickest line, not only the thinnest.
    halo = [pe.withStroke(linewidth=PSEUDOTIME_LINE_THICKEST + 1.0, foreground="white"),
            pe.Normal()]
    drawn.lines.set_path_effects(halo)
    drawn.arrows.set_path_effects(halo)


def _draw_map(xy, group, rank, order, counts, greys, idx, width, dot):
    """One map: a hue for the lineage, colour strength for the rank inside it."""
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import textwrap

    palette, neutral = _lineage_hues()
    hue = {name: palette[i] for i, name in enumerate(order)}
    ranked = np.array([g in hue for g in group])
    no_fate, pruned = greys

    colours = np.zeros((len(xy), 4))
    colours[no_fate] = mcolors.to_rgba(neutral)
    # The pruned cells are lighter than the shared ones, so data that was
    # discarded recedes and the ontology's own refusal is the one that stays
    # visible. Both are grey: neither carries a rank.
    if pruned.any():
        colours[pruned] = _fade(neutral, np.full(int(pruned.sum()), 0.45))
    if ranked.any():
        strength = np.nan_to_num(rank[ranked])
        each = np.array([mcolors.to_rgb(hue[g]) for g in group[ranked]])
        faded = 1 - (1 - each) * (
            PSEUDOTIME_FADE_FLOOR + (1 - PSEUDOTIME_FADE_FLOOR) * strength
        )[:, None]
        colours[ranked] = np.column_stack([faded, np.ones(int(ranked.sum()))])

    aspect = ((xy[:, 1].max() - xy[:, 1].min())
              / (xy[:, 0].max() - xy[:, 0].min()))
    left, right, top, bottom = 0.01, 0.99, 0.965, 0.10
    # The gap has to clear two things that both sit in it: the corner marker,
    # hanging below the map, and the caption, sitting above the key.
    key_h, key_gap = 0.030, 0.17
    block = (right - left) * width * aspect * (1 + key_gap * (1 + key_h) / 2 + key_h)

    fig = plt.figure(figsize=(width, block / (top - bottom)))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, key_h], hspace=key_gap,
                          left=left, right=right, top=top, bottom=bottom)
    ax = fig.add_subplot(gs[0])

    # The cells carrying no rank go down first and flat, so a coloured cell is
    # never hidden by one with nothing to show. Then the palest of the rest, so
    # the strongest land on top rather than whatever came last in the table.
    rest = ~ranked
    ax.scatter(xy[rest, 0], xy[rest, 1], s=dot, c=colours[rest], linewidths=0,
               rasterized=True)
    inside = np.where(ranked)[0][np.argsort(np.nan_to_num(rank[ranked]))]
    ax.scatter(xy[inside, 0], xy[inside, 1], s=dot, c=colours[inside], linewidths=0,
               rasterized=True)

    if idx is not None:
        xc, yc, u, v = _arrows_to_grid(
            xy, _cell_arrows(xy, np.where(np.isfinite(rank), rank, 0.0), group,
                             ranked, idx)
        )
        print(f"  lines drawn on {int(np.isfinite(u).sum()):,} of {u.size:,} squares")
        _draw_lines(ax, xc, yc, u, v)

    _bare(ax)
    # Inside the map's own empty corner rather than below it. Below, the marker
    # falls into the same gap the caption sits in and its name crowds the first
    # line; the bottom-left of a UMAP is empty on any dataset worth drawing.
    _corner_marker(ax, x0=0.02, y0=0.07)

    # Each lineage named on its own cells rather than in a colour key, so the eye
    # does not travel. One too small to hold a name is named underneath instead.
    unnamed = []
    for name in order:
        if counts[name] < PSEUDOTIME_LABEL_SHARE * len(xy):
            unnamed.append(name)
            continue
        here = np.array([g == name for g in group])
        ax.text(np.median(xy[here, 0]), np.median(xy[here, 1]),
                "\n".join(textwrap.wrap(str(name), 18)), fontsize=6.5, color="black",
                ha="center", va="center", linespacing=1.2,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

    cax = fig.add_subplot(gs[1])
    _draw_strength_key(cax)
    for x, text, ha in [(0.0, "start of the lineage", "left"), (1.0, "end", "right")]:
        cax.text(x, -0.8, text, transform=cax.transAxes, fontsize=6.5, ha=ha, va="top")
    caption = ("Colour is the lineage a cell is heading for, its strength how far "
               "along its own lineage the cell sits.\nEvery lineage is ranked on its "
               "own, so a strength in one is not a strength in another.")
    if idx is not None:
        caption += ("\nLines follow the way that rank goes up. They are its slope, "
                    "not a measured rate.")
    cax.set_title(caption, fontsize=6.5, pad=4, linespacing=1.5)

    # Nothing on the map is left unaccounted for: a lineage too small to hold its
    # name, and each of the two groups that carry no rank, are named here.
    note = [f"{name} ({counts[name]:,} cells)" for name in unnamed]
    if no_fate.any():
        note.append(f"in grey, {PSEUDOTIME_NO_FATE} ({int(no_fate.sum()):,} cells)")
    if pruned.any():
        note.append(f"in paler grey, {PSEUDOTIME_PRUNED} ({int(pruned.sum()):,} cells)")
    if note:
        cax.text(0.5, -3.4, "\n".join(textwrap.wrap(
                     "not named on the map: " + "; ".join(note), 92)),
                 transform=cax.transAxes, ha="center", va="top", fontsize=6,
                 linespacing=1.4)
    return fig


def _draw_panels(xy, group, rank, order, counts, greys, width, panels_per_row,
                 dot):
    """One panel per lineage, each on its own scale, the rest of the data grey."""
    import matplotlib.pyplot as plt
    import textwrap
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    _, neutral = _lineage_hues()
    back_dot = PSEUDOTIME_PANEL_BACK_DOT * (dot / PSEUDOTIME_PANEL_DOT)
    no_fate, pruned = greys
    # Light to dark in both layouts, so colour strength means how far along in
    # both. A panel can spend the whole range on one lineage; the map cannot,
    # because it has eight hues to keep apart and darkening the top of each would
    # bring them together.
    ramp = _rank_ramp()

    lo = np.percentile(xy, 0.2, axis=0)
    hi = np.percentile(xy, 99.8, axis=0)
    pad = (hi - lo) * 0.03
    lo, hi = lo - pad, hi + pad
    aspect = (hi[1] - lo[1]) / (hi[0] - lo[0])

    unranked = no_fate | pruned
    panels = list(order) + ([None] if unranked.any() else [])

    # Every panel is placed by hand rather than by a grid. A grid cell is not the
    # panel: set_aspect("equal") shrinks the drawing inside its cell to the map's
    # own proportions, so the space a row occupies cannot be predicted from the
    # grid, which is what leaves gaps between rows and puts the colour bar on top
    # of the bottom one.
    mm = 1 / 25.4
    rows = int(np.ceil(len(panels) / panels_per_row))
    margin, gutter = 2 * mm, 2 * mm
    title_h = 9 * mm             # a cell type name runs to three lines at 6 pt
    bar_h = 16 * mm              # colour bar and its two-line label

    panel_w = (width - 2 * margin - (panels_per_row - 1) * gutter) / panels_per_row
    panel_h = panel_w * aspect
    height = rows * (title_h + panel_h) + bar_h
    print(f"  panel {panel_w / mm:.0f} x {panel_h / mm:.0f} mm, "
          f"figure {width / mm:.0f} x {height / mm:.0f} mm")

    fig = plt.figure(figsize=(width, height))
    for i, name in enumerate(panels):
        r, c = divmod(i, panels_per_row)
        x0 = margin + c * (panel_w + gutter)
        y0 = height - (r + 1) * (title_h + panel_h)
        ax = fig.add_axes([x0 / width, y0 / height, panel_w / width, panel_h / height])
        here = unranked if name is None else np.array([g == name for g in group])
        ax.scatter(xy[~here, 0], xy[~here, 1], s=back_dot, c=PSEUDOTIME_PANEL_BACK,
                   linewidths=0, rasterized=True)
        if name is None:
            # No rank to show, so flat colour: drawn to say how much of the map
            # these cells hold, not to be read on the bar. The two are kept apart
            # because they mean different things -- one is the ontology declining
            # to commit, the other is a cell type discarded before the trajectory.
            for mask, colour in ((no_fate, neutral), (pruned, PSEUDOTIME_PANEL_PRUNED)):
                if mask.any():
                    ax.scatter(xy[mask, 0], xy[mask, 1], s=dot, c=colour,
                               linewidths=0, rasterized=True)
            title = (f"{PSEUDOTIME_NO_FATE}, {int(no_fate.sum()):,} cells\n"
                     f"{PSEUDOTIME_PRUNED}, {int(pruned.sum()):,} cells\nneither is ranked")
        else:
            v = np.nan_to_num(rank[here])
            o = np.argsort(v)                       # palest first
            points = xy[here][o]
            ax.scatter(points[:, 0], points[:, 1], s=dot, c=ramp(v[o]), linewidths=0,
                       rasterized=True)
            # The full cell type name, never shortened, so wrap rather than truncate.
            title = ("\n".join(textwrap.wrap(str(name), 30))
                     + f"\n{int(counts[name]):,} cells")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        _bare(ax)
        ax.set_title(title, fontsize=6, linespacing=1.25, pad=2)

    # One corner marker for the whole figure, in the space beside the colour bar.
    # Every panel is the same map on the same limits, so one per panel would be
    # many copies of one statement, and inside a small panel the two names collide.
    marker = fig.add_axes([margin / width, 3 * mm / height,
                           24 * mm / width, 11 * mm / height])
    marker.set_axis_off()
    _corner_marker(marker, dx=0.55, dy=0.62, x0=0.13, y0=0.16, gap=0.13)

    cax = fig.add_axes([0.30, 9.5 * mm / height, 0.40, 2.4 * mm / height])
    bar = fig.colorbar(ScalarMappable(Normalize(0, 1), ramp), cax=cax,
                       orientation="horizontal")
    bar.outline.set_visible(False)
    bar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    bar.ax.tick_params(labelsize=6, length=2, width=0.8, pad=1)
    cax.set_xlabel("where a cell sits in the order of its own lineage\n"
                   "(rank within that lineage; every panel has its own scale, and "
                   "grey is the rest of the dataset)",
                   fontsize=6.5, labelpad=2, linespacing=1.4)
    return fig


def plot_pseudotime(
    adata,
    output_file: str,
    layout: str = "auto",
    lines: bool = False,
    lineage_key: str = "hector_lineage",
    pseudotime_key: str = "hector_pseudotime",
    basis: str = "X_umap",
    neighbors_basis: str = "X_hector",
    n_neighbors: int = 30,
    panels_per_row: int = 4,
    dot_size: Optional[float] = None,
) -> None:
    """Draw the lineages and each cell's rank within its own, to a PDF or a PNG.

    Needs no ``predictor``: everything comes off the object, so no model has to
    be loaded to draw.

    Two layouts, because a rank is only defined within one lineage and no drawing
    of it may put two lineages on one colour scale:

    - ``"map"`` -- one map, a hue for each lineage and colour strength for the
      rank inside it, so no colour is ever read across two lineages.
    - ``"panels"`` -- one panel per lineage, the whole dataset in grey and that
      lineage faded through a single hue on its own scale.

    **The result is an order, not an amount.** A cell at 0.5 has half its own
    lineage's cells before it. It is not half way along the lineage, the gap
    between two cells is not an amount of anything, and two cells of the same
    value in different lineages are each at the same place in their own order,
    not at the same moment.

    Args:
        adata: AnnData already processed by ``create_pseudotime_analysis``.
        output_file: Where to write. The suffix decides the format; ``.pdf``
            uses editable text where supported by the PDF consumer.
        layout: ``"auto"``, ``"map"`` or ``"panels"``. ``"auto"`` takes the map
            when the lineages fit the colours that stay apart on paper, and the
            panels when they do not. Cell types with no fate above them are not
            counted, so a handful of stray cells cannot cost a dataset its map.
        lines: Draw lines following the way the rank goes up. They need
            ``adata.obsm[neighbors_basis]`` and cost a nearest-neighbour search
            over every cell, minutes on a large dataset. Off by default, and
            ignored by the panels, where each panel already holds one lineage.
        lineage_key: Column containing lineage assignments.
        pseudotime_key: Column containing within-lineage pseudotime ranks.
        basis: Coordinates to draw the cells on.
        neighbors_basis: Coordinates the neighbours for the lines are found in.
            HECTOR's own cell embedding, not the map: the map is the picture, and
            neighbours found on it would make the lines a creature of the drawing.
        n_neighbors: Neighbours per cell for the lines.
        panels_per_row: How many panels sit in a row. The panels layout only.
        dot_size: How big one cell is drawn, in points squared. Left at ``None``
            each layout takes its own default, which suits a dataset of tens of
            thousands of cells; raise it for a smaller one, where the defaults
            leave the cells too faint to see. On the panels the grey ground
            behind the lineage keeps its share of whatever is asked for.

    Returns:
        None. Writes ``output_file``.

    Raises:
        KeyError: If the columns or coordinates it needs are missing.
        ValueError: If ``layout`` is not one of the three, or ``"map"`` is asked
            for with more lineages than there are colours.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required to draw. Install matplotlib.")

    xy, group, rank, fated, on_map = _read_pseudotime(
        adata, lineage_key, pseudotime_key, basis
    )
    present = {g for g in group if g}
    if fated is None:
        print("  no lineage table on this object; every group is taken to be a lineage")
        fated = set(present)
    order, counts = _by_size(group, fated & present)

    # Three groups, and none of them is another. Only the first is ranked.
    no_fate = np.array([g is not None and g not in fated for g in group])
    pruned = np.array([g is None for g in group])

    print(f"{len(xy):,} cells drawn, {len(order)} lineages")
    for name in order:
        print(f"  {str(name)[:52]:54s} {counts[name]:>8,} cells")
    if no_fate.any():
        print(f"  {'(' + PSEUDOTIME_NO_FATE + ')':54s} {int(no_fate.sum()):>8,} cells, "
              "not ranked")
    if pruned.any():
        print(f"  {'(' + PSEUDOTIME_PRUNED + ')':54s} {int(pruned.sum()):>8,} cells, "
              "not ranked")

    if layout not in ("auto", "map", "panels"):
        raise ValueError(f"layout is 'auto', 'map' or 'panels', not {layout!r}")
    if layout == "auto":
        layout = "map" if len(order) <= PSEUDOTIME_HUES else "panels"
        print(f"  layout: {layout}, because there are {len(order)} lineages and the "
              f"map carries {PSEUDOTIME_HUES}")
    elif layout == "map" and len(order) > PSEUDOTIME_HUES:
        raise ValueError(
            f"The map gives every lineage its own colour and there are only "
            f"{PSEUDOTIME_HUES} that stay apart on paper, against {len(order)} "
            "lineages here. Use layout='panels', or merge lineages with the "
            "`lineages` argument of create_pseudotime_analysis."
        )

    idx = None
    if lines and layout == "map":
        idx = _neighbour_index(adata, neighbors_basis, n_neighbors, on_map)
    elif lines:
        print("  the lines are for the map only; each panel already holds one lineage")

    mm = 1 / 25.4
    with _IllustratorSafe():
        if layout == "map":
            fig = _draw_map(xy, group, rank, order, counts, (no_fate, pruned), idx,
                            PSEUDOTIME_MAP_WIDTH_MM * mm,
                            PSEUDOTIME_MAP_DOT if dot_size is None else float(dot_size))
        else:
            fig = _draw_panels(
                xy, group, rank, order, counts, (no_fate, pruned),
                PSEUDOTIME_PANELS_WIDTH_MM * mm, panels_per_row,
                PSEUDOTIME_PANEL_DOT if dot_size is None else float(dot_size))
        fig.savefig(output_file, dpi=600, bbox_inches="tight", pad_inches=0.05)

    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"\n  wrote {output_file}")
    return None


# =============================================================================
# Public API: create_gene_trend_analysis() and plot_gene_trends()
# =============================================================================
#
# Two questions, and they need different instruments.
#
# **Which genes change.** Walk the lineage's cells in pseudotime order, keeping a
# running total of how far each sits above or below the gene's own average. The
# total starts and ends at zero, so only the journey carries anything: it travels
# far only when the cells holding a lot of the gene are bunched together somewhere
# along the order. The statistic is that total's range -- the largest excess any
# stretch of the order has over the rest of it, over every stretch and every width
# at once. Nothing is binned, smoothed or fitted, so nothing is set by anyone.
# P-values come from shuffling the cell order, since the usual distributions
# assume noise that single-cell counts do not have.
#
# **Where a gene peaks.** Not from the walk: every position a walk offers is an
# average over the axis, and an average is pulled towards the middle, so a gene
# switching on at the very end reads as peaking two thirds along. The peak comes
# from a curve fitted through the cells, whose bendiness is chosen per gene by
# holding a fifth of them back and keeping the level that best predicts them.
#
# **The ruler, held fixed.** ``create_pseudotime_analysis`` orders the cell types
# the ontology is silent about by how many genes each cell expresses, so a gene
# tracking that count would look like one that changes along the lineage. Both
# stages work on what a straight line against that count leaves of each gene.

GENE_TRENDS_SHUFFLES = 20           # only sets how finely a small p-value is quoted
GENE_TRENDS_STRATA = 5              # detection-rate groups the null is pooled within
GENE_TRENDS_LEVELS = (4, 6, 8, 12, 20, 40)   # bendiness levels the sweep considers
GENE_TRENDS_FOLDS = 5               # parts the cells are split into for that sweep
GENE_TRENDS_FINE = 400              # where the chosen curve is read, to describe it
GENE_TRENDS_DEGREE = 3              # cubic pieces
GENE_TRENDS_RIDGE = 1e-6            # covers a knot span holding too few cells
# What the stage that reads every cell may hold across all threads at once. Genes
# are taken in chunks sized to fit it, so more threads costs smaller chunks rather
# than memory the machine may not have.
GENE_TRENDS_MEMORY_BUDGET = 6e9
GENE_TRENDS_COPIES = 8              # roughly how many copies of a chunk exist at the peak
GENE_TRENDS_MIN_CHUNK, GENE_TRENDS_MAX_CHUNK = 200, 2000
# Past about four threads the gain flattens: the heavy stage moves data rather than
# calculating, so further threads queue on memory channels instead of on cores.
# Measured on a 12,000-cell lineage, the shuffles took 39 s at one thread, 19 s at
# four and 21 s at eight.
GENE_TRENDS_MAX_THREADS = 6
GENE_TRENDS_STRETCHES = 60          # stretches the cell-type strip is painted in
GENE_TRENDS_CLIP = 2.5              # standard deviations, beyond which colour stops changing
GENE_TRENDS_POINTS = 200            # where the refitted curve is read off, to draw it
GENE_TRENDS_ROWS = 400              # rows per panel: a drawing limit, not a threshold
GENE_TRENDS_WIDTH_MM = 180          # full page width
GENE_TRENDS_PANEL_HEIGHT_MM = 26    # the map itself, per panel
# Cell types named under a panel before the rest are counted. Three, so the key is
# always two rows: at six it ran to three and the last was drawn over the map.
GENE_TRENDS_KEY_ENTRIES = 3
# Okabe and Ito's set, which stays apart for the common kinds of colour blindness.
GENE_TRENDS_STRIP_COLOURS = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                             "#0072B2", "#D55E00", "#CC79A7", "#999999"]


def _gene_trend_threads(n_rounds, n_jobs=None):
    """How many shuffles to run at once.

    ``sched_getaffinity`` rather than ``cpu_count`` because the second reports the
    machine and the first reports what this process was actually granted, which is
    what matters inside a container or under a scheduler.
    """
    if n_jobs is not None:
        return max(1, int(n_jobs))
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 2
    return max(1, min(n_rounds, cores - 1, GENE_TRENDS_MAX_THREADS))


def _gene_trend_chunk(n_cells, threads):
    """Genes held at once by each thread, so the whole stays inside the budget."""
    per_thread = GENE_TRENDS_MEMORY_BUDGET / max(threads, 1)
    return int(np.clip(per_thread / (max(n_cells, 1) * 8 * GENE_TRENDS_COPIES),
                       GENE_TRENDS_MIN_CHUNK, GENE_TRENDS_MAX_CHUNK))


def _normalise_for_trends(counts):
    """Counts scaled to a common library size and log-transformed.

    Stays sparse: ``log1p`` leaves zeros at zero, so the transform touches only the
    stored values.
    """
    import scipy.sparse as sp

    counts = sp.csr_matrix(counts)
    totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    totals[totals == 0] = 1.0
    scaled = sp.diags(float(np.median(totals)) / totals) @ counts
    scaled = sp.csr_matrix(scaled)
    scaled.data = np.log1p(scaled.data)
    return scaled


def _per_gene(values, indptr, counts, reduce_with):
    """Reduce the stored entries of each gene, genes with none giving zero.

    ``reduceat`` cannot be handed a column that stores nothing -- it returns the
    next column's first entry rather than nothing -- so the empty ones are left out
    of the reduction and filled in afterwards.
    """
    out = np.zeros(len(counts))
    filled = counts > 0
    if filled.any():
        out[filled] = reduce_with.reduceat(values, indptr[:-1][filled])
    return out


def _equal_slices(rank, n_slices):
    """Which of ``n_slices`` equal-sized stretches of the order each cell falls in.

    Equal-sized rather than equal-width because the pseudotime is a rank and so is
    uniform by construction: the two come to the same thing, and equal sizes stop
    ties at one value from deciding the edges arbitrarily.
    """
    order = np.argsort(np.asarray(rank, dtype=float), kind="stable")
    slice_of_cell = np.empty(len(order), dtype=int)
    slice_of_cell[order] = np.minimum(
        (np.arange(len(order)) * n_slices) // max(len(order), 1), n_slices - 1
    )
    return slice_of_cell


def _reading_every_cell(walked, which, running_covariate, mean, slope, n_cells,
                        chunk):
    """The extremes of the adjusted running total, read at every cell.

    Only the genes whose total can rise between one detection and the next need
    this. For the remaining genes, the turning points are the detections
    themselves and the sparse path finds them directly.
    """
    steps = np.arange(1, n_cells + 1, dtype=np.float64)
    # One matrix product rather than two rank-one updates: what has to come off is
    # a straight line in the cell number and in the covariate's running total.
    shared = np.column_stack([steps, running_covariate])
    highest = np.zeros(len(which))
    lowest = np.zeros(len(which))
    for start in range(0, len(which), chunk):
        here = which[start:start + chunk]
        total = np.cumsum(
            np.asarray(walked[:, here].todense(), dtype=np.float64), axis=0)
        total -= shared @ np.vstack([mean[here], slope[here]])
        highest[start:start + chunk] = total.max(axis=0)
        lowest[start:start + chunk] = total.min(axis=0)
    return highest, lowest


def _walk(expression, cell_order, covariate=None, chunk=None):
    """How far each gene's running total strays, walking the cells in one order.

    Args:
        expression: Cells-by-genes sparse matrix, already normalised.
        cell_order: The order to walk in -- the pseudotime order for the real
            answer, a shuffle of it for the null.
        covariate: Optional per-cell number to hold fixed. Each gene then has the
            straight line that best predicts it from that number subtracted before
            the walk, so a gene that only tracks the ruler no longer travels.
        chunk: Genes the every-cell stage holds at once.

    Returns:
        One number per gene: the running total's range, scaled by the gene's own
        spread and the square root of the cell count so that genes of very
        different abundance are comparable.
    """
    import scipy.sparse as sp

    n_cells = expression.shape[0]
    walked = sp.csc_matrix(expression[cell_order])
    # Accumulate in double precision so long running totals retain adequate
    # resolution even when the input matrix uses single precision.
    data = np.asarray(walked.data, dtype=np.float64)
    rows, indptr = walked.indices, walked.indptr
    counts = np.diff(indptr)
    n_genes = len(counts)
    if len(data) == 0:
        return np.zeros(n_genes)

    total = _per_gene(data, indptr, counts, np.add)
    mean = total / n_cells
    squared = _per_gene(data * data, indptr, counts, np.add)

    # The running total within each gene, taken from the running total across all
    # of them by subtracting where each gene started.
    running = np.cumsum(data)
    start_value = np.zeros(n_genes)
    filled = counts > 0
    start_value[filled] = (running[indptr[:-1][filled]]
                           - data[indptr[:-1][filled]])
    running = running - np.repeat(start_value, counts)

    # The total just after each cell that expresses the gene, and just before it.
    step = rows + 1.0
    mean_per_entry = np.repeat(mean, counts)
    after = running - step * mean_per_entry
    before = (running - data) - (step - 1.0) * mean_per_entry

    if covariate is None:
        # With nothing held fixed the total only slides down between detections,
        # so every turning point it has is one of the two candidates above.
        highest = np.maximum(_per_gene(after, indptr, counts, np.maximum), 0.0)
        lowest = np.minimum(_per_gene(before, indptr, counts, np.minimum), 0.0)
        spread = np.sqrt(np.maximum(squared / n_cells - mean ** 2, 0.0))
    else:
        centred = np.asarray(covariate, dtype=float)[cell_order]
        centred = centred - centred.mean()
        squared_covariate = float((centred ** 2).sum())
        running_covariate = np.cumsum(centred)
        # Subtracting that straight line shifts every gene's running total by one
        # curve shared with all the others -- the covariate's own running total --
        # times one number per gene, its slope against the covariate. So nothing
        # here needs a second pass over the matrix.
        slope = np.zeros(n_genes)
        if squared_covariate > 0:
            slope = np.asarray(walked.T @ centred,
                               dtype=np.float64).ravel() / squared_covariate
        slope_per_entry = np.repeat(slope, counts)
        before_here = np.where(rows > 0, running_covariate[rows - 1], 0.0)
        highest = np.maximum(_per_gene(
            after - slope_per_entry * running_covariate[rows],
            indptr, counts, np.maximum), 0.0)
        lowest = np.minimum(_per_gene(
            before - slope_per_entry * before_here,
            indptr, counts, np.minimum), 0.0)

        # Between one detection and the next the total falls by the gene's own
        # average plus its slope times that cell's covariate. Where that is
        # positive at every cell the total can only fall, so its turning points are
        # the detections and the two candidates above are the whole story. Where it
        # is not, the total can rise mid-gap and has to be read everywhere.
        rises = ~((mean + slope * centred.min() > 0)
                  & (mean + slope * centred.max() > 0))
        rises &= counts > 0
        if rises.any():
            which = np.where(rises)[0]
            top, bottom = _reading_every_cell(
                walked, which, running_covariate, mean, slope, n_cells,
                chunk or _gene_trend_chunk(n_cells, 1))
            highest[which] = np.maximum(top, 0.0)
            lowest[which] = np.minimum(bottom, 0.0)

        # What the covariate leaves of the gene. The residual is orthogonal to the
        # covariate by construction, so its spread is the gene's own less the part
        # the straight line accounts for.
        variance = squared / n_cells - mean ** 2
        variance -= slope ** 2 * squared_covariate / n_cells
        spread = np.sqrt(np.maximum(variance, 0.0))

    return (highest - lowest) / np.maximum(spread * np.sqrt(n_cells), 1e-12)


def _empirical_p(observed, null, stratum):
    """P-values read off the shuffled statistics rather than a distribution.

    Genes are compared only against shuffled values from their own detection-rate
    group, because a gene seen in 1% of cells and one seen in 90% do not have the
    same null. Floored at one over the number of values in the group plus one --
    below that the shuffles have nothing to say.
    """
    p = np.ones(len(observed))
    for group in np.unique(stratum):
        members = np.where(stratum == group)[0]
        pool = np.sort(null[:, members].ravel())
        beyond = len(pool) - np.searchsorted(pool, observed[members], side="left")
        p[members] = (beyond + 1) / (len(pool) + 1)
    return p


def _benjamini_hochberg(p):
    """Benjamini-Hochberg false discovery rate, as a q-value per gene."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = np.empty(len(p))
    ranked[order] = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    return np.minimum(ranked, 1.0)


def _spline_basis(positions, rank, n_basis):
    """A bendable curve's building blocks, evaluated at the given positions.

    Args:
        positions: Where to evaluate; anything outside the knots is pulled to the
            nearest edge.
        rank: The lineage's cell positions, whose quantiles place the knots.
        n_basis: How bendy the curve may be. The count is nominal: it divides the
            axis at ``n_basis - 2`` points and gives each gene ``n_basis + 2`` free
            numbers.
    """
    from scipy.interpolate import BSpline

    interior = np.quantile(rank, np.linspace(0, 1, n_basis)[1:-1])
    knots = np.concatenate([[rank.min()] * (GENE_TRENDS_DEGREE + 1), interior,
                            [rank.max()] * (GENE_TRENDS_DEGREE + 1)])
    inside = np.clip(np.asarray(positions, dtype=float),
                     knots[GENE_TRENDS_DEGREE], knots[-GENE_TRENDS_DEGREE - 1])
    return BSpline.design_matrix(inside, knots, GENE_TRENDS_DEGREE).toarray()


def _spline_gram(basis):
    """The basis against itself, with a ridge for a knot span holding few cells."""
    gram = basis.T @ basis
    gram += np.eye(gram.shape[0]) * (np.trace(gram) / gram.shape[0]) * GENE_TRENDS_RIDGE
    return gram


def _spline_fit(basis, block):
    """One least-squares solve covering every gene at once."""
    return np.linalg.solve(_spline_gram(basis),
                           np.asarray(block.T @ basis, dtype=np.float64).T)


def _straight_line(block, covariate):
    """Each gene's intercept and slope against the covariate, fitted at once.

    The curve has to be fitted to what the covariate does not account for, or a
    gene that only follows the count of detected genes would still show a peak and
    the peak would be the count's. This is the line to subtract.
    """
    n = block.shape[0]
    sum_c = float(covariate.sum())
    sum_cc = float((covariate ** 2).sum())
    sum_y = np.asarray(block.sum(axis=0)).ravel().astype(np.float64)
    sum_yc = np.asarray(block.T @ covariate, dtype=np.float64).ravel()
    determinant = n * sum_cc - sum_c ** 2
    if abs(determinant) < 1e-12:
        return sum_y / max(n, 1), np.zeros(block.shape[1])
    return ((sum_cc * sum_y - sum_c * sum_yc) / determinant,
            (n * sum_yc - sum_c * sum_y) / determinant)


def _spline_fit_beyond(basis, block, covariate, intercept, slope):
    """The curve through what the straight line leaves of each gene.

    Fitting to ``y - a - b*c`` needs no dense residual: the basis against that
    residual is the basis against ``y``, less ``a`` times the basis against a
    column of ones and ``b`` times the basis against the covariate, and the last
    two are shared by every gene.
    """
    crossed = np.asarray(block.T @ basis, dtype=np.float64).T
    crossed = (crossed - np.outer(basis.sum(axis=0), intercept)
               - np.outer(basis.T @ covariate, slope))
    return np.linalg.solve(_spline_gram(basis), crossed)


def _held_out_error(block, rank, n_basis, folds, covariate=None):
    """Squared error per gene on cells the curve was not fitted on.

    This is what chooses the bendiness, and it is why nobody has to. A curve too
    stiff predicts held-out cells badly because it missed real structure; one too
    floppy predicts them badly because it fitted noise that does not repeat in
    cells it never saw. Everything -- the straight line, the curve, and the target
    the error is measured against -- is fitted on the training cells alone.
    """
    error = np.zeros(block.shape[1])
    for fold in range(GENE_TRENDS_FOLDS):
        train, test = folds != fold, folds == fold
        if not train.any() or not test.any():
            continue
        basis_train = _spline_basis(rank[train], rank, n_basis)
        basis_test = _spline_basis(rank[test], rank, n_basis)
        y_train, y_test = block[train], block[test]
        gram_test = basis_test.T @ basis_test

        if covariate is None:
            coefficients = _spline_fit(basis_train, y_train)
            squared = np.asarray(y_test.multiply(y_test).sum(axis=0)).ravel()
            crossed = np.asarray(y_test.T @ basis_test, dtype=np.float64).T
        else:
            c_train, c_test = covariate[train], covariate[test]
            intercept, slope = _straight_line(y_train, c_train)
            coefficients = _spline_fit_beyond(basis_train, y_train, c_train,
                                              intercept, slope)
            # The target on the held-out cells is the same residual, written out so
            # that nothing dense is ever formed.
            sum_y = np.asarray(y_test.sum(axis=0)).ravel().astype(np.float64)
            sum_yc = np.asarray(y_test.T @ c_test, dtype=np.float64).ravel()
            squared = (np.asarray(y_test.multiply(y_test).sum(axis=0)).ravel()
                       - 2 * intercept * sum_y - 2 * slope * sum_yc
                       + intercept ** 2 * int(test.sum())
                       + 2 * intercept * slope * float(c_test.sum())
                       + slope ** 2 * float((c_test ** 2).sum()))
            crossed = (np.asarray(y_test.T @ basis_test, dtype=np.float64).T
                       - np.outer(basis_test.sum(axis=0), intercept)
                       - np.outer(basis_test.T @ c_test, slope))

        error += (squared - 2 * (coefficients * crossed).sum(axis=0)
                  + (coefficients * (gram_test @ coefficients)).sum(axis=0))
    return error


def _levels_for(n_cells):
    """The bendiness levels this many cells can support.

    Arithmetic, not a chosen floor. A fifth of the cells is held back, so a lineage
    of 30 cells fits on 24 -- and the floppiest curve on offer has 42 free numbers,
    more than there are cells to fit them with. It bites below about 50 cells and
    nowhere else.
    """
    n_train = n_cells - n_cells // GENE_TRENDS_FOLDS
    usable = tuple(n for n in GENE_TRENDS_LEVELS if n + 2 < n_train)
    return usable or (GENE_TRENDS_LEVELS[0],)


def _held_out_parts(names, n_cells):
    """Which part each cell is held back in, decided by the cell's own name.

    By name and not by place in the array, because the array is not a property of
    the cell. The bendiness of a gene's curve is settled by which level predicts
    the held-back cells best, and the levels are usually within a hair of each
    other, so which cells were held back together can affect the selected level.
    Assigning folds by name keeps assignments stable when cells are reordered or
    removed.

    A name that repeats, as a concatenated atlas often has, only puts those cells
    in the same part, which costs nothing. A dataset whose names are all one value
    would leave every other part empty, so that case uses random fold assignment.
    """
    if names is not None:
        parts = np.array([zlib.crc32(str(name).encode()) % GENE_TRENDS_FOLDS
                          for name in names])
        if len(np.unique(parts)) > 1:
            return parts
    return np.random.default_rng(1).integers(0, GENE_TRENDS_FOLDS, size=n_cells)


def _curve_for_genes(expression, rank, passing, covariate=None, names=None):
    """Each gene's curve at the bendiness its own data prefers, described.

    Args:
        expression: Cells-by-genes sparse matrix, normalised.
        rank: Per-cell position in the order.
        passing: Column indices of the genes to fit.
        covariate: Optional per-cell number held fixed, so that the curve is
            through what a straight line against it leaves of the gene.
        names: Per-cell names, which decide the part each cell is held back in.
            Without them the parts are dealt by position, which moves every cell
            behind a removed one; see :func:`_held_out_parts`.

    Returns:
        ``(described, chosen)`` -- a table describing each gene's curve and the
        selected bendiness. Curves are evaluated on the fixed analysis grid and
        are not returned or stored.
    """
    import pandas as pd
    import scipy.sparse as sp
    from concurrent.futures import ThreadPoolExecutor

    block = sp.csc_matrix(expression)[:, passing]
    folds = _held_out_parts(names, len(rank))
    fine = np.linspace(rank.min(), rank.max(), GENE_TRENDS_FINE)
    line = _straight_line(block, covariate) if covariate is not None else None
    levels = _levels_for(len(rank))

    def at_level(n_basis):
        """One bendiness level: its held-out error, and the curve it gives."""
        error = _held_out_error(block, rank, n_basis, folds, covariate)
        basis = _spline_basis(rank, rank, n_basis)
        coefficients = (_spline_fit(basis, block) if line is None
                        else _spline_fit_beyond(basis, block, covariate, *line))
        return error, _spline_basis(fine, rank, n_basis) @ coefficients

    # The levels do not depend on each other, and the work is inside NumPy and
    # SciPy, which release the interpreter's lock.
    with ThreadPoolExecutor(len(levels)) as pool:
        done = list(pool.map(at_level, levels))

    best = np.argmin(np.vstack([error for error, _ in done]), axis=0)
    stacked = np.stack([curve for _, curve in done], axis=0)
    curve = stacked[best, :, np.arange(len(passing))]
    # In standard deviations from the gene's own mean along its own curve, so zero
    # is that gene's average and every number below is comparable between genes of
    # very different abundance.
    centred = curve - curve.mean(axis=1, keepdims=True)
    curve = centred / np.maximum(centred.std(axis=1, keepdims=True), 1e-12)

    described = pd.DataFrame({
        "peak_location": fine[np.argmax(curve, axis=1)],
        "peak_height": curve.max(axis=1),
        "trough_location": fine[np.argmin(curve, axis=1)],
        "trough_depth": curve.min(axis=1),
        "starts_at": curve[:, 0],
        "ends_at": curve[:, -1],
        # How much of the order the gene spends above its own average. A gene
        # concentrated in a short stretch is small and one up across most of the
        # lineage is large -- it is the only column that tells a brief flare from a
        # gene that is simply on for most of the way.
        "share_of_pseudotime_above_average": (curve > 0).mean(axis=1),
    })
    return described, np.asarray(levels)[best]


def _cell_type_at(peaks, rank, cell_types):
    """The cell type that predominates where each gene peaks.

    Read from the same stretches the figure's strip is painted in, so a gene's
    label and the colour above its row always agree.

    This names one point on a gene's curve and nothing more. It does **not** say
    the gene is specific to that cell type, nor that it is expressed in it, nor
    that it belongs to it -- a gene runs the length of the order. Where one cell
    type fills a whole lineage every gene carries its name and the column says
    nothing at all.
    """
    import pandas as pd

    stretch_of_cell = _equal_slices(rank, GENE_TRENDS_STRETCHES)
    painted, centre = [], []
    for stretch in range(GENE_TRENDS_STRETCHES):
        here = stretch_of_cell == stretch
        if not here.any():
            continue
        painted.append(pd.Series(cell_types[here]).value_counts().index[0])
        centre.append(float(rank[here].mean()))
    if not painted:
        return np.full(len(peaks), None, dtype=object)
    nearest = np.abs(np.asarray(peaks, dtype=float)[:, None]
                     - np.asarray(centre)[None, :]).argmin(axis=1)
    return np.asarray(painted, dtype=object)[nearest]


def _gene_trend_inputs(adata):
    """The counts, the gene identities and the cell order this stage needs."""
    import pandas as pd

    from .predictor_support import resolve_expression_source

    missing = [c for c in ("hector_lineage", "hector_pseudotime")
               if c not in adata.obs.columns]
    if missing:
        raise RuntimeError(
            f"create_gene_trend_analysis requires {missing}, written by "
            "create_pseudotime_analysis. Run create_pseudotime_analysis(predictor, "
            "adata, ...) first."
        )

    # Raw counts, and nothing else: this stage scales by library size and takes
    # log1p, so a matrix that has already been logged would be transformed twice.
    resolved = resolve_expression_source(adata, allow_log1p=False)
    var = resolved["var"]
    symbols = None
    for column in ("symbol", "gene_symbol", "feature_name", "gene_name"):
        if column in var.columns:
            symbols = np.asarray(var[column]).astype(str)
            break
    genes = pd.DataFrame({
        "gene_id": np.asarray(var.index).astype(str),
        "symbol": np.asarray(var.index).astype(str) if symbols is None else symbols,
    })
    return resolved["matrix"], genes, resolved["name"]


def create_gene_trend_analysis(
    adata,
    lineages: Optional[List[str]] = None,
    false_discovery_rate: float = 0.05,
    min_cells_per_gene: int = 25,
    n_shuffles: int = GENE_TRENDS_SHUFFLES,
    hold_detected_genes_fixed: bool = True,
    max_cells_per_lineage: Optional[int] = None,
    keep_all: bool = False,
    n_jobs: Optional[int] = None,
    seed: int = 0,
):
    """Which genes change along each lineage's cell order, and in what order.

    One lineage at a time and never pooled, because a position in one lineage's
    pseudotime is not comparable with a position in another.

    Whether a gene changes comes from a walk along the cell order; where it peaks
    comes from a curve whose bendiness the data chooses. Neither has a setting.
    Both can hold fixed the count of genes each cell expresses, which is also the
    covariate used by ``create_pseudotime_analysis`` to break unresolved ties.

    Args:
        adata: AnnData already through ``create_pseudotime_analysis``. Raw counts
            must be resolvable from ``layers['counts']``, ``raw.X`` or ``X``; a
            log-normalised matrix is refused rather than logged twice.
        lineages: Restrict to these. Every lineage by default.
        false_discovery_rate: The share of the returned list that may be wrong.
        min_cells_per_gene: A gene detected in fewer cells of a lineage than this
            is not tested there.
        n_shuffles: Shuffles behind the p-values. Sets how finely a small p-value
            can be quoted, not which genes are found.
        hold_detected_genes_fixed: Off gives the unadjusted answer.
        max_cells_per_lineage: Sample a large lineage down. Every cell by default.
        keep_all: Also return the genes that did not pass, their curve columns
            empty: no curve is fitted for them, and a gene that does nothing still
            has a highest point.
        n_jobs: Shuffles at once. Left alone, the smaller of the cores this process
            was granted and six. The shuffles are drawn before any of them runs, so
            the answer does not depend on it.
        seed: Seed for the shuffles and the sampling.

    Returns:
        A ``pandas.DataFrame``, one row per gene per lineage, sorted by lineage and
        then by peak -- so a lineage's rows top to bottom are the gene ordering.
        Nothing is written onto ``adata``.

        ``lineage``, ``order_in_lineage``, ``gene_id``, ``symbol``, ``n_cells``
            which lineage, 1 for the gene peaking earliest in it, identity, and
            cells of that lineage the gene was detected in.
        ``strength``, ``p_value``, ``q_value``
            the walk's statistic and what the shuffles make of it. ``strength`` is
            signal-to-noise and not effect size: it ranks well-measured genes
            first, so cutting on it drops genes like RHO that are only just
            switching on. Use it to decide what to draw.
        ``peak_location``, ``trough_location``
            where on the 0-to-1 pseudotime the curve is highest and lowest.
        ``peak_height``, ``trough_depth``, ``starts_at``, ``ends_at``
            how far above or below the gene's own average the curve reaches there
            and at each end, in standard deviations of its own curve. ``ends_at``
            near ``starts_at`` does not mean the gene does nothing: a brief flare
            ends where it started.
        ``share_of_pseudotime_above_average``
            how much of the order the gene spends above its own average, which is
            what separates a flare from a gene simply on for most of the lineage.
        ``bendiness``
            how bendy this gene's curve was allowed to be, chosen by the data.
        ``cell_type_at_peak``
            which cell type predominates where the gene peaks -- one point on the
            curve and nothing more, not a claim that the gene is specific to that
            type. Absent without a ``trajectory_target`` column.

    Raises:
        RuntimeError: If ``create_pseudotime_analysis`` has not been run.
        ValueError: If no slot of ``adata`` holds raw counts.

    Note:
        The p-values are anti-conservative, because the pseudotime was estimated
        from the same cells. True of every tool in this family. Read the ordering
        rather than the length of the list.

        The curve stage holds a fifth of the cells back to choose each gene's
        bendiness, and which fifth a cell falls in is read off ``adata.obs_names``
        so that removing one cell does not move any other cell. Renaming the cells
        therefore moves a few peaks; reordering, subsetting or renaming the genes
        does not.
    """
    import pandas as pd
    import scipy.sparse as sp
    from concurrent.futures import ThreadPoolExecutor

    counts, genes, slot = _gene_trend_inputs(adata)
    counts = sp.csr_matrix(counts)
    lineage_of = adata.obs["hector_lineage"].astype(object).to_numpy()
    rank_of = pd.to_numeric(adata.obs["hector_pseudotime"],
                            errors="coerce").to_numpy()
    ranked = np.isfinite(rank_of)
    targets = (adata.obs["trajectory_target"].astype(str).to_numpy()
               if "trajectory_target" in adata.obs.columns else None)
    # Each cell's name decides which part of the curve sweep holds it back, so
    # that removing a cell does not move any other cell into a different part.
    cell_names = np.asarray(adata.obs_names, dtype=str)

    # Silent when it holds the quantity the ordering actually used, which is the
    # ordinary case and needs no explaining. Only the substitute is worth a word.
    ruler = None
    if hold_detected_genes_fixed:
        if "hector_detected_genes" in adata.obs.columns:
            ruler = pd.to_numeric(adata.obs["hector_detected_genes"],
                                  errors="coerce").to_numpy().astype(float)
        else:
            ruler = np.asarray((counts > 0).sum(axis=1)).ravel().astype(float)
            print(
                "\n[Gene trends] adata.obs['hector_detected_genes'] is missing, so "
                "this run counts the genes each cell expresses from the matrix "
                "instead. That is close to, but not the same as, what "
                "create_pseudotime_analysis used. Rerun it to match."
            )

    wanted = [str(name) for name in
              pd.Series(lineage_of[ranked]).value_counts().index]
    if lineages is not None:
        asked = {str(name) for name in lineages}
        unknown = sorted(asked - set(wanted))
        if unknown:
            raise ValueError(
                f"No ranked cells in {unknown}. The lineages on this object are "
                f"{wanted}."
            )
        wanted = [name for name in wanted if name in asked]

    print(f"\n[Gene trends] counts from {slot}, {len(wanted)} lineages, please wait...")

    rng = np.random.default_rng(seed)
    blocks = []
    for name in wanted:
        rows = np.where(ranked & (lineage_of == name))[0]
        if max_cells_per_lineage and len(rows) > max_cells_per_lineage:
            rows = np.sort(rng.choice(rows, max_cells_per_lineage, replace=False))
        rank = rank_of[rows]
        block = counts[rows]

        # Filtered inside the lineage: a gene nothing here expresses cannot be
        # tested here, and leaving it in only costs the others their correction.
        seen = np.asarray((block > 0).sum(axis=0)).ravel()
        testable = np.where(seen >= min_cells_per_gene)[0]
        if len(testable) == 0:
            print(f"  {str(name)[:44]:46s} {len(rows):>7,} cells   no gene reaches "
                  f"{min_cells_per_gene} cells")
            continue
        expression = _normalise_for_trends(block[:, testable])
        covariate = None if ruler is None else ruler[rows]

        workers = _gene_trend_threads(n_shuffles, n_jobs)
        chunk = _gene_trend_chunk(len(rows), workers)
        observed = _walk(expression, np.argsort(rank, kind="stable"), covariate,
                         _gene_trend_chunk(len(rows), 1))
        shuffles = [rng.permutation(len(rows)) for _ in range(n_shuffles)]

        def one(order, expression=expression, covariate=covariate, chunk=chunk):
            return _walk(expression, order, covariate, chunk)

        if workers > 1:
            with ThreadPoolExecutor(workers) as pool:
                null = np.vstack(list(pool.map(one, shuffles)))
        else:
            null = np.vstack([one(order) for order in shuffles])

        detection = seen[testable] / len(rows)
        edges = np.quantile(detection,
                            np.linspace(0, 1, GENE_TRENDS_STRATA + 1)[1:-1])
        p_value = _empirical_p(observed, null, np.searchsorted(edges, detection))
        q_value = _benjamini_hochberg(p_value)

        table = genes.iloc[testable].reset_index(drop=True).copy()
        table.insert(0, "lineage", name)
        table["n_cells"] = seen[testable]
        table["strength"] = observed
        table["p_value"] = p_value
        table["q_value"] = q_value

        passing = np.where(q_value < false_discovery_rate)[0]
        if len(passing):
            described, chosen = _curve_for_genes(expression, rank, passing,
                                                 covariate, cell_names[rows])
            table["bendiness"] = np.nan
            table.loc[table.index[passing], "bendiness"] = chosen
            for column in described.columns:
                values = np.full(len(table), np.nan)
                values[passing] = described[column].to_numpy()
                table[column] = values
            if targets is not None:
                table["cell_type_at_peak"] = pd.NA
                table.loc[table.index[passing], "cell_type_at_peak"] = \
                    _cell_type_at(described["peak_location"].to_numpy(), rank,
                                  targets[rows])
        print(f"  {str(name)[:44]:46s} {len(rows):>7,} cells   "
              f"{len(passing):>7,} of {len(testable):,} genes change", flush=True)

        blocks.append(table if keep_all else table.iloc[passing])

    if not blocks:
        raise RuntimeError(
            "No lineage had cells with a rank and genes to test. Check "
            "adata.obs['hector_pseudotime'] and lower min_cells_per_gene."
        )

    trends = pd.concat(blocks, ignore_index=True)
    if "peak_location" in trends.columns:
        trends = trends.sort_values(["lineage", "peak_location"], kind="stable")
    trends = trends.reset_index(drop=True)
    trends.insert(1, "order_in_lineage",
                  trends.groupby("lineage", observed=True).cumcount() + 1)
    print(f"\n  {len(trends):,} rows. Sorted by lineage then by where the gene "
          f"peaks, so reading a\n  lineage's rows top to bottom is the gene "
          f"ordering.")
    return trends


def _name_rows(fig, panels, font_size=5.5):
    """Name genes beside each map, once the panels' final heights are known.

    How many names fit is a question about the panel in points, not about the
    genes, so the figure is rendered once to measure. Names are then spaced by the
    height of the type: down the list first, pushing each clear of the one above,
    then back up pulling in any that overran the bottom. Where even that is not
    enough an evenly spread subset is kept, rather than a pile of unreadable names.
    """
    fig.canvas.draw()
    for panel in panels:
        axis, n_rows, symbols = panel["axis"], panel["n_rows"], panel["symbols"]
        height_points = axis.get_window_extent().height / fig.dpi * 72
        if height_points <= 0 or n_rows == 0:
            continue
        gap = n_rows * (font_size * 1.25) / height_points

        wanted = [(int(np.where(symbols == name)[0][0]), name)
                  for name in panel["labels"] if name in symbols]
        wanted.sort()
        room = int(height_points / (font_size * 1.25))
        if len(wanted) > room > 0:
            keep = np.linspace(0, len(wanted) - 1, room).round().astype(int)
            wanted = [wanted[i] for i in sorted(set(keep))]

        placed, last = [], -gap
        for row, name in wanted:
            last = max(row, last + gap)
            placed.append([row, last, name])
        for i in range(len(placed) - 1, -1, -1):
            limit = (n_rows - 1.0) if i == len(placed) - 1 else placed[i + 1][1] - gap
            placed[i][1] = max(0.0, min(placed[i][1], limit))

        edge = panel["n_columns"]
        for row, at, name in placed:
            axis.annotate(
                name, xy=(edge - 0.5, row), xytext=(edge + edge * 0.06, at),
                fontsize=font_size, va="center", ha="left", annotation_clip=False,
                arrowprops=dict(arrowstyle="-", linewidth=0.4, color="#888888",
                                shrinkA=0, shrinkB=0),
            ).set_in_layout(True)


def _trend_picture(block, rank, wanted, columns, n_points, clip):
    """Each drawn gene's curve, refitted at the bendiness the table records.

    The bendiness comes from the table, and the curve is refitted on the requested
    plotting grid rather than stored in the analysis table.
    """
    import scipy.sparse as sp

    # The scaling comes from every gene, since it is the cell's whole library, but
    # only the drawn genes' columns need transforming.
    totals = np.asarray(block.sum(axis=1)).ravel().astype(np.float64)
    totals[totals == 0] = 1.0
    drawn = sp.csc_matrix(sp.csc_matrix(block)[:, columns])
    drawn = sp.csc_matrix(sp.diags(float(np.median(totals)) / totals) @ drawn)
    drawn.data = np.log1p(drawn.data)

    read_at = np.linspace(rank.min(), rank.max(), n_points)
    picture = np.empty((len(columns), n_points))
    for level, part in wanted.groupby("bendiness"):
        which = part.index.to_numpy()
        basis = _spline_basis(rank, rank, int(level))
        coefficients = _spline_fit(basis, drawn[:, which])
        picture[which] = (_spline_basis(read_at, rank, int(level))
                          @ coefficients).T

    centred = picture - picture.mean(axis=1, keepdims=True)
    spread = np.maximum(centred.std(axis=1, keepdims=True), 1e-12)
    return np.clip(centred / spread, -clip, clip)


def plot_gene_trends(
    adata,
    table,
    output_file: str,
    max_genes: int = GENE_TRENDS_ROWS,
    q_cutoff: Optional[float] = None,
    label_genes=None,
    panels_per_row: int = 2,
    width_mm: float = GENE_TRENDS_WIDTH_MM,
    panel_height_mm: float = GENE_TRENDS_PANEL_HEIGHT_MM,
    height_mm: Optional[float] = None,
    clip: float = GENE_TRENDS_CLIP,
    n_points: int = GENE_TRENDS_POINTS,
) -> None:
    """Draw the gene trends, one panel per lineage, to a PDF.

    One row per gene, one column per point along the pseudotime, colour the fitted
    expression in standard deviations from that gene's own mean. Rows are sorted
    by ``peak_location``, the table's own column, so the picture and the numbers a
    reader has in front of them cannot disagree. Above each map, the cell types
    occupying that stretch of the order.

    The table decides which genes, their order, and each curve's smoothness; the
    object supplies the cells. Curves are refitted here rather than stored.

    Args:
        adata: The AnnData the table was computed from.
        table: What ``create_gene_trend_analysis`` returned, filtered however you
            like. Rows with no ``peak_location`` are skipped.
        output_file: The suffix decides the format. PDF output uses editable text
            where supported by the PDF consumer.
        max_genes: Rows per panel, taken by ``strength`` -- a drawing limit, not a
            threshold. Anything named in ``label_genes`` is added back regardless,
            since ``strength`` ranks well-measured genes first.
        q_cutoff: Drop rows above this false discovery rate first. ``None`` draws
            what the table holds.
        label_genes: Symbols to name beside the rows: a list applied to every
            panel, or a dict from lineage name to a list.
        panels_per_row: Panels side by side.
        width_mm: Page width. 180 is full page, 120 one and a half columns.
        panel_height_mm: Height of one map, excluding its title, strip and labels.
        height_mm: Page height. ``None`` computes it from the panels.
        clip: Standard deviations beyond which the colour stops changing, so one
            extreme gene cannot flatten every other.
        n_points: Points along the pseudotime each curve is read at.

    Raises:
        KeyError: If the table is missing columns the analysis writes.
        ValueError: If the table names lineages the object does not have, which is
            what a table computed before the lineages changed looks like.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib is required to draw. Install matplotlib.")

    import matplotlib.pyplot as plt
    import pandas as pd
    import scipy.sparse as sp
    from matplotlib.colors import ListedColormap

    needed = ["lineage", "symbol", "gene_id", "strength", "peak_location",
              "bendiness"]
    missing = [c for c in needed if c not in table.columns]
    if missing:
        raise KeyError(
            f"The table is missing {missing}. Pass what "
            "create_gene_trend_analysis returned."
        )
    if q_cutoff is not None and "q_value" in table.columns:
        table = table[table["q_value"] < q_cutoff]
    table = table[table["peak_location"].notna() & table["bendiness"].notna()]

    counts, genes, _ = _gene_trend_inputs(adata)
    counts = sp.csr_matrix(counts)
    column_of = {name: i for i, name in enumerate(genes["gene_id"])}
    lineage_of = adata.obs["hector_lineage"].astype(object).to_numpy()
    rank_of = pd.to_numeric(adata.obs["hector_pseudotime"],
                            errors="coerce").to_numpy()
    targets = (adata.obs["trajectory_target"].astype(str).to_numpy()
               if "trajectory_target" in adata.obs.columns else None)

    on_object = {str(name) for name in pd.unique(lineage_of[np.isfinite(rank_of)])}
    drawn_lineages = [name for name in
                      pd.Series(lineage_of[np.isfinite(rank_of)]).value_counts().index
                      if str(name) in set(table["lineage"].astype(str))]
    stale = sorted(set(table["lineage"].astype(str)) - on_object)
    if stale:
        raise ValueError(
            f"The table names lineages this object does not have: {stale}. It was "
            "computed before create_pseudotime_analysis last ran, so its rows do "
            "not describe these cells. Rerun create_gene_trend_analysis."
        )
    if not drawn_lineages:
        raise ValueError("No lineage in the table has ranked cells on this object.")

    mm = 1 / 25.4
    columns = max(1, int(panels_per_row))
    n_rows = int(np.ceil(len(drawn_lineages) / columns))
    left_mm, right_mm, names_mm, col_gap_mm = 13.0, 4.0, 17.0, 9.0
    panel_w_mm = max(
        10.0,
        (width_mm - left_mm - right_mm - columns * names_mm
         - (columns - 1) * col_gap_mm) / columns)
    title_mm, strip_mm, key_mm, gap_mm = 5.0, 2.4, 8.0, 3.0
    xlabel_mm, row_gap_mm, top_mm, bar_mm = 9.0, 3.0, 4.0, 20.0
    row_pitch_mm = (title_mm + strip_mm + key_mm + gap_mm + panel_height_mm
                    + xlabel_mm + row_gap_mm)
    spare = len(drawn_lineages) % columns
    page_mm = height_mm or (top_mm + n_rows * row_pitch_mm
                            + (0.0 if spare else bar_mm))

    # Placed by hand in figure coordinates rather than by a layout engine, as
    # plot_pseudotime is and for the same reason: with this many axes competing
    # with a key and a colour bar the engine collapses whole panels to nothing,
    # which is silent in the file and visible only in the picture.
    fig = plt.figure(figsize=(width_mm * mm, page_mm * mm), layout="none")
    panels = []
    for i, name in enumerate(drawn_lineages):
        rows = np.where(np.isfinite(rank_of) & (lineage_of == name))[0]
        rank = rank_of[rows]
        here = table[table["lineage"].astype(str) == str(name)]
        labels = (label_genes.get(name, []) if isinstance(label_genes, dict)
                  else list(label_genes or []))
        wanted = pd.concat([
            here.nlargest(max_genes, "strength"),
            here[here["symbol"].isin(labels)],
        ]).drop_duplicates("gene_id").sort_values("peak_location", kind="stable")
        wanted = wanted[wanted["gene_id"].isin(column_of)].reset_index(drop=True)
        if not len(wanted):
            continue
        picture = _trend_picture(
            counts[rows], rank, wanted,
            np.array([column_of[g] for g in wanted["gene_id"]]), n_points, clip)

        r, c = divmod(len(panels), columns)
        left = (left_mm + c * (panel_w_mm + names_mm + col_gap_mm)) / width_mm
        strip_bottom = top_mm + r * row_pitch_mm + title_mm + strip_mm
        map_bottom = strip_bottom + key_mm + gap_mm + panel_height_mm
        strip_axis = fig.add_axes([left, 1 - strip_bottom / page_mm,
                                   panel_w_mm / width_mm, strip_mm / page_mm])
        main_axis = fig.add_axes([left, 1 - map_bottom / page_mm,
                                  panel_w_mm / width_mm,
                                  panel_height_mm / page_mm])

        title = str(name) if len(str(name)) <= 34 else str(name)[:32] + "…"
        strip_axis.set_title(title, fontsize=7.5, pad=4)
        strip_axis.text(-0.16, 3.4, "abcdefghijklmnop"[len(panels)],
                        transform=strip_axis.transAxes, fontsize=12,
                        fontweight="bold", va="top", ha="left")

        if targets is not None:
            stretch_of_cell = _equal_slices(rank, GENE_TRENDS_STRETCHES)
            counted = pd.Series(targets[rows]).value_counts()
            named = counted.index.tolist()[:GENE_TRENDS_KEY_ENTRIES]
            colour_of = {n: GENE_TRENDS_STRIP_COLOURS[i]
                         for i, n in enumerate(named)}
            strip = []
            for stretch in range(GENE_TRENDS_STRETCHES):
                mask = stretch_of_cell == stretch
                top = (pd.Series(targets[rows][mask]).value_counts().index[0]
                       if mask.any() else None)
                strip.append(colour_of.get(top, "#DDDDDD"))
            strip_axis.imshow([list(range(GENE_TRENDS_STRETCHES))], aspect="auto",
                              cmap=ListedColormap(strip))
            entries = [plt.Line2D([], [], marker="s", linestyle="none",
                                  markersize=3.4, color=colour_of[n],
                                  label=n if len(n) <= 22 else n[:20] + "…")
                       for n in named]
            if len(counted) > len(named):
                entries.append(plt.Line2D(
                    [], [], marker="s", linestyle="none", markersize=3.4,
                    color="#DDDDDD",
                    label=f"{len(counted) - len(named)} further cell types"))
            key = strip_axis.legend(handles=entries, loc="upper left",
                                    bbox_to_anchor=(0, -0.9), fontsize=5, ncol=2,
                                    handletextpad=0.35, columnspacing=0.7,
                                    borderpad=0, labelspacing=0.22)
            key.set_in_layout(False)
        strip_axis.set_xticks([])
        strip_axis.set_yticks([])
        for side in strip_axis.spines.values():
            side.set_visible(False)

        # Smooth rather than blocky: the curve does not step, and drawing its
        # samples as hard columns would suggest it does.
        image = main_axis.imshow(picture, aspect="auto", cmap="RdBu_r", vmin=-clip,
                                 vmax=clip, interpolation="bilinear")
        main_axis.set_xticks(np.linspace(0, n_points - 1, 3))
        main_axis.set_xticklabels(["0.00", "0.50", "1.00"])
        main_axis.set_xlabel("Pseudotime", fontsize=7.5)
        if c == 0:
            main_axis.set_ylabel(f"{len(picture):,} genes", fontsize=7)
        main_axis.set_yticks([])
        for side in main_axis.spines.values():
            side.set_visible(False)
        panels.append({"image": image, "axis": main_axis,
                       "symbols": wanted["symbol"].to_numpy(), "labels": labels,
                       "n_rows": len(picture), "n_columns": n_points})

    if not panels:
        plt.close(fig)
        raise ValueError("No gene in the table matches a gene on this object.")

    # In the empty half of the last row when there is one, and below everything
    # when the panels fill it.
    if spare:
        r, c = n_rows - 1, spare
        bar_left = (left_mm + c * (panel_w_mm + names_mm + col_gap_mm)) / width_mm
        bar_bottom = top_mm + r * row_pitch_mm + title_mm + strip_mm + key_mm \
            + gap_mm + panel_height_mm * 0.5
        bar_axis = fig.add_axes([bar_left + 0.02, 1 - bar_bottom / page_mm,
                                 panel_w_mm * 0.75 / width_mm, 2.0 / page_mm])
    else:
        bar_axis = fig.add_axes([0.55, 1 - (page_mm - 12.0) / page_mm, 0.24,
                                 2.0 / page_mm])
    bar = fig.colorbar(panels[0]["image"], cax=bar_axis, orientation="horizontal",
                       ticks=[-clip, 0, clip])
    bar.ax.set_xlabel("Fitted expression, in standard deviations\n"
                      "from each gene's own mean", fontsize=6, labelpad=2)
    bar.ax.tick_params(labelsize=6, pad=1)

    with _IllustratorSafe():
        _name_rows(fig, panels)
        fig.savefig(output_file, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  {len(panels)} lineages drawn, curves refitted to draw them")
    print(f"  wrote {output_file}")
    return None


# =============================================================================
# Adaptive Lineage Color Mapper (LCA-Based Dynamic Coloring)
# =============================================================================

# Visually distinct colors for adaptive lineage coloring
# Ordered by visual prominence for deterministic assignment


# =============================================================================
# Public API: simplify_ontology_tree()
# =============================================================================
