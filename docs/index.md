# HECTOR

Hierarchical Embedding and Contrastive Training for Ontology-guided Recognition
and Trajectory Analysis of single-cell sequencing data.

HECTOR annotates single cells with cell types from the Cell Ontology, then
places them on the ontology tree to compare groups, simplify labels, and order
cells along developmental lineages.

**New here?** [Install the package](installation.md), then work through the
tutorials in order. Each builds on the one before it, and every one uses a
public dataset it downloads itself.

## Tutorials

1. [Basic prediction](tutorials/01_basic_prediction.ipynb) — annotate cell
   types, starting from the raw output of a sequencing run.
2. [Tree visualization](tutorials/02_trajectory_visualization.ipynb) — draw
   your annotated cells on the cell ontology, in two layouts.
3. [Comparing groups on one tree](tutorials/03_trajectory_comparison.ipynb) —
   the same tree once per group, every cell type pinned to the same place.
4. [Fewer, broader labels](tutorials/04_ontology_simplification.ipynb) —
   replace fine-grained labels with a smaller set of broader ones.
5. [The genes behind a call](tutorials/05_IG_markers.ipynb) — which genes
   HECTOR used to decide what each cell is.
6. [Ordering cells along a lineage](tutorials/06_pseudotime.ipynb) — a single
   order of cells across a whole developmental lineage.
7. [The genes that change along a lineage](tutorials/07_gene_trends.ipynb) —
   which genes change along that order, and when.

```{toctree}
:hidden:
:caption: Getting started

installation
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/01_basic_prediction
tutorials/02_trajectory_visualization
tutorials/03_trajectory_comparison
tutorials/04_ontology_simplification
tutorials/05_IG_markers
tutorials/06_pseudotime
tutorials/07_gene_trends
```

```{toctree}
:hidden:
:caption: Project

Source code <https://github.com/Polligator/HECTOR>
```
