# Model performance

How well each model recognises the cell types it was trained on: 650 for the
human model and 340 for the mouse model. Each box is a cell type from the Cell
Ontology, coloured by its F1 score on cells held out of training. Click a box to
open it, and point at one for its scores and cell counts.

These are the trained cell types only. HECTOR also makes zero-shot predictions,
so it can assign cell types it never saw in training; those have no held-out
test cells and are not drawn here.

```{raw} html
<div class="model-view" data-page="performance" data-tall="1.2" data-extra="190">
<div class="model-switch" role="group" aria-label="Model">
<button type="button" data-model="human" aria-pressed="true">Human</button>
<button type="button" data-model="mouse" aria-pressed="false">Mouse</button>
<a class="model-open" target="_blank" rel="noopener" href="../_static/models/human_performance.html">Open full page</a>
</div>
<div class="model-frame-box"><iframe class="model-frame" title="Model performance by cell type" src="../_static/models/human_performance.html"></iframe></div>
</div>
```
