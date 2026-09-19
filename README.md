# HECTOR
Hierarchical Embedding and Contrastive Training for Ontology-guided Recognition and Trajectory Analysis of single-cell sequencing data

HECTOR packages the current HECTOR inference and trajectory analysis code as a Python distribution. The package source is under `hector/`.

**Prefer a graphical interface?** [HECTOR Desktop](https://github.com/Polligator/HECTOR-Desktop) lets you load data, run predictions, and explore results without using the command line. Download the macOS or Windows application from its [Releases page](https://github.com/Polligator/HECTOR-Desktop/releases).

## Installation

On Linux or WSL2 with an NVIDIA graphics card:

```bash
pip install "hector-sc[cuda12]"
```

On everything else — Mac, Windows, or Linux without an NVIDIA card:

```bash
pip install hector-sc
```

Either command installs the complete feature set; there are no other optional
extras to choose from. HECTOR selects its inference backend automatically:
TensorFlow on Linux, Windows, and Intel Macs, or MLX on Apple Silicon Macs.

The `cuda12` extra adds TensorFlow's pip-managed NVIDIA CUDA runtime libraries,
the prebuilt `cupy-cuda12x` wheel used for live VRAM measurement and CUDA
memory-pool management, and the RAPIDS libraries `cuml-cu12` and `cugraph-cu12`,
which accelerate `reduce_dimensions()` and `evaluate_cells()`. Every package in
the extra is gated to Linux, so on Mac and Windows the bracketed form installs
nothing extra and is equivalent to the plain command.

## Model Checkpoints

The packaged registry exposes two model entries hosted in Hugging Face repo
`polligator/HECTOR`. Access to the hosted checkpoints is managed separately on Hugging Face.

- `human` -> file `human.h5` (*Homo sapiens*)
- `mouse` -> file `mouse.h5` (*Mus musculus*)

Basic usage:

```python
import anndata
import hector

adata = anndata.read_h5ad("example_data.h5ad")
predictor = hector.HECTOR("human") # model will automatically download
predictions = predictor.predict(adata)
predictor.write_predictions(adata, predictions)
```


## Hardware Requirements

HECTOR automatically selects an available inference backend and adjusts batch
sizes to the available memory. CPU inference is used where supported when no
accelerator is available. Results are expected to be numerically stable for
fixed inputs and settings, although small differences can occur across
hardware, backends, batch sizes, or row orderings.

| Resource | Guidance |
| --- | --- |
| GPU VRAM | **10 GB or more is recommended** for GPU inference with the published human checkpoint; adaptive batching may permit smaller workloads on devices with less memory. |
| GPU VRAM (comfortable) | **16 GB** provides additional headroom for larger workloads. |
| System RAM | Workload-dependent. Approximately **16 GB** is a useful starting point for the published human checkpoint with a 100,000-cell anchor pool. |
| CPU-only mode | Supported where the selected backend provides CPU execution, but generally slower than accelerated inference. |

These estimates use the published human checkpoint (approximately 57 million
parameters) with a 100,000-cell by 5,000-gene anchor pool. Smaller anchor pools
or fewer genes generally reduce memory requirements, but memory use also
includes fixed model and runtime overhead.


## Citation

```yaml
title: "A shared ontology representation of cell identity and development"
version: "1.0"
license: "Apache-2.0"
repository-code: "https://github.com/Polligator/HECTOR"
```

## License

The source code is released under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) for details.

The project name, logo, icons, and related branding are not licensed under
Apache-2.0. Modified versions should not use the project name or branding in a
way that suggests they are official, endorsed by, or affiliated with the
original project.

## Contributions

External code contributions are not currently accepted.

Bug reports, reproducibility reports, installation issues, and feature suggestions are welcome through GitHub Issues.

This policy helps keep copyright ownership clear while the project is under active research and development.
