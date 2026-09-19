"""HECTOR public package API.

On non-MLX backends, TensorFlow is imported eagerly at ``import hector`` time
so subsequent model loads do not pause for TensorFlow initialization. On macOS,
TensorFlow is skipped when MLX is available. Other heavy dependencies (SciPy,
NetworkX, Plotly, and Matplotlib) remain lazy and are loaded when first used.

For the TensorFlow backend, GPU memory growth is enabled by default
(``TF_FORCE_GPU_ALLOW_GROWTH=true``) so memory remains available to cuML and
cugraph stages of ``HECTOR.evaluate_cells``. Opt out with
``HECTOR_TF_MEMORY_GROWTH=false``.

TF32 tensor-core matmuls are disabled by default so fp32 runs at true fp32
precision and produces consistent encoder and cosine k-NN calculations across
supported backends. Opt back into TF32 speed with ``HECTOR_TF32=1``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

__version__ = "1.0"

__all__ = [
    "HECTOR",
    "InferenceConfig",
    "predict_from_anndata",
    "load_cellranger_data",
    "detect_doublets",
    "diagnose_cellranger_directory",
    "filter_cells",
    "map_gene_symbols",
    "HUMAN_MITO_GENE_IDS",
    "MOUSE_MITO_GENE_IDS",
    "HierarchicalTrajectoryAnalyzer",
    "TrajectoryAnalysisConfig",
    "create_trajectory_analysis",
    "create_trajectory_comparison",
    "create_pseudotime_analysis",
    "plot_pseudotime",
    "create_gene_trend_analysis",
    "plot_gene_trends",
    "simplify_ontology_tree",
    "plot_simplified_tree",
    "discover_inferred_paths",
    "VisualizationCache",
    "compute_elastic_depths",
    "extract_visualization_data",
    "download_model",
    "get_cache_dir",
    "get_model_info",
    "get_model_path",
    "read_checkpoint_version",
    "check_for_update",
    "import_model_file",
    "list_available_models",
    "clear_cache",
    "plot_obs_label_density",
]

DEFAULT_CACHE_ENV_VAR = "HECTOR_MODEL_CACHE"

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "human": {
        "repo_id": "polligator/HECTOR",
        "filename": "human.h5",
        "revision": "main",
        "sha256": None,
        "description": "Public human HECTOR checkpoint",
        "species": "Homo sapiens",
    },
    "mouse": {
        "repo_id": "polligator/HECTOR",
        "filename": "mouse.h5",
        "revision": "main",
        "sha256": None,
        "description": "Public mouse HECTOR checkpoint",
        "species": "Mus musculus",
    },
}


def get_cache_dir(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        resolved_cache_dir = Path(cache_dir).expanduser()
    else:
        resolved_cache_dir = Path(
            os.environ.get(DEFAULT_CACHE_ENV_VAR, Path.home() / ".cache" / "hector" / "models")
        ).expanduser()

    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    return resolved_cache_dir.resolve()


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_available_models() -> dict[str, dict[str, Any]]:
    return {
        model_name: dict(model_info)
        for model_name, model_info in MODEL_REGISTRY.items()
    }


def get_model_info(model_name: str) -> dict[str, Any]:
    normalized_name = model_name.strip().lower()
    if normalized_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown HECTOR model '{model_name}'. Available models: {available}")
    return dict(MODEL_REGISTRY[normalized_name])


def _metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".json")


def _resolve_cached_model_path(model_name: str, cache_dir: str | Path | None = None) -> Path:
    model_info = get_model_info(model_name)
    species_cache_dir = get_cache_dir(cache_dir) / model_name.strip().lower()
    species_cache_dir.mkdir(parents=True, exist_ok=True)
    return (species_cache_dir / model_info["filename"]).resolve()


def _write_download_metadata(
    model_name: str,
    model_path: Path,
    *,
    commit_sha: str | None = None,
    checkpoint_version: str | None = None,
) -> None:
    model_info = get_model_info(model_name)
    metadata = {
        "model_name": model_name,
        "repo_id": model_info["repo_id"],
        "filename": model_info["filename"],
        "revision": model_info.get("revision", "main"),
        "commit_sha": commit_sha,
        "checkpoint_version": checkpoint_version,
    }
    _metadata_path(model_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _validate_checksum(model_path: Path, expected_sha256: str | None) -> None:
    if not expected_sha256:
        return

    observed_sha256 = compute_sha256(model_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for '{model_path}'. "
            f"Expected {expected_sha256}, observed {observed_sha256}."
        )


def read_checkpoint_version(model_path: str | Path) -> str | None:
    """Read checkpoint_version from an HDF5 model file without loading the full model."""
    import h5py
    try:
        with h5py.File(str(model_path), "r") as f:
            version = f.attrs.get("checkpoint_version", None)
            if isinstance(version, bytes):
                version = version.decode("utf-8")
            return version
    except Exception:
        return None


def _get_hf_hub_download():
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required for HECTOR model auto-download. "
            "Install the package dependencies or `pip install huggingface-hub`."
        ) from exc
    return hf_hub_download


def download_model(
    model_name: str,
    *,
    force_download: bool = False,
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> Path:
    normalized_name = model_name.strip().lower()
    model_info = get_model_info(normalized_name)
    cached_model_path = _resolve_cached_model_path(normalized_name, cache_dir=cache_dir)
    hf_hub_download = _get_hf_hub_download()

    hf_hub_download(
        repo_id=model_info["repo_id"],
        filename=model_info["filename"],
        revision=model_info.get("revision", "main"),
        repo_type="model",
        local_dir=str(cached_model_path.parent),
        force_download=force_download,
        token=token,
    )

    if not cached_model_path.exists():
        raise FileNotFoundError(
            f"Downloaded model for '{normalized_name}' was not found at {cached_model_path}."
        )

    _validate_checksum(cached_model_path, model_info.get("sha256"))

    commit_sha = None
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_info["repo_id"])
        commit_sha = info.sha
    except Exception:
        pass

    ckpt_version = read_checkpoint_version(cached_model_path)

    _write_download_metadata(
        normalized_name,
        cached_model_path,
        commit_sha=commit_sha,
        checkpoint_version=ckpt_version,
    )
    return cached_model_path


_UPDATE_CHECK_DONE: set[str] = set()


def _background_update_check(
    model_name: str,
    *,
    cache_dir: str | Path | None = None,
) -> None:
    """Fire-and-forget update check — runs once per model per process."""
    if model_name in _UPDATE_CHECK_DONE:
        return
    _UPDATE_CHECK_DONE.add(model_name)

    if os.environ.get("HECTOR_NO_UPDATE_CHECK", "").lower() in ("1", "true", "yes"):
        return

    import threading

    def _check() -> None:
        try:
            result = check_for_update(model_name, cache_dir=cache_dir, timeout=5.0)
            if result["update_available"]:
                local = result.get("local_version") or "unknown"
                print(
                    f"\n  ❇️  A newer HECTOR model checkpoint is available "
                    f"(local version: {local}).\n"
                    f"     Run:  hector.download_model(\"{model_name}\", "
                    f"force_download=True)\n"
                )
        except Exception:
            pass

    threading.Thread(target=_check, daemon=True).start()


def get_model_path(
    model_name_or_path: str | Path,
    *,
    auto_download: bool = True,
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> Path:
    if isinstance(model_name_or_path, Path):
        candidate_path = model_name_or_path.expanduser().resolve()
        if not candidate_path.exists():
            raise FileNotFoundError(f"Model checkpoint was not found at {candidate_path}.")
        return candidate_path

    normalized_name = model_name_or_path.strip().lower()
    if normalized_name in MODEL_REGISTRY:
        cached_model_path = _resolve_cached_model_path(normalized_name, cache_dir=cache_dir)
        if cached_model_path.exists():
            _validate_checksum(cached_model_path, MODEL_REGISTRY[normalized_name].get("sha256"))
            _background_update_check(normalized_name, cache_dir=cache_dir)
            return cached_model_path
        if not auto_download:
            raise FileNotFoundError(
                f"Model '{normalized_name}' is not present in the local cache at {cached_model_path}."
            )
        return download_model(
            normalized_name,
            force_download=False,
            cache_dir=cache_dir,
            token=token,
        )

    candidate_path = Path(model_name_or_path).expanduser().resolve()
    if not candidate_path.exists():
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise FileNotFoundError(
            f"Model checkpoint was not found at {candidate_path}. "
            f"If you intended to use a registry entry, available models are: {available}."
        )
    return candidate_path


def clear_cache(
    model_name: str | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> None:
    root_cache_dir = get_cache_dir(cache_dir)

    if model_name is None:
        shutil.rmtree(root_cache_dir, ignore_errors=True)
        root_cache_dir.mkdir(parents=True, exist_ok=True)
        return

    model_cache_dir = root_cache_dir / model_name.strip().lower()
    if model_cache_dir.exists():
        shutil.rmtree(model_cache_dir)


def check_for_update(
    model_name: str = "human",
    *,
    cache_dir: str | Path | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Check whether a newer published model is available on Hugging Face.

    A differing remote commit identifies an update because published model
    revisions are advanced on Hugging Face when a new checkpoint is uploaded.

    Returns a dict with keys:
        update_available (bool)
        local_version (str | None) — checkpoint_version of the cached model
        local_commit_sha (str | None)
        remote_commit_sha (str | None)
    """
    model_info = get_model_info(model_name)
    cached_model_path = _resolve_cached_model_path(model_name, cache_dir=cache_dir)
    meta_path = _metadata_path(cached_model_path)

    local_commit_sha = None
    local_version = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            local_commit_sha = meta.get("commit_sha")
            local_version = meta.get("checkpoint_version")
        except Exception:
            pass

    if not local_version and cached_model_path.exists():
        local_version = read_checkpoint_version(cached_model_path)

    from huggingface_hub import HfApi
    api = HfApi()
    info = api.model_info(model_info["repo_id"], timeout=timeout)
    remote_commit_sha = info.sha

    update_available = (
        remote_commit_sha is not None
        and local_commit_sha is not None
        and remote_commit_sha != local_commit_sha
    )

    return {
        "update_available": update_available,
        "local_version": local_version,
        "local_commit_sha": local_commit_sha,
        "remote_commit_sha": remote_commit_sha,
    }


def import_model_file(
    source_path: str | Path,
    model_name: str = "human",
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Import a user-provided .h5 model file into the local cache.

    Returns a dict with:
        checkpoint_version (str | None)
        cache_path (str)
    """
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source model file not found: {source_path}")

    cached_model_path = _resolve_cached_model_path(model_name, cache_dir=cache_dir)

    if source_path != cached_model_path:
        shutil.copy2(source_path, cached_model_path)

    ckpt_version = read_checkpoint_version(cached_model_path)

    _write_download_metadata(
        model_name,
        cached_model_path,
        commit_sha=None,
        checkpoint_version=ckpt_version,
    )

    return {
        "checkpoint_version": ckpt_version,
        "cache_path": str(cached_model_path),
    }


# ---------------------------------------------------------------------------
# Select MLX on macOS when available. Otherwise import TensorFlow eagerly so it
# is initialized before model loading. TensorFlow GPU memory growth remains on
# by default so cuML/cugraph stages can share device memory. User-provided
# environment values take precedence.
# ---------------------------------------------------------------------------
import sys as _sys
if _sys.platform == 'darwin':
    try:
        import mlx.core  # noqa: F401
        _HECTOR_USE_MLX = True
    except ImportError:
        _HECTOR_USE_MLX = False
else:
    _HECTOR_USE_MLX = False

if not _HECTOR_USE_MLX:
    if os.environ.get("HECTOR_TF_MEMORY_GROWTH", "true").lower() != "false":
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf  # noqa: F401

    # Disable TF32 tensor-core matmuls by default to keep fp32 encoder and cosine
    # calculations consistent across supported backends. Set HECTOR_TF32=1 to
    # opt into TF32 execution.
    if os.environ.get("HECTOR_TF32", "0").lower() not in ("1", "true"):
        try:
            tf.config.experimental.enable_tensor_float_32_execution(False)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Lazy-import machinery
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    """Module-level __getattr__ for deferred imports (PEP 562)."""

    # -----------------------------------------------------------------------
    # predictor.py — pulls in TensorFlow, Keras, h5py, anndata, pandas, …
    # -----------------------------------------------------------------------
    if name in ("HECTOR", "InferenceConfig", "predict_from_anndata"):
        from .predictor import HECTOR, InferenceConfig, predict_from_anndata
        _inject_globals(
            HECTOR=HECTOR,
            InferenceConfig=InferenceConfig,
            predict_from_anndata=predict_from_anndata,
        )
        return globals()[name]

    # -----------------------------------------------------------------------
    # predictor_support.py — Cell Ranger I/O helpers, cell QC filter
    # -----------------------------------------------------------------------
    if name in ("detect_doublets", "load_cellranger_data",
                "diagnose_cellranger_directory",
                "filter_cells", "map_gene_symbols",
                "HUMAN_MITO_GENE_IDS", "MOUSE_MITO_GENE_IDS"):
        from .predictor_support import (
            HUMAN_MITO_GENE_IDS,
            MOUSE_MITO_GENE_IDS,
            detect_doublets,
            diagnose_cellranger_directory,
            filter_cells,
            load_cellranger_data,
            map_gene_symbols,
        )
        _inject_globals(
            HUMAN_MITO_GENE_IDS=HUMAN_MITO_GENE_IDS,
            MOUSE_MITO_GENE_IDS=MOUSE_MITO_GENE_IDS,
            detect_doublets=detect_doublets,
            diagnose_cellranger_directory=diagnose_cellranger_directory,
            filter_cells=filter_cells,
            load_cellranger_data=load_cellranger_data,
            map_gene_symbols=map_gene_symbols,
        )
        return globals()[name]

    # -----------------------------------------------------------------------
    # trajectory.py — public API orchestration
    # -----------------------------------------------------------------------
    if name in (
        "HierarchicalTrajectoryAnalyzer",
        "create_trajectory_analysis",
        "create_trajectory_comparison",
        "create_pseudotime_analysis",
        "plot_pseudotime",
        "create_gene_trend_analysis",
        "plot_gene_trends",
    ):
        from .trajectory import (
            HierarchicalTrajectoryAnalyzer,
            create_trajectory_analysis,
            create_trajectory_comparison,
            create_pseudotime_analysis,
            plot_pseudotime,
            create_gene_trend_analysis,
            plot_gene_trends,
        )
        _inject_globals(
            HierarchicalTrajectoryAnalyzer=HierarchicalTrajectoryAnalyzer,
            create_trajectory_analysis=create_trajectory_analysis,
            create_trajectory_comparison=create_trajectory_comparison,
            create_pseudotime_analysis=create_pseudotime_analysis,
            plot_pseudotime=plot_pseudotime,
            create_gene_trend_analysis=create_gene_trend_analysis,
            plot_gene_trends=plot_gene_trends,
        )
        return globals()[name]

    # -----------------------------------------------------------------------
    # trajectory_ontology.py — config, layout engines, color, simplification
    # -----------------------------------------------------------------------
    if name in (
        "TrajectoryAnalysisConfig",
        "VisualizationCache",
        "discover_inferred_paths",
        "compute_elastic_depths",
        "extract_visualization_data",
        "simplify_ontology_tree",
        "plot_simplified_tree",
    ):
        from .trajectory_ontology import (
            TrajectoryAnalysisConfig,
            VisualizationCache,
            compute_elastic_depths,
            discover_inferred_paths,
            extract_visualization_data,
            plot_simplified_tree,
            simplify_ontology_tree,
        )
        _inject_globals(
            TrajectoryAnalysisConfig=TrajectoryAnalysisConfig,
            VisualizationCache=VisualizationCache,
            compute_elastic_depths=compute_elastic_depths,
            discover_inferred_paths=discover_inferred_paths,
            extract_visualization_data=extract_visualization_data,
            plot_simplified_tree=plot_simplified_tree,
            simplify_ontology_tree=simplify_ontology_tree,
        )
        return globals()[name]

    # -----------------------------------------------------------------------
    # trajectory_render.py — standalone diagnostic plot helpers
    # -----------------------------------------------------------------------
    if name == "plot_obs_label_density":
        from .trajectory_render import plot_obs_label_density
        _inject_globals(
            plot_obs_label_density=plot_obs_label_density,
        )
        return globals()[name]

    raise AttributeError(f"module 'hector' has no attribute {name!r}")


def _inject_globals(**kwargs) -> None:
    """Cache resolved names into the module's global namespace.

    This ensures subsequent accesses (e.g. a second ``hector.HECTOR`` call)
    hit the module dict directly and bypass ``__getattr__`` entirely.
    """
    import sys
    module = sys.modules[__name__]
    for attr, value in kwargs.items():
        setattr(module, attr, value)
