"""Regression checks for the runtime/training package split."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_runtime_package_excludes_retraining_modules():
    """Keep model retraining code out of the default package source tree."""
    assert not (ROOT / "pvcracks" / "utils" / "train_functions.py").exists()
    assert not (ROOT / "pvcracks" / "vae" / "VAE_training_1CH.py").exists()
    assert not (
        ROOT / "pvcracks" / "MultiSolSegment" / "training" / "train_channeled_unet_cv_hpo.py"
    ).exists()
    assert (ROOT / "training" / "pyproject.toml").exists()


def test_documentation_notebooks_use_runtime_inference_apis():
    """The published notebooks must not depend on optional training modules."""
    notebook_paths = [
        ROOT / "docs" / "Examples" / "example_VAE_rapidELprocessing.ipynb",
        ROOT / "docs" / "Examples" / "example_multisolsegment_cellELs.ipynb",
        ROOT / "docs" / "Examples" / "example_multisolsegment_test_images.ipynb",
        ROOT / "docs" / "Examples" / "example_download_run_multisolsegment.ipynb",
        ROOT / "docs" / "Examples" / "example_download_run_multisolsegment_benchmarking.ipynb",
    ]
    combined_source = "\n".join(notebook_source(path) for path in notebook_paths)
    assert "train_functions" not in combined_source
    assert "pvcracks.utils import model_io" in combined_source
    assert "pvcracks.vae.inference" in notebook_source(notebook_paths[0])


def test_readthedocs_configuration_builds_source_without_runtime_install():
    """Read the Docs uses source imports plus mocked heavyweight dependencies."""
    config = (ROOT / ".readthedocs.yaml").read_text(encoding="utf-8")
    requirements = (ROOT / "docs" / "requirements.txt").read_text(encoding="utf-8")
    assert "configuration: docs/conf.py" in config
    assert "path: ." not in config
    assert "-r ../requirements.txt" not in requirements
