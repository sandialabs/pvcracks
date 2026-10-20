"""Execute the current model examples in real kernels from an isolated checkout."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import nbformat
import pytest
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    "docs/Examples/example_load_3CH-VAE.ipynb",
    "docs/Examples/example_xgboost.ipynb",
    "docs/Examples/example_validate_pvimage.ipynb",
    "docs/Examples/example_generate_images.ipynb",
    "docs/Examples/example_VAE_rapidELprocessing.ipynb",
    "training/notebooks/train_xgboost.ipynb",
)


@pytest.fixture(scope="module")
def example_checkout(tmp_path_factory):
    """Copy only repo-owned inputs, excluding the original outputs and Box files."""
    checkout = tmp_path_factory.mktemp("notebooks") / "checkout with spaces"
    shutil.copytree(
        ROOT / "pvcracks",
        checkout / "pvcracks",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        ROOT / "training/src",
        checkout / "training/src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / "training/pyproject.toml", checkout / "training/pyproject.toml")
    data = checkout / "docs/data"
    data.mkdir(parents=True)
    for name in (
        "autoencoder_3ch_module_209.npz",
        "autoencoder_3ch_module_209.csv",
        "autoencoder_3ch_reference.csv",
    ):
        shutil.copy2(ROOT / "docs/data" / name, data / name)
    for example in EXAMPLES:
        path = checkout / example
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / example, path)
    return checkout


@pytest.mark.parametrize("example", EXAMPLES)
@pytest.mark.parametrize("launch_directory", [".", "notebook_directory"])
def test_model_example_paths(example_checkout, example, launch_directory):
    """Read inputs, load bundled weights, and write outputs without source paths."""
    path = example_checkout / example
    cwd = example_checkout if launch_directory == "." else path.parent
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    # A clean kernel imports the copied runtime, not an editable install from
    # the original workspace. Limit CPU threads for predictable CI runtimes.
    notebook.cells.insert(
        0, nbformat.v4.new_code_cell("import torch\ntorch.set_num_threads(4)")
    )
    expected_root = repr(str(example_checkout.resolve()))
    assertions = (
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pvcracks\n"
        f"assert ROOT.resolve() == Path({expected_root})\n"
        "assert Path(pvcracks.__file__).resolve() == ROOT / 'pvcracks/__init__.py'\n"
    )
    if example.startswith("docs/"):
        assertions += (
            "assert DATA.resolve() == ROOT / 'docs/data'\n"
            "assert len(masks) == len(metadata) == 26\n"
            "assert checkpoint['base_channels'] == 20\n"
        )
    if example.endswith("example_xgboost.ipynb"):
        assertions += (
            "assert predictions.shape == (26, 3)\n"
            "assert np.isfinite(predictions.to_numpy()).all()\n"
        )
    elif example.endswith("example_VAE_rapidELprocessing.ipynb"):
        assertions += (
            "assert OUTPUT.resolve() == ROOT / 'outputs/autoencoder_3ch'\n"
            "saved = pd.read_csv(OUTPUT / 'analysis.csv')\n"
            "assert saved.shape == analysis.shape\n"
            "assert np.isfinite(saved.filter(like='_predicted').to_numpy()).all()\n"
        )
    elif example.endswith("train_xgboost.ipynb"):
        assertions += (
            "import pvcracks_training\n"
            "assert Path(pvcracks_training.__file__).resolve() == "
            "ROOT / 'training/src/pvcracks_training/__init__.py'\n"
            "assert OUTPUT.resolve() == ROOT / 'outputs/xgboost_training_example'\n"
            "assert (OUTPUT / 'training_summary.csv').is_file()\n"
            "for target in models:\n"
            "    assert (OUTPUT / f'xgb_segmented_{target}.json').is_file()\n"
            "    assert (OUTPUT / f'xgb_segmented_{target}.metadata.json').is_file()\n"
        )
    notebook.cells.append(nbformat.v4.new_code_cell(assertions))
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=180,
        resources={"metadata": {"path": str(cwd)}},
    )
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [
                str(example_checkout),
                str(example_checkout / "training/src"),
            ]
        ),
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    }
    try:
        client.execute(cwd=str(cwd), env=environment)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    assert not any(
        output.output_type == "error"
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
