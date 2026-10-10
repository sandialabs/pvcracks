# pvcracks-training

`pvcracks-training` is the optional companion package for retraining
MultiSolSegment and VAE models. It is separated from the main `pvcracks`
distribution so a normal installation remains focused on the documented
analysis, pretrained-inference, XGBoost, and SPICE workflows.

Install a released pair from PyPI:

```bash
python -m pip install pvcracks pvcracks-training
```

When working from this repository, install both projects in editable mode:

```bash
python -m pip install -e .
python -m pip install -e ./training
```

The `scripts/` directory contains the configurable Ray/HPO experiment retained
from the development repository. Set its data and output locations before
running it on a workstation or HPC system. The package APIs under
`pvcracks_training` provide reusable checkpoint, model, and VAE training
helpers without importing Ray during ordinary `pvcracks` use.
