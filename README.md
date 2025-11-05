<p align="left">
  <img
    src="docs/pvcracks_logo.png"
    height="130"
    alt="PVCracks logo"
  />
  <img
    src="docs/duramat_logo.png"
    height="130"
    alt="DuraMAT logo"
  />
</p>

# PVCracks

## Latest Release

[![DOI](https://img.shields.io/badge/DOI-10.11578%2Fdc.20240606.4-blue)](https://doi.org/10.11578/dc.20240606.4)

## License

[![License: BSD 3-Clause](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](LICENSE)

License details can be found in the "License" folder.

## Overview

PVCracks is the DuraMAT project that investigates the effects of cracks on power loss in photovoltaic (PV) solar cells and tracks crack progression over time. We provide:

- Open-source cell-level imaging and electrical datasets  
- **MultiSolSegment**: segmentation of cracks, busbars, and dark areas in electroluminescence (EL) images  
- **Variational Autoencoder (VAE)**: parameterization & clustering of segmented data  
- **XGBoost model**: estimation of power loss (ΔPMPP) per cell  

## Documentation

[![Read the Docs](https://readthedocs.org/projects/pvcracks/badge/?version=latest)](https://pvcracks.readthedocs.io/en/latest/index.html#)

Full online documentation is available at  
https://pvcracks.readthedocs.io/en/latest/index.html#

Below is a schematic of the repo’s architecture:

<img src="docs/pvcracks_sch.png" width="800">

## Data & Models

All datasets, trained model weights, and additional resources are hosted on DuraMAT DataHub:  
[https://datahub.duramat.org/project/pv-crack-stress-and-power-loss](https://datahub.duramat.org/project/pv-crack-stress-and-power-loss)

- Cell-level EL & electrical data (data publication to follow)  
- MultiSolSegment training images and masks (DOI: [10.21948/2587738](https://doi.org/10.21948/2587738))  
- MultiSolSegment model weights (DOI: [10.21948/2997859](https://doi.org/10.21948/2997859))
- VAE model weights (DOI: [10.21948/2997860](https://doi.org/10.21948/2997860))

## Publications

- **MultiSolSegment**  
  In revision for *Renewable Energies* (Elsevier). Link to follow.

- **Variational Autoencoder (VAE)**  
  EUPVSEC 2024 conference proceeding  
  DOI: [10.4229/EUPVSEC2024/3BO.15.6](https://doi.org/10.4229/EUPVSEC2024/3BO.15.6)

- **Power-Loss Model (XGBoost)**  
  2025 IEEE 53rd Photovoltaic Specialists Conference (PVSC) proceeding  
  DOI: [10.1109/PVSC59419.2025.11132966](https://doi.org/10.1109/PVSC59419.2025.11132966)

## Updates

- Sep. 30th 2025: Please attend our webinar on Nov. 10th. Sign up here:  
  [www.duramat.org/news-and-events/webinars](https://www.duramat.org/news-and-events/webinars)

## Installation

Install [uv](https://docs.astral.sh/uv/).

Clone and install:

```bash
git clone git@github.com:sandialabs/pvcracks.git
cd pvcracks

# Install the package and its dependencies
uv sync
uv pip install -e . # enable intra-project imports
```

An auto-generated `requirements.txt` file for use with `pip` has also been provided for your convenience, but compatibility is not guaranteed.
