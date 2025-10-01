<img src="docs/pvcracks_logo.png" width="200">

# PVCracks

## Latest Release

[![DOI](https://img.shields.io/badge/DOI-10.11578%2Fdc.20240606.4-blue)](https://doi.org/10.11578/dc.20240606.4)

## License

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
License details can be found in: 
vae/License/

## Overview

PVCracks is the DuraMAT project that investigates the effects of cracks on power loss in photovoltaic (PV) solar cells and tracks crack progression over time. We provide:

- Open-source cell-level imaging and electrical datasets  
- **MultiSolSegment**: segmentation of cracks, busbars, and dark areas in electroluminescence (EL) images  
- **Variational Autoencoder (VAE)**: parameterization & clustering of segmented data  
- **XGBoost model**: estimation of power loss (ΔPMPP) per cell  

## Documentation

Below is a schematic of the repos architecture:

<img src="docs/pvcracks_sch.png" width="800">

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

- Sep. 30th  2025: Please attend our webinar on Nov. 17th. Sign up here [www.duramat.org/news-and-events/webinars](https://www.duramat.org/news-and-events/webinars)

## Installation

Clone and install:

```bash
git clone https://github.com/yourusername/pvcracks.git
cd pvcracks

pip install -r vae/requirements.txt
pip install -r retrain/requirements.txt

