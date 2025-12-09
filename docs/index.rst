PVCracks Documentation
=======================

.. raw:: html

.. image:: duramat_logo.png
   :width: 800
   :alt: Repository architecture schematic

Overview
--------

PVCracks is the DuraMAT project that investigates the effects of cracks on power loss in photovoltaic (PV) solar cells and tracks crack progression over time. This effort was lead by the Photovoltaic System Evaluation Lab at Sandia National Laboratories. 
We provide:

- Open‐source cell‐level imaging and electrical datasets  
- **MultiSolSegment**: crack, busbar, and dark‐area segmentation in EL images  
- **Variational Autoencoder (VAE)**: unsupervised parameterization and clustering  
- **XGBoost model**: per‐cell power‐loss estimation (ΔPMPP)  
- **pvspice_lite**: lightweight SPICE tools for I–V curve simulation

Schematic
---------

This is how the workflow is set up. In addition to the SPICE simulation capabilities that are being added.

.. image:: pvcracks_sch.png
   :width: 800
   :alt: Repository architecture schematic

Data & Models
-------------

All datasets, trained model weights, and additional resources are hosted on DuraMAT DataHub:  
`https://datahub.duramat.org/project/pv-crack-stress-and-power-loss <https://datahub.duramat.org/project/pv-crack-stress-and-power-loss>`__

- Cell‐level EL & electrical data (publication pending)  
- MultiSolSegment training images & masks  
  (DOI: `10.21948/2587738 <https://doi.org/10.21948/2587738>`__)  
- MultiSolSegment model weights  
  (DOI: `10.21948/2997859 <https://doi.org/10.21948/2997859>`__)  
- VAE model weights  
  (DOI: `10.21948/2997860 <https://doi.org/10.21948/2997860>`__)

Publications
-------------

- **MultiSolSegment**  
  In revision for *Solar Energy* (Elsevier). Preprint  
  (`10.2139/ssrn.5506469 <https://dx.doi.org/10.2139/ssrn.5506469>`__)

- **Variational Autoencoder (VAE)**  
  EUPVSEC 2024 conference proceeding  
  (`10.4229/EUPVSEC2024/3BO.15.6 <https://doi.org/10.4229/EUPVSEC2024/3BO.15.6>`__)

- **Power-Loss Model (XGBoost)**  
  2025 IEEE 53rd Photovoltaic Specialists Conference (PVSC) proceeding  
  (`10.1109/PVSC59419.2025.11132966 <https://doi.org/10.1109/PVSC59419.2025.11132966>`__)

Updates
-------------

- Dec. 9 2025: DuraMAT webinar is uploaded here:  
  `www.duramat.org/news-and-events/webinars <https://www.duramat.org/news-and-events/webinars>`__  
  `Direct Link to YouTube <https://www.youtube.com/watch?v=4QWsLkpUDd4>`__


Acknowledgments
-------------

Funding was provided as part of the Durable Modules Consortium (DuraMAT), an Energy Materials Network Consortium funded by the U.S. Department of Energy, Office of Energy Efficiency and Renewable Energy, Solar Energy Technologies Office under agreement number 32509. The views expressed in this article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains—and the publisher, by accepting the article for publication, acknowledges—that it retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or to allow others to do so, for U.S. Government purposes.

Sandia National Laboratories is a multimission laboratory managed and operated by National Technology and Engineering Solutions of Sandia, LLC, a wholly owned subsidiary of Honeywell International Inc., for the U.S. Department of Energy’s National Nuclear Security Administration under contract DE-NA0003525. 

Installation
------------

Clone and install:

.. code-block:: bash

  git clone git@github.com:NormanJost/pvcracks.git
  cd pvcracks

  # 1. Upgrade tooling
  pip install --upgrade pip setuptools wheel

  # 2. Install dependencies
  pip install -r requirements.txt

  # 3. Install in editable mode
  pip install -e .    # enables intra‐project imports

Examples
========

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Notebook Examples

   Examples/example_multisolsegment_cellELs.ipynb
   Examples/example_download_run_multisolsegment.ipynb
   Examples/example_load_3CH-VAE.ipynb
   Examples/example_VAE_rapidELprocessing.ipynb
   Examples/example_xgboost.ipynb
   Examples/example_minimodule_build.ipynb

Modules
=======

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: API Reference

   api/pvcracks.utils
   api/pvcracks.vae
   api/pvcracks.powerloss
   api/pvcracks.pvspice_lite

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
