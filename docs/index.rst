PVCracks Documentation
=======================

Version 0.2.0

.. image:: duramat_logo.png
   :width: 800
   :alt: DuraMAT logo

Overview
--------

PVCracks is a DuraMAT software project for studying crack-related power loss
and degradation in photovoltaic modules. It combines cell-level
electroluminescence analysis, machine-learning models, and transparent
electrical circuit simulation.

The package provides:

* MultiSolSegment for crack, busbar, dark-area, and non-cell segmentation.
* A skip-connected 3-channel autoencoder for mask reconstruction and clustering.
* An XGBoost power-loss estimator.
* pvspice tools for cell, module, half-cut module, and module-string IV
  simulations.

Featured notebooks
------------------

Start with the complete EL workflow, or choose one of the focused examples
below. These notebooks use the current three-channel autoencoder and matching
XGBoost models, and include executed tables and figures.

.. list-table:: Current autoencoder and XGBoost workflows
   :header-rows: 1
   :widths: 35 65

   * - Notebook
     - What it shows
   * - :doc:`Full EL analysis <Examples/example_VAE_rapidELprocessing>`
     - Process masks, inspect 3D latent-space clusters, compare signed
       Pmp/Voc changes in percent, and view original EL images in every cluster.
   * - :doc:`Generate a new mask image <Examples/example_generate_images>`
     - Sample a new empirical latent point and synthesize matching skip features
       to decode a mask image, with a check against the example inputs.
   * - :doc:`Load the 3-channel autoencoder <Examples/example_load_3CH-VAE>`
     - Load the bundled checkpoint, reconstruct crack/busbar/dark masks,
       and inspect the 320-dimensional embedding.
   * - :doc:`Inspect reconstruction and clustering <Examples/example_validate_pvimage>`
     - Calculate channel-wise Dice and IoU scores and visualize clusters
       of standardized bottleneck embeddings.
   * - :doc:`Predict Pmp, Voc, and Rsh <Examples/example_xgboost>`
     - Use the matched XGBoost models with all 326 features and compare
       measured and predicted electrical response.

Each notebook page provides a download of its executed ``.ipynb`` file.
Run the notebooks inside a source checkout to retain their paths to the
included example data and model weights. The default examples use 26 module
209 observations; they demonstrate the workflow rather than reproduce the
full manuscript data set or provide independent validation.

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:
   :caption: Featured notebooks

   Full EL analysis and clustering <Examples/example_VAE_rapidELprocessing>
   Generate a new mask image <Examples/example_generate_images>
   Load the 3-channel autoencoder <Examples/example_load_3CH-VAE>
   Reconstruction and latent-space inspection <Examples/example_validate_pvimage>
   Predict Pmp, Voc, and Rsh <Examples/example_xgboost>

Release and documentation versions
----------------------------------

This documentation describes pvcracks 0.2.0. Read the Docs selects the
matching tagged release through its version switcher. Build release tags such
as v0.2.0 after publishing the repository version so users can select the
corresponding documentation and source code together.

Data and models
---------------

Data sets, trained weights, and supporting materials are available from the
DuraMAT DataHub:
https://datahub.duramat.org/project/pv-crack-stress-and-power-loss

* MultiSolSegment training images and masks: DOI
  `10.21948/2587738 <https://doi.org/10.21948/2587738>`_.
* MultiSolSegment model weights: DOI
  `10.21948/2997859 <https://doi.org/10.21948/2997859>`_.
* The current 3-channel autoencoder and matching XGBoost models are bundled
  with the package. Earlier VAE weights are archived at DOI
  `10.21948/2997860 <https://doi.org/10.21948/2997860>`_.

Publications
------------

* MultiSolSegment

  Published in Solar Energy. DOI
  `10.1016/j.solener.2026.114469 <https://doi.org/10.1016/j.solener.2026.114469>`_.

* Variational autoencoder

  EU PVSEC 2024 conference proceeding. DOI
  `10.4229/EUPVSEC2024/3BO.15.6 <https://doi.org/10.4229/EUPVSEC2024/3BO.15.6>`_.

* Power-loss model

  2025 IEEE 53rd Photovoltaic Specialists Conference proceeding. DOI
  `10.1109/PVSC59419.2025.11132966 <https://doi.org/10.1109/PVSC59419.2025.11132966>`_.

* Partial-shading thermal validation

  S. Islam and S. R. Kurtz, *Understanding Heating from Partial Shade of a
  PV Module under Varying Conditions*. DOI
  `10.1109/JPHOTOV.2026.3735999 <https://doi.org/10.1109/JPHOTOV.2026.3735999>`_.

Webinar
-------

**Nov. 10, 2025**, 1 p.m. MT — `pvcracks: Quantifying the Impact of Solar
Cell Cracks on PV Systems <https://www.youtube.com/watch?v=4QWsLkpUDd4>`_,
presented by Norman Jost, Sandia.

Installation
------------

Install the runtime and inference tools:

.. code-block:: bash

   python -m pip install pvcracks

For development from a clone, use the editable installation instead:

.. code-block:: bash

   git clone https://github.com/sandialabs/pvcracks.git
   cd pvcracks
   python -m pip install -e .

The base package includes MultiSolSegment inference, autoencoder clustering, XGBoost
power-loss prediction, and both pvspice subpackages. To retrain a model or run
Ray/HPO workflows, install the companion training distribution:

.. code-block:: bash

   python -m pip install -e "./training[notebooks]"

For a source checkout, install its companion project in editable mode:

.. code-block:: bash

   python -m pip install -e ./training

The current ``pvcracks.vae`` model is a deterministic U-Net autoencoder;
it takes crack, busbar, and dark masks and produces 320 pooled embeddings.
XGBoost uses those unscaled embeddings plus six original-mask measurements.
Training scripts and notebooks live under ``training/``; Ray/HPO dependencies
are optional through ``python -m pip install -e "./training[hpo]"``.

The SPICE examples require a separate ngspice installation. Set the executable
path explicitly in each notebook's configuration cell. This keeps Windows,
Linux, and HPC runs reproducible and avoids relying on system PATH discovery.

More notebook examples
======================

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: Segmentation and circuit examples

   Examples/example_download_run_multisolsegment
   Examples/example_multisolsegment_cellELs
   Examples/example_multisolsegment_test_images
   Examples/example_download_run_multisolsegment_benchmarking
   Examples/example_minimodule_build
   Examples/example_pvspice_PID_pingel
   Examples/example_pvspice_thermal
   Examples/example_pvspice_thermal_all_rows_fit
   Examples/example_pvspice_reverse

Modules
=======

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: API reference

   api/pvcracks.powerloss
   api/pvcracks.pvspice
   api/pvcracks.pvspice_lite
   api/pvcracks.MultiSolSegment
   api/pvcracks.utils
   api/pvcracks.vae
   api/pvcracks.vae.pytorch_ssim
