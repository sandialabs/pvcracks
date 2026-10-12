---
title: 'pvcracks: Open tools for photovoltaic crack analysis and electrical simulation'
tags:
  - Python
  - photovoltaics
  - electroluminescence
  - reliability
  - circuit simulation
authors:
  - name: Norman Jost
    affiliation: "1"
  - name: Brandon K. Byford
    affiliation: "1"
  - name: Rodrigo d. Prado Santamaria
    affiliation: "2"
  - name: Shariful Islam
    affiliation: "3"
  - name: Ojas Sanghi
    affiliation: "1, 4"
  - name: Jennifer L. Braid
    affiliation: "1"
affiliations:
  - name: Sandia National Laboratories, Albuquerque, New Mexico, USA
    index: 1
  - name: Technical University of Denmark, Department of Electrical and Photonics Engineering, Roskilde, Denmark
    index: 2
  - name: University of California, Merced, California, USA
    index: 3
  - name: Massachusetts Institute of Technology, Cambridge, Massachusetts, USA
    index: 4
date: 27 September 2026
bibliography: paper.bib
---

# Summary

pvcracks is an open-source Python package that links photovoltaic (PV)
electroluminescence (EL) images with electrical reliability analysis. It
segments cracks, dark regions, busbars, and non-cell regions in cell EL images;
derives compact image representations; estimates image-associated power loss;
and builds cell-resolved electrical simulations. The `pvspice` subpackage
writes one-diode SPICE circuits and supports cell, module, and string
simulations with spatially varying photocurrent, series resistance, and shunt
resistance. The package, documentation, trained artefacts, and reproducible
examples are available for research and engineering use [@Jost2024].

# Statement of need

EL imaging is widely used to identify manufacturing defects and in-service
damage in PV modules. An EL pattern, however, does not by itself specify the
electrical consequence of a defect. A crack may affect a cell differently
depending on its location, the cell interconnection, operating conditions, and
the associated changes in current or resistance. Analysts therefore need to
move from image-derived features to cell parameters and then to a module or
string current-voltage (IV) curve. This work is often divided among image
scripts, manually edited circuit netlists, and proprietary tools. The resulting
workflow can be difficult to inspect, reproduce, or adapt.

pvcracks provides a documented path from cell EL images to feature
segmentation and data-driven power-loss estimates. It also provides
physically interpretable electrical simulation when cell parameters or
degradation maps are available. The package is intended for PV reliability
researchers, diagnostic-tool developers, and engineers studying cracking,
reverse bias, partial shading, and potential-induced degradation (PID). The
workflows use file-backed, parameterized inputs so users can replace the
included images, IV data, and module layouts with their own data.

# State of the field

The open PV software ecosystem includes general system modelling in
pvlib-python [@Holmgren2018], EL image processing in PVimage [@Pierce2020],
degradation analysis in RdTools [@RdTools], and circuit-based mismatch
simulations in PVMismatch [@PVMismatch]. pvcracks is designed to work alongside
these packages. Its specific contribution is the connection between
EL-derived feature information and cell-resolved electrical simulations.
`pvspice` accepts maps of photocurrent, series resistance, and shunt
resistance. It can therefore represent spatially heterogeneous damage within a
module and across a string.

# Software design

The default `pvcracks` distribution includes the documented analysis tools,
pretrained inference, XGBoost workflow, and `pvspice` without Ray or
experiment-management dependencies. Model retraining and hyperparameter
optimization are supplied through the separately installable
`pvcracks-training` companion distribution. `MultiSolSegment` produces
overlapping masks for cracks, dark regions, busbars, and non-cell areas in
multi-busbar EL images. Sanghi *et al.* describe the segmentation method and
benchmarks [@Sanghi2026]. The labelled images and masks
[@MultiSolSegmentData] and model weights [@MultiSolSegmentWeights] are cited
separately so users can retrieve the exact data and trained artefacts.

The package also includes a convolutional variational autoencoder for compact
image representations used in triage and clustering. A separate XGBoost
workflow relates processed image features to maximum-power loss, following the
crack-power-loss study of Braid *et al.* [@Braid2025]. The pretrained VAE is
archived separately [@VAEWeights]. These models support inspection and
inference; the training distribution is required only when users need to
retrain or tune models.

`pvspice` creates inspectable one-diode circuit netlists, runs a compatible
local SPICE engine, and parses IV and EL outputs. Its API separates netlist
generation, simulation execution, raw-output parsing, and degradation-map
construction. It supports series and half-cut module topologies, bypass diodes,
and full strings. Users can inspect the electrical model, change cell-level
parameters, and run the same workflow with a compatible simulator.

The documentation provides focused examples rather than using this paper to
report application-specific plots. `example_pvspice_reverse.ipynb` demonstrates
reverse-bias and mismatch analysis. `example_pvspice_thermal.ipynb` applies
the workflow to UC Merced partial-shading IV and temperature data. The
experimental context is documented in PVRW proceedings [@PVRW2026Posters] and
by Islam and Kurtz [@IslamKurtz2026JPhotov]. The mini-module example uses
cell-level parameter and EL data from a PVSC cracked-cell study
[@DelPrado2021]. `example_pvspice_PID_pingel.ipynb` applies spatial
shunt-resistance patterns from Pingel *et al.* [@Pingel2010] to a module
string. The examples provide reproducible starting points for new module
layouts and degradation maps. The synthetic EL-IV workflow was also presented
at EU PVSEC [@Jost2024EUPVSEC].

# Research impact statement

pvcracks makes image processing, learned estimation, and cell-resolved circuit
analysis available as inspectable Python code. Tests, examples, documentation,
trained weights, and data references give users clear entry points for
reproducing and extending a workflow. The project is distributed through the
Department of Energy's DuraMAT DataHub and has a citable software release
[@Jost2024]. This paper cites the software, data, trained artefacts, and prior
PV studies at their points of use so users can identify the provenance of each
component.

# Acknowledgements

Funding was provided as part of the Durable Modules Consortium (DuraMAT), an
Energy Materials Network Consortium funded by the U.S. Department of Energy,
Office of Energy Efficiency and Renewable Energy, Solar Energy Technologies
Office under agreement number 32509. Sandia National Laboratories is a
multimission laboratory managed and operated by National Technology and
Engineering Solutions of Sandia, LLC, a wholly owned subsidiary of Honeywell
International Inc., for the U.S. Department of Energy's National Nuclear
Security Administration under contract DE-NA0003525.

# AI usage disclosure

OpenAI Codex (GPT-5) assisted with code refactoring, test scaffolding and
documentation updates. The authors wrote the initial code and made
architectural and design decisions; they reviewed, edited, and validated all
AI-assisted content and retain responsibility for the submitted work.

# References
