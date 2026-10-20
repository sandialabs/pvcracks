## Data for pvcracks examples

The current autoencoder/XGBoost notebooks use `autoencoder_3ch_module_209.npz`
(26 binary crack/busbar/dark masks, 400 x 400) and the aligned scalar metadata
in `autoencoder_3ch_module_209.csv`. These are the successfully segmented module
209 rows from the supplied FinalVAE3CH analysis; one of the 27 original images
was not included in that successful segmentation export. Original row indices
are preserved as `source_index`. The masks are stored as uint8 without pickle.

`autoencoder_3ch_reference.csv` contains the corresponding original exported
embeddings, morphology values, and target measurements.
`xgboost_3ch_reference.csv` contains the source analysis predictions, including
missing entries for rows without a measurement for a target. They support
compatibility checks; they are not independent validation data. Small embedding
differences across PyTorch backends can change predictions near a tree split.
The old VAE latent-analysis CSV has been replaced because its features are
incompatible with the new model pair.

This folder contains data to be used in the examples. Such as, electroluminescence (EL) images from a 9‐cell minimodule provided by the Technical University of Denmark.

- **Source**  
  EL images from DTU, part of the PVSC 2021 dataset:  
  [10.1109/PVSC43889.2021.9519081](https://doi.org/10.1109/PVSC43889.2021.9519081)  

- **Module**  
  9 solar cells wired in series. Data includes both the full‐module measurement and individual‐cell EL captures.

- **Degradation steps**  
  • **Init**: pristine, no mechanical damage  
  • **Deg1**: after first mechanical degradation  
  • **Deg2**: after second mechanical degradation  

### Folder layout

```text
docs/data/EL
├── 209_A1
│   ├── Deg1
│   ├── Deg2
│   └── Init
├── 209_A2
│   ├── Deg1
│   ├── Deg2
│   └── Init
├── 209_A3
│   ├── Deg1
│   ├── Deg2
│   └── Init
├── 209_B1
│   ├── Deg1
│   ├── Deg2
│   └── Init
├── 209_B2
│   ├── Deg1
│   ├── Deg2
│   └── Init
├── 209_B3
│   ├── Deg1
│   ├── Deg2
│   └── Init
├── 209_C1
│   ├── Deg1
│   ├── Deg2
│   └── Init
├── 209_C2
│   ├── Deg1
│   ├── Deg2
│   └── Init
└── 209_C3
    ├── Deg1
    ├── Deg2
    └── Init
```
