"""Load the matched 3-channel autoencoder / XGBoost power-loss models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ("delta_pmp_percent", "delta_voc_percent", "Rsh")


class FeaturePredictor:
    """An XGBoost regressor with its ordered input-column contract.

    DataFrames are selected by column name. Arrays must already contain all
    features in ``feature_cols`` order, including the six morphology features.
    """

    def __init__(self, model, metadata):
        self.model = model
        self.metadata = metadata
        self.feature_cols = tuple(metadata["feature_cols"])

    def predict(self, features):
        if isinstance(features, pd.DataFrame):
            missing = set(self.feature_cols) - set(features.columns)
            if missing:
                raise ValueError(f"Missing model features: {sorted(missing)}")
            values = features.loc[:, list(self.feature_cols)].to_numpy(dtype=float)
        else:
            values = np.asarray(features)
            if values.dtype == object and values.ndim == 1:
                values = np.stack(values)
        if values.ndim != 2 or values.shape[1] != len(self.feature_cols):
            raise ValueError(
                f"Expected 2D features with {len(self.feature_cols)} columns "
                f"(embeddings plus morphology); got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Model features must be finite")
        return self.model.inplace_predict(values)


def load_xgb_model(model_path: str | Path) -> FeaturePredictor:
    """Load a native XGBoost JSON model and its sibling ``.metadata.json``."""
    from xgboost import Booster

    path = Path(model_path)
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    model = Booster(model_file=path)
    if model.num_features() != len(metadata["feature_cols"]):
        raise ValueError("Model and metadata feature counts differ")
    return FeaturePredictor(model, metadata)


def load_all_xgb_models(model_dir: str | Path | None = None):
    """Load the bundled ΔPmp (%), ΔVoc (%), and Rsh models from any cwd."""
    directory = Path(model_dir) if model_dir is not None else Path(__file__).parent
    return {
        target: load_xgb_model(directory / f"xgb_segmented_{target}.json")
        for target in TARGETS
    }


def load_xgb_models(
    pmpp_model_path: str | Path | None = None,
    voc_model_path: str | Path | None = None,
):
    """Return the matching bundled power and Voc predictors.

    Optional paths refer to native JSON models with sibling metadata files.
    The models predict signed percentage changes, ``100 * (post-pre) / pre``;
    a negative value indicates loss. Values are not clipped or sign-flipped.
    """
    directory = Path(__file__).parent
    return (
        load_xgb_model(
            pmpp_model_path or directory / "xgb_segmented_delta_pmp_percent.json"
        ),
        load_xgb_model(
            voc_model_path or directory / "xgb_segmented_delta_voc_percent.json"
        ),
    )


def predict_power_and_voc(latent_vectors, pmpp_model, voc_model) -> pd.DataFrame:
    """Predict signed power / Voc changes from a complete feature table.

    ``latent_vectors`` retains its historical argument name. The current bundled
    models require 326 features: 320 raw embeddings plus six mask measurements.
    The historical output column ``power_loss_%`` contains signed ΔPmp (%).
    """
    values = latent_vectors
    if not isinstance(values, pd.DataFrame):
        values = np.asarray(values)
        if values.dtype == object and values.ndim == 1:
            values = np.stack(values)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D model features; got shape {values.shape}")
    return pd.DataFrame(
        {
            "power_loss_%": pmpp_model.predict(values),
            "Voc_diff_%": voc_model.predict(values),
        },
        index=values.index if isinstance(values, pd.DataFrame) else None,
    )
