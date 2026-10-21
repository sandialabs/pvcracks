"""Train XGBoost on unscaled embeddings and original-mask morphology."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, train

from pvcracks.powerloss.powerloss_functions import TARGETS
from pvcracks.vae.features import CHANNELS


def train_xgb_models(table: pd.DataFrame, output_dir: str | Path, *, seed=30):
    """Reproduce the supplied XGBoost settings and export portable predictors.

    Input rows align the 320 ``latent_N`` columns and six morphology columns
    with signed ``delta_pmp_percent``, ``delta_voc_percent``, and ``Rsh`` targets.
    Each target uses its non-missing rows and an 80/20 random split, as in the
    source notebook. The saved model stays fitted to the training partition.
    These row-level scores do not measure generalization to unseen modules.
    """
    features = [f"latent_{i}" for i in range(320)] + [
        f"{channel}_{stat}"
        for stat in ("fraction", "components")
        for channel in CHANNELS
    ]
    missing = set(features) - set(table.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    summaries = []
    for target in TARGETS:
        if target not in table:
            raise ValueError(f"Missing target column: {target}")
        rows = table.dropna(subset=[target])
        if len(rows) < 10:
            raise ValueError(f"{target} requires at least 10 non-missing rows")
        x = rows.loc[:, features].to_numpy(dtype=float)
        y = rows[target].to_numpy(dtype=float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite features or target values for {target}")
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.20, random_state=seed
        )
        model = train(
            {
                "eta": 0.05,
                "max_depth": 3,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "lambda": 1.0,
                "objective": "reg:squarederror",
                "seed": seed,
            },
            DMatrix(x_train, label=y_train),
            num_boost_round=100,
        )
        predictions = model.inplace_predict(x_test)
        metadata = {
            "feature_cols": features,
            "target": target,
            "r2_train": float(r2_score(y_train, model.inplace_predict(x_train))),
            "r2_test": float(r2_score(y_test, predictions)),
            "r2_all": float(r2_score(y, model.inplace_predict(x))),
            "mae_test": float(mean_absolute_error(y_test, predictions)),
            "rmse_test": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "n_rows": len(rows),
            "seed": seed,
            "test_size": 0.20,
        }
        path = directory / f"xgb_segmented_{target}.json"
        model.save_model(path)
        path.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        summaries.append({k: v for k, v in metadata.items() if k != "feature_cols"})
    summary = pd.DataFrame(summaries)
    summary.to_csv(directory / "training_summary.csv", index=False)
    return summary
