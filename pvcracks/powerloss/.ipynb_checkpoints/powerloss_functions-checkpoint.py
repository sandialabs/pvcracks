# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:04:49 2023

@authors: nrjost
"""

import pickle
import numpy as np
import pandas as pd

def load_xgb_models(
    pmpp_model_path: str = "xgb_model_pmpp_diff_percent_3CH.pkl",
    voc_model_path: str  = "xgb_model_Voc_diff_percent_3CH.pkl"
):
    """
    Load two XGBoost models from disk:
      - A model trained on pmpp_diff_% (power loss %)
      - A model trained on Voc_diff_%

    Parameters
    ----------
    pmpp_model_path : str
        Path to the pickle file for the pmpp_diff_% model.
    voc_model_path : str
        Path to the pickle file for the Voc_diff_% model.

    Returns
    -------
    pmpp_model, voc_model : xgboost.XGBRegressor
        The loaded XGBRegressor models.
    """
    with open(pmpp_model_path, "rb") as f:
        pmpp_model = pickle.load(f)

    with open(voc_model_path, "rb") as f:
        voc_model = pickle.load(f)

    return pmpp_model, voc_model


def predict_power_and_voc(
    latent_vectors,
    pmpp_model,
    voc_model
) -> pd.DataFrame:
    """
    Given latent vectors, predict:
    
    - power loss (%) using the pmpp model  
    - Voc difference (%) using the Voc model

    Parameters
    ----------
    latent_vectors : array-like
        Either:

        - a 2D numpy array of shape (n_samples, latent_dim)
        - a 1D object-dtype numpy array or list of 1D arrays (each of length latent_dim)

    pmpp_model : xgboost.XGBRegressor
        Loaded model for pmpp_diff_%.

    voc_model : xgboost.XGBRegressor
        Loaded model for Voc_diff_%.


    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:

        - ``power_loss_%``
        - ``Voc_diff_%``

        and one row per input latent vector.
    """
    # Convert to numpy array
    X = np.array(latent_vectors)

    # If object-dtype (e.g. array of arrays), stack into a 2D array
    if X.dtype == object:
        X = np.vstack(X)

    if X.ndim != 2:
        raise ValueError(
            f"Expected latent_vectors to be 2D or object-dtype array of 1D vectors; got shape {X.shape}"
        )

    # Make predictions
    power_loss_preds = pmpp_model.predict(X)
    voc_diff_preds   = voc_model.predict(X)

    # Pack into DataFrame
    results = pd.DataFrame({
        "power_loss_%": power_loss_preds,
        "Voc_diff_%":   voc_diff_preds
    })
    return results