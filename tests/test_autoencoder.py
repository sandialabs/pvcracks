"""Compatibility checks against the supplied FinalVAE3CH analysis artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pvcracks.powerloss.powerloss_functions import load_all_xgb_models
from pvcracks.vae import load_unet_ae_crack_plus
from pvcracks.vae.features import extract_features, morphology_features, prepare_masks
from pvcracks.vae.inference import reconstruct_masks, reconstruction_scores

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data"


@pytest.fixture(scope="module")
def model_and_masks():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    model, _ = load_unet_ae_crack_plus(device="cpu")
    masks = np.load(DATA / "autoencoder_3ch_module_209.npz")["masks"]
    yield model, masks
    torch.set_num_threads(previous)


def test_bundled_checkpoint_and_models_match_manifest():
    manifest = json.loads((ROOT / "pvcracks/vae/model_manifest.json").read_text())
    checkpoint = ROOT / "pvcracks/vae" / manifest["checkpoint"]
    assert (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        == manifest["checkpoint_sha256"]
    )
    for entry in manifest["models"].values():
        path = ROOT / "pvcracks/powerloss" / entry["filename"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_embedding_and_morphology_match_source_export(model_and_masks):
    model, masks = model_and_masks
    features = extract_features(model, masks, batch_size=4)
    reference = pd.read_csv(DATA / "autoencoder_3ch_reference.csv")
    assert features.shape == (26, 326)
    # The source embeddings were exported on another PyTorch backend. Small
    # pooling / convolution differences are expected; morphology must agree.
    np.testing.assert_allclose(
        features.iloc[:, :320],
        reference.loc[:, features.columns[:320]],
        atol=0.002,
        rtol=0.002,
    )
    np.testing.assert_allclose(
        features.iloc[:, 320:],
        reference.loc[:, features.columns[320:]],
        atol=1e-7,
        rtol=0,
    )


def test_native_xgboost_predictions_match_original_export_from_any_cwd(
    tmp_path, monkeypatch
):
    features = pd.read_csv(DATA / "autoencoder_3ch_reference.csv")
    reference = pd.read_csv(DATA / "xgboost_3ch_reference.csv").set_index("index")
    reference = reference.loc[features["index"]]
    monkeypatch.chdir(tmp_path)
    for target, model in load_all_xgb_models().items():
        expected = reference[target + "_pred_all"].to_numpy()
        available = np.isfinite(expected)
        predictions = model.predict(features)
        np.testing.assert_allclose(
            predictions[available], expected[available], atol=1e-4
        )
        shuffled = features[features.columns[::-1]]
        np.testing.assert_array_equal(model.predict(shuffled), predictions)
        with pytest.raises(ValueError, match="326"):
            model.predict(np.zeros((1, 320)))


def test_reconstruction_and_encoder_paths_agree(model_and_masks):
    model, masks = model_and_masks
    batch = prepare_masks(masks[:1])
    with torch.inference_mode():
        output = model(batch)
        np.testing.assert_allclose(model.encode(batch), output["embedding"], atol=1e-6)
    probabilities = reconstruct_masks(model, masks[:1], batch_size=1)
    assert probabilities.shape == (1, 3, 400, 400)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    scores = reconstruction_scores(masks[:1], probabilities)
    assert scores.shape == (1, 6)
    assert np.isfinite(scores).all().all()
    assert scores.ge(0).all().all() and scores.le(1).all().all()


def test_component_counts_use_eight_connectivity():
    masks = np.zeros((1, 3, 400, 400), dtype=np.uint8)
    masks[0, 0, 1, 1] = masks[0, 0, 2, 2] = 1
    masks[0, 0, 5, 5] = 1
    features = morphology_features(masks)
    assert features.loc[0, "crack_components"] == 2
    assert features.loc[0, "busbar_components"] == 0
    scores = reconstruction_scores(masks, masks)
    np.testing.assert_array_equal(scores, np.ones((1, 6)))


@pytest.mark.parametrize(
    "masks", [np.zeros((3, 256, 256)), np.full((3, 400, 400), 0.5)]
)
def test_rejects_incompatible_masks(masks):
    with pytest.raises(ValueError):
        prepare_masks(masks)


def test_xgboost_training_exports_loadable_models(tmp_path):
    training = pytest.importorskip("pvcracks_training.powerloss.training")
    table = pd.read_csv(DATA / "autoencoder_3ch_reference.csv")
    summary = training.train_xgb_models(table, tmp_path)
    assert set(summary["target"]) == {"delta_pmp_percent", "delta_voc_percent", "Rsh"}
    for model in load_all_xgb_models(tmp_path).values():
        assert np.isfinite(model.predict(table)).all()
