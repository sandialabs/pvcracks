"""Tests for prediction shaping that do not require bundled ML model weights."""

import numpy as np
import pytest

from pvcracks.powerloss.powerloss_functions import predict_power_and_voc


class OffsetModel:
    """A minimal predictor used to test the public prediction interface."""

    def __init__(self, offset: float):
        self.offset = offset

    def predict(self, values):
        return np.asarray(values).sum(axis=1) + self.offset


def test_predict_power_and_voc_returns_named_columns():
    result = predict_power_and_voc(
        [[1.0, 2.0], [3.0, 4.0]],
        OffsetModel(0.5),
        OffsetModel(-0.5),
    )

    assert result.columns.tolist() == ["power_loss_%", "Voc_diff_%"]
    np.testing.assert_allclose(result["power_loss_%"], [3.5, 7.5])
    np.testing.assert_allclose(result["Voc_diff_%"], [2.5, 6.5])


def test_predict_power_and_voc_rejects_one_dimensional_vectors():
    with pytest.raises(ValueError, match="2D"):
        predict_power_and_voc([1.0, 2.0], OffsetModel(0), OffsetModel(0))
