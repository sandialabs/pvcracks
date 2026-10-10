"""Tests for pvspice_lite data-preparation and netlist helpers."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from pvcracks.pvspice_lite import pvspice_tools
from pvcracks.pvspice_lite.pvspice_helper import cells_to_module, extract_parameters


def test_extract_parameters_returns_one_single_diode_dictionary_per_cell():
    cells = pd.DataFrame(
        {
            "Rs": [0.01, 0.02],
            "Rsh": [100.0, 200.0],
            "I": [5.0, 4.5],
            "Is": [1e-10, 2e-10],
            "N": [1.2, 1.3],
        }
    )

    parameters = extract_parameters(cells)

    assert parameters == [
        {"Rs": 0.01, "Rsh": 100.0, "I": 5.0, "Is": 1e-10, "N": 1.2},
        {"Rs": 0.02, "Rsh": 200.0, "I": 4.5, "Is": 2e-10, "N": 1.3},
    ]


def test_extract_parameters_reports_missing_columns():
    with pytest.raises(ValueError, match="Rsh"):
        extract_parameters(pd.DataFrame({"Rs": [0.01]}))


def test_cells_to_module_validates_layout_before_loading_images():
    with pytest.raises(ValueError, match="More cell images"):
        cells_to_module(
            pd.DataFrame({"ELPath": ["one.png", "two.png"]}),
            (10, 10),
            rows=1,
            columns=1,
        )


def test_minimodule_netlist_uses_absolute_output_paths(tmp_path, monkeypatch):
    """The lite simulator must not depend on ngspice's working directory."""
    fake_pvlib = types.SimpleNamespace(
        ivtools=types.SimpleNamespace(
            utils=types.SimpleNamespace(
                rectify_iv_curve=lambda voltage, current: (
                    np.asarray(voltage),
                    np.asarray(current),
                )
            )
        )
    )
    monkeypatch.setitem(sys.modules, "pvlib", fake_pvlib)
    monkeypatch.setattr(pvspice_tools, "run_ngspice", lambda *_: None)
    monkeypatch.setattr(
        pvspice_tools,
        "ngpsice_read_voltage_current_modules",
        lambda _: (
            np.array([0.0, 1.0]),
            np.array([1.0, 0.0]),
            pd.DataFrame({"V": [0.0, 1.0], "I": [1.0, 0.0]}),
        ),
    )

    pvspice_tools.MiniMod_Spice(
        [{"Rs": 0.01, "Rsh": 100.0, "I": 5.0, "Is": 1e-10, "N": 1.2}],
        tmp_path,
        [0.0, 1.0],
        V_Step=0.1,
        spicepath=tmp_path / "ngspice",
    )

    netlist = (tmp_path / "solar_circuit.cir").read_text(encoding="utf-8")
    assert f"write {(tmp_path / 'solar_circuit.out').resolve().as_posix()}" in netlist
    assert f"write {(tmp_path / 'solar_circuit.txt').resolve().as_posix()}" in netlist
