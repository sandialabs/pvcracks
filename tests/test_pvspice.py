"""Tests for the public pvspice netlist builders and output readers."""

import subprocess

import numpy as np
import pytest

from pvcracks.pvspice import spice_tools
from pvcracks.pvspice.spice_tools import (
    module_to_netlist,
    modules_halfcut_netlist_builder,
    modules_series_netlist_builder,
    ngspice_read_cell_level_ivs,
    ngpsice_read_voltage_current_modules,
    run_ngspice,
)


def _write_ascii_raw(path, variables, points):
    """Write a minimal ngspice-style ASCII raw output for reader tests."""
    lines = [
        "Title: test",
        f"No. Variables: {len(variables)}",
        f"No. Points: {len(points)}",
        "Variables:",
    ]
    for index, variable in enumerate(variables):
        kind = "current" if variable.lower().startswith("i(") else "voltage"
        lines.append(f"\t{index}\t{variable}\t{kind}")
    lines.append("Values:")
    for point_index, point in enumerate(points):
        for value_index, value in enumerate(point):
            prefix = f"{point_index}\t" if value_index == 0 else "\t"
            lines.append(f"{prefix}{value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_series_string_netlist_accepts_module_and_string_parameters(tmp_path):
    netlist = modules_series_netlist_builder(
        tmp_path,
        N_s=6,
        N_modules=2,
        cell_currents_=[5.0] * 6,
        series_resistances_=0.01,
        shunt_resistances_=[100.0] * 12,
        ideality_factor=1.2,
        saturation_current=1e-10,
        breakdown_voltage=15.0,
        v_bypass=20.0,
        v_oc=3.6,
        N_byp=3,
        circuit_name="series",
    )

    assert (tmp_path / "series.cir").is_file()
    assert netlist.count("* bypass diode") == 6
    assert "DbyM1" in netlist
    assert "dc V1 0 8.2 0.1" in netlist
    assert f"write {(tmp_path / 'series.out').resolve().as_posix()}" in netlist
    assert f"write {(tmp_path / 'series.txt').resolve().as_posix()}" in netlist


def test_halfcut_builder_has_two_branches_and_returns_netlist(tmp_path):
    netlist = modules_halfcut_netlist_builder(
        tmp_path,
        N_s=12,
        N_modules=1,
        cell_currents_=2.5,
        series_resistances_=0.02,
        shunt_resistances_=500.0,
        ideality_factor=1.2,
        saturation_current=1e-10,
        breakdown_voltage=15.0,
        v_bypass=20.0,
        v_oc=7.2,
        N_byp=3,
        circuit_name="halfcut",
    )

    assert (tmp_path / "halfcut.cir").is_file()
    assert "M0B0C0" in netlist
    assert "M0B1C0" in netlist
    assert netlist.count("* bypass diode") == 6
    assert netlist.endswith(".end\n")


def test_module_to_netlist_selects_legacy_topology_threshold(tmp_path):
    _, small_is_halfcut = module_to_netlist(
        tmp_path, 60, 5.0, 0.01, 100.0, 1.2, 1e-10, 15.0, 20.0, 36.0
    )
    _, large_is_halfcut = module_to_netlist(
        tmp_path, 132, 2.5, 0.02, 500.0, 1.2, 1e-10, 15.0, 20.0, 40.0
    )

    assert not small_is_halfcut
    assert large_is_halfcut


def test_ascii_raw_reader_uses_named_voltage_and_current_columns(tmp_path):
    raw_file = tmp_path / "result.out"
    raw_file.write_text(
        "\n".join(
            [
                "Title: test",
                "No. Variables: 3",
                "Variables:",
                "\t0\tv(v-sweep)\tvoltage",
                "\t1\tv(1)\tvoltage",
                "\t2\ti(v1)\tcurrent",
                "Values:",
                "0\t0.0",
                "\t0.0",
                "\t5.0",
                "1\t1.0",
                "\t1.0",
                "\t4.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    voltage, current, dataframe = ngpsice_read_voltage_current_modules(raw_file)

    np.testing.assert_allclose(voltage, [0.0, 1.0])
    np.testing.assert_allclose(current, [5.0, 4.5])
    assert dataframe.columns.tolist() == ["V", "I"]


def test_cell_level_iv_reader_reconstructs_voltage_and_current_curves(tmp_path):
    raw_file = tmp_path / "cell-level.out"
    raw_file.write_text(
        "\n".join(
            [
                "Title: two-cell module",
                "No. Variables: 6",
                "No. Points: 2",
                "Variables:",
                "\t0\tv(v-sweep)\tvoltage",
                "\t1\tv(1)\tvoltage",
                "\t2\ti(v1)\tcurrent",
                "\t3\tv(2)\tvoltage",
                "\t4\tv(3)\tvoltage",
                "\t5\tv(4)\tvoltage",
                "Values:",
                "0\t0.0",
                "\t0.0",
                "\t-5.0",
                "\t0.1",
                "\t0.2",
                "\t0.05",
                "1\t1.0",
                "\t1.0",
                "\t-4.0",
                "\t0.8",
                "\t0.5",
                "\t0.3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    long, wide, raw = ngspice_read_cell_level_ivs(
        raw_file, N_s=2, series_resistances=0.1
    )

    assert raw.columns.tolist() == [
        "v(v-sweep)",
        "v(1)",
        "i(v1)",
        "v(2)",
        "v(3)",
        "v(4)",
    ]
    np.testing.assert_allclose(wide["module_I"], [5.0, 4.0])
    np.testing.assert_allclose(wide["V_cell_0"], [-0.2, 0.5])
    np.testing.assert_allclose(wide["V_diode_0"], [-0.1, 0.3])
    np.testing.assert_allclose(wide["V_rs_1"], [0.15, 0.2])
    np.testing.assert_allclose(wide["I_cell_0"], [1.0, -2.0])
    np.testing.assert_allclose(wide["I_cell_1"], [-1.5, -2.0])
    assert long.columns.tolist() == [
        "sweep_V",
        "module_V",
        "module_I",
        "circuit_V",
        "circuit_I",
        "module",
        "branch",
        "cell",
        "cell_index",
        "V_cell",
        "V_diode",
        "V_rs",
        "I_cell",
    ]
    assert long["module"].tolist() == [0, 0, 0, 0]
    assert long["branch"].isna().all()
    assert long["cell"].tolist() == [0, 0, 1, 1]

    long_without_current, wide_without_current, _ = ngspice_read_cell_level_ivs(
        raw_file, N_s=2
    )
    assert not any(column.startswith("I_cell_") for column in wide_without_current)
    assert long_without_current["I_cell"].isna().all()


def test_cell_level_iv_reader_validates_resistance_and_node_inputs(tmp_path):
    raw_file = tmp_path / "incomplete.out"
    raw_file.write_text(
        "\n".join(
            [
                "No. Variables: 4",
                "Variables:",
                "\t0\tv(v-sweep)\tvoltage",
                "\t1\tv(1)\tvoltage",
                "\t2\ti(v1)\tcurrent",
                "\t3\tv(2)\tvoltage",
                "Values:",
                "0\t0.0",
                "\t0.0",
                "\t-5.0",
                "\t0.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="series_resistances"):
        ngspice_read_cell_level_ivs(raw_file, N_s=1, series_resistances=[0.1, 0.2])
    with pytest.raises(ValueError, match="v\\(3\\)"):
        ngspice_read_cell_level_ivs(raw_file, N_s=2)


def test_cell_level_iv_reader_supports_series_module_strings(tmp_path):
    raw_file = tmp_path / "series-string.out"
    _write_ascii_raw(
        raw_file,
        [
            "v(v-sweep)",
            "v(1)",
            "i(v1)",
            "v(2)",
            "v(3)",
            "v(4)",
            "v(m0out)",
            "v(m1c0d)",
            "v(m1n0)",
            "v(m1c1d)",
        ],
        [
            [0.0, 10.0, -2.0, 9.8, 5.0, 4.8, 1.0, 0.8, 0.5, 0.3],
            [1.0, 8.0, -1.5, 7.8, 4.0, 3.8, 0.8, 0.6, 0.4, 0.2],
        ],
    )

    long, wide, _ = ngspice_read_cell_level_ivs(
        raw_file,
        N_s=2,
        N_modules=2,
        topology="series",
        series_resistances=0.1,
    )

    assert set(long["module"]) == {0, 1}
    assert set(long["cell_index"]) == {0, 1, 2, 3}
    np.testing.assert_allclose(wide["string_I"], [2.0, 1.5])
    np.testing.assert_allclose(wide["V_module_0"], [9.0, 7.2])
    np.testing.assert_allclose(wide["V_module_1"], [1.0, 0.8])
    np.testing.assert_allclose(wide["V_module_1_cell_0"], [0.5, 0.4])
    np.testing.assert_allclose(wide["I_module_1_cell_1"], [-2.0, -2.0])


def test_cell_level_iv_reader_supports_halfcut_module_strings(tmp_path):
    raw_file = tmp_path / "halfcut-string.out"
    _write_ascii_raw(
        raw_file,
        [
            "v(v-sweep)",
            "v(1)",
            "i(v1)",
            "v(m0b0c0d)",
            "v(m0b0n0)",
            "v(m0b0c1d)",
            "v(m0b1c0d)",
            "v(m0b1n0)",
            "v(m0b1c1d)",
        ],
        [
            [0.0, 10.0, -4.0, 9.8, 5.0, 4.8, 9.7, 4.9, 4.7],
            [1.0, 8.0, -3.0, 7.8, 4.0, 3.8, 7.7, 3.9, 3.7],
        ],
    )

    long, wide, _ = ngspice_read_cell_level_ivs(
        raw_file,
        N_s=4,
        N_modules=1,
        topology="halfcut",
        series_resistances=[0.1, 0.1, 0.1, 0.1],
    )

    assert set(long["branch"]) == {0, 1}
    assert set(long["cell_index"]) == {0, 1, 2, 3}
    np.testing.assert_allclose(wide["V_module_0"], [10.0, 8.0])
    np.testing.assert_allclose(wide["V_module_0_branch_0_cell_0"], [5.0, 4.0])
    np.testing.assert_allclose(wide["V_module_0_branch_1_cell_1"], [4.9, 3.9])
    np.testing.assert_allclose(wide["I_module_0_branch_1_cell_0"], [-3.0, -3.0])


def test_runner_uses_the_exact_user_supplied_executable_and_log_path(
    tmp_path, monkeypatch
):
    executable = tmp_path / "ngspice.exe"
    console_executable = tmp_path / "ngspice_con.exe"
    circuit = tmp_path / "manual-path.cir"
    log_file = tmp_path / "manual-path.log"
    executable.touch()
    console_executable.touch()
    circuit.write_text("* test\n.end\n", encoding="utf-8")
    invocation = {}

    def fake_run(command, **kwargs):
        invocation["command"] = command
        invocation["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(spice_tools.subprocess, "run", fake_run)
    result = run_ngspice(executable, circuit, capture_output=True, log_file=log_file)

    assert result.returncode == 0
    assert invocation["command"] == [
        str(executable.resolve()),
        "-b",
        "-o",
        str(log_file.resolve()),
        str(circuit.resolve()),
    ]
    assert invocation["kwargs"]["cwd"] == str(tmp_path.resolve())


def test_netlist_rejects_bad_parameter_lengths(tmp_path):
    with pytest.raises(ValueError, match="cell_currents"):
        modules_series_netlist_builder(
            tmp_path,
            N_s=6,
            N_modules=2,
            cell_currents_=[1.0, 2.0],
            series_resistances_=0.01,
            shunt_resistances_=100.0,
            ideality_factor=1.2,
            saturation_current=1e-10,
            breakdown_voltage=15.0,
            v_bypass=20.0,
            v_oc=3.6,
        )
