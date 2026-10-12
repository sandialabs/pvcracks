"""Build, execute, and read transparent photovoltaic ngspice simulations."""

from __future__ import annotations

import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def run_ngspice(
    ngpsice_path: str | os.PathLike[str],
    circuit_file: str | os.PathLike[str],
    *,
    capture_output: bool = False,
    timeout: float | None = None,
    log_file: str | os.PathLike[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an ngspice batch simulation using an explicit executable path.

    ``ngpsice_path`` is intentionally not discovered from an environment
    variable, ``PATH``, or platform-specific installation locations. Callers
    supply the exact executable they intend to run, which makes notebook and
    HPC executions reproducible across Windows and Linux.
    """
    circuit = Path(circuit_file).resolve()
    executable = Path(ngpsice_path).expanduser().resolve()
    if not circuit.is_file():
        raise FileNotFoundError(f"SPICE circuit file does not exist: {circuit}")
    if not executable.is_file():
        raise FileNotFoundError(f"ngspice executable does not exist: {executable}")
    command = [str(executable), "-b"]
    if log_file is not None:
        log = Path(log_file).expanduser().resolve()
        if not log.parent.is_dir():
            raise FileNotFoundError(f"ngspice log directory does not exist: {log.parent}")
        command.extend(["-o", str(log)])
    command.append(str(circuit))
    kwargs: dict[str, object] = {
        "check": True,
        "cwd": str(circuit.parent),
        "text": True,
        "capture_output": capture_output,
        "timeout": timeout,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        return subprocess.run(command, **kwargs)
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip() if error.stderr else str(error)
        raise RuntimeError(f"ngspice failed for {circuit}: {details}") from error


def run_xyce(
    xyce_path: str | os.PathLike[str],
    outputfile: str | os.PathLike[str],
    circuit_file: str | os.PathLike[str],
    *,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Xyce batch simulation and return the completed process."""
    circuit = Path(circuit_file).resolve()
    executable = Path(xyce_path).expanduser().resolve()
    output = Path(outputfile).expanduser().resolve()
    if not circuit.is_file():
        raise FileNotFoundError(f"SPICE circuit file does not exist: {circuit}")
    if not executable.is_file():
        raise FileNotFoundError(f"Xyce executable does not exist: {executable}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Xyce output directory does not exist: {output.parent}")
    return subprocess.run(
        [str(executable), "-r", str(output), str(circuit)],
        check=True,
        cwd=str(circuit.parent),
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def plot_solar_module(
    cell_currents: np.ndarray | list[float],
    cells_wide: int,
    cells_long: int,
    font: int = 18,
    save: bool | str | os.PathLike[str] = False,
):
    """Plot a cell-current map and return its Matplotlib axes."""
    from matplotlib import pyplot as plt

    values = np.asarray(cell_currents, dtype=float)
    expected = cells_wide * cells_long
    if values.size != expected:
        raise ValueError(f"Expected {expected} currents, received {values.size}.")
    figure, axis = plt.subplots(figsize=(10, 5))
    image = axis.imshow(
        values.reshape(cells_wide, cells_long), cmap="plasma", aspect="auto"
    )
    colorbar = figure.colorbar(image, ax=axis, label="Current (A)")
    colorbar.ax.yaxis.label.set_fontsize(font)
    axis.set_xlabel("Cell number along long edge", fontsize=font)
    axis.set_ylabel("Cell number along short edge", fontsize=font)
    axis.set_title("Solar module current distribution", fontsize=font + 2)
    figure.tight_layout()
    if save:
        output = Path("module_current_distr.png") if save is True else Path(save)
        figure.savefig(output, dpi=150, bbox_inches="tight")
    return axis


def _expand_parameter(
    values, name: str, cells_per_module: int, module_count: int
) -> np.ndarray:
    """Normalize scalar, per-module, or full-string parameters."""
    total_cells = cells_per_module * module_count
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 1:
        return np.full(total_cells, array.item())
    if array.size == cells_per_module:
        return np.tile(array, module_count)
    if array.size == total_cells:
        return array
    raise ValueError(
        f"{name} must contain 1, {cells_per_module}, or {total_cells} values; "
        f"got {array.size}."
    )


def _output_node(
    module: int, branch: int | None, cell: int, last_cell: int, module_end: str
) -> str:
    if cell == last_cell:
        return module_end
    if branch is None and module == 0:
        # Preserve the original numeric-node layout for the single-module
        # standard-cell workflow used by the published thermal notebook.
        return str(2 * cell + 3)
    branch_name = "" if branch is None else f"B{branch}"
    return f"M{module}{branch_name}N{cell}"


def _append_cell(
    lines: list[str],
    element_id: str,
    start: str,
    end: str,
    iph: float,
    rs: float,
    rsh: float,
    diode_node: str | None = None,
) -> None:
    diode_node = diode_node or f"{element_id}D"
    lines.extend(
        [
            f"* cell {element_id}",
            f"Rs{element_id} {start} {diode_node} {rs:.12g}",
            f"Rsh{element_id} {diode_node} {end} {rsh:.12g}",
            f"D{element_id} {diode_node} {end} sd1",
            f"Iph{element_id} {end} {diode_node} DC {iph:.12g}",
        ]
    )


def _append_bypass_diodes(
    lines: list[str],
    module: int,
    branch: int | None,
    input_node: str,
    output_node: str,
    cells_in_branch: int,
    bypass_diodes: int,
) -> None:
    if cells_in_branch % bypass_diodes:
        raise ValueError(
            f"{cells_in_branch} cells cannot be divided among {bypass_diodes} "
            "bypass diodes."
        )
    cells_per_substring = cells_in_branch // bypass_diodes
    branch_name = "" if branch is None else f"B{branch}"
    for bypass in range(bypass_diodes):
        first = bypass * cells_per_substring
        last = first + cells_per_substring - 1
        start = (
            input_node
            if first == 0
            else _output_node(
                module, branch, first - 1, cells_in_branch - 1, output_node
            )
        )
        end = _output_node(module, branch, last, cells_in_branch - 1, output_node)
        lines.extend(
            [
                (
                    f"* bypass diode module {module} branch {branch_name} "
                    f"substring {bypass}"
                ),
                f"DbyM{module}{branch_name}{bypass} {end} {start} sdby",
            ]
        )


def _append_series_module(
    lines: list[str],
    module: int,
    input_node: str,
    output_node: str,
    photocurrents: np.ndarray,
    series_resistances: np.ndarray,
    shunt_resistances: np.ndarray,
    bypass_diodes: int,
) -> None:
    cells = photocurrents.size
    for cell in range(cells):
        start = (
            input_node
            if cell == 0
            else _output_node(module, None, cell - 1, cells - 1, output_node)
        )
        end = _output_node(module, None, cell, cells - 1, output_node)
        _append_cell(
            lines,
            f"M{module}C{cell}",
            start,
            end,
            photocurrents[cell],
            series_resistances[cell],
            shunt_resistances[cell],
            str(2 * cell + 2) if module == 0 else None,
        )
    _append_bypass_diodes(
        lines, module, None, input_node, output_node, cells, bypass_diodes
    )


def _append_halfcut_module(
    lines: list[str],
    module: int,
    input_node: str,
    output_node: str,
    photocurrents: np.ndarray,
    series_resistances: np.ndarray,
    shunt_resistances: np.ndarray,
    bypass_diodes: int,
) -> None:
    if photocurrents.size % 2:
        raise ValueError("A half-cut module must contain an even number of cells.")
    cells_per_branch = photocurrents.size // 2
    for branch in range(2):
        offset = branch * cells_per_branch
        for cell in range(cells_per_branch):
            index = offset + cell
            start = (
                input_node
                if cell == 0
                else _output_node(
                    module, branch, cell - 1, cells_per_branch - 1, output_node
                )
            )
            end = _output_node(module, branch, cell, cells_per_branch - 1, output_node)
            _append_cell(
                lines,
                f"M{module}B{branch}C{cell}",
                start,
                end,
                photocurrents[index],
                series_resistances[index],
                shunt_resistances[index],
            )
        _append_bypass_diodes(
            lines,
            module,
            branch,
            input_node,
            output_node,
            cells_per_branch,
            bypass_diodes,
        )


def _build_netlist(
    *,
    path: str | os.PathLike[str],
    cells_per_module: int,
    modules: int,
    cell_currents,
    series_resistances,
    shunt_resistances,
    ideality_factor: float,
    saturation_current: float,
    breakdown_voltage: float,
    bypass_breakdown_voltage: float,
    module_open_circuit_voltage: float,
    topology: str,
    bypass_diodes: int,
    nominal_temperature: float,
    circuit_name: str,
    voltage_step: float,
) -> str:
    """Build and persist an ngspice DC sweep netlist."""
    if cells_per_module <= 0 or modules <= 0:
        raise ValueError("cells_per_module and modules must be positive.")
    if bypass_diodes <= 0 or voltage_step <= 0:
        raise ValueError("bypass_diodes and voltage_step must be positive.")
    if topology not in {"series", "halfcut"}:
        raise ValueError(f"Unknown module topology: {topology}.")
    photocurrents = _expand_parameter(
        cell_currents, "cell_currents", cells_per_module, modules
    )
    cell_rs = _expand_parameter(
        series_resistances, "series_resistances", cells_per_module, modules
    )
    cell_rsh = _expand_parameter(
        shunt_resistances, "shunt_resistances", cells_per_module, modules
    )
    if np.any(cell_rs < 0):
        raise ValueError("series_resistances must be non-negative.")
    if np.any(cell_rsh <= 0):
        raise ValueError("shunt_resistances must be positive.")

    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    circuit_file = directory / f"{circuit_name}.cir"
    output_file = directory / f"{circuit_name}.out"
    text_file = directory / f"{circuit_name}.txt"
    lines = [
        "* pvcracks photovoltaic SPICE simulation",
        (
            f"* topology={topology}, modules={modules}, "
            f"cells_per_module={cells_per_module}"
        ),
        (
            f".model sd1 D (IS={saturation_current:.12g} "
            f"N={ideality_factor:.12g} EG=1.12 "
            f"BV={breakdown_voltage:.12g})"
        ),
        f".model sdby D (IS=1e-7 N=1 BV={bypass_breakdown_voltage:.12g} IBV=20)",
        "V1 1 0 0",
    ]
    for module in range(modules):
        start_index = module * cells_per_module
        stop_index = start_index + cells_per_module
        input_node = "1" if module == 0 else f"M{module - 1}OUT"
        output_node = "0" if module == modules - 1 else f"M{module}OUT"
        parameters = (
            photocurrents[start_index:stop_index],
            cell_rs[start_index:stop_index],
            cell_rsh[start_index:stop_index],
        )
        if topology == "series":
            _append_series_module(
                lines, module, input_node, output_node, *parameters, bypass_diodes
            )
        else:
            _append_halfcut_module(
                lines, module, input_node, output_node, *parameters, bypass_diodes
            )
    sweep_limit = modules * module_open_circuit_voltage + 1.0
    lines.extend(
        [
            f".temp {nominal_temperature:.12g}",
            f".options tnom={nominal_temperature:.12g}",
            ".control",
            "set filetype=ascii",
            f"dc V1 0 {sweep_limit:.12g} {voltage_step:.12g}",
            "print dc v(1) i(V1)",
            f"write {output_file.as_posix()}",
            f"write {text_file.as_posix()}",
            "quit",
            ".endc",
            ".end",
        ]
    )
    netlist = "\n".join(lines) + "\n"
    circuit_file.write_text(netlist, encoding="utf-8")
    return netlist


def module_to_netlist(
    path: str | os.PathLike[str],
    N_s: int,
    cell_currents,
    series_resistances,
    shunt_resistances,
    ideality_factor: float,
    saturation_current: float,
    breakdown_voltage: float,
    v_bypass: float,
    v_oc: float,
    TNOM: float = 25,
    circuit_name: str = "solar_circuit",
) -> tuple[str, bool]:
    """Build one module and return its netlist text and half-cut topology flag."""
    half_cut = N_s > 90
    netlist = _build_netlist(
        path=path,
        cells_per_module=N_s,
        modules=1,
        cell_currents=cell_currents,
        series_resistances=series_resistances,
        shunt_resistances=shunt_resistances,
        ideality_factor=ideality_factor,
        saturation_current=saturation_current,
        breakdown_voltage=breakdown_voltage,
        bypass_breakdown_voltage=v_bypass,
        module_open_circuit_voltage=v_oc,
        topology="halfcut" if half_cut else "series",
        bypass_diodes=3,
        nominal_temperature=TNOM,
        circuit_name=circuit_name,
        voltage_step=0.1,
    )
    return netlist, half_cut


def modules_halfcut_netlist_builder(
    path: str | os.PathLike[str],
    N_s: int,
    N_modules: int,
    cell_currents_,
    series_resistances_,
    shunt_resistances_,
    ideality_factor: float,
    saturation_current: float,
    breakdown_voltage: float,
    v_bypass: float,
    v_oc: float,
    N_byp: int = 3,
    TNOM: float = 25,
    circuit_name: str = "solar_circuit",
    v_step: float = 0.1,
) -> str:
    """Build a series string of half-cut modules from scalar or cell arrays."""
    return _build_netlist(
        path=path,
        cells_per_module=N_s,
        modules=N_modules,
        cell_currents=cell_currents_,
        series_resistances=series_resistances_,
        shunt_resistances=shunt_resistances_,
        ideality_factor=ideality_factor,
        saturation_current=saturation_current,
        breakdown_voltage=breakdown_voltage,
        bypass_breakdown_voltage=v_bypass,
        module_open_circuit_voltage=v_oc,
        topology="halfcut",
        bypass_diodes=N_byp,
        nominal_temperature=TNOM,
        circuit_name=circuit_name,
        voltage_step=v_step,
    )


def modules_series_netlist_builder(
    path: str | os.PathLike[str],
    N_s: int,
    N_modules: int,
    cell_currents_,
    series_resistances_,
    shunt_resistances_,
    ideality_factor: float,
    saturation_current: float,
    breakdown_voltage: float,
    v_bypass: float,
    v_oc: float,
    N_byp: int = 3,
    TNOM: float = 25,
    circuit_name: str = "solar_circuit",
    v_step: float = 0.1,
) -> str:
    """Build a series string of conventional full-cell modules."""
    return _build_netlist(
        path=path,
        cells_per_module=N_s,
        modules=N_modules,
        cell_currents=cell_currents_,
        series_resistances=series_resistances_,
        shunt_resistances=shunt_resistances_,
        ideality_factor=ideality_factor,
        saturation_current=saturation_current,
        breakdown_voltage=breakdown_voltage,
        bypass_breakdown_voltage=v_bypass,
        module_open_circuit_voltage=v_oc,
        topology="series",
        bypass_diodes=N_byp,
        nominal_temperature=TNOM,
        circuit_name=circuit_name,
        voltage_step=v_step,
    )


def _resolve_ngspice_output(
    path: str | os.PathLike[str], output_filename: str
) -> Path:
    requested = Path(path).expanduser()
    output = requested if requested.is_file() else requested / output_filename
    output = output.resolve()
    if not output.is_file():
        raise FileNotFoundError(f"ngspice output file does not exist: {output}")
    return output


def _read_ngspice_ascii_raw(
    path: str | os.PathLike[str], output_filename: str = "solar_circuit.out"
) -> pd.DataFrame:
    """Read an ngspice ASCII raw file into a table of named variables."""
    output = _resolve_ngspice_output(path, output_filename)
    lines = output.read_text(encoding="utf-8", errors="replace").splitlines()
    nvars = None
    npoints = None
    variables_start = None
    values_start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("No. Variables:"):
            nvars = int(stripped.partition(":")[2].strip())
        elif stripped.startswith("No. Points:"):
            npoints = int(stripped.partition(":")[2].strip())
        elif stripped == "Variables:":
            variables_start = index + 1
        elif stripped == "Values:":
            values_start = index + 1
            break
    if nvars is None or variables_start is None or values_start is None:
        raise ValueError(f"{output} is not a complete ngspice ASCII raw file.")

    variables: list[str] = []
    for line in lines[variables_start : values_start - 1]:
        fields = line.split()
        if len(fields) >= 2:
            variables.append(fields[1])
    if len(variables) != nvars:
        raise ValueError(
            f"Expected {nvars} ngspice variables, but found {len(variables)} "
            f"in {output}."
        )

    rows: list[list[float]] = []
    point: list[float] = []
    for line in lines[values_start:]:
        fields = line.split()
        if not fields:
            continue
        try:
            value = (
                float(fields[1])
                if len(fields) >= 2 and fields[0].isdigit()
                else float(fields[0])
            )
        except ValueError as error:
            raise ValueError(
                f"Could not parse ngspice value row {line!r} in {output}."
            ) from error
        point.append(value)
        if len(point) == nvars:
            rows.append(point)
            point = []
    if point:
        raise ValueError(f"Incomplete final ngspice sweep point in {output}.")

    dataframe = pd.DataFrame(rows, columns=variables)
    if npoints is not None and len(dataframe) != npoints:
        warnings.warn(
            f"ngspice header reports {npoints} points, but {len(dataframe)} were parsed.",
            RuntimeWarning,
            stacklevel=2,
        )
    if dataframe.empty:
        raise ValueError(f"No sweep values were found in {output}.")
    return dataframe


def _iter_cell_connections(cells_per_module: int, modules: int, topology: str):
    """Yield pvspice cell nodes in the same order used by the netlist builders."""
    for module in range(modules):
        input_node = "1" if module == 0 else f"M{module - 1}OUT"
        module_end = "0" if module == modules - 1 else f"M{module}OUT"
        if topology == "series":
            for cell in range(cells_per_module):
                start = (
                    input_node
                    if cell == 0
                    else _output_node(
                        module, None, cell - 1, cells_per_module - 1, module_end
                    )
                )
                end = _output_node(
                    module, None, cell, cells_per_module - 1, module_end
                )
                diode = str(2 * cell + 2) if module == 0 else f"M{module}C{cell}D"
                yield module, None, cell, module * cells_per_module + cell, start, diode, end
            continue

        cells_per_branch = cells_per_module // 2
        for branch in range(2):
            for cell in range(cells_per_branch):
                start = (
                    input_node
                    if cell == 0
                    else _output_node(
                        module, branch, cell - 1, cells_per_branch - 1, module_end
                    )
                )
                end = _output_node(
                    module, branch, cell, cells_per_branch - 1, module_end
                )
                diode = f"M{module}B{branch}C{cell}D"
                cell_index = module * cells_per_module + branch * cells_per_branch + cell
                yield module, branch, cell, cell_index, start, diode, end


def ngspice_read_cell_level_ivs(
    path: str | os.PathLike[str],
    N_s: int = 60,
    series_resistances: float | list[float] | np.ndarray | None = None,
    output_filename: str = "solar_circuit.out",
    current_col: str = "i(v1)",
    flip_module_current_if_negative: bool = True,
    N_modules: int = 1,
    topology: str = "series",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reconstruct cell IV curves from a pvspice ngspice ASCII raw output.

    The reader supports the node layouts produced by
    :func:`module_to_netlist`, :func:`modules_series_netlist_builder`, and
    :func:`modules_halfcut_netlist_builder`. It therefore handles conventional
    series modules, strings of series modules, half-cut modules, and strings
    of half-cut modules. The explicitly supplied ``N_s``, ``N_modules``, and
    ``topology`` must match the netlist that produced the raw file.

    Parameters
    ----------
    path
        Directory containing the ngspice ASCII raw output, or the output file
        itself.
    N_s
        Total number of cells in each module. For half-cut modules this is the
        total across both parallel branches.
    series_resistances
        One resistance, a per-module sequence of ``N_s`` resistances, a
        full-string sequence of ``N_s * N_modules`` resistances, or ``None``.
        When omitted, cell-current columns contain ``NaN`` or are absent from
        the wide table.
    output_filename
        Output name used when ``path`` is a directory.
    current_col
        Terminal-current variable in the raw output. If it is absent, the
        last available current variable is used and a warning is emitted.
    flip_module_current_if_negative
        Change a negative median terminal current to the positive PV-current
        convention used by pvspice plots.
    N_modules
        Number of series-connected modules represented by the circuit.
    topology
        ``"series"`` for conventional modules or ``"halfcut"`` for modules
        with two parallel cell branches.

    Returns
    -------
    df_cells_long, df_cells_wide, df_raw
        A long-form table with one row per cell and sweep point, a wide-form
        table with per-cell curves, and the complete raw ngspice table. The
        long form identifies ``module``, ``branch`` (``NaN`` for conventional
        series modules), local ``cell``, and full-string ``cell_index``. For a
        string, the wide table also contains ``string_V``, ``string_I``, and
        ``V_module_<n>`` terminal-voltage columns.
    """
    if N_s <= 0 or N_modules <= 0:
        raise ValueError("N_s and N_modules must be positive integers.")
    if topology not in {"series", "halfcut"}:
        raise ValueError("topology must be either 'series' or 'halfcut'.")
    if topology == "halfcut" and N_s % 2:
        raise ValueError("N_s must be even for a half-cut module.")

    output = _resolve_ngspice_output(path, output_filename)
    df_raw = _read_ngspice_ascii_raw(output)
    columns_lower = {column.lower(): column for column in df_raw.columns}
    sweep_column = columns_lower.get("v(v-sweep)", df_raw.columns[0])
    terminal_voltage_column = columns_lower.get("v(1)", sweep_column)
    selected_current = columns_lower.get(current_col.lower())
    if selected_current is None:
        available_currents = [
            column for column in df_raw.columns if column.lower().startswith("i(")
        ]
        if not available_currents:
            raise ValueError(
                f"Could not find {current_col!r} or another current column in {output}."
            )
        selected_current = available_currents[-1]
        warnings.warn(
            f"Using {selected_current!r} because {current_col!r} is not present "
            f"in {output}.",
            RuntimeWarning,
            stacklevel=2,
        )

    sweep_voltage = df_raw[sweep_column].to_numpy(dtype=float)
    terminal_voltage = df_raw[terminal_voltage_column].to_numpy(dtype=float)
    terminal_current = df_raw[selected_current].to_numpy(dtype=float)
    if flip_module_current_if_negative and np.nanmedian(terminal_current) < 0:
        terminal_current = -terminal_current

    resistances = (
        None
        if series_resistances is None
        else _expand_parameter(
            series_resistances, "series_resistances", N_s, N_modules
        )
    )
    if resistances is not None and np.any(resistances == 0):
        raise ValueError("series_resistances cannot contain zero values.")

    def node_voltage(node: str) -> np.ndarray:
        if str(node) == "0":
            return np.zeros(len(df_raw), dtype=float)
        column = columns_lower.get(f"v({str(node).lower()})")
        if column is None:
            raise ValueError(
                f"Missing required node column v({node}) in {output}. "
                "Confirm N_s, N_modules, and topology match the netlist."
            )
        return df_raw[column].to_numpy(dtype=float)

    wide_columns: dict[str, np.ndarray] = {
        "sweep_V": sweep_voltage,
        "module_V": terminal_voltage,
        "module_I": terminal_current,
        "circuit_V": terminal_voltage,
        "circuit_I": terminal_current,
    }
    if N_modules > 1:
        wide_columns["string_V"] = terminal_voltage
        wide_columns["string_I"] = terminal_current
    for module in range(N_modules):
        module_start = "1" if module == 0 else f"M{module - 1}OUT"
        module_end = "0" if module == N_modules - 1 else f"M{module}OUT"
        wide_columns[f"V_module_{module}"] = (
            node_voltage(module_start) - node_voltage(module_end)
        )
        wide_columns[f"I_module_{module}"] = terminal_current

    long_tables: list[pd.DataFrame] = []
    for module, branch, cell, cell_index, start, diode, end in _iter_cell_connections(
        N_s, N_modules, topology
    ):
        start_voltage = node_voltage(start)
        diode_voltage = node_voltage(diode)
        end_voltage = node_voltage(end)
        cell_voltage = start_voltage - end_voltage
        junction_voltage = diode_voltage - end_voltage
        resistance_voltage = start_voltage - diode_voltage
        cell_current = (
            np.full(len(df_raw), np.nan)
            if resistances is None
            else (diode_voltage - start_voltage) / resistances[cell_index]
        )
        if N_modules == 1 and topology == "series":
            label = f"cell_{cell}"
        elif topology == "series":
            label = f"module_{module}_cell_{cell}"
        else:
            label = f"module_{module}_branch_{branch}_cell_{cell}"
        wide_columns[f"V_{label}"] = cell_voltage
        # Preserve the column names returned by the original one-module
        # reader while using unambiguous names for module strings.
        component_label = str(cell) if N_modules == 1 and topology == "series" else label
        wide_columns[f"V_diode_{component_label}"] = junction_voltage
        wide_columns[f"V_rs_{component_label}"] = resistance_voltage
        if resistances is not None:
            wide_columns[f"I_{label}"] = cell_current
        long_tables.append(
            pd.DataFrame(
                {
                    "sweep_V": sweep_voltage,
                    "module_V": terminal_voltage,
                    "module_I": terminal_current,
                    "circuit_V": terminal_voltage,
                    "circuit_I": terminal_current,
                    "module": module,
                    "branch": branch,
                    "cell": cell,
                    "cell_index": cell_index,
                    "V_cell": cell_voltage,
                    "V_diode": junction_voltage,
                    "V_rs": resistance_voltage,
                    "I_cell": cell_current,
                }
            )
        )

    return pd.concat(long_tables, ignore_index=True), pd.DataFrame(wide_columns), df_raw


def ngpsice_read_voltage_current_modules(
    path: str | os.PathLike[str], output_filename: str = "solar_circuit.out"
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Read v(1) and i(v1) from an ngspice ASCII raw file."""
    dataframe = _read_ngspice_ascii_raw(path, output_filename)
    columns_lower = {column.lower(): column for column in dataframe.columns}
    voltage_column = columns_lower.get("v(1)")
    current_column = columns_lower.get("i(v1)")
    if voltage_column is None or current_column is None:
        raise ValueError(
            f"Expected v(1) and i(v1); found {sorted(columns_lower)}."
        )
    voltage = dataframe[voltage_column].to_numpy(dtype=float)
    current = dataframe[current_column].to_numpy(dtype=float)
    return voltage, current, pd.DataFrame({"V": voltage, "I": current})
