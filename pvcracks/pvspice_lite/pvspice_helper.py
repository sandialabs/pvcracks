"""Data-preparation helpers retained for the pvspice_lite compatibility API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_iv(
    iv_path: str | Path,
    savgol_window: int = 5,
    savgol_order: int = 2,
    samples: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """Read, rectify, lightly smooth, and interpolate a two-column IV curve.

    The input CSV must contain Voltage and Current columns. The result has
    increasing voltage and exactly samples evenly spaced points.
    """
    import pvlib
    from scipy.signal import savgol_filter

    if samples < 2:
        raise ValueError("samples must be at least 2.")
    if savgol_window <= savgol_order or savgol_window % 2 == 0:
        raise ValueError("savgol_window must be odd and larger than savgol_order.")

    iv = pd.read_csv(iv_path)
    required = {"Voltage", "Current"}
    if missing := required.difference(iv.columns):
        raise ValueError(f"IV CSV is missing required columns: {sorted(missing)}.")
    voltage, current = pvlib.ivtools.utils.rectify_iv_curve(
        iv["Voltage"], iv["Current"]
    )
    if len(voltage) < savgol_window:
        raise ValueError(
            f"At least {savgol_window} IV points are required for smoothing."
        )
    voltage = savgol_filter(voltage, savgol_window, savgol_order)
    current = savgol_filter(current, savgol_window, savgol_order)
    interpolated_voltage = np.linspace(voltage[0], voltage[-1], samples)
    interpolated_current = np.interp(interpolated_voltage, voltage, current)
    return interpolated_voltage, interpolated_current


def read_el(path: str | Path) -> np.ndarray:
    """Read an electroluminescence image as a grayscale NumPy array."""
    import cv2

    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read EL image: {image_path}")
    return image


def cells_to_module(
    cells: pd.DataFrame,
    input_size: tuple[int, int],
    *,
    rows: int = 6,
    columns: int = 10,
    path_column: str = "ELPath",
    root: str | Path | None = None,
) -> np.ndarray:
    """Tile cell images named in a DataFrame into a module image.

    Input size follows OpenCV width, height ordering. Relative paths are
    resolved against root when supplied, otherwise against the working folder.
    """
    if path_column not in cells:
        raise ValueError(f"cells must contain a {path_column!r} column.")
    if len(cells) > rows * columns:
        raise ValueError("More cell images were provided than the module layout holds.")
    width, height = input_size
    if width <= 0 or height <= 0:
        raise ValueError("input_size values must be positive.")

    import cv2

    module = np.zeros((height * rows, width * columns), dtype=np.uint8)
    base = Path.cwd() if root is None else Path(root)
    for index, image_value in enumerate(cells[path_column]):
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = base / image_path
        image = cv2.resize(read_el(image_path), (width, height))
        row, column = divmod(index, columns)
        module[
            row * height : (row + 1) * height, column * width : (column + 1) * width
        ] = image
    return module


def extract_parameters(cells: pd.DataFrame) -> list[dict[str, float]]:
    """Return one single-diode parameter dictionary per DataFrame row."""
    required = ("Rs", "Rsh", "I", "Is", "N")
    if missing := set(required).difference(cells.columns):
        raise ValueError(f"cells is missing required columns: {sorted(missing)}.")
    return [
        {name: float(row[name]) for name in required} for _, row in cells.iterrows()
    ]


# Legacy names remain importable while documentation and new notebooks use the
# clear snake_case API above.
Read_IV = read_iv
Read_EL = read_el
cells2Mod = cells_to_module
Extract_Params = extract_parameters
