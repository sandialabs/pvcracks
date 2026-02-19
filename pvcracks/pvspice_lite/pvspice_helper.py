# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:18:35 2024

Helper functions for reading in ELs and IVs.

@author: bbyford, nrjost
"""


def read_IV(IVPath, sav_window=5, sav_order=2, samples=400):
    """
    Read an IV curve from a CSV file, rectify it, smooth it, and interpolate.

    Parameters
    ----------
    IVPath : str
        Path to the CSV file containing IV data with columns 'Voltage' and 'Current'.
    sav_window : int, optional
        Window length for the Savitzky-Golay filter (default is 5).
    sav_order : int, optional
        Polynomial order for the Savitzky-Golay filter (default is 2).
    samples : int, optional
        Number of voltage sample points to interpolate (default is 400).

    Returns
    -------
    vn : numpy.ndarray
        Evenly spaced voltage values after rectification, smoothing, and interpolation.
    Ic : numpy.ndarray
        Current values corresponding to vn after smoothing and interpolation.
    """
    import numpy as np
    import pandas as pd
    import pvlib
    import scipy

    # Read IV
    IV = pd.read_csv(IVPath)
    # Ensure decreasing signal that is positive
    Vc, Ic = pvlib.ivtools.utils.rectify_iv_curve(IV["Voltage"], IV["Current"])
    # lightly smooth with savgol
    Vc = scipy.signal.savgol_filter(Vc, sav_window, sav_order)
    Ic = scipy.signal.savgol_filter(Ic, sav_window, sav_order)

    vn = np.squeeze(np.linspace(Vc[0], np.max(Vc), samples))

    # linear interpret to ensure the generated IV fits the points provided
    Ic = np.interp(vn, Vc, Ic)
    return (vn, Ic)


def read_EL(path):
    """
    Tile individual cell EL images into a single module image.

    Parameters
    ----------
    slice : pandas.DataFrame
        DataFrame with an 'ELPath' column pointing to each cell image.
    input_size : tuple of int
        Desired size (height, width) for each cell image.
    rows : int, optional
        Number of rows in the module layout (default is 6).
    cols : int, optional
        Number of columns in the module layout (default is 10).

    Returns
    -------
    img : numpy.ndarray
        Combined module image formed by tiling resized cell images.
    """
    import cv2

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def cells2Mod(slice, input_size, rows=6, cols=10):
    """
    Tile individual cell EL images into a single module image.

    Parameters
    ----------
    slice : pandas.DataFrame
        DataFrame with an 'ELPath' column pointing to each cell image.
    input_size : tuple of int
        Desired size (height, width) for each cell image.
    rows : int, optional
        Number of rows in the module layout (default is 6).
    cols : int, optional
        Number of columns in the module layout (default is 10).

    Returns
    -------
    img : numpy.ndarray
        Combined module image formed by tiling resized cell images.
    """
    import os

    import cv2
    import numpy as np

    rc = 0
    cc = 0
    img = np.zeros((input_size[0] * rows, input_size[1] * cols))

    for ind, row in slice.iterrows():
        cell = read_EL(f"{os.getcwd()}{row.ELPath}")
        cell = cv2.resize(cell, input_size)

        img[rc : rc + input_size[0], cc : cc + input_size[1]] = cell

        cc += input_size[1]

        if cc >= img.shape[1]:
            cc = 0
            rc += input_size[0]
    return img


def extract_params(slice):
    """
    Extract single-diode model parameters from a DataFrame.

    Parameters
    ----------
    slice : pandas.DataFrame
        DataFrame containing one or more rows with the following columns:
        - Rs  : series resistance
        - Rsh : shunt resistance
        - I   : photo-generated current
        - Is  : saturation current
        - N   : diode ideality factor

    Returns
    -------
    Params : list of dict
        A list of parameter dictionaries, one per row in `slice`. Each dict has keys:
        'Rs', 'Rsh', 'I', 'Is', and 'N', with values taken from the corresponding row.
    """
    Params = []
    for ind, row in slice.iterrows():
        Params.append(
            {"Rs": row.Rs, "Rsh": row.Rsh, "I": row.I, "Is": row.Is, "N": row.N}
        )
    return Params
