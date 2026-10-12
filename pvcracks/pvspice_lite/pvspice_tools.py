# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:18:35 2024

Build pv module netlists, run ngspice and read in output.

@author: nrjost, bbyford
"""

from pathlib import Path
import subprocess


def run_ngspice(ngpsice_path, circuit_file):
    """
    Run ngspice using path and netlist/circuit    
    
    Parameters
    ----------
    ngpsice_path : str
        path to ngspice, example: 'C:/Spice64/bin/ngspice.exe'
    circuit_file : str
        path to netlist, example: 'C:/Spice64/test/solar_circuit.cir'

    Returns
    -------
    Nothing; circuit outputs are written by the netlist to explicit paths.
    
    Notes
    ------
    To install ngspice follow the instructions here:
        https://ngspice.sourceforge.io/index.html

    """
    executable = Path(ngpsice_path).expanduser().resolve()
    circuit = Path(circuit_file).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"ngspice executable does not exist: {executable}")
    if not circuit.is_file():
        raise FileNotFoundError(f"SPICE circuit file does not exist: {circuit}")
    return subprocess.run(
        [str(executable), "-b", str(circuit)],
        check=True,
        cwd=str(circuit.parent),
        text=True,
        capture_output=True,
    )

    
def ngpsice_read_voltage_current_modules(path):
    """
    Read the volatage and current from the ngspice output file
    
    Parameters
    ----------
    path : str
        path where output of spice simulation file is saved

    Returns
    -------
    voltage : float, array
        voltage values
    current : float, array
        current values
    df_iv : float, pandas DataFrame
        IV values
    
    Notes
    -------
    If halfcut is true output IV for a halfcut module interconnection, esle for a series interconnection with full size cells
    This code can be further modified to show current and voltage of specific components such as the bypass diodes, the output file has all nodes of the simulation

    """
    import pandas as pd

    voltage = []
    current = []
    flag = False
    df_iv = pd.DataFrame()
    # Extract data from NGSPICE output file
    with open(path, "r") as output_file:
        lines = output_file.readlines()
        for nl, line in enumerate(lines):
            # print(line)
            
            if flag:
                if line =='\n':
                    current.append(float(lines[nl-1]))
                elif line.split("\t")[0]!='':
                    voltage.append(float(line.split('\t')[1]))              
                    
            if "Values" in line:
                flag=True    
                
                
    df_iv['V'] = voltage
    df_iv['I'] = current
    
    return voltage, current, df_iv

def Create_Cell_NetCode(params, a, b, c):
    """
    Build and simulate a mini-module SPICE circuit, then return its IV curve.

    Parameters
    ----------
    params : list of dict
        List of dictionaries, each containing single-diode parameters for one cell.
    file_path : str
        Directory where the SPICE circuit and output files will be written.
    V : array-like
        Voltage points for the DC sweep.
    V_Step : float, optional
        Step size for the DC voltage sweep (default is 0.001).
    file_name : str, optional
        Suffix for output file names (default is empty).

    Returns
    -------
    vn : numpy.ndarray
        Interpolated voltage values from the simulation output.
    fitted_current : numpy.ndarray
        Interpolated current values corresponding to vn.
    """
    
    elementary_diode = f"""X{a} {a} {b} {c}    Cell Rs={params['Rs']} Rp={params['Rsh']} I={params['I']} BV={30} N={params['N']} Is={params['Is']}
"""
    return(elementary_diode)

def MiniMod_Spice(params, file_path, V, V_Step=0.001, file_name ='', spicepath=''):
    """
    Build and simulate a mini-module SPICE circuit, then return its IV curve.

    Parameters
    ----------
    params : list of dict
        List of dictionaries, each containing single-diode parameters for one cell.
    file_path : str
        Directory where the SPICE circuit and output files will be written.
    V : array-like
        Voltage points for the DC sweep.
    V_Step : float, optional
        Step size for the DC voltage sweep (default is 0.001).
    file_name : str, optional
        Suffix for output file names (default is empty).

    Returns
    -------
    vn : numpy.ndarray
        Interpolated voltage values from the simulation output.
    fitted_current : numpy.ndarray
        Interpolated current values corresponding to vn.
    """
    import pvlib
    import numpy as np

    directory = Path(file_path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    circuit_path = directory / f"solar_circuit{file_name}.cir"
    output_path = directory / f"solar_circuit{file_name}.out"
    text_path = directory / f"solar_circuit{file_name}.txt"
    cell_model_path = Path(__file__).resolve().with_name("SolarCell.cp")
    
    N_s = len(params)
    Measurment_offset = {'I': 0,
                         'Is': 0,
                         'Rs': 0.0035873181014406622,
                         'Rsh': 0,
                         'N': 0}
    # Define nominal temperature
    TNOM=25
    # Crete Cell netlist for series connections
    elementary_diode_netlist = ''
    
        
    for n in range(N_s-1):
        Cell_Params=params[n].copy()
        
        Cell_Params['I']   -= Measurment_offset['I']
        Cell_Params['Is']  -= Measurment_offset['Is']
        Cell_Params['Rs']  -= Measurment_offset['Rs']
        Cell_Params['Rsh'] -= Measurment_offset['Rsh']
        Cell_Params['N']   -= Measurment_offset['N']
        
        
        a = n+n+1
        b = n+n+2
        c = n+n+3
        elementary_diode=Create_Cell_NetCode(Cell_Params,a,b,c)
        
        elementary_diode_netlist+=elementary_diode
    if N_s==1:
        n=0
    else:
        n+=1
    Cell_Params=params[n].copy()
    
        
    
    a = n+n+1
    b = n+n+2
    # last node has to end at ground
    c = 0
    elementary_diode=Create_Cell_NetCode(Cell_Params,a,b,c)
    elementary_diode_netlist+=elementary_diode

    # Create Circuit Content
    circuit_content = f"""
* Import the diode model
.include {cell_model_path.as_posix()}

* Define the DC sweep
V1 1 0 DC 0V

{elementary_diode_netlist}

.control
* set the sweep for the apropriate voltage range in steps of V_Step
dc V1 {V[0]} {V[-1]+0.1} {V_Step}
set filetype=ascii 
write {output_path.as_posix()}
write {text_path.as_posix()}

quit
.endc
"""
    circuit_path.write_text(circuit_content, encoding="utf-8")
    
    # Simulate and read output file
    run_ngspice(spicepath, circuit_path)
    
    _, _, df_iv = ngpsice_read_voltage_current_modules(output_path)
    
    # use PVLIB to snip any negative values
    Vp,Ip = pvlib.ivtools.utils.rectify_iv_curve(df_iv['V'], df_iv['I'])
    
    vn= np.squeeze(np.linspace(V[0],np.max(V), len(V) ))
    
    # linear interpret to ensure the generated IV fits the points provided 
    fitted_current = np.interp(vn, Vp, Ip)
    
    return(vn, fitted_current)
