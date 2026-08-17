# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:18:35 2024
Last edit: April 28th 2025

Simulate modules using spice

@author: nrjost
"""

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
    Nothing the output is saved to the path given in module_to_netlist
    
    Notes
    ------
    To install ngspice follow the instructions here:
        https://ngspice.sourceforge.io/index.html

    """
    import subprocess
    ngspice_command = [ngpsice_path, "-b", circuit_file]
    try:
        subprocess.run(ngspice_command)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Check path for ngspice, current: {ngpsice_path}")
        print(f"Check path for circuit file, current: {circuit_file}")
    
def run_xyce(xyce_path, outputfile, circuit_file):
    """
    Run xyce using path and netlist/circuit    
    
    Parameters
    ----------
    xyce_path : str
        path to xyce, example: 'C:/Xyce_7.8/bin/Xyce.exe'
    circuit_file : str
        path to netlist, example: 'C:/Xyce_7.8/test/solar_circuit.cir'

    Returns
    -------
    Nothing the output is saved to the path given in module_to_netlist
    
    Notes
    ------
    To install xyce follow the instructions here:
        https://xyce.sandia.gov/
    """
    import subprocess
    xyce_command = [xyce_path, "-r", outputfile, circuit_file]
    try:
        subprocess.run(xyce_command)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Check path for Xyce, current: {xyce_path}")
        print(f"Check path for circuit file, current: {circuit_file}")


def plot_solar_module(cell_currents, cells_wide, cells_long, font=18, save=True):
    """
    Plot module with cell currents as color bar
    
    Parameters
    ----------
    cell_currents : array
        currents in A for each solar cell in the module
    cells_wide : int
        number of cells along short edge of module
    cells_long : int
        number of cells along long edge of module
    font : int
        fontsize for plot
    save : bool
        save the plot to the current path

    Returns
    -------
    Does not return anything but plots the cell current map and saves the plot to current path

    """
    import numpy as np
    from matplotlib import pyplot as plt
    
    # Reshape the list of cell currents into a 2D array
    cell_currents_array = np.array(cell_currents).reshape(cells_wide, cells_long)
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    
    # Create a heatmap of the cell currents
    plt.imshow(cell_currents_array, cmap='plasma', aspect='auto')
    
    # Add colorbar for reference
    cbar = plt.colorbar(label='Current (A)')
    cbar.ax.yaxis.label.set_fontsize(font)
    
    # Add labels and title
    plt.xlabel('Cell number (Long Edge)', fontsize=font)
    plt.ylabel('Cell number (Short Edge)', fontsize=font)
    plt.title('Solar Module Current Distribution', fontsize=font+2)
    
    # Show the plot
    if save:
        plt.savefig('module_current_distr.png')
    plt.show()

def module_to_netlist(path, N_s, cell_currents, series_resistances, shunt_resistances, ideality_factor, saturation_current, breakdown_voltage, v_bypass, v_oc, TNOM=25, circuit_name='solar_circuit'):
    """
    Create the module netlist for ngspice if N_s > 100 assumes half cut module with butterfly interconnection, if < 100 series interconnection.
    Both interconnections have 3 bypass diodes. Saves circuit file in path.
    
    Parameters
    ----------
    path : str
        path where to save the circuit file and output of spice simulation
    N_s : int
        number of cells in the module if N_s>100 halfcut
    cell_currents : float, array
        photocurrent of each solar cell, array of 1 or N_s
    series_resistances : float, array
        series resistance of each cell, array of 1 or N_s
    shunt_resistances : float, array
        shunt resistance/ parallel resitance of each cell, array of 1 or N_s
    ideality_factor : float
        ideality factor of the diode
    saturation_current : float
        saturation current of the diode
    breakdown_voltage : float
        breakdown voltage for each cell, e.g. HJT = 30V, Al-BSF = 15V, PERC = 18V, TOPcon = 25V, IBC = 3V
    v_bypass : float
        bypass diode voltage
    v_oc : float
        state open circuit voltage for voltage sweep limit which is v_oc+1
    TNOM : int
        SPICE nominal temperature can be kept at 25 as diode parameters are adjusted with pvlib
    circuit_name : str
        name of circuit file to be saved, default is solar_circuit

    Returns
    -------
    circuit_content : str
        netlist for ngspice simulation
    half_cut : bool
        if cells are halfcut or not
        
    Notes
    -------
    bypass diode voltage, can be estimated with
        v_bypass = abs(breakdown_voltage - ((N_s/3) - 1)*v_oc/(N_s/3) #N_s/3 because we know there are 3 strings
        from ST AN3432 - Doc ID 019041 Rev 1 available @ www.st.com

    """
    
    #Check variable lengths and make into arrays of N_s
    if len(cell_currents) == 1:
        print(f"cell_currents made into array of N_s = {N_s}")
        cell_currents = [cell_currents[0]] * N_s
    if len(series_resistances) == 1:
        print(f"series_resistances made into array of N_s = {N_s}")
        series_resistances = [series_resistances[0]] * N_s
    if len(shunt_resistances) == 1:
        print(f"shunt_resistances made into array of N_s = {N_s}")
        shunt_resistances = [shunt_resistances[0]] * N_s
            
    # Create NGSPICE circuit file
    cells_as_elem_diode = ''
    path = path
    
    if N_s <= 90:
        print(f"All series interconnection of {N_s} solar cells")
        halfcut=False
        for n in range(0, int(N_s)): #enumerates cell number
            photocurrent = cell_currents[n]
            resistance_series = series_resistances[n]
            resistance_shunt = shunt_resistances[n]
            rsn = n
            rpn = n
            dn = n
            ni = n
            rspos = [n+n+1, n+n+2]
            rppos = [n+n+2, n+n+3]
            dpos = [n+n+2, n+n+3]
            ipos = [n+n+3, n+n+2] #inverse current
            if n == int(N_s-1):
                elementary_diode = f"""
*elementary diode model cell {n} byp
Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} 0 {resistance_shunt}
D{dn} {dpos[0]} 0 sd1
I{ni} 0 {ipos[1]} {photocurrent}
*bypass diode
Dby{dn} {0} {int(dpos[0]-N_s/3*2+1)} sdby
""" #two diode would include *D2 2 0 sd2
            elif n == 0:
                elementary_diode = f"""
*elementary diode model
Rs{rsn} 1 2 {resistance_series}
Rp{rpn} 2 3 {resistance_shunt}
D{dn} 2 3 sd1
I{ni} 3 2 {photocurrent}
"""        
            elif n == (N_s/3-1):
                elementary_diode = f"""
*elementary diode model cell {n} byp
Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} {rppos[1]} {resistance_shunt}
D{dn} {dpos[0]} {dpos[1]} sd1
I{ni} {ipos[0]} {ipos[1]} {photocurrent}
*bypass diode
Dby{dn} {dpos[1]} {int(dpos[1]-N_s/3*2)} sdby
*Rt{rpn} 1 {rppos[1]} 0
"""   
            elif n == ((N_s/3)*2)-1:
                elementary_diode = f"""
*elementary diode model cell {n} byp
Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} {rppos[1]} {resistance_shunt}
D{dn} {dpos[0]} {dpos[1]} sd1
I{ni} {ipos[0]} {ipos[1]} {photocurrent}
*bypass diode
Dby{dn} {dpos[1]} {int(dpos[1]-N_s/3*2)} sdby
"""        
            else:
                elementary_diode = f"""
*elementary diode model
Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} {rppos[1]} {resistance_shunt}
D{dn} {dpos[0]} {dpos[1]} sd1
I{ni} {ipos[0]} {ipos[1]} {photocurrent}
"""
            cells_as_elem_diode = cells_as_elem_diode + elementary_diode
        #taken out of diode model:  RS=0 CJO=1e-12 M=0.33
        #second diode out: *.model sd2 D (IS=1e-18 N=2 EG=1.1 RS=0 CJO=1e-12 M=0.33)
    elif 90 < N_s < 200:
        print(f"Half-cut interconnection of {N_s} solar cells")
        halfcut=True
        # ideality_factor = ideality_factor*2
        print(f"Ideality factor = {ideality_factor}")
        for n in range(0, int(N_s)): #enumerates cell number
            if n < N_s/2:
                photocurrent = cell_currents[n]#/2 #half-cut cell half module current
                resistance_series = series_resistances[n]#*4 #adjustment of IV curve
                resistance_shunt = shunt_resistances[n]#*4 #half cut
                rsn = n
                rpn = n
                dn = n
                ni = n
                rspos = [n+n+1, n+n+2]
                rppos = [n+n+2, n+n+3]
                dpos = [n+n+2, n+n+3]
                ipos = [n+n+3, n+n+2] #inverse current
                if n == ((N_s/6)*3)-1:
                    elementary_diode = f"""
*elementary diode model cell {n} byp
Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} 0 {resistance_shunt}
D{dn} {dpos[0]} 0 sd1
I{ni} 0 {ipos[1]} {photocurrent}
*bypass diode
Dby{dn} {0} {int(dpos[0]-N_s/3+1)} sdby
""" #two diode would include *D2 2 0 sd2
                elif n == 0:
                    elementary_diode = f"""
*elementary diode model cell {n}
Rs{rsn} 1 2 {resistance_series}
Rp{rpn} 2 3 {resistance_shunt}
D{dn} 2 3 sd1
I{ni} 3 2 {photocurrent}
"""  
                elif n == (N_s/6-1):
                    elementary_diode = f"""
*elementary diode model cell {n} byp
Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} {rppos[1]} {resistance_shunt}
D{dn} {dpos[0]} {dpos[1]} sd1
I{ni} {ipos[0]} {ipos[1]} {photocurrent}
*bypass diode
Dby{dn} {dpos[1]} {int(dpos[1]-N_s/3)} sdby
*Rt{rpn} 1 {rppos[1]} 0
"""   
                elif n == ((N_s/6)*2)-1:
                    elementary_diode = f"""
*elementary diode model cell {n} byp
Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} {rppos[1]} {resistance_shunt}
D{dn} {dpos[0]} {dpos[1]} sd1
I{ni} {ipos[0]} {ipos[1]} {photocurrent}
*bypass diode
Dby{dn} {dpos[1]} {int(dpos[1]-N_s/3)} sdby
"""        
                else:
                    elementary_diode = f"""
*elementary diode model cell {n}
Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} {rppos[1]} {resistance_shunt}
D{dn} {dpos[0]} {dpos[1]} sd1
I{ni} {ipos[0]} {ipos[1]} {photocurrent}
"""
            elif n >= N_s/2:
                photocurrent = cell_currents[n] #half-cut cell half module current
                resistance_series = series_resistances[n] #adjustment of IV curve
                resistance_shunt = shunt_resistances[n] #half cut
                rsn = n
                rpn = n
                dn = n
                ni = n
                rspos = [n+n+1, n+n+2]
                rppos = [n+n+2, n+n+3]
                dpos = [n+n+2, n+n+3]
                ipos = [n+n+3, n+n+2] #inverse 
                if n == int(N_s-1):
                    elementary_diode = f"""
*elementary diode model cell {n} byp
Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} 0 {resistance_shunt}
D{dn} {dpos[0]} 0 sd1
I{ni} 0 {ipos[1]} {photocurrent}
*bypass diode
""" #two diode would include *D2 2 0 sd2
                elif n == N_s/2:
                    elementary_diode = f"""
*elementary diode model cell {n}
Rs{rsn} 1 {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} {rppos[1]} {resistance_shunt}
D{dn} {dpos[0]} {dpos[1]} sd1
I{ni} {ipos[0]} {ipos[1]} {photocurrent}
"""        
                elif n == ((N_s/6)*4)-1:
                    endpos = int(n-(N_s/3-2))
                    elementary_diode = f"""
*elementary diode model cell {n} byp
Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} {endpos} {resistance_shunt}
D{dn} {dpos[0]} {endpos} sd1
I{ni} {endpos} {ipos[1]} {photocurrent}
*bypass diode
"""   
                elif n == ((N_s/6)*4):
                    firstpos = int(n-(N_s/3-1))
                    elementary_diode = f"""
*elementary diode model cell {n}
Rs{rsn} {firstpos} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} {rppos[1]} {resistance_shunt}
D{dn} {dpos[0]} {dpos[1]} sd1
I{ni} {ipos[0]} {ipos[1]} {photocurrent}
"""   
                elif n == ((N_s/6)*5)-1:
                    endpos = int(n-(N_s/6-2))
                    elementary_diode = f"""
*elementary diode model cell {n} byp
Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} {endpos} {resistance_shunt}
D{dn} {dpos[0]} {endpos} sd1
I{ni} {endpos} {ipos[1]} {photocurrent}
*bypass diode
"""
                elif n == ((N_s/6)*5):
                    firstpos = int(n-(N_s/6-1))
                    elementary_diode = f"""
*elementary diode model cell {n}
Rs{rsn} {firstpos} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} {rppos[1]} {resistance_shunt}
D{dn} {dpos[0]} {dpos[1]} sd1
I{ni} {ipos[0]} {ipos[1]} {photocurrent}
"""        
                else:
                    elementary_diode = f"""
*elementary diode model cell {n}
Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} {rppos[1]} {resistance_shunt}
D{dn} {dpos[0]} {dpos[1]} sd1
I{ni} {ipos[0]} {ipos[1]} {photocurrent}
"""
                
            cells_as_elem_diode = cells_as_elem_diode + elementary_diode
    else:
        print("No available interconnection, to-do: shingles")
        
        
    circuit_content = f"""
* One-diode solar module simulation

* Define the diode model
.model sd1 D (IS={saturation_current} N={ideality_factor} EG=1.12 BV={breakdown_voltage})
.model sdby D (IS=1e-7 N=1 BV={v_bypass} IBV=20)

* Define the DC sweep
V1 1 0

{cells_as_elem_diode}

.temp = {TNOM}
.options tnom={TNOM}
.control
dc V1 0 {v_oc+1} 0.1
print dc v(1) i(D1)
set filetype=ascii 
write {path}/solar_circuit.out
write {path}/solar_circuit.txt

quit
.endc
"""

    with open(f"{path}/{circuit_name}.cir", "w") as circuit_file:
        circuit_file.write(circuit_content)
    
    return circuit_content, halfcut

# def ngpsice_read_voltage_current(path, halfcut, N_s): #Old version
#     import pandas as pd
#     """
#     Read the volatage and current from the ngspice output file
    
#     Parameters
#     ----------
#     path : str
#         path where output of spice simulation file is saved
#     half_cut : bool
#         if cells are halfcut or not
#     N_s : int
#         number of solar cells

#     Returns
#     -------
#     voltage : float, array
#         voltage values
#     current : float, array
#         current values
#     df_iv : float, pandas DataFrame
#         IV values
    
#     Notes
#     -------
#     If halfcut is true output IV for a halfcut module interconnection, esle for a series interconnection with full size cells
#     This code can be further modified to show current and voltage of specific components such as the bypass diodes, the output file has all nodes of the simulation

#     """
#     voltage = []
#     current = []
#     df_iv = pd.DataFrame()
#     # Extract data from NGSPICE output file
#     with open(path+"/solar_circuit.out", "r") as output_file:
#         lines = output_file.readlines()
#         for nl, line in enumerate(lines):
#             if "Values" in line:
#                 if halfcut:
#                     for nv in range(nl+1 ,len(lines)-2, N_s*2): #was 6 before bypassD
#                         # print(nv+N_s+4)
#                         # df = pd.DataFrame()
#                         voltage.append(float(lines[nv].split()[1]))
#                         current.append(float(lines[nv+(N_s*2)-2]))
#                 else:
#                     for nv in range(nl+1 ,len(lines)-2, N_s*2+3): #was 6 before bypassD
#                         print(nv+N_s+4)
#                         # df = pd.DataFrame()
#                         voltage.append(float(lines[nv].split()[1]))
#                         current.append(float(lines[nv+(N_s*2)+1]))
#     df_iv['V'] = voltage
#     df_iv['I'] = current
    
#     return voltage, current, df_iv


    
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
    with open(path+"/solar_circuit.out", "r") as output_file:
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


def modules_halfcut_netlist_builder(path, N_s, N_modules, cell_currents_, series_resistances_, shunt_resistances_,
                             ideality_factor, saturation_current, breakdown_voltage, v_bypass, v_oc,
                             N_byp=3, TNOM=25, circuit_name='solar_circuit'):
    """
    Build a netlist for a half-cut solar module with bypass diodes.
    
    Parameters
    ----------
    path : str
        Path where to save the circuit file and output of spice simulation.
    N_s : int
        Number of solar cells in each module.
    N_modules : int
        Number of modules.
    cell_currents_ : float, array
        Photocurrent of each solar cell.
    series_resistances_ : float, array
        Series resistance of each cell.
    shunt_resistances_ : float, array
        Shunt resistance of each cell.
    ideality_factor : float
        Ideality factor of the diode.
    saturation_current : float
        Saturation current of the diode.
    breakdown_voltage : float
        Breakdown voltage for each cell.
    v_bypass : float
        Bypass diode voltage.
    v_oc : float
        Open circuit voltage for voltage sweep limit.
    N_byp : int
        Number of bypass diodes.
    TNOM : int
        SPICE nominal temperature.
    circuit_name : str
        Name of circuit file to be saved.
    
    Returns
    -------
    circuit_content : str
        Netlist for ngspice simulation.
    """

    import numpy as np

    #Make variables length N_s*N_modules if they are not already that
    if not len(cell_currents_) == N_s*N_modules: 
        cell_currents = np.tile(cell_currents_, N_modules)
    else:
        cell_currents = cell_currents_
    if not len(series_resistances_) == N_s*N_modules:        
        series_resistances = np.tile(series_resistances_, N_modules)
    else:
        series_resistances = series_resistances_
    if not len(shunt_resistances_) == N_s*N_modules:
        shunt_resistances = np.tile(shunt_resistances_, N_modules)
    else:
        shunt_resistances = shunt_resistances_

    #Checking
    print(f"Length of all variables: Rsh {len(shunt_resistances)}, Rs {len(series_resistances)}, Iph {len(cell_currents)}")
            
    # Create NGSPICE circuit file
    cells_as_elem_diode = ''
    path = path
    
    #String end nodes
    N_nodes = N_modules*N_s + 2
    N_cells = N_modules*N_s
    N_cellstring = N_modules/N_byp
    N_sjunction = N_modules*N_byp
    first_break = N_cellstring + 2
    string_junctions = np.arange(N_cells/2/N_sjunction, N_cells/2+1, N_cells/2/N_sjunction)
    
    
    elementary_diode = f"""
    *elementary diode model cell {0}
    Rs{0} {1} {2} {series_resistances[0]}
    Rp{0} {2} {3} {shunt_resistances[0]}
    D{0} {2} {3} sd1
    I{0} {3} {2} {cell_currents[0]}
    """  
    cells_as_elem_diode = cells_as_elem_diode + elementary_diode
    
    for n in range(1, int(N_cells/2)-1): #enumerates all cells
        photocurrent = cell_currents[n]#/2 #half-cut cell half module current
        resistance_series = series_resistances[n]#*4 #adjustment of IV curve
        resistance_shunt = shunt_resistances[n]#*4 #half cut
        
        rsn = n
        rpn = n
        dn = n
        ni = n
    
        rspos = [n+n+1, n+n+2]
        rppos = [n+n+2, n+n+3]
        dpos = [n+n+2, n+n+3]
        ipos = [n+n+3, n+n+2] #inverse current
    
        elementary_diode = f"""
    *elementary diode model cell {n}
    Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
    Rp{rpn} {rppos[0]} {rppos[1]} {resistance_shunt}
    D{dn} {dpos[0]} {dpos[1]} sd1
    I{ni} {ipos[0]} {ipos[1]} {photocurrent}
        """  
        cells_as_elem_diode = cells_as_elem_diode + elementary_diode
    
    n +=1
    rsn = n
    rpn = n
    dn = n
    ni = n
    elementary_diode = f"""
        *elementary diode model cell {n}
        Rs{rsn} {n+n+1} {n+n+2} {series_resistances[n]} 
        Rp{rpn} {n+n+2} {0} {shunt_resistances[n]}
        D{dn} {n+n+2} {0} sd1
        I{ni} {0} {n+n+2} {cell_currents[n]}
        """  
    cells_as_elem_diode = cells_as_elem_diode + elementary_diode
    
    n +=1
    rsn = n
    rpn = n
    dn = n
    ni = n
    elementary_diode = f"""
        *elementary diode model cell {n}
        Rs{rsn} {1} {n+n+2} {series_resistances[n]} 
        Rp{rpn} {n+n+2} {n+n+3} {shunt_resistances[n]}
        D{dn} {n+n+2} {n+n+3} sd1
        I{ni} {n+n+3} {n+n+2} {cell_currents[n]}
        """  
    cells_as_elem_diode = cells_as_elem_diode + elementary_diode
    flag=False
    count_stringjunctions = 0
    for n in range(int(N_cells/2)+1, int(N_cells)-1): #enumerates all cells
        photocurrent = cell_currents[n]#/2 #half-cut cell half module current
        resistance_series = series_resistances[n]#*4 #adjustment of IV curve
        resistance_shunt = shunt_resistances[n]#*4 #half cut
    
        if (n+1)%(N_cells/2/N_sjunction) == 0:        
            node = n+n
            rsn = n
            rpn = n
            dn = n
            ni = n
    
            rspos = [node+1 , node+2] 
            rppos = [node+2 , int(string_junctions[count_stringjunctions]*2+1)] 
            dpos = [node+2 , int(string_junctions[count_stringjunctions]*2+1)] 
            ipos = [int(string_junctions[count_stringjunctions]*2+1), node+2]
            
            elementary_diode = f"""
    *elementary diode model cell {n} bypass
    Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
    Rp{rpn} {rppos[0]} {rppos[1]} {resistance_shunt}
    D{dn} {dpos[0]} {dpos[1]} sd1
    I{ni} {ipos[0]} {ipos[1]} {photocurrent}
            """  
            cells_as_elem_diode = cells_as_elem_diode + elementary_diode
            flag = True
        elif flag:
            node = n+n
            rsn = n
            rpn = n
            dn = n
            ni = n
            
            rspos = [int(string_junctions[count_stringjunctions]*2+1), n+n+2]
            rppos = [n+n+2, n+n+3]
            dpos = [n+n+2, n+n+3]
            ipos = [n+n+3, n+n+2] #inverse current
            
            elementary_diode = f"""
    *elementary diode model cell {n} bypass
    Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
    Rp{rpn} {rppos[0]} {rppos[1]} {resistance_shunt}
    D{dn} {dpos[0]} {dpos[1]} sd1
    I{ni} {ipos[0]} {ipos[1]} {photocurrent}
            """  
            cells_as_elem_diode = cells_as_elem_diode + elementary_diode
            count_stringjunctions += 1
            flag=False
    
        else:
            photocurrent = cell_currents[n]#/2 #half-cut cell half module current
            resistance_series = series_resistances[n]#*4 #adjustment of IV curve
            resistance_shunt = shunt_resistances[n]#*4 #half cut
            
            rsn = n
            rpn = n
            dn = n
            ni = n
        
            rspos = [n+n+1, n+n+2]
            rppos = [n+n+2, n+n+3]
            dpos = [n+n+2, n+n+3]
            ipos = [n+n+3, n+n+2] #inverse current
        
            elementary_diode = f"""
    *elementary diode model cell {n}
    Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
    Rp{rpn} {rppos[0]} {rppos[1]} {resistance_shunt}
    D{dn} {dpos[0]} {dpos[1]} sd1
    I{ni} {ipos[0]} {ipos[1]} {photocurrent}
            """  
            cells_as_elem_diode = cells_as_elem_diode + elementary_diode
    
    n += 1
    rsn = n
    rpn = n
    dn = n
    ni = n
    elementary_diode = f"""
        *elementary diode model cell {n}
        Rs{rsn} {n+n+1} {n+n+2} {series_resistances[n]} 
        Rp{rpn} {n+n+2} {0} {shunt_resistances[n]}
        D{dn} {n+n+2} {0} sd1
        I{ni} {0} {n+n+2} {cell_currents[n]}
        """  
    cells_as_elem_diode = cells_as_elem_diode + elementary_diode
    
    
    node = int(string_junctions[0]*2+1)
    dbyn = 0
    elementary_diode=f'''
    *bypass diode
    Dby{dbyn} {node} {1} sdby
    '''
    cells_as_elem_diode = cells_as_elem_diode + elementary_diode
    
    for pos, junk in enumerate(string_junctions[:-1]):
        
        node = int(junk*2+1)
        second = string_junctions[pos+1]
        node2 = int(second*2+1)
    
        dbyn = pos+1
    
        elementary_diode=f'''
        Dby{dbyn} {node2} {node} sdby
        '''
        cells_as_elem_diode = cells_as_elem_diode + elementary_diode
    elementary_diode=f'''
    Dby{dbyn+1} {0} {node2} sdby
        '''
    cells_as_elem_diode = cells_as_elem_diode + elementary_diode
    
    circuit_paras = f"""
        * One-diode solar module simulation
    * Define the diode model
    .model sd1 D (IS={saturation_current} N={ideality_factor} EG=1.12 BV={breakdown_voltage})
    .model sdby D (IS=1e-7 N=1 BV={v_bypass} IBV=20)
    
    * Define the DC sweep
    V1 1 {0} 
    
    .temp = {TNOM}
    .options tnom={TNOM}
    .control
    dc V1 0 {int(v_oc*N_modules + 1)} 1
    print dc v(1) i()
    set filetype=ascii 
    write {path}/solar_circuit.out
    write {path}/solar_circuit.txt
    
    quit
    .endc
    """
    circuit_content = cells_as_elem_diode + circuit_paras
    
    # Save the circuit file
    with open(f"{path}/{circuit_name}.cir", "w") as circuit_file:
        circuit_file.write(circuit_content)

    circuit_content = f"""
* One-diode solar module simulation

* Define the diode model
.model sd1 D (IS={saturation_current} N={ideality_factor} EG=1.12 BV={breakdown_voltage})
.model sdby D (IS=1e-7 N=1 BV={v_bypass} IBV=20)

* Define the DC sweep
V1 1 0

{cells_as_elem_diode}

.temp = {TNOM}
.options tnom={TNOM}
.control
dc V1 0 {v_oc+1} 0.1
print dc v(1) i(V1)
set filetype=ascii 
write {path}/solar_circuit.out
write {path}/solar_circuit.txt

quit
.endc
"""

def modules_series_netlist_builder(
    path,
    N_s,
    N_modules,
    cell_currents_,
    series_resistances_,
    shunt_resistances_,
    ideality_factor,
    saturation_current,
    breakdown_voltage,
    v_bypass,
    v_oc,
    N_byp=3,
    TNOM=25,
    circuit_name="solar_circuit",
    v_step=0.1,
):
    """
    Build a netlist for a string of standard non-halfcut series-connected modules.
    Each module contains N_s cells in series and N_byp bypass diodes distributed
    evenly across the module substring groups.

    Parameters
    ----------
    path : str
        Path where to save the circuit file and output of spice simulation.
    N_s : int
        Number of cells in each module, e.g. 60.
    N_modules : int
        Number of modules in the string.
    cell_currents_ : float or array
        Photocurrent of each solar cell. Length can be:
          - 1
          - N_s
          - N_s * N_modules
    series_resistances_ : float or array
        Series resistance of each solar cell. Length can be:
          - 1
          - N_s
          - N_s * N_modules
    shunt_resistances_ : float or array
        Shunt resistance of each solar cell. Length can be:
          - 1
          - N_s
          - N_s * N_modules
    ideality_factor : float
        Ideality factor of the diode model.
    saturation_current : float
        Saturation current of the diode model.
    breakdown_voltage : float
        Cell breakdown voltage.
    v_bypass : float
        Bypass diode voltage.
    v_oc : float
        Approximate open-circuit voltage per module for sweep limit.
    N_byp : int
        Number of bypass diodes per module. Default is 3.
    TNOM : float
        SPICE nominal temperature.
    circuit_name : str
        Name of circuit file to be saved.
    v_step : float
        DC sweep voltage step.

    Returns
    -------
    circuit_content : str
        Netlist text.
    """

    import numpy as np
    import os

    total_cells = N_s * N_modules

    def _as_array(x):
        if np.isscalar(x):
            return np.asarray([x], dtype=float)
        return np.asarray(x, dtype=float)

    def _expand_param(arr, name):
        arr = _as_array(arr)
        if len(arr) == 1:
            print(f"{name} made into array of N_s*N_modules = {total_cells}")
            return np.full(total_cells, arr[0], dtype=float)
        elif len(arr) == N_s:
            print(f"{name} tiled from one-module length N_s={N_s} to total length {total_cells}")
            return np.tile(arr, N_modules)
        elif len(arr) == total_cells:
            return arr.astype(float)
        else:
            raise ValueError(
                f"{name} length must be 1, N_s ({N_s}), or N_s*N_modules ({total_cells}), got {len(arr)}"
            )

    cell_currents = _expand_param(cell_currents_, "cell_currents")
    series_resistances = _expand_param(series_resistances_, "series_resistances")
    shunt_resistances = _expand_param(shunt_resistances_, "shunt_resistances")

    print(
        f"Length of all variables: "
        f"Rsh {len(shunt_resistances)}, Rs {len(series_resistances)}, Iph {len(cell_currents)}"
    )

    if N_s % N_byp != 0:
        raise ValueError(f"N_s={N_s} must be divisible by N_byp={N_byp}")

    cells_per_bypass = N_s // N_byp
    cells_as_elem_diode = ""

    # -----------------------------------------------------
    # Build one long series chain across all modules
    # -----------------------------------------------------
    for n in range(total_cells):
        photocurrent = cell_currents[n]
        resistance_series = series_resistances[n]
        resistance_shunt = shunt_resistances[n]

        rsn = n
        rpn = n
        dn = n
        ni = n

        rspos = [n + n + 1, n + n + 2]
        rppos = [n + n + 2, n + n + 3]
        dpos = [n + n + 2, n + n + 3]
        ipos = [n + n + 3, n + n + 2]

        if n == 0:
            elementary_diode = f"""
* elementary diode model cell {n}
Rs{rsn} 1 2 {resistance_series}
Rp{rpn} 2 3 {resistance_shunt}
D{dn} 2 3 sd1
I{ni} 3 2 {photocurrent}
"""
        elif n == total_cells - 1:
            elementary_diode = f"""
* elementary diode model cell {n}
Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} 0 {resistance_shunt}
D{dn} {dpos[0]} 0 sd1
I{ni} 0 {ipos[1]} {photocurrent}
"""
        else:
            elementary_diode = f"""
* elementary diode model cell {n}
Rs{rsn} {rspos[0]} {rspos[1]} {resistance_series}
Rp{rpn} {rppos[0]} {rppos[1]} {resistance_shunt}
D{dn} {dpos[0]} {dpos[1]} sd1
I{ni} {ipos[0]} {ipos[1]} {photocurrent}
"""
        cells_as_elem_diode += elementary_diode

    # -----------------------------------------------------
    # Add bypass diodes for each module
    # Each bypass spans one substring of cells_per_bypass cells
    # -----------------------------------------------------
    dbyn = 0
    for m in range(N_modules):
        module_cell_start = m * N_s

        for b in range(N_byp):
            substring_start = module_cell_start + b * cells_per_bypass
            substring_end = module_cell_start + (b + 1) * cells_per_bypass - 1

            if substring_start == 0:
                start_node = 1
            else:
                start_node = 2 * substring_start + 1

            if substring_end == total_cells - 1:
                end_node = 0
            else:
                end_node = 2 * substring_end + 3

            elementary_diode = f"""
* bypass diode module {m} substring {b}
Dby{dbyn} {end_node} {start_node} sdby
"""
            cells_as_elem_diode += elementary_diode
            dbyn += 1

    # -----------------------------------------------------
    # Netlist / control block
    # -----------------------------------------------------
    total_sweep_voltage = N_modules * v_oc + 1

    circuit_content = f"""
* One-diode solar string simulation (standard non-halfcut modules)

* Define the diode model
.model sd1 D (IS={saturation_current} N={ideality_factor} EG=1.12 BV={breakdown_voltage})
.model sdby D (IS=1e-7 N=1 BV={v_bypass} IBV=20)

* Define the DC sweep source
V1 1 0

{cells_as_elem_diode}

.temp = {TNOM}
.options tnom={TNOM}
.control
dc V1 0 {total_sweep_voltage} {v_step}
print dc v(1) i(V1)
set filetype=ascii
write {path}/solar_circuit.out
write {path}/solar_circuit.txt

quit
.endc
"""

    with open(os.path.join(path, f"{circuit_name}.cir"), "w") as circuit_file:
        circuit_file.write(circuit_content)

    return circuit_content