#!/usr/bin/env python 


# Home-made modules
import opes_analysis_module as opes
# import opes_plot_module as opesplot
import tools

# Other modules
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pprint import pprint
from concurrent.futures import process
from curses.ascii import isdigit

# Start timing
start_time = time.time()

# =========================================
# Setup argparse (allowing the use of flags)
parser = argparse.ArgumentParser(description='Analyse data from an OPES run, using the dumped state file (STATE_WFILE). 1D or 2D only')
# File names
parser.add_argument('--states',
                    '-s',
                    dest='states_file',
                    type=str,
                    default='STATES',
                    help='Name/prefix of the file containing the dumped states with the compressed kernels. (default: %(default)s)')
parser.add_argument('--colvar',
                    '-c',
                    dest='colvar_file',
                    type=str,
                    default='COLVAR',
                    help='Name/prefix of the file containing all data on collective variables. (default: %(default)s)')
parser.add_argument('--kernels',
                    '-k',
                    dest='kernels_file',
                    type=str,
                    default='KERNELS',
                    help='Name of the file containing the kernels. (default: %(default)s)')
parser.add_argument('--fes_prefix',
                    '-o',
                    dest='fes_prefix',
                    type=str,
                    default='FES',
                    help='Prefix of the output file storing all FES information. (default: %(default)s)')
# energy 
parser.add_argument('--temp',
                    '-t',
                    dest='temp',
                    type=float,
                    default=300.0,
                    help='The temperature in Kelvin. (default: %(default)s)')
parser.add_argument('--units',
                    '-u',
                    dest='units',
                    choices=['kT', 'kJ/mol', 'kcal/mol'],
                    type=str,
                    default='kJ/mol',
                    help='Output in these units. (default: %(default)s)')
# about the simulation and processing
parser.add_argument('--process_max',
                    '-m',
                    dest='process_max',
                    type=str,
                    default='last',
                    help='Process all frames (all), only last frame (last) or n frames (int). (default: %(default)s)')

fes_group = parser.add_mutually_exclusive_group()
fes_group.add_argument('--read_fes_from',
                    dest='read_fes_from',
                    type=str,
                    help='Read FES from this file. (default: %(default)s)')
fes_group.add_argument('--calc_fes_from',
                    dest='calc_fes_from',
                    choices=['states', 'colvar', 'kernels'],
                    type=str,
                    default='states',
                    help='Calculate FES from %(choices)s. (default: %(default)s)')

# grid related
parser.add_argument('--min',
                    dest='grid_min',
                    type=str,
                    required=False,
                    help='Lower bounds for the grid (comma seperated)')
parser.add_argument('--max',
                    dest='grid_max',
                    type=str,
                    required=False,
                    help='Upper bounds for the grid (comma seperated)')
parser.add_argument('--bin',
                    dest='grid_bin',
                    type=str,
                    default="100",
                    help='Number of bins for the grid. If 1 value is given, it is used for all dimensions.')
# other options
parser.add_argument('--fmt',
                    dest='fmt',
                    type=str,
                    default='% 12.6f',
                    help='Specify the output format')
parser.add_argument('--cvs',
                    dest='cvs',
                    type=str,
                    required=False,
                    help='Names of the cvs (comma seperated)')
parser.add_argument('--sigmas',
                    dest='sigmas',
                    type=str,
                    required=False,
                    help='Upper bounds for the grid (comma seperated)')
parser.add_argument('--calc_conv',
                    dest='calc_conv',
                    choices=['no', 'deltaFE', 'KLdiv', 'dAlonso', 'all'],
                    type=str,
                    default='all',
                    help='Calculate FES from %(choices)s. (default: %(default)s)')
parser.add_argument('--split_fes_at',
                    dest='split_fes_at',
                    type=float,
                    default=0.0,
                    help='The value at which to split the FES (when calculating deltaFE). (default: %(default)s)')
parser.add_argument('--mintozero',
                    dest='mintozero',
                    action='store_true',
                    default=False,
                    help='Shift the minimum to zero')
parser.add_argument('--lower_dims',
                    dest='mintozero',
                    action='store_false',
                    default=True,
                    help='Also calculate FES for each of the lower dimensions. (default: %(default)s)')
# =========================================

# Process parsed arguments
args = parser.parse_args()

# File name variables
states_file = args.states_file
kernels_file = args.kernels_file
colvar_file = args.colvar_file
fes_prefix = args.fes_prefix
process_max = args.process_max

# Other variables
fmt = args.fmt
calc_fes_from = args.calc_fes_from
read_fes_from = args.read_fes_from
calc_conv = args.calc_conv
split_fes_at = args.split_fes_at
mintozero = args.mintozero

# Other constants
kb = 1.38064852e-23 # Boltzman's constant in m^2 kg s^-2 K^-1
NA = 6.02214086e23 # Avogadro's constant in mol^-1
temp = args.temp

# Energy variables
if args.units == 'kJ/mol':
    # 1 kT = 0.008314459 kJ/mol
    unitfactor = kb * NA * temp / 1000
elif args.units == 'kcal/mol':
    # 1 kJ/mol = 0.238846 kcal/mol
    unitfactor = 0.238846 * kb * NA * temp / 1000
elif args.units == 'kT':
    # 1 kT = 1 kT
    unitfactor = 1

# CVs
if args.cvs != None:
    cvs_from_user = args.cvs.split(',') # list of strings
else:
    cvs_from_user = args.cvs
# Sigmas
if args.sigmas != None:
    sigmas_from_user = list(map(float, args.sigmas.split(','))) # list of floats
else:
    sigmas_from_user = args.sigmas
    
# Grid variables
# Min
if args.grid_min != None:
    grid_min = list(map(float, args.grid_min.split(','))) # list of floats
else:
    grid_min = args.grid_min 
# Max
if args.grid_max != None:
    grid_max = list(map(float, args.grid_max.split(','))) # list of floats
else:
    grid_max = args.grid_max 

# Setup the list of n_bins, later used to make a n_dimensional meshgrid 
grid_bin = list(map(int, args.grid_bin.split(','))) # list of ints

# Check process_max value
if process_max.isdigit() and int(process_max) > 0:
    process_max = int(process_max)
elif process_max not in ['all', 'last']:
    sys.exit(f"ERROR: Illegal value found for process_max ({process_max}). Please choose 'all', 'last' or a positive integer\nNow exiting")


print(f'''Settings...
    Files
        : Statesfile                {states_file}
        : Kernelsfile               {kernels_file}
        : Colvarfile                {colvar_file}
    Energy variables
        : Temperature               {temp} K
        : Units                     {args.units}
        : Conversionfactor          {unitfactor}   
    Grid variables
        : Grid minimum              {grid_min}
        : Grid maximum              {grid_max}
        : Grid binsizes             {",".join(map(str, grid_bin))}
    Other variables
        : Formatting options        {fmt}
        : Calculate convergence     {calc_conv}
        : Minimum to zero           {mintozero}
        : Max processing            {process_max}
runtime: {(time.time() - start_time):.4f} seconds

''')

# Here the main scripts starts is defined
def main():   

    # Read or calculate FES
    start_time = time.time()
    if read_fes_from:
        print("Reading free energy...")
        print(f"\t:from {read_fes_from}...", end="", flush=True)

        fes_list = [pd.read_csv(read_fes_from, delim_whitespace=True)]

        print(f"\t(containing: {len(fes_list[0]['time'].unique())} time point(s))")
    elif calc_fes_from:
        print("Calculating free energy...", flush=True)

        if calc_fes_from == 'states':
            # Load states file
            print(f"\t:loading {states_file}...", end="", flush=True)
            states_data, states_info = opes.read_states_file(states_file)

            # Fetch cvs from file
            cvs_from_file = [col for col in states_data if not (col.startswith('sigma_') or col in ['time', 'height'])]

            # If the cvs are not given, take the file cvs. Otherwise check if they coincide.
            if cvs_from_user == None:
                cvs = cvs_from_file
            else:
                # Check if the given cv list is a subset of cvs found in the states file.
                if all(x in cvs_from_file for x in cvs_from_user):
                    print(f"\n\tNOTE: Your cv ({' and '.join(cvs_from_user)}) is a subset of cv(s) found in the states file ({' and '.join(cvs_from_file)}). Analysis of the other cvs is skipped.")
                    cvs = cvs_from_user
                else:
                    sys.exit(f"\n\tERROR: Your cv(s) ({' and '.join(cvs_from_user)}) do not match cv(s) found in the states file ({' and '.join(cvs_from_file)})\nNow exitting!")

            print(f"\tfound {len(states_info)} states and {len(cvs)} cvs ({' and '.join(cvs)})", flush=True)

            # Setup grid settings, make small dataframe containing all grid_info for each cv.
            grid_info = opes.setup_grid_settings(cvs, grid_min, grid_max, grid_bin)

            # Calculate free energy
            fes_list = opes.fes_from_states(states_data,
                                    states_info,
                                    cvs,
                                    grid_info,
                                    process_max,
                                    mintozero,
                                    unitfactor,
                                    fes_prefix,
                                    fmt)
        elif calc_fes_from == 'colvar':
            # Load colvar file
            print(f"\t:loading {colvar_file}...", end="", flush=True)
            colvar = pd.read_csv(colvar_file, names=tools.get_column_names(colvar_file), delim_whitespace=True, comment="#")

            # Fetch cvs from file
            # I'm taking the column name between the columns 'time' and 'opes.bias', which is where I always find my cvs as defined in plumed.dat
            # PRINT STRIDE=500 FILE=COLVAR ARG=cv_1,(...),cv_n,opes.*,other_variables
            # If you do this differently the results can differ.
            cvs_from_file = colvar.loc[:, 'time':'opes.bias'].columns.values[1:-1]

            # If the cvs are not given, take the file cvs. Otherwise check if they coincide.
            if cvs_from_user == None:
                cvs = cvs_from_file
            else:
                # Check if the given cv list is a subset of cvs found in the states file.
                if all(x in cvs_from_file for x in cvs_from_user):
                    print(f"\n\tNOTE: Your cv ({' and '.join(cvs_from_user)}) is a subset of cv(s) found in the states file ({' and '.join(cvs_from_file)}). Analysis of the other cvs is skipped.")
                    cvs = cvs_from_user
                else:
                    sys.exit(f"\n\tERROR: Your cv(s) ({' and '.join(cvs_from_user)}) do not match cv(s) found in the states file ({' and '.join(cvs_from_file)})\nNow exitting!")

            print(f"\tfound {len(colvar)} time points and {len(cvs)} cvs ({' and '.join(cvs)})", flush=True)

            # Setup grid settings, make small dataframe containing all grid_info for each cv.
            grid_info = opes.setup_grid_settings(cvs, grid_min, grid_max, grid_bin)

            # Determine sigmas
            # If the sigmas are not given, try finding them in the states_file.
            # Output is a dictionary with cvs names as keys and the sigmas as values
            if sigmas_from_user == None:
                print(f"\t\tNo sigmas given. Trying to fetching from {states_file}...")
            
                # Read first header lines (starting with) and look for the sigmas
                
                if os.path.isfile(states_file):
                    sigmas = opes.find_sigmas(states_file)

                    if not sigmas:
                        sys.exit(f"\n\tERROR: Could not find sigmas in {states_file}.\n\tNow exitting!")
                    elif len(sigmas) != len(cvs):
                        sys.exit(f"\n\tERROR: Number of sigmas ({len(sigmas)}) do not match number of cvs ({len(cvs)}).\nNow exitting!")
                    else:
                        print(f"\t\tfound sigmas: {', '.join(list(map(str, sigmas)))} for {' and '.join(cvs)} respectively", flush=True)
                        sigmas_info = dict(zip(cvs, sigmas))
                else:
                    sys.exit(f"\n\tERROR: Could not find {states_file}.\n\tNow exitting!")

            else:
                sigmas = sigmas_from_user
           
                if len(sigmas) != len(cvs):
                    sys.exit(f"\n\tERROR: Number of sigmas ({len(sigmas)}) do not match number of cvs ({len(cvs)}).\n\tNow exitting!")
                else:
                    print(f"\t\tfound sigmas: {', '.join(list(map(str, sigmas)))} for {' and '.join(cvs)} respectively", flush=True)
                    sigmas_info = dict(zip(cvs, sigmas))

            # Calculate free energy
            fes_list = opes.fes_from_colvar(colvar,
                                        cvs,
                                        sigmas_info,
                                        grid_info,
                                        process_max,
                                        mintozero,
                                        unitfactor,
                                        temp,
                                        fes_prefix,
                                        fmt)

        # TODO # elif calc_fes_from == 'kernels':
        # kernels = pd.read_csv(kernels_file, names=tools.get_column_names(kernels_file), delim_whitespace=True, comment="#")

    print(f"runtime: {(time.time() - start_time):.4f} seconds\n\n")

    # Calculating convergence
    start_time = time.time()
    if calc_conv:
        print("Calculating convergence...")

        print("\t:calculating KLdiv, dAlonso and deltaFE")
        
        for fes in fes_list:
            # Calculate the convergence parameters
            convergence = opes.calc_conv(fes,
                                unitfactor,
                                split_fes_at,
                                calc_fes_from,
                                fmt)        
    print(f"runtime: {(time.time() - start_time):.4f} seconds\n\n")

    # Plotting
    # start_time = time.time()
    # print("Plotting...")

    # gnuplot().plot_colvar()

    # print(f"runtime: {(time.time() - start_time):.4f} seconds\n\n")

# Run the program
if __name__ == '__main__':
    main()