#!/usr/bin/env python3

import time
import sys
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from pprint import pprint
from pathlib import Path

# Helpfull home-made modules
import tools 

# Other constants
kb = 1.38064852e-23 # Boltzman's constant in m^2 kg s^-2 K^-1
NA = 6.02214086e23 # Avogadro's constant in mol^-1

def read_states_file(states_file="STATES"):
    ''' Read states file and modify data to fit into two dataframes. 
    1) states_data (dataframe containing the states data)
    2) states_info (containing the extra information, such as zed value/biasfactor etc.'''

    df = pd.read_csv(states_file, delim_whitespace=True, low_memory=False)

    # Extracting all states from the dataframe and set the time as the index value
    states_data = df[~df.iloc[:,1].isin(['SET', 'FIELDS'])]
    states_data.columns = states_data.columns[2:].tolist() + 2 * ['na']
    states_data = states_data.dropna(axis=1).astype({"time": int})
    
    # Getting additional information (all variables starting with #! SET)
    states_info = df[df.iloc[:,1] == 'SET'].iloc[:,2:4]
    states_info.columns = ['variable', 'value']
    g = states_info.groupby(['variable']).cumcount()

    # Pivot table. Now the unique variables are the column names
    states_info = (states_info.set_index([g, 'variable'])['value']
        .unstack(fill_value=0)
        .reset_index(drop=True)
        .rename_axis(None, axis=1))

    # Add a column for the time values.
    states_info['time'] = states_data['time'].unique()

    return states_data, states_info

def setup_grid_settings(cvs, grid_min, grid_max, grid_bin):
    ''' Make small dataframe containing all grid_info for each cv.'''

    # For grid_min and grid_max if no value given, make an array of nan, if only one value is given, use it for all dimensions, otherwise use the user input.
    if grid_min == None:
        grid_min = np.empty(len(cvs))
        grid_min[:] = np.nan
    elif len(grid_min) == 1:
        grid_min = np.repeat(grid_min, len(cvs))
    elif len(grid_min) != len(cvs):
        sys.exit(f"ERROR: Number of grid_min values ({len(grid_min)}, i.e. {grid_min} ) not equal to number dimensions ({len(cvs)}).\nComma separated integers expected after --min\nIf 1 value is given, it is used for all dimensions.\nNow exiting")

    if grid_max == None:
        grid_max = np.empty(len(cvs))
        grid_max[:] = np.nan
    elif len(grid_max) == 1:
        grid_max = np.repeat(grid_max, len(cvs))
    elif len(grid_max) != len(cvs):
        sys.exit(f"ERROR: Number of grid_max values ({len(grid_max)}, i.e. {grid_max} ) not equal to number dimensions ({len(cvs)}).\nComma separated integers expected after --max\nIf 1 value is given, it is used for all dimensions.\nNow exiting")

    # For grid_bin, if only one value is given, use it for all dimensions. Otherwise, use the user input.
    if len(grid_bin) == 1:
        n_bins = np.repeat(grid_bin, len(cvs)) # repeat the bin value n times
    elif len(grid_bin) != len(cvs):
        sys.exit(f"ERROR: Number of grid_bin values ({len(grid_bin)}, i.e. {grid_bin} ) not equal to number dimensions ({len(cvs)}).\nComma separated integers expected after --bin\nIf 1 value is given, it is used for all dimensions.\nNow exiting")
    else:
        n_bins = grid_bin

    # The + 1 comes from the original invemichele script. He says its: #same as plumed sum_hills
    # I'm not sure why an extra bin is added.
    n_bins = [x + 1 for x in n_bins]

    # Make small dataframe containing all grid_info for each cv 
    return pd.DataFrame(np.vstack((grid_min, grid_max, n_bins)), columns=cvs, index=['grid_min', 'grid_max', 'n_bins'])

def find_sigmas(f):
    ''' See if you can find the sigma values in the first n lines of a file.'''
    sigmas_list = []
    with open(f) as fp:
        lines = fp.readlines()
        for index, line in enumerate(lines):
            if not line.startswith('#!'):
                break
            elif 'sigma0' in line:
                sigmas_list.append(float(line.split()[-1]))
    
    return sigmas_list

def fes_from_states(states_data, states_info, cvs, grid_info, process_max, mintozero, unitfactor, fes_prefix, fmt):
    ''' Calculate the free energy surface from the dumped state file (STATE_WFILE).'''
    
    # The powerset of a set S is the set of all subsets of S, including the empty set and S itself
    # First set is the empty set, so it's removed with [1:]
    cvs_powerset = list(tools.powerset(cvs))[1:] 

    # Get list of all different times, which corresponds to unique states (time is the index of the df)
    states_time_list = states_data['time'].unique()

    # Check if there are states
    if len(states_time_list) == 0:
        sys.exit(f"ERROR: No states found.\nNow exiting")

    # See what states to analyse
    if process_max == 'last':
        # Process only last frame
        states_time_list = [states_time_list[-1]]
        print(f"\t\tkeeping last state only")
    elif process_max == 'all' or len(states_time_list) <= int(process_max):
        # If you have less frames than you want to keep or want to keep all frames  
        print(f"\t\tkeeping all ({len(states_time_list)}) states")
    elif len(states_time_list) >= int(process_max):
        # Striding the list of times to analyse.
        last = states_time_list[-1]
        total = len(states_time_list)
        stride = int(np.ceil(len(states_time_list) / float(process_max)))
        states_time_list = states_time_list[::stride]
    
        # Note: I've decided to always add the last frame which is the "final" state, this might give a small discontinuity in the timesteps between the last two frames.
        print(f"\t\tkeeping {len(states_time_list)} of {total} states (except last state)")
        if states_time_list[-1] != last:
            states_time_list = np.concatenate((states_time_list, [last]), axis=None)
            print(f"\t\tNOTE: last frame was added to the end. This might give a small discontinuity in the timesteps between the last two frames.\n")
    else:
        sys.exit(f"ERROR: Something went wrong when striding.")

    # Make a list in which you keep all FES dataframes to return to main
    fes_all = []

    # Loop over all combination of the powerset
    for cv_subset in cvs_powerset:
        
        dimensions = len(cv_subset)
        print(f"\t\tstarting with FES of {' and '.join(cv_subset)} ({dimensions} dimension(s))")

        # Get only relevant data from dataframe (time, cvs, sigma_cvs and height)
        subset_data = states_data[["time"] + list(cv_subset) + ["sigma_" + s for s in cv_subset] + ["height"]]

        # Loop over all states in states_time_list
        for index, state_time in enumerate(states_time_list):

            # Extract the data for that particular state (with time state_time).
            state_data = subset_data[subset_data['time'] == state_time]
            state_info = states_info[states_info['time'] == state_time]

            # Get kernels. They are stored in lists where the index of the list is the CV number.
            centers, sigmas = [], []
            for cv in cv_subset:
                centers.append(state_data[cv].astype(float).values)
                sigmas.append(state_data["sigma_" + cv].astype(float).values)
        
            # Convert from list to numpy arrays
            # Centers are s_0
            centers = np.asarray(centers)
            sigmas = np.asarray(sigmas)
            
            # Get height of the kernels and store in a list
            height = state_data['height'].astype(float).values

            # Get variables
            sum_weights = state_info["sum_weights"].astype(float).iloc[0]
            zed = state_info["zed"].astype(float).iloc[0] * sum_weights
            epsilon = state_info["epsilon"].astype(float).iloc[0]
            cutoff = state_info['kernel_cutoff'].astype(float).iloc[0]
            val_at_cutoff = np.exp(-0.5 * cutoff**2)
            
            # Prepare the grid
            grid_min = grid_info.loc['grid_min', list(cv_subset)].tolist()
            grid_max = grid_info.loc['grid_max', list(cv_subset)].tolist()
            n_bins = grid_info.loc['n_bins', list(cv_subset)].astype(int).tolist()

            # If needed, redefine the bounds (min, max) for each state
            if np.isnan(grid_min).any():
                grid_min = [min(cv_cent) for cv_cent in centers]
            if np.isnan(grid_max).any():
                grid_max = [max(cv_cent) for cv_cent in centers]

            # Define a list with the bounds for each cv
            # [[cv1-min cv1-max] [cv2-min cv2-max] ... [cvn-min cvn-max]]]
            bounds = list(zip(grid_min, grid_max))

            # Make n dimensional meshgrid, where the dimension represents the cvs.
            # Then make all possible combinations of the n dimensions (n cvs)
            mgrid = np.mgrid[[slice(row[0], row[1], n*1j) for row, n in zip(bounds, n_bins)]]

            # List used to save all probablities.
            norm_probabilities = []

            all_combos = np.array(mgrid.T.reshape(-1, dimensions))
            total_len = len(all_combos)

            for i, combo in enumerate(all_combos):
                # Loopless way to compute the delta_s value by broadcasting
                # You have two Numpy arrays A (n x 1) and B (m x 1) of different sizes. 
                # To subtract all elements of B from each element of A you can use this.
                # Thus the elements of the result matrix C (n x m) should be computed as c(i,j) = A(i)-B(j). Is there any direct loop-less computation using Numpy?Broadcasting
                # c = A[:, np.newaxis] - B
                print(f"\t\tWorking on state {(index + 1)} of {len(states_time_list)}\t | {(i/total_len):.0%}   ", end='\r')
                
                # delta_s = s - s0 / sigma
                delta_s = (combo[:, np.newaxis] - centers)/sigmas

                # "the Gaussian function is defined as G(s, s0) = h exp −0.5 SUM(s − s0)/sigma)^2 (eq. 2.64)
                gauss_function = height * (np.maximum(np.exp(-0.5 * np.sum(delta_s**2, axis=0)) - val_at_cutoff, 0))

                # "However, P˜(s) should be normalized not with respect to the full CV space but only over the CV space actually explored up to step n, which we call Wn. Thus, we introduce the normalization factor Z (zed). Finally, we can explicitly write the bias at the nth step as eq (2.63) where epsilon can be seen as a regularization term that ensures that the argument of the logarithm is always greater than zero."
                norm_probabilities.append(np.sum(gauss_function)/zed + epsilon)

            # Reshape back into n-dimensional array
            norm_probabilities = np.array(norm_probabilities).reshape(n_bins)

            # Set minimum value to zero
            if not mintozero:
                max_prob = 1
            else:
                max_prob = np.max(norm_probabilities)

            # Calculate FES
            # F(s) = - 1/beta * log (P/zed + epsilon)
            fes = -unitfactor * np.log(norm_probabilities / max_prob)

            fes_data_array = np.column_stack((all_combos, fes.flatten(), [state_time]*len(all_combos)))

            # Add all fes data to a np array. Later used to make a single FES file.             
            if index == 0:
                fes_complete = fes_data_array
            else:
                fes_complete = np.append(fes_complete, fes_data_array, axis=0)

        print(f"\t\tWorking on state {(index + 1)} of {len(states_time_list)}\t | {(i/total_len):.0%}   ", end="")

        fes_data = pd.DataFrame(fes_complete, columns = list(cv_subset) + ['fes', 'time'], dtype = float)
        fes_data.to_csv(f"{fes_prefix}_{'_'.join(cv_subset)}_states", index=False, sep='\t', float_format=fmt)

        print(f"\t--> outputfile: {fes_prefix}_{'_'.join(cv_subset)}_states\n")

        # Add to the list of dataframes to return
        fes_all.append(fes_data)

    return fes_all


def fes_from_colvar(colvar, cvs, sigmas_info, grid_info, process_max, mintozero, unitfactor, T, fes_prefix, fmt):
    ''' Get free energy surface estimation from the collective variables file (COLVAR).'''

    # The powerset of a set S is the set of all subsets of S, including the empty set and S itself
    # First set is the empty set, so it's removed with [1:]
    cvs_powerset = list(tools.powerset(cvs))[1:] 

    # Get list of all different times, which corresponds to unique states (time is the index of the df)
    colvar_time_list = colvar['time'].unique()
   
    # Check if there are points
    if len(colvar_time_list) == 0:
        sys.exit(f"ERROR: No time points found.\nNow exiting")

    # See what colvar to analyse
    if process_max == 'last':
        # Process only last frame
        colvar_time_list = [colvar_time_list[-1]]
        print(f"\t\tkeeping last state time points")
    elif process_max == 'all' or len(colvar_time_list) <= int(process_max):
        # If you have less frames than you want to keep or want to keep all frames  
        print(f"\t\tkeeping all ({len(colvar_time_list)}) time points")
    elif len(colvar_time_list) >= int(process_max):
        # Striding the list of times to analyse.
        last = colvar_time_list[-1]
        total = len(colvar_time_list)
        stride = int(np.ceil(len(colvar_time_list) / float(process_max)))
        colvar_time_list = colvar_time_list[::stride]
    
        # Note: I've decided to always add the last frame which is the "final" state, this might give a small discontinuity in the timesteps between the last two frames.
        print(f"\t\tkeeping {len(colvar_time_list)} of {total} states (except last state)")
        if colvar_time_list[-1] != last:
            colvar_time_list = np.concatenate((colvar_time_list, [last]), axis=None)
            print(f"\t\tNOTE: last frame was added to the end. This might give a small discontinuity in the timesteps between the last two frames.\n")
    else:
        sys.exit(f"ERROR: Something went wrong when striding.")

    # Make a list in which you keep all FES dataframes to return to main
    fes_all = []

    # Loop over all combination of the powerset
    for cv_subset in cvs_powerset:
        
        dimensions = len(cv_subset)
        print(f"\t\tstarting with FES of {' and '.join(cv_subset)} ({dimensions} dimension(s))")

        # Get only relevant data from dataframe (time, cvs, sigma_cvs and height)
        subset_data = colvar[["time"] + list(cv_subset) + ["opes.bias"]]

        # Loop over all states in colvar_time_list
        for index, colvar_time in enumerate(colvar_time_list):

            # Extract the data up until that state (with time colvar_time).
            colvar_data = subset_data[subset_data['time'].between(0, colvar_time)]

            # Get kernels. They are stored in lists where the index of the list is the CV number.
            centers, sigmas = [], []
            for cv in cv_subset:
                centers.append(colvar_data[cv].astype(float).values)

            # Sigmas are given. Make a vector as long as the dataframe.
            sigmas = [np.repeat(sigmas_info[k], colvar_data.shape[0]) for k in cv_subset]

            # Convert from list to numpy arrays
            centers = np.asarray(centers)
            sigmas = np.asarray(sigmas)

            # Get the bias values in unitless factor.
            # Given is in kJ/mol, so unitfactor is not used but the kbt for kJ/mol.
            bias = colvar_data['opes.bias'].values / (kb * NA * T / 1000)

            # Prepare the grid
            grid_min = grid_info.loc['grid_min', list(cv_subset)].tolist()
            grid_max = grid_info.loc['grid_max', list(cv_subset)].tolist()
            n_bins = grid_info.loc['n_bins', list(cv_subset)].astype(int).tolist()

            # If needed, redefine the bounds (min, max) for each state
            if np.isnan(grid_min).any():
                grid_min = [min(cv_cent) for cv_cent in centers]
            if np.isnan(grid_max).any():
                grid_max = [max(cv_cent) for cv_cent in centers]

            # Define a list with the bounds for each cv
            # [[cv1-min cv1-max] [cv2-min cv2-max] ... [cvn-min cvn-max]]]
            bounds = list(zip(grid_min, grid_max))

            # Make n dimensional meshgrid, where the dimension represents the cvs.
            # Then make all possible combinations of the n dimensions (n cvs)
            mgrid = np.mgrid[[slice(row[0], row[1], n*1j) for row, n in zip(bounds, n_bins)]]

            # List used to save all free energies.
            fes = []

            all_combos = np.array(mgrid.T.reshape(-1, dimensions))
            total_len = len(all_combos)

            for i, combo in enumerate(all_combos):
                # Loopless way to compute the delta_s value by broadcasting
                # You have two Numpy arrays A (n x 1) and B (m x 1) of different sizes. 
                # To subtract all elements of B from each element of A you can use this.
                # Thus the elements of the result matrix C (n x m) should be computed as c(i,j) = A(i)-B(j). Is there any direct loop-less computation using Numpy?Broadcasting
                # c = A[:, np.newaxis] - B
                print(f"\t\tWorking on state {(index + 1)} of {len(colvar_time_list)}\t | {(i/total_len):.0%}   ", end='\r')

                delta_s = (combo[:, np.newaxis] - centers)/sigmas
                
                args = bias - 0.5 * np.sum(delta_s**2, axis=0)

                fes.append(-unitfactor * np.logaddexp.reduce(args))

            # Set minimum value to zero
            if mintozero:
                fes += abs(np.min(fes))

            # Add all fes data to a np array. Later used to make a single FES file.             
            fes_data_array = np.column_stack((all_combos, fes, [colvar_time]*len(all_combos)))
            if index == 0:
                fes_complete = fes_data_array
            else:
                fes_complete = np.append(fes_complete, fes_data_array, axis=0)

        print(f"\t\tWorking on state {(index + 1)} of {len(colvar_time_list)}\t | {(i/total_len):.0%}   ", end="")
        fes_data = pd.DataFrame(fes_complete, columns = list(cv_subset) + ['fes', 'time'], dtype = float)
        fes_data.to_csv(f"{fes_prefix}_{'_'.join(cv_subset)}_colvar", index=False, sep='\t', float_format=fmt)

        print(f"\t--> outputfile: {fes_prefix}_{'_'.join(cv_subset)}_colvar\n")

        # Add to the list of dataframes to return
        fes_all.append(fes_data)

    return fes_all

def calc_conv(fes_df, unitfactor, split_fes_at, fmt, calc_fes_from, write_output=True):
    ''' Calculate different convergence metrics given a free energy landscape.'''

    time_list = fes_df['time'].unique()

    # Reference distribution (p or v1)
    reference = fes_df[fes_df['time'] == fes_df['time'].unique()[-1]]
    
    # cvs
    cvs = [ x for x in reference.columns.values.tolist() if x not in ['time', 'fes']]
    print(f"\t\tcvs: {' and '.join(cvs)}")   

    kldiv_values, jsdiv_values, dalonso_values, dfe_values = [], [], [], []

    for index, time in enumerate(time_list):
        print(f"\t\tWorking on state {(index + 1)} of {len(time_list)}\t| {((index + 1)*100.0)/len(time_list):.1f}%", end='\r')

        # Current distribution (q or v2)
        current = fes_df[fes_df['time'] == time]

        # Free energy estimates
        ref_fe = reference['fes'].values
        cur_fe = current['fes'].values

        # Corresponding probability distributions
        ref = np.exp(-ref_fe / unitfactor)
        cur = np.exp(-cur_fe / unitfactor)

        # Normalized probability distributions
        ref_norm = ref / np.sum(ref)
        cur_norm = cur / np.sum(cur)

        # To adjust for large arear where q = 0, a Bayesian smoothing function is employed. Here a "simulation" is performed of N ideal steps, using the FE from sampling.
        # The new adjusted probability for each of the bins is then (1 + Pi * N) / (M + N), where M is the total number of bins.
        # N is chosen to be big enough to turn 0 values into very small values, without risking python not being able to handle the values.
        # This effect is similar as adding very small values to all gridpoints.
        N, M = 1e9, len(ref_norm) 
        ref_norm_smooth = (N * ref_norm + 1) / (N + M)
        cur_norm_smooth = (N * cur_norm + 1) / (N + M)
            
        # Kullback-Leibler divergence
        kldiv_values.append(tools.kldiv(ref_norm_smooth, cur_norm_smooth))

        # Jensen–Shannon divergence
        jsdiv_values.append(tools.jsdiv(ref_norm_smooth, cur_norm_smooth))

        # Alonso & Echenique metric
        dalonso_values.append(tools.dalonso(ref_norm_smooth, cur_norm_smooth))

        # DeltaFE
        # NB: summing is as accurate as trapz, and logaddexp avoids overflows
        cv = cvs[0]
        fesA = -unitfactor * np.logaddexp.reduce(-1/unitfactor * current[current[cv] < split_fes_at]['fes'].values)
        fesB = -unitfactor * np.logaddexp.reduce(-1/unitfactor * current[current[cv] > split_fes_at]['fes'].values)
        deltaFE = fesB - fesA

        dfe_values.append(deltaFE)
        
    print(f"\t\tWorking on state {(index + 1)} of {len(time_list)}\t| {((index + 1)*100.0)/len(time_list):.1f}%")

    # Dataframe from lists
    conv_df = pd.DataFrame({'time': time_list, 'KLdiv': kldiv_values, 'JSdiv': jsdiv_values, 'dA': dalonso_values, 'deltaFE': dfe_values})
    conv_df.to_csv(f"conv_{'_'.join(cvs)}_{calc_fes_from}.dat", index=False, sep='\t', float_format=fmt)

    print(f"\t\t--> Outputfile: conv_{'_'.join(cvs)}_{calc_fes_from}.dat")           

    return conv_df