"""
update.py: file containing the functions to run the simulations with given conditions

CONTAINS: - simulate_3D: function to run a simulation using the SOR_3D algorithm for PBC
          - simulate_2D: function to run a simulation using the SOR_3D algorithm for PBC
          - shannon:     function to calculate shannon diversity in real time to check for convergence
          - change_grid_R/N: functions for the multigrid solver, to change grids 

"""

import numpy as np

from time import time
from scipy.interpolate import RegularGridInterpolator

from SOR import *
from N_dynamics import *
from R_dynamics import *


#---------------------------------------------------------------------------------------------
# simulate_3D: functiln to run a simulation with PBC, in a quasi-3D setting

def simulate_3D(steps, source, initial_guess, initial_N, param, mat):

    """
    steps:         int, number of steps we want to run the simulation for
    source:        function, (*args: R,N,param,mat; out: nxn source matrix), reaction in RD equation
    initial_guess: matrix, nxnxn_r, initial guess for eq. concentrations
    initial_N:     matrix, nxnxn_s, initial composition of species grid
    param:         dictionary, parameters
    mat:           dictionary, matrices

    RETURNS frames_N:  list, all the frames of the population grid, in a decoded species matrix form
            frames_R:  list, all the frames of the chemicals grid
            frames_up: list, all the frames of the chemicals grid for uptake
            frames_in: list, all the frames of the chemicals grid for the production
            frames_mu: list, all the frames of the liminting modulation of the growth matrix 
            current_R: matrix, nxnxn_r, final configuration of nutrients grid
            current_N: matrix, nxnxn_s, final configuration of population grid
            g_rates:   matrix, nxn, final growth rates
            s_list:    list, time series of shannon diversity

    """

    # start timing simulation
    t0 = time()

    # extract list of all possible species
    all_species = list(range(len(param['g'])))

    # lists to store time steps 
    frames_N  = [decode(initial_N)] 
    frames_R  = [] 
    frames_up = []
    frames_in = []
    frames_mu = []
    s_list = []

    # first iteration
    print('Solving iteration zero, finding equilibrium from initial guess')

    # computing equilibrium concentration at ztep zero
    current_R, up, prod = SOR_3D(initial_N, param, mat, source, initial_guess)
    # computing growth rates on all the grid
    g_rates, mod  = growth_rates(current_R,initial_N,param,mat)
    # performing DB dynamics
    decoded_N, check, most_present = death_birth_periodic(decode(initial_N),g_rates)
    current_N = encode(decoded_N, all_species)

    # store time step
    frames_N.append(decoded_N)
    frames_R.append(current_R)
    frames_up.append(up)
    frames_in.append(prod)
    frames_mu.append(mod)


    for i in range(steps):

        print("Step %i" % (i+1))

        # compute new equilibrium, initial guess is previous equilibrium
        current_R, up, prod = SOR_3D(current_N, param, mat, source, current_R)

        # compute growth rates
        g_rates, mod  = growth_rates(current_R,current_N,param,mat)
        # performe DB dynamics
        decoded_N,check,most_present = death_birth_periodic(decode(current_N),g_rates)
        # check that there is more than one species
        if check == 'vittoria':
            print('winner species is: ', most_present)
            break

        current_N = encode(decoded_N, all_species)

        # save time step
        frames_N.append(decoded_N)
        frames_R.append(current_R)
        frames_up.append(up)
        frames_in.append(prod)
        frames_mu.append(mod)

        # check if shannon diversity has converged
        s = shannon(current_N)
        s_list.append(s)

        if len(s_list)>1000:
            avg = sum(s_list[-200:])/200
            dev = np.sqrt(sum((s_list[-200:]-avg)**2)/200)

            if(np.abs(s-avg)<dev/10000):
                print('shannon diversity has converged')
                break

    # end timing
    t1 = time()
    print(f'\n Time taken to solve for {steps} steps: ', round((t1-t0)/60,4), ' minutes \n')

    return frames_N, frames_R, frames_up, frames_in, frames_mu, current_R, current_N, g_rates, s_list   


#----------------------------------------------------------------------------------------------------
# simulate_2D: functiln to run a simulation with DBC, in a 2D setting

def simulate_2D(steps, source, initial_guess, initial_N, param, mat):

    """
    steps:         int, number of steps we want to run the simulation for
    source:        function, (*args: R,N,param,mat; out: nxn source matrix), reaction in RD equation
    initial_guess: matrix, nxnxn_r, initial guess for eq. concentrations
    initial_N:     matrix, nxnxn_s, initial composition of species grid
    param:         dictionary, parameters
    mat:           dictionary, matrices

    RETURNS frames_N:  list, all the frames of the population grid, in a decoded species matrix form
            frames_R:  list, all the frames of the chemicals grid
            frames_up: list, all the frames of the chemicals grid for uptake
            frames_in: list, all the frames of the chemicals grid for the production
            frames_mu: list, all the frames of the liminting modulation of the growth matrix 
            current_R: matrix, nxnxn_r, final configuration of nutrients grid
            current_N: matrix, nxnxn_s, final configuration of population grid
            g_rates:   matrix, nxn, final growth rates
            s_list:    list, time series of shannon diversity

    """

    # start timing simulation
    t0 = time()

    # extract list of all possible species
    all_species = list(range(len(param['g'])))

    # lists to store time steps 
    frames_N  = [decode(initial_N)] 
    frames_R  = [] 
    frames_up = []
    frames_in = []
    frames_mu = []
    s_list = []

    # first iteration
    print('Solving iteration zero, finding equilibrium from initial guess')

    # computing equilibrium concentration at ztep zero
    current_R, up, prod = SOR_2D(initial_N, param, mat, source, initial_guess)
    # computing growth rates on all the grid
    g_rates, mod  = growth_rates(current_R,initial_N,param,mat)
    # performing DB dynamics
    decoded_N, check, most_present = death_birth_periodic(decode(initial_N),g_rates)
    current_N = encode(decoded_N, all_species)

    # store time step
    frames_N.append(decoded_N)
    frames_R.append(current_R)
    frames_up.append(up)
    frames_in.append(prod)
    frames_mu.append(mod)

    for i in range(steps):

        print("Step %i" % (i+1))

        # compute new equilibrium, initial guess is previous equilibrium
        current_R, up, prod = SOR_2D(current_N, param, mat, source, current_R)

        # compute growth rates
        g_rates, mod  = growth_rates(current_R,current_N,param,mat)
        # performe DB dynamics
        decoded_N,check,most_present = death_birth_periodic(decode(current_N),g_rates)
        # check that there is more than one species
        if check == 'vittoria':
            print('winner species is: ', most_present)
            break

        current_N = encode(decoded_N, all_species)

        # save time step
        frames_N.append(decoded_N)
        frames_R.append(current_R)
        frames_up.append(up)
        frames_in.append(prod)
        frames_mu.append(mod)

        # check if shannon diversity has converged
        s = shannon(current_N)
        s_list.append(s)

        if len(s_list)>1000:
            avg = sum(s_list[-200:])/200
            dev = np.sqrt(sum((s_list[-200:]-avg)**2)/200)

            if(np.abs(s-avg)<dev/10000):
                print('shannon diversity has converged')
                break

    # end timing
    t1 = time()
    print(f'\n Time taken to solve for {steps} steps: ', round((t1-t0)/60,4), ' minutes \n')

    return frames_N, frames_R, frames_up, frames_in, frames_mu, current_R, current_N, g_rates, s_list


#-------------------------------------------------------------------------------------------------
# function to perform simulations in multigrid iterations

def simulate_MG(steps, source, initial_guess, initial_N, param, mat, t):

    """
    steps:         int, number of steps we want to run the simulation for
    source:        function, (*args: R,N,param,mat; out: nxn source matrix), reaction in RD equation
    initial_guess: matrix, nxnxn_r, initial guess for eq. concentrations
    initial_N:     matrix, nxnxn_s, initial composition of species grid
    param:         dictionary, parameters
    mat:           dictionary, matrices
    t:             int, number of refinements

    RETURNS frames_N:  list, all the frames of the population grid, in a decoded species matrix form
            frames_R:  list, all the frames of the chemicals grid
            frames_up: list, all the frames of the chemicals grid for uptake
            frames_in: list, all the frames of the chemicals grid for the production
            frames_mu: list, all the frames of the liminting modulation of the growth matrix 
            current_R: matrix, nxnxn_r, final configuration of nutrients grid
            current_N: matrix, nxnxn_s, final configuration of population grid
            g_rates:   matrix, nxn, final growth rates
            s_list:    list, time series of shannon diversity

    """

    # start timing simulation
    t0 = time()

    # extract list of all possible species
    all_species = list(range(len(param['g'])))

    # list to store steps of the population grid
    # lists to store time steps 
    frames_N  = [decode(initial_N)] 
    frames_R  = [] 
    frames_up = []
    frames_in = []
    frames_mu = []
    s_list = []

    # initial grid size
    n = initial_N.shape[0]

    # first iteration
    print('Solving iteration zero, finding equilibrium from initial guess')

    # computing equilibrium concentration at ztep zero
    current_R, _, _ = SOR_3D(initial_N, param, mat, source, initial_guess)

    # refine result
    finer_R, finer_up, finer_prod = SOR_3D(change_grid_N(initial_N,n*2), param, mat, source, change_grid_R(current_R,n*2))
    for ref in range(3,t+1):
        m = n*ref
        finer_R, finer_up, finer_prod = SOR_3D(change_grid_N(initial_N,m), param, mat, source, change_grid_R(finer_R,m))
    coarser_R = change_grid_R(finer_R,n)

    # computing growth rates on all the grid
    g_rates, mod   = growth_rates(coarser_R,initial_N,param,mat)
    # performing DB dynamics
    decoded_N,check,most_present = death_birth_periodic(decode(initial_N),g_rates)
    current_N = encode(decoded_N, all_species)

    frames_N.append(decoded_N)
    frames_R.append(coarser_R)
    frames_up.append(change_grid_R(finer_up,n))
    frames_in.append(change_grid_R(finer_prod,n))
    frames_mu.append(mod)

    for i in range(steps):

        print("Step %i" % (i+1))

        # compute new equilibrium, initial guess is previous equilibrium
        finer_R, finer_up, finer_prod = SOR_3D(change_grid_N(current_N,t*n), param, mat, source, finer_R)
        coarser_R = change_grid_R(finer_R,n)

        # compute growth rates
        g_rates, mod  = growth_rates(coarser_R,current_N,param,mat)
        # performe DB dynamics
        decoded_N,check,most_present = death_birth_periodic(decode(current_N),g_rates)
        # check that there is more than one species
        if check == 'vittoria':
            print('winner species is: ', most_present)
            break
        current_N = encode(decoded_N, all_species)

        frames_N.append(decoded_N)
        frames_R.append(coarser_R)
        frames_up.append(change_grid_R(finer_up,n))
        frames_in.append(change_grid_R(finer_prod,n))
        frames_mu.append(mod)
        

        # check if shannon diversity has converged
        s = shannon(current_N)
        s_list.append(s)

        if len(s_list)>1000:
            avg = sum(s_list[-200:])/200
            dev = np.sqrt(sum((s_list[-200:]-avg)**2)/200)
            print(dev,avg,s)
            if(np.abs(s-avg)<dev/10000):
                print('shannon diversity has converged')
                break

    # end timing
    t1 = time()
    print(f'\n Time taken to solve for {steps} steps: ', round((t1-t0)/60,4), ' minutes \n')

    return frames_N, frames_R, frames_up, frames_in, frames_mu, current_R, current_N, g_rates, s_list


#-------------------------------------------------------------------------------------------------
# function to calculate Shannon diversity

def shannon(N):

    """
    N: matrix, nxnxn_s, encoded population grid

    RETURNS shannon_entropy, float, value for shannon diversity

    """

    n_s = N.shape[-1]
    n   = N.shape[0]

    # frequencies
    counts = np.zeros((n_s))
    for i in range(n_s):
        counts[i] = np.sum(N[:,:,i])
    freq = counts/(n*n)

    # calculate shannon diversity
    shannon_entropy = -np.sum(freq * np.log(freq + 1e-10))  # avoid log(0)

    return shannon_entropy

#----------------------------------------------------------------------------------------------------
# functions to change grid for multigrid solvers

def change_grid_R(R, m):

    """
    R: matrix, nxnxn_r, resource matrix
    m: int, new number of grid points

    RETURNS resized_R: matrix, mxmxn_r, resized resources matrix

    """

    n, _, n_r = R.shape

    old_x, old_y = np.arange(n), np.arange(n)
    new_x, new_y = np.linspace(0, n-1, m), np.linspace(0, n-1, m)
    
    resized_R = np.zeros((m, m, n_r))

    if m > n:
        # Interpolation case: New grid is finer
        for i in range(n_r):
            interpolator = RegularGridInterpolator((old_x, old_y), R[:, :, i])
            new_grid_points = np.array(np.meshgrid(new_x, new_y, indexing='ij')).T.reshape(-1, 2)
            resized_R[:, :, i] = interpolator(new_grid_points).reshape(m, m)
    else:
        # Averaging case: New grid is coarser
        scale_factor = n / m
        for i in range(m):
            for j in range(m):
                x_start = int(i * scale_factor)
                x_end = int((i + 1) * scale_factor)
                y_start = int(j * scale_factor)
                y_end = int((j + 1) * scale_factor)
                
                for k in range(n_r):
                    resized_R[i, j, k] = np.mean(R[x_start:x_end, y_start:y_end, k])

    return resized_R

def change_grid_N(N, m):

    """
    N: matrix, nxnxn_s, old species grid
    m: int, new size for grid

    RETURNS new_N: matrix, mxmxn_s, reshaped species grid
    
    """

    n, _, n_s = N.shape
    new_N = np.zeros((m, m, n_s))

    if m == n:
        return N  # No changes needed

    if m > n:
        # If the new grid is larger, copy the values from the old grid
        for i in range(m):
            for j in range(m):
                new_N[i, j] = N[i * n // m, j * n // m]
    else:
        # If the new grid is smaller, use the central point value from the finer grid
        scale_factor = n / m
        half_scale = scale_factor / 2
        for i in range(m):
            for j in range(m):
                center_i = int(i * scale_factor + half_scale)
                center_j = int(j * scale_factor + half_scale)
                new_N[i, j] = N[center_i, center_j]
    
    return new_N
