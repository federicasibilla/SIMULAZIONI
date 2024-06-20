"""
auxotrophs simulation 

"""

import numpy as np
import pandas as pd

import cProfile
import pstats
import os

from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

from R_dynamics import *
from N_dynamics import *
from visualize  import *
from update import *
from SOR import *

# create paths to save results
path = os.path.splitext(os.path.abspath(__file__))[0]
executable_name = os.path.basename(__file__).split('.')[0]
results_dir = f"{path}_results"
graphs_dir = os.path.join(results_dir, "graphs")
matric_dir = os.path.join(results_dir, "matrices")
output_dir = os.path.join(results_dir, "outputs")
os.makedirs(graphs_dir, exist_ok=True)
os.makedirs(matric_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


# initialize R0
n_r = 3
n_s = 2
n   = 40

R0  = np.zeros((n, n, n_r))
# saturate resource c everywhere
R0[:,:,0]=10
R0[:,:,1]=0
R0[:,:,2]=0
g   = np.array([0.4,0.6]) 
m   = np.array([0.,0.])

initial_guess = np.random.uniform(90, 100, size=(n,n,n_r))

# initialize species grid: random
N = np.random.randint(2, size=(n, n,n_s))
N[N[:,:,0]==1,1]=0
N[N[:,:,0]==0,1]=1
#N = np.zeros((n,n,n_s))
#N[:,20:,0]=1
#N[:,:20,1]=1


# define parameters
param = {
    # model parameters
    'R0' : R0.copy(),                                  # initial conc. nxnxn_r [monod constants]
    'w'  : np.array([20,20,20]),                       # energy conversion     [energy/mass]
    'l'  : np.array([0.5,0.1,0.1]),                    # leakage               [adim]
    'g'  : g,                                          # growth conv. factors  [1/energy]
    'm'  : m,                                          # maintainance requ.    [energy/time]
    'ext': np.array([10.,0.,0.]),
    'rapp': 0.01,
    
    # sor algorithm parameters
    'n'  : n,                                          # grid points in each dim
    'sor': 1.85,                                       # relaxation parameter
    'L'  : 40,                                         # grid true size        [length]
    'D'  : 1e1,                                        # diffusion constant    [area/time] 
    'acc': 1e-3,                                       # maximum accepted stopping criterion   
    'ref': 0                                           # number of grid refinements to perform 
}

# make matrices
up_mat   = np.array([[1,0.,1],[1.,1.,0.]])
met_mat  = np.array([[0.,0.,0.],[1.,0.,1],[1,1,0.]])
sign_mat = np.array([[1.,0.,0.],[1.,0.,0.]])
mat_ess  = np.array([[0.,0.,1.],[0.,1.,0.]])
spec_met = np.array([[0.,1,0.],[0.,0.,1]])

mat = {
    'uptake'  : up_mat,
    'met'     : met_mat,
    'sign'    : sign_mat,
    'ess'     : mat_ess,
    'spec_met': spec_met
}

# save in output file all parameters
with open(f"{output_dir}/parameters.txt", 'w') as file:
    file.write("Simulation Parameters:\n")
    for key, value in param.items():
        file.write(f"{key}: {value}\n")
    file.write("\nMatrices:\n")
    for key, value in mat.items():
        file.write(f"{key}:\n{value}\n\n")

last_2_frames_N, mod, current_R, current_N, g_rates, s_list, abundances = simulate_MG(10, f, initial_guess, N, param, mat,1)

# save results as csv
np.save(f'{output_dir}/R_fin.npz', current_R)
np.save(f'{output_dir}/N_fin.npz', current_N)
np.save(f'{output_dir}/2N.npz', last_2_frames_N)
np.save(f'{output_dir}/mod_fin.npz', mod)
np.save(f'{output_dir}/g_fin.npz', g_rates)
np.save(f'{output_dir}/shannon.npz', s_list)
np.save(f'{output_dir}/abundances.npz', abundances)

# save plots
R_ongrid(current_R,graphs_dir)
G_ongrid(g_rates,encode(last_2_frames_N[-2], np.array(np.arange(n_s))),graphs_dir)
N_ongrid(current_N,graphs_dir)
R_ongrid_3D(current_R,graphs_dir)
vis_abundances(abundances,s_list,graphs_dir)
makenet(met_mat,matric_dir)
vispreferences(mat,matric_dir)

