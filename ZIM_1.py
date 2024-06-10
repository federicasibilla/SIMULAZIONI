"""
S1.py: file for simulation number 1 in 'CR model project'
       the interest here is how richness varies when interaction ranges are constant for all interactions
       but the fraction of positive to negative interaction is changed; simulations are run for 8 species
       and 1 carbon source and 24 chemical byproducts

"""

import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pstats 

from visualize import *
from update import *
from well_mixed import *

n_s = 8
n_r = 25

n = 100

# list to contain ratios to investigate
ratios = [0.5,0.8]

# first of all random matrices are created 

# random uptake matrix with entries in (0,1)
up_mat = np.random.rand(n_s,n_r)

# random metabolic matrix (with constraints)
met_mat = np.random.rand(n_r,n_r)*(np.random.rand(n_r, n_r) > 0.8) # make metabolic matrix sparce
row_sum = np.sum(met_mat, axis=1)
non_zero_indices = row_sum != 0
met_mat[non_zero_indices] /= row_sum[non_zero_indices, None]    # columns sum to 1
met_mat[0,:] = 0                                                # carbon source is not produced
np.fill_diagonal(met_mat, 0)                                    # diagonal elements should be 0

# random sign matrices (list should be as long as ratios list)
sign_mat_list = []
for ratio in ratios:
    elements = np.random.choice([-1, 1], size=n_s*n_r, replace=True, p=[1 - ratio/(ratio+1), ratio/(ratio+1)])
    sign_mat = np.reshape(elements, (n_s, n_r))
    sign_mat[:,0]=1 # carbon source positive for all
    sign_mat_list.append(sign_mat)


# no essential nutrients (only catabolic cross-feeding)
mat_ess = np.zeros((n_s,n_r))

# no auxotrophies (anyone can produce what metabolism allows)
spec_met = np.ones((n_s,n_r))

# totally symmetric g and m
g = np.ones((n_s))*0.5
m = np.zeros((n_s))

# no reinsertion of chemicals
R0 = np.zeros((n,n,n_r))
R0[:,:,0] = 10
R0_wm = np.zeros((n_r))
R0_wm[0] = 10
tau = np.zeros((n_r))+np.inf
tau[0]=1


# define parameters
param = {
    # model parameters
    'R0' : R0.copy(),                                  # initial conc. nxnxn_r [monod constants]
    'w'  : np.ones((n_r))*20,                          # energy conversion     [energy/mass]
    'l'  : np.ones((n_r))*0.6,                         # leakage               [adim]
    'g'  : g,                                          # growth conv. factors  [1/energy]
    'm'  : m,                                          # maintainance requ.    [energy/time]
    'ext': R0_wm,                                      # used for 2D and wm   
    'tau' : tau,                                       # chemicals dilution                             
    'tau_s': 1.,                                       # species dilution
    
    # sor algorithm parameters
    'n'  : n,                                          # grid points in each dim
    'sor': 1.85,                                       # relaxation parameter
    'L'  : 100,                                        # grid true size        [length]
    'D'  : 1e2,                                        # diffusion constant    [area/time] 
    'rapp': 0.01,                                      # ration between Dz and Dxy
    'acc': 1e-3,                                       # maximum accepted stopping criterion 
}

# define matrices dict
mat = {
    'uptake'  : up_mat,
    'met'     : met_mat,
    'ess'     : mat_ess,
    'spec_met': spec_met
}

# initial guesses and conditions
R_space_ig = np.ones((n,n,n_r))*10
N0_space   = encode(np.random.randint(0, 8, size=(n,n)),np.array(np.arange(n_s)))
N0_wm      = np.bincount(decode(N0_space).flatten(), minlength=n_s)/(n*n)

# define function to calculate richness
def richness_wm(N):
    return np.sum(N>0.01)
def richness_sp(N):
    return len(np.unique(np.ravel(decode(N))))

# simulate for different ratios and store results
s1_wm_richness = []
s1_space_richness = []

profiler = cProfile.Profile()
profiler.enable()

for i,ratio in enumerate(ratios):
    mat['sign'] = sign_mat_list[i]
    # well mixed
    N_wm, R_wm = run_wellmixed(N0_wm,param,mat,dR_dt,dN_dt)
    s1_wm_richness.append(richness_wm(N_wm[-1]))
    # spatial
    frames_N, _, _, _, _, current_R, current_N, g_rates, _ = simulate_3D(2, f, R_space_ig, N0_space, param, mat)
    s1_space_richness.append(richness_sp(current_N))

print(s1_wm_richness,s1_space_richness)

# plot results
plt.figure(figsize=(8, 6))

plt.scatter(ratios, s1_wm_richness, color='blue', marker='o', label='well mixed')
plt.scatter(ratios, s1_space_richness, color='red', marker='s', label='spatial')

plt.xlabel('Ratio of negative to positive interactions')
plt.ylabel('Richness')
plt.title('Richness dependence on interaction types')
plt.legend()

plt.grid(True)
plt.savefig('s1.png')

# for my use:
vis_wm(N_wm,R_wm)
R_ongrid(current_R)
G_ongrid(g_rates,encode(frames[-2], np.array(np.arange(n_s))))
N_ongrid(current_N)
R_ongrid_3D(current_R)
abundances(frames)
makenet(met_mat)
vispreferences(mat)

profiler.disable()
stats = pstats.Stats(profiler.dump_stats())

# Ordina le statistiche per nome della funzione
stats.sort_stats('time')

with open("stats_profiling.txt", "w") as f:
    stats.stream = f
    stats.print_stats()
