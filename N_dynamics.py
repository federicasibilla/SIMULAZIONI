"""
N_dynamics.py: file containing the functions related to the population grid and its update.
               This corresponds to the code that determins the cellular automaton update rules.

CONTAINS: - growth_rates: the function to compute growth rates starting from equilibrium R and
            the state vector of the population grid
          - death_birth_periodic: update rule for one automaton step in PBC case
          - death_birth_periodic: update rule for one automaton step in DBC case
          - encode: function to one hot encode species matrix
          - decode: function to decode species matrix

"""

import numpy as np

from numpy import random

#--------------------------------------------------------------------------------------------
# growth_rates(R,N,param) function to calculate the growth rates of each individual
# based on their intrinsic conversion factor and on the concentrations of resources on the
# underlying grids of equilibrium concentrations

def growth_rates(R,N,param,mat):

    """
    R:     matrix, nxnxn_r, state of resources
    N:     matrix, nxnxn_s, state of population
    param: dictionary, parameters
    mat:   dictionary, matrices

    RETURNS growth matrix: matrix, nxn, growth rates of each individual
            mod:           matrix, nxnx2, first layer is mu, second layer is limiting
    
    """

    species  = np.argmax(N,axis=2)                           # matrix specifiying species at each site
    growth_matrix = np.zeros((N.shape[0],N.shape[1]))        # matrix to store growth rates

    # identify auxotrophies on the grid
    mask = np.zeros((R.shape[0],R.shape[1],R.shape[2]))     
    mask[mat['ess'][species]!=0] = 1                         # mask to match present species with ess. nut.

    # calculate MM at each site and mask for essential resources
    upp      = R/(R+1)
    up_ess  = np.where(mask==0, 1, upp)                      # 1 for non ess, r/(1+r) for ess

    # find limiting nutrient and calculate correspoinding mu modulation
    lim = np.argmin(up_ess, axis=2)
    mu_lim  = np.min(up_ess,axis=2)

    # create modulation mask
    mu  = np.zeros((R.shape[0],R.shape[1],R.shape[2]))
    mu += mu_lim[:, :, np.newaxis]
    mu[np.arange(R.shape[0])[:, None], np.arange(R.shape[1]),lim] = 1  # ensure that limiting nutrient is not modulated

    # modulation matrix
    mod = np.zeros((R.shape[0],R.shape[1],2))
    mod[:,:,0] = mu_lim
    mod[:,:,1] = lim

    # modulate uptake and insert uptake rates (this is also depletion)
    uptake = upp*mu*mat['uptake'][species]

    n_s = N.shape[2]
    
    for i in range(n_s):

        # check actual leackage rate (no leakage if no production)
        realized_met = mat['met']*mat['spec_met'][i][:, np.newaxis]
        l = param['l'].copy()
        l[np.sum(realized_met,axis=0)==0] = 0
       
        species_i_matrix = N[:, :, i]                                 # level i of N matrix is matrix of species i
        growth_matrix += species_i_matrix*param['g'][i]*(np.sum(uptake*mat['sign'][i]*(1-l)-param['m'][i],axis=2))
        
    return growth_matrix, mod


#-------------------------------------------------------------------------------------------------
# define death_birth(state,G) the rule of update of a single step of the automaton in the PBC case

def death_birth_periodic(state,G):

    """
    state: matrix, nxn, containing the species grid (decoded)
    G:     matrix, nxn, containing growth rates at each site

    RETURNS state:    matrix, nxn, new grid (decoded)
            ancora:   string, 'vittoria' if one species has won, 'ancora' if there are more than 1 species present
            max_spec: int, identity of the most present species

    """

    # choose cell to kill
    i = random.randint(0, state.shape[0]-1) 
    j = random.randint(0, state.shape[0]-1)

    # look at the 8 neighbours (numbered from the bottom counterclockwise)and index for pbc
    n1j = n8j = n7j = j-1
    n3j = n4j = n5j = (j+1) % state.shape[0]
    n2j = n6j       = j

    n7i = n6i = n5i = i-1
    n1i = n2i = n3i = (i+1) % state.shape[0]
    n8i = n4i       = i

    # only kill cells close to an interface (keep searching)
    while ((state[i,j]==np.array([state[n1i,n1j],state[n2i,n2j],state[n3i,n3j],state[n4i,n4j],
                   state[n5i,n5j],state[n6i,n6j],state[n7i,n7j],state[n8i,n8j]])).all()):


                   if (state[0,0]==state).all():
                       print('one species has taken up all colony space')
                       return state, 'vittoria', state[0,0]
                       
                   i = random.randint(0, state.shape[0]-1) 
                   j = random.randint(0, state.shape[0]-1)

                   # look at the 8 neighbours (numbered from the bottom counterclockwise)and index for pbc
                   n1j = n8j = n7j = j-1
                   n3j = n4j = n5j = (j+1) % state.shape[0]
                   n2j = n6j       = j
               
                   n7i = n6i = n5i = i-1
                   n1i = n2i = n3i = (i+1) % state.shape[0]
                   n8i = n4i       = i

    i_s = np.array([n1i,n2i,n3i,n4i,n5i,n6i,n7i,n8i])
    j_s = np.array([n1j,n2j,n3j,n4j,n5j,n6j,n7j,n8j])

    # create probability vector from growth rates vector
    gr = np.array([G[n1i,n1j],G[n2i,n2j],G[n3i,n3j],G[n4i,n4j],
                   G[n5i,n5j],G[n6i,n6j],G[n7i,n7j],G[n8i,n8j]
    ])
    # set to zero probability for negative growth rates
    gr[gr<0] = 0
    if sum(gr)!=0:
        prob = gr/(sum(gr))
    else:
        prob = np.ones((len(gr)))/len(gr)

    # extraction of the winner cell
    winner_idx = np.random.choice(np.arange(len(gr)), p=prob)

    # reproduction
    state[i,j]=state[i_s[winner_idx],j_s[winner_idx]]

    # max species present
    max_spec = np.argmax(np.bincount(state.ravel()))

    return state, 'ancora', max_spec


#-----------------------------------------------------------------------------------------------
# define death_birth(state,G) the rule of update of a single step of the automaton in DBC case

def death_birth(state,G):

    """
    state: matrix, nxn, containing the species grid (decoded)
    G:     matrix, nxn, containing growth rates at each site

    RETURNS state:    matrix, nxn, new grid (decoded)
            ancora:   string, 'vittoria' if one species has won, 'ancora' if there are more than 1 species present
            max_spec: int, identity of the most present species

    """

    # create a padded grid of states
    padded_state = np.full((state.shape[0]+2, state.shape[1]+2), np.nan)
    padded_state[1:state.shape[0]+1,1:state.shape[1]+1] = state
    padded_growth = np.full((state.shape[0]+2, state.shape[1]+2), np.nan)
    padded_growth[1:state.shape[0]+1,1:state.shape[1]+1] = G

    # create neighbrohood kernel
    kernel = np.array([[1, 1, 1],
                      [1, np.nan, 1],
                      [1, 1, 1]])

    # choose cell to kill
    i = random.randint(1, state.shape[0]) 
    j = random.randint(1, state.shape[0])

    # look at the neighbours (numbered from the bottom counterclockwise)
    flat_neig  = np.ravel((padded_state[i-1:i+2,j-1:j+2]*kernel))       # species composition of neighbrohood (flat)
    valid_idx  = np.where(~np.isnan(flat_neig))                         # indices of the flat vector to consider as real neig.
    valid_neig = flat_neig[valid_idx]                                   # correspondent values

    # only kill cells close to an interface (keep searching)
    while ((state[i-1,j-1]==valid_neig).all()):


                   if (state[0,0]==state).all():
                       print('one species has taken up all colony space')
                       return state, 'vittoria', state[0,0]
                       
                   i = random.randint(1, state.shape[0]) 
                   j = random.randint(1, state.shape[0])

                   # look at the neighbours (numbered from the bottom counterclockwise)
                   flat_neig  = np.ravel((padded_state[i-1:i+2,j-1:j+2]*kernel))       # species composition of neighbrohood (flat)
                   valid_idx  = np.where(~np.isnan(flat_neig))                         # indices of the flat vector to consider as real neig.
                   valid_neig = flat_neig[valid_idx]                                   # correspondent values


    # create probability vector from growth rates vector
    flat_gr      = np.ravel((padded_growth[i-1:i+2,j-1:j+2]*kernel))
    valid_idx_gr = np.where(~np.isnan(flat_gr))
    valid_gr     = flat_gr[valid_idx_gr]

    # set to zero probability for negative growth rates
    valid_gr[valid_gr<0] = 0

    if sum(valid_gr)!=0:
        prob = valid_gr/(sum(valid_gr))
    else:
        prob = np.ones((len(valid_gr)))/len(valid_gr)

    # extraction of the winner cell
    winner_idx = np.random.choice(np.arange(len(valid_gr)), p=prob)
    # extraction of its identity
    winner_id  = valid_neig[winner_idx]

    # reproduction
    state[i-1,j-1]=winner_id

    # max species present
    max_spec = np.argmax(np.bincount(state.ravel()))

    return state, 'ancora', max_spec


#--------------------------------------------------------------------------------------
# define encoding(N) function to one-hot encode the species matrix

def encode(N, all_species):

    """
    N:           matrix, nxn, species (integer values, each integer corresponds to a species)
    all_species: list of all possible species

    RETURNS one_hot_N: matrix, nxnxn_s, species identity is one-hot-encoded

    """

    n_s = len(all_species)

    one_hot_N = np.zeros((*N.shape, n_s), dtype=int)

    # encoding
    for i, value in enumerate(all_species):
        one_hot_N[:, :, i] = (N == value).astype(int)

    return one_hot_N


#--------------------------------------------------------------------------------------
# define decoding(N) function to one-hot decode the species matrix

def decode(N):

    """
    N: matrix, nxnxn_s, containing one hot encoded species

    RETURNS decoded_N: matrix, nxn, decoded N matrix

    """

    # max index on axis m (one-hot dimension)
    decoded_N = np.argmax(N, axis=-1)

    return decoded_N