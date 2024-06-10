"""
well_mixed.py: file to store definition of the correspondent well-mixed model

CONTAINS: - dR_dt: function storing the dynamics of resources
          - dN_dt: function storing the dynamics of species
          - run_wellmixed: function to run the well-mixed simulation

"""

import numpy as np
import scipy



#-------------------------------------------------------------------------------------------
# define chemicals dynamics

def dR_dt(R,N,param,mat):

    """
    R: vector, n_r, current resource concentration
    N: vecotr, n_s, current species abundance
    param, mat: dictionaries, parameters and matrice

    RETURNS (ext+prod-out)**2: vector, n_r, the time derivative of nutrients concentration (reaction part of RD equation)
                                
    """

    n_s = N.shape[0]
    n_r = R.shape[0]

    up = mat['uptake']

    # resource loss due to uptake (not modulated by essentials)
    out = np.dot((R*up/(1+R)).T,N.T)

    # species specific metabolism and renormalization
    D_species = np.tile(mat['met'].T,(n_s,1,1))*(np.tile(mat['spec_met'],(1,1,n_r)).reshape(n_s,n_r,n_r)) 
    sums = np.sum(D_species, axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        D_s_norma = np.where(sums != 0, D_species / sums, D_species)
    up_eff = (param['w']*param['l']*(up*R/(1+R)))*N[:,np.newaxis]
    prod = np.sum(np.einsum('ij,ijk->ik', up_eff, D_s_norma),axis=0)/param['w'] # vector long n_r with produced chemicals
    
    # resource replenishment
    ext = 1/param['tau']*(param['ext']-R)

    return (ext+prod-out)**2

#--------------------------------------------------------------------------------------------------
# define species dynamics

def dN_dt(t,N,R_eq,param,mat):

    """
    R: vector, n_r, current resource concentration
    N: vecotr, n_s, current species abundance
    param, mat: dictionaries, parameters and matrice

    RETURNS N*(growth_vector-1/param['tau_s']), vector, n_s, the new state of species, n_s

    """

    up = mat['uptake'] 

    # check essential nutrients presence (at each site)
    up_eff = np.ones((up.shape))
    for i in range(len(N)):
        # calculate essential nutrients modulation for each species
        if (np.sum(mat['ess'][i]!=0)):
            mu  = np.min(R_eq[mat['ess'][i]==1]/(R_eq[mat['ess'][i]==1]+1))
            lim = np.where(mat['ess'][i] == 1)[np.argmin(R_eq[mat['ess'][i]==1]/(R_eq[mat['ess'][i]==1]+1))]
            up_eff[i]=mat['uptake'][i]*mu      # modulate uptakes 
            up_eff[i,lim]=mat['uptake'][i,lim] # restore uptake of the limiting one to max

    growth_vector = param['g']*(np.sum(param['w']*up_eff*mat['sign']*R_eq/(1+R_eq),axis=1)-param['m'])

    return N*(growth_vector-1/param['tau_s'])


#-----------------------------------------------------------------------------------------------
# function for the whole simulation

def run_wellmixed(N0,param,mat,dR,dN):

    """
    N0: initial state vector of species n_s
    param,mat: matrices and parameters dictionaries

    RETURNS N: list, contains vectors n_s of species time series
            R: list, contains vectors n_r of chemnicals time series

    """

    N = [N0.copy()]
    R = []

    guess = np.ones((param['ext'].shape))*10
    diff = N0
    N_prev = N0
    i = 0
 
    while((np.abs(diff) > 0.001*N_prev).any()):

        print("N_iter_wm %d \r" % (i), end='')

        R_eq = scipy.optimize.least_squares(dR, guess, args=(N_prev,param,mat),bounds=(0,np.inf)).x
        
        # integrate N one step
        N_out = scipy.integrate.solve_ivp(dN, (0,1), N_prev, method='RK45', args=(np.array(R_eq),param,mat))
        N_out = N_out.y[:, -1]

        diff = N_out-N_prev
        N_prev += diff
        guess = R_eq

        N.append(N_out)
        R.append(R_eq)

        i +=1

        if i>1000:
            break

    N, R = np.array(N),np.array(R)
    N = N/np.sum(N,axis=1)[:,np.newaxis]

    return N,R