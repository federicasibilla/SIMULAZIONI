"""
visualization.py: file containing the functions to make plots 

CONTAINS:
    - R_ongrid: takes the matrix nxnxn_R with concentrations and returns the plot of such distributions 
                in 2D, one surface for each nutrient on the grid it is defined on
    - R_ongrid_3D: takes the matrix nxnxn_R with concentrations and returns the plot of such distributions 
                in 3D, one surface for each nutrient on the grid it is defined on
    - N_ongrid: takes the matrix nxnxn_s with ones and zeros representing which species is present
                on each sites and returns the plot, with one color for each species
    - G_ongrid: takes the nxn matrix of growth rates and returns the plot on grid
    - makenet:  function to draw the metabolic network, takes the metabolic matrix as input
    - vispreferences: function to visualize the uptake preferences
    - abundances: function to visualize the abundances time series
    - vis_wm: function to visualize time seriues of well-mixed case

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import networkx as nx
import seaborn as sns
import os

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from networkx.drawing.nx_agraph import to_agraph
from matplotlib.animation import FuncAnimation,ArtistAnimation
from matplotlib.animation import PillowWriter
from scipy.stats import entropy
sns.set(style='whitegrid')

#---------------------------------------------------------------------------------
# define R_ongrid(R) to visualize the equilibrium concentrations for each resource

def R_ongrid(R):

    """
    R: matrix, nxnxn_r, chemicals concentrations

    PLOTS the equilibrium concentrations for each nutrient

    """

    # plotting path and saving name
    executable_name = os.path.basename(__file__).split('.')[0]
    plot_path = os.path.join('/Users/federicasibilla/Documenti/Tesi/SIMULAZIONI/graphs/', executable_name + '/R_fin.png')

    # create the grid
    x = np.arange(R.shape[0])
    y = np.arange(R.shape[0])
    X, Y = np.meshgrid(x, y)

    # R matrix as function of x and y plot (one plot per nutrient)
    n_r = R.shape[2]
    fig, axs = plt.subplots(1, n_r, figsize=(18, 6))

    if n_r == 1:
        im = axs.imshow(R[:, :, 0], cmap='ocean')
        fig.colorbar(im)
        axs.set_xlabel('x')
        axs.set_ylabel('y')
        axs.set_title('Resource {}'.format(1))

    else:

        for i in range(n_r):
            ax = axs[i]
            im = ax.imshow(R[:, :, i], cmap='ocean')
            fig.colorbar(im, ax=ax, label='Concentration')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('Resource {}'.format(i+1))

    plt.savefig(plot_path)
    plt.close()

    return

#--------------------------------------------------------------------------------
# same but 3D

def R_ongrid_3D(R):
    """
    R: matrix, nxnxn_r, chemicals concentrations

    PLOTS the equilibrium concentrations for each nutrient

    """

    # plotting path and saving name
    executable_name = os.path.basename(__file__).split('.')[0]
    plot_path = os.path.join('/Users/federicasibilla/Documenti/Tesi/SIMULAZIONI/graphs/', executable_name + '/R_fin_3D.png')

    # R matrix as function of x and y plot (one plot per nutrient)
    n_r = R.shape[2]
    fig = plt.figure(figsize=( n_r*4,10))

    if n_r!=1:
        for r in range(n_r):
            ax = fig.add_subplot(1, n_r, r+1, projection='3d')
            x = np.linspace(0, 1, R.shape[0])
            y = np.linspace(0, 1, R.shape[1])
            X, Y = np.meshgrid(x, y)
            ax.plot_surface(X, Y, R[:,:,r])
            ax.set_title(f'Resource {r+1}')

    else:
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(0, 1, R.shape[0])
        y = np.linspace(0, 1, R.shape[1])
        X, Y = np.meshgrid(x, y)
        ax.plot_surface(X, Y, R[:,:,0])
        ax.set_title(f'Resource {1}')

    plt.savefig(plot_path)
    plt.close()

    return

#---------------------------------------------------------------------------------
# define N_ongrid(R) to visualize the disposition of species 

def N_ongrid(N):

    """
    N: matrix, nxnxn_s containing nxn elements with length n_s composed by all zeros and
       one corresponding to the species present in the grid point (1 species per grid point)

    PLOTS the grid with current species disposition

    """

    # plotting path and saving name
    executable_name = os.path.basename(__file__).split('.')[0]
    plot_path = os.path.join('/Users/federicasibilla/Documenti/Tesi/SIMULAZIONI/graphs/', executable_name + '/N_fin.png')

    # define colors for species distinction
    cmap = plt.cm.get_cmap('bwr', N.shape[2])  
    norm = mc.Normalize(vmin=0, vmax=N.shape[2]-1)

    # plot gird
    colors = cmap(norm(np.argmax(N, axis=2)))
    plt.figure(figsize=(8, 8))

    plt.imshow(colors, interpolation='nearest')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm,ticks=np.arange(N.shape[2]),label='Species')

    plt.savefig(plot_path)
    plt.close()

    return

#---------------------------------------------------------------------------------
# define G_ongrid(G) function to visualize growth rates

def G_ongrid(G,N):

    """
    G: matrix, nxn, growth rates matrix
    N: matrix, nxnxn_s, hot encoded species state

    RETURNS grid with color gradient corresponding to growth rates

    """

    # plotting path and saving name
    executable_name = os.path.basename(__file__).split('.')[0]
    plot_path = os.path.join('/Users/federicasibilla/Documenti/Tesi/SIMULAZIONI/graphs/', executable_name + '/G_rates.png')

    fig = plt.figure(figsize=(N.shape[2]*10,10))

    for species in range(N.shape[2]):

        ax = fig.add_subplot(1, N.shape[2], species+1)
        cmap = plt.cm.get_cmap('summer') 

        # Apply the mask to the G matrix
        G_masked = np.where(N[:,:,species] == 1, G, np.nan)

        # Plot the masked G matrix
        im = ax.imshow(G_masked, cmap=cmap)

        # Add a colorbar with the label 'Growth rate'
        fig.colorbar(im, ax=ax, label='Growth rate')

        ax.set_title(f'Species {species+1}')

    # Save the figure with all subplots
    plt.savefig(plot_path)

    plt.close()

    return

#---------------------------------------------------------------------------------
# define makenet(met_matrix) to visualize the metabolic processes network, with
# resources as nodes and allocation magnitude as edges thikness

def makenet(met_matrix):

    """
    met_matrix: matrix, n_rxn_r, with resources as rows and columns and allocation rates as
                entries

    RETURNS the graph of metabolic allocations

    """

    # plotting path and saving name
    executable_name = os.path.basename(__file__).split('.')[0]
    plot_path = os.path.join('/Users/federicasibilla/Documenti/Tesi/SIMULAZIONI/matrices/', executable_name + '/met_net.png')

    G = nx.DiGraph()

    for i in range(met_matrix.shape[0]):
        for j in range(met_matrix.shape[1]):
            G.add_edge(f"Res{j+1}", f"Res{i+1}", weight=met_matrix[i, j])

    # draw graph
    agraph = to_agraph(G)
    agraph.layout(prog='dot', args='-GK=0.5 -Gsep=3 -Ncolor=lightblue -Nstyle=filled -Npenwidth=2 -Ecolor=gray -Nnodesep=0.1')
    for edge in agraph.edges():
        weight = G[edge[0]][edge[1]]['weight']
        agraph.get_edge(*edge).attr['penwidth'] = weight * 5
    img = agraph.draw(format='png')
    with open(plot_path, 'wb') as f:
        f.write(img)

    return

#---------------------------------------------------------------------------------
# defining vispreferences(up_mat) function to visualize the uptake preferences 
# of the different species

def vispreferences(mat):

    """
    mat: matrix,  n_sxn_r, uptake of the different species and resources

    RETURNS a graph to visualize uptake preferences 

    """

    # plotting path and saving name
    executable_name = os.path.basename(__file__).split('.')[0]
    plot_path = os.path.join('/Users/federicasibilla/Documenti/Tesi/SIMULAZIONI/matrices/', executable_name + '/up_pref.png')

    up_mat = mat['uptake']*mat['sign']

    plt.figure(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, up_mat.shape[1]))  

    legend = 0
    for i in range(up_mat.shape[0]):
        offset = 0 
        offset_neg = 0
    
        for j in range(up_mat.shape[1]):
            lunghezza_segmento = up_mat[i, j]  
            if (lunghezza_segmento>=0):
                if legend<up_mat.shape[1]:
                    plt.bar(i, lunghezza_segmento, bottom=offset, width=0.8, color=colors[j], label=f'Res {j+1}')
                    offset += lunghezza_segmento
                    legend +=1
                else:
                    plt.bar(i, lunghezza_segmento, bottom=offset, width=0.8, color=colors[j])
                    offset += lunghezza_segmento
            else:
                if legend<up_mat.shape[1]:
                    plt.bar(i, lunghezza_segmento, bottom=offset_neg, width=0.8, color=colors[j], label=f'Res {j+1}')
                    offset_neg += lunghezza_segmento
                    legend +=1
                else:
                    plt.bar(i, lunghezza_segmento, bottom=offset_neg, width=0.8, color=colors[j])
                    offset_neg += lunghezza_segmento



    plt.xlabel('Species')
    plt.ylabel('Uptake')
    plt.title('Consumer preferences')
    plt.legend()
    plt.grid(True) 

    plt.savefig(plot_path)
    plt.close()

    return


#---------------------------------------------------------------------------------------------------------
# function to visualize aboundances and determin if steady state is reached

def abundances(steps):

    """
    steps: list of matrices where each element represents a point in time
    
    RETURNS abundance_series: time series matrix of abundances

    """

    # plotting path and saving name
    executable_name = os.path.basename(__file__).split('.')[0]
    plot_path = os.path.join('/Users/federicasibilla/Documenti/Tesi/SIMULAZIONI/graphs/', executable_name + '/abundances.png')

    # Initialize an empty list to store abundance time series
    abundance_series = []

    # Get the number of unique integers present in the matrices
    num_unique_integers = len(np.unique(steps[0]))

    # Iterate over each time step (matrix)
    for step in steps:
        # Initialize an empty list to store abundances for this time step
        step_abundances = []

        # Count the occurrences of each integer in the matrix
        for i in range(num_unique_integers):
            abundance = np.count_nonzero(step == i)
            step_abundances.append(abundance)

        # Append the abundances for this time step to the abundance series
        abundance_series.append(step_abundances)

    # Convert the list of lists to a numpy array
    abundance_series = np.array(abundance_series)

    # Calculate Shannon diversity for each time step
    shannon_diversity = []
    for step in abundance_series:
        p_i = step / np.sum(step)
        shannon_diversity.append(entropy(p_i))

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot abundance time series
    for i in range(num_unique_integers):
        axs[0].plot(abundance_series[:, i], label=f'Species {i+1}')

    axs[1].axhline(y=np.log(len(np.unique(steps[0]))), color='red', linestyle='--', label='Max Possible Diversity')

    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Abundance')
    axs[0].set_title('Abundance Time Series')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Shannon diversity
    axs[1].plot(shannon_diversity, color='orange')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Shannon Diversity')
    axs[1].set_title('Shannon Diversity Time Series')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(plot_path)

    return abundance_series


#------------------------------------------------------------------------------------------------
# visualize well mixed time series

def vis_wm(N,R):

    """
    N: list of vectors, n_s, species abundances
    R: list of vectors, n_r, chemicals concentrations

    RETURNS time series of well mixed simulation

    """

    # plotting path and saving name
    executable_name = os.path.basename(__file__).split('.')[0]
    plot_path = os.path.join('/Users/federicasibilla/Documenti/Tesi/SIMULAZIONI/graphs/', executable_name + '/wm.png')

    # Plot Time Series for Species
    plt.figure(figsize=(8, 5))
    plt.title("Time Series for Species")
    plt.xlabel("Time")
    plt.ylabel("Population")

    colors = plt.cm.tab10.colors  

    for i in range(N.shape[1]):                                       
        plt.plot(N[:, i], label=f'Species {i}', color=colors[i], linewidth=1)

    plt.legend()
    plt.grid(True)  
    plt.tight_layout()  
    plt.savefig('well_mixed_N.png')
    plt.close()

    # Plot Time Series for Resources
    plt.figure(figsize=(8, 5))
    plt.title("Time Series for Resources")
    plt.xlabel("Time")
    plt.ylabel("Concentration")

    for i in range(R.shape[1]):                                       
      plt.plot(R[:, i], label=f'Resource {i}', color=plt.cm.viridis(i/30), linewidth=1)

    plt.legend()
    plt.grid(True)  
    plt.tight_layout()  
    plt.savefig(plot_path)
    plt.close()

    return