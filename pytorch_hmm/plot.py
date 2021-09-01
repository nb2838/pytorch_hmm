import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm 

def plot_state_diagram(z, x):
    """
    Plots a simple state diagram to analyze results of an hmm. 

    z: list of states of dimension 1. 
    x: array of dimension (num_dims, t) corresponding to observed values
    """
    fig, ax = plt.subplots(2, 1, sharex=True)
    for i in range(len(x)):
        ax[0].plot(x[i])

        
    ax[1].imshow([z], cmap=plt.get_cmap('tab20c'), aspect='auto')
    plt.show()




