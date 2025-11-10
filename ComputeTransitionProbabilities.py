"""ComputeTransitionProbabilities.py

Template to compute the transition probability matrix.

Dynamic Programming and Optimal Control
Fall 2025
Programming Exercise

Contact: Antonio Terpin aterpin@ethz.ch
Authors: Marius Baumann, Antonio Terpin

--
ETH Zurich
Institute for Dynamic Systems and Control
--
"""

import numpy as np
from Const import Const
import itertools
from utils import *

def compute_transition_probabilities(C:Const) -> np.array:
    """Computes the transition probability matrix P.

    Args:
        C (Const): The constants describing the problem instance.

    Returns:
        np.array: Transition probability matrix of shape (K,K,L), where:
            - K is the size of the state space;
            - L is the size of the input space.
            - P[i,j,l] corresponds to the probability of transitioning
              from the state i to the state j when input l is applied.
    """
    P = np.zeros((C.K, C.K, C.L))
    
    #Create the state space
    state_space = C.state_space
    input_space = C.input_space
    print(f"state_space length: {len(state_space)}")

    # TODO fill the transition probability matrix P here
    # Remember the state space ordering convention: First loop around the bird heights, second around velocities, 
    # third, and so on for obstacles distances, last ones for row indices of obstacles' heights 

    #The transition probability first computes the dynamics considering the disturbances, and then assigns probabilities based on the
    #reachable next states. This has to be done for all the possible states in the state space
    
    #first dummy implementation which is not optimized

    ### POSE DYNAMICS ###
    # C.K is the already filtered state space that satisfies these constraints
    # state_space now contains all the information needed to compute the dynamics and the probability function
    
    #height is a deterministic update, so it can be calculated vectorially without the need of for loops
    y_k = np.array([s[0] for s in state_space]) #(K,)
    v_k = np.array([s[1] for s in state_space]) #(K,)

    d_k = np.array([s[2 : 2 + C.M] for s in state_space]) #(K,M) 
    h_k = np.array([s[2 + C.M : ] for s in state_space])  #(K,M)
    
    y_k1 = compute_height_dynamics(y_k = y_k, v_k=v_k, C=C) #(K,)
    
    v_dev_inter = np.arange(-C.V_dev, C.V_dev + 1)
    v_k1 = compute_vel_dynamics(v_k, input_space, v_dev_inter, C) #(K, input_space, 2*V_dev +1)

    d_k1, h_k1 = compute_obst_dynamics(d_k, h_k, y_k, C) #tuple((K,M,2), (K, M, len(S_h), 2))

    return P


    
        