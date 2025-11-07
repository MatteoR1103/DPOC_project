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

    for s in state_space:
        #unpack state 
        y_k, v_k, d_values, h_values = s
        for u_k in input_space:
            y_k1, v_k1 = compute_pose_dynamics(y_k, v_k, u_k, C)

        
    return P

def compute_pose_dynamics(y_k, v_k, u_k, C: Const):
    """ Computes next state velocities and height
    Args:
        y_k : current height
        v_k : current velocity
    Returns:
        tuple(y_k1 (int), v_k1): next state height and all possible velocities
    """
    v_max = C.V_max
    g = C.g
    v_dev = C.V_dev
    flap_space_dim = 2*v_dev+1

    y_k1 = min(max(y_k + v_k, 0), C.Y -1)
    
    if u_k == C.U_no_flap or u_k == C.U_weak: 
        w_flap_k = np.zeros(flap_space_dim)
    else: 
        #w_k_flap takes different values distributed uniformly around [-V_dev, V_dev]
        
        w_flap_k = np.linspace(-v_dev, v_dev, flap_space_dim) #flap_space_dim

    v_k1 = np.min(np.max(v_k + u_k + w_flap_k - g, -v_max), v_max) #flap_space_dim. Each velocity has the same probability of being generated

    return y_k1, v_k1


def compute_obst_dynamics(d_k, h_k, C: Const):
    """ Computes obstacle dynamics: new obstacle distances and heights based on transition and spawn disturbance
        Args:
            d_k : first obst distance
            h_k : first obst height
            y_k : current height
            C   : constants 
        Returns:
            int: 0 for is_collision, 1 for is_passing or 2 for is_drifting
    """
    #compute intermediate dynamics first, then factor in spawn disturbances
    #only used for valid states, otherwise probability is already computed as 0 

    
        