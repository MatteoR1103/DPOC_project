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

    _ = C.state_to_index(C.state_space[0])
    index_map = C._state_indexing
    
    #print(f"state_space length: {len(state_space)}")

    w_h_dim = len(C.S_h)

    # TODO fill the transition probability matrix P here

    #The transition probability first computes the dynamics considering the disturbances, and then assigns probabilities based on the
    #reachable next states. This has to be done for all the possible states in the state space

    ### POSE DYNAMICS ###
    # C.K is the already filtered state space that satisfies these constraints
    # state_space now contains all the information needed to compute the dynamics and the probability function
    
    #height is a deterministic update, so it can be calculated vectorially without the need of for loops
    #Current heights, velocities and obstacles' poses
    y_k = np.array([s[0] for s in state_space]) #(K,)
    v_k = np.array([s[1] for s in state_space]) #(K,)

    d_k = np.array([s[2 : 2 + C.M] for s in state_space]) #(K,M) 
    h_k = np.array([s[2 + C.M : ] for s in state_space])  #(K,M)
    
    #Next height dynamics
    y_k1 = compute_height_dynamics(y_k = y_k, v_k=v_k, C=C) #(K,)
    
    #Next vel dynamics
    v_dev_inter = np.arange(-C.V_dev, C.V_dev + 1)
    flap_space_dim = len(v_dev_inter)
    v_k1 = compute_vel_dynamics(v_k, input_space, v_dev_inter, C) #(K, input_space, 2*V_dev +1)

    #Next obstacles' dynamics
    d_k1, h_k1, p_spawn = compute_obst_dynamics(d_k, h_k, y_k, C) #tuple((K,M,2), (K, M, len(S_h), 2), (K,))

    # now we need to factor in all probabilities at once. The probability of going from one state to the other when applying control action u
    # depends on the following factors: 
    # - prob(y_k, y_k1)= 1 because deterministic
    # - prob(v_k, v_k1)= depends on the distribution of w_k_flap which is uniform in {-v_dev, +v_dev}. The returned array of velocities 
    #   has equal probability for each v_k, u
    # - prob(d_k, d_k1)= depends on p_spawn. With p_spawn we have d_k1[:,:,1], with 1-p_spawn we have d_k1[:,:,0]
    # - prob(h_k, h_k1)= depends on p_spawn. With p_spawn we have h_k1[:,:,:,1] with 1-p_spawn we have h_k1[:,:,:,0]. 
    #   Values are uniformly distributed along axis=2 (0-indexed). For h_k1[:,:,:,0] they are all the same. For h_k1[:,:,:,0] they take same probability

    # - Total prob of going from one state to another is: 1 * 1/(2v_dev + 1) * p_spawn * 1/(len(C.S_h)) for spawn case
    # - Total prob of going from one state to another is: 1 * 1/(2v_dev + 1) * (1-p_spawn) for no-spawn case
    # - sum(prob)=1! GOOD
    
    # TODO: for each state build an array that contains all the possible next_states. For how this was built, the calculation of 
    # probabilities is actually the same for each current state and can be done vectorially

    current_states = np.arange(C.K)
    for i in range(2): #no spawn (i=0) / spawn (i=1)
        for u in range(C.L):
            if u == 0 or u == 1: # no_flap/weak_flap
                if i == 0: #no spawn
                    tuples = [(y_k1[s], v_k1[s, u, 0], *d_k1[s, :, i], *h_k1[s, :, 0, i]) for s in range(C.K)]     
                    indices = [index_map[t] for t in tuples]
                    P[current_states, indices, u] = (1-p_spawn) #deterministic update for velocities for no flap or weak flap, only stochastic thing is no_spawn 
                
                else: #spawn 
                    for h in range(len(C.S_h)):    
                        tuples = [(y_k1[s], v_k1[s, u, 0], *d_k1[s, :, i], *h_k1[s, :, h, i]) for s in range(C.K)]     
                        indices = [index_map[t] for t in tuples]
                        P[current_states, indices, u] = (1/w_h_dim)*(p_spawn)

            else: #strong flap
                if i == 0: #no spawn
                    for v in range(flap_space_dim):
                        tuples = [(y_k1[s], v_k1[s, u, v], *d_k1[s, :, i], *h_k1[s, :, 0, i]) for s in range(C.K)]     
                        indices = [index_map[t] for t in tuples]
                        P[current_states, indices, u] = (1/flap_space_dim)*(1-p_spawn) #stochastic update for velocities this time 
                
                else: #spawn 
                    for v in range(flap_space_dim):
                        for h in range(len(C.S_h)):    
                            tuples = [(y_k1[s], v_k1[s, u, v], *d_k1[s, :, i], *h_k1[s, :, h, i]) for s in range(C.K)]     
                            indices = [index_map[t] for t in tuples]
                            P[current_states, indices, u] = (1/flap_space_dim)*(1/w_h_dim)*(p_spawn) #stochastic update for velocities this time 
            
    return P


    
        