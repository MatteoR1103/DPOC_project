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
    
    half = (C.G - 1) // 2
    not_in_gap = abs(y_k - h_k[:,0]) > half
    in_col_1 = d_k[:,0] == 0   #mask selcting where we have an obst in 1st column

    is_colliding_mask = not_in_gap & in_col_1 #(K,) array of booleans indicating whether it's passing or not
    valid = np.logical_not(is_colliding_mask)
    valid_indices = np.nonzero(valid)[0] 

    y_k_valid = y_k[valid]
    v_k_valid = v_k[valid]
    
    d_k_valid = d_k[valid, :]
    h_k_valid = h_k[valid, :]

    K_valid = y_k_valid.shape[0]

    print(f"K_valid = {K_valid}")
    
    # FROM NOW ON K HAS TO BE INTENDED AS THE VALID (NON-COLLIDING) STATE DIMENSION

    #Next height dynamics
    y_k1 = compute_height_dynamics(y_k = y_k_valid, v_k=v_k_valid, C=C) #(K,)

    #Next vel dynamics
    v_dev_inter = np.arange(-C.V_dev, C.V_dev + 1)
    flap_space_dim = len(v_dev_inter)
    v_k1 = compute_vel_dynamics(v_k_valid, input_space, v_dev_inter, C) #(K, input_space, 2*V_dev +1)

    #Next obstacles' dynamics
    d_k1, h_k1, p_spawn = compute_obst_dynamics(d_k_valid, h_k_valid, y_k_valid, C) #tuple((K,M,2), (K, M, len(S_h), 2), (K,))

    y_k1_int = y_k1.astype(int)
    v_k1_int = v_k1.astype(int)
    d_k1_int = d_k1.astype(int)
    h_k1_int = h_k1.astype(int)

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

    current_states = valid_indices
    for i in range(2): #no spawn (i=0) / spawn (i=1)
        for u in range(C.L):
            if u == 0 or u == 1: # no_flap/weak_flap
                if i == 0: #no spawn
                    tuples = [(y_k1_int[s], v_k1_int[s, u, 0], *d_k1_int[s, :, i], *h_k1_int[s, :, 0, i]) for s in range(K_valid)]     
                    indices = [index_map[t] for t in tuples]
                    #print(f"indices and current states: {len(indices), len(current_states)}")
                    np.add.at(P[:, :, u], (current_states, indices), (1 - p_spawn)) 
                    #deterministic update for velocities for no flap or weak flap, only stochastic thing is no_spawn 
                    #np.add.at handles cases in which multiple indices are the same, and adds probabilities up
                
                else: #spawn 
                    for h in range(len(C.S_h)):    
                        tuples = [(y_k1_int[s], v_k1_int[s, u, 0], *d_k1_int[s, :, i], *h_k1_int[s, :, h, i]) for s in range(K_valid)]     
                        indices = [index_map[t] for t in tuples]
                        #print(f"indices and current states: {len(indices), len(current_states)}")
                        np.add.at(P[:, :, u], (current_states, indices), (1/w_h_dim)*(p_spawn))

            else: #strong flap
                if i == 0: #no spawn
                    for v in range(flap_space_dim):
                        tuples = [(y_k1_int[s], v_k1_int[s, u, v], *d_k1_int[s, :, i], *h_k1_int[s, :, 0, i]) for s in range(K_valid)]     
                        indices = [index_map[t] for t in tuples]
                        #print(f"indices and current states: {len(indices), len(current_states)}")
                        np.add.at(P[:, :, u], (current_states, indices), (1/flap_space_dim)*(1-p_spawn)) #stochastic update for velocities this time 
                
                else: #spawn 
                    for v in range(flap_space_dim):
                        for h in range(len(C.S_h)):    
                            tuples = [(y_k1_int[s], v_k1_int[s, u, v], *d_k1_int[s, :, i], *h_k1_int[s, :, h, i]) for s in range(K_valid)]     
                            indices = [index_map[t] for t in tuples]
                            #print(f"indices and current states: {len(indices), len(current_states)}")
                            np.add.at(P[:, :, u], (current_states, indices), (1/flap_space_dim)*(1/w_h_dim)*(p_spawn))
    """
    PROBABILITY CHECK PASSED
    sum_probs = P.sum(axis=1)  # shape (K, L)

    # Identify rows that do not sum to 1 within a small tolerance
    bad_mask = np.abs(sum_probs - 1.0) > 1e-8
    bad_indices = np.argwhere(bad_mask)

    if bad_indices.size == 0:
        print("All (state, action) pairs correctly normalize to 1.")
    else:
        print(f"Found {len(bad_indices)} (state, action) pairs not normalized:")
        for k, u in bad_indices:
            print(f"  State {k}, Action {u}, Sum = {sum_probs[k, u]:.6f}")
    """
    
    return P


    
        