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
import time 
from scipy.sparse import coo_matrix

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
    #PRECOMPUTATIONS: 
    # Create the state space and build numerical encoding for later indexing 
    input_space = C.input_space
    state_space = np.array(C.state_space, dtype=np.int64)

    state_space_for_keys = state_space.copy()
    state_space_for_keys[:, 1] += C.V_max
    dims = [
            len(C.S_y),
            len(C.S_v),
            len(C.S_d1),
            *([len(C.S_d)] * (C.M - 1)),
            *([len(C.S_h)] * C.M),
            ]
    #dims contains the whole state space, also the INVALID STATES. But we don't care if we create an encoding also for invalid arrays, because then the map
    #is built only out of the valid state space 
    stride = np.cumprod([1] + dims[:-1])
    #the stride defines for each parameter of the tuple(y, v, d1, d2, ...., dM, h1, h2, ..., hM), how fast it varies:
    
    #Here we choose y to vary fastest (stride[0]=1), then v varies as soon as y runs out of dimension, so varies each len(C.S_y). Then the third object
    # varies as soon as both run out of dimension, so each len(C.S_y)*len(C.S_v), which is what cumprod is doing
    # To effectively encode the states in the state space we multiply each parameter for the corresponding number on the stride, and sum them up together 
    encoded = (state_space_for_keys * stride).sum(axis=1)
    #Now we effectively need to create the indexing array. To do so we must remember that keys are not consecutive because of invalid states
    #Therefore we fill the array with -1 to flag with invalid states, and the array must be at least max_key dimensional. 
    max_key = encoded.max()
    state_index_map = -np.ones(max_key + 1, dtype=np.int32)

    #Fill the array with the unique encoding keys. This is an inverse array: given the key it returns the corresponding state. If state s has key k,
    # and the key can be computed by (s*stride), the map returns the state corresponding to s (can be retrieved thanks to encoded that computes for every valid state)  
    state_index_map[encoded] = np.arange(C.K)
    

    # TODO fill the transition probability matrix P here
    w_h_dim = len(C.S_h)
    #The transition probability first computes the dynamics considering the disturbances, and then assigns probabilities based on the
    #reachable next states. This has to be done for all the possible states in the state space

    ### POSE DYNAMICS ONE-SHOT COMPUTATION###
    #Current heights, velocities and obstacles' poses
    
    y_k = np.array([s[0] for s in state_space]) #(K,)
    v_k = np.array([s[1] for s in state_space]) #(K,)

    d_k = np.array([s[2 : 2 + C.M] for s in state_space]) #(K,M) 
    h_k = np.array([s[2 + C.M : ] for s in state_space])  #(K,M)
    
    half = (C.G - 1) // 2
    not_in_gap = abs(y_k - h_k[:,0]) > half
    in_col_1 = d_k[:,0] == 0   #mask selcting where we have an obst in 1st column

    is_colliding_mask = not_in_gap & in_col_1 #(K,) array of booleans indicating whether it's colliding or not
    valid = np.logical_not(is_colliding_mask)
    valid_indices = np.nonzero(valid)[0] 

    y_k_valid = y_k[valid]
    v_k_valid = v_k[valid]
    
    d_k_valid = d_k[valid, :]
    h_k_valid = h_k[valid, :]
    
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
    for u in range(C.L):
        #start_loop = time.time()
        all_cstates = []
        all_nstates = []
        all_probs = []
        for i in range(2): #no spawn (i=0) / spawn (i=1)
            if u == 0 or u == 1: # no_flap/weak_flap
                if i == 0: #no spawn
                    tuples = np.column_stack((y_k1_int, v_k1_int[:, u, 0] + C.V_max, d_k1_int[:, :, 0], h_k1_int[:, :, 0, 0]))
                    encoded_next = (tuples * stride).sum(axis=1)
                    indices = state_index_map[encoded_next]
                    #indices = np.array([index_map[tuple(row)] for row in tuples], dtype=np.int32) #(K_valid, )

                    all_cstates.append(current_states)
                    all_nstates.append(indices)
                    all_probs.append(1-p_spawn)#deterministic update for velocities for no flap or weak flap, only stochastic thing is no_spawn 
                    
                else: #spawn 
                    for h in range(len(C.S_h)):  
                        tuples = np.column_stack((y_k1_int, v_k1_int[:, u, 0] + C.V_max, d_k1_int[:, :, 1], h_k1_int[:, :, h, 1]))
                        encoded_next = (tuples * stride).sum(axis=1)
                        indices = state_index_map[encoded_next]
                        #indices = np.array([index_map[tuple(row)] for row in tuples], dtype=np.int32)
                        all_cstates.append(current_states)
                        all_nstates.append(indices)
                        all_probs.append((1/w_h_dim)*(p_spawn))
                              

            else: #strong flap
                if i == 0: #no spawn
                    for v in range(flap_space_dim):
                        tuples = np.column_stack((y_k1_int, v_k1_int[:, u, v] + C.V_max, d_k1_int[:, :, 0], h_k1_int[:, :, 0, 0]))
                        encoded_next = (tuples * stride).sum(axis=1)
                        indices = state_index_map[encoded_next]
                        #indices = np.array([index_map[tuple(row)] for row in tuples], dtype=np.int32)
                        all_cstates.append(current_states)
                        all_nstates.append(indices)
                        all_probs.append((1/flap_space_dim)*(1-p_spawn))
                else: #spawn 
                    for v in range(flap_space_dim):
                        for h in range(len(C.S_h)):    
                            tuples = np.column_stack((y_k1_int, v_k1_int[:, u, v] + C.V_max, d_k1_int[:, :, 1], h_k1_int[:, :, h, 1]))
                            encoded_next = (tuples * stride).sum(axis=1)
                            indices = state_index_map[encoded_next]
                            #indices = np.array([index_map[tuple(row)] for row in tuples], dtype=np.int32)
                            all_cstates.append(current_states)
                            all_nstates.append(indices)
                            all_probs.append((1/flap_space_dim)*(1/w_h_dim)*(p_spawn))
        

        all_cstates = np.concatenate(all_cstates)
        all_nstates = np.concatenate(all_nstates)
        all_probs = np.concatenate(all_probs)
        np.add.at(P[:, :, u], (all_cstates, all_nstates), all_probs)
    
    
    return P


    
        