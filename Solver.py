"""Solver.py

Template to solve the stochastic shortest path problem.

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
from ComputeTransitionProbabilities import *
from ComputeExpectedStageCosts import *
from scipy.sparse import csr_matrix


"""def solution(C: Const) -> tuple[np.ndarray, np.ndarray]:
    # You're free to use the functions below, implemented in the previous
    # tasks, or come up with something else.
    # If you use them, you need to add the corresponding imports 
    # at the top of this file.
    # P = compute_transition_probabilities(C)
    # Q = compute_expected_stage_cost(C)
    
    # TODO: implement Value Iteration, Policy Iteration, Linear Programming 
    # or a combination of these

    # Load dynamics and costs
    P = compute_transition_probabilities(C)     # (K, K, L)
    Q = compute_expected_stage_cost(C)          # (K, L)

    K= C.K

    # Initialize
    J = np.zeros(K)
    current_policy = np.zeros(K, dtype=int)     # start with all actions = 0 (no flap)

    diff = True

    while diff:
        old_policy = current_policy.copy()

        # ---------------------------------------
        # 1) POLICY EVALUATION
        # ---------------------------------------

        a_idx = current_policy
        # Extract P_pi(i,j) = P(i,j,a_i)
        P_pi = np.take_along_axis(
            P,
            a_idx[:, None, None],
            axis=2
        )[..., 0]     # shape (K,K)

        # Extract Q_pi(i) = Q(i,a_i)
        Q_pi = np.take_along_axis(Q, a_idx[:, None], axis=1)[:, 0]  # shape (K,)

        # Evaluate J through fixed number of Bellman iterations
        for _ in range(2500):
            J = Q_pi + P_pi @ J

        # ---------------------------------------
        # 2) POLICY IMPROVEMENT
        # ---------------------------------------

        # Compute Q(i,u) + sum_j P(i,j,u)*J(j)
        J_Q = Q + np.sum(P * J[None, :, None], axis=1)   # shape (K,L)

        # Best action = argmin
        current_policy = np.argmin(J_Q, axis=1)

        diff = np.any(current_policy != old_policy)

    # Final optimal cost from last policy to be returned
    J_Q = Q + np.sum(P * J[None, :, None], axis=1)
    J_opt = np.min(J_Q, axis=1)
    u_opt_ind = current_policy
    u_opt = np.array([C.input_space[ind] for ind in u_opt_ind])

    return J_opt, u_opt
"""



def solution(C: Const) -> tuple[np.ndarray, np.ndarray]:
    #P = compute_transition_probabilities(C)   # (K, K, L)
    Q = compute_expected_stage_cost(C)        # (K, L)
    K, L = C.K, C.L
    #P_sparse = []
    #for u in range(C.L):
    #    P_sparse.append(csr_matrix(P[:, :, u]))
    
    P_sparse = []
    #P = np.zeros((C.K, C.K, C.L))
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
        P_sparse.append(coo_matrix((all_probs, (all_cstates, all_nstates)), shape=(C.K, C.K)).tocsr())

    J = np.zeros(K)
    #max_iters = 1000
    tol = 1e-5
    do = True
    count = 0
    while do:
        # Bellman backup: for each (i,u)
        # J_new(i,u) = Q(i,u) + sum_j P(i,j,u) * J(j)
        #J_Q = Q + np.sum(P * J[None, :, None], axis=1)  # (K, L)
        #J_new = J_Q.min(axis=1)                         # (K,)
        J_new = np.empty_like(J)
        count += 1

        for u in range(C.L):
            J_new_u = Q[:, u] + P_sparse[u].dot(J)
            if u == 0:
                J_new = J_new_u
            else:
                J_new = np.minimum(J_new, J_new_u)

        if np.max(np.abs(J_new - J)) < tol:
            break

        J = J_new

    # Extract policy corresponding to final J
    J_Q = Q + np.sum(P * J[None, :, None], axis=1)      # (K, L)
    u_opt_ind = np.argmin(J_Q, axis=1)
    u_opt = np.array([C.input_space[ind] for ind in u_opt_ind])
    print(f"Count{count}")


    return J, u_opt
