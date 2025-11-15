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
from scipy.sparse import csr_matrix, coo_matrix

def modified_policy(
    C: Const,
    P_sparse: list[csr_matrix],
    Q: np.ndarray,
    eval_iters: int = 20,   # number of times THE POLICY gets improved (threshold is ~20)
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Modified policy iteration using sparse transition matrices.

    Args:
        C: problem constants
        P_sparse: list of length C.L, each element is a csr_matrix of shape (K, K)
                  with P_sparse[u][i, j] = P(i -> j | action u)
        Q: expected stage cost, shape (K, L)
        eval_iters: number of Bellman iterations for policy evaluation step
        tol: tolerance for early stopping of the evaluation loop

    Returns:
        J_opt: optimal cost-to-go, shape (K,)
        u_opt: optimal control values (actual inputs, not indices), shape (K,)
    """
    # Initialize value function and an initial proper policy (all "no flap")
    K, L = C.K, C.L
    J = np.zeros(K)
    current_policy = np.zeros(K, dtype=int)  # action indices in {0, ..., L-1}

    while True:
        old_policy = current_policy.copy()  # save the old policy
        a_idx = current_policy

        # ---------- 1) Policy evaluation (approximate) ----------
        for _ in range(eval_iters):
            J_old = J.copy()

            # Precompute P_u * J for all actions w.r.t. J_old
            PJ_list = [P_sparse[u].dot(J_old) for u in range(L)]

            # Update J only on the rows that use each action u, the others are invalid
            for u in range(L):
                mask = (a_idx == u)
                if not np.any(mask):
                    continue
                J[mask] = Q[mask, u] + PJ_list[u][mask]

            # Optional early stop if evaluation converged enough (unlikely)
            if np.max(np.abs(J - J_old)) < tol:
                break

        # ---------- 2) Policy improvement ----------
        # For each action u, compute Q(i,u) + sum_j P(i,j,u) * J(j)
        Q_plus = np.empty((K, L))
        for u in range(L):
            Q_plus[:, u] = Q[:, u] + P_sparse[u].dot(J)

        current_policy = np.argmin(Q_plus, axis=1)

        # Check for policy convergence
        if np.all(current_policy == old_policy):
            break

    # Final optimal cost and policy
    J_opt = Q_plus.min(axis=1)
    u_opt_ind = current_policy
    u_opt = np.array([C.input_space[idx] for idx in u_opt_ind])

    # J_opt improvement since we did not give it enough time
    #J_opt, _ = value_iteration(C, P_sparse, Q, J_init = J_opt)
    return J_opt, u_opt




def value_iteration(C: Const, P_sparse: list[csr_matrix], Q, J_init = None, tol: float = 1e-5, iters: int = 500) -> tuple[np.ndarray, np.ndarray]:
    J = np.zeros(C.K) #if (J_init is None) else J_init
    count = 0
    while True:
        # Bellman backup: for each (i,u)
        J_new = np.empty_like(J)
        count += 1
        
        for u in range(C.L):
            J_new = Q[:, u] + P_sparse[u].dot(J)
            J = np.minimum(J_new, J)
        
        if (count % iters) == iters-1: 
            J_conv = J.copy()

        if (count % iters) == 0: 
            if np.max(np.abs(J_conv - J)) < tol:
                break


    # Extract policy corresponding to final J
    #end = time.time()
    expected_value = np.column_stack([(P_sparse[u].dot(J))for u in range(C.L)])
    J_Q = Q + expected_value
    u_opt_ind = np.argmin(J_Q, axis=1)
    u_opt = np.array([C.input_space[ind] for ind in u_opt_ind])

    print(f"Count: {count}")
    return (J, u_opt) 

def GS_value_iteration(C: Const, P_sparse: list[csr_matrix], Q, tol: float = 1e-5, iters: int = 500) -> tuple[np.ndarray, np.ndarray]:
    J = np.zeros(C.K)

    start_idx = []
    indptr_idx = []
    probs = []
    for u in range(C.L):
        start_idx.append(P_sparse[u].indices)
        indptr_idx.append(P_sparse[u].indptr)
        probs.append(P_sparse[u].data)

    while True:
        # Bellman backup: for each (i,u)
        for iter in range(iters): 
            for i in range(C.K):
                J_cands = []
                for u in range(C.L):
                    st_idx = start_idx[u]
                    iptr_idx = indptr_idx[u]
                    pr = probs[u]
                    next_idx = st_idx[iptr_idx[i]:iptr_idx[i+1]]
                    p_ij = pr[iptr_idx[i]:iptr_idx[i+1]]
                    next_V = J[next_idx]
                    if p_ij.size == 0 : 
                        J_u = Q[i, u]
                    else: 
                        J_u = Q[i, u] + np.dot(p_ij, next_V) 
                    J_cands.append(J_u)
                
                J[i] = min(J_cands)
            if iter == iters - 2: 
                J_new = J.copy()
                
        if np.max(np.abs(J_new - J)) < tol:
            break

    # Extract policy corresponding to final J
    #end = time.time()
    expected_value = np.column_stack([(P_sparse[u].dot(J))for u in range(C.L)])
    J_Q = Q + expected_value
    u_opt_ind = np.argmin(J_Q, axis=1)
    u_opt = np.array([C.input_space[ind] for ind in u_opt_ind])

    return J, u_opt

def solution(C: Const) -> tuple[np.ndarray, np.ndarray]:
    Q = compute_expected_stage_cost(C)        # (K, L)
    K, L = C.K, C.L

    start = time.time()
    P_sparse = []
    #PRECOMPUTATIONS: 
    # Create the state space and build numerical encoding for later indexing 
    input_space = C.input_space
    state_space = np.array(C.state_space, dtype=np.int64)

    state_space_for_keys = state_space.copy()
    state_space_for_keys[:, 1] += C.V_max    #offset to avoid negative indices offset 
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
    w_h_dim = len(C.S_h)
    
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
   
    current_states = valid_indices
    for u in range(C.L):
        all_cstates = []
        all_nstates = []
        all_probs = []
        for i in range(2): #no spawn (i=0) / spawn (i=1)
            if u == 0 or u == 1: # no_flap/weak_flap
                if i == 0: #no spawn
                    tuples = np.column_stack((y_k1_int, v_k1_int[:, u, 0] + C.V_max, d_k1_int[:, :, 0], h_k1_int[:, :, 0, 0]))
                    encoded_next = (tuples * stride).sum(axis=1)
                    indices = state_index_map[encoded_next]
                    all_cstates.append(current_states)
                    all_nstates.append(indices)
                    all_probs.append(1-p_spawn)#deterministic update for velocities for no flap or weak flap, only stochastic thing is no_spawn 
                    
                else: #spawn 
                    for h in range(len(C.S_h)):  
                        tuples = np.column_stack((y_k1_int, v_k1_int[:, u, 0] + C.V_max, d_k1_int[:, :, 1], h_k1_int[:, :, h, 1]))
                        encoded_next = (tuples * stride).sum(axis=1)
                        indices = state_index_map[encoded_next]
                        all_cstates.append(current_states)
                        all_nstates.append(indices)
                        all_probs.append((1/w_h_dim)*(p_spawn))
                              

            else: #strong flap
                if i == 0: #no spawn
                    for v in range(flap_space_dim):
                        tuples = np.column_stack((y_k1_int, v_k1_int[:, u, v] + C.V_max, d_k1_int[:, :, 0], h_k1_int[:, :, 0, 0]))
                        encoded_next = (tuples * stride).sum(axis=1)
                        indices = state_index_map[encoded_next]
                        all_cstates.append(current_states)
                        all_nstates.append(indices)
                        all_probs.append((1/flap_space_dim)*(1-p_spawn))
                else: #spawn 
                    for v in range(flap_space_dim):
                        for h in range(len(C.S_h)):    
                            tuples = np.column_stack((y_k1_int, v_k1_int[:, u, v] + C.V_max, d_k1_int[:, :, 1], h_k1_int[:, :, h, 1]))
                            encoded_next = (tuples * stride).sum(axis=1)
                            indices = state_index_map[encoded_next]
                            all_cstates.append(current_states)
                            all_nstates.append(indices)
                            all_probs.append((1/flap_space_dim)*(1/w_h_dim)*(p_spawn))
        
        all_cstates = np.concatenate(all_cstates)
        all_nstates = np.concatenate(all_nstates)
        all_probs = np.concatenate(all_probs)
        P_sparse.append(coo_matrix((all_probs, (all_cstates, all_nstates)), shape=(C.K, C.K)).tocsr())

    end = time.time()
    print(f"Time elapsed with sparse matrix construction: {end-start}")
    #compare_dense_sparse(P, P_sparse)
    
    (J, u_opt) = value_iteration(C, P_sparse, Q, tol=1e-5, iters=500)
    print(f"VI time: {end-start}")

    return (J, u_opt)
