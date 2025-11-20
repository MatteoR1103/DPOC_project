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
import time
from Const import Const
from ComputeTransitionProbabilities import *
from ComputeExpectedStageCosts import *
from scipy.sparse import csr_matrix, coo_matrix, # eye, vstack

def solution(C: Const) -> tuple[np.ndarray, np.ndarray]:
    start = time.perf_counter()
    P_sparse = []
    # PRECOMPUTATIONS: build reduced state-space encodings and masks
    (
        _,
        encoded_to_compact,
        stride,
        valid_indices,
        y_k_valid,
        v_k_valid,
        d_k_valid,
        h_k_valid,
        compact_size,
        K_valid,
    ) = build_state_space(C)

    # Compute dynamics (heights, velocities, obstacles, spawn probabilities)
    (
        y_k1_int,
        v_k1_int,
        d_k1_int,
        h_k1_int,
        p_spawn,
    ) = build_dynamics(C, y_k_valid, v_k_valid, d_k_valid, h_k_valid)
   
    q = np.array([-1, -1 + C.lam_weak, -1 + C.lam_strong])

    # Build reduced sparse transition matrices
    P_sparse = build_P_sparse(
        C,
        y_k1_int=y_k1_int,
        v_k1_int=v_k1_int,
        d_k1_int=d_k1_int,
        h_k1_int=h_k1_int,
        encoded_to_compact=encoded_to_compact,
        stride=stride,
        valid_indices=valid_indices,
        p_spawn=p_spawn,
        compact_size=compact_size,
    )
    end = time.perf_counter()
    print(f"Time for precomps: {end-start}")

    n_state_threshold = 3500 # threshold to switch between methods
    J_compact, u_opt_compact = modified_policy(C=C, P_sparse=P_sparse, q=q, K_valid=K_valid+1, tol=1e-5) if C.K > n_state_threshold else value_iteration_in_place(C=C, P_sparse=P_sparse, q=q, K_valid=K_valid+1, tol=1e-5)

    # Expand results back to full state-space size C.K
    J_full = np.full(C.K, -1.0)
    u_opt_full = np.zeros(C.K)
    # map optimal costs into J_opt_full; ignore dummy last entry
    J_full[valid_indices] = J_compact[:K_valid]
    # map actions into u_opt_full length; ignore dummy last entry
    u_opt_full[valid_indices] = u_opt_compact[:K_valid]

    return (J_full, u_opt_full)

# HELPER FUNCTIONS BELOW
def build_state_space(C: Const):
    """Build state-space arrays and indexing helpers.

    Returns a tuple:
      (input_space, state_space, state_index_map, stride,
       valid_indices, y_k_valid, v_k_valid, d_k_valid, h_k_valid)
    """
    state_space = np.array(C.state_space, dtype=np.int64)

    y_k = np.array([s[0] for s in state_space])
    v_k = np.array([s[1] for s in state_space])
    d_k = np.array([s[2 : 2 + C.M] for s in state_space])
    h_k = np.array([s[2 + C.M : ] for s in state_space])

    half = (C.G - 1) // 2
    not_in_gap = abs(y_k - h_k[:, 0]) > half
    in_col_1 = d_k[:, 0] == 0
    is_colliding_mask = not_in_gap & in_col_1
    valid = np.logical_not(is_colliding_mask)
    valid_indices = np.nonzero(valid)[0]

    y_k_valid = y_k[valid]
    v_k_valid = v_k[valid]
    d_k_valid = d_k[valid, :]
    h_k_valid = h_k[valid, :]
    
    state_space_for_keys = state_space.copy()
    state_space_for_keys[:, 1] += C.V_max
    dims = [
        len(C.S_y),
        len(C.S_v),
        len(C.S_d1),
        *([len(C.S_d)] * (C.M - 1)),
        *([len(C.S_h)] * C.M),
    ]
    stride = np.cumprod([1] + dims[:-1])

    encoded = (state_space_for_keys * stride).sum(axis=1)
    max_key = int(encoded.max())

    # Build compact mapping: map encoded keys directly to compact indices
    K_valid = valid_indices.size
    dummy_idx = K_valid  # last index is dummy for all the invalid states
    # encoded -> compact (direct array). All invalid encoded keys map to dummy
    encoded_to_compact = np.full(max_key + 1, dummy_idx, dtype=np.int32)
    encoded_valid = encoded[valid_indices].astype(np.int64)
    encoded_to_compact[encoded_valid] = np.arange(K_valid, dtype=np.int32)
    compact_size = K_valid + 1
    
    return (
        state_space,
        encoded_to_compact,
        stride,
        valid_indices,
        y_k_valid,
        v_k_valid,
        d_k_valid,
        h_k_valid,
        compact_size,
        K_valid,
    )

def build_P_sparse(
    C: Const,
    y_k1_int: np.ndarray,
    v_k1_int: np.ndarray,
    d_k1_int: np.ndarray,
    h_k1_int: np.ndarray,
    encoded_to_compact: np.ndarray,
    stride: np.ndarray,
    valid_indices: np.ndarray,
    p_spawn: np.ndarray,
    compact_size: int,
) -> list:
    """Build list of CSR transition matrices (one per action).

    This builds compact CSR matrices of shape (compact_size, compact_size),
    where all invalid/full-missing next states are aggregated into the dummy
    index (last row/column = compact_size-1).
    Returns: list of length C.L with CSR matrices shape (compact_size, compact_size).
    """
    P_sparse = []
    # current_states = valid_indices
    # compact rows correspond to valid states in the same order -> 0..K_valid-1
    K_valid = valid_indices.size
    current_states_compact = np.arange(K_valid, dtype=np.int32)
    w_h_dim = len(C.S_h)
    flap_space_dim = v_k1_int.shape[2]

    for u in range(C.L):
        all_cstates = []
        all_nstates = []
        all_probs = []

        for i in range(2):  # no spawn (i=0) / spawn (i=1)
            if u == 0 or u == 1:  # no_flap/weak_flap
                if i == 0:  # no spawn
                    tuples = np.column_stack((
                        y_k1_int,
                        v_k1_int[:, u, 0] + C.V_max,
                        d_k1_int[:, :, 0],
                        h_k1_int[:, :, 0, 0],
                    ))
                    encoded_next = (tuples * stride).sum(axis=1).astype(np.int64)
                    indices_compact = encoded_to_compact[encoded_next]
                    all_cstates.append(current_states_compact)
                    all_nstates.append(indices_compact)
                    all_probs.append(1 - p_spawn)
                else:  # spawn
                    for h in range(w_h_dim):
                        tuples = np.column_stack((
                            y_k1_int,
                            v_k1_int[:, u, 0] + C.V_max,
                            d_k1_int[:, :, 1],
                            h_k1_int[:, :, h, 1],
                        ))
                        encoded_next = (tuples * stride).sum(axis=1).astype(np.int64)
                        indices_compact = encoded_to_compact[encoded_next]
                        all_cstates.append(current_states_compact)
                        all_nstates.append(indices_compact)
                        all_probs.append((1 / w_h_dim) * (p_spawn))
            else:  # strong flap
                if i == 0:  # no spawn
                    for v in range(flap_space_dim):
                        tuples = np.column_stack((
                            y_k1_int,
                            v_k1_int[:, u, v] + C.V_max,
                            d_k1_int[:, :, 0],
                            h_k1_int[:, :, 0, 0],
                        ))
                        encoded_next = (tuples * stride).sum(axis=1).astype(np.int64)
                        indices_compact = encoded_to_compact[encoded_next]
                        all_cstates.append(current_states_compact)
                        all_nstates.append(indices_compact)
                        all_probs.append((1 / flap_space_dim) * (1 - p_spawn))
                else:  # spawn
                    for v in range(flap_space_dim):
                        for h in range(w_h_dim):
                            tuples = np.column_stack((
                                y_k1_int,
                                v_k1_int[:, u, v] + C.V_max,
                                d_k1_int[:, :, 1],
                                h_k1_int[:, :, h, 1],
                            ))
                            encoded_next = (tuples * stride).sum(axis=1).astype(np.int64)
                            indices_compact = encoded_to_compact[encoded_next]
                            all_cstates.append(current_states_compact)
                            all_nstates.append(indices_compact)
                            all_probs.append((1 / flap_space_dim) * (1 / w_h_dim) * (p_spawn))

        all_cstates = np.concatenate(all_cstates)
        all_nstates = np.concatenate(all_nstates)
        all_probs = np.concatenate(all_probs)

        P_sparse.append(coo_matrix((all_probs, (all_cstates, all_nstates)), shape=(compact_size, compact_size)).tocsr())

    return P_sparse

def build_dynamics(C: Const, y_k_valid: np.ndarray, v_k_valid: np.ndarray, d_k_valid: np.ndarray, h_k_valid: np.ndarray):
        """Compute next-state dynamics and return integer arrays + spawn probs.

        Returns:
            y_k1_int, v_k1_int, d_k1_int, h_k1_int, p_spawn
        """
        # Next height dynamics (deterministic)
        y_k1 = compute_height_dynamics(y_k=y_k_valid, v_k=v_k_valid, C=C)

        # Next vel dynamics
        v_dev_inter = np.arange(-C.V_dev, C.V_dev + 1)
        v_k1 = compute_vel_dynamics(v_k_valid, C.input_space, v_dev_inter, C)

        # Next obstacles' dynamics
        d_k1, h_k1, p_spawn = compute_obst_dynamics(d_k_valid, h_k_valid, y_k_valid, C)

        y_k1_int = y_k1.astype(int)
        v_k1_int = v_k1.astype(int)
        d_k1_int = d_k1.astype(int)
        h_k1_int = h_k1.astype(int)

        return y_k1_int, v_k1_int, d_k1_int, h_k1_int, p_spawn

def build_P_policy(C: Const, P_sparse, policy, K_valid):

    state_list = []
    next_state_list = []
    prob_list = []
    
    for u in range(C.L):
        rows_u = np.where(policy == u)[0]

        sub = P_sparse[u][rows_u, :] 
        sub_coo = sub.tocoo()

        next_state_list.append(sub_coo.col)
        prob_list.append(sub_coo.data)
    
        state_list.append(rows_u[sub_coo.row])

    states = np.concatenate(state_list)
    next_states = np.concatenate(next_state_list)
    probs = np.concatenate(prob_list)
    P_pi = csr_matrix((probs, (states, next_states)), shape=(K_valid, K_valid))
    return P_pi

def modified_policy(
    C: Const,
    K_valid: int,
    P_sparse: list[csr_matrix],
    q: np.ndarray,
    eval_iters: int = 300,   # number of times THE POLICY gets improved (threshold is ~20, but higher give improvs)
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
    # Determine compact state dimension 
    J = np.zeros(K_valid)
    Q = np.tile(q, (K_valid, 1))

    current_policy = np.zeros(K_valid, dtype=int)  # action indices in {0, ..., L-1}

    start_p_iter = time.time()
    while True:
        old_policy = current_policy.copy()  # save the old policy
        P_policy = build_P_policy(C, P_sparse=P_sparse, policy=current_policy, K_valid=K_valid)    
        Q_pi = Q[np.arange(K_valid), current_policy] 
        
        #Value Improvement
        for _ in range(eval_iters):
            #J_old = J.copy()
            J = Q_pi + P_policy.dot(J)
            # if np.max(np.abs(J - J_old))< tol: 
            #     print("BROKE")
            #     break


        expected_value = np.column_stack([(P_sparse[u].dot(J))for u in range(C.L)])
        J_Q = q + expected_value
        current_policy = np.argmin(J_Q, axis=1)

        # Check for policy convergence
        if np.all(current_policy == old_policy):
            break
    
    end_p_iter = time.time()
    print(f"Policy improvement time : {end_p_iter-start_p_iter}")
    
    #DO IMPROVEMENT ON THE SETTLED POLICY
    count = 0
    check_conv_iters = 20
    start_iter = time.time()

    while True:
        J_old = J.copy()
        J = Q_pi + P_policy.dot(J_old)
        count +=1
        if (count % check_conv_iters) == 0:
            if np.max(np.abs(J - J_old))< tol: 
                break
    
    end_iter = time.time()

    print(f"Value improvement time : {end_iter-start_iter}")

    u_opt_ind = current_policy
    u_opt = np.array([C.input_space[idx] for idx in u_opt_ind])
    
    return J, u_opt
    
def value_iteration_in_place(C: Const, P_sparse: list[csr_matrix], q: np.ndarray, K_valid: int, J_init=None, tol: float = 1e-5, iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    # Determine compact state dimension 
    #J = np.zeros(K_valid)

    if J_init is None:
        J = np.zeros(K_valid)
    else:
        J = J_init.copy()  # should be length K

    count = 0
    start = time.time()

    J_new = np.zeros_like(J)
    J_conv = np.empty(K_valid)

    while True:
        # Bellman backup: for each (i,u)
        count += 1

        for u in range(C.L):
            J_new = q[u] + P_sparse[u].dot(J)
            J = np.minimum(J_new, J)

        if (count % iters) == iters - 1:
            J_conv = J.copy()

        if (count % iters) == 0:
            if np.max(np.abs(J_conv - J)) < tol:
                print(f"Optimized VI converged in {count} iterations.")
                break
    end = time.time()

    print(f"Time for value: {end-start}")

    # Extract policy corresponding to final J (compact indices)
    expected_value = np.column_stack([(P_sparse[u].dot(J)) for u in range(C.L)])
    J_Q = q + expected_value
    u_opt_ind = np.argmin(J_Q, axis=1)
    u_opt = np.array([C.input_space[ind] for ind in u_opt_ind])

    return J, u_opt

def value_iteration_anderson(
    C: Const,
    P_sparse: list[csr_matrix],
    q: np.ndarray,
    K_valid: int,
    # J_init: np.ndarray = None,
    tol: float = 1e-5,
    m: int = 10,           # History size of the Js
    reg: float = 1e-8,     # Regularization for stability, otherwise diverges
) -> tuple[np.ndarray, np.ndarray]:
    """
    Value Iteration with Anderson Acceleration.
    Args:
        C: problem constants
        P_sparse: list of length C.L, each element is a csr_matrix of shape (K, K)
                  with P_sparse[u][i, j] = P(i -> j | action u)
        Q: expected stage cost, shape (K, L)
        tol: tolerance for convergence
        m: history size for Anderson Acceleration
        reg: regularization parameter for stability in least-squares solve
    """
    L = C.L

    # --- Helper function for the Bellman Operator T(J) ---
    def bellman_operator(J_in: np.ndarray) -> np.ndarray:
        J_cands = np.empty((K_valid, L))
        for u in range(L):
            J_cands[:, u] = q[u] + P_sparse[u].dot(J_in)
        return np.min(J_cands, axis=1)
    # ---------------------------------------------------
    # J is the current estimate of the value function (vector of length K).
    # Using a (K, L) array here causes P_sparse[u].dot(J) to return (K, L),
    # which cannot be stored into a single column of `J_cands`.
    J = np.zeros(K_valid)
    #if J_init is None:
    #    J = np.empty((C.K, C.L)) # Initial guess
    #else:
    #    J = J_init.copy()  # Start from the provided initial guess

    # History buffers for AA
    J_hist = np.zeros((m, K_valid))  # Stores the last m J_k vectors
    G_hist = np.zeros((m, K_valid))  # Stores the last m g_k = T(J_k) - J_k vectors

    count = 0
    while True:
        count += 1

        # 1. Apply the Bellman operator
        T_J = bellman_operator(J)    
        # 2. Calculate the residual
        g = T_J - J
        # 3. Store in history (using a circular buffer)
        k = (count - 1) % m
        J_hist[k] = J
        G_hist[k] = g

        # --- Convergence Check ---
        # We check every step (or `check_every`)
        # We check the *true* residual max|T(J) - J|
        residual = np.max(np.abs(g))
        if residual < tol:
            print(f"Anderson VI converged in {count} iterations.")
            break
        
        # --- Anderson Acceleration Step ---
        if count < m:
            # While filling the buffer, just do a standard VI step
            J = T_J
        else:
            # History is full. Solve the least-squares problem.
            # We want to find alpha to minimize ||g_k + G_k @ alpha||^2
            # G_k = [g_{k-1}-g_k, ..., g_{k-m}-g_k]
            
            G_matrix = G_hist.T  # (K, m)
            GTG = G_matrix.T @ G_matrix   # (m, m)
            GTg = G_matrix.T @ g          # (m,)
            
            # Solve (GTG + reg*I) * alpha = -GTg
            try:
                alpha = np.linalg.solve(GTG + reg * np.eye(m), -GTg)
            except np.linalg.LinAlgError:
                # if the matrix is singular, just do a standard step
                J = T_J
                continue

            # 4. Compute the accelerated update
            T_J_hist = J_hist + G_hist  # History of T(J_i)
            T_J_diffs = T_J_hist - T_J  # (m, K)
            
            # J_next = T_J + (alpha @ T_J_diffs)
            J = T_J + (T_J_diffs.T @ alpha)

    # --- Extract Policy ---
    expected_value = np.column_stack([(P_sparse[u].dot(J)) for u in range(C.L)])
    J_Q = q + expected_value
    u_opt_ind = np.argmin(J_Q, axis=1)
    u_opt = np.array([C.input_space[ind] for ind in u_opt_ind])

    return (J, u_opt)

