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

def value_iteration(C: Const, P_sparse: list[csr_matrix], Q: np.ndarray, J_init = None, tol: float = 1e-5, iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    
    if J_init is None:
        J = np.zeros(C.K) # Normal INIT
    else:
        J = J_init.copy()  # Start from the provided initial guess
    
    count = 0
    while True:
        # Bellman backup: for each (i,u)
        J_new = np.empty_like(J)
        count += 1

        for u in range(C.L):
            J_new_u = Q[:, u] + P_sparse[u].dot(J)
            if u == 0:
                J_new = J_new_u
            else:
                J_new = np.minimum(J_new, J_new_u)

        if (count % iters) == 0:
            if np.max(np.abs(J_new - J)) < tol:
                print(f"Pure VI converged in {count} iterations.")
                break

        J = J_new

        # Extract policy corresponding to final J

    expected_value = np.column_stack([(P_sparse[u].dot(J))for u in range(C.L)])
    J_Q = Q + expected_value
    u_opt_ind = np.argmin(J_Q, axis=1)
    u_opt = np.array([C.input_space[ind] for ind in u_opt_ind])

    return J, u_opt

# The idea of initializing VI using PI might still be valid, keep it in mind.
def value_iteration_in_place(C: Const, P_sparse: list[csr_matrix], Q: np.ndarray, J_init = None, tol: float = 1e-5, iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    if J_init is None:
        J = np.zeros(C.K) # Normal INIT
    else:
        J = J_init.copy()  # Start from the provided initial guess
    
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
                print(f"Optimized VI converged in {count} iterations.")
                break
    
    # Extract policy corresponding to final J
    expected_value = np.column_stack([(P_sparse[u].dot(J))for u in range(C.L)])
    J_Q = Q + expected_value
    u_opt_ind = np.argmin(J_Q, axis=1)
    u_opt = np.array([C.input_space[ind] for ind in u_opt_ind])

    print(f"Count: {count}")
    return (J, u_opt) 


def value_iteration_anderson(
    C: Const,
    P_sparse: list[csr_matrix],
    Q: np.ndarray,
    J_init: np.ndarray = None,
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
    K, L = C.K, C.L
    
    # --- Helper function for the Bellman Operator T(J) ---
    def bellman_operator(J_in: np.ndarray) -> np.ndarray:
        J_cands = np.empty((K, L))
        for u in range(L):
            J_cands[:, u] = Q[:, u] + P_sparse[u].dot(J_in)
        return np.min(J_cands, axis=1)
    # ---------------------------------------------------
    if J_init is None:
        J = np.empty((C.K, C.L)) # Initial guess
    else:
        J = J_init.copy()  # Start from the provided initial guess

    # History buffers for AA
    J_hist = np.zeros((m, K))  # Stores the last m J_k vectors
    G_hist = np.zeros((m, K))  # Stores the last m g_k = T(J_k) - J_k vectors

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
    J_Q = Q + expected_value
    u_opt_ind = np.argmin(J_Q, axis=1)
    u_opt = np.array([C.input_space[ind] for ind in u_opt_ind])

    return (J, u_opt)


def value_iteration_accelerated(C: Const, P_sparse: list[csr_matrix], Q,
                                tol: float = 1e-5,
                                check_every: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Your fast, action-asynchronous VI, now with Nesterov Acceleration.
    """
    K, L = C.K, C.L
    
    # 1. Start with a better J_init.
    # This is a one-step lookahead cost, much better than zeros.
    J = np.min(Q, axis=1)
    
    J_last = J.copy()
    w = 1.0  # Nesterov momentum weight
    count = 0
    
    while True:
        count += 1
        
        # --- Nesterov Momentum Step ---
        # 1. Calculate momentum weight
        w_last = w
        w = (1.0 + np.sqrt(1.0 + 4.0 * w_last**2)) / 2.0
        momentum = (w_last - 1.0) / w
        
        # 2. Extrapolate J to a new point Y
        # This is the "smarter" point to start the update from.
        Y = J + momentum * (J - J_last)
        
        # 3. Save current J to be J_last for the *next* iteration
        J_last = J.copy()
        # -------------------------------

        # --- Your fast update loop, but starting from Y ---
        # We compute J = T(Y)
        J_candidates = np.empty((K, L))
        for u in range(L):
            J_candidates[:, u] = Q[:, u] + P_sparse[u].dot(Y)
        
        J = np.min(J_candidates, axis=1)
        # -------------------------------
        
        # --- Parametrized Convergence Check ---
        # We check every N iterations for speed.
        # We must check against J_last (the *true* previous J), not Y.
        if (count % check_every) == 0:
            residual = np.max(np.abs(J_last - J))
            
            if residual < tol:
                print(f"Accelerated VI converged in {count} iterations.")
                break

    # Extract policy
    expected_value = np.column_stack([(P_sparse[u].dot(J)) for u in range(L)])
    J_Q = Q + expected_value
    u_opt_ind = np.argmin(J_Q, axis=1)
    u_opt = np.array([C.input_space[ind] for ind in u_opt_ind])

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
    start = time.time()
    Q = compute_expected_stage_cost(C)        # (K, L)
    P_sparse = []
    # PRECOMPUTATIONS: build state-space encodings and masks
    (
        _,
        state_index_map,
        stride,
        valid_indices,
        y_k_valid,
        v_k_valid,
        d_k_valid,
        h_k_valid,
    ) = build_state_space(C)

    # Compute dynamics (heights, velocities, obstacles, spawn probabilities)
    (
        y_k1_int,
        v_k1_int,
        d_k1_int,
        h_k1_int,
        p_spawn,
    ) = build_dynamics(C, y_k_valid, v_k_valid, d_k_valid, h_k_valid)
   
    # Initialize J_init using one-step lookahead costs for valid states
    J_init = np.zeros(C.K)
    J_1_step_costs = np.min(Q, axis=1)
    # Apply these costs *only* to the "valid" (alive) states.
    J_init[valid_indices] = J_1_step_costs[valid_indices]

    # Build sparse transition matrices (one CSR per action)
    P_sparse = build_P_sparse(
        C,
        y_k1_int=y_k1_int,
        v_k1_int=v_k1_int,
        d_k1_int=d_k1_int,
        h_k1_int=h_k1_int,
        state_index_map=state_index_map,
        stride=stride,
        valid_indices=valid_indices,
        p_spawn=p_spawn,
    )
    end = time.time()

    print(f"Time for precomps: {end-start}")
    (J, u_opt) = value_iteration(C, P_sparse, Q, tol=1e-5, J_init = J_init)
    return (J, u_opt)

# HELPER FUNCTIONS BELOW
def build_state_space(C: Const):
    """Build state-space arrays and indexing helpers.

    Returns a tuple:
      (input_space, state_space, state_index_map, stride,
       valid_indices, y_k_valid, v_k_valid, d_k_valid, h_k_valid)
    """
    # input_space = C.input_space
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
    stride = np.cumprod([1] + dims[:-1])

    encoded = (state_space_for_keys * stride).sum(axis=1)
    max_key = int(encoded.max())
    state_index_map = -np.ones(max_key + 1, dtype=np.int32)
    state_index_map[encoded] = np.arange(C.K)

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

    return (
        #input_space,
        state_space,
        state_index_map,
        stride,
        valid_indices,
        y_k_valid,
        v_k_valid,
        d_k_valid,
        h_k_valid,
    )

def build_P_sparse(
    C: Const,
    y_k1_int: np.ndarray,
    v_k1_int: np.ndarray,
    d_k1_int: np.ndarray,
    h_k1_int: np.ndarray,
    state_index_map: np.ndarray,
    stride: np.ndarray,
    valid_indices: np.ndarray,
    p_spawn: np.ndarray,
) -> list:
    """Build list of CSR transition matrices (one per action).

    This extracts the block that was previously inline in `solution`.
    Returns: list of length C.L with CSR matrices shape (C.K, C.K).
    """
    P_sparse = []
    current_states = valid_indices
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
                    encoded_next = (tuples * stride).sum(axis=1)
                    indices = state_index_map[encoded_next]
                    all_cstates.append(current_states)
                    all_nstates.append(indices)
                    all_probs.append(1 - p_spawn)
                else:  # spawn
                    for h in range(len(C.S_h)):
                        tuples = np.column_stack((
                            y_k1_int,
                            v_k1_int[:, u, 0] + C.V_max,
                            d_k1_int[:, :, 1],
                            h_k1_int[:, :, h, 1],
                        ))
                        encoded_next = (tuples * stride).sum(axis=1)
                        indices = state_index_map[encoded_next]
                        all_cstates.append(current_states)
                        all_nstates.append(indices)
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
                        encoded_next = (tuples * stride).sum(axis=1)
                        indices = state_index_map[encoded_next]
                        all_cstates.append(current_states)
                        all_nstates.append(indices)
                        all_probs.append((1 / flap_space_dim) * (1 - p_spawn))
                else:  # spawn
                    for v in range(flap_space_dim):
                        for h in range(len(C.S_h)):
                            tuples = np.column_stack((
                                y_k1_int,
                                v_k1_int[:, u, v] + C.V_max,
                                d_k1_int[:, :, 1],
                                h_k1_int[:, :, h, 1],
                            ))
                            encoded_next = (tuples * stride).sum(axis=1)
                            indices = state_index_map[encoded_next]
                            all_cstates.append(current_states)
                            all_nstates.append(indices)
                            all_probs.append((1 / flap_space_dim) * (1 / w_h_dim) * (p_spawn))

        all_cstates = np.concatenate(all_cstates)
        all_nstates = np.concatenate(all_nstates)
        all_probs = np.concatenate(all_probs)

        P_sparse.append(coo_matrix((all_probs, (all_cstates, all_nstates)), shape=(C.K, C.K)).tocsr())

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
