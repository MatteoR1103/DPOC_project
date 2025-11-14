"""ComputeExpectedStageCosts.py

Template to compute the expected stage cost.

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

def compute_expected_stage_cost(C: Const) -> np.array:
    """Computes the expected stage cost for the given problem.

    Args:
        C (Const): The constants describing the problem instance.

    Returns:
        np.array: Expected stage cost Q of shape (K,L), where
            - K is the size of the state space;
            - L is the size of the input space; and
            - Q[i,l] corresponds to the expected stage cost incurred when using input l at state i.
    """
    Q = np.ones((C.K, C.L)) * np.inf
    half = (C.G - 1) // 2

    state_space = np.array(C.state_space, dtype=np.int64)

    y_k = np.array([s[0] for s in state_space]) #(K,)
    d_k = np.array([s[2 : 2 + C.M] for s in state_space]) #(K,M) 
    h_k = np.array([s[2 + C.M : ] for s in state_space])  #(K,M)
    
    not_in_gap = abs(y_k - h_k[:,0]) > half
    in_col_1 = d_k[:,0] == 0   #mask selcting where we have an obst in 1st column

    is_colliding_mask = not_in_gap & in_col_1 #(K,) array of booleans indicating whether it's colliding or not
    valid = np.logical_not(is_colliding_mask)


    # TODO fill the expected stage cost Q here
    Q[:, 0] = -1
    Q[:, 1] = C.lam_weak-1
    Q[:, 2] = C.lam_strong-1
    return Q