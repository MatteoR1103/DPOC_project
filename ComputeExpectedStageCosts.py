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

    # TODO fill the expected stage cost Q here

    return Q