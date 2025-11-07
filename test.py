"""test.py

Python script containg some tests to check correctness of the implementation.
NOTE: these tests are not exhaustive. You can add more tests if you want.
Passing all tests here does not guarantee full correctness of your code.

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

import pickle
import numpy as np

from Const import Const
from ComputeTransitionProbabilities import compute_transition_probabilities
from ComputeExpectedStageCosts import compute_expected_stage_cost
from Solver import solution

RTOL = 1e-4
ATOL = 1e-7

def apply_overrides_and_instantiate(overrides: dict) -> Const:
    """Apply overrides to Const class and instantiate it.
    
    Args:
        overrides (dict): Dictionary with attribute names as keys and
            the values to override as values.
        
    Returns:
        Const: Instance of Const with overridden attributes.
    """
    for k, v in overrides.items():
        if hasattr(Const, k):
            setattr(Const, k, v)
    return Const()
    

def run_test(test_nr: int) -> None:
    """Run a single test case.
    
    Args:
        test_nr (int): Test case number.
    """
    print("-----------")
    print(f"Test {test_nr}")
    # Load constants overrides
    with open(f"tests/test{test_nr}.pkl", "rb") as f:
        overrides = pickle.load(f)
    C = apply_overrides_and_instantiate(overrides)

    # Load goldens
    gold = np.load(f"tests/test{test_nr}.npz")

    # Compute fresh
    P = compute_transition_probabilities(C)
    Q = compute_expected_stage_cost(C)
    
    passed = True
    
    # check all rows in [0,1] (within tolerance)
    row_sums = P.sum(axis=1)  # (K, L)
    eps = 1e-10
    if not (np.all(row_sums <= 1 + eps) and np.all(row_sums >= -eps)):
        print("[ERROR] Some probability row sums are outside [0,1].")
        passed = False
    else:
        mins = row_sums.min(axis=0)
        maxs = row_sums.max(axis=0)
        print(f"Row-sum min/max per action: min {mins}, max {maxs}")

    if not np.allclose(P, gold["P"], rtol=RTOL, atol=ATOL):
        print("Wrong transition probabilities")
        passed = False
    else:
        print("Correct transition probabilities")

    if not np.allclose(Q, gold["Q"], rtol=RTOL, atol=ATOL):
        print("Wrong expected stage costs")
        passed = False
    else:
        print("Correct expected stage costs")

    J_opt, u_opt = solution(C)
    if not np.allclose(J_opt, gold["J"], rtol=RTOL, atol=ATOL):
        print("Wrong optimal cost")
        passed = False
    else:
        print("Correct optimal cost")

    if "u" in gold.files:
        if not np.array_equal(u_opt, gold["u"]):
            print("Policy differs from golden (may be OK if ties exist)")
        else:
            print("Policy matches golden")

    print("Result:", "PASSED" if passed else "FAILED")

def main() -> None:
    """Main function to run all tests."""
    n_tests = 4
    for test_nr in range(n_tests):
        run_test(test_nr)
    print("-----------")

if __name__ == "__main__":
    main()
