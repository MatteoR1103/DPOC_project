"""Const.py
 
Python script containg the definition of the class Const
that holds all the problem constants.
 
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
 
from math import ceil
from itertools import product
from typing import List, Tuple
 
class Const:
    """Class containing all problem constants.
    
    Feel free to tweak these parameters to test your solution, BUT be careful!
    As described in the assignment, some combinations might lead to
    an invalid problem.
    """
    # ----- Grid & Physics -----
    X: int = 8                      # Number of columns (X) of the grid
    
    @property
    def S_d1(self) -> List[int]:
        """Set of admissible distances d1.
        
        Returns:
            List[int]: list of admissible distances
        """
        return list(range(self.X))
    
    @property
    def S_d(self) -> List[int]:
        """Set of admissible distances d (for d2,...,dM).
        
        Returns:
            List[int]: list of admissible distances
        """
        return [0] + list(range(self.D_min, self.X))
 
    Y: int = 14                     # Number of columns (Y) of the grid
    
    @property
    def S_y(self) -> List[int]:
        """Set of admissible vertical positions y.
        
        Returns:
            List[int]: list of admissible vertical positions
        """
        return list(range(self.Y))
 
    V_max: int = 2                  # Vertical velocity bound
    
    @property
    def S_v(self) -> List[int]:
        """Set of admissible vertical velocities v.
        
        Returns:
            List[int]: list of admissible vertical velocities
        """
        return list(range(-self.V_max, self.V_max + 1))
 
    U_no_flap: int = 0              # No upward impulse input
    U_weak: int = 2                 # Weak upward impulse input
    U_strong: int = 3               # Strong upward impulse input
 
    V_dev: int = 1                  # Strong flap altitude deviation:
                                    # w_flap in {-V_dev, ... , V_dev}
    
    @property
    def W_v(self) -> List[int]:
        """Set of admissible flap disturbances w_flap.
        
        Returns:
            List[int]: list of admissible flap disturbances
        """
        return list(range(-self.V_dev, self.V_dev + 1))
 
    D_min: int = 4                  # Minimum spacing between obstacles
                                    # (0 < D_min < X)
    G: int = 3                      # Gap size (must be odd)
    S_h: List[int] = [5, 9, 13]     # Set of admissible gap centers h_i
                                    # (h_i must be in {0, ..., Y-1})
 
    g: int = 1                      # Gravity (downwards accelaration per step)
 
    @property
    def M(self) -> int:
        """Maximum number of obstacles on the grid.
        
        Returns:
            int: maximum number of obstacles
        """
        return ceil(self.X / self.D_min)
    
    @property
    def state_space(self) -> List[Tuple[int, ...]]:
        """Returns the full state space as a list of tuples.
        
        Returns:
            List[Tuple[int, ...]]: list of admissible states
        """
        if not hasattr(self, '_state_space'):
            # Caching
            iterables = (
                [self.S_y, self.S_v, self.S_d1]
                + [self.S_d] * (self.M - 1)
                + [self.S_h] * self.M
            )
            self._state_space = [
                tuple(x) for x in product(*iterables) if self.is_valid_state(x)
            ]
        return self._state_space
 
    def state_to_index(self, x: Tuple[int, ...]) -> int:
        """Get index of state x = (y, v, d1, d2, ..., dM, h1, h2, ..., hM).
        
        Args:
            x: state tuple (y, v, d1, d2, ..., dM, h1, h2, ..., hM)
        
        Returns:
            int: index of state x in state_space
        """
        if not hasattr(self, '_state_indexing'):
            # Caching
            index: dict[Tuple[int, ...], int] = {}
            for idx, s in enumerate(self.state_space):
                index[s] = idx
            self._state_indexing = index
        if self.is_valid_state(x):
            return self._state_indexing[x]
        raise KeyError(f"[ERROR] state {x} does not exist in state_space.")
 
    @property
    def K(self) -> int:
        """Returns the size of the state space.
        
        Returns:
            int: size of state space
        """
        return len(self.state_space)
 
    # ----- Cost ------
    lam_weak: float = 0.5            # Cost factor on weak input effort
    lam_strong: float = 0.7          # Cost factor on strong input effort
 
    @property
    def input_space(self) -> List[int]:
        """Returns the full input space as a list.
        
        Returns:
            List[int]: list of admissible inputs
        """
        return [self.U_no_flap, self.U_weak, self.U_strong]
    
    @property
    def L(self) -> int:
        """Returns the size of the input space.
        
        Returns:
            int: size of input space
        """
        return len(self.input_space)
 
    def is_valid_state(self, x):
        """Checks if state s is valid for our statespace.
        
        Args:
            x: state tuple (y, v, d1, d2, ..., dM, h1, h2, ..., hM)
 
        Returns:
            bool: True if state is valid, False otherwise
        """
 
        D = list(x[2 : 2 + self.M])
        H = list(x[2 + self.M :])
        
        if sum(D) > self.X - 1:
            return False
 
        if D[1] == 0 and D[0] <= 0:
            return False
        
        if D[0] not in self.S_d1:
            return False
        
        if any(d not in self.S_d for d in D[1:]):
            return False
        
        if any(h not in self.S_h for h in H):
            return False
 
        if any(d == 0 and h != self.S_h[0] for d, h in zip(D[1:], H[1:])):
            return False
        
        # once a zero appears in d2, ..., dM, all later entries must be zero
        zero_seen = False
        for d in D[1:]:
            if zero_seen and d != 0:
                return False
            if d == 0:
                zero_seen = True
        return True