"""utils.py

Python script containg utility functions. Modify if needed,
but be careful as these functions are used, e.g., in simulation.py.

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

from Const import Const
import numpy as np

def spawn_probability(C: Const, s: int) -> float:
    """Distance-dependent spawn probability p_spawn(s).
    
    Args:
        C (Const): The constants describing the problem instance.
        s (int): Free distance, as defined in the assignment.

    Returns:
        float: The spawn probability p_spawn(s).
    """
    return max(min((s - C.D_min + 1) / (C.X - C.D_min), 1.0), 0.0)

def is_in_gap(C: Const, y: int, h1: int) -> bool:
    """Returns true if bird in gap.
    
    Args:
        C (Const): The constants describing the problem instance.
        y (int): Vertical position of the bird.
        h1 (int): Center of the gap of the first obstacle.

    Returns:
        bool: True if bird is in the gap, False otherwise.
    """
    half = (C.G - 1) // 2
    return abs(y - h1) <= half

def is_passing(C: Const, y: int, d1: int, h1: int) -> bool:
    """Return true if bird is currently passing the gap without colliding.
    
    Args:
        C (Const): The constants describing the problem instance.
        y (int): Vertical position of the bird.
        d1 (int): Distance to the first obstacle.
        h1 (int): Center of the gap of the first obstacle.

    Returns:
        bool: True if bird is passing the gap, False otherwise.
    """
    return (d1 == 0) and is_in_gap(C, y, h1)

def is_collision(C: Const, y: int, d1: int, h1: int) -> bool:
    """Return true if bird is colliding with obstacle.
    
    Args:
        C (Const): The constants describing the problem instance.
        y (int): Vertical position of the bird.
        d1 (int): Distance to the first obstacle.
        h1 (int): Center of the gap of the first obstacle.

    Returns:
        bool: True if bird is colliding with obstacle, False otherwise.
    """
    return (d1 == 0) and not is_in_gap(C, y, h1)


def compute_pose_dynamics(y_k, v_k, u_k, C: Const):
    """ Computes next state velocities and height
    Args:
        y_k : current height
        v_k : current velocity
    Returns:
        tuple(y_k1 (int), v_k1): next state height and all possible velocities
    """
    v_max = C.V_max
    g = C.g
    v_dev = C.V_dev
    flap_space_dim = 2*v_dev+1

    y_k1 = min(max(y_k + v_k, 0), C.Y -1)
    
    if u_k == C.U_no_flap or u_k == C.U_weak: 
        w_flap_k = np.zeros(flap_space_dim)
    else: 
        #w_k_flap takes different values distributed uniformly around [-V_dev, V_dev]
        
        w_flap_k = np.linspace(-v_dev, v_dev, flap_space_dim) #flap_space_dim

    v_k1 = np.min(np.max(v_k + u_k + w_flap_k - g, -v_max), v_max) #flap_space_dim. Each velocity has the same probability of being generated

    return y_k1, v_k1


def compute_obst_dynamics(d_k, h_k, y_k, C: Const):
    """ Computes obstacle dynamics: new obstacle distances and heights based on transition and spawn disturbance
        Args:
            d_k : list obstacle distances
            h_k : list obstacle heights
            C   : constants 
        Returns:
            tuple(d_k1, h_k1) new obstacle poses
    """
    #compute intermediate dynamics first, then factor in spawn disturbances
    #only used for valid states, otherwise probability is already computed as 0 

    #Intermediate variables. considering that collision states are already considered elsewhere

    d_int = []

    if is_passing(C, y=y_k, d1=d_k[0], h1=h_k[0]):
        for i in range(len(d_k)): 
            if i == 0:
                d_int.append(d_k[i+1]-1)    #the distance to the first becomes the distance from first to second -1
            elif i == C.M :                 
                d_int.append(0)             #dummy value waiting for spawn disturbance for the next dynamics
            else: 
                d_int.append(d_k[i+1])      #shift indices 
    else:                                   #considering we are already ruling out is_collision states normal drifting
        for i in range(len(d_k)): 
            if i == 0:
                d_int.append(d_k[i]-1)      #the distance to the first becomes the distance from first to second -1
            else: 
                d_int.append(d_k[i+1])
    
    #now that we have the intermediate dynamics, We can see how spawning works 
    s = C.X - 1 - sum(d_int)
    p_spawn = spawn_probability(C=C, s=s)

    