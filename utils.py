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

def spawn_probability_vec(C: Const, s) -> float:
    """Distance-dependent spawn probability p_spawn(s).
    
    Args:
        C (Const): The constants describing the problem instance.
        s np.ndarray (K,): Free distances, as defined in the assignment.

    Returns:
        np.ndarray(K,) float: The spawn probabilities p_spawn(s).
    """
    return np.clip(((s - C.D_min + 1) / (C.X - C.D_min)), 0.0, 1.0)

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

def compute_height_dynamics(y_k, v_k, C: Const): 
    """
    Compute the next-state altitudes (y_{k+1}) deterministically.

    Args:
        y_k (np.ndarray): Array of shape (K,) containing current altitudes.
        v_k (np.ndarray): Array of shape (K,) containing current vertical velocities.
        C (Const): Problem constants 

    Returns:
        np.ndarray: Array of shape (K,) containing next-state altitudes,
                    clipped to remain within [0, C.Y - 1].
    """
    return np.clip((y_k + v_k), 0, C.Y -1)#.astype(int)

def compute_vel_dynamics(v_k, input_space, v_dev_inter, C: Const):
    """
    Compute the next-state velocity array indexed by state, input, and flap disturbance.

    Args:
        v_k (np.ndarray): Array of shape (K,) containing current velocities.
        input_space (np.ndarray): Array of shape (L,) with discrete input values
                                  [U_no_flap, U_weak, U_strong].
        v_dev_inter (np.ndarray): Array of shape (2 * C.V_dev + 1,)
                                  containing the sampled flap disturbances.
        C (Const): Problem constants (fields: g, V_max, V_dev, etc.).

    Returns:
        np.ndarray: Array of shape (K, L, 2 * C.V_dev + 1) containing all possible
                    next velocities for each state, input, and flap disturbance.
    """
     
    v_max = C.V_max
    g = C.g
    flap_space_dim = v_dev_inter.shape[0]
    #The next velocities are deterministic if the input is no_flap or weak: K-dimensional arrays
    v_next_no_flap = np.clip((v_k + input_space[0] - g), -v_max, v_max)#.astype(int)
    v_next_weak = np.clip((v_k + input_space[1] - g), -v_max, v_max)#.astype(int)

    #augment v_k and v_dev_inter for broadcasting: (Kxflap_space_dim)-dimensional arrays
    # Each velocity has the same probability of being generated
    v_next_strong = np.clip((v_k[:, None]+ input_space[2] + v_dev_inter[None, :] - g), -v_max, v_max)#.astype(int)

    K_valid = v_k.shape[0]

    next_v = np.empty((K_valid, C.L, flap_space_dim))
    next_v[:, 0, :] = v_next_no_flap[:, None]
    next_v[:, 1, :] = v_next_weak[:, None]
    next_v[:, 2, :] = v_next_strong

    return next_v

def compute_obst_dynamics(d_k, h_k, y_k, C: Const):
    """
        Compute obstacle dynamics (new distances and heights) based on current state and spawn disturbances.

        Args:
            d_k (np.ndarray): Array of shape (K, M) containing all obstacles' distances.
            h_k (np.ndarray): Array of shape (K, M) containing all obstacles' heights.
            y_k (np.ndarray): Array of shape (K,) containing all bird altitudes.
            C (Const): Constant parameters 

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - d_next: np.ndarray of shape (K, M, 2)
                Updated obstacle distances for no-spawn and spawn cases.
                - h_next: np.ndarray of shape (K, M, len(C.S_h), 2)
                Updated obstacle heights for all possible spawn heights and spawn/no-spawn cases.
                - p_spawn: np.ndarray of shape (K,)
                Obstacle spawn probability for each state.
    """
    
    #compute intermediate dynamics first, then factor in spawn disturbances
    #only used for valid states, otherwise probability is already computed as 0 

    #Intermediate variables. considering that collision states are already considered elsewhere
    K_valid = d_k.shape[0]
    half = (C.G - 1) // 2 
    h_d = C.S_h[0]

    in_gap = abs(y_k - h_k[:,0]) <= half
    in_col_1 = d_k[:,0] == 0   #mask selcting where we have an obst in 1st column

    is_passing_mask = in_gap & in_col_1 #(K,) array of booleans indicating whether it's passing or not
    is_drifting_mask = np.logical_not(is_passing_mask)#(K,) array of booleans indicating whether it's drifting or not

    #compute dynamics for passing scenario
    passing_obst_d = d_k[is_passing_mask, :] #(is_passing, M) array
    d_int_passing = np.empty((passing_obst_d.shape))
    
    d_int_passing[:, 0] = passing_obst_d[:, 1] - 1 #the distance to the first becomes the distance from first to second -1
    d_int_passing[:,1:(C.M-1)] = passing_obst_d[:, 2:] #shift indices 
    d_int_passing[:, -1] = 0                       #dummy value waiting for spawn disturbance for the next dynamics

    passing_obst_h = h_k[is_passing_mask, :] #(is_passing, M) array
    h_int_passing = np.empty((passing_obst_h.shape))
    
    h_int_passing[:, 0:(C.M-1)] = passing_obst_h[:, 1:] #shift indices 
    h_int_passing[:, -1] = h_d                      #dummy value waiting for spawn disturbance for the next dynamics

    #compute dynamics for normal drifting scenario
    drifting_obst_d = d_k[is_drifting_mask, :]
    d_int_drifting = np.empty((drifting_obst_d.shape))
    
    d_int_drifting[:, 0] = drifting_obst_d[:, 0] - 1 #the distance to the first becomes the distance from first - 1
    d_int_drifting[:, 1:] = drifting_obst_d[:, 1:]   #others remain unchanged 

    h_int_drifting = h_k[is_drifting_mask, :]           #(is_passing, M) array

    #put everything back together into a (K,M) array 
    d_int = np.empty((K_valid, C.M))
    d_int[is_passing_mask, :] = d_int_passing
    d_int[is_drifting_mask, :] = d_int_drifting
    
    h_int = np.empty((K_valid, C.M))
    h_int[is_passing_mask, :] = h_int_passing
    h_int[is_drifting_mask, :] = h_int_drifting

    #now that we have the intermediate dynamics, We can see how spawning works
    s = C.X - 1 - np.sum(d_int, axis=1)  #(K_valid, )

    p_spawn = spawn_probability_vec(C, s)

    # compute next state obstacles. for obstacles' distances it's either the same as the intermediate, or = s if it has spawned
    # create all possible outcomes, and probability will be taken into account later on in the ComputeTransitionProbabilities

    d_next_no_spawn = d_int.copy()
    h_next_no_spawn = h_int.copy()

    d_next_spawn = d_int.copy()
    h_next_spawn = h_int.copy()
    
    #CREATE A DUMMY STATE FOR P_SPAWN = 1 CASES SO AS NOT TO CREATE MAPPING KeyError 
    mask_spawn1 = np.isclose(p_spawn, 1.0, atol=1e-8)
    d_next_no_spawn[mask_spawn1, :] = d_k[0, :] 
    h_next_no_spawn[mask_spawn1, :] = h_k[0, :] 

    mask_mmin = (d_int[:, 1:] == 0) #(K_valid,M-1) mask
    mask_smin = (s >= C.D_min)
    has_zero = np.any(mask_mmin, axis=1) #finds if the row K has a 0 (condition of the mask = True,1) or not 

    has_both = has_zero & mask_smin
    
    #we need to pick the first TRUE (what argmax does since true=1) only if has_zero is true, otherwise it's 0 because not found
    #If not found, set as C.M as in the statement. Need +1 because we found the mask as (K_valid, M-1)
    mmin = np.where(has_zero, 1 + np.argmax(mask_mmin, axis=1), C.M-1) #(K_valid,) indices of where the first zero is
    
    #we need now to index for all the rows the column in which mmin was found
    #np.arange(K_valid) pairs each mmin with the corresponding row
    rows = np.arange(K_valid)
    d_next_spawn[rows[has_both], mmin[has_both]] = s[has_both]

    #create an array to contain the possible heights in case of spawn 
    w_k_h = np.array(C.S_h)
    #stack next distances into a single array of dimensions (K,M,2)
    d_next = np.stack((d_next_no_spawn, d_next_spawn), axis = 2)

    #create result array indexed by state, number of obstacles, possible spawn heights, spawn/no_spawn cases
    h_next = np.empty((K_valid, C.M, len(C.S_h), 2))
    h_next[:, :, :, 0] = h_next_no_spawn[:, :, None]
    h_next[:, :, :, 1] = h_next_spawn[:, :, None]
    h_next[rows[has_both], mmin[has_both], :, 1] = w_k_h[None, :]      #every row gets filled with the possible heights for each mmin 

    return d_next, h_next, p_spawn




    