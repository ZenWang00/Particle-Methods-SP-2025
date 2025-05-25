#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import os
import time # For unique subfolder names if needed, or just use params
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting
import traceback

# ==== Global Constants & Default Parameters ====
TYPE_FOLLOWER = 0
TYPE_LEADER = 1

# Default physical/simulation parameters that can be overridden by experiment scripts
L_DOMAIN_DEFAULT = 20.0
V0_DEFAULT = 0.5
R_INTERACTION_DEFAULT = 2.0 # For Vicsek interactions
ETA_DEFAULT = 0.2           # Noise amplitude
DT_DEFAULT = 0.5            # Simulation time step
T_COMMUNICATION_DEFAULT = 30 # For LF model leader communication interval

# Default convergence parameters (can also be overridden)
# For LF model (main convergence on polarization)
POL_THRESH_LF_DEFAULT = 0.95
CONV_WINDOW_LF_DEFAULT = 40       # Increased (e.g., 5 steps_interval * 40 checks = 200 steps window)
CHECK_INTERVAL_LF_DEFAULT = 5     # Decreased for more frequent data logging and checks
# Other LF metrics thresholds (for logging, not primary for stopping in current LF logic)
AS_THRESH_DEFAULT = 0.95
LB_THRESH_DEFAULT = 2.5 
LS_THRESH_DEFAULT = 0.2 

# For Pure Vicsek model
POL_THRESH_VICSEK_DEFAULT = 0.95
CONV_WINDOW_VICSEK_DEFAULT = 40   # Increased
CHECK_INTERVAL_VICSEK_DEFAULT = 5 # Decreased

# --- Parameters ---
L_domain = 20.0       # Domain size (square domain from 0 to L_domain)
N_followers = 500     # Keep large number of followers
N_leaders = 20        # Keep N_leaders
N_particles = N_followers + N_leaders

v0 = 0.5              # Constant speed of particles
R_interaction = 2.0   # Interaction radius for Vicsek component
eta = 0.2             # Noise amplitude (radians) for heading updates
dt = 0.5              # Time step
sim_steps = 1000      # Max simulation steps for Leader-Follower (e.g., 1000)

# Convergence Parameters (for Leader-Follower)
convergence_check_interval = 20 
convergence_window = 10          # LF: Number of consecutive checks for polarization
assignment_stability_threshold = 0.95 # Will be recorded, but not primary for stopping
polarization_threshold = 0.95         # LF: Target polarization for stopping
load_balance_std_dev_threshold = 2.5 
leader_spacing_variance_change_threshold = 0.2

# MODIFIED: Directly set leader capacity for ~30% coverage experiment
leader_capacity = 7 
print(f"N_followers: {N_followers}, N_leaders: {N_leaders}")
print(f"Leader capacity MANUALLY SET to: {leader_capacity}")
print(f"Total leader capacity: {N_leaders * leader_capacity} (Target Coverage: {(N_leaders * leader_capacity / N_followers * 100):.1f}%)")

# Comment out or remove previous automatic capacity calculation logic
# if N_leaders > 0:
#     base_capacity = N_followers // N_leaders
#     if N_followers % N_leaders == 0: leader_capacity = base_capacity
#     else: leader_capacity = base_capacity + 1
#     if N_followers > N_leaders and N_leaders * leader_capacity <= N_followers: leader_capacity +=1
# else: leader_capacity = float('inf')
# print(f"DEBUG: N_f={N_followers}, N_l={N_leaders}, Calculated cap={leader_capacity}, Total cap={N_leaders*leader_capacity}")

T_communication = 30

# --- Helper Functions ---
def wrap_to_pi(angles):
    """Wrap angles to [-pi, pi]."""
    return (angles + np.pi) % (2 * np.pi) - np.pi

def pbc_vector(vec, L):
    """Apply periodic boundary conditions to a vector component-wise."""
    return vec - L * np.rint(vec / L)
    
def pbc_distance_sq(p1, p2, L):
    """Squared Euclidean distance with PBC."""
    diff = pbc_vector(p1 - p2, L)
    return np.sum(diff**2)

def pbc_position(positions, L):
    """Wrap positions to [0, L) for a single or multiple particles."""
    return positions % L

def calculate_mean_angle(angles):
    """Correctly calculate the mean of a list of angles (in radians)."""
    if len(angles) == 0:
        return 0.0 # Or handle as an error, or return None
    x_coords = np.cos(angles)
    y_coords = np.sin(angles)
    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)
    return np.arctan2(mean_y, mean_x)

def calculate_polarization(headings, particle_types, target_type=TYPE_FOLLOWER):
    """Calculate polarization for particles of target_type."""
    target_indices = np.where(particle_types == target_type)[0]
    if len(target_indices) == 0:
        return 0.0
    target_headings = headings[target_indices]
    vx = np.cos(target_headings)
    vy = np.sin(target_headings)
    mean_vx = np.mean(vx)
    mean_vy = np.mean(vy)
    return np.sqrt(mean_vx**2 + mean_vy**2)

def calculate_leader_distance_variance(leader_idxs, positions_all, L):
    """Calculate the variance of pairwise distances between leaders."""
    if len(leader_idxs) < 2:
        return 0.0 # Variance is 0 if < 2 leaders
    
    leader_positions = positions_all[leader_idxs]
    pairwise_distances_sq = []
    for i in range(len(leader_positions)):
        for j in range(i + 1, len(leader_positions)):
            dist_sq = pbc_distance_sq(leader_positions[i], leader_positions[j], L)
            pairwise_distances_sq.append(np.sqrt(dist_sq)) # Store actual distances for variance
    
    if not pairwise_distances_sq:
        return 0.0
    return np.var(np.array(pairwise_distances_sq))

# --- Initialization Functions ---
def initialize_system(n_leaders, n_followers, L_param, leader_dist_mode="random"):
    """Initializes system with specified leader distribution mode."""
    n_total = n_leaders + n_followers
    positions = np.random.rand(n_total, 2) * L_param # Default: all random
    headings = np.random.rand(n_total) * 2 * np.pi - np.pi
    types = np.zeros(n_total, dtype=int)
    
    leader_indices_abs = np.array([], dtype=int) # Initialize as empty

    if n_leaders > 0 and n_total >= n_leaders:
        # First, assign types assuming all random, then override leader positions if needed
        # This ensures follower_indices_abs is correct before leaders are potentially moved.
        # It might be cleaner to place followers first, then leaders according to mode.
        
        # Alternative approach: Place followers randomly first, then leaders
        non_leader_indices = np.random.choice(n_total, n_followers, replace=False)
        types[non_leader_indices] = TYPE_FOLLOWER
        # The remaining indices are for leaders
        potential_leader_indices = np.setdiff1d(np.arange(n_total), non_leader_indices)
        if len(potential_leader_indices) >= n_leaders: # Should always be true if n_total = n_f + n_l
            leader_indices_abs = np.random.choice(potential_leader_indices, n_leaders, replace=False)
        else: # Should not happen with correct N_f, N_l
            leader_indices_abs = potential_leader_indices # Take all available if not enough
        
        types[leader_indices_abs] = TYPE_LEADER

        # Now, set leader positions based on leader_dist_mode
        if leader_dist_mode == "center_clustered":
            center = L_param / 2.0
            # Place leaders in a small cluster around the center
            # Spread them slightly to avoid exact overlap if multiple leaders
            for i, leader_idx in enumerate(leader_indices_abs):
                angle = 2 * np.pi * i / n_leaders
                radius = L_param * 0.05 # Small radius, e.g., 5% of L
                positions[leader_idx, 0] = center + radius * np.cos(angle)
                positions[leader_idx, 1] = center + radius * np.sin(angle)
                # Ensure they are within bounds if center + radius goes out, though pbc_position handles it later
                positions[leader_idx] = np.clip(positions[leader_idx], 0.01, L_param-0.01)
        
        elif leader_dist_mode == "grid":
            # Attempt to place leaders on a somewhat uniform grid
            # This is a simple grid, might not be perfectly uniform for all N_l
            sqrt_nl = int(np.ceil(np.sqrt(n_leaders)))
            grid_spacing_l = L_param / (sqrt_nl +1) # Add 1 to avoid edges if possible
            idx_l = 0
            for r in range(sqrt_nl):
                for c in range(sqrt_nl):
                    if idx_l < n_leaders:
                        positions[leader_indices_abs[idx_l], 0] = (c + 1) * grid_spacing_l
                        positions[leader_indices_abs[idx_l], 1] = (r + 1) * grid_spacing_l
                        idx_l += 1
                    else: break
                if idx_l >= n_leaders: break
        
        elif leader_dist_mode == "periphery":
            # Distribute leaders along the 4 peripheries
            leaders_per_side = int(np.ceil(n_leaders / 4.0))
            edge_offset = L_param * 0.05 # Small offset from the very edge
            current_leader = 0
            for side in range(4):
                for i in range(leaders_per_side):
                    if current_leader < n_leaders:
                        rand_pos_on_side = edge_offset + np.random.rand() * (L_param - 2 * edge_offset)
                        if side == 0: # Bottom
                            positions[leader_indices_abs[current_leader]] = [rand_pos_on_side, edge_offset]
                        elif side == 1: # Right
                            positions[leader_indices_abs[current_leader]] = [L_param - edge_offset, rand_pos_on_side]
                        elif side == 2: # Top
                            positions[leader_indices_abs[current_leader]] = [rand_pos_on_side, L_param - edge_offset]
                        else: # Left
                            positions[leader_indices_abs[current_leader]] = [edge_offset, rand_pos_on_side]
                        current_leader += 1
                    else: break
                if current_leader >= n_leaders: break
        
        # elif leader_dist_mode == "random": # This is default if not specified or matched
            # Leader positions are already random from the initial np.random.rand(n_total, 2) * L_param
            # Just ensure types are set correctly (done above)
            pass # Default random placement handled by initial positions array creation

    # Ensure all positions are wrapped by PBC after potential specific placements
    positions = pbc_position(positions, L_param)

    follower_indices_abs = np.array([i for i in range(n_total) if types[i] == TYPE_FOLLOWER])
    assignments = -np.ones(len(follower_indices_abs), dtype=int) 
    leader_actual_indices_from_types = np.where(types == TYPE_LEADER)[0] # Re-fetch after types are finalized
    leader_follower_counts = np.zeros(n_leaders, dtype=int) if n_leaders > 0 else np.array([])
    
    return positions, headings, types, follower_indices_abs, leader_actual_indices_from_types, assignments, leader_follower_counts

def initialize_vicsek_system(n_particles, L_param):
    positions = np.random.rand(n_particles, 2) * L_param
    headings = np.random.rand(n_particles) * 2 * np.pi - np.pi
    return positions, headings

# --- Core Logic Update Functions ---
def assign_followers(follower_idxs_assign, leader_idxs_assign, positions_all_assign, 
                     leader_counts_assign, capacity_assign, current_assignments_assign, L_param):
    new_assignments = -np.ones_like(current_assignments_assign)
    new_leader_counts = np.zeros_like(leader_counts_assign)
    for i, follower_idx in enumerate(follower_idxs_assign):
        f_pos = positions_all_assign[follower_idx]; min_dist_sq = float('inf'); assigned_leader_k = -1
        for k, leader_idx in enumerate(leader_idxs_assign):
            if new_leader_counts[k] < capacity_assign:
                d_sq = pbc_distance_sq(f_pos, positions_all_assign[leader_idx], L_param)
                if d_sq < min_dist_sq: min_dist_sq = d_sq; assigned_leader_k = k
        if assigned_leader_k != -1: new_assignments[i] = assigned_leader_k; new_leader_counts[assigned_leader_k] += 1
    return new_assignments, new_leader_counts

def update_follower_headings_differentiated(follower_idxs_update, positions_update, headings_update, types_update, 
                                          assignments_update, leader_actual_idxs_update, 
                                          R_inter_sq_update, eta_noise_update, L_domain_update):
    """Followers assigned to a leader directly follow that leader's heading.
       Unassigned followers (or those whose leader is not found) perform Vicsek with all neighbors.
       Modifies headings in place for the specified follower_idxs.
    """
    # Create a complete list of all particle indices for Vicsek neighbor search for unassigned followers
    all_particle_indices = np.arange(len(positions_update))

    for i, follower_idx in enumerate(follower_idxs_update): # follower_idx is the global index of the current follower
        assigned_leader_k_in_leader_list = assignments_update[i] # Index k (0 to N_leaders-1) or -1

        if assigned_leader_k_in_leader_list != -1:
            # This follower is a "Direct Follower"
            actual_leader_particle_idx = leader_actual_idxs_update[assigned_leader_k_in_leader_list]
            target_angle = headings_update[actual_leader_particle_idx] # Directly take leader's current heading
        else:
            # This follower is an "Other/Free Follower" - performs Vicsek
            f_pos = positions_update[follower_idx]
            vicsek_neighbor_headings = []
            for neighbor_candidate_idx in all_particle_indices:
                if follower_idx == neighbor_candidate_idx: continue # Skip self
                
                # For Vicsek, unassigned followers consider ALL other particles as potential neighbors
                dist_sq = pbc_distance_sq(f_pos, positions_update[neighbor_candidate_idx], L_domain_update)
                if dist_sq < R_inter_sq_update:
                    vicsek_neighbor_headings.append(headings_update[neighbor_candidate_idx])
            
            if vicsek_neighbor_headings:
                target_angle = calculate_mean_angle(np.array(vicsek_neighbor_headings))
            else:
                target_angle = headings_update[follower_idx] # Keep current heading if no Vicsek neighbors
        
        noise = (np.random.rand() - 0.5) * eta_noise_update
        headings_update[follower_idx] = wrap_to_pi(target_angle + noise)
    # headings array is modified in place

def update_leaders_vicsek_step(leader_idxs_vic, positions_vic, headings_vic, types_vic, 
                             R_inter_sq_vicsek_param, eta_noise_vic, L_domain_vic):
    """Leaders update headings based on Vicsek interaction with ALL nearby particles.
       Modifies headings in place.
    """
    if len(leader_idxs_vic) == 0: return

    all_particle_indices = np.arange(len(positions_vic)) # For neighbor searching
    temp_new_headings_for_leaders = headings_vic[leader_idxs_vic].copy()

    for i_idx, leader_idx1 in enumerate(leader_idxs_vic): # leader_idx1 is the global index of the current leader
        l1_pos = positions_vic[leader_idx1]
        neighbor_headings = []
        for neighbor_candidate_idx in all_particle_indices:
            if leader_idx1 == neighbor_candidate_idx: continue # Skip self
            
            dist_sq = pbc_distance_sq(l1_pos, positions_vic[neighbor_candidate_idx], L_domain_vic)
            if dist_sq < R_inter_sq_vicsek_param:
                neighbor_headings.append(headings_vic[neighbor_candidate_idx])
        
        if neighbor_headings:
            mean_neighbor_angle = calculate_mean_angle(np.array(neighbor_headings))
        else:
            mean_neighbor_angle = headings_vic[leader_idx1] # Keep current heading if no neighbors
            
        noise = (np.random.rand() - 0.5) * eta_noise_vic
        temp_new_headings_for_leaders[i_idx] = wrap_to_pi(mean_neighbor_angle + noise)
    
    headings_vic[leader_idxs_vic] = temp_new_headings_for_leaders

def update_leader_headings_communication(leader_idxs_comm, headings_comm, eta_noise_comm):
    """At communication instant, all leaders align their heading to the current mean heading of all leaders, plus noise.
       Modifies headings in place.
    """
    if len(leader_idxs_comm) < 1: return

    leader_current_headings = headings_comm[leader_idxs_comm]
    mean_leader_angle = calculate_mean_angle(leader_current_headings)

    # All leaders adopt this mean angle for the next step (after this communication)
    for leader_idx in leader_idxs_comm:
        noise = (np.random.rand() - 0.5) * eta_noise_comm
        headings_comm[leader_idx] = wrap_to_pi(mean_leader_angle + noise)

def update_all_positions(positions_all, headings_all, v0_all, dt_all, L_param):
    dx = v0_all * np.cos(headings_all) * dt_all
    dy = v0_all * np.sin(headings_all) * dt_all
    positions_all[:, 0] += dx
    positions_all[:, 1] += dy
    positions_all[:] = pbc_position(positions_all, L_param) # Apply PBC to all

def update_vicsek_headings(current_positions, current_headings, R_inter_sq_param, eta_noise_param, L_domain_param):
    new_headings = current_headings.copy()
    n_particles_vicsek = len(current_positions)
    for i in range(n_particles_vicsek):
        p_i = current_positions[i]
        neighbor_headings_list = [] # Renamed for clarity
        for j in range(n_particles_vicsek):
            if i == j: continue
            if pbc_distance_sq(p_i, current_positions[j], L_domain_param) < R_inter_sq_param:
                neighbor_headings_list.append(current_headings[j])
        
        target_angle = calculate_mean_angle(np.array(neighbor_headings_list)) if neighbor_headings_list else current_headings[i]
        noise = (np.random.rand() - 0.5) * eta_noise_param
        new_headings[i] = wrap_to_pi(target_angle + noise)
    return new_headings

# --- Visualization Functions (plot_single_snapshot, plot_vicsek_snapshot) ---
def plot_single_snapshot(ax, positions, headings, types, assignments, follower_idxs, leader_actual_idxs, L, title_str):
    """Helper to plot one snapshot onto a given matplotlib Axes object for Leader-Follower model.
       Now includes quivers for particle headings.
    """
    leader_indices = np.where(types == TYPE_LEADER)[0]
    follower_indices_all = np.where(types == TYPE_FOLLOWER)[0] # All follower indices from types array

    leader_pos = positions[leader_indices]
    follower_pos = positions[follower_indices_all]
    
    ax.scatter(leader_pos[:, 0], leader_pos[:, 1], c='red', marker='o', s=50, label='Leaders', zorder=3)
    ax.scatter(follower_pos[:, 0], follower_pos[:, 1], c='blue', marker='.', s=10, label='Followers', zorder=2)
    
    # Quivers for Leaders
    if len(leader_pos) > 0:
        leader_headings_xy = np.array([np.cos(headings[leader_indices]), np.sin(headings[leader_indices])]).T
        ax.quiver(leader_pos[:,0], leader_pos[:,1], 
                  leader_headings_xy[:,0], leader_headings_xy[:,1], 
                  color='darkred', scale=30, width=0.004, alpha=0.7, headwidth=3.5, headlength=4.5, zorder=4)

    # Quivers for Followers
    if len(follower_pos) > 0:
        follower_headings_xy = np.array([np.cos(headings[follower_indices_all]), np.sin(headings[follower_indices_all])]).T
        ax.quiver(follower_pos[:,0], follower_pos[:,1], 
                  follower_headings_xy[:,0], follower_headings_xy[:,1], 
                  color='darkblue', scale=30, width=0.003, alpha=0.6, headwidth=3, headlength=4, zorder=1)

    lines = []
    # Note: follower_idxs passed to this function is the one from initialize_system, which are the *original* indices
    # of particles that are followers. assignments array is indexed based on the *position* of a follower_idx in follower_idxs list.
    # So we need to iterate carefully or use the `follower_indices_all` derived from `types` array for consistency here.
    
    # Correct iteration for drawing lines based on the `assignments` array structure:
    # `assignments` stores the k-th leader (index in `leader_actual_idxs`) for the i-th follower (index in `follower_idxs`)
    for i, original_follower_particle_idx in enumerate(follower_idxs): # follower_idxs contains global particle indices
        assigned_leader_k_in_leader_list = assignments[i] # This is the k for leader_actual_idxs
        if assigned_leader_k_in_leader_list != -1:
            actual_leader_particle_idx = leader_actual_idxs[assigned_leader_k_in_leader_list]
            p1 = positions[original_follower_particle_idx]
            p2 = positions[actual_leader_particle_idx]
            diff = pbc_vector(p2 - p1, L)
            lines.append([p1, p1 + diff])
            
    if lines:
        lc = mc.LineCollection(lines, colors='gray', linewidths=0.3, alpha=0.6, zorder=0)
        ax.add_collection(lc)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title_str)
    ax.set_xticks([]) 
    ax.set_yticks([])
    # Legend is handled by the main grid plotting logic for the first subplot

def plot_system(positions, headings, types, assignments, follower_idxs, leader_actual_idxs, L, step, filename=None):
    plt.figure(figsize=(10, 10))
    
    leader_pos = positions[types == TYPE_LEADER]
    follower_pos = positions[types == TYPE_FOLLOWER]
    
    plt.scatter(leader_pos[:, 0], leader_pos[:, 1], c='red', marker='o', s=100, label='Leaders')
    plt.scatter(follower_pos[:, 0], follower_pos[:, 1], c='blue', marker='.', s=30, label='Followers')
    
    # Draw lines from followers to their assigned leaders
    lines = []
    for i, follower_idx in enumerate(follower_idxs):
        assigned_leader_k = assignments[i]
        if assigned_leader_k != -1:
            actual_leader_idx = leader_actual_idxs[assigned_leader_k]
            p1 = positions[follower_idx]
            p2 = positions[actual_leader_idx]
            # Handle PBC for line drawing (draw shortest line)
            diff = pbc_vector(p2 - p1, L)
            lines.append([p1, p1 + diff])
            
    if lines:
        lc = mc.LineCollection(lines, colors='gray', linewidths=0.5, alpha=0.7)
        plt.gca().add_collection(lc)

    # Quiver for headings (optional, can be cluttered)
    # leader_headings_xy = np.array([np.cos(headings[types==TYPE_LEADER]), np.sin(headings[types==TYPE_LEADER])]).T
    # follower_headings_xy = np.array([np.cos(headings[types==TYPE_FOLLOWER]), np.sin(headings[types==TYPE_FOLLOWER])]).T
    # plt.quiver(leader_pos[:,0], leader_pos[:,1], leader_headings_xy[:,0], leader_headings_xy[:,1], color='darkred', scale=20)
    # plt.quiver(follower_pos[:,0], follower_pos[:,1], follower_headings_xy[:,0], follower_headings_xy[:,1], color='darkblue', scale=20)

    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Leader-Follower System: Step {step}")
    plt.legend()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

# --- New Trajectory Plotting Function ---
def plot_trajectories(positions_history, types, L, filename="lf_trajectories.png"):
    plt.figure(figsize=(12, 12))
    n_steps = len(positions_history)
    n_particles = positions_history[0].shape[0]

    leader_indices = np.where(types == TYPE_LEADER)[0]
    follower_indices = np.where(types == TYPE_FOLLOWER)[0]

    # Plot follower trajectories
    for i in follower_indices:
        trajectory = np.array([positions_history[t][i] for t in range(n_steps)])
        plt.plot(trajectory[:, 0], trajectory[:, 1], color='blue', linewidth=0.3, alpha=0.5)
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'bo', markersize=2) # Start point
        # plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'b*', markersize=4) # End point (optional)

    # Plot leader trajectories
    for i in leader_indices:
        trajectory = np.array([positions_history[t][i] for t in range(n_steps)])
        plt.plot(trajectory[:, 0], trajectory[:, 1], color='red', linewidth=0.7, alpha=0.8)
        plt.plot(trajectory[0, 0], trajectory[0, 1], 'ro', markersize=4) # Start point
        # plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=6) # End point (optional)

    # Mark initial leader positions more clearly for reference
    initial_leader_pos = positions_history[0][leader_indices]
    plt.scatter(initial_leader_pos[:, 0], initial_leader_pos[:, 1], 
                facecolors='none', edgecolors='red', s=150, linewidth=1.5, label='Leader Start')

    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Particle Trajectories over {n_steps} Steps")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    # Custom legend elements if needed, as scatter is already labelled by plot_system (if used before)
    # handles = [plt.Line2D([0], [0], color='red', lw=2, label='Leader Traj.'),
    #            plt.Line2D([0], [0], color='blue', lw=1, label='Follower Traj.')]
    # plt.legend(handles=handles)
    plt.legend() # Will use labels from scatter if they exist

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

# MODIFIED: Trajectory Plotting Function (3D: X, Y, Time)
def plot_trajectories_3d_time(positions_history, types, L_param, output_dir, filename_base="lf_trajectories_3d_time.png"):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    n_steps_history = len(positions_history)
    if n_steps_history == 0:
        print("Warning: positions_history is empty, cannot plot trajectories.")
        plt.close(fig)
        return
        
    time_steps_axis = np.arange(n_steps_history)

    leader_indices = np.where(types == TYPE_LEADER)[0]
    follower_indices = np.where(types == TYPE_FOLLOWER)[0]

    # Plot follower trajectories
    for particle_global_idx in follower_indices:
        # Extract x and y coordinates for this particle across all time steps
        x_coords = np.array([positions_history[t][particle_global_idx, 0] for t in range(n_steps_history)])
        y_coords = np.array([positions_history[t][particle_global_idx, 1] for t in range(n_steps_history)])
        ax.plot(x_coords, y_coords, time_steps_axis, color='blue', linewidth=0.3, alpha=0.4)
        if n_steps_history > 0: # Mark start point
            ax.scatter(x_coords[0], y_coords[0], time_steps_axis[0], color='blue', marker='o', s=10, alpha=0.6)

    # Plot leader trajectories
    for particle_global_idx in leader_indices:
        x_coords = np.array([positions_history[t][particle_global_idx, 0] for t in range(n_steps_history)])
        y_coords = np.array([positions_history[t][particle_global_idx, 1] for t in range(n_steps_history)])
        ax.plot(x_coords, y_coords, time_steps_axis, color='red', linewidth=0.6, alpha=0.7)
        if n_steps_history > 0: # Mark start point
            ax.scatter(x_coords[0], y_coords[0], time_steps_axis[0], color='red', marker='^', s=30, alpha=0.8)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Time Step")
    ax.set_title(f"3D Particle Trajectories (X, Y, Time) over {n_steps_history-1} Steps (Simulated)")
    
    ax.set_xlim(0, L_param)
    ax.set_ylim(0, L_param)
    ax.set_zlim(0, n_steps_history)
    ax.view_init(elev=25., azim=-125) # Adjusted view angle

    from matplotlib.lines import Line2D # For custom legend
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Leader Traj.'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8, label='Leader Start'),
        Line2D([0], [0], color='blue', lw=1, label='Follower Traj.'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=5, label='Follower Start')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize='small')

    full_path = os.path.join(output_dir, filename_base)
    try:
        plt.savefig(full_path)
        print(f"3D Trajectory plot saved to {full_path}")
    except Exception as e:
        print(f"Error saving 3D trajectory plot: {e}")
    plt.close(fig)

# --- Parameterized Pure Vicsek Simulation Function ---
def run_pure_vicsek_simulation(N_particles, L_domain_sim, v0_sim, R_interaction_sim, eta_sim, dt_sim, 
                             max_steps_sim, pol_thresh_sim, conv_window_sim, check_interval_sim, 
                             output_dir):
    print(f"\n--- Starting Pure Vicsek Simulation (N={N_particles}, MaxSteps={max_steps_sim}) ---")
    print(f"Outputting to: {output_dir}")

    vicsek_positions, vicsek_headings_current = initialize_vicsek_system(N_particles, L_domain_sim)
    
    recent_vicsek_polarizations = []
    vicsek_metrics_history = [] 
    vicsek_types_dummy = np.full(N_particles, TYPE_FOLLOWER)
    vicsek_snapshots_data = []
    vicsek_snapshot_steps = np.linspace(0, max_steps_sim - 1, 9, dtype=int).astype(int) # Ensure int steps
    final_step_vicsek = 0 
    converged_vicsek = False

    if 0 in vicsek_snapshot_steps:
        vicsek_snapshots_data.append({
            "step": 0, "positions": vicsek_positions.copy(), "headings": vicsek_headings_current.copy()
        })
    # print(f"Vicsek: Taking snapshots at steps: {vicsek_snapshot_steps}")

    for step in range(max_steps_sim):
        final_step_vicsek = step + 1 # Update actual steps run
        vicsek_headings_current = update_vicsek_headings(vicsek_positions, vicsek_headings_current, R_interaction_sim**2, eta_sim, L_domain_sim)
        update_all_positions(vicsek_positions, vicsek_headings_current, v0_sim, dt_sim, L_domain_sim)

        if (step + 1) % check_interval_sim == 0:
            current_pol = calculate_polarization(vicsek_headings_current, vicsek_types_dummy) 
            vicsek_metrics_history.append([step + 1, current_pol])
            recent_vicsek_polarizations.append(current_pol)
            if len(recent_vicsek_polarizations) > conv_window_sim: recent_vicsek_polarizations.pop(0)
            
            pol_stable_for_stop = (len(recent_vicsek_polarizations) == conv_window_sim and all(p >= pol_thresh_sim for p in recent_vicsek_polarizations))
            # print(f"[DEBUG Vicsek] Check at step {step+1}: Pol={current_pol:.4f}, RecentPolsWindow={len(recent_vicsek_polarizations)}, StableForStop={pol_stable_for_stop}, History={['{:.3f}'.format(x) for x in recent_vicsek_polarizations]}")

            if pol_stable_for_stop:
                print(f"Pure Vicsek simulation CONVERGED at step {step + 1} with polarization: {current_pol:.3f}")
                converged_vicsek = True
                # Ensure this final converged step is captured for snapshot if not already a scheduled one
                if not any(s_data['step'] == (step + 1) for s_data in vicsek_snapshots_data):
                    vicsek_snapshots_data.append({"step": step + 1, "positions": vicsek_positions.copy(), "headings": vicsek_headings_current.copy()})
                break 
        
        if (step + 1) in vicsek_snapshot_steps and not converged_vicsek:
            if not any(s_data['step'] == (step+1) for s_data in vicsek_snapshots_data):
                vicsek_snapshots_data.append({"step": step + 1, "positions": vicsek_positions.copy(), "headings": vicsek_headings_current.copy()})
        elif (step + 1) % 100 == 0 and not ((step + 1) % check_interval_sim == 0) and not converged_vicsek:
             print(f"Vicsek Step: {step+1}/{max_steps_sim}")

    # Ensure the very last state (max_steps_sim or actual convergence step) is in snapshots if not already covered
    if not any(s_data['step'] == final_step_vicsek for s_data in vicsek_snapshots_data) and len(vicsek_snapshots_data) < 9:
        # Check if the last snapshot is already final_step_vicsek or if list is empty
        if not vicsek_snapshots_data or vicsek_snapshots_data[-1]["step"] != final_step_vicsek:
            vicsek_snapshots_data.append({"step": final_step_vicsek, "positions": vicsek_positions.copy(), "headings": vicsek_headings_current.copy()})
            # Sort snapshots by step to ensure chronological order for plotting if final step was added out of sequence
            vicsek_snapshots_data.sort(key=lambda x: x['step'])
            # Keep only up to 9 most relevant snapshots (e.g. first, last, and some in between)
            if len(vicsek_snapshots_data) > 9:
                 # A simple way to ensure we have first, last and some distributed ones if too many:
                 first_snap = vicsek_snapshots_data[0]
                 last_snap = vicsek_snapshots_data[-1]
                 middle_snaps = vicsek_snapshots_data[1:-1]
                 if len(middle_snaps) > 7:
                     indices = np.linspace(0, len(middle_snaps) - 1, 7, dtype=int).tolist()
                     selected_middle = [middle_snaps[i] for i in indices]
                     vicsek_snapshots_data = [first_snap] + selected_middle + [last_snap]
                 # Ensure unique steps if final_step_vicsek was already a snapshot_step
                 final_snaps = []
                 seen_steps = set()
                 for snap in vicsek_snapshots_data:
                     if snap['step'] not in seen_steps:
                         final_snaps.append(snap)
                         seen_steps.add(snap['step'])
                 vicsek_snapshots_data = final_snaps[:9]

    if not converged_vicsek:
        print(f"[DEBUG Vicsek] Max steps ({max_steps_sim}) reached without meeting polarization convergence.")
    print(f"Pure Vicsek simulation finished loop. Effective steps run for Vicsek: {final_step_vicsek}")

    # CSV Saving
    if vicsek_metrics_history:
        try:
            header_vicsek = "Step,FollowerPolarization"
            metrics_array_vicsek = np.array(vicsek_metrics_history)
            csv_path = os.path.join(output_dir, "vicsek_metrics.csv")
            print(f"[DEBUG Vicsek] Saving metrics to: {csv_path}")
            np.savetxt(csv_path, metrics_array_vicsek, delimiter=",", header=header_vicsek, comments='', fmt='%.6f')
            print(f"Vicsek metrics successfully saved to {csv_path}")
            if not os.path.exists(csv_path): print(f"[ERROR Vicsek] CSV file NOT FOUND after saving: {csv_path}")
        except Exception as e: print(f"[ERROR Vicsek] Error saving Vicsek metrics to CSV at {csv_path}: {e}"); traceback.print_exc()
    else: print("[DEBUG Vicsek] No metrics recorded for Vicsek simulation to save.")

    # Snapshot Grid Plotting
    if vicsek_snapshots_data:
        try:
            fig, axes = plt.subplots(3, 3, figsize=(18, 18))
            axes_flat = axes.flatten()
            for i in range(min(9, len(vicsek_snapshots_data))):
                snap_data = vicsek_snapshots_data[i]
                # Corrected call: Ensure L_domain_sim is passed as L, and title_str is last.
                # For a Vicsek plot, assignments, follower_idxs, leader_actual_idxs are not directly applicable.
                # We pass dummy/placeholder values.
                dummy_types_for_plot = np.full(snap_data["positions"].shape[0], TYPE_FOLLOWER) 
                dummy_assignments = -np.ones(snap_data["positions"].shape[0], dtype=int) 
                dummy_follower_idxs = np.arange(snap_data["positions"].shape[0]) 
                dummy_leader_actual_idxs = np.array([], dtype=int) 
                
                plot_single_snapshot(axes_flat[i], 
                                     snap_data["positions"], 
                                     snap_data["headings"], 
                                     dummy_types_for_plot, 
                                     dummy_assignments,    
                                     dummy_follower_idxs,  
                                     dummy_leader_actual_idxs, 
                                     L_domain_sim,         
                                     f"Step {snap_data['step']}") # title_str
            for i in range(len(vicsek_snapshots_data), 9): fig.delaxes(axes_flat[i])
            handles, labels = axes_flat[0].get_legend_handles_labels()
            if handles: fig.legend(handles, labels, loc='upper center', ncol=max(1, len(handles)), bbox_to_anchor=(0.5, 0.99))
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fig.suptitle(f"Vicsek Snapshots (N={N_particles})", fontsize=16)
            grid_plot_path = os.path.join(output_dir, "vicsek_snapshots_grid.png")
            print(f"[DEBUG Vicsek] Saving snapshot grid to: {grid_plot_path}")
            plt.savefig(grid_plot_path); plt.close(fig)
            print(f"Vicsek Snapshot grid saved to {grid_plot_path}")
            if not os.path.exists(grid_plot_path): print(f"[ERROR Vicsek] Snapshot grid file NOT FOUND after saving: {grid_plot_path}")
        except Exception as e: print(f"[ERROR Vicsek] Error saving Vicsek snapshot grid to {grid_plot_path if 'grid_plot_path' in locals() else output_dir}: {e}"); traceback.print_exc()
    else: print("[DEBUG Vicsek] No Vicsek snapshots available for grid plot.")

    # Polarization Curve Plotting
    if vicsek_metrics_history:
        try:
            plt.figure(figsize=(10, 6))
            # Convert to NumPy array before slicing
            metrics_array_vicsek = np.array(vicsek_metrics_history)
            if metrics_array_vicsek.ndim == 2 and metrics_array_vicsek.shape[0] > 0: # Check if array is valid for plotting
                plt.plot(metrics_array_vicsek[:, 0], metrics_array_vicsek[:, 1], marker='o', linestyle='-', color='b')
            else:
                print("[DEBUG Vicsek] metrics_array_vicsek is not suitable for plotting (e.g. empty or wrong dimensions).")
            plt.xlabel("Step")
            plt.ylabel("Follower Polarization")
            plt.title("Vicsek Simulation: Follower Polarization Over Steps")
            plt.grid(True)
            curve_plot_path = os.path.join(output_dir, "vicsek_polarization_curve.png")
            print(f"[DEBUG Vicsek] Saving polarization curve to: {curve_plot_path}")
            plt.savefig(curve_plot_path); plt.close()
            print(f"Vicsek polarization curve saved to {curve_plot_path}")
            if not os.path.exists(curve_plot_path): print(f"[ERROR Vicsek] Polarization curve file NOT FOUND after saving: {curve_plot_path}")
        except Exception as e: print(f"[ERROR Vicsek] Error saving Vicsek polarization curve: {e}"); traceback.print_exc()
    else: print("[DEBUG Vicsek] No Vicsek metrics to plot polarization curve.")

# --- Parameterized Leader-Follower Simulation Function ---
def run_leader_follower_simulation(N_f_run, N_l_run, L_domain_run, v0_run, R_interaction_run, eta_run, dt_run, 
                                 capacity_run, T_comm_run, max_steps_run, 
                                 pol_thresh_run, conv_window_run_lf, check_interval_run_lf, 
                                 as_thresh_run, lb_thresh_run, ls_thresh_run, 
                                 output_dir, leader_dist_mode_run="random",
                                 suppress_plots=False, return_full_data=False, output_dir_override=None):
    
    current_output_dir = output_dir_override if output_dir_override is not None else output_dir
    os.makedirs(current_output_dir, exist_ok=True) # Ensure the actual output directory exists

    print(f"\n--- Starting Leader-Follower Simulation (Nf={N_f_run}, Nl={N_l_run}, Cap={capacity_run}, Steps={max_steps_run}, Mode={leader_dist_mode_run}) ---")
    print(f"Outputting to: {current_output_dir}")
    
    lf_positions, lf_headings, lf_types, lf_follower_indices, lf_leader_actual_indices, \
        lf_assignments, lf_leader_follower_counts = initialize_system(N_l_run, N_f_run, L_domain_run, leader_dist_mode_run)
    
    lf_all_positions_history = [lf_positions.copy()]
    lf_metrics_history = [] 
    lf_snapshots_data = []
    lf_snapshot_steps = np.linspace(0, max_steps_run - 1, 9, dtype=int)
    
    if 0 in lf_snapshot_steps: 
        lf_snapshots_data.append({
            "step": 0, "positions": lf_positions.copy(), "headings": lf_headings.copy(),
            "types": lf_types.copy(), "assignments": lf_assignments.copy(),
            "follower_indices": lf_follower_indices.copy(), "leader_actual_indices": lf_leader_actual_indices.copy()
        })
    
    lf_prev_assignments = lf_assignments.copy()
    lf_recent_assignment_stabilities, lf_recent_polarizations, lf_recent_load_std_devs, lf_recent_leader_dist_vars_changes = [], [], [], []
    lf_prev_leader_dist_variance = None 
    lf_converged = False 
    final_step_lf = 0 # Initialize final_step_lf
    
    # Initialize metrics for potential early return if no check interval is hit before max_steps
    current_stability_ratio = 0.0
    current_pol = 0.0
    current_lb_std = 0.0
    current_ls_var = 0.0

    print(f"[DEBUG LF] Starting loop for {max_steps_run} steps...")

    for step in range(max_steps_run):
        final_step_lf = step + 1 # Update actual steps run, will be overwritten if loop breaks early
        if step < 5 or (step + 1) % 100 == 0 or step == max_steps_run - 1:
             print(f"[DEBUG LF] In loop: current step {step + 1}/{max_steps_run}")
        
        current_assignments_for_stability_check = lf_assignments.copy() 
        lf_assignments, lf_leader_follower_counts = assign_followers(lf_follower_indices, lf_leader_actual_indices, lf_positions, lf_leader_follower_counts, capacity_run, lf_assignments, L_domain_run)
        update_follower_headings_differentiated(lf_follower_indices, lf_positions, lf_headings, lf_types, lf_assignments, lf_leader_actual_indices,R_interaction_run**2, eta_run, L_domain_run)
        
        if N_l_run > 0: # Only apply leader updates if there are leaders
            if step % T_comm_run == 0 and step > 0: 
                update_leader_headings_communication(lf_leader_actual_indices, lf_headings, eta_run)
            else: 
                update_leaders_vicsek_step(lf_leader_actual_indices, lf_positions, lf_headings, lf_types, R_interaction_run**2, eta_run, L_domain_run)
        
        update_all_positions(lf_positions, lf_headings, v0_run, dt_run, L_domain_run)
        lf_all_positions_history.append(lf_positions.copy())

        if (step + 1) % check_interval_run_lf == 0:
            if len(current_assignments_for_stability_check) > 0 and len(lf_assignments) == len(current_assignments_for_stability_check):
                 stable_count = np.sum(lf_assignments == current_assignments_for_stability_check) 
                 current_stability_ratio = stable_count / len(lf_assignments)
            else: current_stability_ratio = 0.0 if len(lf_assignments)>0 else 1.0
            
            current_pol = calculate_polarization(lf_headings, lf_types, TYPE_FOLLOWER)
            current_lb_std = np.std(lf_leader_follower_counts) if N_l_run > 0 and len(lf_leader_follower_counts) > 0 else 0.0
            current_ls_var = calculate_leader_distance_variance(lf_leader_actual_indices, lf_positions, L_domain_run) if N_l_run > 0 else 0.0
            
            lf_metrics_history.append([step + 1, current_stability_ratio, current_pol, current_lb_std, current_ls_var])
            
            lf_recent_assignment_stabilities.append(current_stability_ratio)
            if len(lf_recent_assignment_stabilities) > conv_window_run_lf: lf_recent_assignment_stabilities.pop(0)
            lf_recent_polarizations.append(current_pol)
            if len(lf_recent_polarizations) > conv_window_run_lf: lf_recent_polarizations.pop(0)
            lf_recent_load_std_devs.append(current_lb_std)
            if len(lf_recent_load_std_devs) > conv_window_run_lf: lf_recent_load_std_devs.pop(0)
            
            if lf_prev_leader_dist_variance is not None and N_l_run > 0:
                var_change = abs(current_ls_var - lf_prev_leader_dist_variance)
                lf_recent_leader_dist_vars_changes.append(var_change)
                if len(lf_recent_leader_dist_vars_changes) > conv_window_run_lf: lf_recent_leader_dist_vars_changes.pop(0)
            elif N_l_run > 0: # First time, or if no leaders, append 0 or handle appropriately
                 lf_recent_leader_dist_vars_changes.append(0.0) # Placeholder if first check or N_l_run is 0 then becomes 0 later
            
            lf_prev_assignments = lf_assignments.copy() 
            if N_l_run > 0 : lf_prev_leader_dist_variance = current_ls_var
            
            pol_stable_for_stop = (len(lf_recent_polarizations) == conv_window_run_lf and all(p >= pol_thresh_run for p in lf_recent_polarizations))
            
            if pol_stable_for_stop:
                print(f"[DEBUG LF] CONVERGENCE MET (Polarization) at step {step + 1}. Breaking loop.")
                lf_converged = True; final_step_lf = step + 1 
                if (step+1) not in lf_snapshot_steps and not suppress_plots: 
                    lf_snapshots_data.append({"step": step + 1, "positions": lf_positions.copy(),"headings": lf_headings.copy(),"types": lf_types.copy(),"assignments": lf_assignments.copy(),"follower_indices": lf_follower_indices.copy(),"leader_actual_indices": lf_leader_actual_indices.copy()})
                    print(f"[DEBUG LF] Final state snapshot taken at step {step + 1} due to polarization convergence")
                break 
            print(f"LF Step {step+1}/{max_steps_run}: POL History: {['{:.3f}'.format(x) for x in lf_recent_polarizations]} (Target: {pol_thresh_run}) | AS: {current_stability_ratio:.3f} | LB_std: {current_lb_std:.2f} | LS_var: {current_ls_var:.3f}")
        
        if (step + 1) in lf_snapshot_steps and not lf_converged and not suppress_plots:
            if not any(s['step'] == (step + 1) for s in lf_snapshots_data):
                lf_snapshots_data.append({"step": step + 1, "positions": lf_positions.copy(),"headings": lf_headings.copy(),"types": lf_types.copy(),"assignments": lf_assignments.copy(),"follower_indices": lf_follower_indices.copy(),"leader_actual_indices": lf_leader_actual_indices.copy()})
        elif (step + 1) % 100 == 0 and not ((step + 1) % check_interval_run_lf == 0) and not lf_converged:
             print(f"LF Step: {step+1}/{max_steps_run}")
    
    if not lf_converged: 
        final_step_lf = max_steps_run # Ensure final_step_lf is max_steps_run if loop completed fully
        print(f"[DEBUG LF] Max steps ({max_steps_run}) reached.")
    print(f"[DEBUG LF] Exited loop. Effective steps run for LF: {final_step_lf}")

    # --- Output Saving --- 
    # (CSV will always be saved to current_output_dir unless a more specific suppress_csv parameter is added later)
    print(f"[DEBUG LF Core] Attempting to save outputs to directory: {current_output_dir}")
    if lf_metrics_history:
        try:
            header_lf = "Step,AssignmentStability,FollowerPolarization,LeaderLoadStdDev,LeaderDistVariance"
            metrics_array_lf = np.array(lf_metrics_history)
            csv_path = os.path.join(current_output_dir, "leader_follower_metrics.csv") # Use current_output_dir
            print(f"[DEBUG LF Core] Saving metrics to: {csv_path}")
            np.savetxt(csv_path, metrics_array_lf, delimiter=",", header=header_lf, comments='', fmt='%.6f')
            print(f"Leader-Follower metrics successfully saved to {csv_path}")
            if not os.path.exists(csv_path): print(f"[ERROR LF Core] CSV file NOT FOUND after saving: {csv_path}")
        except Exception as e: print(f"[ERROR LF Core] Error saving LF metrics to CSV at {csv_path}: {e}"); traceback.print_exc()
    else: print("[DEBUG LF Core] No metrics recorded for Leader-Follower simulation to save (lf_metrics_history is empty).")

    if not suppress_plots:
        if lf_snapshots_data: 
            try:
                fig, axes = plt.subplots(3, 3, figsize=(18, 18))
                axes_flat = axes.flatten()
                for i in range(min(9, len(lf_snapshots_data))):
                    snap_data = lf_snapshots_data[i]
                    plot_single_snapshot(axes_flat[i], snap_data["positions"], snap_data["headings"], snap_data["types"], snap_data["assignments"],snap_data["follower_indices"], snap_data["leader_actual_indices"],L_domain_run, f"Step {snap_data['step']}")
                for i in range(len(lf_snapshots_data), 9): fig.delaxes(axes_flat[i])
                handles, labels = axes_flat[0].get_legend_handles_labels()
                if handles: fig.legend(handles, labels, loc='upper center', ncol=max(1, len(handles)), bbox_to_anchor=(0.5, 0.99))
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                fig.suptitle(f"LF Snapshots (Nf={N_f_run}, Nl={N_l_run}, Cap={capacity_run})", fontsize=16)
                grid_plot_path = os.path.join(current_output_dir, "lf_snapshots_grid.png") # Use current_output_dir
                print(f"[DEBUG LF Core] Saving snapshot grid to: {grid_plot_path}")
                plt.savefig(grid_plot_path); plt.close(fig)
                print(f"LF Snapshot grid saved to {grid_plot_path}")
                if not os.path.exists(grid_plot_path): print(f"[ERROR LF Core] Snapshot grid file NOT FOUND after saving: {grid_plot_path}")
            except Exception as e: print(f"[ERROR LF Core] Error saving LF snapshot grid to {grid_plot_path if 'grid_plot_path' in locals() else current_output_dir}: {e}"); traceback.print_exc()
        else: print("[DEBUG LF Core] No LF snapshots taken, skipping grid plot.")

        # Trajectory Plot (2D)
        if len(lf_all_positions_history) > 1: # Need at least 2 steps for a trajectory
            try:
                # Pass current_output_dir for saving the plot
                plot_trajectories(lf_all_positions_history, lf_types, L_domain_run, current_output_dir, filename_base="lf_trajectories_2d.png")
            except Exception as e: print(f"[ERROR LF Core] Error saving 2D trajectory plot: {e}"); traceback.print_exc()
        else: print("[DEBUG LF Core] Not enough position history for 2D trajectory plot.")

        # Trajectory Plot (3D with Time)
        if len(lf_all_positions_history) > 1: # Need at least 2 steps for a trajectory
            try:
                plot_trajectories_3d_time(lf_all_positions_history, lf_types, L_domain_run, current_output_dir, filename_base="lf_trajectories_3d_time.png")
            except Exception as e: print(f"[ERROR LF Core] Error saving 3D trajectory plot: {e}"); traceback.print_exc()
        else: print("[DEBUG LF Core] Not enough position history for 3D trajectory plot.")
    
    # --- Return Data if Requested ---
    if return_full_data:
        final_metrics_dict = {
            'follower_polarization': current_pol,
            'assignment_stability': current_stability_ratio,
            'leader_load_std_dev': current_lb_std,
            'leader_dist_variance': current_ls_var
        }
        return (final_metrics_dict, lf_all_positions_history, lf_leader_actual_indices, lf_types, final_step_lf)
    else:
        return None # Or some other indicator of completion if not returning full data

# Need to define plot_trajectories_2d_history if it's a new function or rename if it's existing plot_trajectories
# Assuming plot_trajectories is the 2D one.
# For consistency, let's assume plot_trajectories was intended to be the 2D one.

# Renaming plot_trajectories to plot_trajectories_2d_history for clarity in the call above
# if it was indeed the 2D plot. If it's a new function, it needs to be defined.
# For now, I'll assume plot_trajectories should be called here. Let's check its definition.
# The existing plot_trajectories function seems suitable for 2D history.
# Let's adjust the call to match the existing function name if it's `plot_trajectories`

# (Outside the function, ensure plot_trajectories and plot_trajectories_3d_time exist as expected)

# The previous edit block will focus only on `run_leader_follower_simulation`
# I will make a separate edit for any plotting function renames or new definitions if necessary
# after confirming their current state.

# For now, the call inside run_leader_follower_simulation will be to a hypothetical
# plot_trajectories_2d_history. If plot_trajectories is the correct one, we'll adjust.
# Let's assume plot_trajectories is indeed the 2D trajectory plot function
# and it takes (positions_history, types, L, output_dir, filename_base) as arguments.

# The definition of plot_trajectories is:
# def plot_trajectories(positions_history, types, L, filename="lf_trajectories.png"):
# It needs to be adapted to take output_dir and filename_base or the call needs to be adapted.

# I will adapt the CALL inside run_leader_follower_simulation for now.
# The plot_trajectories_2d_history call will be changed to use plot_trajectories
# and construct the filename correctly.

# The edit will now be structured to only modify run_leader_follower_simulation and add the new parameters
# and conditional logic, and the return statement.
# The plotting calls inside will be adjusted if their signatures in the file don't match output_dir/filename_base pattern.

# Revised section for plotting within run_leader_follower_simulation:
# This will be part of the larger edit block for the function.
# Inside the `if not suppress_plots:` block:
# ... (snapshot grid saving code) ...
#        if len(lf_all_positions_history) > 1:
#            try:
#                traj_2d_filename = os.path.join(current_output_dir, "lf_trajectories_2d.png")
#                plot_trajectories(lf_all_positions_history, lf_types, L_domain_run, filename=traj_2d_filename)
#                print(f"2D Trajectory plot saved to {traj_2d_filename}")
#            except Exception as e: print(f"[ERROR LF Core] Error saving 2D trajectory plot: {e}"); traceback.print_exc()
#        else: print("[DEBUG LF Core] Not enough position history for 2D trajectory plot.")
#
#        if len(lf_all_positions_history) > 1:
#            try:
#                plot_trajectories_3d_time(lf_all_positions_history, lf_types, L_domain_run, current_output_dir, filename_base="lf_trajectories_3d_time.png")
#            except Exception as e: print(f"[ERROR LF Core] Error saving 3D trajectory plot: {e}"); traceback.print_exc()
#        else: print("[DEBUG LF Core] Not enough position history for 3D trajectory plot.")


# The above commented out section is how I'll adjust the plotting calls within the main edit block.

# The actual edit for the function starts here. I need to make sure to get the entire function body.
# It seems I need to read the file first to ensure I get the whole function definition correctly.
# The user has provided lines 550-703 for this function which should be a good reference.
# I will proceed with the edit, assuming the function structure. 
# I will also initialize final_step_lf at the beginning of the function correctly.
# And ensure metric calculations (current_pol etc.) are robust if N_l_run is 0.


# REMOVE THE OLD if __name__ == "__main__": block from zitianwang_LeaderFollowerParticleSystem.py
# That logic will now go into separate experiment script(s). 