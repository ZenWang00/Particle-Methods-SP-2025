# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange # For performance optimization
import os
import time

# --- Simulation Parameters ---
SIGMA = 1.0         # Lennard-Jones distance parameter (reduced units)
EPSILON = 1.0       # Lennard-Jones energy parameter (reduced units)
MASS = 1.0          # Particle mass (reduced units)
KB = 1.0            # Boltzmann constant (reduced units)
RC = 2.5            # Cut-off radius (in units of SIGMA)
RC_SQ = RC * RC     # Squared cut-off radius (for efficiency)
L = 30.0            # Box size (units of SIGMA)
DT_DEFAULT = 0.005  # Default time step (reduced units)
TAU_T = DT_DEFAULT / 0.0025 # Berendsen thermostat relaxation time (tau)

# Create output directory
OUTPUT_DIR = "lj_simulation_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Core Physics Functions (Numba Optimized) ---

@njit(cache=True)
def minimum_image_convention(dr, L):
    """Applies minimum image convention to a distance vector."""
    return dr - L * np.round(dr / L)

@njit(cache=True)
def potential_energy_pair(r_sq, rc_sq, epsilon):
    """Calculates truncated Lennard-Jones potential energy for a pair."""
    if r_sq >= rc_sq:
        return 0.0
    # sigma = 1, so sigma^6/r^6 = 1/r^6 etc.
    inv_r_sq = 1.0 / r_sq
    inv_r_6 = inv_r_sq * inv_r_sq * inv_r_sq
    inv_r_12 = inv_r_6 * inv_r_6

    # Potential at rc for truncation shift
    inv_rc_sq = 1.0 / rc_sq
    inv_rc_6 = inv_rc_sq * inv_rc_sq * inv_rc_sq
    inv_rc_12 = inv_rc_6 * inv_rc_6
    U_rc = 4.0 * epsilon * (inv_rc_12 - inv_rc_6)

    U_r = 4.0 * epsilon * (inv_r_12 - inv_r_6)
    return U_r - U_rc

@njit(cache=True)
def force_pair(dr_vec, r_sq, rc_sq, epsilon):
    """Calculates truncated Lennard-Jones force vector for a pair."""
    if r_sq >= rc_sq:
        return np.zeros(2) # 2D simulation

    # sigma = 1
    inv_r_sq = 1.0 / r_sq
    inv_r_6 = inv_r_sq * inv_r_sq * inv_r_sq
    inv_r_12 = inv_r_6 * inv_r_6

    # F_chi(r) = 48 * epsilon * chi / r^2 * [(sigma/r)^12 - 0.5 * (sigma/r)^6]
    # Force magnitude = -dU/dr = 24*epsilon/r * [2*(sigma/r)^12 - (sigma/r)^6]
    # F_vec = (Force magnitude / r) * dr_vec
    # F_vec = (24 * epsilon / r_sq) * [2*(sigma/r)^12 - (sigma/r)^6] * dr_vec
    # F_vec = (24 * epsilon * inv_r_sq) * [2 * inv_r_12 - inv_r_6] * dr_vec

    # Using the formula provided in the assignment:
    # F_chi(r) = 48 * epsilon * chi / r^2 * [(sigma/r)^12 - 0.5 * (sigma/r)^6]
    # F_vec = (48 * epsilon / r^2) * [(sigma/r)^12 - 0.5 * (sigma/r)^6] * dr_vec
    # Note: Epsilon=1, Sigma=1
    magnitude_factor = 48.0 * epsilon * inv_r_sq * (inv_r_12 - 0.5 * inv_r_6)
    return magnitude_factor * dr_vec

# --- Cell List Implementation (Numba Optimized) ---

@njit(cache=True)
def build_cell_list(positions, L, rc):
    """Builds a cell list for efficient neighbor finding."""
    n_cells_dim = int(np.floor(L / rc))
    cell_size = L / n_cells_dim # Ensure cells cover the box exactly
    
    # Check cell size requirement
    # if cell_size < rc:
         # print(f"Warning: Cell size ({cell_size:.4f}) is smaller than cutoff radius ({rc:.4f}). Neighbor search might be incorrect.")

    head = -np.ones(n_cells_dim * n_cells_dim, dtype=np.int32)
    linked_list = -np.ones(positions.shape[0], dtype=np.int32)
    cell_map = np.empty((positions.shape[0], 2), dtype=np.int32) # Stores (cell_x, cell_y) for each particle

    for i in range(positions.shape[0]):
        # Ensure particle is inside the primary box for cell assignment
        pos = positions[i] % L
        cell_x = int(pos[0] // cell_size)
        cell_y = int(pos[1] // cell_size)
        
        # Handle potential edge cases due to floating point
        cell_x = min(cell_x, n_cells_dim - 1)
        cell_y = min(cell_y, n_cells_dim - 1)
        
        cell_map[i, 0] = cell_x
        cell_map[i, 1] = cell_y

        cell_idx = cell_x + cell_y * n_cells_dim
        linked_list[i] = head[cell_idx]
        head[cell_idx] = i

    return head, linked_list, cell_map, n_cells_dim, cell_size

@njit(cache=True, parallel=True)
def calculate_forces_and_potential_cl(positions, L, rc_sq, epsilon, head, linked_list, cell_map, n_cells_dim):
    """Calculates forces and potential energy using cell lists."""
    N = positions.shape[0]
    forces = np.zeros((N, 2))
    potential_energy_total = 0.0

    # Loop over cells
    for cell_y in prange(n_cells_dim):
        for cell_x in prange(n_cells_dim):
            cell_idx_1 = cell_x + cell_y * n_cells_dim

            # Iterate through particles in the current cell
            i = head[cell_idx_1]
            while i != -1:
                pos_i = positions[i]

                # Interactions within the same cell
                j = linked_list[i]
                while j != -1:
                    dr_vec = pos_i - positions[j]
                    dr_vec = minimum_image_convention(dr_vec, L)
                    r_sq = np.sum(dr_vec * dr_vec)

                    if r_sq < rc_sq:
                        f_pair = force_pair(dr_vec, r_sq, rc_sq, epsilon)
                        forces[i] += f_pair
                        forces[j] -= f_pair # Newton's third law
                        potential_energy_total += potential_energy_pair(r_sq, rc_sq, epsilon)
                    j = linked_list[j]

                # Interactions with neighboring cells (including diagonals)
                for dy in range(-1, 2): # dy = -1, 0, 1
                    for dx in range(-1, 2): # dx = -1, 0, 1
                        # Skip self-cell (already done) and only consider upper/right neighbors
                        # to avoid double counting
                        if dy == 0 and dx <= 0: continue # dx=0: same cell, dx<0: left neighbor already counted
                        if dy == -1: continue # Bottom neighbors already counted

                        neigh_cell_x = (cell_x + dx) % n_cells_dim
                        neigh_cell_y = (cell_y + dy) % n_cells_dim
                        cell_idx_2 = neigh_cell_x + neigh_cell_y * n_cells_dim

                        j = head[cell_idx_2]
                        while j != -1:
                            dr_vec = pos_i - positions[j]
                            dr_vec = minimum_image_convention(dr_vec, L)
                            r_sq = np.sum(dr_vec * dr_vec)

                            if r_sq < rc_sq:
                                f_pair = force_pair(dr_vec, r_sq, rc_sq, epsilon)
                                forces[i] += f_pair
                                forces[j] -= f_pair # Newton's third law
                                potential_energy_total += potential_energy_pair(r_sq, rc_sq, epsilon)
                            j = linked_list[j]

                i = linked_list[i] # Move to the next particle in the current cell

    # Potential energy is counted once per pair in the loop structure
    return forces, potential_energy_total


# --- Integration and Thermostat (Numba Optimized) ---

@njit(cache=True)
def velocity_verlet_step1(positions, velocities, forces, dt, mass, L):
    """First part of the Velocity Verlet algorithm."""
    # Update positions: r(t + dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
    accel = forces / mass
    positions += velocities * dt + 0.5 * accel * dt * dt
    # Apply periodic boundary conditions
    positions = positions % L # Wrap around
    
    # Update velocities partially: v(t + dt/2) = v(t) + 0.5*a(t)*dt
    velocities += 0.5 * accel * dt
    return positions, velocities

@njit(cache=True)
def velocity_verlet_step2(velocities, forces_new, dt, mass):
    """Second part of the Velocity Verlet algorithm."""
    # Update velocities: v(t + dt) = v(t + dt/2) + 0.5*a(t + dt)*dt
    accel_new = forces_new / mass
    velocities += 0.5 * accel_new * dt
    return velocities

@njit(cache=True)
def apply_berendsen_thermostat(velocities, T_current, T_target, dt, tau_T, N, kB):
    """Applies the Berendsen thermostat velocity scaling."""
    if tau_T <= 0: # Safety check or if thermostat is off
        return velocities
        
    lambda_sq = 1.0 + (dt / tau_T) * (T_target / T_current - 1.0)
    # Prevent instability if T_current is very close to 0
    if lambda_sq < 0: 
        print("Warning: Berendsen lambda^2 negative. T_current likely near zero.")
        return velocities # Or apply a minimum lambda if needed
    
    scaling_factor = np.sqrt(lambda_sq)
    velocities *= scaling_factor
    return velocities

# --- Initialization Functions ---

def initialize_positions(N, L):
    """Initializes particle positions on a square lattice."""
    n_dim = int(np.ceil(np.sqrt(N))) # Number of particles per dimension
    if n_dim * n_dim != N:
        print(f"Warning: N={N} is not a perfect square. Using {n_dim*n_dim} particles instead.")
        N = n_dim * n_dim
    
    spacing = L / n_dim
    positions = np.zeros((N, 2))
    idx = 0
    for i in range(n_dim):
        for j in range(n_dim):
            positions[idx, 0] = (i + 0.5) * spacing
            positions[idx, 1] = (j + 0.5) * spacing
            idx += 1
    return positions, N

def initialize_velocities(N, T_initial, mass, kB):
    """Initializes velocities randomly with zero total momentum."""
    # Draw from Gaussian distribution (related to Maxwell-Boltzmann)
    # Standard deviation for velocity component: sqrt(kB*T/m)
    std_dev = np.sqrt(kB * T_initial / mass)
    velocities = np.random.normal(0.0, std_dev, (N, 2))

    # Enforce zero total momentum
    total_momentum = np.sum(velocities * mass, axis=0)
    velocities -= total_momentum / (N * mass)

    # Optional: Rescale to exact initial temperature (Berendsen will handle it anyway)
    # ke_initial = 0.5 * mass * np.sum(velocities**2)
    # T_current = ke_initial / (N * kB) # In 2D, KE = N * kB * T (DoF = 2N, KE = DoF/2 * kB * T)
    # scale_factor = np.sqrt(T_initial / T_current)
    # velocities *= scale_factor
    
    return velocities

# --- Diagnostics and Analysis ---

@njit(cache=True)
def calculate_kinetic_energy(velocities, mass):
    """Calculates the total kinetic energy."""
    return 0.5 * mass * np.sum(velocities**2)

@njit(cache=True)
def calculate_temperature(kinetic_energy, N, kB):
    """Calculates the instantaneous temperature."""
    # KE = DoF/2 * kB * T. In 2D, DoF = 2N. So KE = N * kB * T
    if N == 0: return 0.0
    return kinetic_energy / (N * kB)

@njit(cache=True)
def calculate_total_momentum(velocities, mass):
    """Calculates the total momentum vector."""
    return np.sum(velocities * mass, axis=0)

@njit(cache=True)
def calculate_rdf(positions, L, N, rc, dr_rdf, max_dist):
    """Calculates the Radial Distribution Function (RDF), g(r)."""
    n_bins = int(np.ceil(max_dist / dr_rdf))
    hist = np.zeros(n_bins, dtype=np.int64)
    # distances_sq = [] # Store distances for normalization check if needed
    pair_count = 0

    # Effective cutoff for RDF pair search: max_dist
    # Need cell list capable of searching up to max_dist
    # Simplest approach: Rebuild cell list with larger effective 'rc' if max_dist > rc
    # For now, assume max_dist <= L/2. A simpler loop is used here for clarity,
    # but for large N, using the cell list approach even for RDF is better.
    
    # Simple pairwise loop (less efficient for large N than using cell lists)
    for i in range(N):
        for j in range(i + 1, N): # Avoid double counting and self-interaction
            dr_vec = positions[i] - positions[j]
            dr_vec = minimum_image_convention(dr_vec, L)
            r_sq = np.sum(dr_vec*dr_vec)
            r = np.sqrt(r_sq)
            
            if r < max_dist:
                bin_index = int(r // dr_rdf)
                if bin_index < n_bins:
                    hist[bin_index] += 2 # Count pair for both i->j and j->i perspective
                    pair_count +=1


    # Normalization
    rdf = np.zeros(n_bins, dtype=np.float64)
    number_density = N / (L * L)
    bin_centers = (np.arange(n_bins) + 0.5) * dr_rdf

    for k in range(n_bins):
        r_lower = k * dr_rdf
        r_upper = (k + 1) * dr_rdf
        # Volume (area in 2D) of the shell
        shell_volume = np.pi * (r_upper**2 - r_lower**2)
        # Number of ideal gas particles in the shell
        ideal_gas_count = number_density * shell_volume
        
        if ideal_gas_count > 1e-9 and hist[k] > 0: # Avoid division by zero
             # Normalization factor: N accounts for averaging over all central particles
             # The hist already contains counts from N*(N-1)/2 pairs times 2 = N*(N-1) interactions
             # Normalize by ideal gas count and number of central particles (N)
             rdf[k] = hist[k] / (N * ideal_gas_count)


    # For RDF using cell lists (more efficient):
    # Adapt calculate_forces_and_potential_cl to only calculate distances
    # and increment histogram bins, extending neighbor search if max_dist > cell_size.
    # Normalization logic remains the same.

    return rdf, bin_centers

# --- Plotting Functions ---

def plot_energies(times, E_kin, E_pot, E_tot, dt, N, filename):
    """Plots kinetic, potential, and total energy vs. time."""
    plt.figure(figsize=(10, 6))
    plt.plot(times, E_kin, label='Kinetic Energy')
    plt.plot(times, E_pot, label='Potential Energy')
    plt.plot(times, E_tot, label='Total Energy')
    plt.xlabel("Time (reduced units)")
    plt.ylabel("Energy (reduced units)")
    plt.title(f"Energy Evolution (N={N}, dt={dt})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved energy plot to {os.path.join(OUTPUT_DIR, filename)}")


def plot_temperature(times, temperature, T_target, dt, N, use_thermostat, filename):
    """Plots temperature vs. time."""
    plt.figure(figsize=(10, 6))
    plt.plot(times, temperature, label='System Temperature')
    if use_thermostat and T_target is not None:
        plt.axhline(T_target, color='r', linestyle='--', label=f'Target T = {T_target}')
    plt.xlabel("Time (reduced units)")
    plt.ylabel("Temperature (reduced units)")
    plt.title(f"Temperature Evolution (N={N}, dt={dt}, Thermostat: {use_thermostat})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved temperature plot to {os.path.join(OUTPUT_DIR, filename)}")

def plot_momentum(times, momentum, dt, N, filename):
    """Plots total momentum components vs. time."""
    plt.figure(figsize=(10, 6))
    plt.plot(times, momentum[:, 0], label='Total Momentum X')
    plt.plot(times, momentum[:, 1], label='Total Momentum Y')
    plt.xlabel("Time (reduced units)")
    plt.ylabel("Momentum (reduced units)")
    plt.title(f"Total Momentum Conservation Check (N={N}, dt={dt})")
    # Set y-limits close to zero to see deviations
    max_mom = np.max(np.abs(momentum))
    if max_mom < 1e-10: max_mom = 1e-10 # Avoid zero limits
    plt.ylim(-max_mom * 1.5, max_mom * 1.5)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved momentum plot to {os.path.join(OUTPUT_DIR, filename)}")

def plot_rdf(rdf, bin_centers, N, T, filename):
    """Plots the radial distribution function."""
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, rdf, marker='o', linestyle='-', markersize=4)
    plt.xlabel("Distance r (reduced units, $\\sigma=1$)")
    plt.ylabel("g(r)")
    plt.title(f"Radial Distribution Function (N={N}, T={T})")
    plt.axhline(1.0, color='grey', linestyle='--', label='Ideal Gas g(r)=1')
    plt.grid(True)
    plt.xlim(left=0, right=max(bin_centers[-1], RC * 1.5)) # Show beyond rc
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved RDF plot to {os.path.join(OUTPUT_DIR, filename)}")

def plot_particle_distribution(positions, L, N, T, step, filename):
    """Plots the particle positions."""
    plt.figure(figsize=(8, 8))
    plt.scatter(positions[:, 0], positions[:, 1], s=10)
    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.xlabel("x (reduced units)")
    plt.ylabel("y (reduced units)")
    plt.title(f"Particle Distribution (N={N}, T={T}, Step={step})")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved particle distribution plot to {os.path.join(OUTPUT_DIR, filename)}")


# --- Main Simulation Function ---

def run_simulation(N, T_initial, n_steps, dt, use_thermostat, T_target=None, tau_T=TAU_T,
                   equilibration_steps=1000, rdf_sampling_freq=100, plot_freq=500,
                   rdf_max_dist=None, rdf_bin_size=0.05):
    """Runs the Lennard-Jones simulation."""

    print(f"\n--- Starting Simulation: N={N}, T_initial={T_initial:.2f}, dt={dt}, Thermostat={use_thermostat} ({'T_target='+str(T_target) if use_thermostat else 'NVE'}) ---")
    start_time = time.time()

    # Initialization
    positions, N = initialize_positions(N, L) # N might be adjusted
    velocities = initialize_velocities(N, T_initial, MASS, KB)

    if rdf_max_dist is None:
        rdf_max_dist = L / 2.0 # Sensible default max distance for RDF

    # Calculate initial forces and energy
    head, linked_list, cell_map, n_cells_dim, cell_size = build_cell_list(positions, L, RC)
    forces, potential_energy = calculate_forces_and_potential_cl(positions, L, RC_SQ, EPSILON, head, linked_list, cell_map, n_cells_dim)
    kinetic_energy = calculate_kinetic_energy(velocities, MASS)
    total_energy = kinetic_energy + potential_energy
    temperature = calculate_temperature(kinetic_energy, N, KB)
    total_momentum = calculate_total_momentum(velocities, MASS)

    # Data storage
    times = np.zeros(n_steps // plot_freq + 1)
    E_kin_data = np.zeros_like(times)
    E_pot_data = np.zeros_like(times)
    E_tot_data = np.zeros_like(times)
    T_data = np.zeros_like(times)
    P_data = np.zeros((times.shape[0], 2)) # Store momentum vector

    # RDF data
    rdf_hist = np.zeros(int(np.ceil(rdf_max_dist / rdf_bin_size)), dtype=np.int64)
    rdf_samples = 0

    # Store initial state
    times[0] = 0.0
    E_kin_data[0] = kinetic_energy
    E_pot_data[0] = potential_energy
    E_tot_data[0] = total_energy
    T_data[0] = temperature
    P_data[0] = total_momentum
    plot_idx = 1

    # --- Simulation Loop ---
    for step in range(1, n_steps + 1):
        # 1. Velocity Verlet Step 1 (Update positions, half-update velocities)
        positions, velocities = velocity_verlet_step1(positions, velocities, forces, dt, MASS, L)

        # 2. Calculate forces at new positions using Cell Lists
        head, linked_list, cell_map, n_cells_dim, _ = build_cell_list(positions, L, RC)
        forces, potential_energy = calculate_forces_and_potential_cl(positions, L, RC_SQ, EPSILON, head, linked_list, cell_map, n_cells_dim)

        # 3. Velocity Verlet Step 2 (Finish velocity update)
        velocities = velocity_verlet_step2(velocities, forces, dt, MASS)

        # 4. Apply Thermostat (if enabled)
        kinetic_energy = calculate_kinetic_energy(velocities, MASS)
        temperature = calculate_temperature(kinetic_energy, N, KB)
        if use_thermostat and T_target is not None:
            velocities = apply_berendsen_thermostat(velocities, temperature, T_target, dt, tau_T, N, KB)
            # Recalculate KE and T after thermostatting for accurate recording
            kinetic_energy = calculate_kinetic_energy(velocities, MASS)
            temperature = calculate_temperature(kinetic_energy, N, KB)

        # 5. Calculate total energy and momentum
        total_energy = kinetic_energy + potential_energy
        total_momentum = calculate_total_momentum(velocities, MASS)

        # 6. Data Recording & Analysis
        if step % plot_freq == 0:
            current_time = step * dt
            if plot_idx < len(times):
                times[plot_idx] = current_time
                E_kin_data[plot_idx] = kinetic_energy
                E_pot_data[plot_idx] = potential_energy
                E_tot_data[plot_idx] = total_energy
                T_data[plot_idx] = temperature
                P_data[plot_idx] = total_momentum
                plot_idx += 1

        # 7. RDF Calculation (after equilibration)
        if step > equilibration_steps and step % rdf_sampling_freq == 0:
             # Use the simpler RDF function for now
             rdf_step, _ = calculate_rdf(positions, L, N, RC, rdf_bin_size, rdf_max_dist)
             # Accumulate the histogram counts (more robust than averaging g(r) directly)
             # Need to modify calculate_rdf to return hist if accumulating this way.
             # Simpler: Calculate g(r) each time and average g(r).
             if rdf_samples == 0:
                 rdf_avg = rdf_step
             else:
                 rdf_avg = (rdf_avg * rdf_samples + rdf_step) / (rdf_samples + 1)
             rdf_samples += 1
             # Note: For proper averaging, accumulate the histogram counts (hist from calculate_rdf)
             # and normalize *once* at the end. The current averaging of g(r) is an approximation.

        # Print progress
        if step % (n_steps // 10) == 0:
             print(f"Step {step}/{n_steps} | Time: {step*dt:.2f} | T: {temperature:.4f} | E_tot: {total_energy:.4f} | Mom: {total_momentum}")


    # --- Post-Simulation Analysis & Plotting ---
    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

    # Trim data arrays if simulation stopped early or plot_freq didn't align perfectly
    times = times[:plot_idx]
    E_kin_data = E_kin_data[:plot_idx]
    E_pot_data = E_pot_data[:plot_idx]
    E_tot_data = E_tot_data[:plot_idx]
    T_data = T_data[:plot_idx]
    P_data = P_data[:plot_idx]

    # Generate Plots
    sim_id = f"N{N}_Tinit{T_initial:.1f}_dt{dt}"
    if use_thermostat:
        sim_id += f"_NVT_T{T_target:.1f}"
    else:
        sim_id += "_NVE"

    plot_energies(times, E_kin_data, E_pot_data, E_tot_data, dt, N, f"energy_{sim_id}.png")
    plot_temperature(times, T_data, T_target if use_thermostat else None, dt, N, use_thermostat, f"temperature_{sim_id}.png")
    if not use_thermostat: # Only check momentum conservation strictly in NVE
        plot_momentum(times, P_data, dt, N, f"momentum_{sim_id}.png")
    plot_particle_distribution(positions, L, N, T_data[-1], n_steps, f"distribution_{sim_id}_final.png")

    # Plot RDF if sampled
    if rdf_samples > 0:
        # Final normalization of RDF (if hist was accumulated) would happen here
        # Using the averaged g(r) for now:
        _, bin_centers = calculate_rdf(positions, L, N, RC, rdf_bin_size, rdf_max_dist) # Get bin centers
        plot_rdf(rdf_avg, bin_centers, N, T_target if use_thermostat else T_data[-1], f"rdf_{sim_id}.png")
    else:
        print("No RDF data collected (simulation might be too short or equilibration steps too long).")

    # Equilibration estimation
    # Find rough equilibration time by looking at when kinetic/potential energy stabilize
    try:
        # Look for stabilization in potential energy (often clearer than kinetic)
        mid_point = len(E_pot_data) // 2
        if mid_point > 10: # Need enough data
            mean_later = np.mean(E_pot_data[mid_point:])
            std_later = np.std(E_pot_data[mid_point:])
            # Find first point after start that's within, say, 2 std dev of the later mean
            equil_idx = np.argmax( (np.abs(E_pot_data[10:] - mean_later) < 2*std_later) & (times[10:] > 0) ) + 10
            equil_time_est = times[equil_idx] if equil_idx < len(times) else times[-1]
            print(f"Estimated equilibration time (rough): ~{equil_time_est:.2f} reduced time units (based on Potential Energy)")
        else:
            print("Not enough data points to estimate equilibration time reliably.")
    except Exception as e:
        print(f"Could not estimate equilibration time: {e}")

    # Energy fluctuations vs dt (for NVE)
    if not use_thermostat:
        energy_fluctuation = np.std(E_tot_data[len(E_tot_data)//2:]) # Fluctuations after potential equilibration
        print(f"Standard deviation of Total Energy (after equilibration): {energy_fluctuation:.6g}")
        print(f"Relative energy drift: {(E_tot_data[-1] - E_tot_data[0])/E_tot_data[0] if E_tot_data[0] != 0 else 'N/A'}")

    print(f"--- Simulation Complete: {sim_id} ---")
    return positions, velocities # Return final state if needed


# --- Running Simulations as per Homework ---

if __name__ == "__main__":

    print("========= Part (a): NVE Simulations =========")
    T_init_a = 0.5 # A reasonable starting temperature
    N_values_a = [100, 400, 900] # Test different N (400 is a perfect square)
    dt_values_a = [0.01, 0.005, 0.001] # Test different time steps
    n_steps_a = 10000 # Number of steps for NVE analysis

    for N in N_values_a:
        # Check dt impact (using one N value)
        if N == 400: # Use a mid-size system for dt checks
             for dt in dt_values_a:
                 run_simulation(N=N, T_initial=T_init_a, n_steps=n_steps_a, dt=dt,
                                use_thermostat=False, T_target=None,
                                equilibration_steps=n_steps_a // 5, # Estimate equilibration within first 20%
                                plot_freq=n_steps_a // 50) # Plot 50 points

        # Run with default dt for other N values
        dt_default = DT_DEFAULT
        run_simulation(N=N, T_initial=T_init_a, n_steps=n_steps_a, dt=dt_default,
                       use_thermostat=False, T_target=None,
                       equilibration_steps=n_steps_a // 5,
                       plot_freq=n_steps_a // 50)

    print("\n========= Part (b): NVT Simulations =========")
    n_steps_b = 20000 # Longer simulation for NVT equilibration and RDF
    equilibration_b = 5000 # Allow sufficient time for equilibration before RDF
    dt_b = DT_DEFAULT  # Use a stable dt found in part (a)

    nvt_cases = [
        {"N": 100, "T": 0.1},
        {"N": 100, "T": 1.0},
        {"N": 625, "T": 1.0}, # 625 is a perfect square (25x25)
        {"N": 900, "T": 1.0}, # 900 is a perfect square (30x30)
    ]

    for case in nvt_cases:
        N = case["N"]
        T = case["T"]
        # Start near the target temperature for faster equilibration
        run_simulation(N=N, T_initial=T, n_steps=n_steps_b, dt=dt_b,
                       use_thermostat=True, T_target=T, tau_T=TAU_T,
                       equilibration_steps=equilibration_b,
                       rdf_sampling_freq=50,  # Sample RDF frequently after equilibration
                       plot_freq=n_steps_b // 100) # Plot 100 points

    print("\nAll simulations finished. Check the '{}' directory for plots.".format(OUTPUT_DIR))