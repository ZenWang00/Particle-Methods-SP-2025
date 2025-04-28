#!/usr/bin/env python3
"""
This code implements a 2D Lennard-Jones simulation according to the assignment:
- Domain: a 30×30 square with periodic boundaries.
- Lennard-Jones potential: σ = 1, ε = 1, truncated (and shifted) at r_c = 2.5.
- Integration: Velocity-Verlet method.
- Acceleration: Using a cell list.
- Ensembles: Both NVE (no thermostat) and NVT (using a Berendsen thermostat, dt/τ = 0.0025) are implemented.
- RDF: Calculated with a bin width of dr = 0.05.

Test cases:
  (a) NVE: N = 100, 225, 400, 625, 900.
  (b) NVT: 
        - N = 100, T_target = 0.1 and T_target = 1.0;
        - N = 225, 625, 900, T_target = 1.0.
        
To help the system reach equilibrium (so that the RDF converges), a long pre-equilibration
phase (20,000 steps) and a production run (50,000 steps) are used. For each test case,
three separate figures (energy vs. time, temperature vs. time, and RDF) are automatically saved.
The energy data are displayed in three separate subplots in one figure.
All output files are saved to:
    /Users/zitian/Particle-Methods/homework3/simulation_results/
Summary statistics are printed to the console.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os

try:
    from numba import njit, prange
    USE_NUMBA = True
    print("Numba found, JIT-accelerated routines will be used.")
except ImportError:
    USE_NUMBA = False
    print("Numba not found, running in pure Python mode.")


L = 30.0               # Domain size (30x30)
rc = 2.5               # Cut-off radius
rc2 = rc * rc         # Square of cutoff
sigma = 1.0          # Lennard-Jones sigma (reduced units)
epsilon = 1.0        # Lennard-Jones epsilon (reduced units)
mass = 1.0           # Particle mass

cell_size = rc       # cell list cell size
n_cells = int(L / cell_size)
small_r2 = 1e-12     # threshold to avoid division by zero

# Output folder (absolute path)
output_folder = "/Users/zitian/Particle-Methods/homework3/simulation_results5"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def minimum_image(dx, L):
    """Apply the minimum image convention."""
    return dx - L * np.rint(dx / L)

if USE_NUMBA:
    @njit
    def min_image_numba(dx, L):
        return dx - L * np.rint(dx / L)
else:
    min_image_numba = minimum_image


def compute_forces(positions):
    N = positions.shape[0]
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    # Build cell list.
    cells = [[[] for _ in range(n_cells)] for _ in range(n_cells)]
    for i in range(N):
        ix = int(positions[i, 0] / cell_size) % n_cells
        iy = int(positions[i, 1] / cell_size) % n_cells
        cells[ix][iy].append(i)
    for ix in range(n_cells):
        for iy in range(n_cells):
            cell_particles = cells[ix][iy]
            for dix in (-1, 0, 1):
                for diy in (-1, 0, 1):
                    jx = (ix + dix) % n_cells
                    jy = (iy + diy) % n_cells
                    neighbor_particles = cells[jx][jy]
                    for i in cell_particles:
                        for j in neighbor_particles:
                            if (i < j) or (dix != 0 or diy != 0):
                                dx = positions[j, 0] - positions[i, 0]
                                dy = positions[j, 1] - positions[i, 1]
                                dx = minimum_image(dx, L)
                                dy = minimum_image(dy, L)
                                r2 = dx*dx + dy*dy
                                if r2 < small_r2:
                                    r2 = small_r2
                                if r2 < rc2:
                                    inv_r2 = 1.0 / r2
                                    inv_r6 = inv_r2**3
                                    inv_r12 = inv_r6**2
                                    U = 4.0 * epsilon * (inv_r12 - inv_r6)
                                    inv_rc2 = 1.0/(rc*rc)
                                    inv_rc6 = inv_rc2**3
                                    inv_rc12 = inv_rc6**2
                                    U_shift = 4.0 * epsilon * (inv_rc12 - inv_rc6)
                                    potential_energy += (U - U_shift)
                                    force_scalar = 48.0 * epsilon * (inv_r12 - 0.5*inv_r6) * inv_r2
                                    fx = force_scalar * dx
                                    fy = force_scalar * dy
                                    forces[i, 0] += fx
                                    forces[i, 1] += fy
                                    forces[j, 0] -= fx
                                    forces[j, 1] -= fy
    return forces, potential_energy


if USE_NUMBA:
    @njit
    def compute_forces_numba(positions):
        N = positions.shape[0]
        forces = np.zeros_like(positions)
        potential_energy = 0.0
        max_per_cell = 100
        cell_list = -np.ones((n_cells, n_cells, max_per_cell), dtype=np.int32)
        cell_counts = np.zeros((n_cells, n_cells), dtype=np.int32)
        for i in range(N):
            ix = int(positions[i, 0] / cell_size) % n_cells
            iy = int(positions[i, 1] / cell_size) % n_cells
            count = cell_counts[ix, iy]
            cell_list[ix, iy, count] = i
            cell_counts[ix, iy] += 1
        for ix in range(n_cells):
            for iy in range(n_cells):
                for a in range(cell_counts[ix, iy]):
                    i = cell_list[ix, iy, a]
                    for dix in (-1, 0, 1):
                        for diy in (-1, 0, 1):
                            jx = (ix + dix) % n_cells
                            jy = (iy + diy) % n_cells
                            for b in range(cell_counts[jx, jy]):
                                j = cell_list[jx, jy, b]
                                if (i < j) or (dix != 0 or diy != 0):
                                    dx = positions[j, 0] - positions[i, 0]
                                    dy = positions[j, 1] - positions[i, 1]
                                    dx = dx - L * np.rint(dx / L)
                                    dy = dy - L * np.rint(dy / L)
                                    r2 = dx*dx + dy*dy
                                    if r2 < small_r2:
                                        r2 = small_r2
                                    if r2 < rc*rc:
                                        inv_r2 = 1.0 / r2
                                        inv_r6 = inv_r2**3
                                        inv_r12 = inv_r6**2
                                        U = 4.0 * epsilon * (inv_r12 - inv_r6)
                                        inv_rc2 = 1.0 / (rc*rc)
                                        inv_rc6 = inv_rc2**3
                                        inv_rc12 = inv_rc6**2
                                        U_shift = 4.0 * epsilon * (inv_rc12 - inv_rc6)
                                        potential_energy += (U - U_shift)
                                        force_scalar = 48.0 * epsilon * (inv_r12 - 0.5*inv_r6) * inv_r2
                                        fx = force_scalar * dx
                                        fy = force_scalar * dy
                                        forces[i, 0] += fx
                                        forces[i, 1] += fy
                                        forces[j, 0] -= fx
                                        forces[j, 1] -= fy
        return forces, potential_energy
    force_function = compute_forces_numba
else:
    force_function = compute_forces

def compute_rdf(positions, dr=0.05, r_max=None):
    N = positions.shape[0]
    if r_max is None:
        r_max = L/2.0
    nbins = int(r_max/dr)
    rdf_hist = np.zeros(nbins)
    for i in range(N-1):
        for j in range(i+1, N):
            dx = positions[j,0] - positions[i,0]
            dy = positions[j,1] - positions[i,1]
            dx = minimum_image(dx, L)
            dy = minimum_image(dy, L)
            r = np.sqrt(dx*dx+dy*dy)
            if r < r_max:
                bin_index = int(r/dr)
                rdf_hist[bin_index] += 2
    rho = N/(L*L)
    r_values = (np.arange(nbins)+0.5)*dr
    shell_areas = 2.0*np.pi*r_values*dr
    ideal_counts = shell_areas*rho*N
    g_r = rdf_hist/ideal_counts
    return r_values, g_r


class LJSimulation:
    def __init__(self, N, dt=1e-4, nsteps=50000, ensemble='NVE', T_target=1.0, pre_equil_steps=20000):
        """
        Parameters:
            N: Number of particles.
            dt: Time step.
            nsteps: Production steps (observables recorded during these steps).
            ensemble: 'NVE' or 'NVT'.
            T_target: Target temperature (for NVT).
            pre_equil_steps: Pre-equilibration steps.
        """
        self.N = N
        self.dt = dt
        self.nsteps = nsteps
        self.ensemble = ensemble
        self.T_target = T_target
        self.pre_equil_steps = pre_equil_steps
        
        self.thermo_factor = dt / (dt / 0.0025)  # Berendsen thermostat parameter
        
        self.positions = self.initialize_positions()
        self.velocities = self.initialize_velocities(initial_T=T_target if ensemble=='NVT' else 0.5)
        self.forces, self.potential_energy = force_function(self.positions)
        
        self.kinetic_energies = []
        self.potential_energies = []
        self.total_energies = []
        self.temperatures = []
    
    def initialize_positions(self):
        N_side = int(np.ceil(np.sqrt(self.N)))
        grid = np.linspace(0, L, N_side, endpoint=False)
        pos = []
        for x in grid:
            for y in grid:
                if len(pos) < self.N:
                    perturb = (L/N_side)*0.01*np.random.randn(2)
                    pos.append([x + L/(2*N_side) + perturb[0], y + L/(2*N_side) + perturb[1]])
        return np.array(pos)
    
    def initialize_velocities(self, initial_T):
        velocities = np.random.randn(self.N, 2)
        com = np.mean(velocities, axis=0)
        velocities -= com
        current_T = np.mean(np.sum(velocities**2, axis=1)) / 2.0
        scaling = np.sqrt(initial_T/current_T)
        velocities *= scaling
        return velocities
    
    def apply_periodic(self, positions):
        return positions % L
    
    def current_temperature(self):
        kinetic = 0.5*np.sum(self.velocities**2)
        T_inst = kinetic/self.N
        return T_inst, kinetic
    
    def run(self):
        dt = self.dt
        print(f"Pre-equilibration for {self.pre_equil_steps} steps (N={self.N}, Ensemble={self.ensemble}, Target T={'N/A' if self.ensemble=='NVE' else self.T_target})...")
        for _ in range(self.pre_equil_steps):
            self.positions += self.velocities * dt + 0.5 * self.forces/mass * dt**2
            self.positions = self.apply_periodic(self.positions)
            forces_old = self.forces.copy()
            self.forces, _ = force_function(self.positions)
            self.velocities += 0.5*(forces_old+self.forces)/mass * dt
            if self.ensemble == 'NVT':
                T_inst, _ = self.current_temperature()
                lambda_factor = np.sqrt(1+0.0025*(self.T_target/T_inst-1))
                self.velocities *= lambda_factor
        print("Pre-equilibration complete. Production run begins...")
        t_start = time.time()
        for step in range(self.nsteps):
            self.positions += self.velocities * dt + 0.5 * self.forces/mass * dt**2
            self.positions = self.apply_periodic(self.positions)
            forces_old = self.forces.copy()
            self.forces, pot_energy = force_function(self.positions)
            self.potential_energy = pot_energy
            self.velocities += 0.5*(forces_old+self.forces)/mass * dt
            if self.ensemble == 'NVT':
                T_inst, _ = self.current_temperature()
                lambda_factor = np.sqrt(1+0.0025*(self.T_target/T_inst-1))
                self.velocities *= lambda_factor
            T_inst, kin_energy = self.current_temperature()
            tot_energy = kin_energy + self.potential_energy
            self.temperatures.append(T_inst)
            self.kinetic_energies.append(kin_energy)
            self.potential_energies.append(self.potential_energy)
            self.total_energies.append(tot_energy)
            if step % (self.nsteps//10) == 0:
                print(f"Step {step}/{self.nsteps}: T = {T_inst:.3e}, E_tot = {tot_energy:.3e}")
        t_end = time.time()
        print("Production run complete in {:.2f} seconds.".format(t_end-t_start))
        avg_temp = np.mean(self.temperatures)
        std_temp = np.std(self.temperatures)
        avg_energy = np.mean(self.total_energies)
        std_energy = np.std(self.total_energies)
        print(f"Average Temperature: {avg_temp:.3e}, STD: {std_temp:.3e}")
        print(f"Average Total Energy: {avg_energy:.3e}, STD: {std_energy:.3e}")
    
    def run_extension(self, additional_steps):
        dt = self.dt
        for _ in range(additional_steps):
            self.positions += self.velocities * dt + 0.5 * self.forces/mass * dt**2
            self.positions = self.apply_periodic(self.positions)
            forces_old = self.forces.copy()
            self.forces, pot_energy = force_function(self.positions)
            self.potential_energy = pot_energy
            self.velocities += 0.5*(forces_old+self.forces)/mass * dt
            if self.ensemble == 'NVT':
                T_inst, _ = self.current_temperature()
                lambda_factor = np.sqrt(1+0.0025*(self.T_target/T_inst-1))
                self.velocities *= lambda_factor
            T_inst, kin_energy = self.current_temperature()
            tot_energy = kin_energy + self.potential_energy
            self.temperatures.append(T_inst)
            self.kinetic_energies.append(kin_energy)
            self.potential_energies.append(self.potential_energy)
            self.total_energies.append(tot_energy)
    
    def plot_energies(self, save_fig=True, filename=None):
        steps = np.arange(self.nsteps)
        # Create 3 separate subplots for each energy type.
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8,10))
        axs[0].plot(steps, self.kinetic_energies, color='navy')
        axs[0].set_ylabel("Kinetic Energy")
        axs[0].set_title("Kinetic Energy vs Time")
        axs[1].plot(steps, self.potential_energies, color='darkred')
        axs[1].set_ylabel("Potential Energy")
        axs[1].set_title("Potential Energy vs Time")
        axs[2].plot(steps, self.total_energies, color='darkgreen', linewidth=2)
        axs[2].set_xlabel("Time step")
        axs[2].set_ylabel("Total Energy")
        axs[2].set_title("Total Energy vs Time")
        fig.suptitle(f"Energy Components vs Time, N={self.N}, Ensemble={self.ensemble}")
        fig.tight_layout(rect=[0,0,1,0.96])
        if save_fig:
            extra = f"_T={self.T_target}" if self.ensemble=="NVT" else ""
            if filename is None:
                filename = f"{output_folder}/EnergyComponents_N={self.N}_Ensemble={self.ensemble}{extra}.png"
            plt.savefig(filename)
            plt.close(fig)
        else:
            plt.show()
    
    def plot_temperature(self, save_fig=True, filename=None):
        steps = np.arange(self.nsteps)
        fig = plt.figure(figsize=(8,6))
        plt.plot(steps, self.temperatures, label="Temperature", color='magenta')
        plt.xlabel("Time step")
        plt.ylabel("Temperature")
        plt.title(f"Temperature vs Time (N={self.N}, Ensemble={self.ensemble})")
        plt.legend()
        if save_fig:
            extra = f"_T={self.T_target}" if self.ensemble=="NVT" else ""
            if filename is None:
                filename = f"{output_folder}/Temperature_N={self.N}_Ensemble={self.ensemble}{extra}.png"
            plt.savefig(filename)
            plt.close(fig)
        else:
            plt.show()
    
    def plot_rdf(self, save_fig=True, filename=None):
        r_vals, g_r = compute_rdf(self.positions)
        fig = plt.figure(figsize=(8,6))
        plt.plot(r_vals, g_r, '-', color='deepskyblue', alpha=0.6, label="g(r) line")
        plt.scatter(r_vals, g_r, s=5, color='deepskyblue', alpha=0.4, label="g(r) points")
        plt.xlabel("r")
        plt.ylabel("g(r)")
        plt.title(f"RDF (N={self.N}, Ensemble={self.ensemble})")
        plt.xlim(0, np.max(r_vals))
        plt.yscale('log')
        plt.legend()
        if save_fig:
            extra = f"_T={self.T_target}" if self.ensemble=="NVT" else ""
            if filename is None:
                filename = f"{output_folder}/RDF_N={self.N}_Ensemble={self.ensemble}{extra}.png"
            plt.savefig(filename)
            plt.close(fig)
        else:
            plt.show()

def run_until_equilibrium(sim, window_size=5000, tolerance=0.01, max_extensions=10):
    ext_count = 0
    prev_avg = None
    while ext_count < max_extensions:
        if len(sim.temperatures) < 2 * window_size:
            print("Not enough data for equilibrium check, extending production run...")
            sim.run_extension(window_size)
            ext_count += 1
            continue
        current_window = sim.temperatures[-window_size:]
        new_avg = np.mean(current_window)
        if prev_avg is None:
            prev_avg = new_avg
        rel_change = abs(new_avg - prev_avg) / abs(new_avg) if new_avg != 0 else 0
        print(f"Extension {ext_count}: New average temperature = {new_avg:.3e}, relative change = {rel_change:.3e}")
        if rel_change < tolerance:
            print("Equilibrium reached based on temperature stability criterion.")
            break
        else:
            prev_avg = new_avg
            sim.run_extension(window_size)
            ext_count += 1
    if ext_count == max_extensions:
        print("Maximum extensions reached. The system may still not be fully equilibrated.")

def run_test_case(tc):
    print("\n--------------------------------------------------")
    msg = f"Running test case: N = {tc['N']}, Ensemble = {tc['ensemble']}"
    if tc['ensemble'] == 'NVT':
        msg += f", T_target = {tc['T_target']}"
    print(msg)
    
    sim = LJSimulation(
        N=tc["N"],
        dt=1e-4,
        nsteps=tc.get("nsteps", 50000),
        ensemble=tc["ensemble"],
        T_target=tc.get("T_target", 0.5),
        pre_equil_steps=tc.get("pre_equil_steps", 20000)
    )
    sim.run()
    if sim.ensemble == "NVT":
        run_until_equilibrium(sim, window_size=5000, tolerance=0.01, max_extensions=10)
    
    sim.plot_energies()
    sim.plot_temperature()
    sim.plot_rdf()
    
    print(f"Test case (N={tc['N']}, Ensemble={tc['ensemble']}"
          f"{', T='+str(tc['T_target']) if tc['ensemble']=='NVT' else ''}) completed.\n")

def main():
    test_cases = [
        # NVE cases: N = 100, 225, 400, 625, 900.
        {"N": 100, "ensemble": "NVE"},
        {"N": 225, "ensemble": "NVE"},
        {"N": 400, "ensemble": "NVE"},
        {"N": 625, "ensemble": "NVE"},
        {"N": 900, "ensemble": "NVE"},
        # NVT cases:
        {"N": 100, "ensemble": "NVT", "T_target": 0.1},
        {"N": 100, "ensemble": "NVT", "T_target": 1.0},
        {"N": 225, "ensemble": "NVT", "T_target": 1.0},
        {"N": 625, "ensemble": "NVT", "T_target": 1.0},
        {"N": 900, "ensemble": "NVT", "T_target": 1.0},
    ]
    for tc in test_cases:
        run_test_case(tc)
    print("All test cases completed. Please check the output folder:")
    print(output_folder)

if __name__ == "__main__":
    main()
