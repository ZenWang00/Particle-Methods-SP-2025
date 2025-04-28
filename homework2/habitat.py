import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Global parameters
prr = 0.02      # Rabbit reproduction probability
pwe = 0.02      # Wolf eating probability
rc = 0.5        # Wolf predation radius
pwr = 0.02      # Wolf replication probability per eaten rabbit
twd = 50        # Wolf starvation threshold (steps)
initial_rabbits = 900
initial_wolves = 100

def apply_periodic(arr, L):
    return arr % L

def simulate_vectorized(sim_steps, L, sigma, trd):
    # Rabbits: columns [x, y, age]
    rabbits = np.empty((initial_rabbits, 3))
    rabbits[:, 0] = np.random.uniform(0, L, initial_rabbits)
    rabbits[:, 1] = np.random.uniform(0, L, initial_rabbits)
    rabbits[:, 2] = np.random.randint(1, trd, initial_rabbits)
    
    # Wolves: columns [x, y, hunger]
    wolves = np.empty((initial_wolves, 3))
    wolves[:, 0] = np.random.uniform(0, L, initial_wolves)
    wolves[:, 1] = np.random.uniform(0, L, initial_wolves)
    wolves[:, 2] = 0

    rabbit_counts = []
    wolf_counts = []

    for t in range(sim_steps):
        # --- Update rabbits: movement, aging, reproduction, and death ---
        if rabbits.shape[0] > 0:
            n_rabbits = rabbits.shape[0]
            rabbits[:, 0] = apply_periodic(rabbits[:, 0] + np.random.normal(0, sigma, n_rabbits), L)
            rabbits[:, 1] = apply_periodic(rabbits[:, 1] + np.random.normal(0, sigma, n_rabbits), L)
            rabbits[:, 2] += 1
            # Reproduction: add new rabbits with age 0 at same position
            repro_mask = np.random.rand(n_rabbits) < prr
            new_rabbits = np.column_stack((rabbits[repro_mask, 0],
                                            rabbits[repro_mask, 1],
                                            np.zeros(np.sum(repro_mask))))
            rabbits = np.vstack((rabbits, new_rabbits))
            # Remove rabbits exceeding lifetime
            rabbits = rabbits[rabbits[:, 2] < trd]
        else:
            rabbits = np.empty((0, 3))
        
        # --- Update wolves: movement, predation, replication, and starvation ---
        n_wolves = wolves.shape[0]
        if n_wolves > 0:
            wolves[:, 0] = apply_periodic(wolves[:, 0] + np.random.normal(0, sigma, n_wolves), L)
            wolves[:, 1] = apply_periodic(wolves[:, 1] + np.random.normal(0, sigma, n_wolves), L)
        new_wolves = []
        for i in range(n_wolves):
            if rabbits.shape[0] == 0:
                wolves[i, 2] += 1
                new_wolves.append(wolves[i])
                continue
            # Compute distances using periodic boundaries
            dx = np.abs(rabbits[:, 0] - wolves[i, 0])
            dx = np.minimum(dx, L - dx)
            dy = np.abs(rabbits[:, 1] - wolves[i, 1])
            dy = np.minimum(dy, L - dy)
            dist = np.sqrt(dx**2 + dy**2)
            # Find rabbits in predation radius
            indices = np.where(dist < rc)[0]
            ate = False
            if indices.size > 0:
                eat_mask = np.random.rand(indices.size) < pwe
                eaten_indices = indices[eat_mask]
                if eaten_indices.size > 0:
                    ate = True
                    rep_mask = np.random.rand(eaten_indices.size) < pwr
                    num_new = np.sum(rep_mask)
                    if num_new > 0:
                        new_wolves.extend([[wolves[i, 0], wolves[i, 1], 0]] * int(num_new))
                    rabbits = np.delete(rabbits, eaten_indices, axis=0)
            wolves[i, 2] = 0 if ate else wolves[i, 2] + 1
            if wolves[i, 2] < twd:
                new_wolves.append(wolves[i])
        wolves = np.array(new_wolves) if new_wolves else np.empty((0, 3))

        rabbit_counts.append(rabbits.shape[0])
        wolf_counts.append(wolves.shape[0])
    return np.array(rabbit_counts), np.array(wolf_counts)

def lv_system(t, y, alpha, beta, delta, gamma):
    R, W = y
    return [alpha * R - beta * R * W, delta * R * W - gamma * W]

# Solve the Lotka-Volterra ODE
alpha, beta, delta, gamma = 0.05, 0.0001, 0.0001, 0.05
t_span = (0, 2000)
t_eval = np.linspace(t_span[0], t_span[1], 2001)
lv_sol = solve_ivp(lambda t, y: lv_system(t, y, alpha, beta, delta, gamma),
                   t_span, [initial_rabbits, initial_wolves], t_eval=t_eval)

sim_steps = 2000
# Case (a): sigma=0.5, trd=100, L=10
rabbits_a, wolves_a = simulate_vectorized(sim_steps, L=10, sigma=0.5, trd=100)
# Case (b): sigma=0.5, trd=50, L=10
rabbits_b, wolves_b = simulate_vectorized(sim_steps, L=10, sigma=0.5, trd=50)
# Case (c): sigma=0.05, trd=100, L=8
rabbits_c, wolves_c = simulate_vectorized(sim_steps, L=8, sigma=0.05, trd=100)

# Plot simulation results for three cases
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
time = np.arange(sim_steps)
axs[0].plot(time, rabbits_a, label="Rabbits (a)")
axs[0].plot(time, wolves_a, label="Wolves (a)")
axs[0].set_title("Case (a): sigma=0.5, trd=100, L=10")
axs[0].legend()

axs[1].plot(time, rabbits_b, label="Rabbits (b)")
axs[1].plot(time, wolves_b, label="Wolves (b)")
axs[1].set_title("Case (b): sigma=0.5, trd=50, L=10")
axs[1].legend()

axs[2].plot(time, rabbits_c, label="Rabbits (c)")
axs[2].plot(time, wolves_c, label="Wolves (c)")
axs[2].set_title("Case (c): sigma=0.05, trd=100, L=8")
axs[2].legend()

axs[2].set_xlabel("Time Step")
for ax in axs:
    ax.set_ylabel("Population")
plt.tight_layout()
plt.show()

# Plot LV model solution
plt.figure(figsize=(10, 5))
plt.plot(lv_sol.t, lv_sol.y[0], label="Rabbits (LV)")
plt.plot(lv_sol.t, lv_sol.y[1], label="Wolves (LV)")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Lotka–Volterra Model Solution")
plt.legend()
plt.show()
