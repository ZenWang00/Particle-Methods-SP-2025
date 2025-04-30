# preliminary_dt_experiment.py

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import Dpd  

dt_list = [0.001, 0.01, 0.1]
n_steps = 2000        
record_interval = 100 

data_dir = 'data/preliminary'
fig_dir  = 'figures/preliminary'
os.makedirs(fig_dir, exist_ok=True)

all_results = {}

for dt in dt_list:
    print(f"\n=== Running preliminary for dt = {dt} ===")
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    # Monkey-patch dt 和 sigma
    Dpd.dt    = dt
    Dpd.sigma = np.sqrt(2 * Dpd.gamma * Dpd.kT)
    
    Dpd.run_preliminary()
    
    arr = np.load(os.path.join(data_dir, 'preliminary.npz'))
    steps = arr['steps']
    temps = arr['temps']
    
    all_results[dt] = (steps, temps)

plt.figure(figsize=(8,6))
for dt, (steps, temps) in all_results.items():
    plt.plot(steps, temps, label=f'Δt = {dt}')
plt.xlabel('Step')
plt.ylabel('Temperature')
plt.title('Preliminary Test: Temperature vs Step for Various Δt')
plt.legend()
plt.grid(True)
out_png = os.path.join(fig_dir, 'temp_vs_step_dt_comparison.png')
plt.savefig(out_png, dpi=300)
print(f"\nSaved comparison plot to {out_png}")
