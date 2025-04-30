# plot.py
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# ==== Clear figures directory ====
fig_dir = 'figures'
if os.path.exists(fig_dir):
    shutil.rmtree(fig_dir)
os.makedirs(os.path.join(fig_dir, 'preliminary'), exist_ok=True)
os.makedirs(os.path.join(fig_dir, 'couette'), exist_ok=True)
os.makedirs(os.path.join(fig_dir, 'poiseuille'), exist_ok=True)

# ==== Preliminary Visualization ====
prelim = np.load('data/preliminary/preliminary.npz')
vel = prelim['vel']         # shape (N,2)
steps_p = prelim['steps']   # shape (M,)
temps_p = prelim['temps']   # shape (M,)

# 1. Temperature vs Step (omit <1000)
mask = steps_p >= 0
plt.figure()
plt.plot(steps_p[mask], temps_p[mask], 'o-')
plt.xlabel('Step')
plt.ylabel('Temperature')
plt.title('Preliminary: Temperature vs Step (from step 1000)')
plt.grid(True)
plt.savefig(os.path.join(fig_dir, 'preliminary', 'temperature_vs_step.png'))
plt.close()

# 2. Speed Distribution
speeds = np.linalg.norm(vel, axis=1)
plt.figure()
plt.hist(speeds, bins=50, density=True, edgecolor='black')
plt.xlabel('Speed')
plt.ylabel('Probability Density')
plt.title('Preliminary: Speed Distribution')
plt.grid(True)
plt.savefig(os.path.join(fig_dir, 'preliminary', 'speed_distribution.png'))
plt.close()

# ==== Couette Visualization ====
coup = np.load('data/couette/couette.npz')
yedges_c = coup['yedges']
steps_c = coup['steps']
temps_c = coup['temps']
vprofs = coup['vprofs']  # shape (M, nb)
e2es = coup['e2es']      # shape (M, n_chains)
ycenters_c = 0.5 * (yedges_c[:-1] + yedges_c[1:])

# 1. Temperature vs Step
plt.figure()
plt.plot(steps_c, temps_c, 's-')
plt.xlabel('Step')
plt.ylabel('Temperature')
plt.title('Couette: Temperature vs Step')
plt.grid(True)
plt.savefig(os.path.join(fig_dir, 'couette', 'temperature_vs_step.png'))
plt.close()

# 2. Velocity Profiles (symlog scale)
plt.figure()
for i, step in enumerate(steps_c):
    plt.plot(vprofs[i], ycenters_c, label=f'{step}', alpha=0.7)
plt.xscale('symlog', linthresh=0.1)
plt.xlabel('⟨v_x⟩ (symlog)')
plt.ylabel('y')
plt.title('Couette: Velocity Profiles (symlog)')
plt.legend(fontsize='small', ncol=2)
plt.grid(True)
plt.savefig(os.path.join(fig_dir, 'couette', 'vprof_symlog.png'))
plt.close()

# 3. Velocity Evolution Heatmap
plt.figure()
plt.imshow(vprofs.T, origin='lower', aspect='auto',
           extent=[steps_c[0], steps_c[-1], ycenters_c[0], ycenters_c[-1]],
           cmap='viridis')
plt.colorbar(label='⟨v_x⟩')
plt.xlabel('Step')
plt.ylabel('y')
plt.title('Couette: Velocity Evolution Heatmap')
plt.savefig(os.path.join(fig_dir, 'couette', 'vprof_heatmap.png'))
plt.close()

# 4. e2e Box Plot (skip step 0)
mask_e = [i for i, s in enumerate(steps_c) if s >= 1000]
plt.figure()
plt.boxplot([e2es[i] for i in mask_e], labels=[str(steps_c[i]) for i in mask_e], sym='')
plt.xlabel('Step')
plt.ylabel('End-to-End Distance')
plt.title('Couette: Chain Extension (e2e) (from step 1000)')
plt.grid(True)
plt.savefig(os.path.join(fig_dir, 'couette', 'e2e_boxplot.png'))
plt.close()

# ==== Poiseuille Visualization ====
pos = np.load('data/poiseuille/poiseuille.npz')
yedges_o = pos['yedges']
steps_o = pos['steps']
temps_o = pos['temps']
mean_vx_o = pos['mean_vx']
vprofs_o = pos['vprofs']    # shape (M, nb)
concs_o = pos['concs']      # shape (M, nb)
Rgs_o = pos['Rgs']          # shape (M, n_rings)
ycenters_o = 0.5 * (yedges_o[:-1] + yedges_o[1:])

# 1. Temperature & Mean v_x vs Step
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(steps_o, temps_o, 'r-o', label='Temperature')
ax2.plot(steps_o, mean_vx_o, 'b--s', label='Mean v_x')
ax1.set_xlabel('Step')
ax1.set_ylabel('Temperature', color='r')
ax2.set_ylabel('Mean v_x', color='b')
plt.title('Poiseuille: Temperature & Mean v_x vs Step')
ax1.grid(True)
fig.savefig(os.path.join(fig_dir, 'poiseuille', 'temp_meanvx_vs_step.png'))
plt.close()

# 2. Velocity Profiles (symlog scale)
plt.figure()
for i, step in enumerate(steps_o):
    plt.plot(vprofs_o[i], ycenters_o, label=f'{step}', alpha=0.7)
plt.xscale('symlog', linthresh=0.1)
plt.xlabel('⟨v_x⟩ (symlog)')
plt.ylabel('y')
plt.title('Poiseuille: Velocity Profiles (symlog)')
plt.legend(fontsize='small', ncol=2)
plt.grid(True)
plt.savefig(os.path.join(fig_dir, 'poiseuille', 'vprof_symlog.png'))
plt.close()

# 3. Velocity Evolution Heatmap
plt.figure()
plt.imshow(vprofs_o.T, origin='lower', aspect='auto',
           extent=[steps_o[0], steps_o[-1], ycenters_o[0], ycenters_o[-1]],
           cmap='plasma')
plt.colorbar(label='⟨v_x⟩')
plt.xlabel('Step')
plt.ylabel('y')
plt.title('Poiseuille: Velocity Evolution Heatmap')
plt.savefig(os.path.join(fig_dir, 'poiseuille', 'vprof_heatmap.png'))
plt.close()

# 4. Concentration Heatmap
plt.figure()
plt.imshow(concs_o.T, origin='lower', aspect='auto',
           extent=[steps_o[0], steps_o[-1], ycenters_o[0], ycenters_o[-1]],
           cmap='magma')
plt.colorbar(label='Concentration')
plt.xlabel('Step')
plt.ylabel('y')
plt.title('Poiseuille: Concentration Heatmap')
plt.savefig(os.path.join(fig_dir, 'poiseuille', 'conc_heatmap.png'))
plt.close()

# 5. Rg Violin Plot
plt.figure()
plt.violinplot([Rgs_o[i] for i in range(len(steps_o))], showmeans=True)
plt.xticks(np.arange(1, len(steps_o)+1), [str(s) for s in steps_o])
plt.xlabel('Step')
plt.ylabel('Radius of Gyration (Rg)')
plt.title('Poiseuille: Rg Distribution Over Time')
plt.grid(True)
plt.savefig(os.path.join(fig_dir, 'poiseuille', 'Rg_violinplot.png'))
plt.close()

print('Plots generated in figures/ directory.')
