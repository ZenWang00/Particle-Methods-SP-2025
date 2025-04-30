# dpd_simulation.py

import os
import glob
import numpy as np
from tqdm import trange
from numba import njit, prange
import logging

# ==== Clear old logs on each run ====
if os.path.exists('logs/simulation.log'):
    os.remove('logs/simulation.log')
for folder in ['logs/preliminary', 'logs/couette', 'logs/poiseuille']:
    if os.path.isdir(folder):
        for file in glob.glob(os.path.join(folder, '*.log')):
            os.remove(file)

# ==== Setup directories ====
os.makedirs('data/preliminary', exist_ok=True)
os.makedirs('data/couette', exist_ok=True)
os.makedirs('data/poiseuille', exist_ok=True)
os.makedirs('logs/preliminary', exist_ok=True)
os.makedirs('logs/couette', exist_ok=True)
os.makedirs('logs/poiseuille', exist_ok=True)

# ==== Logging Configuration ====
logger = logging.getLogger('dpd')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.FileHandler('logs/simulation.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# ==== Global Constants ====
L = 15.0           # Domain size
rc = 1.0           # Cutoff radius
rho = 4.0          # Fluid density
dt = 0.001         # Time step
mass = 1.0
gamma = 4.5
kT = 1.0
sigma = np.sqrt(2 * gamma * kT)  # Fluctuation-dissipation balance
cell_size = rc

# Particle types
TYPE_F, TYPE_W, TYPE_A, TYPE_B = 0, 1, 2, 3

# ==== Utility Functions ====
@njit
def apply_pbc(dx):
    if dx > 0.5*L: return dx - L
    if dx < -0.5*L: return dx + L
    return dx

@njit
def wR(r):
    return 1.0 - r/rc if r < rc else 0.0

@njit
def wD(r):
    wr = wR(r)
    return wr * wr

def build_cell_list(pos):
    n = int(np.floor(L / cell_size))
    head = -1 * np.ones((n, n), dtype=np.int32)
    linked = -1 * np.ones(pos.shape[0], dtype=np.int32)
    for i in range(pos.shape[0]):
        cx = int(pos[i,0] / cell_size) % n
        cy = int(pos[i,1] / cell_size) % n
        linked[i] = head[cx, cy]
        head[cx, cy] = i
    return head, linked, n

@njit
def integrate(pos, vel, acc):
    for i in range(pos.shape[0]):
        vel[i] += 0.5 * acc[i] * dt
        pos[i] += vel[i] * dt
        for d in range(2):
            if pos[i,d] >= L: pos[i,d] -= L
            if pos[i,d] < 0:  pos[i,d] += L

@njit
def compute_dpd_forces(pos, vel, acc, ptype, head, linked, n_cells, aij):
    N = pos.shape[0]
    acc[:,:] = 0.0
    sqrt_dt = np.sqrt(dt)
    for i in prange(N):
        xi = int(pos[i,0] / cell_size) % n_cells
        yi = int(pos[i,1] / cell_size) % n_cells
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cx = (xi + dx) % n_cells
                cy = (yi + dy) % n_cells
                j = head[cx, cy]
                while j != -1:
                    if j > i:
                        rij = pos[i] - pos[j]
                        rij[0] = apply_pbc(rij[0]); rij[1] = apply_pbc(rij[1])
                        r = np.sqrt(rij[0]**2 + rij[1]**2)
                        if r < rc:
                            eij = rij / r
                            vij = vel[i] - vel[j]
                            fc = aij[ptype[i], ptype[j]] * (1 - r/rc)
                            fd = -gamma * wD(r) * (vij[0]*eij[0] + vij[1]*eij[1])
                            # Random force
                            u1 = np.random.rand(); u2 = np.random.rand()
                            theta = np.sqrt(-2.0*np.log(u1)) * np.cos(2*np.pi*u2)
                            fr = sigma * wR(r) * theta / sqrt_dt
                            ftotal = (fc + fd + fr) * eij
                            acc[i] += ftotal / mass
                            acc[j] -= ftotal / mass
                    j = linked[j]

@njit
def compute_spring_forces(pos, acc, bonds, KS, rS):
    for b in range(bonds.shape[0]):
        i, j = bonds[b]
        rij = pos[i] - pos[j]
        r = np.sqrt(rij[0]**2 + rij[1]**2)
        if r > 1e-12:
            eij = rij / r
            fs = KS * (1 - r/rS)
            fvec = fs * eij
            acc[i] += fvec / mass
            acc[j] -= fvec / mass

# ==== Initialization Helpers ====
def create_chain(n):
    positions, types, bonds = [], [], []
    for _ in range(n):
        base = len(positions)
        pts = np.random.rand(7,2) * L
        for k in range(7):
            positions.append(pts[k]); types.append(TYPE_A if k<2 else TYPE_B)
        for k in range(6): bonds.append((base+k, base+k+1))
    return np.array(positions), np.array(types), np.array(bonds, dtype=np.int32)

def create_ring(n):
    positions, types, bonds = [], [], []
    for _ in range(n):
        base = len(positions)
        pts = np.random.rand(9,2) * L
        for k in range(9): positions.append(pts[k]); types.append(TYPE_A)
        for k in range(9): bonds.append((base+k, base+((k+1)%9)))
    return np.array(positions), np.array(types), np.array(bonds, dtype=np.int32)

def create_walls(mode):
    positions, types, velocities = [], [], []
    v = 5.0 if mode=='couette' else 0.0
    n_wall = int(rho * L * rc / 2)
    ys = np.random.rand(n_wall) * rc
    for y in ys:
        positions.append([np.random.rand()*L, y]); types.append(TYPE_W); velocities.append([v,0])
    for y in ys:
        positions.append([np.random.rand()*L, y+L-rc]); types.append(TYPE_W); velocities.append([-v,0])
    return np.array(positions), np.array(types), np.array(velocities)

# ==== Simulations ====
def run_preliminary():
    steps, chunk = 2000, 500
    logger.info('Starting Preliminary')
    N = int(rho * L * L)
    pos = np.random.rand(N,2) * L
    vel = np.zeros((N,2)); acc = np.zeros_like(pos)
    ptype = np.full(N, TYPE_F)
    aij = np.full((4,4), 25.0)

    head, linked, nc = build_cell_list(pos)
    compute_dpd_forces(pos, vel, acc, ptype, head, linked, nc, aij)

    steps_list, temps_list = [], []
    for step in trange(steps, desc='Preliminary'):
        integrate(pos, vel, acc)
        head, linked, nc = build_cell_list(pos)
        compute_dpd_forces(pos, vel, acc, ptype, head, linked, nc, aij)
        integrate(pos, vel, acc)
        if (step+1) % 100 == 0:
            vcom = vel.mean(axis=0); vel -= vcom
        # metrics
        speed = np.linalg.norm(vel,axis=1)
        temp = np.mean(speed**2)*mass/2.0
        steps_list.append(step)
        temps_list.append(temp)
    # Save vel, and temperature evolution
    np.savez_compressed('data/preliminary/preliminary.npz',
                        vel=vel,
                        steps=np.array(steps_list),
                        temps=np.array(temps_list))
    logger.info('Preliminary saved')


def run_couette():
    steps, chunk, nb = 10000, 2000, 50
    logger.info('Starting Couette')
    cps, ct, cb = create_chain(42)
    wps, wt, wvs = create_walls('couette')
    Nc, Nw = len(cps), len(wps)
    Nf = int(rho*L*L) - Nc - Nw
    fps = np.random.rand(Nf,2)*L
    vel_f = np.zeros((Nf,2)); ft = np.full(Nf, TYPE_F)

    pos = np.vstack([cps, wps, fps])
    vel = np.vstack([np.zeros_like(cps), wvs, vel_f])
    ptype = np.concatenate([ct, wt, ft])
    bonds = cb
    aij = np.array([[50,25,25,200],[25,1,300,200],[25,300,25,200],[200,200,200,0]])
    acc = np.zeros_like(pos)
    head, linked, nc = build_cell_list(pos)

    snaps = list(range(0, steps, 1000))
    if snaps[-1] != steps-1: snaps.append(steps-1)
    yedges = np.linspace(0, L, nb+1)

    steps_c, temps_c, vprofs, e2es = [], [], [], []
    for step in trange(steps, desc='Couette'):
        compute_dpd_forces(pos, vel, acc, ptype, head, linked, nc, aij)
        compute_spring_forces(pos, acc, bonds, KS=100, rS=0.1)
        integrate(pos, vel, acc)
        pos[Nc:Nc+Nw] += wvs * dt
        head, linked, nc = build_cell_list(pos)
        compute_dpd_forces(pos, vel, acc, ptype, head, linked, nc, aij)
        integrate(pos, vel, acc)
        if step in snaps:
            speed = np.linalg.norm(vel,axis=1)
            temp = np.mean(speed**2)*mass/2.0
            vprof = np.zeros(nb); cnt = np.zeros(nb)
            for i in range(len(pos)):
                if ptype[i] != TYPE_W:
                    idx = int(pos[i,1]/L*nb)
                    vprof[idx] += vel[i,0]; cnt[idx] += 1
            vprof /= np.maximum(cnt,1)
            e2e = np.array([np.linalg.norm([apply_pbc(pos[i1,0]-pos[i0,0]),
                                            apply_pbc(pos[i1,1]-pos[i0,1])])
                             for i0,i1 in [(i*7, i*7+6) for i in range(42)]])
            steps_c.append(step)
            temps_c.append(temp)
            vprofs.append(vprof)
            e2es.append(e2e)
    # Save Couette data
    np.savez_compressed('data/couette/couette.npz',
                        yedges=yedges,
                        steps=np.array(steps_c),
                        temps=np.array(temps_c),
                        vprofs=np.array(vprofs),
                        e2es=np.array(e2es))
    logger.info('Couette saved')


def run_poiseuille():
    steps, chunk, nb = 10000, 2000, 50
    logger.info('Starting Poiseuille')
    rps, rt, rb = create_ring(10)
    wps, wt, wvs = create_walls('poiseuille')
    Nr, Nw = len(rps), len(wps)
    Nf = int(rho*L*L) - Nr - Nw
    fps = np.random.rand(Nf,2)*L
    vel_f = np.zeros((Nf,2)); ft = np.full(Nf, TYPE_F)

    pos = np.vstack([rps, wps, fps])
    vel = np.vstack([np.zeros_like(rps), wvs, vel_f])
    ptype = np.concatenate([rt, wt, ft])
    bonds = rb
    aij = np.array([[50,25,200],[25,25,200],[200,200,0]])
    acc = np.zeros_like(pos)
    head, linked, nc = build_cell_list(pos)

    snaps = list(range(0, steps, 1000))
    if snaps[-1] != steps-1: snaps.append(steps-1)
    yedges = np.linspace(0, L, nb+1)

    steps_o, temps_o, mean_vx_o, max_vx_o = [], [], [], []
    vprofs_o, concs_o, Rgs_o, center_y_o = [], [], [], []
    for step in trange(steps, desc='Poiseuille'):
        compute_dpd_forces(pos, vel, acc, ptype, head, linked, nc, aij)
        compute_spring_forces(pos, acc, bonds, KS=100, rS=0.3)
        acc[:,0] += 0.3/mass
        integrate(pos, vel, acc)
        head, linked, nc = build_cell_list(pos)
        compute_dpd_forces(pos, vel, acc, ptype, head, linked, nc, aij)
        integrate(pos, vel, acc)
        if step in snaps:
            speed = np.linalg.norm(vel, axis=1)
            temp = np.mean(speed**2)*mass/2.0
            fv = vel[ptype==TYPE_F,0]
            mean_vx = fv.mean(); max_vx = fv.max()
            centers = np.array([pos[i*9:(i+1)*9].mean(axis=0) for i in range(len(rt)//9)])
            cy = centers[:,1]
            center_y = cy.mean()
            vprof = np.zeros(nb); cnt = np.zeros(nb)
            for i in range(len(pos)):
                if ptype[i] != TYPE_W:
                    idx = int(pos[i,1]/L*nb)
                    vprof[idx] += vel[i,0]; cnt[idx] += 1
            vprof /= np.maximum(cnt,1)
            conc = np.zeros(nb); cntc = np.zeros(nb)
            for y in cy:
                idx = int(y/L*nb)
                conc[idx] += 1; cntc[idx] += 1
            conc /= np.maximum(cntc,1)
            Rg = np.array([np.sqrt(np.mean(np.sum((pos[i*9:(i+1)*9] - centers[j])**2,axis=1)))
                           for j,i in enumerate(range(0, Nr, 9))])
            steps_o.append(step)
            temps_o.append(temp)
            mean_vx_o.append(mean_vx); max_vx_o.append(max_vx)
            vprofs_o.append(vprof); concs_o.append(conc)
            Rgs_o.append(Rg); center_y_o.append(center_y)
    # Save Poiseuille data
    np.savez_compressed('data/poiseuille/poiseuille.npz',
                        yedges=yedges,
                        steps=np.array(steps_o),
                        temps=np.array(temps_o),
                        mean_vx=np.array(mean_vx_o),
                        max_vx=np.array(max_vx_o),
                        vprofs=np.array(vprofs_o),
                        concs=np.array(concs_o),
                        Rgs=np.array(Rgs_o),
                        center_y=np.array(center_y_o))
    logger.info('Poiseuille saved')

if __name__=='__main__':
    run_preliminary()
    run_couette()
    run_poiseuille()
