import numpy as np
from .loop import Loop, Edge

def circular_mean(angles):
    # mean of angles on unit circle
    if len(angles) == 0:
        return 0.0
    v = np.exp(1j * angles)
    m = v.mean()
    return np.angle(m)

def compute_order_param(angles):
    if len(angles) == 0:
        return 0.0
    v = np.exp(1j * angles)
    return np.abs(v.mean())

class ResonanceField:
    def __init__(self, loops, edges, cfg, rng):
        self.loops = loops
        self.edges = edges
        self.cfg = cfg
        self.rng = rng

    def harmony(self):
        # H = sum J_ij * cos(phi_i - phi_j)
        phis = np.array([L.phi for L in self.loops])
        H = 0.0
        for e in self.edges:
            d = phis[e.i] - phis[e.j]
            H += e.w * np.cos(d)
        return float(H)

    def tile_assignments(self, positions, k, tile_count0):
        # 2D only for scaffold simplicity
        L = len(positions)
        # tile_count increases by factor (cell_ratio)^k
        ratio = max(1, int(self.cfg['coarse']['cell_ratio']))
        tile_count = max(1, int(tile_count0 // (ratio**k)))
        tile_count = max(1, tile_count)
        # normalize to [0,1)
        mins = np.array([self.cfg['domain']['min'], self.cfg['domain']['min']])
        maxs = np.array([self.cfg['domain']['max'], self.cfg['domain']['max']])
        span = (maxs - mins)
        grid = np.floor(((positions - mins) / span) * tile_count).astype(int)
        grid = np.clip(grid, 0, tile_count - 1)
        # linear index
        tile_ids = grid[:,0] * tile_count + grid[:,1]
        return tile_ids, tile_count

    def detect_bindings(self, phis, tile_ids, tile_count, theta):
        # compute R per tile and mark bound if R > theta
        R_per_tile = np.zeros(tile_count*tile_count, dtype=float)
        counts = np.zeros_like(R_per_tile)
        for tidx, phi in zip(tile_ids, phis):
            R_per_tile[tidx] += np.exp(1j*phi)
            counts[tidx] += 1
        # avoid divide by zero
        mask = counts > 0
        R_per_tile[mask] = np.abs(R_per_tile[mask] / counts[mask])
        R_per_tile[~mask] = 0.0
        bound_tiles = R_per_tile > theta
        is_bound = bound_tiles[tile_ids]
        return is_bound, R_per_tile

