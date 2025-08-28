import argparse, os, json, time
from datetime import datetime
import numpy as np
import yaml
import matplotlib.pyplot as plt

from .loop import Loop, Edge
from .resonance import ResonanceField, compute_order_param
from .coarse import coarse_grain
from .hasr import hasr_update
from .chrono import FrictionModel

def seed_loops(cfg, rng):
    n = int(cfg['n_loops'])
    d = int(cfg['space_dim'])
    dom_min = cfg['domain']['min']
    dom_max = cfg['domain']['max']
    positions = rng.uniform(dom_min, dom_max, size=(n, d))

    # Motifs: seed a few clusters in phase + space
    m = int(cfg['motifs']['count'])
    phase_spread = float(cfg['motifs']['phase_spread'])
    spatial_spread = float(cfg['motifs']['spatial_spread'])
    centers_idx = rng.integers(0, n, size=m)
    centers_pos = positions[centers_idx]
    centers_phi = rng.uniform(0, 2*np.pi, size=m)

    phis = rng.uniform(0, 2*np.pi, size=n)
    # move a slice of points toward cluster centers (both space + phase)
    for c in range(m):
        # choose a band for this center
        mask = np.linalg.norm(positions - centers_pos[c], axis=1) < spatial_spread
        phis[mask] = (centers_phi[c] + rng.normal(0, phase_spread, size=mask.sum())) % (2*np.pi)

    omegas = rng.uniform(0.5, 1.5, size=n)
    amps = rng.uniform(0.8, 1.2, size=n)
    loops = [Loop(i, float(phis[i]), float(omegas[i]), float(amps[i]), {}, 0.0, 1.0, positions[i]) for i in range(n)]
    return loops

def build_edges(cfg, rng, n):
    # For scalability without heavy libs, connect each node to 'avg_degree' random neighbors (undirected)
    k = int(cfg['avg_degree'])
    edges = []
    for i in range(n):
        # sample k neighbors without replacement
        # avoid self loops
        neighbors = set()
        while len(neighbors) < k:
            j = int(rng.integers(0, n))
            if j != i:
                neighbors.add(j)
        for j in neighbors:
            if i < j:
                w = float(rng.uniform(0.2, 0.8))
                edges.append(Edge(i, j, w))
    return edges

def export_lattice(loops, edges, out_path):
    data = {
        "nodes": [{"id": L.id, "phi": L.phi, "omega": L.omega, "amp": L.amp, "x": float(L.pos[0]), "y": float(L.pos[1])} for L in loops],
        "edges": [{"source": e.i, "target": e.j, "w": e.w} for e in edges]
    }
    with open(out_path, "w") as f:
        json.dump(data, f)

def plot_R_vs_scale(Rks, out_png):
    plt.figure()
    xs = list(range(len(Rks)))
    plt.plot(xs, Rks, marker='o')
    plt.xlabel("Scale k")
    plt.ylabel("Order parameter R_k (mean over tiles)")
    plt.title("Order vs. Scale")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def plot_frame(loops, is_bound, out_png):
    # 2D scatter: bound vs unbound (no custom colors per instructions)
    import matplotlib.pyplot as plt
    pos = np.array([L.pos for L in loops])
    plt.figure()
    # unbound first
    mask_unb = ~is_bound
    plt.scatter(pos[mask_unb,0], pos[mask_unb,1], s=2, alpha=0.4)
    mask_b = is_bound
    plt.scatter(pos[mask_b,0], pos[mask_b,1], s=3, alpha=0.9)
    plt.xlabel("x"); plt.ylabel("y"); plt.title("Bindings at this scale")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def run(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get('seed', 42))
    rng = np.random.default_rng(seed)

    loops = seed_loops(cfg, rng)
    edges = build_edges(cfg, rng, len(loops))

    field = ResonanceField(loops, edges, cfg, rng)
    friction = FrictionModel(
        zeta=cfg['friction']['zeta'],
        decay=cfg['friction']['decay'],
        min_theta=cfg['binding'].get('min_theta', 0.35),
        max_theta=cfg['binding'].get('max_theta', 0.85),
        adapt_rate=cfg['binding'].get('adapt_rate', 0.02),
    )

    tile_count0 = int(cfg['window']['tile_count'])
    theta = float(cfg['binding']['theta0'])
    adaptive = bool(cfg['binding'].get('adaptive', True))

    # Artifacts
    run_dir = os.path.join(cfg['artifacts_dir'], f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    frames_dir = os.path.join(run_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    metrics = {"Rk": [], "Hk": [], "bound_fraction": []}

    # Iterate scales
    K = int(cfg['coarse']['levels'])
    for k in range(K):
        phis = np.array([L.phi for L in loops])
        positions = np.array([L.pos for L in loops])
        tile_ids, tile_count = field.tile_assignments(positions, k, tile_count0)

        is_bound, R_tiles = field.detect_bindings(phis, tile_ids, tile_count, theta)
        bound_fraction = float(is_bound.mean())

        # HASR updates
        edges = hasr_update(edges, is_bound, lr=cfg['hasr']['lr'], decay=cfg['hasr']['decay'])

        # Metrics
        Rk = float(R_tiles.mean())
        Hk = field.harmony()
        metrics["Rk"].append(Rk)
        metrics["Hk"].append(Hk)
        metrics["bound_fraction"].append(bound_fraction)

        # Export per-scale lattice JSON and optional frame
        export_lattice(loops, edges, os.path.join(run_dir, f"lattice_scale_{k}.json"))
        if cfg['render'].get('export_frames', True):
            plot_frame(loops, is_bound, os.path.join(frames_dir, f"scale_{k}.png"))

        # Adapt theta using friction/TSL
        if adaptive:
            theta = friction.adapt_theta(theta, bound_fraction)

        # Coarse-grain to next scale unless last
        if k < K - 1:
            loops, edges = coarse_grain(loops, edges, tile_ids, tile_count, cfg)

    # Final exports
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    plot_R_vs_scale(metrics["Rk"], os.path.join(run_dir, "order_vs_scale.png"))
    export_lattice(loops, edges, os.path.join(run_dir, "lattice_top.json"))

    print(f"Run complete. Artifacts in: {run_dir}")
    return run_dir

def main():
    p = argparse.ArgumentParser(description="UMST Substrate Zoomer")
    p.add_argument("-c", "--config", dest="config", default="configs/default.yaml")
    args = p.parse_args()
    run(args.config)

if __name__ == "__main__":
    main()
