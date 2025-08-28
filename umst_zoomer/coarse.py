import numpy as np
from .loop import Loop, Edge
from .resonance import circular_mean

def coarse_grain(loops, edges, tile_ids, tile_count, cfg):
    # Aggregate loops by tile into macro-loops (centroids & circular means)
    tile_size = tile_count * tile_count
    tiles = [[] for _ in range(tile_size)]
    for idx, L in enumerate(loops):
        tiles[tile_ids[idx]].append(idx)

    macro_loops = []
    id_map = {}  # map old tile index -> new macro id
    for t_idx, idxs in enumerate(tiles):
        if not idxs:
            continue
        phis = [loops[i].phi for i in idxs]
        amps = [loops[i].amp for i in idxs]
        pos = np.mean([loops[i].pos for i in idxs], axis=0)
        omega = np.mean([loops[i].omega for i in idxs])
        phi_bar = circular_mean(phis)
        amp_bar = float(np.mean(amps))
        macro_id = len(macro_loops)
        id_map[t_idx] = macro_id
        macro_loops.append(Loop(macro_id, phi_bar, omega, amp_bar, {}, 0.0, 1.0, pos))

    # Build macro edges by summing weights across tiles
    macro_edges_dict = {}
    for e in edges:
        # Skip edges whose endpoints were mapped to the same empty tile
        # We need the original tile ids passed in (tile ids per loop)
        # This function assumes caller provides tile_ids for the same scale used to build tiles.
        pass

    # We need loop->tile mapping; reconstruct from input tile_ids
    # Build per-loop tile lookup
    # tile_ids corresponds to loops in order, so we can map
    loop_to_tile = tile_ids

    agg = {}
    for e in edges:
        ti = loop_to_tile[e.i]
        tj = loop_to_tile[e.j]
        if ti not in id_map or tj not in id_map:
            continue
        mi = id_map[ti]
        mj = id_map[tj]
        if mi == mj:
            continue
        key = (min(mi, mj), max(mi, mj))
        agg[key] = agg.get(key, 0.0) + e.w

    macro_edges = [Edge(i, j, float(w)) for (i, j), w in agg.items()]
    return macro_loops, macro_edges
