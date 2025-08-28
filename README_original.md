# UMST Substrate Zoomer (v0.1)

A minimal, runnable scaffold that demonstrates the **substrate zoom** idea using our custom UMST-style primitives:
- seed a cloud of oscillator **loops**
- detect **resonance** & **binding**
- apply **HASR** reinforcement and **friction/TSL** modulation
- **coarse-grain** across scales
- compute order parameters (R_k) and a harmony functional H
- export **JSON metrics**, **PNG plots**, and a **lattice.json** for 3D/GLB pipelines

This is an **application harness** intended to sit on top of the UMST engine (MPU/ChronoLayer/SSP), but is runnable standalone for experimentation and paper figures.

> Default config is sized for laptop/dev. Increase `n_loops` gradually for larger studies.

## Quick start

```bash
# (optional) create a virtual env
python -m venv .venv && source .venv/bin/activate

# install
pip install -r requirements.txt

# run (uses configs/default.yaml)
python -m umst_zoomer.run -c configs/default.yaml
```

Artifacts will be written under `artifacts/run_YYYYMMDD_HHMMSS/`:
- `metrics.json`: per-scale metrics (R_k, H_k, bound_fraction)
- `order_vs_scale.png`: plot of R_k vs. scale
- `frames/scale_k.png`: scatter snapshots colored by binding state
- `lattice_scale_k.json`: node/edge export per scale for external GLB pipelines
- `lattice_top.json`: final top-scale lattice export
