# Engine Wiring: MPU / ChronoLayer / TSL

This scaffold is self-contained, but exposes clear adapters to plug into the full UMST engine:

- **MPU (Memory Processing Unit):** symbolic payloads on loops (`payload` dict) and cross-loop reinforcement live here.
- **ChronoLayer:** temporal windows and decay; we expose `t0/t1` and a simple exponential decay.
- **TSL (Temporal Smoothing Layer):** smoothing of recent inputs to stabilize the perceived present; modeled via friction `zeta` and adaptive thresholding `theta` in this scaffold.

To integrate real modules, implement or wrap the methods within `chrono.py` and swap kernels in `resonance.py`.
