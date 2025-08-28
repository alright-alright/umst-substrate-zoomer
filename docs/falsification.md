# Falsification Protocols

To avoid misleading visualizations, run ablations and nulls:

1. **Null Seed:** shuffle phases and randomize couplings → lattice should not emerge (R_k stays low).
2. **-HASR:** disable reinforcement; expect weaker or no growth in R_k across scales.
3. **High Friction:** raise zeta; binding thresholds become strict; order emergence stalls.
4. **Fixed-θ:** disable adaptive thresholds; robustness should drop.

The scaffold prints and exports metrics per scale to support these checks.
