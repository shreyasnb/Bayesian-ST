# Tracking and process evidence

This directory contains the engineering quick profile. It tests execution and numerical checks; the paper comparison uses a separate seed stream and the paper profile after the general validation gate passes.

The experiment runner hashes scientific sources, configuration, task inputs, and JSON checkpoint results. A completed manifest also hashes every retained CSV and NPZ artifact. Before using completed evidence, call `verify_component` from `lib.beamforming_validation` or use the figure readers, which call it.

Partial resume assumes that the preassigned NPZ trajectory files in `trajectories/` remain intact. Those arrays are written separately from scalar JSON checkpoints; during an incomplete run, restoring a scalar checkpoint does not independently verify or recreate its NPZ file. If an incomplete run's trajectory files have been removed or changed, preserve that incomplete directory and rerun the same frozen protocol in a fresh output directory. A completed directory is checked against its artifact hashes and must not be silently repaired or overwritten.

`paired.csv` uses independent trajectories as the replicate. It keeps the complete planned cohort, reports numerical-resolution counts separately, and marks missing endpoints rather than dropping them. Its confidence intervals are exploratory pointwise paired Student intervals. The scalar posterior samples and fixed queries are not counted as independent trajectories. `risk_calibration.csv` first averages losses across the fixed queries within each trajectory.
