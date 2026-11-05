# Causal mean tracking on contact trajectories

Policies were selected on validation before the fresh held-out seeds were read. Measurement variance is fixed at 1e-6 in noisy and noiseless runs. The true-normal alignment is an unavailable diagnostic and was excluded from tuning.

| Condition | Policy | Terminal MSE | Time-average MSE | Forecast MSE |
| --- | --- | ---: | ---: | ---: |
| anisotropic | identity | 5.1653e-05 | 4.8272e-05 | 5.2621e-05 |
| anisotropic | oracle | 3.5594e-05 | 4.0624e-05 | 3.5611e-05 |
| anisotropic | original | 5.1653e-05 | 4.8272e-05 | 5.2621e-05 |
| anisotropic | rolling | 5.1085e-05 | 4.8407e-05 | 5.1534e-05 |
| anisotropic | velocity | 5.1285e-05 | 5.0304e-05 | 5.1624e-05 |
| plane | identity | 2.768e-09 | 3.6614e-08 | 2.7951e-09 |
| plane | oracle | 2.768e-09 | 3.6614e-08 | 2.7951e-09 |
| plane | original | 2.768e-09 | 3.6614e-08 | 2.7951e-09 |
| plane | rolling | 4.0726e-08 | 6.5969e-08 | 4.1905e-08 |
| plane | velocity | 2.768e-09 | 3.6614e-08 | 2.7951e-09 |
| sphere_large | identity | 1.2702e-06 | 3.606e-06 | 1.8786e-06 |
| sphere_large | oracle | 3.566e-09 | 1.4351e-07 | 3.5662e-09 |
| sphere_large | original | 1.2702e-06 | 3.606e-06 | 1.8786e-06 |
| sphere_large | rolling | 2.5635e-06 | 5.465e-06 | 3.6074e-06 |
| sphere_large | velocity | 3.8281e-07 | 8.4307e-07 | 4.6143e-07 |
| sphere_small | identity | 1.0845e-06 | 2.1267e-06 | 1.5873e-06 |
| sphere_small | oracle | 2.0628e-09 | 1.0324e-07 | 2.0718e-09 |
| sphere_small | original | 1.0845e-06 | 2.1267e-06 | 1.5873e-06 |
| sphere_small | rolling | 8.9247e-07 | 1.5939e-06 | 1.1356e-06 |
| sphere_small | velocity | 1.262e-07 | 1.2114e-06 | 1.3758e-07 |
| weak | identity | 1.6971e-06 | 8.0049e-06 | 2.6497e-06 |
| weak | oracle | 6.1091e-09 | 1.3306e-07 | 6.1703e-09 |
| weak | original | 1.6971e-06 | 8.0049e-06 | 2.6497e-06 |
| weak | rolling | 2.096e-06 | 8.9258e-06 | 3.1193e-06 |
| weak | velocity | 8.8594e-07 | 2.7804e-06 | 9.9213e-07 |

## What this isolates

Residual noiseless error under identity memory measures stale-state bias; its sensor likelihood scale was not changed. A causal velocity forecast can reduce or increase that bias, because motion estimated from noisy modes can itself be inaccurate. The oracle shows what exact state alignment could achieve with unavailable geometry, not what a robot can implement.

Mode equality with independently transported weighted projector least squares is checked under the same forecast and memory. Thus any Stage 1 advantage belongs to forecast/memory choices, not to retaining a posterior distribution.

The anisotropic-resistance stress scores the actual KKT mobility. It violates the isotropic projector observation law and remains outside the matched model.

All per-observation forecast/corrected errors, including startup, are in error_paths.npz. trial0_paths.npz retains raw A, priors, nominal rotations, mode projectors and physical trajectories. No confidence interval is a simultaneous-time guarantee.
