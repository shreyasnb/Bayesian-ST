# Process uncertainty and distribution-family comparison

The final physical comparison follows the later contact tracking guide. It first selects a causal mean forecast, then separately selects report temperature and process spread. It replaces the earlier proposed broad beta/likelihood-scale scan with the requested bounded Stage 1/Stage 2 design. The raw historical pilot remains unchanged.

The comparison is paired across **64 independent small-sphere trajectories**. Positive paired gains below mean lower loss for the Bingham process prediction; intervals are pointwise 95% Student intervals.

| Question | Held-out result |
| --- | --- |
| Does the causal forecast reduce lag? | Terminal mode MSE changes from 1.085 to 0.126 (mm/s)², a 88.4% reduction. Time-average MSE improves 43.0%. These gains are available to point methods. |
| What does spread calibration change? | With T=2, clean coverage changes from 98.0% to 100.0%; width from 0.938 to 1.326 mm/s; interval score from 0.979 to 1.326 mm/s (lower is better). The mode is unchanged. |
| Does process Bingham beat the cheap Gaussian? | It has no resolved terminal MSE advantage. The Gaussian-minus-Bingham paired MSE difference is -0.001381 [-0.006124, 0.003362] (mm/s)² (-0.62% of Gaussian loss). |
| What does computation cost? | Measured filter cost, including the common forecast tracker, is 2.38 versus 2.05 ms/observation. One 27-component uncertainty report averages 145 versus 73 ms. |

For uncertainty, Bingham/Gaussian clean-component coverage is **99.5%/99.5%**, with mean interval width **1.395/1.392 mm/s** and interval score **1.405/1.402 mm/s** (lower is better). The paired Gaussian-minus-Bingham score difference is -0.0028 [-0.0081, 0.0026] mm/s. Lower MSE alone would not establish better uncertainty.

The whole-trajectory tradeoff differs from the terminal result: time-average MSE is 1.162 for the raw forecast, 0.578 for process Bingham and 0.600 (mm/s)² for the Gaussian. Point memory was selected for terminal MSE, so this is a secondary comparison of the retained policies, not a search for the best possible time-average point estimator. Temperature was fitted on the curved-condition validation mixture; more coverage on this one endpoint can mean unnecessarily wide intervals.

An exploratory case shows where process uncertainty is useful: when the second and third force components are divided by five, the temperature control covers 94.8% at width 2.12 mm/s. Process Bingham gives 95.4% at 1.71 mm/s; the Gaussian gives 95.5% at 1.69 mm/s. Useful uncertainty propagation is supported there, while the full Bingham family still has no demonstrated premium over the Gaussian.

Time-average loss, same-parameter Gaussian checks, all misses and the other contact conditions are in the [full findings](../contact_tracking/findings.md). Cost scope: median of three separate serial runs of preassigned small-sphere trajectory 0; shared forecast included. Average report cost does not guarantee a 50 ms deadline. These numbers separate improved dynamics from any additional benefit of a distribution. The finite validation scan selected a lag of 32 at the extended boundary; it does not establish a globally optimal forecast.


Full physical results and limitations: [contact tracking findings](../contact_tracking/findings.md).

## Matched process

| Kappa | Method | Trials | MSE | Clean coverage | Interval score | Unresolved |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | Fixed forgetting diagnostic | 200 | 0.20822 | 0.783 | 1.5964 | 0 |
| 4 | Full-density reference | 16 | 0.15005 | 0.954 | 0.8199 | 0 |
| 4 | Process Bingham | 200 | 0.13291 | 0.955 | 0.8189 | 0 |
| 4 | Recursive Gaussian | 200 | 0.21124 | 0.860 | 1.2705 | 0 |
| 4 | Same-parameter Laplace | 200 | 0.13336 | 0.930 | 0.8262 | 26 |
| 20 | Fixed forgetting diagnostic | 200 | 0.091105 | 0.837 | 0.9086 | 0 |
| 20 | Full-density reference | 16 | 0.056013 | 0.972 | 0.5923 | 1 |
| 20 | Process Bingham | 200 | 0.075424 | 0.953 | 0.6664 | 0 |
| 20 | Recursive Gaussian | 200 | 0.085358 | 0.920 | 0.7649 | 0 |
| 20 | Same-parameter Laplace | 200 | 0.075151 | 0.935 | 0.6643 | 1 |

This dimensionless process has known kappa and uniform initialization. It uses 200 independent trajectories per kappa; the first 16 per kappa are the full-density reference subset, fixed before testing. Do not compare a 16-trajectory reference rate directly with a 200-trajectory method rate as if they were paired. The complete matched metrics retain method rows on that same subset for paired inspection. Refined full-density failures remain visible; successful development is not a certificate for every held-out target.

## What the experiment can support

1. Explicit propagation is judged after mean and spread calibration, using proper scores and widths as well as coverage. A dynamics improvement is available to the point and Gaussian controls.

2. The same-A Laplace comparison isolates global density shape; the recursive Gaussian comparison also includes local transition and measurement approximations. Broad, ambiguous posteriors and concentrated physical paths are reported separately.

3. The physical plant is not the isotropic Watson transition. Good matched-process calibration does not establish physical robustness; a wrong resistance law remains a likelihood failure.

4. Correction and mean prediction are distinguished from occasional 27-component continuous-CDF reports. Use measured wall time and flags, not only the conjugate update cost, to judge a 20 Hz implementation.

5. An uncertainty-led paper needs a material width/score/decision advantage beyond the cheap Gaussian. Where that evidence is absent or numerically unresolved, the defensible claim is narrower identification, exact conditional updating and controlled theory/numerics, not general calibrated contact prediction.
