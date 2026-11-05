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

Time-average loss, same-parameter Gaussian checks, all misses and the other contact conditions are in the [full findings](../results/contact_tracking/findings.md). Cost scope: median of three separate serial runs of preassigned small-sphere trajectory 0; shared forecast included. Average report cost does not guarantee a 50 ms deadline. These numbers separate improved dynamics from any additional benefit of a distribution. The finite validation scan selected a lag of 32 at the extended boundary; it does not establish a globally optimal forecast.
