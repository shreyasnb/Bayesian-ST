# Contact-normal study

The observations use the real conditional model u=P_X v+e. P_X is a tangent-plane projector; its normal line has Bingham parameter -A. Known applied free velocity and measured wrist rotation are supplied to every corresponding estimator. True surface normals and positions are scoring-only.

The quick profile has 4 validation and 8 independent test datasets per static input program, 4 validation and 8 test trajectories per physical condition. Query coordinates and reporting times are not independent experimental repeats.

Every static command has the same Euclidean norm. Balanced and directionally concentrated programs therefore have identical total energy at every horizon. Their local Fisher information need not be the same. The stationary normal is Haar-distributed in the Monte Carlo comparison; the fixed e3 density illustration is a separate preassigned example.

## Comparators and decisions

The modal plane equals independent weighted conditional projector least squares. The mean, mode and matched isotropized normal mean use identical A in the shape comparison. A separate globally validated scalar contracts the modal plane toward the known-rank isotropic mean. Ridge regression also regularizes toward that known-rank isotropic mean, rather than zero; it uses force and measured velocity. Residual PCA uses v-u and therefore has access to both signals, while discarding some conditional-likelihood information.

The same-parameter local Gaussian uses the modal Hessian of -A. Its exponential-map mean is integrated; local cut-locus mass and numerical refinement have separate flags. Tied precision fails explicitly. The physical recursive Gaussian uses a six-observation data-only initialization, Joseph correction and the same measured frame rotations; Q is selected separately on validation. It is not the same-parameter control.

For the physical contact, each method selects its own beta or Q on independent terminal response loss. The mean and matched-isotropization rows share beta by construction. The identity-frame ablation retains the same beta and data. G=R_t^T R_(t-1) is derived from wrist orientation independent of the hidden surface normal, not an oracle normal alignment.

The expected decision is v^B=[I+lambda(I-Pbar)]^(-1) v_des. Cost uses a soft inadmissible-command penalty, not a guarantee of feasible contact motion or a closed-loop controller. The expected penalty is linear in I-Pbar. Unconstrained ridge maps are scored for response prediction only and have no Eq.36 cost.

## Static paired response comparisons

      kind  t    reference     endpoint  trials  finite_pairs          gain         lower        upper                                                 positive  numerical_flags_retained
  balanced  1    isotropic response_mse       8             8  2.785121e-06 -6.999885e-06 1.257013e-05 reference minus posterior mean; independent trajectories                      True
  balanced  1      laplace response_mse       8             8  2.903619e-06 -7.070957e-06 1.287819e-05 reference minus posterior mean; independent trajectories                      True
  balanced  1         mode response_mse       8             8  5.650487e-05 -4.296649e-06 1.173064e-04 reference minus posterior mean; independent trajectories                      True
  balanced  1  regularized response_mse       8             8  1.834044e-05 -3.918999e-06 4.059989e-05 reference minus posterior mean; independent trajectories                      True
  balanced  1 residual_pca response_mse       8             8  5.203063e-05 -1.061758e-05 1.146789e-04 reference minus posterior mean; independent trajectories                      True
  balanced  1        ridge response_mse       8             8  9.651175e-06 -5.620203e-06 2.492255e-05 reference minus posterior mean; independent trajectories                      True
  balanced  2    isotropic response_mse       8             8  1.621566e-06 -3.723854e-06 6.966987e-06 reference minus posterior mean; independent trajectories                      True
  balanced  2      laplace response_mse       8             8  1.374016e-06 -9.415195e-06 1.216323e-05 reference minus posterior mean; independent trajectories                      True
  balanced  2         mode response_mse       8             8  3.406313e-05 -2.329962e-05 9.142587e-05 reference minus posterior mean; independent trajectories                      True
  balanced  2  regularized response_mse       8             8  2.376663e-05 -2.074428e-05 6.827754e-05 reference minus posterior mean; independent trajectories                      True
  balanced  2 residual_pca response_mse       8             8  4.739793e-05 -8.639592e-06 1.034354e-04 reference minus posterior mean; independent trajectories                      True
  balanced  2        ridge response_mse       8             8  1.985670e-05 -7.522212e-06 4.723560e-05 reference minus posterior mean; independent trajectories                      True
  balanced  4    isotropic response_mse       8             8  1.037311e-06 -1.135859e-06 3.210481e-06 reference minus posterior mean; independent trajectories                      True
  balanced  4      laplace response_mse       8             8  1.235652e-06 -5.720380e-06 8.191683e-06 reference minus posterior mean; independent trajectories                      True
  balanced  4         mode response_mse       8             8  1.662536e-05 -2.185418e-05 5.510489e-05 reference minus posterior mean; independent trajectories                      True
  balanced  4  regularized response_mse       8             8  1.037120e-05 -1.909641e-05 3.983880e-05 reference minus posterior mean; independent trajectories                      True
  balanced  4 residual_pca response_mse       8             8  4.658181e-05 -2.592280e-05 1.190864e-04 reference minus posterior mean; independent trajectories                      True
  balanced  4        ridge response_mse       8             8  3.409066e-05  1.572542e-05 5.245590e-05 reference minus posterior mean; independent trajectories                      True
  balanced  8    isotropic response_mse       8             8 -9.421049e-09 -1.835070e-07 1.646649e-07 reference minus posterior mean; independent trajectories                      True
  balanced  8      laplace response_mse       8             8 -2.428499e-07 -7.863051e-07 3.006053e-07 reference minus posterior mean; independent trajectories                      True
  balanced  8         mode response_mse       8             8  6.546212e-07 -1.420769e-06 2.730012e-06 reference minus posterior mean; independent trajectories                      True
  balanced  8  regularized response_mse       8             8 -3.810420e-07 -1.352737e-06 5.906527e-07 reference minus posterior mean; independent trajectories                      True
  balanced  8 residual_pca response_mse       8             8  1.646546e-05 -4.986717e-06 3.791764e-05 reference minus posterior mean; independent trajectories                      True
  balanced  8        ridge response_mse       8             8  2.221729e-05  8.610789e-06 3.582379e-05 reference minus posterior mean; independent trajectories                      True
  balanced 16    isotropic response_mse       8             8 -1.087927e-07 -4.011396e-07 1.835541e-07 reference minus posterior mean; independent trajectories                      True
  balanced 16      laplace response_mse       8             8 -2.957041e-07 -9.967219e-07 4.053138e-07 reference minus posterior mean; independent trajectories                      True
  balanced 16         mode response_mse       8             8  9.303535e-08 -5.985124e-07 7.845831e-07 reference minus posterior mean; independent trajectories                      True
  balanced 16  regularized response_mse       8             8 -1.832909e-07 -9.354773e-07 5.688955e-07 reference minus posterior mean; independent trajectories                      True
  balanced 16 residual_pca response_mse       8             8  2.943621e-05 -1.729077e-05 7.616319e-05 reference minus posterior mean; independent trajectories                      True
  balanced 16        ridge response_mse       8             8  1.816313e-05  9.720901e-06 2.660535e-05 reference minus posterior mean; independent trajectories                      True
  balanced 32    isotropic response_mse       8             8 -1.835466e-09 -5.837324e-09 2.166393e-09 reference minus posterior mean; independent trajectories                      True
  balanced 32      laplace response_mse       8             8 -1.215241e-09 -4.431930e-09 2.001448e-09 reference minus posterior mean; independent trajectories                      True
  balanced 32         mode response_mse       8             8  6.946984e-08 -4.039926e-08 1.793389e-07 reference minus posterior mean; independent trajectories                      True
  balanced 32  regularized response_mse       8             8 -5.741719e-09 -3.258964e-08 2.110620e-08 reference minus posterior mean; independent trajectories                      True
  balanced 32 residual_pca response_mse       8             8  9.525190e-06 -5.505648e-06 2.455603e-05 reference minus posterior mean; independent trajectories                      True
  balanced 32        ridge response_mse       8             8  1.502205e-05  8.490836e-06 2.155326e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  1    isotropic response_mse       8             8  2.785121e-06 -6.999885e-06 1.257013e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  1      laplace response_mse       8             8  2.903619e-06 -7.070957e-06 1.287819e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  1         mode response_mse       8             8  5.650487e-05 -4.296649e-06 1.173064e-04 reference minus posterior mean; independent trajectories                      True
unbalanced  1  regularized response_mse       8             8  1.834044e-05 -3.918999e-06 4.059989e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  1 residual_pca response_mse       8             8  5.203063e-05 -1.061758e-05 1.146789e-04 reference minus posterior mean; independent trajectories                      True
unbalanced  1        ridge response_mse       8             8  9.651175e-06 -5.620203e-06 2.492255e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  2    isotropic response_mse       8             8  2.880549e-06 -9.917382e-06 1.567848e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  2      laplace response_mse       8             8  2.593078e-06 -5.301494e-06 1.048765e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  2         mode response_mse       8             8  5.176924e-05 -1.775204e-05 1.212905e-04 reference minus posterior mean; independent trajectories                      True
unbalanced  2  regularized response_mse       8             8  2.918228e-05 -2.089732e-05 7.926188e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  2 residual_pca response_mse       8             8  8.952062e-05  3.154673e-05 1.474945e-04 reference minus posterior mean; independent trajectories                      True
unbalanced  2        ridge response_mse       8             8  8.700740e-06 -1.541760e-05 3.281908e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  4    isotropic response_mse       8             8  7.615173e-06 -4.143766e-06 1.937411e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  4      laplace response_mse       8             8  1.976136e-08 -3.001979e-06 3.041502e-06 reference minus posterior mean; independent trajectories                      True
unbalanced  4         mode response_mse       8             8  5.458432e-05  1.544294e-06 1.076243e-04 reference minus posterior mean; independent trajectories                      True
unbalanced  4  regularized response_mse       8             8  3.396624e-05 -5.241574e-06 7.317405e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  4 residual_pca response_mse       8             8  9.015795e-05  2.605196e-05 1.542639e-04 reference minus posterior mean; independent trajectories                      True
unbalanced  4        ridge response_mse       8             8  1.130865e-05  2.219688e-06 2.039762e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  8    isotropic response_mse       8             8  3.038530e-06 -5.738998e-06 1.181606e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  8      laplace response_mse       8             8  2.352798e-06 -6.553118e-06 1.125871e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  8         mode response_mse       8             8  1.863799e-05 -2.432569e-05 6.160168e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  8  regularized response_mse       8             8  7.932732e-06 -1.700403e-05 3.286949e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  8 residual_pca response_mse       8             8  4.819368e-05 -2.003380e-05 1.164212e-04 reference minus posterior mean; independent trajectories                      True
unbalanced  8        ridge response_mse       8             8  1.718216e-05 -1.940978e-06 3.630530e-05 reference minus posterior mean; independent trajectories                      True
unbalanced 16    isotropic response_mse       8             8  2.916078e-06 -4.734855e-06 1.056701e-05 reference minus posterior mean; independent trajectories                      True
unbalanced 16      laplace response_mse       8             8  1.852899e-06 -5.230733e-06 8.936530e-06 reference minus posterior mean; independent trajectories                      True
unbalanced 16         mode response_mse       8             8  1.461438e-05 -2.382136e-05 5.305012e-05 reference minus posterior mean; independent trajectories                      True
unbalanced 16  regularized response_mse       8             8  1.350350e-05 -2.259910e-05 4.960610e-05 reference minus posterior mean; independent trajectories                      True
unbalanced 16 residual_pca response_mse       8             8  4.848578e-05 -3.155755e-06 1.001273e-04 reference minus posterior mean; independent trajectories                      True
unbalanced 16        ridge response_mse       8             8  3.567576e-05  1.506600e-05 5.628551e-05 reference minus posterior mean; independent trajectories                      True
unbalanced 32    isotropic response_mse       8             8  3.600105e-06 -2.453593e-06 9.653803e-06 reference minus posterior mean; independent trajectories                      True
unbalanced 32      laplace response_mse       8             8  2.935347e-06 -3.817993e-06 9.688687e-06 reference minus posterior mean; independent trajectories                      True
unbalanced 32         mode response_mse       8             8  1.559365e-05 -1.022307e-05 4.141036e-05 reference minus posterior mean; independent trajectories                      True
unbalanced 32  regularized response_mse       8             8  1.424041e-05 -9.605871e-06 3.808669e-05 reference minus posterior mean; independent trajectories                      True
unbalanced 32 residual_pca response_mse       8             8  5.093294e-05 -7.654387e-06 1.095203e-04 reference minus posterior mean; independent trajectories                      True
unbalanced 32        ridge response_mse       8             8  2.522812e-05 -3.448491e-06 5.390473e-05 reference minus posterior mean; independent trajectories                      True

## Physical paired response comparisons

condition  t     reference     endpoint  trials  finite_pairs          gain         lower        upper                                                 positive  numerical_flags_retained
    plane  8      gaussian response_mse       8             8  1.145676e-08 -6.667563e-08 8.958914e-08 reference minus posterior mean; independent trajectories                      True
    plane  8     isotropic response_mse       8             8 -3.287757e-10 -1.571866e-09 9.143147e-10 reference minus posterior mean; independent trajectories                      True
    plane  8 mean_identity response_mse       8             8  8.947298e-06  3.788078e-06 1.410652e-05 reference minus posterior mean; independent trajectories                      True
    plane  8          mode response_mse       8             8  4.107047e-08 -4.748619e-08 1.296271e-07 reference minus posterior mean; independent trajectories                      True
    plane  8 mode_identity response_mse       8             8  9.218097e-06  3.836956e-06 1.459924e-05 reference minus posterior mean; independent trajectories                      True
    plane  8   mode_same_A response_mse       8             8  4.107047e-08 -4.748619e-08 1.296271e-07 reference minus posterior mean; independent trajectories                      True
    plane  8  residual_pca response_mse       8             8  5.617950e-06 -6.574449e-07 1.189334e-05 reference minus posterior mean; independent trajectories                      True
    plane  8         ridge response_mse       8             8  1.597947e-05  5.417981e-06 2.654095e-05 reference minus posterior mean; independent trajectories                      True
    plane 24      gaussian response_mse       8             8 -1.079020e-07 -2.410360e-07 2.523191e-08 reference minus posterior mean; independent trajectories                      True
    plane 24     isotropic response_mse       8             8  2.588498e-11 -1.128318e-10 1.646018e-10 reference minus posterior mean; independent trajectories                      True
    plane 24 mean_identity response_mse       8             8  1.175485e-05  4.643170e-06 1.886652e-05 reference minus posterior mean; independent trajectories                      True
    plane 24          mode response_mse       8             8  9.817450e-09 -4.134033e-09 2.376893e-08 reference minus posterior mean; independent trajectories                      True
    plane 24 mode_identity response_mse       8             8  1.186071e-05  4.700078e-06 1.902135e-05 reference minus posterior mean; independent trajectories                      True
    plane 24   mode_same_A response_mse       8             8  9.817450e-09 -4.134033e-09 2.376893e-08 reference minus posterior mean; independent trajectories                      True
    plane 24  residual_pca response_mse       8             8  1.064916e-06 -5.167282e-07 2.646559e-06 reference minus posterior mean; independent trajectories                      True
    plane 24         ridge response_mse       8             8  2.954843e-06  1.782109e-06 4.127577e-06 reference minus posterior mean; independent trajectories                      True
    plane 48      gaussian response_mse       8             8  1.010074e-08 -2.137752e-08 4.157899e-08 reference minus posterior mean; independent trajectories                      True
    plane 48     isotropic response_mse       8             8  1.549060e-11 -3.022278e-11 6.120398e-11 reference minus posterior mean; independent trajectories                      True
    plane 48 mean_identity response_mse       8             8  1.094120e-06  3.511916e-09 2.184728e-06 reference minus posterior mean; independent trajectories                      True
    plane 48          mode response_mse       8             8  5.851814e-10 -1.913501e-09 3.083864e-09 reference minus posterior mean; independent trajectories                      True
    plane 48 mode_identity response_mse       8             8  1.099422e-06  5.041108e-09 2.193802e-06 reference minus posterior mean; independent trajectories                      True
    plane 48   mode_same_A response_mse       8             8  5.851814e-10 -1.913501e-09 3.083864e-09 reference minus posterior mean; independent trajectories                      True
    plane 48  residual_pca response_mse       8             8  8.849034e-07  7.876494e-08 1.691042e-06 reference minus posterior mean; independent trajectories                      True
    plane 48         ridge response_mse       8             8  1.544527e-06  8.941201e-07 2.194935e-06 reference minus posterior mean; independent trajectories                      True
   sphere  8      gaussian response_mse       8             8 -2.172417e-06 -1.091561e-05 6.570778e-06 reference minus posterior mean; independent trajectories                      True
   sphere  8     isotropic response_mse       8             8  2.824732e-10 -3.058959e-06 3.059524e-06 reference minus posterior mean; independent trajectories                      True
   sphere  8 mean_identity response_mse       8             8  9.586058e-06 -1.072861e-05 2.990072e-05 reference minus posterior mean; independent trajectories                      True
   sphere  8          mode response_mse       8             8  1.369187e-07 -1.284052e-05 1.311435e-05 reference minus posterior mean; independent trajectories                      True
   sphere  8 mode_identity response_mse       8             8  3.033396e-05 -3.078819e-05 9.145611e-05 reference minus posterior mean; independent trajectories                      True
   sphere  8   mode_same_A response_mse       8             8  1.369187e-07 -1.284052e-05 1.311435e-05 reference minus posterior mean; independent trajectories                      True
   sphere  8  residual_pca response_mse       8             8  1.502970e-05 -4.667591e-05 7.673531e-05 reference minus posterior mean; independent trajectories                      True
   sphere  8         ridge response_mse       8             8  4.035530e-05  2.707085e-05 5.363975e-05 reference minus posterior mean; independent trajectories                      True
   sphere 24      gaussian response_mse       8             8 -1.331859e-06 -4.616213e-06 1.952495e-06 reference minus posterior mean; independent trajectories                      True
   sphere 24     isotropic response_mse       8             8 -1.166125e-07 -4.796816e-07 2.464566e-07 reference minus posterior mean; independent trajectories                      True
   sphere 24 mean_identity response_mse       8             8  3.251747e-05  1.182704e-05 5.320790e-05 reference minus posterior mean; independent trajectories                      True
   sphere 24          mode response_mse       8             8 -2.775014e-07 -1.804041e-06 1.249039e-06 reference minus posterior mean; independent trajectories                      True
   sphere 24 mode_identity response_mse       8             8  4.012416e-05  1.158356e-05 6.866475e-05 reference minus posterior mean; independent trajectories                      True
   sphere 24   mode_same_A response_mse       8             8 -2.775014e-07 -1.804041e-06 1.249039e-06 reference minus posterior mean; independent trajectories                      True
   sphere 24  residual_pca response_mse       8             8  4.214847e-06 -1.998153e-06 1.042785e-05 reference minus posterior mean; independent trajectories                      True
   sphere 24         ridge response_mse       8             8  4.208506e-05  2.766904e-05 5.650107e-05 reference minus posterior mean; independent trajectories                      True
   sphere 48      gaussian response_mse       8             8 -6.960146e-06 -1.916226e-05 5.241964e-06 reference minus posterior mean; independent trajectories                      True
   sphere 48     isotropic response_mse       8             8  8.731846e-07 -6.094259e-07 2.355795e-06 reference minus posterior mean; independent trajectories                      True
   sphere 48 mean_identity response_mse       8             8  6.562637e-07 -1.440905e-05 1.572157e-05 reference minus posterior mean; independent trajectories                      True
   sphere 48          mode response_mse       8             8  3.931620e-06 -2.613360e-06 1.047660e-05 reference minus posterior mean; independent trajectories                      True
   sphere 48 mode_identity response_mse       8             8 -2.223588e-06 -2.278813e-05 1.834095e-05 reference minus posterior mean; independent trajectories                      True
   sphere 48   mode_same_A response_mse       8             8  3.931620e-06 -2.613360e-06 1.047660e-05 reference minus posterior mean; independent trajectories                      True
   sphere 48  residual_pca response_mse       8             8  5.402907e-06  9.762319e-07 9.829583e-06 reference minus posterior mean; independent trajectories                      True
   sphere 48         ridge response_mse       8             8  3.379899e-05  1.301866e-05 5.457931e-05 reference minus posterior mean; independent trajectories                      True

## Numerical and interpretation limits

{
  "seconds": 15.569266151000193,
  "rows": {
    "static_metrics": 672,
    "static_fisher": 96,
    "dynamic_metrics": 432,
    "dynamic_bounds": 96,
    "higher_dimension": 48,
    "bounds": 16,
    "paired_static": 72,
    "paired_dynamic": 48,
    "story_checks": 30
  },
  "static_numerical_failures": 0,
  "static_local_gaussian_unresolved": 22,
  "dynamic_numerical_failures": 0,
  "dynamic_local_gaussian_unresolved": 1,
  "higher_numerical_failures": 0,
  "story_numerical_failures": 0,
  "bound_valid": 32,
  "bound_violations": 0,
  "maximum_modal_audit_error": 1.9895196601282805e-13,
  "maximum_decision_identity_error": 4.336808689942018e-19
}

Posterior conditional mass, numerical integration accuracy and repeated-data inclusion are different claims. Forgetting under a changing normal does not inherit a stationary confidence guarantee. Scalar intervals are clean-response quantiles, not noisy velocity prediction intervals.

The long stationary cohort checks the conditional modal theorem through residual r=v-u=(nn^T)v-e. Its normal-line mode is the complementary tangent-plane mode, and chordal errors agree. Bounds use known noise and deterministic inputs/transport and are pointwise at predeclared times. Truth-dependent drift is an evaluation quantity, not an online observable certificate. Invalid conditions remain NaN.

Higher-dimensional examples are generalized maintained single linear constraints under isotropic resistance. They do not silently claim a seven-joint robot whose known Jacobian would be additional information. Normal chordal error and dimension-normalized response error are separate, avoiding artificial improvement simply because a single constraint occupies a smaller fraction of a large space.

All original contact results remain unchanged. Full worker checkpoints are regenerable from seeds; compact metrics, selections, story arrays and every numerical/local-validity flag are saved. Wall clocks during parallel generation are contended and do not establish hardware-independent complexity.
