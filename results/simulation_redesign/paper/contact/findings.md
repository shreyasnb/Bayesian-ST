# Contact-normal study

The observations use the real conditional model u=P_X v+e. P_X is a tangent-plane projector; its normal line has Bingham parameter -A. Known applied free velocity and measured wrist rotation are supplied to every corresponding estimator. True surface normals and positions are scoring-only.

The paper profile has 32 validation and 128 independent test datasets per static input program, 16 validation and 64 test trajectories per physical condition. Query coordinates and reporting times are not independent experimental repeats.

Every static command has the same Euclidean norm. Balanced and directionally concentrated programs therefore have identical total energy at every horizon. Their local Fisher information need not be the same. The stationary normal is Haar-distributed in the Monte Carlo comparison; the fixed e3 density illustration is a separate preassigned example.

## Comparators and decisions

The modal plane equals independent weighted conditional projector least squares. The mean, mode and matched isotropized normal mean use identical A in the shape comparison. A separate globally validated scalar contracts the modal plane toward the known-rank isotropic mean. Ridge regression also regularizes toward that known-rank isotropic mean, rather than zero; it uses force and measured velocity. Residual PCA uses v-u and therefore has access to both signals, while discarding some conditional-likelihood information.

The same-parameter local Gaussian uses the modal Hessian of -A. Its exponential-map mean is integrated; local cut-locus mass and numerical refinement have separate flags. Tied precision fails explicitly. The physical recursive Gaussian uses a six-observation data-only initialization, Joseph correction and the same measured frame rotations; Q is selected separately on validation. It is not the same-parameter control.

For the physical contact, each method selects its own beta or Q on independent terminal response loss. The mean and matched-isotropization rows share beta by construction. The identity-frame ablation retains the same beta and data. G=R_t^T R_(t-1) is derived from wrist orientation independent of the hidden surface normal, not an oracle normal alignment.

The expected decision is v^B=[I+lambda(I-Pbar)]^(-1) v_des. Cost uses a soft inadmissible-command penalty, not a guarantee of feasible contact motion or a closed-loop controller. The expected penalty is linear in I-Pbar. Unconstrained ridge maps are scored for response prediction only and have no Eq.36 cost.

## Static paired response comparisons

      kind  t    reference     endpoint  trials  finite_pairs          gain         lower        upper                                                 positive  numerical_flags_retained
  balanced  1    isotropic response_mse     128           128  1.709378e-06  1.832198e-09 3.416924e-06 reference minus posterior mean; independent trajectories                      True
  balanced  1      laplace response_mse     128           128  2.656451e-06  1.129686e-06 4.183217e-06 reference minus posterior mean; independent trajectories                      True
  balanced  1         mode response_mse     128           128  5.460662e-05  4.364441e-05 6.556882e-05 reference minus posterior mean; independent trajectories                      True
  balanced  1  regularized response_mse     128           128  2.806083e-06  3.579724e-08 5.576369e-06 reference minus posterior mean; independent trajectories                      True
  balanced  1 residual_pca response_mse     128           128  7.355515e-05  6.242402e-05 8.468628e-05 reference minus posterior mean; independent trajectories                      True
  balanced  1        ridge response_mse     128           128  3.327577e-06  8.074803e-07 5.847673e-06 reference minus posterior mean; independent trajectories                      True
  balanced  2    isotropic response_mse     128           128  1.262711e-06  1.148156e-07 2.410607e-06 reference minus posterior mean; independent trajectories                      True
  balanced  2      laplace response_mse     128           128  1.778651e-06  1.150515e-08 3.545797e-06 reference minus posterior mean; independent trajectories                      True
  balanced  2         mode response_mse     128           128  2.899845e-05  2.021687e-05 3.778003e-05 reference minus posterior mean; independent trajectories                      True
  balanced  2  regularized response_mse     128           128  3.330674e-06  6.163180e-08 6.599716e-06 reference minus posterior mean; independent trajectories                      True
  balanced  2 residual_pca response_mse     128           128  7.037695e-05  5.816052e-05 8.259337e-05 reference minus posterior mean; independent trajectories                      True
  balanced  2        ridge response_mse     128           128  1.100727e-05  6.116006e-06 1.589854e-05 reference minus posterior mean; independent trajectories                      True
  balanced  4    isotropic response_mse     128           128  9.489409e-07  1.987267e-07 1.699155e-06 reference minus posterior mean; independent trajectories                      True
  balanced  4      laplace response_mse     128           128  1.203271e-06 -6.432902e-08 2.470871e-06 reference minus posterior mean; independent trajectories                      True
  balanced  4         mode response_mse     128           128  1.431052e-05  8.708698e-06 1.991234e-05 reference minus posterior mean; independent trajectories                      True
  balanced  4  regularized response_mse     128           128  3.618218e-06  1.278980e-06 5.957456e-06 reference minus posterior mean; independent trajectories                      True
  balanced  4 residual_pca response_mse     128           128  5.657851e-05  4.477049e-05 6.838653e-05 reference minus posterior mean; independent trajectories                      True
  balanced  4        ridge response_mse     128           128  3.077400e-05  2.649008e-05 3.505791e-05 reference minus posterior mean; independent trajectories                      True
  balanced  8    isotropic response_mse     128           128  5.676121e-07  7.178454e-08 1.063440e-06 reference minus posterior mean; independent trajectories                      True
  balanced  8      laplace response_mse     128           128  5.669078e-07 -1.248613e-07 1.258677e-06 reference minus posterior mean; independent trajectories                      True
  balanced  8         mode response_mse     128           128  5.644063e-06  2.515582e-06 8.772543e-06 reference minus posterior mean; independent trajectories                      True
  balanced  8  regularized response_mse     128           128  2.676985e-06  6.725878e-07 4.681382e-06 reference minus posterior mean; independent trajectories                      True
  balanced  8 residual_pca response_mse     128           128  5.152679e-05  4.059909e-05 6.245449e-05 reference minus posterior mean; independent trajectories                      True
  balanced  8        ridge response_mse     128           128  2.378501e-05  2.052187e-05 2.704814e-05 reference minus posterior mean; independent trajectories                      True
  balanced 16    isotropic response_mse     128           128  1.050569e-07 -1.312629e-07 3.413767e-07 reference minus posterior mean; independent trajectories                      True
  balanced 16      laplace response_mse     128           128  1.766803e-07 -2.408530e-07 5.942136e-07 reference minus posterior mean; independent trajectories                      True
  balanced 16         mode response_mse     128           128  1.109792e-06 -5.100362e-07 2.729621e-06 reference minus posterior mean; independent trajectories                      True
  balanced 16  regularized response_mse     128           128  6.434147e-07 -6.699436e-07 1.956773e-06 reference minus posterior mean; independent trajectories                      True
  balanced 16 residual_pca response_mse     128           128  3.969457e-05  3.185144e-05 4.753771e-05 reference minus posterior mean; independent trajectories                      True
  balanced 16        ridge response_mse     128           128  2.424032e-05  2.180649e-05 2.667414e-05 reference minus posterior mean; independent trajectories                      True
  balanced 32    isotropic response_mse     128           128 -5.905903e-09 -1.281002e-08 9.982107e-10 reference minus posterior mean; independent trajectories                      True
  balanced 32      laplace response_mse     128           128  1.438880e-09 -1.448232e-09 4.325993e-09 reference minus posterior mean; independent trajectories                      True
  balanced 32         mode response_mse     128           128  1.463284e-07  8.348559e-08 2.091712e-07 reference minus posterior mean; independent trajectories                      True
  balanced 32  regularized response_mse     128           128  2.166521e-08 -5.060826e-09 4.839125e-08 reference minus posterior mean; independent trajectories                      True
  balanced 32 residual_pca response_mse     128           128  2.402855e-05  1.800832e-05 3.004878e-05 reference minus posterior mean; independent trajectories                      True
  balanced 32        ridge response_mse     128           128  1.897555e-05  1.729343e-05 2.065766e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  1    isotropic response_mse     128           128  1.709378e-06  1.832198e-09 3.416924e-06 reference minus posterior mean; independent trajectories                      True
unbalanced  1      laplace response_mse     128           128  2.656451e-06  1.129686e-06 4.183217e-06 reference minus posterior mean; independent trajectories                      True
unbalanced  1         mode response_mse     128           128  5.460662e-05  4.364441e-05 6.556882e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  1  regularized response_mse     128           128  2.806083e-06  3.579724e-08 5.576369e-06 reference minus posterior mean; independent trajectories                      True
unbalanced  1 residual_pca response_mse     128           128  7.355515e-05  6.242402e-05 8.468628e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  1        ridge response_mse     128           128  3.327577e-06  8.074803e-07 5.847673e-06 reference minus posterior mean; independent trajectories                      True
unbalanced  2    isotropic response_mse     128           128  2.567548e-06  7.413225e-07 4.393774e-06 reference minus posterior mean; independent trajectories                      True
unbalanced  2      laplace response_mse     128           128  1.915497e-06  3.265849e-07 3.504408e-06 reference minus posterior mean; independent trajectories                      True
unbalanced  2         mode response_mse     128           128  3.967675e-05  2.963644e-05 4.971706e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  2  regularized response_mse     128           128  5.253560e-06  2.088843e-06 8.418277e-06 reference minus posterior mean; independent trajectories                      True
unbalanced  2 residual_pca response_mse     128           128  6.926585e-05  5.760679e-05 8.092492e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  2        ridge response_mse     128           128  9.272878e-06  5.207782e-06 1.333797e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  4    isotropic response_mse     128           128  2.408726e-06  9.066761e-07 3.910776e-06 reference minus posterior mean; independent trajectories                      True
unbalanced  4      laplace response_mse     128           128  7.298899e-07 -6.190492e-07 2.078829e-06 reference minus posterior mean; independent trajectories                      True
unbalanced  4         mode response_mse     128           128  1.973533e-05  1.183974e-05 2.763092e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  4  regularized response_mse     128           128  5.101525e-06  1.054801e-06 9.148250e-06 reference minus posterior mean; independent trajectories                      True
unbalanced  4 residual_pca response_mse     128           128  6.095647e-05  4.878712e-05 7.312583e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  4        ridge response_mse     128           128  2.275862e-05  1.871916e-05 2.679807e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  8    isotropic response_mse     128           128  1.961586e-06  6.472850e-07 3.275887e-06 reference minus posterior mean; independent trajectories                      True
unbalanced  8      laplace response_mse     128           128  8.222481e-07 -3.866244e-07 2.031121e-06 reference minus posterior mean; independent trajectories                      True
unbalanced  8         mode response_mse     128           128  1.121776e-05  5.111986e-06 1.732353e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  8  regularized response_mse     128           128  5.167802e-06  2.303954e-06 8.031649e-06 reference minus posterior mean; independent trajectories                      True
unbalanced  8 residual_pca response_mse     128           128  5.358090e-05  4.071900e-05 6.644280e-05 reference minus posterior mean; independent trajectories                      True
unbalanced  8        ridge response_mse     128           128  2.935474e-05  2.524269e-05 3.346680e-05 reference minus posterior mean; independent trajectories                      True
unbalanced 16    isotropic response_mse     128           128  7.595316e-07 -2.824981e-08 1.547313e-06 reference minus posterior mean; independent trajectories                      True
unbalanced 16      laplace response_mse     128           128 -1.204143e-07 -9.226594e-07 6.818308e-07 reference minus posterior mean; independent trajectories                      True
unbalanced 16         mode response_mse     128           128  4.419107e-06  6.791402e-07 8.159075e-06 reference minus posterior mean; independent trajectories                      True
unbalanced 16  regularized response_mse     128           128  2.110259e-06 -6.139047e-07 4.834423e-06 reference minus posterior mean; independent trajectories                      True
unbalanced 16 residual_pca response_mse     128           128  5.527348e-05  4.204828e-05 6.849867e-05 reference minus posterior mean; independent trajectories                      True
unbalanced 16        ridge response_mse     128           128  3.689814e-05  3.317474e-05 4.062155e-05 reference minus posterior mean; independent trajectories                      True
unbalanced 32    isotropic response_mse     128           128  7.212198e-07 -1.083046e-07 1.550744e-06 reference minus posterior mean; independent trajectories                      True
unbalanced 32      laplace response_mse     128           128  2.144078e-07 -3.667870e-07 7.956026e-07 reference minus posterior mean; independent trajectories                      True
unbalanced 32         mode response_mse     128           128  3.515270e-06  6.561495e-08 6.964925e-06 reference minus posterior mean; independent trajectories                      True
unbalanced 32  regularized response_mse     128           128  2.644176e-06 -4.891095e-07 5.777462e-06 reference minus posterior mean; independent trajectories                      True
unbalanced 32 residual_pca response_mse     128           128  4.317056e-05  3.133467e-05 5.500645e-05 reference minus posterior mean; independent trajectories                      True
unbalanced 32        ridge response_mse     128           128  4.085928e-05  3.753869e-05 4.417987e-05 reference minus posterior mean; independent trajectories                      True

## Physical paired response comparisons

condition   t     reference     endpoint  trials  finite_pairs          gain         lower         upper                                                 positive  numerical_flags_retained
    plane   8      gaussian response_mse      64            64  6.063213e-08 -4.930719e-08  1.705715e-07 reference minus posterior mean; independent trajectories                      True
    plane   8     isotropic response_mse      64            64  5.312710e-10 -8.701888e-10  1.932731e-09 reference minus posterior mean; independent trajectories                      True
    plane   8 mean_identity response_mse      64            64  1.208181e-06  5.526661e-07  1.863695e-06 reference minus posterior mean; independent trajectories                      True
    plane   8          mode response_mse      64            64  5.366953e-08  3.040257e-08  7.693648e-08 reference minus posterior mean; independent trajectories                      True
    plane   8 mode_identity response_mse      64            64  1.291968e-06  6.168310e-07  1.967104e-06 reference minus posterior mean; independent trajectories                      True
    plane   8   mode_same_A response_mse      64            64  5.366953e-08  3.040257e-08  7.693648e-08 reference minus posterior mean; independent trajectories                      True
    plane   8  residual_pca response_mse      64            64  5.933963e-06  3.895770e-06  7.972156e-06 reference minus posterior mean; independent trajectories                      True
    plane   8         ridge response_mse      64            64  1.487816e-05  1.289730e-05  1.685902e-05 reference minus posterior mean; independent trajectories                      True
    plane  40      gaussian response_mse      64            64  3.089346e-08 -3.416022e-09  6.520293e-08 reference minus posterior mean; independent trajectories                      True
    plane  40     isotropic response_mse      64            64  1.415762e-11 -9.913053e-12  3.822829e-11 reference minus posterior mean; independent trajectories                      True
    plane  40 mean_identity response_mse      64            64  7.071630e-06  6.243011e-06  7.900250e-06 reference minus posterior mean; independent trajectories                      True
    plane  40          mode response_mse      64            64  1.805950e-09  8.159330e-10  2.795967e-09 reference minus posterior mean; independent trajectories                      True
    plane  40 mode_identity response_mse      64            64  7.106757e-06  6.274358e-06  7.939156e-06 reference minus posterior mean; independent trajectories                      True
    plane  40   mode_same_A response_mse      64            64  1.805950e-09  8.159330e-10  2.795967e-09 reference minus posterior mean; independent trajectories                      True
    plane  40  residual_pca response_mse      64            64  1.137387e-06  6.855563e-07  1.589218e-06 reference minus posterior mean; independent trajectories                      True
    plane  40         ridge response_mse      64            64  2.705732e-06  2.365624e-06  3.045840e-06 reference minus posterior mean; independent trajectories                      True
    plane  80      gaussian response_mse      64            64  1.460201e-08 -2.615910e-10  2.946562e-08 reference minus posterior mean; independent trajectories                      True
    plane  80     isotropic response_mse      64            64  1.576214e-12 -1.996586e-12  5.149013e-12 reference minus posterior mean; independent trajectories                      True
    plane  80 mean_identity response_mse      64            64  2.010742e-05  1.907298e-05  2.114186e-05 reference minus posterior mean; independent trajectories                      True
    plane  80          mode response_mse      64            64  3.316065e-10  1.209004e-10  5.423126e-10 reference minus posterior mean; independent trajectories                      True
    plane  80 mode_identity response_mse      64            64  2.015483e-05  1.911784e-05  2.119181e-05 reference minus posterior mean; independent trajectories                      True
    plane  80   mode_same_A response_mse      64            64  3.316065e-10  1.209004e-10  5.423126e-10 reference minus posterior mean; independent trajectories                      True
    plane  80  residual_pca response_mse      64            64  5.952123e-07  3.736505e-07  8.167741e-07 reference minus posterior mean; independent trajectories                      True
    plane  80         ridge response_mse      64            64  1.555866e-06  1.341570e-06  1.770162e-06 reference minus posterior mean; independent trajectories                      True
    plane 160      gaussian response_mse      64            64  5.632408e-09 -1.743724e-10  1.143919e-08 reference minus posterior mean; independent trajectories                      True
    plane 160     isotropic response_mse      64            64  2.111100e-13 -2.545502e-13  6.767702e-13 reference minus posterior mean; independent trajectories                      True
    plane 160 mean_identity response_mse      64            64  1.076043e-06  9.261936e-07  1.225892e-06 reference minus posterior mean; independent trajectories                      True
    plane 160          mode response_mse      64            64  7.894158e-11  3.134131e-11  1.265419e-10 reference minus posterior mean; independent trajectories                      True
    plane 160 mode_identity response_mse      64            64  1.077501e-06  9.274378e-07  1.227565e-06 reference minus posterior mean; independent trajectories                      True
    plane 160   mode_same_A response_mse      64            64  7.894158e-11  3.134131e-11  1.265419e-10 reference minus posterior mean; independent trajectories                      True
    plane 160  residual_pca response_mse      64            64  2.509476e-07  1.424345e-07  3.594607e-07 reference minus posterior mean; independent trajectories                      True
    plane 160         ridge response_mse      64            64  6.940631e-07  5.839152e-07  8.042109e-07 reference minus posterior mean; independent trajectories                      True
   sphere   8      gaussian response_mse      64            64 -9.189805e-07 -5.120628e-06  3.282667e-06 reference minus posterior mean; independent trajectories                      True
   sphere   8     isotropic response_mse      64            64  1.172895e-06 -6.952783e-07  3.041068e-06 reference minus posterior mean; independent trajectories                      True
   sphere   8 mean_identity response_mse      64            64 -2.605711e-07 -1.924459e-06  1.403317e-06 reference minus posterior mean; independent trajectories                      True
   sphere   8          mode response_mse      64            64  4.807600e-06 -3.022967e-06  1.263817e-05 reference minus posterior mean; independent trajectories                      True
   sphere   8 mode_identity response_mse      64            64  3.662324e-06 -5.275672e-06  1.260032e-05 reference minus posterior mean; independent trajectories                      True
   sphere   8   mode_same_A response_mse      64            64  4.807600e-06 -3.022967e-06  1.263817e-05 reference minus posterior mean; independent trajectories                      True
   sphere   8  residual_pca response_mse      64            64  3.367990e-05  1.885529e-05  4.850451e-05 reference minus posterior mean; independent trajectories                      True
   sphere   8         ridge response_mse      64            64  3.786301e-05  3.199790e-05  4.372813e-05 reference minus posterior mean; independent trajectories                      True
   sphere  40      gaussian response_mse      64            64 -2.040738e-06 -4.916517e-06  8.350421e-07 reference minus posterior mean; independent trajectories                      True
   sphere  40     isotropic response_mse      64            64  5.080083e-07 -5.552830e-07  1.571300e-06 reference minus posterior mean; independent trajectories                      True
   sphere  40 mean_identity response_mse      64            64  1.073935e-07 -4.314038e-07  6.461907e-07 reference minus posterior mean; independent trajectories                      True
   sphere  40          mode response_mse      64            64  2.338485e-06 -2.075144e-06  6.752113e-06 reference minus posterior mean; independent trajectories                      True
   sphere  40 mode_identity response_mse      64            64  2.959317e-06 -2.742348e-06  8.660983e-06 reference minus posterior mean; independent trajectories                      True
   sphere  40   mode_same_A response_mse      64            64  2.338485e-06 -2.075144e-06  6.752113e-06 reference minus posterior mean; independent trajectories                      True
   sphere  40  residual_pca response_mse      64            64  9.442344e-06  2.174952e-06  1.670974e-05 reference minus posterior mean; independent trajectories                      True
   sphere  40         ridge response_mse      64            64  4.810943e-05  4.454536e-05  5.167350e-05 reference minus posterior mean; independent trajectories                      True
   sphere  80      gaussian response_mse      64            64 -5.656561e-06 -1.040565e-05 -9.074737e-07 reference minus posterior mean; independent trajectories                      True
   sphere  80     isotropic response_mse      64            64  5.916900e-07 -6.858555e-07  1.869236e-06 reference minus posterior mean; independent trajectories                      True
   sphere  80 mean_identity response_mse      64            64  2.248395e-07 -7.860175e-07  1.235696e-06 reference minus posterior mean; independent trajectories                      True
   sphere  80          mode response_mse      64            64  2.611349e-06 -2.661142e-06  7.883840e-06 reference minus posterior mean; independent trajectories                      True
   sphere  80 mode_identity response_mse      64            64  1.582727e-06 -3.503143e-06  6.668598e-06 reference minus posterior mean; independent trajectories                      True
   sphere  80   mode_same_A response_mse      64            64  2.611349e-06 -2.661142e-06  7.883840e-06 reference minus posterior mean; independent trajectories                      True
   sphere  80  residual_pca response_mse      64            64  9.442938e-06  7.888836e-07  1.809699e-05 reference minus posterior mean; independent trajectories                      True
   sphere  80         ridge response_mse      64            64  4.064935e-05  3.630829e-05  4.499041e-05 reference minus posterior mean; independent trajectories                      True
   sphere 160      gaussian response_mse      64            64 -2.535738e-07 -4.264083e-07 -8.073934e-08 reference minus posterior mean; independent trajectories                      True
   sphere 160     isotropic response_mse      64            64 -2.815671e-09 -5.319908e-09 -3.114340e-10 reference minus posterior mean; independent trajectories                      True
   sphere 160 mean_identity response_mse      64            64  2.958788e-06  2.233475e-06  3.684101e-06 reference minus posterior mean; independent trajectories                      True
   sphere 160          mode response_mse      64            64  2.285201e-08  2.632212e-09  4.307180e-08 reference minus posterior mean; independent trajectories                      True
   sphere 160 mode_identity response_mse      64            64  3.059658e-06  2.304596e-06  3.814721e-06 reference minus posterior mean; independent trajectories                      True
   sphere 160   mode_same_A response_mse      64            64  2.285201e-08  2.632212e-09  4.307180e-08 reference minus posterior mean; independent trajectories                      True
   sphere 160  residual_pca response_mse      64            64  1.194261e-06  2.006765e-07  2.187845e-06 reference minus posterior mean; independent trajectories                      True
   sphere 160         ridge response_mse      64            64  4.369988e-05  4.240928e-05  4.499047e-05 reference minus posterior mean; independent trajectories                      True

## Numerical and interpretation limits

{
  "seconds": 123.9123755720002,
  "rows": {
    "static_metrics": 10752,
    "static_fisher": 1536,
    "dynamic_metrics": 4608,
    "dynamic_bounds": 1024,
    "higher_dimension": 384,
    "bounds": 256,
    "paired_static": 72,
    "paired_dynamic": 64,
    "story_checks": 32
  },
  "static_numerical_failures": 0,
  "static_local_gaussian_unresolved": 323,
  "dynamic_numerical_failures": 0,
  "dynamic_local_gaussian_unresolved": 7,
  "higher_numerical_failures": 0,
  "story_numerical_failures": 0,
  "bound_valid": 448,
  "bound_violations": 0,
  "maximum_modal_audit_error": 1.9895196601282805e-12,
  "maximum_decision_identity_error": 6.505213034913027e-19
}

Posterior conditional mass, numerical integration accuracy and repeated-data inclusion are different claims. Forgetting under a changing normal does not inherit a stationary confidence guarantee. Scalar intervals are clean-response quantiles, not noisy velocity prediction intervals.

The long stationary cohort checks the conditional modal theorem through residual r=v-u=(nn^T)v-e. Its normal-line mode is the complementary tangent-plane mode, and chordal errors agree. Bounds use known noise and deterministic inputs/transport and are pointwise at predeclared times. Truth-dependent drift is an evaluation quantity, not an online observable certificate. Invalid conditions remain NaN.

Higher-dimensional examples are generalized maintained single linear constraints under isotropic resistance. They do not silently claim a seven-joint robot whose known Jacobian would be additional information. Normal chordal error and dimension-normalized response error are separate, avoiding artificial improvement simply because a single constraint occupies a smaller fraction of a large space.

All original contact results remain unchanged. Full worker checkpoints are regenerable from seeds; compact metrics, selections, story arrays and every numerical/local-validity flag are saved. Wall clocks during parallel generation are contended and do not establish hardware-independent complexity.
