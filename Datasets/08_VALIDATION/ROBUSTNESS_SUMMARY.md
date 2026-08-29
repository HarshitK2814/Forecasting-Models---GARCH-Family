# Robustness checks — summary

Run 2026-08-29 09:51. Researcher A, plan item "Robustness checks".

## 1. Sub-sample stability (pre/post COVID-19, GJR-skewt)
COVID split at 2020-02-20 (crisis-window start). Compare `Persistence` (alpha + 0.5*gamma + beta) and the skew parameter `lam_skew` across the two halves.

| Code   | Period     |    N |    LogLik |      alpha |   gamma_asym |     beta |   Persistence |   eta_dof |   lam_skew |
|:-------|:-----------|-----:|----------:|-----------:|-------------:|---------:|--------------:|----------:|-----------:|
| SPX    | pre_COVID  | 7591 |  -9595.87 | 0.00168885 |    0.163646  | 0.9031   |      0.986612 |   7.17458 | -0.107181  |
| SPX    | post_COVID | 1635 |  -2245.83 | 0          |    0.181955  | 0.881935 |      0.972912 |   7.85674 | -0.206792  |
| NDX    | pre_COVID  | 7591 | -12793.9  | 0.0258748  |    0.110377  | 0.913322 |      0.994386 |   9.14691 | -0.119122  |
| NDX    | post_COVID | 1635 |  -2778.87 | 0.00332844 |    0.148316  | 0.893961 |      0.971448 |   9.09169 | -0.192561  |
| UKX    | pre_COVID  | 7613 |  -9951.79 | 0.00970477 |    0.122818  | 0.914023 |      0.985136 |  11.4807  | -0.0916284 |
| UKX    | post_COVID | 1641 |  -1927.06 | 0.026766   |    0.20297   | 0.817203 |      0.945454 |   5.06786 | -0.125919  |
| DAX    | pre_COVID  | 7613 | -11846.7  | 0.0160176  |    0.121065  | 0.910836 |      0.987386 |   8.91399 | -0.0970138 |
| DAX    | post_COVID | 1656 |  -2362.75 | 0.0156226  |    0.188041  | 0.855665 |      0.965308 |   5.8452  | -0.127239  |
| NKY    | pre_COVID  | 7400 | -12401.2  | 0.0250582  |    0.132916  | 0.889572 |      0.981088 |   8.18787 | -0.0672094 |
| NKY    | post_COVID | 1589 |  -2649.76 | 0.046491   |    0.164945  | 0.79237  |      0.921334 |   7.65814 | -0.0490853 |
| HSI    | pre_COVID  | 7441 | -12303.2  | 0.0297685  |    0.0697817 | 0.923397 |      0.988057 |   7.06307 | -0.0302551 |
| HSI    | post_COVID | 1599 |  -2844.84 | 0.0518716  |    0.0427135 | 0.868775 |      0.942003 |   8.19553 |  0.0241579 |

## 2. Innovation distribution (Normal vs Student-t vs skew-t)
AIC differences of 2+ units are conventionally decisive. `DeltaAIC_t_vs_Normal` and `DeltaAIC_GJRskewt_vs_t` should both be positive and large if heavy tails and asymmetry are real features of the data, not overfitting.

| Code   |   AIC_Normal |   AIC_t |   AIC_GJR_skewt |   DeltaAIC_t_vs_Normal |   DeltaAIC_GJRskewt_vs_t | BestSpec     |   Persistence_Normal |   Persistence_GJRskewt |
|:-------|-------------:|--------:|----------------:|-----------------------:|-------------------------:|:-------------|---------------------:|-----------------------:|
| DAX    |      29119   | 28696.9 |         28466.5 |                422.109 |                  230.385 | EGARCH-skewt |             0.980946 |               0.9855   |
| HSI    |      30862.4 | 30414.4 |         30351.4 |                448.008 |                   63.002 | EGARCH-skewt |             0.987576 |               0.986817 |
| NDX    |      31667.5 | 31415.7 |         31196.8 |                251.793 |                  218.896 | EGARCH-skewt |             0.989355 |               0.992122 |
| NKY    |      30644.9 | 30322.9 |         30145   |                321.936 |                  177.951 | EGARCH-skewt |             0.974606 |               0.975795 |
| SPX    |      24523.1 | 24057.3 |         23731.7 |                465.783 |                  325.588 | EGARCH-skewt |             0.98299  |               0.985171 |
| UKX    |      24334   | 24067   |         23835   |                266.974 |                  232.058 | EGARCH-skewt |             0.979677 |               0.980223 |

## 3. Sampling-frequency sensitivity (Hansen-Lunde scale factor, 5/10/15/30-min RV)
If `PctDiff_*_vs_5min` is small (a few percent), the choice of 5-min sampling for the primary realized-measure series is not doing unacknowledged work relative to coarser, even-lower-noise alternatives.

| Code   |   ScaleFactor_5min |   N_5min |   ScaleFactor_10min |   N_10min |   ScaleFactor_15min |   N_15min |   ScaleFactor_30min |   N_30min |   PctDiff_10min_vs_5min |   PctDiff_15min_vs_5min |   PctDiff_30min_vs_5min |
|:-------|-------------------:|---------:|--------------------:|----------:|--------------------:|----------:|--------------------:|----------:|------------------------:|------------------------:|------------------------:|
| SPX    |            1.91102 |     3754 |             2.00911 |      3754 |             2.08741 |      3754 |             2.2481  |      3754 |                 5.13297 |               9.23029   |                17.6389  |
| NDX    |            1.82135 |     3732 |             1.90783 |      3732 |             1.99013 |      3732 |             2.18777 |      3732 |                 4.74847 |               9.26671   |                20.1183  |
| UKX    |            1.86004 |     3727 |             1.81923 |      3726 |             1.85875 |      3726 |             2.0079  |      3724 |                -2.19421 |              -0.0695563 |                 7.94916 |
| DAX    |            1.7099  |     3306 |             1.78653 |      3306 |             1.8126  |      3306 |             1.97202 |      3306 |                 4.48146 |               6.0065    |                15.3298  |
| NKY    |            2.9745  |     3542 |             3.07651 |      3540 |             3.27092 |      3540 |             3.51132 |      3537 |                 3.42946 |               9.96527   |                18.0472  |
| HSI    |            2.31202 |     3575 |             2.44635 |      3575 |             2.57116 |      3575 |             2.96191 |      3571 |                 5.81008 |              11.2086    |                28.1093  |

## 4. Refit-cadence sensitivity (SPX, GJR-skewt, 21-day vs 63-day expanding refit)
`Corr_63v21` close to 1 and a small `MeanAbsRelDiff_63v21` mean the rolling engine's REFIT_EVERY=21 choice is not materially different from a coarser, cheaper cadence - i.e. the 21-day default is a compute-cost choice, not a result-changing one.

| Code   |   Cadence |   NRefits |   MeanSigmaHat |   Elapsed_s |   MeanAbsRelDiff_63v21 |   MaxAbsRelDiff_63v21 |   Corr_63v21 |
|:-------|----------:|----------:|---------------:|------------:|-----------------------:|----------------------:|-------------:|
| SPX    |        21 |       155 |     0.00952242 |     21.7765 |             0.00136987 |             0.0314237 |     0.999977 |
| SPX    |        63 |        52 |     0.00952475 |     12.3187 |             0.00136987 |             0.0314237 |     0.999977 |

## 5. NKY missing-RV robustness (RealGARCH, QLIKE against RVProxy)
Does the realized-information result depend on including NKY, or on the days its realized measure was recursion-imputed (2016-17 feed outage; causal Hansen-Lunde warm-up window)? `NKY_pct_days_imputed` is informational (Note column), not a loss.

| Comparison                 |   N_obs |      QLIKE |          RMSE | Note   |
|:---------------------------|--------:|-----------:|--------------:|:-------|
| all_six_markets            |   17310 |   0.191769 |   0.000227418 | nan    |
| five_markets_excl_NKY      |   15095 |   0.187605 |   0.000222031 | nan    |
| NKY_observed_RV_days       |    2176 |   0.22116  |   0.000263228 | nan    |
| NKY_imputed_recursion_days |      39 |   0.163862 |   9.27851e-05 | nan    |
| NKY_pct_days_imputed       |    3057 | nan        | nan           | 27.5%  |
