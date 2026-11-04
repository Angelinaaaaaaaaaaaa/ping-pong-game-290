
### Does preparation matter? (one-sided, matched − mismatched preparation)

| profile | mechanism | gate (mech − off) | matched − mismatched |
|---|---|---|---|
| aggressive | preparation | -0.112 [-0.175, -0.052] | +0.424 [+0.354, +0.493] |
| aggressive | preparation_cal | -0.113 [-0.173, -0.050] | +0.389 [+0.316, +0.462] |
| moderate_aggressive | preparation_cal | -0.073 [-0.143, -0.007] | +0.467 [+0.400, +0.533] |

### Rally length (both players random choices)

| profile | condition | truncated | mean steps | returns / rally |
|---|---|---|---|---|
| aggressive | off | 0.285 | 1160 | 16.32 |
| aggressive | preparation | 0.140 | 849 | 11.62 |
| aggressive | preparation_cal | 0.115 | 856 | 11.56 |
| moderate_aggressive | off | 0.520 | 1681 | 25.93 |
| moderate_aggressive | preparation_cal | 0.215 | 981 | 14.53 |

### Table 1 — aggressive (mean ± sd over seeds)

| Method | Static mean | Static min | Random | IBR (Q) | IBR (sim) | Other proposed |
|---|---|---|---|---|---|---|
| α-potential, PGA γ0.99 | 0.728 ± 0.021 | 0.525 ± 0.018 | 0.632 ± 0.020 | 0.610 ± 0.043 | 0.258 ± 0.038 | 0.308 ± 0.083 vs sampled, 0.247 ± 0.067 vs argmax |
| Self-play, sampled | 0.761 ± 0.011 | 0.565 ± 0.036 | 0.752 ± 0.033 | 0.742 ± 0.026 | 0.317 ± 0.031 | 0.692 ± 0.083 |
| Self-play, argmax | 0.797 ± 0.014 | 0.568 ± 0.032 | 0.823 ± 0.023 | 0.810 ± 0.046 | 0.492 ± 0.047 | 0.753 ± 0.067 |

Strategic computation per decision (aggressive): α-potential PGA 221 ms, simulated IBR 26.9 s

### α-potential ablations — aggressive (seed 0 unless noted)

| Phi run | train rallies | Q1 MSE (held-out) | α max | α p95 | α median | best epoch | static mean | static min | random | IBR (Q) |
|---|---|---|---|---|---|---|---|---|---|---|
| g07_s0 | 3411 | 0.185 | 1.88 | 0.77 | 0.20 | 1150 | 0.708 | 0.505 | 0.635 | 0.575 |
| g095_s0 | 3411 | 0.517 | 2.27 | 1.46 | 0.31 | 1000 | 0.687 | 0.285 | 0.660 | 0.620 |
| g099_mse_s0 | 3411 | 0.754 | 3.60 | 1.74 | 0.42 | 1000 | 0.697 | 0.320 | 0.615 | 0.565 |
| g099_n1705_s0 | 1705 | 0.926 | 2.34 | 1.93 | 0.15 | 1000 | 0.715 | 0.470 | 0.590 | 0.540 |
| g099_n426_s0 | 426 | 1.053 | 2.59 | 1.72 | 0.19 | 1000 | 0.689 | 0.405 | 0.640 | 0.505 |
| g099_n852_s0 | 852 | 1.032 | 2.34 | 1.89 | 0.15 | 975 | 0.683 | 0.395 | 0.575 | 0.470 |
| g099_s0 | 3411 | 0.754 | 2.36 | 1.85 | 0.22 | 1000 | 0.705 | 0.505 | 0.620 | 0.560 |
| g099_s1 | 3411 | 0.787 | 2.35 | 1.83 | 0.20 | 800 | 0.733 | 0.540 | 0.620 | 0.635 |
| g099_s2 | 3411 | 0.841 | 2.37 | 1.87 | 0.21 | 975 | 0.746 | 0.530 | 0.655 | 0.635 |
| g099_trainloss_s0 | 3790 | 0.731 | 2.34 | 1.83 | 0.27 | 1042 | 0.716 | 0.435 | 0.670 | 0.620 |
| g09_s0 | 3411 | 0.370 | 2.28 | 1.19 | 0.29 | 1325 | 0.697 | 0.420 | 0.670 | 0.575 |

### Self-play discount ablation — aggressive (sampled, seed 0)

| γ | static mean | static min | random |
|---|---|---|---|
| 0.99 | 0.766 | 0.525 | 0.785 |
| 0.9 | 0.726 | 0.530 | 0.705 |
| 0.7 | 0.744 | 0.495 | 0.755 |

### Table 1 — moderate_aggressive (mean ± sd over seeds)

| Method | Static mean | Static min | Random | IBR (Q) | IBR (sim) | Other proposed |
|---|---|---|---|---|---|---|
| α-potential, PGA γ0.99 | 0.712 ± 0.021 | 0.345 ± 0.097 | 0.707 ± 0.023 | 0.575 ± 0.015 | 0.438 ± 0.022 | 0.305 ± 0.083 vs sampled, 0.193 ± 0.105 vs argmax |
| Self-play, sampled | 0.792 ± 0.007 | 0.432 ± 0.059 | 0.777 ± 0.032 | 0.738 ± 0.008 | 0.542 ± 0.069 | 0.695 ± 0.083 |
| Self-play, argmax | 0.850 ± 0.032 | 0.540 ± 0.060 | 0.875 ± 0.030 | 0.823 ± 0.025 | 0.642 ± 0.031 | 0.807 ± 0.105 |

Strategic computation per decision (moderate_aggressive): α-potential PGA 223 ms, simulated IBR 26.6 s

### α-potential ablations — moderate_aggressive (seed 0 unless noted)

| Phi run | train rallies | Q1 MSE (held-out) | α max | α p95 | α median | best epoch | static mean | static min | random | IBR (Q) |
|---|---|---|---|---|---|---|---|---|---|---|
| g07_s0 | 3451 | 0.152 | 1.78 | 0.70 | 0.18 | 1500 | 0.656 | 0.220 | 0.680 | 0.550 |
| g095_s0 | 3451 | 0.516 | 2.23 | 1.58 | 0.30 | 975 | 0.692 | 0.335 | 0.670 | 0.655 |
| g099_mse_s0 | 3451 | 0.791 | 3.57 | 1.87 | 0.44 | 1275 | 0.702 | 0.455 | 0.665 | 0.570 |
| g099_n1725_s0 | 1725 | 0.912 | 2.63 | 1.94 | 0.16 | 975 | 0.740 | 0.375 | 0.730 | 0.660 |
| g099_n431_s0 | 431 | 1.010 | 2.57 | 1.74 | 0.19 | 975 | 0.760 | 0.430 | 0.725 | 0.565 |
| g099_n862_s0 | 862 | 0.925 | 2.53 | 1.87 | 0.17 | 1025 | 0.798 | 0.415 | 0.770 | 0.690 |
| g099_s0 | 3451 | 0.791 | 2.37 | 1.92 | 0.18 | 1025 | 0.694 | 0.455 | 0.685 | 0.560 |
| g099_s1 | 3451 | 0.672 | 2.32 | 1.86 | 0.17 | 1000 | 0.735 | 0.310 | 0.705 | 0.590 |
| g099_s2 | 3451 | 0.750 | 2.37 | 1.91 | 0.16 | 900 | 0.707 | 0.270 | 0.730 | 0.575 |
| g099_trainloss_s0 | 3834 | 0.755 | 2.45 | 1.87 | 0.19 | 1447 | 0.660 | 0.250 | 0.635 | 0.565 |
| g09_s0 | 3451 | 0.334 | 2.15 | 1.19 | 0.28 | 1050 | 0.668 | 0.260 | 0.675 | 0.580 |

### Self-play discount ablation — moderate_aggressive (sampled, seed 0)

| γ | static mean | static min | random |
|---|---|---|---|
| 0.99 | 0.790 | 0.450 | 0.795 |
| 0.9 | 0.797 | 0.415 | 0.830 |
| 0.7 | 0.838 | 0.555 | 0.750 |
