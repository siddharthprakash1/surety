# Option-prior correction: kev_compact

- collection: `artifacts/m1/kev_compact.jsonl`
- prior+T: `PriorCorrector` (one log-prior offset per recurring action id, L2-penalised maximum likelihood on cal; page-element ids keep 0), then T fitted on the corrected cal rows; +prior rows refit both on every draw
- the offsets, T and the gates all use the same cal rows, so the finite-sample statements are approximate, as in M1; test_shift (held-out sites) has no guarantee

## Offsets fitted on the original cal split

BLOCKED -2.85, WAIT -0.84, GO_BACK -0.51, PRESS_ESCAPE -0.14, DONE -0.11, PRESS_ENTER +0.02, SCROLL_UP +0.27, SCROLL_DOWN +0.69, TYPE_TEXT +1.23, CLICK +2.67

## Accuracy and calibration (original split)

| split | task_type | n | acc debiased+T | ECE debiased+T | acc prior+T | ECE prior+T |
|---|---|---|---|---|---|---|
| cal | all | 600 | 0.477 | 0.095 | 0.590 | 0.117 |
| cal | action_prediction | 300 | 0.350 | 0.137 | 0.577 | 0.161 |
| cal | element_grounding | 300 | 0.603 | 0.222 | 0.603 | 0.130 |
| test_in | all | 400 | 0.485 | 0.086 | 0.643 | 0.074 |
| test_in | action_prediction | 200 | 0.335 | 0.125 | 0.650 | 0.093 |
| test_in | element_grounding | 200 | 0.635 | 0.233 | 0.635 | 0.127 |
| test_shift | all | 600 | 0.458 | 0.077 | 0.588 | 0.098 |
| test_shift | action_prediction | 300 | 0.310 | 0.174 | 0.570 | 0.169 |
| test_shift | element_grounding | 300 | 0.607 | 0.204 | 0.607 | 0.089 |


## Error ranking with prior+T (AUROC for 'top option wrong')

| split | task_type | n | error_rate | AUROC p_top | AUROC margin | AUROC entropy | AUROC learned |
|---|---|---|---|---|---|---|---|
| test_in | all | 400 | 0.357 | 0.750 | 0.756 | 0.738 | 0.743 |
| test_in | action_prediction | 200 | 0.350 | 0.707 | 0.689 | 0.728 | 0.687 |
| test_in | element_grounding | 200 | 0.365 | 0.807 | 0.822 | 0.755 | 0.801 |
| test_shift | all | 600 | 0.412 | 0.714 | 0.739 | 0.695 | 0.750 |
| test_shift | action_prediction | 300 | 0.430 | 0.666 | 0.671 | 0.658 | 0.673 |
| test_shift | element_grounding | 300 | 0.393 | 0.811 | 0.833 | 0.752 | 0.826 |


## Calibration draws (200; offsets, T, model and gates refit per draw)

| eps | split | policy | draws | auto_err | auto_err_2.5–97.5% | P(auto_err>eps) | automation | automation_2.5–97.5% |
|---|---|---|---|---|---|---|---|---|
| 0.05 | test_in | risk@0.05 | 200 | 0.034 | [0.015, 0.053] | 0.040 | 0.224 | [0.177, 0.263] |
| 0.05 | test_in | learned@0.05 | 200 | 0.049 | [0.022, 0.083] | 0.410 | 0.304 | [0.257, 0.363] |
| 0.05 | test_in | learned_pac@0.05 | 200 | 0.037 | [0.018, 0.065] | 0.125 | 0.273 | [0.230, 0.325] |
| 0.05 | test_in | risk@0.05 +prior | 200 | 0.051 | [0.030, 0.077] | 0.505 | 0.325 | [0.262, 0.383] |
| 0.05 | test_in | learned@0.05 +prior | 200 | 0.052 | [0.022, 0.083] | 0.490 | 0.356 | [0.290, 0.413] |
| 0.05 | test_in | learned_pac@0.05 +prior | 200 | 0.040 | [0.020, 0.068] | 0.185 | 0.306 | [0.235, 0.372] |
| 0.05 | test_shift | risk@0.05 | 200 | 0.037 | [0.027, 0.045] | 0.000 | 0.226 | [0.203, 0.252] |
| 0.05 | test_shift | learned@0.05 | 200 | 0.046 | [0.038, 0.055] | 0.130 | 0.302 | [0.278, 0.328] |
| 0.05 | test_shift | learned_pac@0.05 | 200 | 0.038 | [0.030, 0.045] | 0.005 | 0.274 | [0.242, 0.303] |
| 0.05 | test_shift | risk@0.05 +prior | 200 | 0.058 | [0.047, 0.075] | 0.785 | 0.321 | [0.273, 0.360] |
| 0.05 | test_shift | learned@0.05 +prior | 200 | 0.072 | [0.050, 0.095] | 0.940 | 0.364 | [0.310, 0.413] |
| 0.05 | test_shift | learned_pac@0.05 +prior | 200 | 0.052 | [0.037, 0.073] | 0.485 | 0.308 | [0.248, 0.362] |
| 0.1 | test_in | risk@0.1 | 200 | 0.067 | [0.043, 0.090] | 0.000 | 0.301 | [0.258, 0.338] |
| 0.1 | test_in | learned@0.1 | 200 | 0.098 | [0.065, 0.140] | 0.435 | 0.419 | [0.367, 0.485] |
| 0.1 | test_in | learned_pac@0.1 | 200 | 0.083 | [0.052, 0.125] | 0.140 | 0.388 | [0.335, 0.450] |
| 0.1 | test_in | risk@0.1 +prior | 200 | 0.088 | [0.052, 0.118] | 0.255 | 0.432 | [0.382, 0.483] |
| 0.1 | test_in | learned@0.1 +prior | 200 | 0.102 | [0.062, 0.143] | 0.540 | 0.488 | [0.422, 0.555] |
| 0.1 | test_in | learned_pac@0.1 +prior | 200 | 0.088 | [0.052, 0.128] | 0.220 | 0.453 | [0.382, 0.522] |
| 0.1 | test_shift | risk@0.1 | 200 | 0.054 | [0.040, 0.070] | 0.000 | 0.294 | [0.268, 0.315] |
| 0.1 | test_shift | learned@0.1 | 200 | 0.095 | [0.075, 0.120] | 0.260 | 0.411 | [0.375, 0.455] |
| 0.1 | test_shift | learned_pac@0.1 | 200 | 0.078 | [0.058, 0.098] | 0.020 | 0.379 | [0.343, 0.422] |
| 0.1 | test_shift | risk@0.1 +prior | 200 | 0.099 | [0.080, 0.117] | 0.445 | 0.445 | [0.410, 0.475] |
| 0.1 | test_shift | learned@0.1 +prior | 200 | 0.129 | [0.110, 0.145] | 1.000 | 0.517 | [0.477, 0.550] |
| 0.1 | test_shift | learned_pac@0.1 +prior | 200 | 0.116 | [0.095, 0.133] | 0.915 | 0.482 | [0.430, 0.522] |


## The same draws by task type

| eps | split | task_type | policy | draws | auto_err | auto_err_2.5–97.5% | P(auto_err>eps) | automation | automation_2.5–97.5% |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | test_in | action_prediction | risk@0.05 | 200 | 0.008 | [0.000, 0.021] | 0.000 | 0.081 | [0.048, 0.111] |
| 0.05 | test_in | action_prediction | learned@0.05 | 200 | 0.046 | [0.018, 0.079] | 0.385 | 0.192 | [0.146, 0.247] |
| 0.05 | test_in | action_prediction | learned_pac@0.05 | 200 | 0.034 | [0.010, 0.063] | 0.115 | 0.164 | [0.116, 0.222] |
| 0.05 | test_in | action_prediction | risk@0.05 +prior | 200 | 0.049 | [0.020, 0.081] | 0.460 | 0.273 | [0.207, 0.333] |
| 0.05 | test_in | action_prediction | learned@0.05 +prior | 200 | 0.077 | [0.030, 0.127] | 0.855 | 0.378 | [0.297, 0.459] |
| 0.05 | test_in | action_prediction | learned_pac@0.05 +prior | 200 | 0.057 | [0.024, 0.102] | 0.585 | 0.315 | [0.222, 0.397] |
| 0.05 | test_in | element_grounding | risk@0.05 | 200 | 0.059 | [0.027, 0.091] | 0.685 | 0.366 | [0.282, 0.432] |
| 0.05 | test_in | element_grounding | learned@0.05 | 200 | 0.052 | [0.020, 0.094] | 0.465 | 0.416 | [0.350, 0.486] |
| 0.05 | test_in | element_grounding | learned_pac@0.05 | 200 | 0.040 | [0.015, 0.075] | 0.230 | 0.382 | [0.330, 0.442] |
| 0.05 | test_in | element_grounding | risk@0.05 +prior | 200 | 0.052 | [0.029, 0.086] | 0.525 | 0.377 | [0.303, 0.463] |
| 0.05 | test_in | element_grounding | learned@0.05 +prior | 200 | 0.026 | [0.010, 0.054] | 0.040 | 0.333 | [0.273, 0.391] |
| 0.05 | test_in | element_grounding | learned_pac@0.05 +prior | 200 | 0.023 | [0.010, 0.043] | 0.010 | 0.298 | [0.231, 0.365] |
| 0.05 | test_shift | action_prediction | risk@0.05 | 200 | 0.015 | [0.010, 0.020] | 0.000 | 0.083 | [0.070, 0.100] |
| 0.05 | test_shift | action_prediction | learned@0.05 | 200 | 0.045 | [0.037, 0.057] | 0.130 | 0.179 | [0.157, 0.203] |
| 0.05 | test_shift | action_prediction | learned_pac@0.05 | 200 | 0.038 | [0.030, 0.047] | 0.005 | 0.160 | [0.137, 0.187] |
| 0.05 | test_shift | action_prediction | risk@0.05 +prior | 200 | 0.072 | [0.053, 0.103] | 0.990 | 0.286 | [0.233, 0.347] |
| 0.05 | test_shift | action_prediction | learned@0.05 +prior | 200 | 0.118 | [0.077, 0.160] | 1.000 | 0.406 | [0.323, 0.490] |
| 0.05 | test_shift | action_prediction | learned_pac@0.05 +prior | 200 | 0.087 | [0.057, 0.123] | 0.995 | 0.326 | [0.247, 0.410] |
| 0.05 | test_shift | element_grounding | risk@0.05 | 200 | 0.059 | [0.040, 0.070] | 0.700 | 0.370 | [0.330, 0.403] |
| 0.05 | test_shift | element_grounding | learned@0.05 | 200 | 0.047 | [0.037, 0.053] | 0.110 | 0.424 | [0.393, 0.457] |
| 0.05 | test_shift | element_grounding | learned_pac@0.05 | 200 | 0.039 | [0.030, 0.047] | 0.000 | 0.387 | [0.343, 0.427] |
| 0.05 | test_shift | element_grounding | risk@0.05 +prior | 200 | 0.044 | [0.033, 0.053] | 0.175 | 0.356 | [0.310, 0.390] |
| 0.05 | test_shift | element_grounding | learned@0.05 +prior | 200 | 0.025 | [0.017, 0.033] | 0.000 | 0.321 | [0.283, 0.347] |
| 0.05 | test_shift | element_grounding | learned_pac@0.05 +prior | 200 | 0.018 | [0.013, 0.027] | 0.000 | 0.291 | [0.250, 0.330] |
| 0.1 | test_in | action_prediction | risk@0.1 | 200 | 0.019 | [0.000, 0.039] | 0.000 | 0.109 | [0.073, 0.146] |
| 0.1 | test_in | action_prediction | learned@0.1 | 200 | 0.078 | [0.041, 0.134] | 0.155 | 0.267 | [0.216, 0.338] |
| 0.1 | test_in | action_prediction | learned_pac@0.1 | 200 | 0.069 | [0.035, 0.117] | 0.060 | 0.247 | [0.197, 0.309] |
| 0.1 | test_in | action_prediction | risk@0.1 +prior | 200 | 0.080 | [0.040, 0.122] | 0.215 | 0.365 | [0.301, 0.434] |
| 0.1 | test_in | action_prediction | learned@0.1 +prior | 200 | 0.149 | [0.078, 0.221] | 0.945 | 0.541 | [0.444, 0.631] |
| 0.1 | test_in | action_prediction | learned_pac@0.1 +prior | 200 | 0.130 | [0.067, 0.199] | 0.795 | 0.501 | [0.416, 0.589] |
| 0.1 | test_in | element_grounding | risk@0.1 | 200 | 0.115 | [0.072, 0.152] | 0.740 | 0.492 | [0.429, 0.543] |
| 0.1 | test_in | element_grounding | learned@0.1 | 200 | 0.118 | [0.067, 0.170] | 0.745 | 0.571 | [0.490, 0.655] |
| 0.1 | test_in | element_grounding | learned_pac@0.1 | 200 | 0.098 | [0.056, 0.157] | 0.430 | 0.529 | [0.462, 0.609] |
| 0.1 | test_in | element_grounding | risk@0.1 +prior | 200 | 0.096 | [0.050, 0.135] | 0.415 | 0.498 | [0.436, 0.566] |
| 0.1 | test_in | element_grounding | learned@0.1 +prior | 200 | 0.056 | [0.024, 0.097] | 0.010 | 0.436 | [0.365, 0.498] |
| 0.1 | test_in | element_grounding | learned_pac@0.1 +prior | 200 | 0.045 | [0.016, 0.087] | 0.000 | 0.404 | [0.339, 0.476] |
| 0.1 | test_shift | action_prediction | risk@0.1 | 200 | 0.032 | [0.020, 0.053] | 0.000 | 0.121 | [0.100, 0.150] |
| 0.1 | test_shift | action_prediction | learned@0.1 | 200 | 0.092 | [0.073, 0.113] | 0.170 | 0.257 | [0.223, 0.297] |
| 0.1 | test_shift | action_prediction | learned_pac@0.1 | 200 | 0.080 | [0.057, 0.097] | 0.010 | 0.236 | [0.200, 0.270] |
| 0.1 | test_shift | action_prediction | risk@0.1 +prior | 200 | 0.111 | [0.077, 0.147] | 0.670 | 0.384 | [0.330, 0.447] |
| 0.1 | test_shift | action_prediction | learned@0.1 +prior | 200 | 0.216 | [0.183, 0.240] | 1.000 | 0.607 | [0.550, 0.663] |
| 0.1 | test_shift | action_prediction | learned_pac@0.1 +prior | 200 | 0.196 | [0.156, 0.227] | 1.000 | 0.565 | [0.490, 0.617] |
| 0.1 | test_shift | element_grounding | risk@0.1 | 200 | 0.076 | [0.060, 0.090] | 0.000 | 0.467 | [0.437, 0.490] |
| 0.1 | test_shift | element_grounding | learned@0.1 | 200 | 0.098 | [0.070, 0.137] | 0.380 | 0.565 | [0.510, 0.627] |
| 0.1 | test_shift | element_grounding | learned_pac@0.1 | 200 | 0.077 | [0.060, 0.107] | 0.045 | 0.522 | [0.480, 0.577] |
| 0.1 | test_shift | element_grounding | risk@0.1 +prior | 200 | 0.088 | [0.070, 0.103] | 0.045 | 0.505 | [0.480, 0.527] |
| 0.1 | test_shift | element_grounding | learned@0.1 +prior | 200 | 0.042 | [0.033, 0.053] | 0.000 | 0.427 | [0.387, 0.460] |
| 0.1 | test_shift | element_grounding | learned_pac@0.1 +prior | 200 | 0.036 | [0.033, 0.047] | 0.000 | 0.399 | [0.350, 0.440] |


## Adaptive gates on test_shift streams, eps=0.05, prior+T (200 site orders and audit draws)

| policy | audit_rate | streams | auto_err | auto_err_2.5–97.5% | P(auto_err>eps) | tail_auto_err | automation | automation_2.5–97.5% |
|---|---|---|---|---|---|---|---|---|
| arisk (γ 0.02) | 0.2 | 200 | 0.049 | [0.023, 0.093] | 0.435 | 0.051 | 0.263 | [0.153, 0.403] |
| alearned (γ 0.005, κ 0.2, dr): default | 0.2 | 200 | 0.042 | [0.023, 0.065] | 0.200 | 0.042 | 0.266 | [0.167, 0.363] |


## Adaptive gates on test_shift streams, eps=0.1, prior+T (200 site orders and audit draws)

| policy | audit_rate | streams | auto_err | auto_err_2.5–97.5% | P(auto_err>eps) | tail_auto_err | automation | automation_2.5–97.5% |
|---|---|---|---|---|---|---|---|---|
| arisk (γ 0.02) | 0.2 | 200 | 0.095 | [0.058, 0.140] | 0.375 | 0.096 | 0.414 | [0.297, 0.543] |
| alearned (γ 0.005, κ 0.2, dr): default | 0.2 | 200 | 0.082 | [0.058, 0.115] | 0.105 | 0.080 | 0.406 | [0.333, 0.497] |

