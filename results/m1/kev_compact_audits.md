# Uniform vs risk-targeted audits: kev_compact

- collection: `artifacts/m1/kev_compact.jsonl`; probabilities debiased+T (M1); audit budget 0.2 of acted steps; escalated steps are always labelled
- targeted: q ∝ √(ê(1 − ê)) (alearned, doubly-robust feedback) or √ê (arisk, importance weighting), ê the cross-fitted error model; scaled so the mean q over the calibration steps the gate acts on is the budget; clipped to [floor, 1] (`surety.gate.audit.AuditPolicy`)
- every arm sees the same streams and audit coins (an acted step is audited iff u < q); 'uniform at x' spends the targeted arm's realised audit rate, for an equal-cost comparison
- floor and margin were chosen on calibration-site streams only (re-drawn cal / test_in splits, test_in streamed in blocks of sites); held-out numbers were not used to choose
- supervisor calls/step = escalations + audits per step; tail = the last 200 steps

## eps = 0.05

### Held-out streams (200 site orders of test_shift): the same gates, three audit rules

| config | streams | auto_err | auto_err_2.5–97.5% | P(auto_err>eps) | tail_auto_err | automation | automation_2.5–97.5% | audits/acted | supervisor calls/step |
|---|---|---|---|---|---|---|---|---|---|
| arisk (γ 0.02), uniform | 200 | 0.048 | [0.023, 0.080] | 0.400 | 0.050 | 0.236 | [0.133, 0.340] | 0.206 | 0.812 |
| arisk (γ 0.02), targeted (floor 0.05) | 200 | 0.047 | [0.027, 0.072] | 0.345 | 0.047 | 0.241 | [0.141, 0.320] | 0.217 | 0.811 |
| arisk (γ 0.02), targeted (floor 0.1) | 200 | 0.047 | [0.027, 0.073] | 0.355 | 0.048 | 0.240 | [0.158, 0.325] | 0.216 | 0.811 |
| arisk (γ 0.02), uniform at 0.217 | 200 | 0.047 | [0.022, 0.080] | 0.360 | 0.047 | 0.235 | [0.133, 0.332] | 0.223 | 0.817 |
| arisk (γ 0.02), uniform at 0.216 | 200 | 0.047 | [0.023, 0.083] | 0.340 | 0.046 | 0.235 | [0.148, 0.327] | 0.222 | 0.817 |
| alearned (κ 0), uniform | 200 | 0.047 | [0.027, 0.072] | 0.350 | 0.049 | 0.304 | [0.233, 0.365] | 0.201 | 0.757 |
| alearned (κ 0), targeted (floor 0.05) | 200 | 0.046 | [0.032, 0.067] | 0.305 | 0.047 | 0.304 | [0.247, 0.355] | 0.209 | 0.760 |
| alearned (κ 0), targeted (floor 0.1) | 200 | 0.047 | [0.030, 0.067] | 0.320 | 0.047 | 0.304 | [0.246, 0.357] | 0.209 | 0.760 |
| alearned (κ 0), uniform at 0.209 | 200 | 0.047 | [0.027, 0.067] | 0.335 | 0.048 | 0.303 | [0.233, 0.362] | 0.212 | 0.761 |
| alearned (κ 0.2, default), uniform | 200 | 0.036 | [0.022, 0.060] | 0.090 | 0.037 | 0.267 | [0.198, 0.338] | 0.201 | 0.787 |
| alearned (κ 0.2, default), targeted (floor 0.05) | 200 | 0.037 | [0.020, 0.053] | 0.060 | 0.038 | 0.272 | [0.193, 0.332] | 0.210 | 0.785 |
| alearned (κ 0.2, default), targeted (floor 0.1) | 200 | 0.037 | [0.020, 0.055] | 0.075 | 0.039 | 0.273 | [0.193, 0.330] | 0.210 | 0.785 |
| alearned (κ 0.2, default), uniform at 0.210 | 200 | 0.038 | [0.023, 0.057] | 0.080 | 0.039 | 0.273 | [0.208, 0.332] | 0.213 | 0.785 |


### Calibration-site streams (200 re-drawn cal / test_in splits): choosing floor and margin

| config | streams | auto_err | auto_err_2.5–97.5% | P(auto_err>eps) | tail_auto_err | automation | automation_2.5–97.5% | audits/acted | supervisor calls/step |
|---|---|---|---|---|---|---|---|---|---|
| alearned (κ 0.2, default), uniform | 200 | 0.040 | [0.020, 0.063] | 0.155 | 0.041 | 0.273 | [0.190, 0.347] | 0.203 | 0.783 |
| alearned (κ 0), targeted (floor 0.05) | 200 | 0.049 | [0.025, 0.077] | 0.390 | 0.050 | 0.301 | [0.230, 0.363] | 0.201 | 0.760 |
| alearned (κ 0.05), targeted (floor 0.05) | 200 | 0.046 | [0.025, 0.070] | 0.300 | 0.047 | 0.293 | [0.200, 0.353] | 0.202 | 0.766 |
| alearned (κ 0.1), targeted (floor 0.05) | 200 | 0.044 | [0.020, 0.065] | 0.265 | 0.044 | 0.285 | [0.175, 0.355] | 0.203 | 0.773 |
| alearned (κ 0.15), targeted (floor 0.05) | 200 | 0.042 | [0.020, 0.068] | 0.145 | 0.043 | 0.280 | [0.187, 0.347] | 0.204 | 0.777 |
| alearned (κ 0.2), targeted (floor 0.05) | 200 | 0.039 | [0.022, 0.063] | 0.155 | 0.041 | 0.271 | [0.180, 0.338] | 0.201 | 0.784 |
| alearned (κ 0), targeted (floor 0.1) | 200 | 0.049 | [0.025, 0.075] | 0.400 | 0.050 | 0.301 | [0.215, 0.360] | 0.202 | 0.760 |
| alearned (κ 0.05), targeted (floor 0.1) | 200 | 0.046 | [0.025, 0.068] | 0.300 | 0.046 | 0.292 | [0.197, 0.350] | 0.203 | 0.768 |
| alearned (κ 0.1), targeted (floor 0.1) | 200 | 0.044 | [0.022, 0.062] | 0.235 | 0.044 | 0.284 | [0.200, 0.355] | 0.202 | 0.773 |
| alearned (κ 0.15), targeted (floor 0.1) | 200 | 0.042 | [0.022, 0.068] | 0.160 | 0.043 | 0.280 | [0.187, 0.347] | 0.203 | 0.777 |
| alearned (κ 0.2), targeted (floor 0.1) | 200 | 0.040 | [0.020, 0.063] | 0.145 | 0.042 | 0.272 | [0.180, 0.338] | 0.202 | 0.783 |

Pick: **alearned (κ 0.15), targeted (floor 0.05)**

### Held-out streams: the pick

| config | streams | auto_err | auto_err_2.5–97.5% | P(auto_err>eps) | tail_auto_err | automation | automation_2.5–97.5% | audits/acted | supervisor calls/step |
|---|---|---|---|---|---|---|---|---|---|
| alearned (κ 0.15), targeted (floor 0.05) | 200 | 0.039 | [0.023, 0.057] | 0.065 | 0.041 | 0.281 | [0.220, 0.332] | 0.204 | 0.777 |
| alearned (κ 0.15), uniform | 200 | 0.038 | [0.023, 0.055] | 0.105 | 0.039 | 0.276 | [0.202, 0.335] | 0.202 | 0.779 |
| alearned (κ 0.2, default), uniform | 200 | 0.036 | [0.022, 0.060] | 0.090 | 0.037 | 0.267 | [0.198, 0.338] | 0.201 | 0.787 |


## eps = 0.1

### Held-out streams (200 site orders of test_shift): the same gates, three audit rules

| config | streams | auto_err | auto_err_2.5–97.5% | P(auto_err>eps) | tail_auto_err | automation | automation_2.5–97.5% | audits/acted | supervisor calls/step |
|---|---|---|---|---|---|---|---|---|---|
| arisk (γ 0.02), uniform | 200 | 0.094 | [0.060, 0.133] | 0.350 | 0.098 | 0.348 | [0.267, 0.432] | 0.202 | 0.722 |
| arisk (γ 0.02), targeted (floor 0.05) | 200 | 0.094 | [0.057, 0.130] | 0.335 | 0.097 | 0.353 | [0.267, 0.425] | 0.216 | 0.723 |
| arisk (γ 0.02), targeted (floor 0.1) | 200 | 0.095 | [0.065, 0.137] | 0.325 | 0.098 | 0.353 | [0.292, 0.425] | 0.214 | 0.722 |
| arisk (γ 0.02), uniform at 0.216 | 200 | 0.094 | [0.062, 0.142] | 0.330 | 0.099 | 0.350 | [0.278, 0.440] | 0.219 | 0.726 |
| arisk (γ 0.02), uniform at 0.214 | 200 | 0.094 | [0.060, 0.135] | 0.340 | 0.098 | 0.350 | [0.273, 0.428] | 0.217 | 0.726 |
| alearned (κ 0), uniform | 200 | 0.096 | [0.068, 0.128] | 0.355 | 0.098 | 0.411 | [0.360, 0.465] | 0.201 | 0.671 |
| alearned (κ 0), targeted (floor 0.05) | 200 | 0.095 | [0.070, 0.122] | 0.355 | 0.095 | 0.410 | [0.357, 0.455] | 0.200 | 0.672 |
| alearned (κ 0), targeted (floor 0.1) | 200 | 0.096 | [0.073, 0.122] | 0.360 | 0.097 | 0.411 | [0.365, 0.454] | 0.201 | 0.672 |
| alearned (κ 0), uniform at 0.200 | 200 | 0.096 | [0.068, 0.128] | 0.360 | 0.099 | 0.411 | [0.360, 0.465] | 0.201 | 0.671 |
| alearned (κ 0), uniform at 0.201 | 200 | 0.095 | [0.068, 0.128] | 0.335 | 0.097 | 0.410 | [0.360, 0.465] | 0.202 | 0.673 |
| alearned (κ 0.2, default), uniform | 200 | 0.075 | [0.050, 0.102] | 0.040 | 0.077 | 0.372 | [0.307, 0.422] | 0.201 | 0.703 |
| alearned (κ 0.2, default), targeted (floor 0.05) | 200 | 0.077 | [0.053, 0.107] | 0.055 | 0.080 | 0.376 | [0.323, 0.428] | 0.205 | 0.701 |
| alearned (κ 0.2, default), targeted (floor 0.1) | 200 | 0.077 | [0.055, 0.105] | 0.055 | 0.080 | 0.376 | [0.330, 0.427] | 0.204 | 0.701 |
| alearned (κ 0.2, default), uniform at 0.205 | 200 | 0.075 | [0.050, 0.103] | 0.035 | 0.078 | 0.372 | [0.313, 0.420] | 0.205 | 0.704 |
| alearned (κ 0.2, default), uniform at 0.204 | 200 | 0.075 | [0.050, 0.103] | 0.035 | 0.078 | 0.372 | [0.313, 0.420] | 0.205 | 0.704 |


### Calibration-site streams (200 re-drawn cal / test_in splits): choosing floor and margin

| config | streams | auto_err | auto_err_2.5–97.5% | P(auto_err>eps) | tail_auto_err | automation | automation_2.5–97.5% | audits/acted | supervisor calls/step |
|---|---|---|---|---|---|---|---|---|---|
| alearned (κ 0.2, default), uniform | 200 | 0.078 | [0.050, 0.113] | 0.090 | 0.080 | 0.375 | [0.300, 0.453] | 0.202 | 0.701 |
| alearned (κ 0), targeted (floor 0.05) | 200 | 0.099 | [0.067, 0.140] | 0.435 | 0.101 | 0.422 | [0.357, 0.485] | 0.202 | 0.664 |
| alearned (κ 0.05), targeted (floor 0.05) | 200 | 0.095 | [0.065, 0.130] | 0.345 | 0.097 | 0.413 | [0.343, 0.485] | 0.204 | 0.671 |
| alearned (κ 0.1), targeted (floor 0.05) | 200 | 0.090 | [0.062, 0.130] | 0.220 | 0.092 | 0.402 | [0.337, 0.475] | 0.202 | 0.679 |
| alearned (κ 0.15), targeted (floor 0.05) | 200 | 0.085 | [0.060, 0.113] | 0.120 | 0.086 | 0.390 | [0.307, 0.455] | 0.203 | 0.689 |
| alearned (κ 0.2), targeted (floor 0.05) | 200 | 0.079 | [0.050, 0.110] | 0.080 | 0.081 | 0.376 | [0.300, 0.443] | 0.203 | 0.700 |
| alearned (κ 0), targeted (floor 0.1) | 200 | 0.100 | [0.070, 0.138] | 0.460 | 0.101 | 0.423 | [0.362, 0.487] | 0.202 | 0.663 |
| alearned (κ 0.05), targeted (floor 0.1) | 200 | 0.095 | [0.062, 0.130] | 0.330 | 0.097 | 0.412 | [0.342, 0.487] | 0.204 | 0.672 |
| alearned (κ 0.1), targeted (floor 0.1) | 200 | 0.090 | [0.062, 0.133] | 0.235 | 0.092 | 0.401 | [0.327, 0.473] | 0.201 | 0.680 |
| alearned (κ 0.15), targeted (floor 0.1) | 200 | 0.085 | [0.058, 0.113] | 0.135 | 0.087 | 0.390 | [0.310, 0.458] | 0.203 | 0.689 |
| alearned (κ 0.2), targeted (floor 0.1) | 200 | 0.079 | [0.052, 0.107] | 0.075 | 0.081 | 0.376 | [0.295, 0.440] | 0.203 | 0.701 |

Pick: **alearned (κ 0.2), targeted (floor 0.05)**

### Held-out streams: the pick

| config | streams | auto_err | auto_err_2.5–97.5% | P(auto_err>eps) | tail_auto_err | automation | automation_2.5–97.5% | audits/acted | supervisor calls/step |
|---|---|---|---|---|---|---|---|---|---|
| alearned (κ 0.2), targeted (floor 0.05) | 200 | 0.077 | [0.053, 0.107] | 0.055 | 0.080 | 0.376 | [0.323, 0.428] | 0.205 | 0.701 |
| alearned (κ 0.2), uniform | 200 | 0.075 | [0.050, 0.102] | 0.040 | 0.077 | 0.372 | [0.307, 0.422] | 0.201 | 0.703 |
| alearned (κ 0.2, default), uniform | 200 | 0.075 | [0.050, 0.102] | 0.040 | 0.077 | 0.372 | [0.307, 0.422] | 0.201 | 0.703 |

