# Group-conditional risk control: kev_compact

- collection: `artifacts/m1/kev_compact.jsonl`; 200 calibration draws (cal re-drawn from cal ∪ test_in, T pooled and refit); per_group = one RiskConformal per task type on that type's cal rows; test_in = calibration sites, test_shift = held-out sites

## test_in

| eps | gate | group | draws | auto_err | auto_err_2.5–97.5% | P(auto_err>eps) | automation |
|---|---|---|---|---|---|---|---|
| 0.05 | marginal | action_prediction | 200 | 0.008 | [0.000, 0.021] | 0.000 | 0.081 |
| 0.05 | marginal | all | 200 | 0.034 | [0.015, 0.053] | 0.040 | 0.224 |
| 0.05 | marginal | element_grounding | 200 | 0.059 | [0.027, 0.091] | 0.685 | 0.366 |
| 0.05 | per_group | action_prediction | 200 | 0.051 | [0.015, 0.092] | 0.505 | 0.167 |
| 0.05 | per_group | all | 200 | 0.044 | [0.022, 0.063] | 0.255 | 0.231 |
| 0.05 | per_group | element_grounding | 200 | 0.037 | [0.015, 0.066] | 0.125 | 0.296 |
| 0.1 | marginal | action_prediction | 200 | 0.019 | [0.000, 0.039] | 0.000 | 0.109 |
| 0.1 | marginal | all | 200 | 0.067 | [0.043, 0.090] | 0.000 | 0.301 |
| 0.1 | marginal | element_grounding | 200 | 0.115 | [0.072, 0.152] | 0.740 | 0.492 |
| 0.1 | per_group | action_prediction | 200 | 0.098 | [0.052, 0.151] | 0.460 | 0.241 |
| 0.1 | per_group | all | 200 | 0.081 | [0.052, 0.110] | 0.085 | 0.309 |
| 0.1 | per_group | element_grounding | 200 | 0.063 | [0.029, 0.098] | 0.015 | 0.377 |


## test_shift

| eps | gate | group | draws | auto_err | auto_err_2.5–97.5% | P(auto_err>eps) | automation |
|---|---|---|---|---|---|---|---|
| 0.05 | marginal | action_prediction | 200 | 0.015 | [0.010, 0.020] | 0.000 | 0.083 |
| 0.05 | marginal | all | 200 | 0.037 | [0.027, 0.045] | 0.000 | 0.226 |
| 0.05 | marginal | element_grounding | 200 | 0.059 | [0.040, 0.070] | 0.700 | 0.370 |
| 0.05 | per_group | action_prediction | 200 | 0.074 | [0.057, 0.083] | 1.000 | 0.188 |
| 0.05 | per_group | all | 200 | 0.055 | [0.047, 0.060] | 0.905 | 0.238 |
| 0.05 | per_group | element_grounding | 200 | 0.037 | [0.027, 0.040] | 0.000 | 0.287 |
| 0.1 | marginal | action_prediction | 200 | 0.032 | [0.020, 0.053] | 0.000 | 0.121 |
| 0.1 | marginal | all | 200 | 0.054 | [0.040, 0.070] | 0.000 | 0.294 |
| 0.1 | marginal | element_grounding | 200 | 0.076 | [0.060, 0.090] | 0.000 | 0.467 |
| 0.1 | per_group | action_prediction | 200 | 0.118 | [0.093, 0.130] | 0.890 | 0.264 |
| 0.1 | per_group | all | 200 | 0.090 | [0.078, 0.098] | 0.005 | 0.322 |
| 0.1 | per_group | element_grounding | 200 | 0.063 | [0.043, 0.073] | 0.000 | 0.381 |


## Adaptive risk on test_shift streams, eps=0.05, audit rate 0.2

| policy | gate | group | streams | auto_err | P(auto_err>eps) | automation |
|---|---|---|---|---|---|---|
| arisk@0.05 | one gate | action_prediction | 200 | 0.034 | 0.135 | 0.108 |
| arisk@0.05 | one gate | all | 200 | 0.048 | 0.400 | 0.236 |
| arisk@0.05 | one gate | element_grounding | 200 | 0.063 | 0.730 | 0.364 |
| arisk@0.05 | per group | action_prediction | 200 | 0.054 | 0.505 | 0.136 |
| arisk@0.05 | per group | all | 200 | 0.049 | 0.435 | 0.217 |
| arisk@0.05 | per group | element_grounding | 200 | 0.045 | 0.300 | 0.298 |


## Adaptive risk on test_shift streams, eps=0.1, audit rate 0.2

| policy | gate | group | streams | auto_err | P(auto_err>eps) | automation |
|---|---|---|---|---|---|---|
| arisk@0.1 | one gate | action_prediction | 200 | 0.091 | 0.350 | 0.213 |
| arisk@0.1 | one gate | all | 200 | 0.094 | 0.350 | 0.348 |
| arisk@0.1 | one gate | element_grounding | 200 | 0.098 | 0.370 | 0.483 |
| arisk@0.1 | per group | action_prediction | 200 | 0.105 | 0.480 | 0.228 |
| arisk@0.1 | per group | all | 200 | 0.091 | 0.265 | 0.329 |
| arisk@0.1 | per group | element_grounding | 200 | 0.078 | 0.085 | 0.430 |
