# Uncertainty: kev_compact

- collection: `artifacts/m1/kev_compact.jsonl`; 200 calibration draws, 2000 site-bootstrap resamples, 200 adaptive streams

## Calibration draws: cal re-drawn from cal ∪ test_in (200 draws), T and gates refit, debiased+T

| eps | policy | split | draws | auto_err | auto_err_2.5–97.5% | P(auto_err>eps) | automation | automation_2.5–97.5% |
|---|---|---|---|---|---|---|---|---|
| 0.05 | fitted@0.05 | test_in | 200 | 0.051 | [0.027, 0.080] | 0.465 | 0.227 | [0.187, 0.263] |
| 0.05 | fitted@0.05 | test_shift | 200 | 0.065 | [0.057, 0.072] | 1.000 | 0.245 | [0.230, 0.270] |
| 0.05 | per_choice@0.05 | test_in | 200 | 0.037 | [0.020, 0.065] | 0.120 | 0.210 | [0.172, 0.250] |
| 0.05 | per_choice@0.05 | test_shift | 200 | 0.049 | [0.042, 0.057] | 0.350 | 0.228 | [0.208, 0.253] |
| 0.05 | conformal@0.05 | test_in | 200 | 0.000 | [0.000, 0.000] | 0.000 | 0.029 | [0.017, 0.040] |
| 0.05 | conformal@0.05 | test_shift | 200 | 0.000 | [0.000, 0.000] | 0.000 | 0.027 | [0.023, 0.030] |
| 0.05 | risk@0.05 | test_in | 200 | 0.034 | [0.015, 0.053] | 0.040 | 0.224 | [0.177, 0.263] |
| 0.05 | risk@0.05 | test_shift | 200 | 0.037 | [0.027, 0.045] | 0.000 | 0.226 | [0.203, 0.252] |
| 0.1 | fitted@0.1 | test_in | 200 | 0.099 | [0.062, 0.133] | 0.470 | 0.326 | [0.270, 0.383] |
| 0.1 | fitted@0.1 | test_shift | 200 | 0.109 | [0.090, 0.125] | 0.875 | 0.347 | [0.313, 0.378] |
| 0.1 | per_choice@0.1 | test_in | 200 | 0.067 | [0.040, 0.100] | 0.020 | 0.288 | [0.240, 0.345] |
| 0.1 | per_choice@0.1 | test_shift | 200 | 0.074 | [0.065, 0.088] | 0.000 | 0.305 | [0.283, 0.332] |
| 0.1 | conformal@0.1 | test_in | 200 | 0.000 | [0.000, 0.000] | 0.000 | 0.047 | [0.033, 0.063] |
| 0.1 | conformal@0.1 | test_shift | 200 | 0.001 | [0.000, 0.002] | 0.000 | 0.043 | [0.037, 0.052] |
| 0.1 | risk@0.1 | test_in | 200 | 0.067 | [0.043, 0.090] | 0.000 | 0.301 | [0.258, 0.338] |
| 0.1 | risk@0.1 | test_shift | 200 | 0.054 | [0.040, 0.070] | 0.000 | 0.294 | [0.268, 0.315] |


## Site-cluster bootstrap: original cal fit, test sites resampled with replacement

| eps | policy | split | n | sites | auto_err | site_boot_95ci | automation | site_boot_95ci |
|---|---|---|---|---|---|---|---|---|
| 0.05 | fitted@0.05 | test_in | 400 | 167 | 0.040 | [0.022, 0.063] | 0.207 | [0.169, 0.247] |
| 0.05 | fitted@0.05 | test_shift | 600 | 176 | 0.060 | [0.037, 0.082] | 0.235 | [0.196, 0.278] |
| 0.05 | per_choice@0.05 | test_in | 400 | 167 | 0.030 | [0.013, 0.048] | 0.203 | [0.164, 0.244] |
| 0.05 | per_choice@0.05 | test_shift | 600 | 176 | 0.048 | [0.027, 0.068] | 0.223 | [0.186, 0.263] |
| 0.05 | conformal@0.05 | test_in | 400 | 167 | 0.000 | [0.000, 0.000] | 0.037 | [0.017, 0.060] |
| 0.05 | conformal@0.05 | test_shift | 600 | 176 | 0.000 | [0.000, 0.000] | 0.030 | [0.014, 0.052] |
| 0.05 | risk@0.05 | test_in | 400 | 167 | 0.028 | [0.011, 0.046] | 0.198 | [0.158, 0.237] |
| 0.05 | risk@0.05 | test_shift | 600 | 176 | 0.033 | [0.018, 0.048] | 0.222 | [0.176, 0.271] |
| 0.1 | fitted@0.1 | test_in | 400 | 167 | 0.090 | [0.062, 0.121] | 0.318 | [0.275, 0.362] |
| 0.1 | fitted@0.1 | test_shift | 600 | 176 | 0.105 | [0.075, 0.138] | 0.338 | [0.287, 0.392] |
| 0.1 | per_choice@0.1 | test_in | 400 | 167 | 0.058 | [0.034, 0.082] | 0.280 | [0.237, 0.325] |
| 0.1 | per_choice@0.1 | test_shift | 600 | 176 | 0.070 | [0.047, 0.092] | 0.302 | [0.252, 0.356] |
| 0.1 | conformal@0.1 | test_in | 400 | 167 | 0.000 | [0.000, 0.000] | 0.052 | [0.029, 0.079] |
| 0.1 | conformal@0.1 | test_shift | 600 | 176 | 0.002 | [0.000, 0.005] | 0.043 | [0.025, 0.065] |
| 0.1 | risk@0.1 | test_in | 400 | 167 | 0.058 | [0.037, 0.079] | 0.268 | [0.222, 0.318] |
| 0.1 | risk@0.1 | test_shift | 600 | 176 | 0.047 | [0.027, 0.066] | 0.283 | [0.236, 0.334] |


## Adaptive risk on test_shift, eps=0.05: site order and audits redrawn per stream

| policy | audit_rate | streams | auto_err | auto_err_2.5–97.5% | P(auto_err>eps) | tail_auto_err | automation | automation_2.5–97.5% |
|---|---|---|---|---|---|---|---|---|
| arisk@0.05 | 0.05 | 200 | 0.056 | [0.012, 0.148] | 0.460 | 0.062 | 0.208 | [0.050, 0.430] |
| arisk@0.05 | 0.2 | 200 | 0.048 | [0.023, 0.080] | 0.400 | 0.050 | 0.236 | [0.133, 0.340] |


## Adaptive risk on test_shift, eps=0.1: site order and audits redrawn per stream

| policy | audit_rate | streams | auto_err | auto_err_2.5–97.5% | P(auto_err>eps) | tail_auto_err | automation | automation_2.5–97.5% |
|---|---|---|---|---|---|---|---|---|
| arisk@0.1 | 0.05 | 200 | 0.091 | [0.022, 0.183] | 0.395 | 0.091 | 0.283 | [0.100, 0.488] |
| arisk@0.1 | 0.2 | 200 | 0.094 | [0.060, 0.133] | 0.350 | 0.098 | 0.348 | [0.267, 0.432] |

