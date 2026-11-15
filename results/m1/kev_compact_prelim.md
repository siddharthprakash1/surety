# M1 analysis: kev_compact

- collection: `/Users/siddharthprakash/Desktop/couchbase-v2/surety/runs/kev_compact.jsonl`
- rows: cal 600, test_in 400, test_shift 600
- backend: kev
- mode: compact

## H1: calibration

| split | variant | n | acc | ECE | Brier | conf_err |
|---|---|---|---|---|---|---|
| cal | first_run | 600 | 0.478 | 0.150 | 0.728 | 0.000 |
| cal | debiased | 600 | 0.477 | 0.160 | 0.743 | 0.000 |
| cal | debiased+T | 600 | 0.477 | 0.095 | 0.711 | 0.000 |
| test_in | first_run | 400 | 0.485 | 0.167 | 0.720 | 0.000 |
| test_in | debiased | 400 | 0.485 | 0.168 | 0.740 | 0.000 |
| test_in | debiased+T | 400 | 0.485 | 0.086 | 0.704 | 0.000 |
| test_shift | first_run | 600 | 0.453 | 0.114 | 0.731 | 0.000 |
| test_shift | debiased | 600 | 0.458 | 0.134 | 0.746 | 0.000 |
| test_shift | debiased+T | 600 | 0.458 | 0.077 | 0.720 | 0.002 |


## H2: gates fitted on cal (debiased+T)

| eps | policy | split | n | auto_err | automation | coverage | set_size |
|---|---|---|---|---|---|---|---|
| 0.05 | fixed@0.55 | test_in | 400 | 0.060 | 0.258 | 0.940 | 18.045 |
| 0.05 | fixed@0.55 | test_shift | 600 | 0.080 | 0.282 | 0.920 | 17.385 |
| 0.05 | fixed@0.7 | test_in | 400 | 0.020 | 0.135 | 0.980 | 19.767 |
| 0.05 | fixed@0.7 | test_shift | 600 | 0.020 | 0.167 | 0.980 | 19.128 |
| 0.05 | margin@0.1 | test_in | 400 | 0.270 | 0.655 | 0.730 | 9.360 |
| 0.05 | margin@0.1 | test_shift | 600 | 0.293 | 0.662 | 0.707 | 10.212 |
| 0.05 | fitted@0.05 | test_in | 400 | 0.040 | 0.207 | 0.960 | 18.733 |
| 0.05 | fitted@0.05 | test_shift | 600 | 0.060 | 0.235 | 0.940 | 18.492 |
| 0.05 | per_choice@0.05 | test_in | 400 | 0.030 | 0.203 | 0.970 | 18.745 |
| 0.05 | per_choice@0.05 | test_shift | 600 | 0.048 | 0.223 | 0.952 | 18.533 |
| 0.05 | conformal@0.05 | test_in | 400 | 0.000 | 0.037 | 0.935 | 8.707 |
| 0.05 | conformal@0.05 | test_shift | 600 | 0.000 | 0.028 | 0.938 | 8.715 |
| 0.1 | fixed@0.55 | test_in | 400 | 0.060 | 0.258 | 0.940 | 18.045 |
| 0.1 | fixed@0.55 | test_shift | 600 | 0.080 | 0.282 | 0.920 | 17.385 |
| 0.1 | fixed@0.7 | test_in | 400 | 0.020 | 0.135 | 0.980 | 19.767 |
| 0.1 | fixed@0.7 | test_shift | 600 | 0.020 | 0.167 | 0.980 | 19.128 |
| 0.1 | margin@0.1 | test_in | 400 | 0.270 | 0.655 | 0.730 | 9.360 |
| 0.1 | margin@0.1 | test_shift | 600 | 0.293 | 0.662 | 0.707 | 10.212 |
| 0.1 | fitted@0.1 | test_in | 400 | 0.090 | 0.318 | 0.910 | 17.105 |
| 0.1 | fitted@0.1 | test_shift | 600 | 0.105 | 0.338 | 0.895 | 16.458 |
| 0.1 | per_choice@0.1 | test_in | 400 | 0.058 | 0.280 | 0.943 | 17.288 |
| 0.1 | per_choice@0.1 | test_shift | 600 | 0.070 | 0.302 | 0.930 | 16.633 |
| 0.1 | conformal@0.1 | test_in | 400 | 0.000 | 0.052 | 0.895 | 6.810 |
| 0.1 | conformal@0.1 | test_shift | 600 | 0.002 | 0.043 | 0.887 | 6.615 |


## H2: ACI on test_shift, eps=0.05

| policy | audit_rate | n | auto_err | automation | final_alpha |
|---|---|---|---|---|---|
| conformal@0.05 | - | 600 | 0.000 | 0.028 | - |
| aci@0.05 | 0 | 600 | 0.000 | 0.022 | 0.026 |
| aci@0.05 | 0.05 | 600 | 0.000 | 0.022 | 0.026 |
| aci@0.05 | 0.2 | 600 | 0.000 | 0.022 | 0.026 |


## H2: ACI on test_shift, eps=0.1

| policy | audit_rate | n | auto_err | automation | final_alpha |
|---|---|---|---|---|---|
| conformal@0.1 | - | 600 | 0.002 | 0.043 | - |
| aci@0.1 | 0 | 600 | 0.002 | 0.042 | 0.073 |
| aci@0.1 | 0.05 | 600 | 0.002 | 0.042 | 0.073 |
| aci@0.1 | 0.2 | 600 | 0.002 | 0.042 | 0.073 |


![risk-coverage](kev_compact_prelim.png)
