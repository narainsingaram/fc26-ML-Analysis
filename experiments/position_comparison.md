# Position-Specific OVR Models

Separate regressors per role reduce averaging across very different responsibilities.

## Metrics by Group
| Group | MAE | RMSE | R² | Train N | Test N |
|---|---|---|---|---|---|
| GK | 0.462 | 0.633 | 0.993 | 1611 | 403 |
| DEF | 0.606 | 0.820 | 0.985 | 4784 | 1197 |
| MID | 0.820 | 1.066 | 0.977 | 5332 | 1333 |
| ATT | 0.755 | 0.973 | 0.983 | 2570 | 643 |

## Top Features per Group (Permutation Importance)
### GK
- num__Acceleration: 0.056
- cat__Position_GK: 0.047
- num__DEF: 0.039
- cat__Preferred foot_Left: 0.035
- num__SHO: 0.022
_Files_: metrics → `artifacts/position_models/gk/metrics.json`, importances → `artifacts/position_models/gk/feature_importance.csv`

### DEF
- num__PHY: 0.638
- num__DEF: 0.026
- num__Heading Accuracy: 0.021
- num__Ball Control: 0.011
- num__Sliding Tackle: 0.007
_Files_: metrics → `artifacts/position_models/def/metrics.json`, importances → `artifacts/position_models/def/feature_importance.csv`

### MID
- num__Composure: 0.453
- num__Ball Control: 0.053
- num__PHY: 0.033
- num__DEF: 0.011
- num__Finishing: 0.008
_Files_: metrics → `artifacts/position_models/mid/metrics.json`, importances → `artifacts/position_models/mid/feature_importance.csv`

### ATT
- num__PAS: 0.423
- num__Composure: 0.086
- num__DEF: 0.024
- num__Finishing: 0.019
- num__Ball Control: 0.011
_Files_: metrics → `artifacts/position_models/att/metrics.json`, importances → `artifacts/position_models/att/feature_importance.csv`
