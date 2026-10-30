# Environment for the time-series notebooks

The notebooks 07c to 07h need two packages on top of `requirements.txt`: AutoTS and Prophet (Prophet is in
`requirements.txt` too). They are listed in `requirements-extra.txt`. tsfresh was installed as well, for the demand
profiles of a notebook that was later removed because the profiles routed every product to the same model
(`docs/PREREGISTRATION_TS.md`, change 5); it is no longer needed.

```
pip install -r requirements.txt
pip install -r requirements-extra.txt
```

## What we checked

We first ran `pip install --dry-run autots tsfresh` in the project environment (`sku_env`, Python 3.12.9). It only
added new packages and changed none of the pinned ones, so no separate environment is needed. After installing, a
throwaway script imported each package, extracted tsfresh features from two short series, fitted a Theta model and a
Prophet model, and listed the AutoTS model groups. All of it ran without errors.

| Package | Version | Why it is here |
|---|---|---|
| autots | 1.0.4 | AutoTS contender in 07f |
| tsfresh | 0.21.2 | demand profiles, since removed (no longer needed) |
| prophet | 1.5.0 | Prophet contender in 07f (cmdstanpy 1.3.0 underneath) |
| statsmodels | 0.15.0 | unchanged; ThetaModel, VARMAX and SARIMAX come from here |
| numpy | 2.5.3 | unchanged |
| pandas | 3.0.6 | unchanged |
| scikit-learn | 1.9.1 | unchanged |

## Darts

Darts is optional and left out. `pip install --dry-run "u8darts[notorch]"` wanted to downgrade pandas from 3.0.6 to
2.3.3 and scikit-learn from 1.9.1 to 1.7.2, and to add catboost, shap and statsforecast. That would change the
environment every published number was produced in, so we did not install it. If Darts is wanted later, it should go
in its own environment.

## Windows

This runs on Windows. The notebooks run their fits with joblib, as 07a does, and put `src/` on `PYTHONPATH` so the
worker processes can import the project code.
