# Published Surrogate Artifacts

These files provide a ready-to-use four-output surrogate for the inference
examples in the root README:

- `best_silu.pt`: PyTorch `state_dict` for the architecture in
  `Pipeline/FNN/FNN_shared.py`
- `scaler_X.pkl`: input `StandardScaler` for the 26 features in `FEATURE_NAMES`
- `scaler_y.pkl`: output `StandardScaler` for $E_x$, $E_y$, $G_{xy}$, and
  $\nu_{xy}$

The artifacts were trained with 5,000 accepted geometries, seed 2, and a
70/15/15 train-validation-test split. Held-out metrics are reported in the root
README. The three files must be kept together because inference depends on the
exact training-time feature and target scaling.

## SHA-256

```text
best_silu.pt  A6463DDB7A533363ABA86856E2DE7DD0AF4CC4BD7C8807CE46B7305DC5EC919B
scaler_X.pkl  F16ECA4E8FB4BA18355B24B4D5634B7C216FC8B3E6AEBDD783F916E5CB1CDF87
scaler_y.pkl  650E2252177B60E833EE505AFD41FAA2975D12479051DC2AEA841278C145296E
```

Only load model and scaler files from trusted sources. Scikit-learn/joblib
artifacts use Python serialization and are not safe to open when obtained from
an untrusted third party.
