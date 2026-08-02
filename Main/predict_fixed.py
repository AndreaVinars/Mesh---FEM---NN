



import os
import sys
from pathlib import Path

import numpy as np
import torch
import joblib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FNN_PIPELINE_DIR = REPOSITORY_ROOT / "Pipeline" / "FNN"
if str(FNN_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(FNN_PIPELINE_DIR))

from FNN_shared import FEATURE_NAMES, FNN, build_feature_vector


MODEL_DIR = Path(
    os.environ.get("FNN_MODEL_DIR", REPOSITORY_ROOT / "models")
).expanduser()

MODEL_PATH = MODEL_DIR / "best_silu.pt"
SCALER_X_PATH = MODEL_DIR / "scaler_X.pkl"
SCALER_Y_PATH = MODEL_DIR / "scaler_y.pkl"


def load_surrogate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FNN().to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    return model, scaler_X, scaler_y, device


def predict_custom(
    model,
    scaler_X,
    scaler_y,
    x1, y1, rx1, ry1, angle1_deg,
    x2, y2, rx2, ry2, angle2_deg,
    device
):
    model.eval()

    x_custom, _ = build_feature_vector(
        x1, y1, rx1, ry1, angle1_deg,
        x2, y2, rx2, ry2, angle2_deg,
    )

    if x_custom.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"x_custom has {x_custom.shape[1]} features, "
            f"but model expects {len(FEATURE_NAMES)}."
        )

    if hasattr(scaler_X, "n_features_in_"):
        if x_custom.shape[1] != scaler_X.n_features_in_:
            raise ValueError(
                f"Feature mismatch: x_custom has {x_custom.shape[1]} features, "
                f"but scaler_X expects {scaler_X.n_features_in_}."
            )

    x_custom_s = scaler_X.transform(x_custom).astype(np.float32)
    x_custom_t = torch.from_numpy(x_custom_s).to(device)

    with torch.no_grad():
        y_pred_s = model(x_custom_t).cpu().numpy()

    y_pred = scaler_y.inverse_transform(y_pred_s)

    result = {
        "E_x [GPa]": float(y_pred[0, 0]),
        "E_y [GPa]": float(y_pred[0, 1]),
        "G_xy [GPa]": float(y_pred[0, 2]),
        "NU_xy [-]": float(y_pred[0, 3]),
    }

    return result
