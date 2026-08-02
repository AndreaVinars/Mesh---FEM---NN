"""Run inference with the trained four-property FNN surrogate.

By default, model and scaler artifacts are loaded from the repository's root
``models`` directory. Set ``FNN_MODEL_DIR`` to use artifacts stored in another
location.
"""

import os
from pathlib import Path

import numpy as np
import torch
import joblib

from FNN_shared import FNN, build_feature_vector


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
MODEL_DIR = Path(
    os.environ.get("FNN_MODEL_DIR", REPOSITORY_ROOT / "models")
).expanduser()

MODEL_PATH = MODEL_DIR / "best_silu.pt"
SCALER_X_PATH = MODEL_DIR / "scaler_X.pkl"
SCALER_Y_PATH = MODEL_DIR / "scaler_y.pkl"


def load_surrogate_model():
    """Load the trained model, both scalers, and the selected compute device.

    Returns:
        The evaluation-ready model, input scaler, target scaler, and PyTorch
        device used for inference.
    """

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
    """Predict effective properties for a unit cell containing two ellipses.

    The ten geometric inputs are converted to the same 26-feature row used
    during training. The function returns predicted properties together with a
    named dictionary of the generated feature values. Stiffness predictions
    are reported in GPa and ``NU_xy`` is dimensionless.
    """

    model.eval()

    x_custom, feature_data = build_feature_vector(
        x1, y1, rx1, ry1, angle1_deg,
        x2, y2, rx2, ry2, angle2_deg,
    )

    # Validate compatibility with scaler artifacts created by scikit-learn.
    if (
        hasattr(scaler_X, "n_features_in_")
        and x_custom.shape[1] != scaler_X.n_features_in_
    ):
        raise ValueError(
            f"Feature mismatch: x_custom has {x_custom.shape[1]} features, "
            f"but scaler_X expects {scaler_X.n_features_in_}."
        )

    x_custom_s = scaler_X.transform(x_custom).astype(np.float32)
    x_custom_t = torch.from_numpy(x_custom_s).to(device)

    with torch.no_grad():
        y_pred_s = model(x_custom_t).cpu().numpy()

    # Convert standardized network outputs back to their physical scale.
    y_pred = scaler_y.inverse_transform(y_pred_s)

    result = {
        "E_x [GPa]": float(y_pred[0, 0]),
        "E_y [GPa]": float(y_pred[0, 1]),
        "G_xy [GPa]": float(y_pred[0, 2]),
        "NU_xy [-]": float(y_pred[0, 3]),
    }

    return result, feature_data
