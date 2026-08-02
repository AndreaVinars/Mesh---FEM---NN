"""Train the four-output feed-forward neural-network surrogate.

The model maps 26 geometric features to ``E_x``, ``E_y``, ``G_xy``, and
``NU_xy``. Data are split into training, validation, and test sets; scalers are
fitted on the training subset; and early stopping selects the saved model.
"""

# =============================================================================
# Imports and logging setup
# =============================================================================
import os
import copy
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib

from FNN_shared import (
    BASE_FEATURE_NAMES,
    FEATURE_NAMES,
    TARGET_NAMES,
    FNN,
    add_derived_features,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
FNN_WORK_DIR = Path(
    os.environ.get("FNN_WORK_DIR", REPOSITORY_ROOT)
).expanduser()
CSV_PATH = Path(
    os.environ.get(
        "FNN_DATASET_PATH",
        REPOSITORY_ROOT
        / "data"
        / "direct_contraction"
        / "ml_data_direct_contraction_seed_30.csv",
    )
).expanduser()

LOG_LEVEL = logging.INFO
LOG_TO_FILE = True  # Set to False to disable file logging.


def _setup_logger(work_dir: Path) -> logging.Logger:
    """Create training log handlers without side effects during module import."""

    logger = logging.getLogger("FNN")
    logger.setLevel(LOG_LEVEL)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if LOG_TO_FILE:
        logs_dir = work_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_path}")

    return logger



def main():
    """Train, evaluate, save, and plot the effective-property surrogate."""

    work_dir = FNN_WORK_DIR.resolve()
    dataset_path = CSV_PATH if CSV_PATH.is_absolute() else REPOSITORY_ROOT / CSV_PATH
    dataset_path = dataset_path.resolve()

    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)
    logger = _setup_logger(work_dir)

    # =============================================================================
    # Reproducibility and compute device
    # =============================================================================
    SEED = 2
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # =============================================================================
    # Load, validate, split, and scale the dataset
    # =============================================================================
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"CSV not found: {dataset_path}. Generate a dataset first or set "
            "FNN_DATASET_PATH."
        )
    df = pd.read_csv(dataset_path, sep=";", decimal=",")

    required_cols = [*TARGET_NAMES, *BASE_FEATURE_NAMES]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
       raise KeyError(f"Missing required columns in CSV: {missing}")

    # Elastic moduli must be positive. Negative Poisson ratios are retained
    # because auxetic effective responses are physically possible.
    initial_rows = len(df)
    df = df[(df["E_x"] > 0) & (df["E_y"] > 0) & (df["G_xy"] > 0)]
    removed_rows = initial_rows - len(df)
    logger.info(
        f"Removed {removed_rows} rows with nonpositive elastic moduli. "
        f"Remaining: {len(df)} samples."
    )

    df = add_derived_features(df)


    # Assemble the feature matrix in the shared, fixed feature order.
    X = df[list(FEATURE_NAMES)].values.astype(np.float32)

    # Reject invalid feature values before fitting the scalers.
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN/inf (check A1/A2 and other columns).")

    # Assemble the target matrix in the shared, fixed target order.
    # Convert the three stiffness targets from MPa to GPa; NU_xy is dimensionless.

    y = df[list(TARGET_NAMES)].values.astype(np.float32)  # Convert targets to a NumPy matrix.
    y[:,0:3] = y[:,0:3] / 1000

    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

    # Split the dataset into 70% training, 15% validation, and 15% test data.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED, shuffle=True)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, shuffle=True)

    # Fit separate input and target scalers on the training subset only.
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler_X.transform(X_val).astype(np.float32)
    X_test_s  = scaler_X.transform(X_test).astype(np.float32)

    y_train_s = scaler_y.fit_transform(y_train).astype(np.float32)
    y_val_s   = scaler_y.transform(y_val).astype(np.float32)
    y_test_s  = scaler_y.transform(y_test).astype(np.float32)

    logger.info(f"X_train scaled mean: {X_train_s.mean():.4f}, std: {X_train_s.std():.4f}")

    # =============================================================================
    # PyTorch datasets and data loaders
    # =============================================================================
    # Convert standardized arrays to float32 tensors for model training.
    X_train_t = torch.from_numpy(X_train_s)
    y_train_t = torch.from_numpy(y_train_s)
    X_val_t   = torch.from_numpy(X_val_s)
    y_val_t   = torch.from_numpy(y_val_s)
    X_test_t  = torch.from_numpy(X_test_s)
    y_test_t  = torch.from_numpy(y_test_s)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset   = TensorDataset(X_val_t, y_val_t)
    test_dataset  = TensorDataset(X_test_t, y_test_t)

    # Shuffle training batches and use larger deterministic validation/test batches.
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=256, shuffle=False, pin_memory=True, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False, pin_memory=True, num_workers=0)

    # =============================================================================
    # Training and evaluation helpers
    # =============================================================================
    def run_epoch_train(model, loader, criterion, optimizer):

        """One training epoch: forward + loss + backward + optimizer step."""

        model.train()
        total_loss = 0.0
        n = 0

        for Xb, yb in loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred = model(Xb)
            loss = criterion(pred, yb)

            loss.backward()
            optimizer.step()

            bs = Xb.size(0)
            total_loss += loss.item() * bs
            n += bs

        return total_loss / max(n, 1)

    @torch.no_grad()
    def run_epoch_eval(model, loader, criterion):

        """One evaluation epoch: no gradients, return average loss."""

        model.eval()
        total_loss = 0.0
        n = 0
        for Xb, yb in loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(Xb)
            loss = criterion(pred, yb)

            bs = Xb.size(0)
            total_loss += loss.item() * bs  # Accumulate sample-weighted batch loss.
            n += bs
        return total_loss / max(n, 1)

    @torch.no_grad()
    def predict_scaled(model, loader):

        """Return predictions and targets in the scaled space (same as during training)."""

        model.eval()
        preds, trues = [], []
        for Xb, yb in loader:
            Xb = Xb.to(device, non_blocking=True)
            pred = model(Xb).detach().cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())
        return np.vstack(preds), np.vstack(trues)

    def rmse(a, b):
        """Compute one root mean squared error value per target column."""

        return np.sqrt(np.mean((a - b) ** 2, axis=0))

    def mae(a, b):
        """Compute one mean absolute error value per target column."""

        return np.mean(np.abs(a - b), axis=0)

    # =============================================================================
    # Training with early stopping
    # =============================================================================
    def train_model(
        lr,
        weight_decay,
        max_epochs,
        patience,
        min_delta,
        save_path=None):

        """
        Training setup:
        - SmoothL1Loss(beta=0.11)
        - AdamW optimizer
        - ReduceLROnPlateau scheduler
        - Early stopping based on validation loss
        """

        model = FNN().to(device)

        # Smooth L1 is less sensitive to outliers than mean squared error.
        criterion = nn.SmoothL1Loss(beta=0.11, reduction = "mean")  # Mean reduction yields one batch loss.

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=50,
            min_lr=1e-6)

        best_state = None
        best_val = float("inf")
        bad_epochs = 0

        train_losses = []
        val_losses = []

        logger.info(
            "Start training | "
            f"lr={lr:g}, weight_decay={weight_decay:g}, max_epochs={max_epochs}, "
            f"patience={patience}, min_delta={min_delta:g}, save_path={save_path}")

        for epoch in range(1, max_epochs + 1):

            tr = run_epoch_train(model, train_loader, criterion, optimizer)
            va = run_epoch_eval(model, val_loader, criterion)

            train_losses.append(tr)
            val_losses.append(va)

            scheduler.step(va)

            improved = (best_val - va) > min_delta
            if improved:
                best_val = va
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
                if save_path:
                    torch.save(best_state, save_path)
                    logger.info(f"Saved new best model → {save_path} (best_val={best_val:.6f})")
            else:
                bad_epochs += 1

            if epoch == 1 or epoch % 50 == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[SiLU] Epoch {epoch:4d} | lr: {cur_lr:.2e} | "
                    f"train loss: {tr:.6f} | val loss: {va:.6f} | best: {best_val:.6f}")

            if bad_epochs >= patience:
                logger.info(
                    f"[SiLU] Early stopping at epoch {epoch} "
                    f"(no val improvement for {patience} epochs). Best val loss: {best_val:.6f}")
                break

        # Restore the weights from the epoch with the lowest validation loss.
        if best_state is not None:
            model.load_state_dict(best_state)

        history = {"train": train_losses,"val": val_losses,"best_val": best_val}

        return model, history

    # =============================================================================
    # Run training and save the fitted artifacts
    # =============================================================================
    os.makedirs("models", exist_ok=True)

    model_silu, hist_silu = train_model(
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=3000,
        patience=500,
        min_delta=1e-6,
        save_path="models/best_silu.pt")

    joblib.dump(scaler_X, "models/scaler_X.pkl")
    joblib.dump(scaler_y, "models/scaler_y.pkl")

    logger.info("Saved scalers: models/scaler_X.pkl and models/scaler_y.pkl")

    # =============================================================================
    # Evaluate the selected model on the held-out test set
    # =============================================================================
    def eval_on_test(model):

        """Evaluate on the test set in the original y scale."""

        y_pred_s, y_true_s = predict_scaled(model, test_loader)

        # Transform predictions and references back to physical units.
        y_pred = scaler_y.inverse_transform(y_pred_s)
        y_true = scaler_y.inverse_transform(y_true_s)

        rmse_values = rmse(y_true, y_pred)
        mae_values = mae(y_true, y_pred)
        metrics ={}

        for target, r, a in zip(TARGET_NAMES, rmse_values, mae_values, strict=True):

            metrics[target] = {"RMSE": float(r),
                             "MAE": float(a)}

        for target in TARGET_NAMES:

            logger.info(f"{target}: "
                        f"RMSE={metrics[target]['RMSE']:.6f} | "
                        f"MAE={metrics[target]['MAE']:.6f}")

        return y_true, y_pred, metrics

    y_true_silu, y_pred_silu, m_silu = eval_on_test(model_silu)


    # =============================================================================
    # Save training-history and prediction-quality plots
    # =============================================================================
    os.makedirs("plots", exist_ok=True)

    units = ["GPa", "GPa", "GPa", "-"]

    for i, (target, unit) in enumerate(zip(TARGET_NAMES, units, strict=True)):

        y_true_target = y_true_silu[:, i]
        y_pred_target = y_pred_silu[:, i]

        plt.figure()
        plt.scatter(y_true_target, y_pred_target)
        mn = float(min(y_pred_target.min(), y_true_target.min()))
        mx = float(max(y_pred_target.max(), y_true_target.max()))

        plt.plot([mn, mx], [mn, mx], "--")
        plt.title(f"True vs Predicted - {target}")
        plt.ylabel(f"Predicted Value [{unit}]")
        plt.xlabel(f"True Value [{unit}]")
        plt.tight_layout()
        TRUE_PRED = os.path.join("plots", f"SILU_{target}.png")
        plt.savefig(TRUE_PRED, dpi=200, bbox_inches="tight")
        plt.close()


    for i, (target, unit) in enumerate(zip(TARGET_NAMES, units, strict=True)):

        y_target = y[:, i]

        plt.figure()
        plt.hist(y_target, bins=40, edgecolor="black")
        plt.xlabel(f"{target} [{unit}]")
        plt.ylabel("Count")
        plt.title(f"{target} distribution")

        TARGET_HIST = os.path.join("plots", f"{target}_distribution.png")
        plt.savefig(TARGET_HIST, dpi=200, bbox_inches="tight")
        plt.close()

    plt.figure()
    plt.plot(hist_silu["train"], label="Train")
    plt.plot(hist_silu["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("SmoothL1 loss")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()

    LOSS = os.path.join("plots", "LOSS.png")
    plt.savefig(LOSS, dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
