# =============================================================================
# 0) Imports + logging setup
# =============================================================================
import os
import copy
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ---------------------------
# Logging (console + optional file)
# ---------------------------
LOG_LEVEL = logging.INFO
LOG_TO_FILE = True  # set False if you don't want a log file

logger = logging.getLogger("FNN")
logger.setLevel(LOG_LEVEL)
logger.handlers.clear()  # avoid duplicated handlers when re-running in Spyder

_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

# Console handler
_ch = logging.StreamHandler()
_ch.setLevel(LOG_LEVEL)
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

# File handler (optional)
if LOG_TO_FILE:
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    _fh = logging.FileHandler(log_path, encoding="utf-8")
    _fh.setLevel(LOG_LEVEL)
    _fh.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.info(f"Logging to file: {log_path}")


def main():
    # =============================================================================
    # 1) Reproducibility + device
    # =============================================================================
    SEED = 3
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # =============================================================================
    # 2) Load + split + scale
    # =============================================================================
    CSV_PATH = os.path.join("data", "ml_data.csv")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH} (put your dataset into ./data/)")
    df = pd.read_csv(CSV_PATH, sep=";", decimal=",")
    
    required_cols = ["E_eff",
                     "x1","y1","rx1","ry1","angle1_sin","angle1_cos",
                     "x2","y2","rx2","ry2","angle2_sin","angle2_cos",
                     "d","dx","dy","A1","A2"]
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
       raise KeyError(f"Missing required columns in CSV: {missing}")
    
    # (a) Basic target cleaning
    initial_rows = len(df)
    df = df[(df["E_eff"] > 0)]
    removed_rows = initial_rows - len(df)
    logger.info(f"Removed {removed_rows} rows with E_eff ≤ 0. Remaining: {len(df)} samples.")

    # (b) X (feature matrix)
    # Feature set: 17 features (2 ellipses + derived values)
    X = df[["x1", "y1", "rx1", "ry1", "angle1_sin", "angle1_cos",
            "x2", "y2", "rx2", "ry2", "angle2_sin", "angle2_cos",
            "d", "dx", "dy", "A1", "A2"]].values.astype(np.float32)

    # (d) Sanity check on X
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X contains NaN/inf (check A1/A2 and other columns).")

    # (e) y (target)
    # Divide by 1000 (assumption: scaling from MPa to GPa)
    y = df["E_eff"].values.reshape(-1, 1).astype(np.float32) / 1000.0

    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

    # (f) Split 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED, shuffle=True)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, shuffle=True)

    # (g) Standardization (fit on train, transform val/test)
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
    # 3) Torch datasets / dataloaders
    # =============================================================================
    # Convert to torch Tensors (float32 is ideal for GPU)
    X_train_t = torch.from_numpy(X_train_s)
    y_train_t = torch.from_numpy(y_train_s)
    X_val_t   = torch.from_numpy(X_val_s)
    y_val_t   = torch.from_numpy(y_val_s)
    X_test_t  = torch.from_numpy(X_test_s)
    y_test_t  = torch.from_numpy(y_test_s)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset   = TensorDataset(X_val_t, y_val_t)
    test_dataset  = TensorDataset(X_test_t, y_test_t)

    # Batch size: smaller for train (shuffle), bigger for val/test (no shuffle)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=256, shuffle=False, pin_memory=True, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False, pin_memory=True, num_workers=0)

    # =============================================================================
    # 4) Model (FNN)
    # =============================================================================
    class FNN(nn.Module):
        
        """
        Simple MLP: 17 → 32 → 32 → 32 → 32 → 1 with SiLU activations.

        This architecture was selected after testing multiple hidden-layer widths and
        several random seeds on this dataset. I also compared common activations
        (Tanh, GELU, ReLU, Sigmoid), and SiLU provided the best validation performance.
        """

        def __init__(self):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(17, 32),
                nn.SiLU(),
                nn.Linear(32, 32),
                nn.SiLU(),
                nn.Linear(32, 32),
                nn.SiLU(),
                nn.Linear(32, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x)

    # =============================================================================
    # 5) Train/eval helpers
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
            total_loss += loss.item() * bs
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
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    # =============================================================================
    # 6) Training with early stopping
    # =============================================================================
    def train_model(
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=1000,
        patience=200,
        min_delta=1e-6,
        save_path=None):
        
        """
        Training setup:
        - SmoothL1Loss(beta=0.2)
        - AdamW optimizer
        - ReduceLROnPlateau scheduler
        - Early stopping based on validation loss
        """
        
        model = FNN().to(device)

        # Loss: more robust to outliers than MSE
        criterion = nn.SmoothL1Loss(beta=0.2)

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

        # Load best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        history = {"train": train_losses, "val": val_losses, "best_val": best_val}
        return model, history

    # =============================================================================
    # 7) Run training
    # =============================================================================
    os.makedirs("models", exist_ok=True)

    model_silu, hist_silu = train_model(
        lr=1e-3,
        max_epochs=2000,
        patience=200,
        save_path="models/best_silu.pt",)

    # =============================================================================
    # 8) Test evaluation
    # =============================================================================
    def eval_on_test(model, name):
        
        """Evaluate on the test set in the original y scale."""
        
        y_pred_s, y_true_s = predict_scaled(model, test_loader)

        # Back to original scale (inverse scaling)
        y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1))
        y_true = scaler_y.inverse_transform(y_true_s.reshape(-1, 1))

        metrics = {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred)}

        logger.info(
            f"[{name}] Test metrics (original scale): "
            f"RMSE={metrics['RMSE']:.6f} | MAE={metrics['MAE']:.6f}"
        )
        return y_true, y_pred, metrics

    y_true_silu, y_pred_silu, m_silu = eval_on_test(model_silu, "SiLU")

    # =============================================================================
    # 9) Plots (saved into plots/)
    # =============================================================================
    os.makedirs("plots", exist_ok=True)

    # (a) Loss curves
    plt.figure()
    plt.plot(hist_silu["train"], label="SiLU train")
    plt.plot(hist_silu["val"], label="SiLU val")
    plt.xlabel("Epoch")
    plt.ylabel("SmoothL1 loss (scaled y)")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    LOSS = os.path.join("plots", "LOSS.png")
    plt.savefig(LOSS, dpi=200, bbox_inches="tight")
    logger.info(f"Saved plot: {LOSS}")
    plt.show()

    # (b) True vs Pred (SiLU)
    plt.figure()
    plt.scatter(y_true_silu, y_pred_silu, alpha=0.6, label="SiLU")
    mn = float(min(y_true_silu.min(), y_pred_silu.min()))
    mx = float(max(y_true_silu.max(), y_pred_silu.max()))
    plt.plot([mn, mx], [mn, mx], "--", color="red", label="Ideal")
    plt.xlabel("True E_eff")
    plt.ylabel("Predicted E_eff")
    plt.title("True vs Predicted (SiLU) - test")
    plt.legend()
    plt.tight_layout()
    SILU = os.path.join("plots", "SILU.png")
    plt.savefig(SILU, dpi=200, bbox_inches="tight")
    logger.info(f"Saved plot: {SILU}")
    plt.show()

    # (c) Target distribution (y is in GPa after /1000)
    plt.figure()
    plt.hist(y, bins=40, edgecolor="black")
    plt.xlabel("E_eff [GPa]")
    plt.ylabel("Count")
    plt.title("Target distribution")
    TARGET_HIST = os.path.join("plots", "TARGET_HIST.png")
    plt.tight_layout()
    plt.savefig(TARGET_HIST, dpi=200, bbox_inches="tight")
    logger.info(f"Saved plot: {TARGET_HIST}")
    plt.show()


if __name__ == "__main__":
    main()
