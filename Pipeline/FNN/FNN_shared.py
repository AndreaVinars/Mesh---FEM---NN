"""Shared feature definitions and architecture for FNN training and inference.

This module is the single source of truth for feature order, target order,
derived geometric quantities, and the four-output surrogate architecture used
by both ``FNN.py`` and ``predict.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch.nn as nn


BASE_FEATURE_NAMES = (
    "x1", "y1", "rx1", "ry1", "angle1_sin", "angle1_cos",
    "x2", "y2", "rx2", "ry2", "angle2_sin", "angle2_cos",
    "d", "dx", "dy", "A1", "A2", "delta_theta",
)

DERIVED_FEATURE_NAMES = (
    "A_diff", "A_ratio", "aspect_1", "aspect_2",
    "rx_sum", "ry_sum", "rx_diff", "ry_diff",
)

FEATURE_NAMES = BASE_FEATURE_NAMES + DERIVED_FEATURE_NAMES
TARGET_NAMES = ("E_x", "E_y", "G_xy", "NU_xy")

RATIO_EPSILON = 1e-8


class FNN(nn.Module):
    """Map 26 geometric features to four effective material properties.

    The network contains five 32-unit SiLU hidden layers, with dropout after
    the first two layers, followed by one output per entry in ``TARGET_NAMES``.
    """

    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), 32),
            nn.SiLU(),
            nn.Dropout(0.01),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Dropout(0.01),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, len(TARGET_NAMES)),
        )

    def forward(self, x):
        return self.net(x)


def add_derived_features(df: Any) -> Any:
    """Add the eight derived feature columns to ``df`` in place and return it."""

    df["A_diff"] = np.round(np.abs(df["A1"] - df["A2"]), 3)
    df["A_ratio"] = np.round(df["A1"] / (df["A2"] + RATIO_EPSILON), 3)
    df["aspect_1"] = np.round(df["rx1"] / (df["ry1"] + RATIO_EPSILON), 3)
    df["aspect_2"] = np.round(df["rx2"] / (df["ry2"] + RATIO_EPSILON), 3)
    df["rx_sum"] = np.round(df["rx1"] + df["rx2"], 3)
    df["ry_sum"] = np.round(df["ry1"] + df["ry2"], 3)
    df["rx_diff"] = np.round(np.abs(df["rx1"] - df["rx2"]), 3)
    df["ry_diff"] = np.round(np.abs(df["ry1"] - df["ry2"]), 3)

    return df


def build_feature_vector(
    x1: float,
    y1: float,
    rx1: float,
    ry1: float,
    angle1_deg: float,
    x2: float,
    y2: float,
    rx2: float,
    ry2: float,
    angle2_deg: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Build one inference row from the geometry of two ellipses.

    The returned array follows ``FEATURE_NAMES`` exactly. The accompanying
    dictionary exposes the same values by name for diagnostics and reporting.
    """

    if min(rx1, ry1, rx2, ry2) <= 0:
        raise ValueError("Ellipse semi-axes must be positive.")
    if rx1 < ry1 or rx2 < ry2:
        raise ValueError("The model expects rx >= ry for both ellipses.")

    angle1_rad = np.radians(angle1_deg)
    angle2_rad = np.radians(angle2_deg)

    angle1_sin = round(np.sin(angle1_rad), 4)
    angle1_cos = round(np.cos(angle1_rad), 4)
    angle2_sin = round(np.sin(angle2_rad), 4)
    angle2_cos = round(np.cos(angle2_rad), 4)

    delta_theta = round(
        np.sin(angle2_rad) * np.sin(angle1_rad)
        + np.cos(angle2_rad) * np.cos(angle1_rad),
        4,
    )

    dx = round(x1 - x2, 2)
    dy = round(y1 - y2, 2)
    d = round(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), 2)

    area1 = round(float(np.pi) * rx1 * ry1, 2)
    area2 = round(float(np.pi) * rx2 * ry2, 2)

    feature_values = (
        x1, y1, rx1, ry1, angle1_sin, angle1_cos,
        x2, y2, rx2, ry2, angle2_sin, angle2_cos,
        d, dx, dy, area1, area2, delta_theta,
        round(abs(area1 - area2), 3),
        round(area1 / (area2 + RATIO_EPSILON), 3),
        round(rx1 / (ry1 + RATIO_EPSILON), 3),
        round(rx2 / (ry2 + RATIO_EPSILON), 3),
        round(rx1 + rx2, 3),
        round(ry1 + ry2, 3),
        round(abs(rx1 - rx2), 3),
        round(abs(ry1 - ry2), 3),
    )

    feature_data = dict(zip(FEATURE_NAMES, feature_values, strict=True))
    feature_vector = np.asarray([feature_values], dtype=np.float32)

    return feature_vector, feature_data
