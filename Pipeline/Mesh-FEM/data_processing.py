"""CalculiX postprocessing and dataset export for periodic RVE simulations.

This module parses element stresses, element volumes, and selected nodal
displacements from CalculiX ``.dat`` files. It computes homogenized stresses,
derives effective orthotropic elastic constants using either the constitutive
matrix or direct-contraction workflow, and exports ML-ready datasets and
diagnostic summaries.

Author: Andrea Vinarš
"""

import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_calculix_float(value: str) -> float:
    """Parse standard and Fortran-style scientific notation."""

    return float(value.replace("D", "E").replace("d", "e"))


def calculate_sigma_hom(dat_path: str) -> np.ndarray:
    """Return the volume-averaged stress vector from a CalculiX ``.dat`` file.

    The returned component order is ``[sigma_x, sigma_y, tau_xy]``. If the
    file contains multiple result increments, only the latest stress/volume
    block is retained. Integration-point stresses are averaged per element,
    then element averages are weighted by ``EVOL``. Stress values retain the
    unit system used in the CalculiX model; this project uses MPa.

    Args:
        dat_path: Path to a CalculiX text result file.

    Raises:
        RuntimeError: If required sections are missing, malformed, non-finite,
        duplicated, or inconsistent between stress and volume outputs.
    """

    stress_data = []
    volume_data = []

    reading_stress = False
    reading_volume = False

    with open(dat_path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            lower = line.lower()

            if "stresses" in lower and "(elem" in lower:
                # Starting a new stress section discards any earlier increment.
                stress_data = []
                volume_data = []
                reading_stress = True
                reading_volume = False
                continue

            if "volume" in lower and "(element" in lower:
                volume_data = []
                reading_stress = False
                reading_volume = True
                continue

            if " for set " in lower and " and time " in lower:
                # Any other CalculiX result header ends the active data section.
                reading_stress = False
                reading_volume = False
                continue

            if not line:
                continue

            if line.startswith("*"):
                reading_stress = False
                reading_volume = False
                continue

            if reading_stress:
                parts = line.split()

                try:
                    element_id = int(parts[0])
                except (IndexError, ValueError):
                    continue

                if len(parts) < 8:
                    raise RuntimeError(
                        f"Invalid stress data in {dat_path}, line {line_number}: {line}"
                    )

                try:
                    sxx = _parse_calculix_float(parts[2])
                    syy = _parse_calculix_float(parts[3])
                    sxy = _parse_calculix_float(parts[5])
                except ValueError as exc:
                    raise RuntimeError(
                        f"Invalid stress data in {dat_path}, line {line_number}: {line}"
                    ) from exc

                stress_data.append(
                    {
                        "element_id": element_id,
                        "Sxx_MPa": sxx,
                        "Syy_MPa": syy,
                        "Sxy_MPa": sxy,
                    }
                )

            if reading_volume:
                parts = line.split()

                try:
                    element_id = int(parts[0])
                except (IndexError, ValueError):
                    continue

                if len(parts) != 2:
                    raise RuntimeError(
                        f"Invalid volume data in {dat_path}, line {line_number}: {line}"
                    )

                try:
                    elem_volume = _parse_calculix_float(parts[1])
                except ValueError as exc:
                    raise RuntimeError(
                        f"Invalid volume data in {dat_path}, line {line_number}: {line}"
                    ) from exc

                volume_data.append(
                    {
                        "element_id": element_id,
                        "element_volume": elem_volume,
                    }
                )

    stress_df = pd.DataFrame(stress_data)
    volume_df = pd.DataFrame(volume_data)

    if stress_df.empty:
        raise RuntimeError(f"No stress data found in {dat_path}")

    if volume_df.empty:
        raise RuntimeError(f"No volume data found in {dat_path}")

    stress_columns = ["Sxx_MPa", "Syy_MPa", "Sxy_MPa"]

    if not np.isfinite(stress_df[stress_columns].to_numpy()).all():
        raise RuntimeError(f"Stress data contains non-finite values in {dat_path}")

    if not np.isfinite(volume_df["element_volume"].to_numpy()).all():
        raise RuntimeError(f"Volume data contains non-finite values in {dat_path}")

    if volume_df["element_id"].duplicated().any():
        raise RuntimeError(f"Duplicate element volumes found in {dat_path}")

    stress_df = stress_df.groupby("element_id")[stress_columns].mean().reset_index()

    stress_ids = set(stress_df["element_id"])
    volume_ids = set(volume_df["element_id"])

    if stress_ids != volume_ids:
        raise RuntimeError(
            f"Inconsistent stress and volume data in {dat_path}: "
            f"{len(stress_ids - volume_ids)} elements without volume and "
            f"{len(volume_ids - stress_ids)} elements without stress"
        )

    df = volume_df.merge(stress_df, on="element_id", how="inner")

    if df.empty:
        raise RuntimeError("Stress and volume data could not be merged.")

    total_volume = df["element_volume"].sum()

    if total_volume <= 0:
        raise RuntimeError("Total element volume is zero or negative.")

    sigma_x = (df["Sxx_MPa"] * df["element_volume"]).sum() / total_volume
    sigma_y = (df["Syy_MPa"] * df["element_volume"]).sum() / total_volume
    tau_xy = (df["Sxy_MPa"] * df["element_volume"]).sum() / total_volume

    return np.array([sigma_x, sigma_y, tau_xy], dtype=float)


def read_node_displacement(
    dat_path: str,
    node_id: int,
) -> Dict[str, float]:
    """Read the latest displacement result recorded for a selected node.

    Returns:
        A mapping with ``ux``, ``uy``, and ``uz`` components.

    Raises:
        RuntimeError: If the requested node is absent from displacement output.
    """

    node_id = int(node_id)
    reading_displacements = False
    latest_displacement = None

    with open(dat_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            lower = stripped.lower()

            if not stripped:
                continue

            if "displacements" in lower:
                reading_displacements = True
                continue

            if reading_displacements:
                parts = stripped.split()

                if len(parts) < 4:
                    continue

                try:
                    current_node = int(parts[0])
                    ux = _parse_calculix_float(parts[1])
                    uy = _parse_calculix_float(parts[2])
                    uz = _parse_calculix_float(parts[3])
                except ValueError:
                    continue

                if current_node == node_id:
                    latest_displacement = {
                        "ux": ux,
                        "uy": uy,
                        "uz": uz,
                    }

    if latest_displacement is None:
        raise RuntimeError(f"Displacement of node {node_id} was not found in {dat_path}")

    return latest_displacement


def compute_effective_constants_from_C(
    sigma_EX: np.ndarray,
    sigma_EY: np.ndarray,
    sigma_XY: np.ndarray,
    eps0: float,
    gamma0: float,
) -> Dict[str, float]:
    """Compute effective constants from three strain-controlled load cases.

    The homogenized stress vectors form the columns of the in-plane stiffness
    matrix ``C``. Its inverse ``S`` is used to obtain ``E_x``, ``E_y``,
    ``G_xy``, and ``NU_xy``. The three moduli retain the stress unit, which is
    MPa in this project, while ``NU_xy`` is dimensionless.
    """

    C = np.column_stack(
        [
            sigma_EX / eps0,
            sigma_EY / eps0,
            sigma_XY / gamma0,
        ]
    )

    logger.info(f"Matrix C:\n{C}")

    S = np.linalg.inv(C)

    E_x = 1.0 / S[0, 0]
    E_y = 1.0 / S[1, 1]
    G_xy = 1.0 / S[2, 2]

    # For uniaxial x stress, nu_xy = -eps_y / eps_x.
    NU_xy = -S[1, 0] / S[0, 0]

    return {
        "E_x": float(E_x),
        "E_y": float(E_y),
        "G_xy": float(G_xy),
        "NU_xy": float(NU_xy),
    }


def compute_effective_constants_with_contraction(
    sigma_EX: np.ndarray,
    sigma_EY: np.ndarray,
    sigma_XY: np.ndarray,
    disp_EX: Dict[str, float],
    plate_height: float,
    eps0: float,
    gamma0: float,
) -> Dict[str, float]:
    """Compute effective constants from direct-response load cases.

    ``E_x``, ``E_y``, and ``G_xy`` are obtained from the primary homogenized
    stress response in EX, EY, and XY. ``NU_xy`` is obtained from the free
    transverse displacement measured during EX loading. The moduli are
    returned in MPa for the project input convention; ``NU_xy`` is
    dimensionless.
    """

    E_x = sigma_EX[0] / eps0
    E_y = sigma_EY[1] / eps0
    G_xy = sigma_XY[2] / gamma0

    eps_y = disp_EX["uy"] / plate_height
    NU_xy = -eps_y / eps0

    return {
        "E_x": float(E_x),
        "E_y": float(E_y),
        "G_xy": float(G_xy),
        "NU_xy": float(NU_xy),
    }


def params_csv_histograms(
    seed: int,
    all_params: List[Dict],
    output_dir: str,
    dataset_mode: str,
) -> None:
    """Write successful simulation data and parameter histograms.

    Args:
        seed: LHS seed included in the dataset filename.
        all_params: Parameter dictionaries from successful simulations.
        output_dir: Directory used for the CSV and histogram image.
        dataset_mode: Homogenization workflow used to generate the targets.
    """

    output_dir = os.path.join(output_dir, dataset_mode)
    os.makedirs(output_dir, exist_ok=True)

    data: List[Dict] = []

    # Build one flat ML record per successful geometry.
    for i, params in enumerate(all_params):
        data.append(
            {
                "SIMULATION": i,
                "dataset_mode": dataset_mode,
                "x1": params["hole1"]["x"],
                "y1": params["hole1"]["y"],
                "rx1": params["hole1"]["rx"],
                "ry1": params["hole1"]["ry"],
                "angle1": params["hole1"]["angle"],
                "angle1_sin": round(np.sin(np.radians(params["hole1"]["angle"])), 4),
                "angle1_cos": round(np.cos(np.radians(params["hole1"]["angle"])), 4),
                "x2": params["hole2"]["x"],
                "y2": params["hole2"]["y"],
                "rx2": params["hole2"]["rx"],
                "ry2": params["hole2"]["ry"],
                "angle2": params["hole2"]["angle"],
                "angle2_sin": round(np.sin(np.radians(params["hole2"]["angle"])), 4),
                "angle2_cos": round(np.cos(np.radians(params["hole2"]["angle"])), 4),
                "delta_theta": round(
                    np.sin(np.radians(params["hole2"]["angle"]))
                    * np.sin(np.radians(params["hole1"]["angle"]))
                    + np.cos((np.radians(params["hole2"]["angle"])))
                    * np.cos((np.radians(params["hole1"]["angle"]))),
                    4,
                ),
                "dx": round(params["hole1"]["x"] - params["hole2"]["x"], 2),
                "dy": round(params["hole1"]["y"] - params["hole2"]["y"], 2),
                "d": round(
                    np.sqrt(
                        (params["hole1"]["x"] - params["hole2"]["x"]) ** 2
                        + (params["hole1"]["y"] - params["hole2"]["y"]) ** 2
                    ),
                    2,
                ),
                "A1": round(
                    float(np.pi) * params["hole1"]["rx"] * params["hole1"]["ry"], 2
                ),
                "A2": round(
                    float(np.pi) * params["hole2"]["rx"] * params["hole2"]["ry"], 2
                ),
                "E_x": params["E_x"],
                "E_y": params["E_y"],
                "G_xy": params["G_xy"],
                "NU_xy": params["NU_xy"],
                "avg_quality": params["avg_quality"],
            }
        )

    df = pd.DataFrame(data)
    csv_file = os.path.join(output_dir, f"ml_data_{dataset_mode}_seed_{seed}.csv")

    # A semicolon separator avoids conflicts with the configured decimal comma.
    df.to_csv(csv_file, index=False, sep=";", decimal=",")

    # Create a compact overview of the sampled geometric parameters.
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))
    fig.suptitle("GEOMETRICAL PARAMETERS", fontsize=16, fontweight="bold")

    # Entries are: column name, subplot row, subplot column, and color.
    params_to_plot = [
        ("x1", 0, 0, "red"),
        ("y1", 0, 1, "blue"),
        ("rx1", 0, 2, "magenta"),
        ("ry1", 0, 3, "yellow"),
        ("angle1", 0, 4, "green"),
        ("x2", 1, 0, "red"),
        ("y2", 1, 1, "blue"),
        ("rx2", 1, 2, "magenta"),
        ("ry2", 1, 3, "yellow"),
        ("angle2", 1, 4, "green"),
    ]

    for param_name, row, col, color in params_to_plot:
        axes[row, col].hist(
            df[param_name], bins=30, color=color, alpha=0.7, edgecolor="black"
        )
        axes[row, col].set_title(f"{param_name} distribution", fontweight="bold")

        if param_name.startswith("angle"):
            axes[row, col].set_xlabel(f"{param_name} [°]")
        else:
            axes[row, col].set_xlabel(f"{param_name} [mm]")

        axes[row, col].set_ylabel("n")
        axes[row, col].grid(True, alpha=0.5)

    plt.tight_layout()
    stat_file = os.path.join(output_dir, f"param_histograms_{dataset_mode}.png")
    plt.savefig(stat_file, dpi=200, bbox_inches="tight")
    logger.info(f"Statistics saved: {stat_file}")

    plt.close()


def save_rejected_csv(
    rejected_results: List[Dict],
    output_dir: str,
) -> None:
    """Save skipped and failed geometries for diagnostics or resimulation.

    This function enables debugging of problematic geometries and facilitates
    potential re-simulation of failed cases.

    Args:
        rejected_results: Result dictionaries with failed or skipped status.
        output_dir: Directory used for the rejected-simulation CSV.
    """

    if not rejected_results:
        return

    os.makedirs(output_dir, exist_ok=True)
    data: List[Dict] = []

    for result in rejected_results:
        params = result.get("params")
        if not params:
            continue

        row = {
            "SIMULATION": result["index"],
            "STATUS": result["status"],
            "REASON": result.get("reason", result.get("error", "Unknown")),
            "x1": params["hole1"]["x"],
            "y1": params["hole1"]["y"],
            "rx1": params["hole1"]["rx"],
            "ry1": params["hole1"]["ry"],
            "angle1": params["hole1"]["angle"],
            "angle1_sin": round(np.sin(np.radians(params["hole1"]["angle"])), 4),
            "angle1_cos": round(np.cos(np.radians(params["hole1"]["angle"])), 4),
            "x2": params["hole2"]["x"],
            "y2": params["hole2"]["y"],
            "rx2": params["hole2"]["rx"],
            "ry2": params["hole2"]["ry"],
            "angle2": params["hole2"]["angle"],
            "angle2_sin": round(np.sin(np.radians(params["hole2"]["angle"])), 4),
            "angle2_cos": round(np.cos(np.radians(params["hole2"]["angle"])), 4),
        }
        data.append(row)

    if data:
        df = pd.DataFrame(data)
        csv_file = os.path.join(output_dir, "rejected_simulations.csv")
        df.to_csv(csv_file, index=False, sep=";", decimal=",")

        logger.info(f"Rejected simulations saved: {csv_file}")
        logger.info(f"  - Total rejected: {len(data)}")

        status_counts = df["STATUS"].value_counts()
        for status, count in status_counts.items():
            logger.info(f"    * {status}: {count}")

    plt.close()
