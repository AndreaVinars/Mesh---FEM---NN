"""Configuration normalization and per-load-case path construction.

The simulation reads YAML data once and stores normalized values in a shared
``SimpleNamespace``. Worker functions then use this context instead of reading
the configuration file independently in each process.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict


VALID_CALCULATION_MODES = {"direct_contraction", "from_c"}


def _resolve_path(value: str, base_dir: Path) -> Path:
    """Return an absolute path, resolving relative values from ``base_dir``."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def build_ctx(
    cfg: Dict[str, Any],
    config_path: Path,
) -> SimpleNamespace:
    """Normalize YAML configuration values into a worker-safe context.

    Relative working and CalculiX paths are resolved from the configuration
    file directory. Relative output paths are resolved from ``working_dir``.
    The ``CALCULIX_CMD`` environment variable, when set, overrides the
    executable path from YAML.

    Args:
        cfg: Parsed YAML configuration mapping.
        config_path: Path to the YAML file from which ``cfg`` was loaded.

    Returns:
        A namespace containing normalized geometry, loading, material, mesh,
        execution, sampling, and output settings.

    Raises:
        ValueError: If ``calculation_mode`` is not supported.
    """

    ctx = SimpleNamespace()

    # -------------------------------------------------------------------------
    # Resolve paths
    # -------------------------------------------------------------------------
    cfg_dir = config_path.resolve().parent

    working_dir = _resolve_path(cfg["paths"]["working_dir"], cfg_dir)

    calculix_value = os.environ.get("CALCULIX_CMD", cfg["paths"]["calculix"])
    calculix_path = _resolve_path(calculix_value, cfg_dir)

    input_dir = _resolve_path(cfg["output"]["input_files"], working_dir)
    output_dir = _resolve_path(cfg["output"]["output_files"], working_dir)
    data_dir = _resolve_path(cfg["output"]["data_dir"], working_dir)

    logs_dir = working_dir / "logs"

    # -------------------------------------------------------------------------
    # Plate
    # -------------------------------------------------------------------------
    ctx.plate_width = float(cfg["plate"]["width"])
    ctx.plate_height = float(cfg["plate"]["height"])
    ctx.thickness = float(cfg["plate"]["thickness"])

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------
    ctx.eps0 = float(cfg["loading"]["eps0"])
    ctx.gamma0 = float(cfg["loading"]["gamma0"])

    # -------------------------------------------------------------------------
    # Material
    # -------------------------------------------------------------------------
    ctx.material_name = str(cfg["material"]["name"])
    ctx.material_E = float(cfg["material"]["youngs_modulus"])
    ctx.material_nu = float(cfg["material"]["poisson_ratio"])

    # -------------------------------------------------------------------------
    # Mesh
    # -------------------------------------------------------------------------
    ctx.size_min = float(cfg["mesh"]["size_min"])
    ctx.size_max = float(cfg["mesh"]["size_max"])

    # -------------------------------------------------------------------------
    # Mesh quality
    # -------------------------------------------------------------------------
    ctx.poor_quality_threshold = float(
        cfg["mesh_quality"].get("poor_quality_threshold", 0.6)
    )
    ctx.max_poor_elements = int(cfg["mesh_quality"]["max_poor_elements"])

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------
    ctx.calculation_mode = (
        str(cfg["simulation"].get("calculation_mode", "direct_contraction"))
        .strip()
        .lower()
    )

    if ctx.calculation_mode not in VALID_CALCULATION_MODES:
        raise ValueError(
            f"Invalid calculation_mode: {ctx.calculation_mode!r}. "
            f"Choose one of: {sorted(VALID_CALCULATION_MODES)}"
        )

    ctx.num_sims = int(cfg["simulation"]["num_simulations"])
    ctx.n_jobs = int(cfg["simulation"]["n_jobs"])
    ctx.seed = int(cfg["simulation"]["seed"])
    ctx.timeout_seconds = int(cfg["simulation"]["timeout_seconds"])

    # -------------------------------------------------------------------------
    # LHS
    # -------------------------------------------------------------------------
    ctx.lower_bounds = list(cfg["lhs_bounds"]["lower"])
    ctx.upper_bounds = list(cfg["lhs_bounds"]["upper"])

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    ctx.working_dir = working_dir
    ctx.input_dir = input_dir
    ctx.output_dir = output_dir
    ctx.data_dir = data_dir
    ctx.logs_dir = logs_dir
    ctx.calculix_path = calculix_path

    return ctx


def make_dirs(ctx: SimpleNamespace) -> None:
    """Create the working, input, output, data, and log directories."""

    ctx.working_dir.mkdir(parents=True, exist_ok=True)
    ctx.input_dir.mkdir(parents=True, exist_ok=True)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    ctx.data_dir.mkdir(parents=True, exist_ok=True)
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)


def make_load_case(
    i: int,
    params: Dict[str, Dict[str, float]],
    logger: Any,
    ctx: SimpleNamespace,
    load_case: str,
) -> SimpleNamespace:
    """Create file paths and metadata for one geometry/load-case pair.

    File stems follow the pattern ``sim_0001_EX``, ``sim_0001_EY``, and
    ``sim_0001_XY``. CalculiX writes temporary outputs next to the input file;
    completed result paths point to ``ctx.output_dir``.
    """

    case = SimpleNamespace()

    case.i = i
    case.load_case = load_case
    case.job_name = load_case
    case.name = f"sim_{i:04d}_{load_case}"

    case.params = params
    case.logger = logger
    case.ctx = ctx

    case.inp_path = ctx.input_dir / f"{case.name}.inp"
    case.inp_base = case.inp_path.with_suffix("")
    case.d_path = ctx.input_dir / f"{case.name}.12d"

    case.dat_path = ctx.output_dir / f"{case.name}.dat"
    case.frd_path = ctx.output_dir / f"{case.name}.frd"
    case.log_path = ctx.output_dir / f"{case.name}.log"
    case.sta_path = ctx.output_dir / f"{case.name}.sta"

    return case
