"""Shared utilities for configuration, logging, sampling, and mesh checks."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import yaml
from scipy.stats import qmc


@contextmanager
def tqdm_joblib(t) -> Iterator[None]:
    """Forward completed joblib batches to a tqdm progress bar."""

    import joblib

    class TqdmCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            t.update(self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmCallback
    try:
        yield t
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        t.close()


# =============================================================================
# Logging
# =============================================================================


def _make_formatter() -> logging.Formatter:
    """Create the timestamped formatter shared by all simulation loggers."""

    return logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_simulation_logger(case) -> logging.Logger:
    """Create an isolated file logger for one parallel geometry worker."""

    ctx = case.ctx
    logs_dir = ctx.logs_dir

    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"Simulation_{case.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # A reused interpreter may still hold handlers from an earlier run.
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    log_file = logs_dir / f"{case.name}.log"
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(_make_formatter())
    logger.addHandler(fh)

    return logger


def setup_main_logger(ctx) -> logging.Logger:
    """Create the main orchestration logger for file and console output."""

    logs_dir = ctx.logs_dir

    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("Main")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    fh = logging.FileHandler(logs_dir / "main.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(_make_formatter())
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_make_formatter())
    logger.addHandler(ch)

    return logger


def close_logger(logger: logging.Logger) -> None:
    """Flush and close logger handlers to release file locks."""

    for h in list(logger.handlers):
        try:
            h.flush()
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)


def safe_gmsh_finalize() -> None:
    """Best-effort Gmsh cleanup for long-running parallel batches."""

    import gmsh

    try:
        if gmsh.isInitialized():
            try:
                gmsh.clear()
            except Exception:
                pass
            gmsh.finalize()
    except Exception:
        pass


# =============================================================================
# Configuration loading
# =============================================================================


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file with safe parsing."""

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")

    return config


# =============================================================================
# Parameter generation (LHS)
# =============================================================================


def generate_lhs_params_independent(
    ctx: SimpleNamespace,
    seed: int = 1,
) -> List[Dict[str, Dict[str, float]]]:
    """Generate two-ellipse geometries with Latin Hypercube Sampling.

    Each sample contains ten values in the order
    ``x, y, rx, ry, angle`` for hole 1 and then hole 2. Semi-axes are sorted
    after scaling so that ``rx >= ry`` for every generated ellipse.
    """

    sampler = qmc.LatinHypercube(d=10, seed=seed)
    sample = sampler.random(ctx.num_sims)

    params_scaled = qmc.scale(sample, ctx.lower_bounds, ctx.upper_bounds)

    params_list: List[Dict[str, Dict[str, float]]] = []
    for row in params_scaled:
        # Sorting avoids rejection sampling, at the cost of slightly changing
        # the marginal distributions of the two semi-axes.
        rx1, ry1 = sorted([row[2], row[3]], reverse=True)
        rx2, ry2 = sorted([row[7], row[8]], reverse=True)

        hole1 = {
            "x": round(float(row[0]), 2),
            "y": round(float(row[1]), 2),
            "rx": round(float(rx1), 2),
            "ry": round(float(ry1), 2),
            "angle": round(float(row[4]), 1),
        }
        hole2 = {
            "x": round(float(row[5]), 2),
            "y": round(float(row[6]), 2),
            "rx": round(float(rx2), 2),
            "ry": round(float(ry2), 2),
            "angle": round(float(row[9]), 1),
        }
        params_list.append({"hole1": hole1, "hole2": hole2})

    return params_list


def mesh_quality_report(
    element_qualities: np.ndarray,
    logger: logging.Logger,
    ctx: SimpleNamespace,
) -> Tuple[float, int]:
    """Log the quality distribution and return its mean and poor count."""

    logger.info("")
    logger.info("ELEMENT QUALITY HISTOGRAM:")
    for lower in np.arange(-0.1, 1.0, 0.1):
        upper = lower + 0.1
        mask = (element_qualities >= lower) & (element_qualities < upper)
        count = int(np.sum(mask))
        pct = 100.0 * count / max(len(element_qualities), 1)
        logger.info(f" {lower:.2f}-{upper:.2f}: {count:5d} ({pct:5.1f}%)")

    poor_count = int(np.sum(element_qualities < ctx.poor_quality_threshold))
    avg_quality = float(np.mean(element_qualities)) if len(element_qualities) else 0.0
    logger.info(f"Average quality: {avg_quality:.3f}")
    logger.info(f"Poor elements (<{ctx.poor_quality_threshold:.2f}): {poor_count}")
    logger.info("")

    return avg_quality, poor_count


def count_nonpositive_jacobian(
    logger: logging.Logger,
    tol: float = 0.0,
) -> int:
    """Count 2D elements whose minimum Jacobian determinant is not positive.

    Gmsh evaluates ``minDetJac`` across each element. Values less than or equal
    to ``tol`` indicate an inverted or degenerate element for the current mesh.
    """

    import gmsh

    _, element_tags_by_type, _ = gmsh.model.mesh.getElements(dim=2)

    nonempty_tag_groups = [
        np.asarray(tags, dtype=int) for tags in element_tags_by_type if len(tags) > 0
    ]
    if not nonempty_tag_groups:
        raise RuntimeError("The Gmsh model contains no 2D elements.")

    element_tags = np.concatenate(nonempty_tag_groups)

    min_det_jac = np.array(
        gmsh.model.mesh.getElementQualities(
            elementTags=element_tags,
            qualityName="minDetJac",
        ),
        dtype=float,
    )

    bad_elems = int(np.sum(min_det_jac <= tol))

    if bad_elems > 0:
        logger.warning(
            f"Found {bad_elems} elements with nonpositive Jacobian determinant."
        )

    return bad_elems
