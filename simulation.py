


"""
PARAMETRIC TENSION PLATE FEA
Automated Batch Simulation for Material Characterization

Pipeline:
1) Load simulation configuration from a YAML file
2) Generate geometric samples using Latin Hypercube Sampling (LHS)
3) Build plate geometries with two elliptical holes per sample
4) Generate adaptive FE meshes (field-based sizing) and run mesh QC
5) Run parallel structural analyses using CalculiX
6) Compute effective Young's modulus via homogenization for each case
7) Export aggregated results to CSV + optional diagnostics

Author: Andrea Vinarš
"""

from __future__ import annotations

import logging
import multiprocessing
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gmsh
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless runs
import numpy as np
import yaml
from scipy.stats import qmc
from joblib import Parallel, delayed

from data_processing import (
    calculate_youngs_modulus,
    params_csv_histograms,
    save_rejected_csv)


from contextlib import contextmanager
from tqdm import tqdm

@contextmanager
def tqdm_joblib(t):
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
    return logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")


def setup_simulation_logger(sim_number: int, logs_dir: Path) -> logging.Logger:
    
    """
    Create a dedicated file logger for one simulation (safe for parallel runs).
    """
    
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"Simulation_{sim_number:04d}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Reset handlers (important in some parallel/interactive contexts)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    log_file = logs_dir / f"sim_{sim_number:04d}.log"
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(_make_formatter())
    logger.addHandler(fh)

    return logger


def setup_main_logger(logs_dir: Path) -> logging.Logger:
    
    """
    Main logger for orchestration (file + console).
    """
    
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
    
    """
    Close all handlers to avoid file locks (especially on Windows).
    """
    
    for h in list(logger.handlers):
        try:
            h.flush()
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)


def safe_gmsh_finalize() -> None:
    
    """
    Best-effort Gmsh cleanup (avoids crashes/locks in long parallel runs).
    """
    
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
# Config loading + normalization
# =============================================================================


def load_config(config_path: Path) -> Dict[str, Any]:
    
    """
    Load YAML configuration.
    """
    
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: Dict[str, Any], config_path: Path) -> Dict[str, Path]:
    
    """
    Resolve paths to absolute Path objects.
    - If working_dir is relative, interpret it relative to the config file location.
    """
    
    cfg_dir = config_path.resolve().parent

    working_dir = Path(cfg["paths"]["working_dir"])
    if not working_dir.is_absolute():
        working_dir = (cfg_dir / working_dir).resolve()

    calculix_path = Path(cfg["paths"]["calculix"])
    if not calculix_path.is_absolute():
        calculix_path = (cfg_dir / calculix_path).resolve()

    input_dir = Path(cfg["output"]["input_files"])
    output_dir = Path(cfg["output"]["output_files"])
    data_dir = Path(cfg["output"]["data_dir"])

    # Make dirs relative to working_dir if needed
    if not input_dir.is_absolute():
        input_dir = (working_dir / input_dir).resolve()
    if not output_dir.is_absolute():
        output_dir = (working_dir / output_dir).resolve()
    if not data_dir.is_absolute():
        data_dir = (working_dir / data_dir).resolve()

    logs_dir = working_dir / "logs"

    return {
        "working_dir": working_dir,
        "calculix": calculix_path,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "data_dir": data_dir,
        "logs_dir": logs_dir,
    }


# =============================================================================
# Parameter generation (LHS)
# =============================================================================


def generate_lhs_params_independent(
    n_samples: int,
    lower_bounds: List[float],
    upper_bounds: List[float],
    seed: int = 1,
) -> List[Dict[str, Dict[str, float]]]:
    
    """
    Generate geometric parameter sets using Latin Hypercube Sampling (LHS).

    Each sample encodes 2 ellipses (10 dimensions total):
      hole1: x, y, rx, ry, angle
      hole2: x, y, rx, ry, angle
    """
    
    sampler = qmc.LatinHypercube(d=10, seed=seed)
    sample = sampler.random(n_samples)

    params_scaled = qmc.scale(sample, lower_bounds, upper_bounds)

    params_list: List[Dict[str, Dict[str, float]]] = []
    for row in params_scaled:
        
        # Design choice (author note):
        # I enforce rx >= ry by sorting (rx, ry) after LHS scaling.
        # This avoids rejection sampling (ry > rx cases) and keeps the pipeline fast/stable.
        # Trade-off: marginal distributions are slightly distorted vs. unconstrained LHS.
        # Empirically, this approach preserved low cross-correlation better than my earlier constrained-sampling variant.
        
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


# =============================================================================
# Mesh generation (Gmsh)
# =============================================================================


def create_mesh(
    params: Dict[str, Dict[str, float]],
    sim_number: int,
    plate_width: float,
    plate_height: float,
    size_min: float,
    size_max: float,
) -> Dict[str, int]:
    
    """
    Create a 2D mesh for a rectangular plate with two elliptical holes.

    Meshing uses field-based sizing (Distance -> Min -> Threshold) to provide
    robust refinement around holes even when CAD boolean operations split curves.
    """
    
    x1 = params["hole1"]["x"]
    y1 = params["hole1"]["y"]
    rx1 = params["hole1"]["rx"]
    ry1 = params["hole1"]["ry"]
    angle1 = params["hole1"]["angle"]

    x2 = params["hole2"]["x"]
    y2 = params["hole2"]["y"]
    rx2 = params["hole2"]["rx"]
    ry2 = params["hole2"]["ry"]
    angle2 = params["hole2"]["angle"]

    gmsh.model.add(f"tensile_plate_{sim_number:04d}")

    # Base plate (XY plane)
    plate = gmsh.model.occ.addRectangle(0, 0, 0, plate_width, plate_height)

    # Ellipse 1
    e1_curve = gmsh.model.occ.addEllipse(x1, y1, 0, rx1, ry1)
    gmsh.model.occ.rotate([(1, e1_curve)], x1, y1, 0, 0, 0, 1, angle1 * np.pi / 180.0)
    e1_loop = gmsh.model.occ.addCurveLoop([e1_curve])
    hole1 = gmsh.model.occ.addPlaneSurface([e1_loop])

    # Ellipse 2
    e2_curve = gmsh.model.occ.addEllipse(x2, y2, 0, rx2, ry2)
    gmsh.model.occ.rotate([(1, e2_curve)], x2, y2, 0, 0, 0, 1, angle2 * np.pi / 180.0)
    e2_loop = gmsh.model.occ.addCurveLoop([e2_curve])
    hole2 = gmsh.model.occ.addPlaneSurface([e2_loop])

    # Center points used by distance fields
    center1 = gmsh.model.occ.addPoint(x1, y1, 0)
    center2 = gmsh.model.occ.addPoint(x2, y2, 0)

    gmsh.model.occ.synchronize()

    # Boolean cut: plate - holes
    cut_result = gmsh.model.occ.cut([(2, plate)], [(2, hole1), (2, hole2)])
    if not cut_result[0]:
        raise RuntimeError("Boolean cut failed - holes might be outside plate or invalid geometry.")

    new_plate_tag = cut_result[0][0][1]
    gmsh.model.occ.synchronize()

    # -------------------------------------------------------------------------
    # Field-based mesh sizing (Distance -> Min -> Threshold)
    # -------------------------------------------------------------------------
    # Author note:
    # I use distance-field sizing instead of curve-based refinement because intersections can split curves into
    # multiple entities, making direct ellipse-curve refinement unreliable.

    sampling1 = 100 if rx1 < 1 else 200
    sampling2 = 100 if rx2 < 1 else 200

    field1 = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(field1, "PointsList", [center1])
    gmsh.model.mesh.field.setNumber(field1, "Sampling", sampling1)

    field2 = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(field2, "PointsList", [center2])
    gmsh.model.mesh.field.setNumber(field2, "Sampling", sampling2)

    field_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(field_min, "FieldsList", [field1, field2])

    min_ry = min(ry1, ry2)
    max_rx = max(rx1, rx2)

    # Apply threshold: fine mesh near holes, coarse mesh far away
    field_threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(field_threshold, "InField", field_min)
    gmsh.model.mesh.field.setNumber(field_threshold, "SizeMin", float(size_min))
    gmsh.model.mesh.field.setNumber(field_threshold, "SizeMax", float(size_max))
    gmsh.model.mesh.field.setNumber(field_threshold, "DistMin", min_ry)
    gmsh.model.mesh.field.setNumber(field_threshold, "DistMax", max_rx * 3.0)

    gmsh.model.mesh.field.setAsBackgroundMesh(field_threshold)

    # Global meshing options
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # algorithm id is version-dependent in Gmsh
    gmsh.option.setNumber("Mesh.Smoothing", 3)
    gmsh.option.setNumber("Mesh.MinCircleNodes", 32)
    gmsh.option.setNumber("Mesh.MinCurveNodes", 8)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)  # avoid fighting the sizing field

    # Generate and optimize mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D")  # 2D optimization for the first-order elements
    gmsh.model.mesh.setOrder(2)            # Second-order elements
    # HighOrder optimizations turned out to be too aggressive and would often break down the kernel

    return {"new_plate_tag": new_plate_tag}


def mesh_quality_report(
    element_qualities: np.ndarray,
    logger: logging.Logger,
    poor_quality_threshold: float = 0.6,
) -> Tuple[float, int]:
    
    """
    Log a compact histogram + return (avg_quality, poor_count).
    """
    
    logger.info("")
    logger.info("ELEMENT QUALITY HISTOGRAM:")
    for lower in np.arange(0.0, 1.0, 0.1):
        upper = lower + 0.1
        mask = (element_qualities >= lower) & (element_qualities < upper)
        count = int(np.sum(mask))
        pct = 100.0 * count / max(len(element_qualities), 1)
        logger.info(f" {lower:.2f}-{upper:.2f}: {count:5d} ({pct:5.1f}%)")

    poor_count = int(np.sum(element_qualities < poor_quality_threshold))
    avg_quality = float(np.mean(element_qualities)) if len(element_qualities) else 0.0
    logger.info(f"Average quality: {avg_quality:.3f}")
    logger.info(f"Poor elements (<{poor_quality_threshold:.2f}): {poor_count}")
    logger.info("")
    return avg_quality, poor_count


def collect_boundary_nodes(
    plate_width: float,
    tolerance: float = 1e-4,
) -> Tuple[List[int], List[int]]:
    
    """
    Collect node tags on left (x=0) and right (x=plate_width) boundaries.

    TOLERANCE: Mesh nodes are exactly at x=0 and x=width, but floating point
    representation can be x=1e-16 or x=width+1e-16.
    1e-4 >> typical node coordinate error (~1e-12) but << min element size (0.1)
    """
    
    left_nodes: List[int] = []
    right_nodes: List[int] = []

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords.reshape(-1, 3)

    for i, tag in enumerate(node_tags):
        x = float(node_coords[i][0])
        if abs(x) < tolerance:
            left_nodes.append(int(tag))
        if abs(x - plate_width) < tolerance:
            right_nodes.append(int(tag))

    return left_nodes, right_nodes


def format_nset(name: str, nodes: List[int], per_line: int = 16) -> str:
    
    """
    Create a CalculiX *NSET block for a given list of nodes.
    """
    
    lines = [f"*NSET, NSET={name}"]
    current: List[str] = []

    for n in nodes:
        current.append(str(n))
        if len(current) >= per_line:
            lines.append(", ".join(current))
            current = []

    if current:
        lines.append(", ".join(current))

    return "\n".join(lines) + "\n"


# =============================================================================
# CalculiX input generation
# =============================================================================


def generate_inp(
    params: Dict[str, Dict[str, float]],
    ccx_input: Path,
    sim_number: int,
    logger: logging.Logger,
    cfg: Dict[str, Any],
) -> Tuple[Path, float, int]:
    
    """
    Generate a CalculiX .inp file for one simulation, including mesh QC.

    Returns:
        (inp_path, avg_quality, poor_quality_count)
    """
    
    plate_width = float(cfg["plate"]["width"])
    plate_height = float(cfg["plate"]["height"])
    thickness = float(cfg["plate"]["thickness"])
    elongation = float(cfg["loading"]["elongation"])
    material_e = float(cfg["material"]["youngs_modulus"])
    material_nu = float(cfg["material"]["poisson_ratio"])
    material_name = str(cfg["material"]["name"])

    poor_quality_threshold = float(cfg["mesh_quality"].get("poor_quality_threshold", 0.6))
    temp_mesh_file = ccx_input.with_name(f"_temp_mesh_{sim_number:04d}.inp")

    gmsh.initialize()

    try:
        
        # Suppress Gmsh console output for performance
        # This only affects Gmsh internal messages, not Python logging
        gmsh.option.setNumber("General.Terminal", 0)

        # Log geometry parameters
        h1, h2 = params["hole1"], params["hole2"]
        logger.info("")
        logger.info("GEOMETRY PARAMETERS:")
        logger.info(f"  Plate: {plate_width} x {plate_height} mm, thickness = {thickness} mm")
        logger.info(f"  Hole 1: center=({h1['x']}, {h1['y']}), rx={h1['rx']}, ry={h1['ry']}, angle={h1['angle']}°")
        logger.info(f"  Hole 2: center=({h2['x']}, {h2['y']}), rx={h2['rx']}, ry={h2['ry']}, angle={h2['angle']}°")
        logger.info("")

        # Build mesh
        mesh_cfg = cfg.get("mesh", {})
        size_min = float(mesh_cfg.get("size_min"))
        size_max = float(mesh_cfg.get("size_max"))
    
        geom = create_mesh(
               params,
               sim_number,
               plate_width,
               plate_height,
               size_min=size_min,
               size_max=size_max)

        new_plate_tag = geom["new_plate_tag"]

        # Physical group for element set naming
        plate_pg = gmsh.model.addPhysicalGroup(2, [new_plate_tag])
        gmsh.model.setPhysicalName(2, plate_pg, "Plate")

        # Mesh quality
        element_data = gmsh.model.mesh.getElements(dim=2)
        element_tags = element_data[1][0]
        element_qualities = np.array(
            gmsh.model.mesh.getElementQualities(elementTags=element_tags),
            dtype=float,
        )

        avg_quality, poor_quality_elements = mesh_quality_report(
            element_qualities,
            logger=logger,
            poor_quality_threshold=poor_quality_threshold,
        )

        # Boundary node sets
        left_nodes, right_nodes = collect_boundary_nodes(plate_width=plate_width, tolerance=1e-4)
        logger.info(f"FIXED NODES: {len(left_nodes)} | LOAD NODES: {len(right_nodes)}")

        # Export mesh to temporary .inp (Gmsh format) and then patch it
        gmsh.write(str(temp_mesh_file))
        mesh_content = temp_mesh_file.read_text(encoding="utf-8", errors="ignore")

        # Modify node definition to include all nodes in a single node set.
        # This allows accessing all nodal results in CalculiX output files.
        mesh_content = mesh_content.replace("*NODE", "*NODE, NSET=NALL", 1)

        nset_fixed = format_nset("Fixed", left_nodes)
        nset_load = format_nset("Load", right_nodes)

        calculix_input = f"""{mesh_content}

{nset_fixed}

{nset_load}

*MATERIAL, NAME={material_name}
*ELASTIC
{material_e}, {material_nu}

*SOLID SECTION, ELSET=Plate, MATERIAL={material_name}
{thickness}

*STEP, NLGEOM=NO
*STATIC

** SUPPORT CONSTRAINTS
*BOUNDARY
Fixed, 1, 3, 0.0

** LOADING
*BOUNDARY
Load, 1, 1, {elongation}

** OUTPUT
*EL PRINT, ELSET=Plate
S

*END STEP
"""

        ccx_input.parent.mkdir(parents=True, exist_ok=True)
        ccx_input.write_text(calculix_input, encoding="utf-8")
        logger.info(f"Input file created: {ccx_input}")

        return ccx_input, avg_quality, poor_quality_elements

    finally:
        safe_gmsh_finalize()
        if temp_mesh_file.exists():
            try:
                temp_mesh_file.unlink()
            except Exception:
                pass


# =============================================================================
# CalculiX execution
# =============================================================================


def run_calculix_simulation(
    inp_path: Path,
    sim_number: int,
    logger: logging.Logger,
    calculix_path: Path,
    output_dir: Path,
    timeout_seconds: int,
) -> bool:
    
    """
    Run a CalculiX simulation for given .inp and move results into output_dir.
    """
    
    try:
        logger.info("Starting CalculiX simulation...")

        inp_base = inp_path.with_suffix("")  # full path without extension

        result = subprocess.run(
            [str(calculix_path), str(inp_base)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(inp_path.parent))
        
        if result.returncode != 0:
            logger.error("CalculiX returned a non-zero exit code.")
            logger.error(f"STDERR:\n{result.stderr}")
            return False

        output_dir.mkdir(parents=True, exist_ok=True)

        # CCX outputs are created next to the inp_base
        output_extensions = [".dat", ".frd", ".log", ".sta"]
        for ext in output_extensions:
            src_file = inp_base.with_suffix(ext)
            dst_file = output_dir / f"sim_{sim_number:04d}{ext}"

            if src_file.exists():
                shutil.move(str(src_file), str(dst_file))
                logger.info(f"Moved {ext} to {output_dir}")
            else:
                # Note: .log file is only created by CalculiX if there are warnings/errors
                if ext == ".log":
                    logger.info("No .log file generated (no warnings).")
                else:
                    logger.warning(f"Expected output not found: {src_file.name}")

        return True

    except subprocess.TimeoutExpired:
        logger.error("Timeout - simulation took too long.")
        return False
    except FileNotFoundError:
        logger.error(f"CalculiX not found at: {calculix_path}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while running CalculiX: {e}")
        return False


# =============================================================================
# Parallel worker
# =============================================================================


def run_single_simulation(
    i: int,
    params: Dict[str, Dict[str, float]],
    cfg: Dict[str, Any],
    paths: Dict[str, Path],
) -> Dict[str, Any]:
    
    """
    Run one simulation end-to-end (mesh -> inp -> optional skip -> ccx -> postproc).
    """
    
    logger = setup_simulation_logger(i, paths["logs_dir"])

    plate_width = float(cfg["plate"]["width"])
    elongation = float(cfg["loading"]["elongation"])
    timeout_seconds = int(cfg["simulation"]["timeout_seconds"])
    max_poor_elements = int(cfg["mesh_quality"]["max_poor_elements"])

    input_dir = paths["input_dir"]
    output_dir = paths["output_dir"]
    calculix_path = paths["calculix"]

    try:
        input_dir.mkdir(parents=True, exist_ok=True)
        ccx_input = input_dir / f"sim_{i:04d}.inp"

        logger.info("=" * 50)
        logger.info(f"STARTING SIMULATION {i:04d}")
        logger.info("=" * 50)

        # 1) Generate .inp and compute mesh quality
        ccx_input, avg_quality, poor_quality_elements = generate_inp(
            params=params,
            ccx_input=ccx_input,
            sim_number=i,
            logger=logger,
            cfg=cfg)

        params["avg_quality"] = avg_quality
        params["poor_elements"] = poor_quality_elements

        # 2) Skip early if mesh is too poor (saves a lot of runtime)
        if poor_quality_elements > max_poor_elements:
            reason = f"poor_elements ({poor_quality_elements}) > {max_poor_elements}"
            logger.warning(f"SKIPPED - {reason}")
            return {"index": i, "status": "skipped", "reason": reason, "params": params}

        # 3) Run CalculiX
        success = run_calculix_simulation(
            inp_path=ccx_input,
            sim_number=i,
            logger=logger,
            calculix_path=calculix_path,
            output_dir=output_dir,
            timeout_seconds=timeout_seconds)

        if not success:
            logger.error("Simulation failed (FEA solver error).")
            return {"index": i, "status": "failed", "reason": "FEA solver error", "params": params}

        # 4) Post-processing: compute effective modulus
        dat_path = output_dir / f"sim_{i:04d}.dat"
        if not dat_path.exists():
            logger.error(f"Missing .dat output for post-processing: {dat_path}")
            return {"index": i, "status": "failed", "reason": "missing dat output", "params": params}

        E_eff = calculate_youngs_modulus(
            width=plate_width,
            elongation=elongation,
            inp_path=str(ccx_input),
            dat_path=str(dat_path))
        
        if E_eff is None:
            reason = "postproc returned None (no stress data parsed?)"
            logger.error(reason)
            return {"index": i, "status": "failed", "reason": reason, "params": params}
        
        try:
            E_eff_val = float(E_eff)
        
        except (TypeError, ValueError):
            reason = f"postproc returned non-numeric value: {E_eff!r}"
            logger.error(reason)
            return {"index": i, "status": "failed", "reason": reason, "params": params}
        
        if (not np.isfinite(E_eff_val)) or (E_eff_val <= 0):
            reason = f"invalid E_eff={E_eff_val} (NaN/Inf/<=0)"
            logger.error(reason)
            return {"index": i, "status": "failed", "reason": reason, "params": params}
        
        params["E_eff"] = E_eff_val
        logger.info(f"Simulation completed successfully. E_eff = {E_eff_val:.2f} MPa")
        return {"index": i, "status": "success", "params": params}

    finally:
        close_logger(logger)


# =============================================================================
# Main
# =============================================================================


def configure_multiprocessing() -> None:
    
    """
    Windows requires 'spawn' for compatibility with some native libs (incl. Gmsh).
    """
    
    # On Windows, 'spawn' is the supported start method and is safer with native libraries (e.g., Gmsh).
    if sys.platform.startswith("win"):
        multiprocessing.set_start_method("spawn", force=True)


def main(config_path: str = "config.yaml") -> None:
    configure_multiprocessing()

    config_path_p = Path(config_path)
    cfg = load_config(config_path_p)
    paths = resolve_paths(cfg, config_path_p)

    # Create base directories
    paths["working_dir"].mkdir(parents=True, exist_ok=True)
    paths["input_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["data_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)

    main_logger = setup_main_logger(paths["logs_dir"])

    # Extract key settings
    num_sims = int(cfg["simulation"]["num_simulations"])
    n_jobs = int(cfg["simulation"]["n_jobs"])
    seed = int(cfg["simulation"]["seed"])

    plate_width = float(cfg["plate"]["width"])
    plate_height = float(cfg["plate"]["height"])
    elongation = float(cfg["loading"]["elongation"])
    material_e = float(cfg["material"]["youngs_modulus"])
    material_nu = float(cfg["material"]["poisson_ratio"])

    lower_bounds = list(cfg["lhs_bounds"]["lower"])
    upper_bounds = list(cfg["lhs_bounds"]["upper"])

    main_logger.info("=" * 70)
    main_logger.info("PARAMETRIC FEA SIMULATION - TENSILE PLATE WITH ELLIPTICAL HOLES")
    main_logger.info("=" * 70)
    main_logger.info("")
    main_logger.info("Configuration:")
    main_logger.info(f" Working dir: {paths['working_dir']}")
    main_logger.info(f" CalculiX: {paths['calculix']}")
    main_logger.info(f" Total simulations: {num_sims}")
    main_logger.info(f" Parallel jobs: {n_jobs}")
    main_logger.info(f" Plate dimensions: {plate_width} x {plate_height} mm")
    main_logger.info(f" Elongation: {elongation} mm")
    main_logger.info(f" Material: E={material_e} MPa, nu={material_nu}")
    main_logger.info("")

    # Generate parameter sets
    all_params = generate_lhs_params_independent(
        n_samples=num_sims,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        seed=seed,
    )

    start_time = time.time()

    with tqdm_joblib(tqdm(total=num_sims, desc="Simulations", unit="sim")):
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(run_single_simulation)(i, all_params[i], cfg, paths)
            for i in range(num_sims)
    )

    
    elapsed = time.time() - start_time
    main_logger.info(f"Total execution time: {elapsed:.1f} seconds")

    # Summaries
    successful = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    errors = sum(1 for r in results if r["status"] == "error")

    main_logger.info("")
    main_logger.info("=" * 70)
    main_logger.info("PARALLEL SIMULATION RESULTS:")
    main_logger.info("=" * 70)
    main_logger.info(f"Total simulations: {num_sims}")
    main_logger.info(f"Successful: {successful}")
    main_logger.info(f"Skipped: {skipped}")
    main_logger.info(f"Failed: {failed}")
    main_logger.info(f"Errors: {errors}")

    rejected_results = [r for r in results if r["status"] in {"skipped", "failed", "error"}]
    successful_results = [r["params"] for r in results if r["status"] == "success" and r.get("params") is not None]

    if successful_results:
        main_logger.info("")
        main_logger.info("Generating CSV and PNG files with parameters...")
        params_csv_histograms(successful_results, str(paths["data_dir"]))
        main_logger.info("Done!")

    # Author note:
    # In my runs (multiple seeds and current mesh/QC settings), only a small fraction of cases failed due to invalid
    # elements (e.g., nonpositive Jacobians) causing solver termination. Rejected cases are saved for diagnostics.
    # TODO: Consider automatic remediation (retry with stricter mesh settings) for borderline meshes.
    
    if rejected_results:
        main_logger.info("")
        main_logger.info(f"Found {len(rejected_results)} problematic simulations.")
        save_rejected_csv(rejected_results, str(paths["data_dir"]))

    main_logger.info("")
    main_logger.info(f"All logs saved to: {paths['logs_dir']}")
    main_logger.info("Individual simulation logs: sim_XXXX.log")
    main_logger.info("Main log: main.log")

    close_logger(main_logger)


if __name__ == "__main__":
    # CLI usage:
    # python simulation.py path/to/config.yaml
    # If omitted, defaults to "config.yaml" in the working directory.
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(cfg_path)
