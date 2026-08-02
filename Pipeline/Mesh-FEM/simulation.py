"""Parallel periodic-RVE simulations for plates with two elliptical holes.

Pipeline:
1. Load and normalize the YAML configuration.
2. Sample two-ellipse geometries with Latin Hypercube Sampling.
3. Build one Gmsh mesh and periodic edge mapping per geometry.
4. Apply mesh-quality fallbacks when curved second-order elements are invalid.
5. Reuse the accepted mesh for EX, EY, and XY CalculiX analyses.
6. Compute effective in-plane elastic constants from homogenized responses.
7. Export successful samples and diagnostics for rejected geometries.

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
from types import SimpleNamespace
from typing import Any, Dict

import gmsh
import matplotlib

# Batch workers must not open interactive plotting windows.
matplotlib.use("Agg")

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from data_processing import (
    calculate_sigma_hom,
    compute_effective_constants_from_C,
    compute_effective_constants_with_contraction,
    params_csv_histograms,
    read_node_displacement,
    save_rejected_csv,
)
from equation_block_corners import (
    build_equation_block_after_mesh,
    set_periodic_edges_before_mesh,
)
from simulation_context import build_ctx, make_dirs, make_load_case
from Helper_functions import (
    close_logger,
    count_nonpositive_jacobian,
    generate_lhs_params_independent,
    load_config,
    mesh_quality_report,
    safe_gmsh_finalize,
    setup_main_logger,
    setup_simulation_logger,
    tqdm_joblib,
)

# =============================================================================
# Mesh generation (Gmsh)
# =============================================================================


def create_base_model(
    params: Dict[str, Dict[str, float]],
    sim_number: int,
    logger: logging.Logger,
    ctx: SimpleNamespace,
) -> Dict[str, Any]:
    """Build and validate one periodic Gmsh mesh.

    The accepted mesh, periodic equation block, corner tags, and quality
    metrics are returned for reuse by all three CalculiX load cases.
    """

    temp_mesh_file = ctx.input_dir / f"_temp_mesh_{sim_number:04d}.inp"

    plate_width = ctx.plate_width
    plate_height = ctx.plate_height
    size_min = ctx.size_min
    size_max = ctx.size_max

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

    gmsh.initialize()

    try:
        gmsh.option.setNumber("General.Terminal", 0)

        logger.info("")
        logger.info("GEOMETRY PARAMETERS:")
        logger.info(
            f"  Plate: {plate_width} x {plate_height} mm,"
            f"thickness = {ctx.thickness} mm"
        )
        logger.info(
            f"  Hole 1: center=({x1}, {y1}), rx={rx1}, ry={ry1}, angle={angle1}°"
        )
        logger.info(
            f"  Hole 2: center=({x2}, {y2}), rx={rx2}, ry={ry2}, angle={angle2}°"
        )
        logger.info("")

        gmsh.model.add(f"tensile_plate_{sim_number:04d}")

        # Construct the rectangular RVE and subtract both elliptical holes.
        plate = gmsh.model.occ.addRectangle(0, 0, 0, plate_width, plate_height)

        e1_curve = gmsh.model.occ.addEllipse(x1, y1, 0, rx1, ry1)
        gmsh.model.occ.rotate(
            [(1, e1_curve)],
            x1,
            y1,
            0,
            0,
            0,
            1,
            angle1 * np.pi / 180.0,
        )
        e1_loop = gmsh.model.occ.addCurveLoop([e1_curve])
        hole1 = gmsh.model.occ.addPlaneSurface([e1_loop])

        e2_curve = gmsh.model.occ.addEllipse(x2, y2, 0, rx2, ry2)
        gmsh.model.occ.rotate(
            [(1, e2_curve)],
            x2,
            y2,
            0,
            0,
            0,
            1,
            angle2 * np.pi / 180.0,
        )
        e2_loop = gmsh.model.occ.addCurveLoop([e2_curve])
        hole2 = gmsh.model.occ.addPlaneSurface([e2_loop])

        gmsh.model.occ.synchronize()

        cut_result = gmsh.model.occ.cut(
            [(2, plate)],
            [(2, hole1), (2, hole2)],
        )

        if not cut_result[0]:
            raise RuntimeError(
                "Boolean cut failed - holes might be outside plate or invalid geometry."
            )

        new_plate_tag = cut_result[0][0][1]
        gmsh.model.occ.synchronize()

        # Refine the mesh near either hole using the nearest center distance.
        center1 = gmsh.model.occ.addPoint(x1, y1, 0)
        center2 = gmsh.model.occ.addPoint(x2, y2, 0)

        field1 = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field1, "PointsList", [center1])

        field2 = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field2, "PointsList", [center2])

        field_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(field_min, "FieldsList", [field1, field2])

        field_threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field_threshold, "InField", field_min)
        gmsh.model.mesh.field.setNumber(field_threshold, "SizeMin", float(size_min))
        gmsh.model.mesh.field.setNumber(field_threshold, "SizeMax", float(size_max))
        gmsh.model.mesh.field.setNumber(field_threshold, "DistMin", min(ry1, ry2))
        gmsh.model.mesh.field.setNumber(field_threshold, "DistMax", max(rx1, rx2) * 3.0)

        gmsh.model.mesh.field.setAsBackgroundMesh(field_threshold)

        # Global options use Gmsh's Frontal-Delaunay algorithm in 2D.
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Smoothing", 3)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        # Register opposite-edge mappings before generating any mesh nodes.
        periodic_edges = set_periodic_edges_before_mesh(
            surface_tag=new_plate_tag,
            height=plate_height,
            width=plate_width,
            logger=logger,
            tol=1e-6,
        )

        # Generate and optimize mesh
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Laplace2D")
        gmsh.model.mesh.setOrder(2)

        # Mesh quality
        element_data = gmsh.model.mesh.getElements(dim=2)
        element_tag_groups = element_data[1]
        if not element_tag_groups:
            raise RuntimeError(f"{ctx}: mesh contains no 2D elements")
        element_tags = np.concatenate(element_tag_groups)

        element_qualities = np.array(
            gmsh.model.mesh.getElementQualities(elementTags=element_tags),
            dtype=float,
        )

        bad_elems = sum(element_qualities <= 0)

        bad_jacobian_elements = count_nonpositive_jacobian(logger=logger, tol=0.0)

        if bad_elems > 0 or bad_jacobian_elements > 0:
            logger.warning(
                f"{ctx}: curved second-order mesh has {bad_elems} invalid elements. "
                f"{ctx}: curved second-order mesh has {bad_jacobian_elements} negative jacobians"
                "Regenerating as straight-sided second-order mesh"
            )

            # Clear existing mesh, keep geometry
            gmsh.model.mesh.clear()

            gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)

            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.optimize("Laplace2D")
            gmsh.model.mesh.setOrder(2)

            # Mesh quality
            element_data = gmsh.model.mesh.getElements(dim=2)
            element_tag_groups = element_data[1]
            if not element_tag_groups:
                raise RuntimeError(f"{ctx}: mesh contains no 2D elements")
            element_tags = np.concatenate(element_tag_groups)

            element_qualities = np.array(
                gmsh.model.mesh.getElementQualities(elementTags=element_tags),
                dtype=float,
            )

            bad_elems = sum(element_qualities <= 0)

            bad_jacobian_elements = count_nonpositive_jacobian(
                logger=logger, tol=0.0
            )

        if bad_elems > 0 or bad_jacobian_elements > 0:
            logger.warning(
                f"{ctx}: straight-sided second-order mesh has {bad_elems} invalid elements. "
                f"{ctx}: straight-sided second-order mesh has {bad_jacobian_elements} negative jacobians"
                "Regenerating as finer first-order mesh"
            )

            # Clear existing mesh, keep geometry
            gmsh.model.mesh.clear()

            gmsh.model.mesh.field.setNumber(
                field_threshold, "SizeMin", float(size_min / 2.0)
            )
            gmsh.model.mesh.field.setNumber(
                field_threshold, "SizeMax", float(size_max / 2.0)
            )

            # Generate and optimize mesh
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.optimize("Laplace2D")

            # Mesh quality
            element_data = gmsh.model.mesh.getElements(dim=2)
            element_tag_groups = element_data[1]
            if not element_tag_groups:
                raise RuntimeError(f"{ctx}: mesh contains no 2D elements")
            element_tags = np.concatenate(element_tag_groups)

            element_qualities = np.array(
                gmsh.model.mesh.getElementQualities(elementTags=element_tags),
                dtype=float,
            )

            bad_elems = sum(element_qualities <= 0)

            bad_jacobian_elements = count_nonpositive_jacobian(
                logger=logger, tol=0.0
            )

            if bad_elems > 0 or bad_jacobian_elements > 0:
                raise RuntimeError(
                    f"{ctx}: even finer first-order mesh has {bad_elems} invalid elements "
                    f"{ctx}: first-order mesh has {bad_jacobian_elements} negative jacobians"
                )

        # Node equations must be built from the final accepted mesh.
        (
            equation_block,
            corner_1,
            corner_2,
            corner_3,
            corner_4,
        ) = build_equation_block_after_mesh(
            periodic_edges=periodic_edges,
            height=plate_height,
            width=plate_width,
            logger=logger,
            tol=1e-6,
        )

        # The physical name becomes the CalculiX element set ``Plate``.
        plate_pg = gmsh.model.addPhysicalGroup(2, [new_plate_tag])
        gmsh.model.setPhysicalName(2, plate_pg, "Plate")

        avg_quality, poor_quality_elements = mesh_quality_report(
            element_qualities=element_qualities,
            logger=logger,
            ctx=ctx,
        )

        # Export once; all load cases reuse the exact same mesh and equations.
        gmsh.write(str(temp_mesh_file))

        mesh_content = temp_mesh_file.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        mesh_content = mesh_content.replace("*NODE", "*NODE, NSET=NALL", 1)

        return {
            "mesh_content": mesh_content,
            "equation_block": equation_block,
            "corner_1": corner_1,
            "corner_2": corner_2,
            "corner_3": corner_3,
            "corner_4": corner_4,
            "avg_quality": avg_quality,
            "poor_quality_elements": poor_quality_elements,
        }

    finally:
        safe_gmsh_finalize()

        if temp_mesh_file.exists():
            try:
                temp_mesh_file.unlink()
            except Exception:
                pass


# =============================================================================
# CalculiX input generation
# =============================================================================


def generate_inp(
    mesh_data: Dict[str, Any],
    case: SimpleNamespace,
) -> Path:
    """Write one CalculiX input deck from the shared periodic mesh.

    ``from_c`` prescribes all macroscopic strain components required to build
    one column of the constitutive matrix. ``direct_contraction`` leaves the
    transverse reference displacement free in the normal load cases so that
    Poisson contraction can be measured directly during EX.
    """

    ctx = case.ctx
    logger = case.logger
    load_case = case.load_case
    ccx_input = case.inp_path

    if ctx.calculation_mode == "from_c":
        u2x = 0.0
        u2y = 0.0
        u4x = 0.0
        u4y = 0.0

        if load_case == "EX":
            u2x = ctx.eps0 * ctx.plate_width

        elif load_case == "EY":
            u4y = ctx.eps0 * ctx.plate_height

        elif load_case == "XY":
            u2y = ctx.gamma0 * ctx.plate_width

        else:
            raise ValueError("load_case needs to be EX, EY or XY")

        calculix_input = f"""{mesh_data["mesh_content"]}

{mesh_data["equation_block"]}

*MATERIAL, NAME={ctx.material_name}
*ELASTIC
{ctx.material_E}, {ctx.material_nu}

*SOLID SECTION, ELSET=Plate, MATERIAL={ctx.material_name}
{ctx.thickness}

*STEP, NLGEOM=NO
*STATIC

** REFERENCE-CORNER BOUNDARY CONDITIONS
*BOUNDARY
{mesh_data["corner_1"]}, 1, 3, 0.0
{mesh_data["corner_2"]}, 1, 1, {u2x}
{mesh_data["corner_2"]}, 2, 2, {u2y}
{mesh_data["corner_4"]}, 1, 1, {u4x}
{mesh_data["corner_4"]}, 2, 2, {u4y}

** OUTPUT
*EL PRINT, ELSET=Plate, GLOBAL=YES
S, EVOL

*END STEP
"""

    elif ctx.calculation_mode == "direct_contraction":
        u2x = 0.0
        u2y = 0.0
        u4x = 0.0
        u4y = 0.0

        if load_case == "EX":
            u2x, u2y = ctx.eps0 * ctx.plate_width, 0.0

            deformation = f"""{mesh_data["corner_2"]}, 1, 1, {u2x}
{mesh_data["corner_2"]}, 2, 2, {u2y}"""

        elif load_case == "EY":
            u2y = 0.0
            u4y = ctx.eps0 * ctx.plate_height

            deformation = f"""{mesh_data["corner_4"]}, 2, 2, {u4y}
{mesh_data["corner_2"]}, 2, 2, {u2y}"""

        elif load_case == "XY":
            # Simple shear is prescribed as uy = gamma_xy * x.
            u2x, u2y = 0.0, ctx.gamma0 * ctx.plate_width
            u4y = 0.0

            deformation = f"""{mesh_data["corner_4"]}, 2, 2, {u4y}
{mesh_data["corner_2"]}, 1, 1, {u2x}
{mesh_data["corner_2"]}, 2, 2, {u2y}"""

        else:
            raise ValueError("load_case must be EX, EY or XY")

        calculix_input = f"""{mesh_data["mesh_content"]}

*NSET, NSET=CORNERS
{mesh_data["corner_4"]}, {mesh_data["corner_2"]}

{mesh_data["equation_block"]}

*MATERIAL, NAME={ctx.material_name}
*ELASTIC
{ctx.material_E}, {ctx.material_nu}

*SOLID SECTION, ELSET=Plate, MATERIAL={ctx.material_name}
{ctx.thickness}

*STEP, NLGEOM=NO
*STATIC

** REFERENCE-CORNER BOUNDARY CONDITIONS
*BOUNDARY
{mesh_data["corner_1"]}, 1, 3, 0.0
{mesh_data["corner_4"]}, 1, 1, {u4x}
{deformation}

** OUTPUT TO .DAT
*EL PRINT, ELSET=Plate, GLOBAL=YES
S, EVOL

*NODE PRINT, NSET=CORNERS, FREQUENCY=1
U

*END STEP
"""

    ccx_input.parent.mkdir(parents=True, exist_ok=True)
    ccx_input.write_text(calculix_input, encoding="utf-8")

    logger.info(f"Input file created: {ccx_input}")

    return ccx_input


# =============================================================================
# CalculiX execution
# =============================================================================


def run_calculix_simulation(case: SimpleNamespace) -> bool:
    """Run one CalculiX job and move generated outputs to the result directory."""

    ctx = case.ctx
    logger = case.logger

    try:
        logger.info(f"Starting CalculiX simulation for {case.load_case}...")

        result = subprocess.run(
            [str(ctx.calculix_path), str(case.inp_base)],
            capture_output=True,
            text=True,
            timeout=ctx.timeout_seconds,
            cwd=str(case.inp_path.parent),
        )

        if result.returncode != 0:
            logger.error("CalculiX returned a non-zero exit code.")
            logger.error(f"STDOUT:\n{result.stdout}")
            logger.error(f"STDERR:\n{result.stderr}")
            return False

        ctx.output_dir.mkdir(parents=True, exist_ok=True)

        for ext in [".dat", ".frd", ".log", ".sta"]:
            src_file = case.inp_base.with_suffix(ext)
            dst_file = ctx.output_dir / f"{case.name}{ext}"

            if src_file.exists():
                shutil.move(str(src_file), str(dst_file))
                logger.info(f"Moved {ext} to {ctx.output_dir}")
            else:
                if ext == ".log":
                    logger.info("No .log file generated (no warnings).")
                else:
                    logger.warning(f"Expected output not found: {src_file.name}")

        return True

    except subprocess.TimeoutExpired:
        logger.error("Timeout - simulation took too long.")
        return False

    except FileNotFoundError:
        logger.error(f"CalculiX not found at: {ctx.calculix_path}")
        return False

    except Exception as e:
        logger.error(f"Unexpected error while running CalculiX: {e}")
        return False


# =============================================================================
# Parallel worker
# =============================================================================


def run_single_geometry(
    i: int,
    params: Dict[str, Dict[str, float]],
    ctx: SimpleNamespace,
) -> Dict[str, Any]:
    """Run EX, EY, and XY analyses for one sampled geometry.

    A single accepted mesh is reused across all load cases. The returned status
    record is consumed by the main process to build successful and rejected
    datasets without raising worker exceptions across process boundaries.
    """

    geometry_case = SimpleNamespace()
    geometry_case.i = i
    geometry_case.name = f"sim_{i:04d}"
    geometry_case.ctx = ctx

    logger = setup_simulation_logger(geometry_case)

    try:
        logger.info("=" * 50)
        logger.info(f"STARTING GEOMETRY {i:04d}")
        logger.info("=" * 50)

        # Build and validate the mesh once for this geometry.
        mesh_data = create_base_model(
            params=params,
            sim_number=i,
            logger=logger,
            ctx=ctx,
        )

        params["avg_quality"] = mesh_data["avg_quality"]
        params["poor_elements"] = mesh_data["poor_quality_elements"]

        # Reject the geometry before solving if too many elements are poor.
        if mesh_data["poor_quality_elements"] > ctx.max_poor_elements:
            reason = (
                f"poor_elements ({mesh_data['poor_quality_elements']}) "
                f"> {ctx.max_poor_elements}"
            )

            logger.warning(f"SKIPPED GEOMETRY - {reason}")

            return {
                "index": i,
                "status": "skipped",
                "reason": reason,
                "params": params,
            }

        sigma_by_case = {}
        disp_EX = None

        for load_case in ["EX", "EY", "XY"]:
            case = make_load_case(
                i=i,
                params=params,
                logger=logger,
                ctx=ctx,
                load_case=load_case,
            )

            logger.info("")
            logger.info("-" * 50)
            logger.info(f"LOAD CASE: {load_case}")
            logger.info("-" * 50)

            generate_inp(
                mesh_data=mesh_data,
                case=case,
            )

            success = run_calculix_simulation(case)

            if not success:
                return {
                    "index": i,
                    "status": "failed",
                    "reason": f"FEA solver error in load case {load_case}",
                    "params": params,
                }

            if not case.dat_path.exists():
                return {
                    "index": i,
                    "status": "failed",
                    "reason": f"missing dat output for load case {load_case}",
                    "params": params,
                }

            try:
                sigma_by_case[load_case] = calculate_sigma_hom(str(case.dat_path))

                if ctx.calculation_mode == "direct_contraction" and load_case == "EX":
                    disp_EX = read_node_displacement(
                        dat_path=str(case.dat_path),
                        node_id=mesh_data["corner_4"],
                    )

                # Keep failed-case files for diagnosis; remove successful
                # intermediate inputs and large result files after parsing.
                for path in [case.inp_path, case.dat_path, case.frd_path, case.d_path]:
                    try:
                        path.unlink()
                        logger.info(f"Deleted file: {path}")
                    except FileNotFoundError:
                        pass

            except Exception as e:
                logger.error(f"Postprocessing failed for {load_case}: {e}")

                return {
                    "index": i,
                    "status": "failed",
                    "reason": f"postprocessing failed for {load_case}: {e}",
                    "params": params,
                }

            logger.info(f"{load_case}: sigma_hom = {sigma_by_case[load_case]}")

        # Convert the three homogenized responses to effective constants.
        try:
            if ctx.calculation_mode == "from_c":
                effective = compute_effective_constants_from_C(
                    sigma_EX=sigma_by_case["EX"],
                    sigma_EY=sigma_by_case["EY"],
                    sigma_XY=sigma_by_case["XY"],
                    eps0=ctx.eps0,
                    gamma0=ctx.gamma0,
                )

            elif ctx.calculation_mode == "direct_contraction":
                if disp_EX is None:
                    raise RuntimeError("Missing displacement data for EX load case.")

                effective = compute_effective_constants_with_contraction(
                    sigma_EX=sigma_by_case["EX"],
                    sigma_EY=sigma_by_case["EY"],
                    sigma_XY=sigma_by_case["XY"],
                    disp_EX=disp_EX,
                    plate_height=ctx.plate_height,
                    eps0=ctx.eps0,
                    gamma0=ctx.gamma0,
                )

        except Exception as e:
            logger.error(f"Failed to compute effective constants: {e}")

            return {
                "index": i,
                "status": "failed",
                "reason": f"failed to compute effective constants: {e}",
                "params": params,
            }

        params.update(effective)

        logger.info(
            "Geometry completed successfully. "
            f"E_x = {params['E_x']:.2f} MPa, "
            f"E_y = {params['E_y']:.2f} MPa, "
            f"G_xy = {params['G_xy']:.2f} MPa, "
            f"NU_xy = {params['NU_xy']:.4f}"
        )

        return {
            "index": i,
            "status": "success",
            "params": params,
        }

    except Exception as e:
        logger.error(f"Unexpected error in geometry {i:04d}: {e}")

        return {
            "index": i,
            "status": "error",
            "reason": str(e),
            "params": params,
        }

    finally:
        close_logger(logger)


# =============================================================================
# Main
# =============================================================================


def configure_multiprocessing() -> None:
    """Use the Windows-safe process start method required by Gmsh workers."""

    if sys.platform.startswith("win"):
        multiprocessing.set_start_method("spawn", force=True)


def main(config_path: str = "config.yaml") -> None:
    """Run the configured simulation batch and export aggregate results."""

    configure_multiprocessing()

    config_path_p = Path(config_path)
    cfg = load_config(config_path_p)
    ctx = build_ctx(cfg, config_path_p)

    dataset_mode = ctx.calculation_mode

    make_dirs(ctx=ctx)

    num_sims = ctx.num_sims
    n_jobs = ctx.n_jobs
    seed = ctx.seed

    plate_width = ctx.plate_width
    plate_height = ctx.plate_height
    material_e = ctx.material_E
    material_nu = ctx.material_nu

    main_logger = setup_main_logger(ctx)

    main_logger.info("=" * 70)
    main_logger.info(
        "PARAMETRIC FEA SIMULATION - TENSILE PLATE WITH ELLIPTICAL HOLES"
    )
    main_logger.info("=" * 70)
    main_logger.info("")
    main_logger.info("Configuration:")
    main_logger.info(f" Working dir: {ctx.working_dir}")
    main_logger.info(f" CalculiX: {ctx.calculix_path}")
    main_logger.info(f" Total geometries: {num_sims}")
    main_logger.info(f" Total load-case analyses: {num_sims * 3}")
    main_logger.info(f" Parallel jobs: {n_jobs}")
    main_logger.info(f" Plate dimensions: {plate_width} x {plate_height} mm")
    main_logger.info(f" eps0: {ctx.eps0}")
    main_logger.info(f" gamma0: {ctx.gamma0}")
    main_logger.info(f" Material: E={material_e} MPa, nu={material_nu}")
    main_logger.info("")

    all_params = generate_lhs_params_independent(
        ctx=ctx,
        seed=seed,
    )

    start_time = time.time()

    with tqdm_joblib(tqdm(total=ctx.num_sims, desc="Geometries", unit="geom")):
        results = Parallel(n_jobs=ctx.n_jobs, verbose=0)(
            delayed(run_single_geometry)(i, all_params[i], ctx)
            for i in range(ctx.num_sims)
        )

    elapsed = time.time() - start_time
    main_logger.info(f"Total execution time: {elapsed:.1f} seconds")

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

    rejected_results = [
        result
        for result in results
        if result["status"] in {"skipped", "failed", "error"}
    ]
    successful_results = [
        result["params"]
        for result in results
        if result["status"] == "success" and result.get("params") is not None
    ]

    if successful_results:
        main_logger.info("")
        main_logger.info("Generating CSV and PNG files with parameters...")
        params_csv_histograms(
            seed=seed,
            all_params=successful_results,
            output_dir=str(ctx.data_dir),
            dataset_mode=dataset_mode,
        )
        main_logger.info("Done!")

    if rejected_results:
        main_logger.info("")
        main_logger.info(f"Found {len(rejected_results)} problematic simulations.")
        save_rejected_csv(rejected_results, str(ctx.data_dir))

    main_logger.info("")
    main_logger.info(f"All logs saved to: {ctx.logs_dir}")
    main_logger.info("Individual simulation logs: sim_XXXX.log")
    main_logger.info("Main log: main.log")

    close_logger(main_logger)


if __name__ == "__main__":
    # Usage: python simulation.py [path/to/config.yaml]
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(cfg_path)
