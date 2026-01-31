



"""
PARAMETRIC TENSION PLATE FEA
Automated Batch Simulation for Material Characterization

This script:
1. Loads simulation configuration from YAML file
2. Generates geometric samples using Latin Hypercube Sampling (LHS)
3. Creates plate geometries with two elliptical holes per sample point
4. Generates adaptive FE meshes and performs quality control
5. Runs parallel structural analyses using CalculiX
6. Computes effective Young's modulus via homogenization for each case
7. Exports aggregated results to CSV with full parameter mappings

Author: Andrea Vinarš

"""

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)  # Required on Windows, 
                                                       # fork() fails with Gmsh

import gmsh
import os
import sys
import logging
import numpy as np
import subprocess
import shutil
import yaml
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
from joblib import Parallel, delayed
import time
import tqdm
from scipy.stats import qmc
from typing import Dict, List, Tuple, Optional, Any

from data_processing import calculate_youngs_modulus, params_csv_histograms, save_rejected_csv


# =============================================================================
# Logging Configuration
# =============================================================================

def setup_simulation_logger(sim_number: int) -> logging.Logger:
    """
    Configure a logger for a specific simulation that writes to a dedicated file.
    
    Each simulation gets its own log file to avoid conflicts in parallel execution.
    
    Args:
        sim_number: Simulation identifier for naming the log file
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"Simulation_{sim_number}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # File handler for this simulation
    log_file = f"logs/sim_{sim_number:04d}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Format: [TIMESTAMP] [LEVEL] Message
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def setup_main_logger() -> logging.Logger:
    """
    Configure the main logger for the orchestration process.
    
    Returns:
        Configured logger that writes to both console and main log file
    """
    logger = logging.getLogger("Main")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    os.makedirs("logs", exist_ok=True)
    
    # File handler for main log
    file_handler = logging.FileHandler("logs/main.log", mode='w')
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Also print to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Load Configuration from YAML
# =============================================================================

def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load simulation configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        Configuration dictionary with all simulation parameters
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Load configuration
config = load_config()

# Set working directory
os.chdir(config['paths']['working_dir'])

# Extract paths
CALCULIX_PATH = config['paths']['calculix']


# Extract plate geometry
PLATE_WIDTH = config['plate']['width']
PLATE_HEIGHT = config['plate']['height']
THICKNESS = config['plate']['thickness']

# Extract loading conditions
ELONGATION = config['loading']['elongation']

# Extract material properties
MATERIAL_E = config['material']['youngs_modulus']
MATERIAL_NU = config['material']['poisson_ratio']

# Extract simulation settings
NUM_SIMULATIONS = config['simulation']['num_simulations']
N_JOBS = config['simulation']['n_jobs']
SEED = config['simulation']['seed']
TIMEOUT_SECONDS = config['simulation']['timeout_seconds']

# Extract LHS bounds
LHS_LOWER_BOUNDS = config['lhs_bounds']['lower']
LHS_UPPER_BOUNDS = config['lhs_bounds']['upper']

# Extract mesh quality thresholds
MAX_POOR_ELEMENTS = config['mesh_quality']['max_poor_elements']

# Extract output directories
INPUT_DIR = config['output']['input_files']
OUTPUT_DIR = config['output']['output_files']
DATA_DIR = config['output']['data_dir']


# =============================================================================
# Parameter Generation
# =============================================================================

def generate_lhs_params_independent(
    n_samples: int, 
    seed: int = 1) -> List[Dict[str, Dict[str, float]]]:
    
    """
    Generate parameter sets using Latin Hypercube Sampling (LHS).
    
    Creates randomized but well-distributed samples across the parameter space
    for two elliptical holes. Each hole is defined by:
      - Center position (x, y)
      - Semi-axes (rx, ry) where rx >= ry
      - Rotation angle (degrees)
    
    Args:
        n_samples: Number of parameter sets to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of parameter dictionaries, each containing 'hole1' and 'hole2' configs
    """
    
    sampler = qmc.LatinHypercube(d=10, seed=seed)
    sample = sampler.random(n_samples)
    
    # Scale samples to parameter bounds
    params_scaled = qmc.scale(sample, LHS_LOWER_BOUNDS, LHS_UPPER_BOUNDS)
    
    params_list: List[Dict[str, Dict[str, float]]] = []
    
    for row in params_scaled:
        
        # Sorting enforces rx >= ry while keeping the correlation between variables low.
        # This method lowered correlation by 0.14 (tested myself) in comparison with method which I uses before,
        # to set ry lower boundary to 0.3 and upper boundary to generated rx
        # Note: This distorts the marginal LHS distributions slightly, but eliminates 
        # the need to reject simulations where ry > rx.
        
        rx1, ry1 = sorted([row[2], row[3]], reverse=True)
        
        rx2, ry2 = sorted([row[7], row[8]], reverse=True)
        
        hole1 = {
            'x': round(row[0], 2),
            'y': round(row[1], 2),
            'rx': round(rx1, 2),
            'ry': round(ry1, 2),
            'angle': round(row[4], 1)
        }
        hole2 = {
            'x': round(row[5], 2),
            'y': round(row[6], 2),
            'rx': round(rx2, 2),
            'ry': round(ry2, 2),
            'angle': round(row[9], 1)
        }
        params_list.append({'hole1': hole1, 'hole2': hole2})
    
    return params_list


# =============================================================================
# Mesh Generation
# =============================================================================

def create_mesh(
    params: Dict[str, Dict[str, float]], 
    sim_number: int) -> Dict[str, int]:
    
    """
    Create a 2D mesh for the plate with elliptical holes using Gmsh.
    
    Uses threshold-based mesh sizing to refine elements near hole boundaries.
    
    Args:
        params: Dictionary with 'hole1' and 'hole2' geometric parameters
        sim_number: Simulation identifier for naming
    
    Returns:
        Dictionary containing Gmsh entity tags for the plate and ellipses
    """
    
    # Extract hole parameters
    x1 = params['hole1']['x']
    y1 = params['hole1']['y']
    rx1 = params['hole1']['rx']
    ry1 = params['hole1']['ry']
    angle1 = params['hole1']['angle']

    x2 = params['hole2']['x']
    y2 = params['hole2']['y']
    rx2 = params['hole2']['rx']
    ry2 = params['hole2']['ry']
    angle2 = params['hole2']['angle']
    
    # Initialize Gmsh model
    gmsh.model.add(f"tensile_plate_{sim_number}")
    
    # Create rectangular plate
    plate = gmsh.model.occ.addRectangle(0, 0, 0, PLATE_WIDTH, PLATE_HEIGHT)

    # First elliptical hole
    e1_curve = gmsh.model.occ.addEllipse(x1, y1, 0, rx1, ry1)
    gmsh.model.occ.rotate([(1, e1_curve)], x1, y1, 0, 0, 0, 1, angle1 * np.pi / 180)
    e1_loop = gmsh.model.occ.addCurveLoop([e1_curve])
    hole1 = gmsh.model.occ.addPlaneSurface([e1_loop])

    # Second elliptical hole
    e2_curve = gmsh.model.occ.addEllipse(x2, y2, 0, rx2, ry2)
    gmsh.model.occ.rotate([(1, e2_curve)], x2, y2, 0, 0, 0, 1, angle2 * np.pi / 180)
    e2_loop = gmsh.model.occ.addCurveLoop([e2_curve])
    hole2 = gmsh.model.occ.addPlaneSurface([e2_loop])

    gmsh.model.occ.synchronize()

    # Boolean cut: plate - holes
    cut_result = gmsh.model.occ.cut([(2, plate)], [(2, hole1), (2, hole2)])
    new_plate_tag = cut_result[0][0][1]

    gmsh.model.occ.synchronize()

    # -------------------------------------------------------------------------
    # Define mesh size field with threshold-based refinement
    # -------------------------------------------------------------------------
    def create_distance_field(curve_tag: int, radius: float) -> int:
        
        """
        Create a distance field from a curve for mesh sizing.
        
        Args:
            curve_tag: Gmsh curve entity tag
            radius: Characteristic radius for sampling density
        
        Returns:
            Field ID for the created distance field
        """
        
    # Field-based meshing is required when holes intersect. Without it, Gmsh 
    # interprets intersections as separate geometric entities, creating 
    # duplicate boundary curves that prevent direct ellipse-based refinement.
        
        field = gmsh.model.mesh.field.add("Distance")
        sampling = 100 if radius < 0.7 else 200

        gmsh.model.mesh.field.setNumber(field, "Sampling", sampling)
        gmsh.model.mesh.field.setNumbers(field, "EdgesList", [curve_tag])
        return field

    # Create distance fields for both ellipses
    field1 = create_distance_field(e1_curve, rx1)
    field2 = create_distance_field(e2_curve, rx2)

    # Combine distance fields using minimum
    field_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(field_min, "FieldsList", [field1, field2])

    # Apply threshold: fine mesh near holes, coarse mesh far away
    field_threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(field_threshold, "InField", field_min)
    gmsh.model.mesh.field.setNumber(field_threshold, "SizeMin", 0.1)
    gmsh.model.mesh.field.setNumber(field_threshold, "SizeMax", 0.8)
    gmsh.model.mesh.field.setNumber(field_threshold, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(field_threshold, "DistMax", 5.0)

    gmsh.model.mesh.field.setAsBackgroundMesh(field_threshold)

    # Generate and optimize mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)           # Second-order elements
    gmsh.model.mesh.optimize("Netgen")    # Netgen optimization
    gmsh.model.mesh.optimize("Gmsh")      # Gmsh optimization

    return {
        'new_plate_tag': new_plate_tag,
        'e1_c': e1_curve,
        'e2_c': e2_curve
    }


# =============================================================================
# CalculiX Input File Generation
# =============================================================================

def generate_inp(
    params: Dict[str, Dict[str, float]], 
    ccx_input: str, 
    sim_number: int,
    logger: logging.Logger) -> Tuple[str, float, int]:
    
    """
    Generate CalculiX input file (.inp) from geometric parameters.
    
    Creates a complete FEA model including mesh, material definition,
    boundary conditions, and output requests.
    
    Args:
        params: Dictionary with hole geometric parameters
        ccx_input: Output path for the .inp file
        sim_number: Simulation identifier
        logger: Logger instance for this simulation
    
    Returns:
        Tuple of (input_file_path, average_element_quality, poor_quality_count)
    """
    
    gmsh.initialize()
    
    # Extract hole parameters for logging
    x1 = params['hole1']['x']
    y1 = params['hole1']['y']
    rx1 = params['hole1']['rx']
    ry1 = params['hole1']['ry']
    angle1 = params['hole1']['angle']
    x2 = params['hole2']['x']
    y2 = params['hole2']['y']
    rx2 = params['hole2']['rx']
    ry2 = params['hole2']['ry']
    angle2 = params['hole2']['angle']
    
    # Log geometry parameters
    logger.info("")
    logger.info("GEOMETRY PARAMETERS:")
    logger.info(f"  Plate: {PLATE_WIDTH} x {PLATE_HEIGHT} mm, thickness = {THICKNESS} mm")
    logger.info(f"  Hole 1: center=({x1}, {y1}), rx={rx1}, ry={ry1}, angle={angle1}°")
    logger.info(f"  Hole 2: center=({x2}, {y2}), rx={rx2}, ry={ry2}, angle={angle2}°")
    logger.info("")
    
    try:
        # Suppress Gmsh console output for performance
        # This only affects Gmsh internal messages, not Python logging
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Smoothing", 20)
        gmsh.option.setNumber("Mesh.MinimumCurvePoints", 5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)

        # Generate mesh
        geom = create_mesh(params, sim_number)
        new_plate_tag = geom['new_plate_tag']

        # -------------------------------------------------------------------------
        # Assess mesh quality
        # -------------------------------------------------------------------------
        element_data = gmsh.model.mesh.getElements(dim=2)
        element_tags = element_data[1][0]
        element_qualities = gmsh.model.mesh.getElementQualities(elementTags=element_tags)

        logger.info("")
        logger.info("ELEMENT QUALITY:")
        for lower in np.arange(0, 1, 0.1):
            upper = lower + 0.1
            count = int(np.sum((element_qualities >= lower) & (element_qualities < upper)))
            pct = 100 * count / len(element_qualities)
            logger.info(f" {lower:.2f}-{upper:.2f}: {count:4d} ({pct:5.1f}%)")

        poor_quality_elements = int(np.sum(element_qualities < 0.6))
        avg_quality = float(np.mean(element_qualities))
        logger.info(f" Average quality: {avg_quality:.3f}")
        logger.info("")

        # -------------------------------------------------------------------------
        # Identify boundary nodes for constraints
        # -------------------------------------------------------------------------
        
        # TOLERANCE: Mesh nodes are exactly at x=0 and x=width, but floating point
        # representation can be x=1e-16 or x=width+1e-16. 
        # 1e-4 >> typical node coordinate error (~1e-12) but << min element size (0.1)
        
        TOLERANCE = 1e-4
        left_nodes: List[int] = []
        right_nodes: List[int] = []
        
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = node_coords.reshape(-1, 3)
        
        for i, tag in enumerate(node_tags):
            x = node_coords[i][0]
            if abs(x) < TOLERANCE:
                left_nodes.append(int(tag))
            if abs(x - PLATE_WIDTH) < TOLERANCE:
                right_nodes.append(int(tag))

        logger.info(f"FIXED NODES: {len(left_nodes)} | LOAD NODES: {len(right_nodes)}")

        # Create physical group for the plate
        gmsh.model.addPhysicalGroup(2, [new_plate_tag], tag=3, name="Plate")

        # Write temporary mesh file
        temp_mesh_file = f"_temp_mesh_{sim_number}.inp"
        gmsh.write(temp_mesh_file)

        # Read mesh content
        with open(temp_mesh_file, "r") as f:
            mesh_content = f.read()

        # Modify node definition to include all nodes in a single node set.
        # This allows accessing all nodal results in CalculiX output files.
        
        mesh_content = mesh_content.replace("*NODE", "*NODE, NSET=NALL")
        
        # -------------------------------------------------------------------------
        # Build node sets for boundary conditions
        # -------------------------------------------------------------------------
        # Fixed boundary (left edge): constrain all DOFs
        nset_fixed = "*NSET, NSET=Fixed\n"
        for i, node in enumerate(left_nodes):
            if i > 0 and i % 16 == 0:
                nset_fixed += "\n"
            nset_fixed += f"{node}, "
        nset_fixed = nset_fixed.rstrip(", ") + "\n"

        # Load boundary (right edge): apply elongation in x-direction
        nset_load = "*NSET, NSET=Load\n"
        for i, node in enumerate(right_nodes):
            if i > 0 and i % 16 == 0:
                nset_load += "\n"
            nset_load += f"{node}, "
        nset_load = nset_load.rstrip(", ") + "\n"

        # -------------------------------------------------------------------------
        # Assemble complete CalculiX input file
        # -------------------------------------------------------------------------
        calculix_input = f"""{mesh_content}
        
{nset_fixed}

{nset_load}

*MATERIAL, NAME={config['material']['name']}
*ELASTIC
{MATERIAL_E}, {MATERIAL_NU}

*SOLID SECTION, ELSET=Plate, MATERIAL={config['material']['name']}
{THICKNESS}

*STEP, NLGEOM=NO
*STATIC

**SUPPORT CONSTRAINTS
*BOUNDARY
Fixed, 1, 3, 0.0

**LOADING
*BOUNDARY
Load, 1, 1, {ELONGATION}

**OUTPUT
*EL PRINT, ELSET=Plate
S

*END STEP
"""

        with open(ccx_input, "w") as f:
            f.write(calculix_input)

        logger.info(f"Input file created: {ccx_input}")
        
        return ccx_input, avg_quality, poor_quality_elements

    finally:
        gmsh.finalize()  
        
        # Clean up temporary mesh file
        temp_file = f"_temp_mesh_{sim_number}.inp"
        if os.path.exists(temp_file):
            os.remove(temp_file)


# =============================================================================
# FEA Solver Execution
# =============================================================================

def run_calculix_simulation(
    inp_filename: str, 
    sim_number: int,
    logger: logging.Logger ) -> bool:
    
    """
    Execute CalculiX FEA simulation.
    
    Runs the static analysis and moves output files to the output directory.
    
    Args:
        inp_filename: Path to the CalculiX input file (.inp)
        sim_number: Simulation identifier
        logger: Logger instance for this simulation
    
    Returns:
        True if simulation completed successfully, False otherwise
    """

    try:
        logger.info("Starting CalculiX simulation...")

        # Extract base filename without extension
        inp_base = inp_filename.replace(".inp", "")

        result = subprocess.run(
            [CALCULIX_PATH, inp_base],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS
        )

        if result.returncode == 0:
            # Move output files to output directory
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_extensions = ['.dat', '.frd', '.log', '.sta']
            
            for ext in output_extensions:
                src_file = f"{INPUT_DIR}/sim_{sim_number:04d}{ext}"
                dst_file = f"{OUTPUT_DIR}/sim_{sim_number:04d}{ext}"

                if os.path.exists(src_file):
                    shutil.move(src_file, dst_file)
                    logger.info(f"Moved {ext} file to {OUTPUT_DIR}/")
                else:
                    # Note: .log file is only created by CalculiX if there are warnings/errors
                    if ext == '.log':
                        logger.info(f"No {ext} file generated (simulation completed without warnings)")
                    else:
                        logger.warning(f"{ext} file not found")

            return True
        else:
            logger.error("Analysis error:")
            logger.error(f"STDERR: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Timeout - simulation took too long")
        return False
    except FileNotFoundError:
        logger.error(f"CalculiX not found at: {CALCULIX_PATH}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


# =============================================================================
# Parallel Simulation Runner
# =============================================================================

def run_simulations_parallel(
    i: int, 
    params: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    
    """
    Execute a single simulation in the parallel batch.
    
    Orchestrates the complete workflow: mesh generation, input file creation,
    FEA execution, and post-processing.
    
    Args:
        i: Simulation index
        params: Geometric parameters for this simulation
    
    Returns:
        Result dictionary with status and computed data
    """
    
    # Setup dedicated logger for this simulation
    logger = setup_simulation_logger(i)
    
    try:
        os.makedirs(INPUT_DIR, exist_ok=True)
        ccx_input = f"{INPUT_DIR}/sim_{i:04d}.inp"

        logger.info("=" * 50)
        logger.info(f"STARTING SIMULATION {i}")
        logger.info("=" * 50)

        # Generate input file and assess mesh quality
        ccx_input, avg_quality, poor_quality_elements = generate_inp(
            params, ccx_input, sim_number=i, logger=logger
        )
        params['avg_quality'] = avg_quality
        
        # Run FEA simulation
        success = run_calculix_simulation(ccx_input, sim_number=i, logger=logger)

        # Check for mesh quality issues
        skip_reason: Optional[str] = None
        if poor_quality_elements > MAX_POOR_ELEMENTS:
            skip_reason = f"poor_elements ({poor_quality_elements}) > {MAX_POOR_ELEMENTS}"

        if skip_reason:
            logger.warning(f"SKIPPED - {skip_reason}")
            return {
                'index': i,
                'status': 'skipped',
                'reason': skip_reason,
                'params': params
            }

        if success:
            # Calculate effective Young's modulus from results
            E_eff = calculate_youngs_modulus(
                width=PLATE_WIDTH, 
                elongation=ELONGATION, 
                inp_path=ccx_input, 
                dat_path=f"{OUTPUT_DIR}/sim_{i:04d}.dat"
            )
            params['E_eff'] = E_eff
            
            logger.info(f"Simulation completed successfully. E_eff = {E_eff:.2f} MPa")
            
            return {
                'index': i,
                'status': 'success',
                'params': params
            }
        else:
            logger.error("Simulation failed")
            return {
                'index': i,
                'status': 'failed',
                'reason': 'FEA solver error', 
                'params': params  
            }
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            'index': i, 
            'status': 'error', 
            'params': params, 
            'error': str(e)
        }


# =============================================================================
# Main Program
# =============================================================================

if __name__ == "__main__":


    main_logger = setup_main_logger()
    
    main_logger.info("=" * 70)
    main_logger.info("PARAMETRIC FEA SIMULATION - TENSILE PLATE WITH ELLIPTICAL HOLES")
    main_logger.info("=" * 70)

    main_logger.info("")
    main_logger.info("Configuration:")
    main_logger.info(f" Total simulations: {NUM_SIMULATIONS}")
    main_logger.info(f" Parallel jobs (CPU cores): {N_JOBS}")
    main_logger.info(f" Plate dimensions: {PLATE_WIDTH} x {PLATE_HEIGHT} mm")
    main_logger.info(f" Elongation: {ELONGATION} mm")
    main_logger.info(f" Material: Steel (E={MATERIAL_E} MPa, nu={MATERIAL_NU})")
    main_logger.info("")
    
    # Generate parameter sets using LHS
    all_params = generate_lhs_params_independent(n_samples=NUM_SIMULATIONS, seed=SEED)
    
    start_time = time.time()

    # Run simulations in parallel
    results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(run_simulations_parallel)(i, all_params[i])
        for i in tqdm.tqdm(range(NUM_SIMULATIONS), desc="Simulations", unit="sim")
    )

    elapsed_time = time.time() - start_time

    main_logger.info(f"Total execution time: {elapsed_time:.1f} seconds")

    # Count results by status
    successful_sims = sum(1 for r in results if r['status'] == 'success')
    failed_sims = sum(1 for r in results if r['status'] == 'failed')
    error_sims = sum(1 for r in results if r['status'] == 'error')
    skipped_sims = sum(1 for r in results if r['status'] == 'skipped')

    main_logger.info("")
    main_logger.info("=" * 70)
    main_logger.info("PARALLEL SIMULATION RESULTS:")
    main_logger.info("=" * 70)
    main_logger.info(f"Total simulations: {NUM_SIMULATIONS}")
    main_logger.info(f"Successful: {successful_sims}")
    main_logger.info(f"Skipped: {skipped_sims}")
    main_logger.info(f"Failed: {failed_sims}")
    main_logger.info(f"Errors: {error_sims}")
    
    # Collect rejected results for analysis
    rejected_results = [
        r for r in results 
        if r['status'] in ['skipped', 'failed', 'error']
    ]

    # Collect successful results for dataset generation
    successful_results = [
        r['params']
        for r in results
        if r['status'] == 'success' and r['params'] is not None
    ]

    # Generate CSV and histograms for successful simulations
    if successful_results:
        main_logger.info("")
        main_logger.info("Generating CSV and PNG files with parameters...")
        params_csv_histograms(successful_results, DATA_DIR)
        main_logger.info("Done!")
        
    # Save rejected simulations for debugging (only 16 out of 5000 simulations were rejected (seed=1))
    if rejected_results:
        main_logger.info("")
        main_logger.info(f"Found {len(rejected_results)} problematic simulations:")
        save_rejected_csv(rejected_results, DATA_DIR)
    
    main_logger.info("")
    main_logger.info("All logs saved to logs/ directory")
    main_logger.info("Individual simulation logs: logs/sim_XXXX.log")

    main_logger.info("Main log: logs/main.log")
