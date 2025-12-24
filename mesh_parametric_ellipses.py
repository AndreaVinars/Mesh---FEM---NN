"""
PARAMETRIC FEA SIMULATION - TENSION PLATE WITH ELLIPSES
Sequential Version (Non-Parallelized)

This script performs a parametric Finite Element Analysis (FEA) of a tension plate
with two randomly positioned and oriented elliptical holes. It uses Gmsh for mesh
generation and CalculiX for structural analysis.

Key features:
    - Random ellipse parameter generation
    - Adaptive mesh refinement around holes
    - Automated CalculiX input file generation
    - Von Mises stress analysis and visualization
    - Post-processing with CSV export and histogram generation
    - Organized output: input_files, output_files, and data directories

Dependencies:
    - gmsh (mesh generation)
    - numpy (numerical computations)
    - pandas (data manipulation)
    - matplotlib (visualization)
    - CalculiX (structural solver)

Author: Andrea Vinarš
"""

import gmsh
import os
import sys
import shutil
import random
import numpy as np 
import subprocess
import matplotlib.pyplot as plt
import pandas as pd

# ========== CONFIGURATION ==========
# Plate dimensions [mm]
PLATE_WIDTH = 15
PLATE_HEIGHT = 15

# ========== PATHS - CONFIGURE BEFORE RUNNING ==========

# PROJECT DIRECTORY
# Where all simulation files will be saved (input_files/, output_files/, data/)
# Examples:
#   Windows: r'C:/Users/YourName/Desktop/FEA_Project'
#   Linux:   r'/home/username/projects/fea_analysis'
#   Relative: r'.'  (current directory)
PROJECT_DIR = r"-"

# CALCULIX SOLVER - ccx_static.exe EXECUTABLE
# Find it in your CalculiX installation folder
# Examples:
#   r'C:/Program Files/calculix/ccx_static.exe'
#   r'D:/Tools/CalculiX_2.23/ccx_static.exe'
# REQUIRED: Script will not work without this!
CALCULIX_PATH = r"-/ccx_static.exe"

# CALCULIX GUI VISUALIZER - cgx_STATIC.exe EXECUTABLE (OPTIONAL)
# Find it in the same folder as ccx_static.exe
# Examples:
#   r'C:/Program Files/calculix/cgx_STATIC.exe'
#   r'D:/Tools/CalculiX_2.23/cgx_STATIC.exe'
# NOTE: Analysis will work without this, but visualization will be skipped
CGX_PATH = r"-/cgx_STATIC.exe"

os.chdir(PROJECT_DIR)
w, h = PLATE_WIDTH, PLATE_HEIGHT

"""========================================================================="""

def create_mesh(params, sim_number, visualize=False):
    """
    Create a 2D finite element mesh with two elliptical holes.
    
    This function uses Gmsh to:
    1. Create a rectangular plate
    2. Add two elliptical holes with specified positions and orientations
    3. Generate an adaptive mesh with refined elements near the holes
    4. Apply mesh smoothing and optimization
    
    Parameters
    ----------
    params : dict
        Dictionary containing hole parameters:
        - 'hole1': dict with 'x', 'y', 'rx', 'ry', 'angle'
        - 'hole2': dict with 'x', 'y', 'rx', 'ry', 'angle'
    sim_number : int
        Simulation identifier for logging
    visualize : bool, optional
        If True, display the geometry in Gmsh GUI (default: False)
    
    Returns
    -------
    dict
        Dictionary with mesh component IDs:
        - 'new_plate_tag': ID of the plate surface after cutting holes
        - 'e1_c': ID of first ellipse curve
        - 'e2_c': ID of second ellipse curve
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
    
    # Create base rectangular plate
    plate = gmsh.model.occ.addRectangle(0, 0, 0, w, h)
    
    # Create first elliptical hole
    e1_c = gmsh.model.occ.addEllipse(x1, y1, 0, rx1, ry1)
    gmsh.model.occ.rotate([(1, e1_c)], x1, y1, 0, 0, 0, 1, angle1*np.pi/180)
    e1_cl = gmsh.model.occ.addCurveLoop([e1_c])
    hole1 = gmsh.model.occ.addPlaneSurface([e1_cl])
    
    # Create second elliptical hole
    e2_c = gmsh.model.occ.addEllipse(x2, y2, 0, rx2, ry2)
    gmsh.model.occ.rotate([(1, e2_c)], x2, y2, 0, 0, 0, 1, angle2*np.pi/180)
    e2_cl = gmsh.model.occ.addCurveLoop([e2_c])
    hole2 = gmsh.model.occ.addPlaneSurface([e2_cl])
    
    gmsh.model.occ.synchronize()
    
    # Boolean operation: cut holes from plate
    cut_result = gmsh.model.occ.cut([(2, plate)], [(2, hole1), (2, hole2)])
    new_plate_tag = cut_result[0][0][1]    
    
    gmsh.model.occ.synchronize()
    
    # Define distance fields for adaptive mesh refinement
    def create_field(curve_tag, radius):
        """Create a distance-based mesh field around a curve."""
        field = gmsh.model.mesh.field.add("Distance")
        # Use higher sampling density for small holes
        sampling = 100 if radius < 0.7 else 200
        
        gmsh.model.mesh.field.setNumber(field, "Sampling", sampling)
        gmsh.model.mesh.field.setNumbers(field, "EdgesList", [curve_tag])
        return field
    
    # Create distance fields for both holes
    field1 = create_field(e1_c, rx1)
    field2 = create_field(e2_c, rx2)
    
    # Combine fields: use minimum distance to either hole
    field3 = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(field3, "FieldsList", [field1, field2])
    
    # Threshold field: transition from fine to coarse mesh
    field4 = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(field4, "InField", field3)
    gmsh.model.mesh.field.setNumber(field4, "SizeMin", 0.1)   # Fine mesh near holes
    gmsh.model.mesh.field.setNumber(field4, "SizeMax", 0.8)   # Coarse mesh far away
    gmsh.model.mesh.field.setNumber(field4, "DistMin", 0.0)   # Distance transition start
    gmsh.model.mesh.field.setNumber(field4, "DistMax", 3.0)   # Distance transition end
    
    gmsh.model.mesh.field.setAsBackgroundMesh(field4)
    
    # Generate 2D mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)  # Use 2nd order (quadratic) elements
    gmsh.model.mesh.optimize("Netgen")  # Optimize with Netgen
    gmsh.model.mesh.optimize("Gmsh")    # Optimize with Gmsh
    
    # Display mesh in Gmsh GUI if requested
    if visualize == True:
        gmsh.fltk.run()
    
    return {'new_plate_tag': new_plate_tag,
            'e1_c': e1_c,
            'e2_c': e2_c}

"""========================================================================="""

# ========== GMSH FUNCTIONS ==========

def visualize_geometry(params, sim_number=0):
    """
    Display the geometry with elliptical holes in the Gmsh GUI.
    
    Creates a visual representation of the plate geometry for user inspection
    before running the analysis.
    
    Parameters
    ----------
    params : dict
        Hole parameters dictionary
    sim_number : int, optional
        Simulation identifier (default: 0)
    """
    
    gmsh.initialize()
    gmsh.model.add("Visualization")
    
    create_mesh(params, sim_number, visualize=True)
    
    gmsh.finalize()

"""========================================================================="""

def generate_inp(params, ccx_input, sim_number):
    """
    Generate CalculiX input file (.inp) from geometry and parameters.
    
    This function:
    1. Creates the mesh using Gmsh
    2. Identifies fixed and loaded boundary nodes
    3. Defines material properties and boundary conditions
    4. Writes a complete CalculiX input file
    
    Parameters
    ----------
    params : dict
        Hole parameters dictionary
    ccx_input : str
        Output filename for CalculiX input file
    sim_number : int
        Simulation identifier
    
    Returns
    -------
    str
        Path to generated input file
    """
    
    gmsh.initialize()
    
    # Mesh quality settings
    gmsh.option.setNumber("Mesh.Smoothing", 20)
    gmsh.option.setNumber("Mesh.MinimumCurvePoints", 5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    
    gmsh.model.add("Tension_plate")
    
    # Create mesh
    geom = create_mesh(params, sim_number, visualize=False)
    new_plate_tag = geom['new_plate_tag']

    # MESH QUALITY REPORT
    # Retrieve and analyze element quality metrics
    element_data = gmsh.model.mesh.getElements(dim=2)
    element_tags = element_data[1][0]
    element_qualities = gmsh.model.mesh.getElementQualities(elementTags=element_tags)
    
    print(f"\n[SIM {sim_number}] ELEMENT QUALITY:")
    for lower in np.arange(0, 1, 0.1):
        upper = lower + 0.1
        count = np.sum((element_qualities >= lower) & (element_qualities < upper))
        pct = 100 * count / len(element_qualities)
        print(f"  {lower:.2f}-{upper:.2f}: {count:4d} ({pct:5.1f}%)")

    avg_quality = np.mean(element_qualities)
    print(f"  Average quality: {avg_quality:.3f}\n")

    # DEFINING FIXED AND LOAD NODES
    # Identify nodes on left edge (fixed) and right edge (loaded)
    TOL = 1e-4
    left_nodes, right_nodes = [], []
    
    # Extract node coordinates from mesh
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    # Reshape array: convert from 1D vector to 2D array (N x 3)
    node_coords = node_coords.reshape(-1, 3)
    
    # Classify nodes by x-coordinate
    for i, tag in enumerate(node_tags):
        x = node_coords[i][0]
        
        # Fixed nodes: x ≈ 0 (left edge)
        if abs(x) < TOL:
            left_nodes.append(int(tag))
        
        # Load nodes: x ≈ plate_width (right edge)
        if abs(x - w) < TOL:
            right_nodes.append(int(tag))
    
    print(f"[SIM {sim_number}] FIXED NODES: {len(left_nodes)}  |  LOAD NODES: {len(right_nodes)}")
    
    # PHYSICAL GROUPS
    # Define a physical group for the plate (needed for material assignment)
    gmsh.model.addPhysicalGroup(2, [new_plate_tag], tag=3, name="Plate")
    
    # SAVING MESH - Write temporary mesh file
    temp_mesh_file = f"_temp_mesh_{sim_number}.inp"
    gmsh.write(temp_mesh_file)
    gmsh.finalize()
    
    # READING MESH FILE
    # Read the mesh content to embed in the CalculiX input file
    with open(temp_mesh_file, "r") as k:
        mesh_content = k.read()
    
    # CREATING NSET SETS
    # Node sets for boundary conditions (CalculiX format)
    nset_fixed = "*NSET, NSET=Fixed\n"
    for i, node in enumerate(left_nodes):
        if i > 0 and i % 16 == 0:
            nset_fixed += "\n"
        nset_fixed += f"{node}, "
    nset_fixed = nset_fixed.rstrip(", ") + "\n"
    
    nset_load = "*NSET, NSET=Load\n"
    for i, node in enumerate(right_nodes):
        if i > 0 and i % 16 == 0:
            nset_load += "\n"
        nset_load += f"{node}, "
    nset_load = nset_load.rstrip(", ") + "\n"
    
    # CREATING CALCULIX INPUT FILE
    # Assemble complete CalculiX input file with mesh, materials, and boundary conditions
    calculix_input = f"""{mesh_content}

{nset_fixed}

{nset_load}

*MATERIAL, NAME=Steel
*ELASTIC
210000, 0.3

*SOLID SECTION, ELSET=Plate, MATERIAL=Steel
1.0

*STEP, NLGEOM=NO, INC=100
*STATIC
0.01, 1.0

**BOUNDARY CONDITIONS (Fixed support)
*BOUNDARY
Fixed, 1, 3, 0.0

**LOADING (Uniaxial tension in X direction)
*BOUNDARY
Load, 1, 1, 0.005

**OUTPUT REQUESTS
*NODE PRINT
U, S

*EL PRINT, ELSET=Plate
S

*NODE FILE
U

*EL FILE
S

*END STEP
"""
    
    # Write CalculiX input file
    with open(ccx_input, "w") as f:
        f.write(calculix_input)
    
    print(f"[SIM {sim_number}] Input file generated: {ccx_input}")
    
    # Clean up temporary mesh file
    if os.path.exists(temp_mesh_file):
        os.remove(temp_mesh_file)
    
    return ccx_input, avg_quality

"""========================================================================="""

def generate_random_params():
    """
    Generate random parameters for two elliptical holes.
    
    Each hole is defined by:
    - (x, y): center coordinates within the plate
    - (rx, ry): semi-major and semi-minor axes
    - angle: rotation angle in degrees [0-360]
    
    Constraints:
    - Holes must be at least 2mm from plate edges
    - rx > ry (major axis > minor axis)
    - rx in [0.4, 2.0] mm
    - ry in [0.3, rx] mm
    
    Returns
    -------
    dict
        Dictionary with two holes, each containing x, y, rx, ry, angle
    """
    
    def generate_hole():
        """Generate random parameters for a single elliptical hole."""
        
        min_dist = 2.0  # Minimum distance from plate edge [mm]
        
        # Random center coordinates
        x = round(random.uniform(min_dist, w - min_dist), 2)
        y = round(random.uniform(min_dist, h - min_dist), 2)
        
        # Generate major and minor axes (rx > ry guaranteed)
        rx = round(random.uniform(0.4, 2.0), 2)  # Major semi-axis [mm]
        ry = round(random.uniform(0.3, rx), 2)   # Minor semi-axis [mm]
        
        # Random orientation angle
        angle = round(random.uniform(0, 360), 1)  # Angle [degrees]
        
        return {'x': x, 'y': y, 'rx': rx, 'ry': ry, 'angle': angle}
    
    hole1 = generate_hole()
    hole2 = generate_hole()
    
    return {
        'hole1': hole1,
        'hole2': hole2
    }

"""========================================================================="""

def run_calculix_simulation(inp_filename, sim_number):
    """
    Execute CalculiX structural analysis.
    
    Runs the CalculiX solver (ccx) with the generated input file and
    handles common error conditions. Output files are automatically moved
    to the output_files directory.
    
    Parameters
    ----------
    inp_filename : str
        Path to CalculiX input file
    sim_number : int
        Simulation identifier for logging
    
    Returns
    -------
    bool
        True if analysis completed successfully, False otherwise
    """
    
    try:
        print(f"[SIM {sim_number}] Running CalculiX simulation...")
        
        # Extract filename without .inp extension (CalculiX convention)
        inp_base = inp_filename.replace(".inp", "")
        
        # Execute CalculiX solver
        result = subprocess.run(
            [CALCULIX_PATH, inp_base],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"[SIM {sim_number}] Analysis completed successfully!")
            
            # Move output files to output_files directory
            os.makedirs("output_files", exist_ok=True)
            output_extensions = ['.dat', '.frd', '.log', '.sta']
            for ext in output_extensions:
                src_file = f"{inp_base}{ext}"
                dst_file = f"output_files/sim_{sim_number:04d}{ext}"
                if os.path.exists(src_file):
                    shutil.move(src_file, dst_file)
                    print(f"[SIM {sim_number}] Moved {ext} file to output_files/")
            
            return True
        else:
            print(f"[SIM {sim_number}] Analysis error:")
            print(f"[SIM {sim_number}] STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[SIM {sim_number}] Timeout - simulation took too long (>60s)")
        return False
    except FileNotFoundError:
        print(f"[SIM {sim_number}] CalculiX not found at: {CALCULIX_PATH}")
        return False
    except Exception as e:
        print(f"[SIM {sim_number}] Unexpected error: {e}")
        return False

"""========================================================================="""

def von_mises(sxx, syy, szz, sxy, sxz, syz):
    """
    Calculate Von Mises equivalent stress.
    
    Von Mises stress is a scalar value that represents the equivalent uniaxial
    stress from a multiaxial stress state. Used to predict material failure.
    
    Parameters
    ----------
    sxx, syy, szz : float
        Normal stresses [Pa]
    sxy, sxz, syz : float
        Shear stresses [Pa]
    
    Returns
    -------
    float
        Von Mises equivalent stress [Pa]
    """
    
    return np.sqrt(0.5*((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 + 
                         6*(sxy**2 + sxz**2 + syz**2)))
    
"""========================================================================="""

def stress_histograms(sim_number):
    """
    Extract and visualize Von Mises stress distribution. 
    
    Reads stress results from CalculiX output file and creates a histogram
    showing the distribution of Von Mises stresses across integration points.
    
    NOTE: For >10 simulations, disable function to save time and avoid chaos!
    
    Parameters
    ----------
    sim_number : int
        Simulation identifier
    
    Returns
    -------
    numpy.ndarray or None
        Array of Von Mises stress values, or None if file not found
    """
    
    # Skip if output file doesn't exist
    dat_file = f"output_files/sim_{sim_number:04d}.dat"
    
    vm = []
    
    # Read stress data from CalculiX output file
    with open(dat_file) as k:
        for line in k:
            
            parts = line.split()
            
            # Filter lines with 8 columns and numeric first two values
            if len(parts) == 8 and parts[0].isdigit() and parts[1].isdigit():
                
                try:
                    # Extract stress components (columns 2-7)
                    sxx, syy, szz, sxy, sxz, syz = map(float, parts[2:])
                    vm.append(von_mises(sxx, syy, szz, sxy, sxz, syz))
                    
                except ValueError:
                    continue
    
    if not vm:
        print(f"[SIM {sim_number}] No von Mises values found")
        return None
        
    vm = np.array(vm)
    
    # Print stress statistics
    print(f"\n[SIM {sim_number}] VON MISES STRESS STATISTICS:")
    print(f"  n = {len(vm)}")
    print(f"  min = {vm.min():.2f} MPa")
    print(f"  max = {vm.max():.2f} MPa")
    
    # Create histogram
    os.makedirs("data", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(vm, bins=50, color="blue", edgecolor="black", alpha=0.6)
    plt.xlabel("von Mises stress (Pa)")
    plt.ylabel("Number of integration points")
    plt.title(f"von Mises stress distribution - SIM {sim_number}")
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    
    # Save histogram in data directory
    plt.savefig(f"data/stress_histogram_sim_{sim_number:04d}.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    
    return vm
    
"""========================================================================="""        

def params_csv_histograms(all_params, output_dir):
    """
    Export simulation parameters to CSV and create distribution histograms.
    
    Generates a CSV file with all simulation parameters and creates a 2x5 grid
    of histograms showing the distribution of each geometric parameter across
    all simulations.
    
    Parameters
    ----------
    all_params : list of dict
        List of parameter dictionaries from successful simulations
    output_dir : str
        Directory for output files
    """
    
    os.makedirs(output_dir, exist_ok=True)
    data = []
    
    # Extract parameters from each simulation
    for i, params in enumerate(all_params):
        data.append({
            "SIMULATION": i,
            'x1': params['hole1']['x'],
            'y1': params['hole1']['y'],
            'rx1': params['hole1']['rx'],
            'ry1': params['hole1']['ry'],
            'angle1': params['hole1']['angle'],
            'x2': params['hole2']['x'],
            'y2': params['hole2']['y'],
            'rx2': params['hole2']['rx'],
            'ry2': params['hole2']['ry'],
            'angle2': params['hole2']['angle'],
            'avg_quality': params['avg_quality']
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    csv_file = os.path.join(output_dir, "geom_params.csv")
    df.to_csv(csv_file, index=False, sep=";", decimal=",")

    # Create 2x5 histogram grid
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))
    fig.suptitle("GEOMETRICAL PARAMETERS", fontsize=16, fontweight="bold")
    
    # Define histogram parameters for each subplot
    params_to_plot = [
        ('x1', 0, 0, 'red'),
        ('y1', 0, 1, 'blue'),
        ('rx1', 0, 2, 'magenta'),
        ('ry1', 0, 3, 'yellow'),
        ('angle1', 0, 4, 'green'),
        ('x2', 1, 0, 'red'),
        ('y2', 1, 1, 'blue'),
        ('rx2', 1, 2, 'magenta'),
        ('ry2', 1, 3, 'yellow'),
        ('angle2', 1, 4, 'green'),
    ]
    
    # Create histograms
    for param_name, row, col, color in params_to_plot:
        axes[row, col].hist(df[param_name], bins=30, color=color, alpha=0.7, edgecolor="black")
        axes[row, col].set_title(f"{param_name} distribution", fontweight="bold")
        
        # Set appropriate labels
        if param_name.startswith('angle'):
            axes[row, col].set_xlabel(f"{param_name} [°]")
        else:
            axes[row, col].set_xlabel(f"{param_name} [mm]")
        
        axes[row, col].set_ylabel("n")
        axes[row, col].grid(True, alpha=0.5)
    
    plt.tight_layout()
    stat_file = os.path.join(output_dir, "param_histograms.png")
    plt.savefig(stat_file, dpi=200, bbox_inches="tight")
    print(f"Statistics saved: {stat_file}")
    
    plt.close()

"""========================================================================="""

def calculix_gui_visualization(sim_number, output_dir="output_files"):
    """
    Open CalculiX result file in CGX visualization tool.
    
    Parameters
    ----------
    sim_number : int
        Simulation identifier
    output_dir : str, optional
        Directory containing output files (default: "output_files")
    """
    
    frd_file = f"{output_dir}/sim_{sim_number:04d}.frd"
    
    try:
        subprocess.Popen([CGX_PATH, frd_file])
        print(f"[SIM {sim_number}] CalculiX GUI opened!")
    except Exception as e:
        print(f"[SIM {sim_number}] Error: {e}")
        
"""========================================================================="""
# ========== MAIN LOOP ==========

def run_simulation(i):
    """
    Execute a single FEA simulation.
    
    Workflow:
    1. Generate random hole parameters
    2. Display geometry (for first simulation only)
    3. Generate CalculiX input file
    4. Run analysis
    5. Extract and visualize stress results
    6. Open CGX viewer (for first simulation only)
    
    Parameters
    ----------
    i : int
        Simulation index
    
    Returns
    -------
    dict or None
        Parameter dictionary if successful, None if failed
    """
    try:
        print(f"\n{'='*70}")
        print(f"SIMULATION {i+1}")
        print(f"{'='*70}")
        
        # Generate random parameters
        params = generate_random_params()
        
        # VISUALIZATION OF FIRST PLATE
        if i == 0:
            print(f"\n[SIM {i}] Visualizing first plate...")
            visualize_geometry(params, sim_number=i)
            
            response = input("\nContinue with simulations (y/n)? ")
            if response.lower() != 'y':
                print("Execution stopped.")
                sys.exit(0)
        
        # GENERATE INPUT FILE
        os.makedirs("input_files", exist_ok=True)
        ccx_input = f"input_files/sim_{i:04d}.inp"
        
        ccx_input, avg_quality = generate_inp(params, ccx_input, sim_number=i)
        params['avg_quality'] = avg_quality
        
        # RUN SIMULATION
        success = run_calculix_simulation(ccx_input, sim_number=i)
        
        if success:
            
            stress_histograms(i)
            
            if i == 0:
               calculix_gui_visualization(i, "output_files")
            
            return params
        else:
            print(f"[SIM {i}] Simulation failed")
            return None
            
    except Exception as e:
        print(f"[SIM {i}] X Error in simulation: {e}")
        import traceback
        traceback.print_exc()
        return None

"""========================================================================="""
# ========== MAIN PROGRAM ==========

if __name__ == "__main__":
    
    print(f"{'='*70}")
    print("PARAMETRIC FEA SIMULATION - TENSION PLATE WITH ELLIPSES")
    print("Sequential Version (Non-Parallelized)")
    print(f"{'='*70}\n")
    
    results = []
    successful_sims = 0
    
    num_simulations = 10  # Number of simulations to run
    
    # Run simulations sequentially
    for i in range(num_simulations):
        result = run_simulation(i)
        if result is not None:
            results.append(result)
            successful_sims += 1
            print(f"[SIM {i}] Simulation {i+1}/{num_simulations} completed successfully")
        else:
            print(f"[SIM {i}] X Simulation {i+1}/{num_simulations} failed")
    
    # Print summary results
    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"Total simulations: {num_simulations}")
    print(f"Successful simulations: {successful_sims}")
    print(f"Failed simulations: {num_simulations - successful_sims}")
    print("Pipeline completed!")
    print(f"{'='*70}")
    
    # Post-processing: generate statistics and histograms
    if results:
       print(f"\n{'='*70}")
       print("Generating CSV and PNG files")
       print(f"{'='*70}")
       params_csv_histograms(results, "data")
