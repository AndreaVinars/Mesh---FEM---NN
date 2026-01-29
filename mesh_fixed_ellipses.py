

"""
TENSION PLATE FEA WITH HOMOGENIZATION
Single Geometry Analysis for Effective Young's Modulus Calculation

This script:
1. Creates a rectangular plate geometry with two elliptical holes
2. Generates an adaptive FE mesh using distance-based refinement  
3. Identifies boundary nodes for constraints and loads
4. Generates a CalculiX input file with material and boundary conditions
5. Runs the structural analysis using CalculiX solver
6. Computes homogenized stress and effective Young's modulus from results
7. Exports results 

Author: Andrea Vinarš
"""

import gmsh
import os
import numpy as np
import subprocess
import pandas as pd

# ========== WORKING DIRECTORY ==========
os.chdir("-")
calculix_path = r"-/ccx_static.exe"

# ========== GEOMETRY PARAMETERS ==========
# Plate dimensions and applied load [mm]
width, height = 12, 12
elongation = 0.005  # Prescribed displacement [mm]

# Elliptical hole positions and dimensions [mm]

# Hole 1 parameters
x1, y1 = 8.93, 7.37        # Center coordinates [mm]
radX1, radY1 = 2.56, 0.57  # Semi-major and semi-minor axes [mm]
angle1 = 170               # Rotation angle [degrees]

# Hole 2 parameters  
x2, y2 = 4.57, 5.44        # Center coordinates [mm]
radX2, radY2 = 1.64, 0.73  # Semi-major and semi-minor axes [mm]
angle2 = -147.6            # Rotation angle [degrees]

# ========== POST-PROCESSING FUNCTIONS ==========

def triangle_area(p1, p2, p3):
    """
    Calculates the triangle area using cross product formula.
    Used for area-weighted stress homogenization.
    
    Args:
        p1, p2, p3: Tuple of (x, y) coordinates
        
    Returns:
        float: Area in mm^2
    """
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def calculate_E(inp_path, dat_path, output="results.csv"):
    
    """
    Parses CalculiX results and computes homogenized mechanical properties.
    
    Process:
    1. Parses .inp file to extract mesh (nodes and elements)
    2. Parses .dat file to extract element stresses (Sxx)
    3. Calculates element areas from nodal coordinates
    4. Computes area-weighted average stress (homogenization)
    5. Calculates effective Young's modulus: E_eff = stress / strain
    
    Args:
        inp_path: Path to CalculiX input file (mesh definition)
        dat_path: Path to CalculiX output file (stress results)
        output: CSV filename for results export
        
    Returns:
        pandas.DataFrame: Element-wise data (id, area, stress)
    """
    
    # ---------- PARSE MESH FILE ----------
    # Extract nodes and elements from CalculiX .inp file
    nodes = {}
    elements = []
    
    with open(inp_path) as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Parse node definitions: *NODE keyword followed by ID, X, Y, Z
        if line.startswith('*NODE'):
            i += 1
            while i < len(lines) and not lines[i].startswith('*'):
                parts = lines[i].split(',')
                if len(parts) >= 4:
                    nodes[int(parts[0])] = (float(parts[1]), float(parts[2]))
                i += 1
            continue
        
        # Parse element definitions: CPS6 elements (6-node triangles)
        # Only the first 3 nodes (corner nodes) are used for area calculation
        if line.startswith('*ELEMENT'):
            i += 1
            data = []
            while i < len(lines) and not lines[i].startswith('*'):
                parts = [x for x in lines[i].split(',') if x.strip()]
                if parts:
                    data.extend(parts)
                    if len(data) >= 7:  # Element ID + 6 nodes
                        elements.append({
                            'id': int(data[0]),
                            'corners': [int(data[1]), int(data[2]), int(data[3])]
                        })
                        data = []
                i += 1
            continue
        
        i += 1
    
    # ---------- CALCULATE ELEMENT AREAS ----------
    # Compute area for each triangular element from nodal coordinates
    areas = []
    
    for elem in elements:
        try:
            corners = elem['corners']
            p1, p2, p3 = nodes[corners[0]], nodes[corners[1]], nodes[corners[2]]
            areas.append({
                'element_id': elem['id'],
                'Area_mm2': triangle_area(p1, p2, p3)
            })
        except:

            pass
        
    areas_df = pd.DataFrame(areas)

    # ---------- PARSE STRESS RESULTS ----------
    # Extract Sxx stress components from CalculiX .dat file
    stress_data = []
    reading = False
    
    with open(dat_path) as f:
        for line in f:
            line = line.strip()
            
            # Detect start of stress output section
            if line.startswith('stresses (elem'):
                reading = True
                continue
            
            # End of stress section (marked by asterisk)
            if reading and line.startswith('*'):
                break
            
            # Parse stress components: element_id, integration_point, Sxx, Syy, Szz, Sxy...
            if reading and line:
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        element_id = int(parts[0])
                        sxx = float(parts[2])  # Sxx is the 3rd column
                        
                        stress_data.append({
                            'element_id': element_id,
                            'Sxx_MPa': sxx
                        })
                    except:
                        continue
    
    stress_df = pd.DataFrame(stress_data)
    if stress_df.empty:
        print(f"[ERROR] No stress data found in {dat_path}")
        return pd.DataFrame(columns=['element_id', 'Sxx_MPa'])
    
    # Average stress values over element integration points
    stress_df = stress_df.groupby('element_id')['Sxx_MPa'].mean().reset_index()
    
    # ---------- HOMOGENIZATION ----------
    """
    Compute effective Young's modulus using area-weighted averaging:
    
    1. Calculate area-weighted average stress: 
       sigma_hom = sum(sigma_i * A_i) / sum(A_i)
       
    2. Applied strain: epsilon = elongation / width
    
    3. Effective modulus: E_eff = sigma_hom / epsilon
    """
    # Merge area and stress data (inner join keeps only elements with both data)
    df = areas_df.merge(stress_df, on='element_id', how='inner')
    
    # Area-weighted average stress 
    total_area = df['Area_mm2'].sum()
    weighted_stress = (df['Sxx_MPa'] * df['Area_mm2']).sum()
    homogenized_stress = weighted_stress / total_area
    
    strain = elongation / width
    
    # Effective Young's modulus = stress / strain
    effective_youngs_modulus = homogenized_stress / strain
    
    print(f"[HOMOGENIZATION] Effective Young's Modulus: {round(effective_youngs_modulus, 2)} MPa")
    
    # Export results for manual verification in excel
    df.to_csv(output, index=False, sep=';', decimal=',')
    
    return df


# ========== INITIALIZE GMSH ==========
gmsh.initialize()
gmsh.model.add("tensile_plate_fea")

# ========== GEOMETRY CREATION ==========
# Create base rectangular plate
plate = gmsh.model.occ.addRectangle(0, 0, 0, width, height)

# ---------- ELLIPSE 1 ----------
# Create first elliptical hole with rotation
e1_c = gmsh.model.occ.addEllipse(x1, y1, 0, radX1, radY1)   
gmsh.model.occ.rotate([(1, e1_c)], x1, y1, 0, 0, 0, 1, angle1*np.pi/180)
e1_cl = gmsh.model.occ.addCurveLoop([e1_c])
hole1 = gmsh.model.occ.addPlaneSurface([e1_cl])

# ---------- ELLIPSE 2 ----------
# Create second elliptical hole with rotation
e2_c = gmsh.model.occ.addEllipse(x2, y2, 0, radX2, radY2)
gmsh.model.occ.rotate([(1, e2_c)], x2, y2, 0, 0, 0, 1, angle2*np.pi/180)
e2_cl = gmsh.model.occ.addCurveLoop([e2_c])
hole2 = gmsh.model.occ.addPlaneSurface([e2_cl])

gmsh.model.occ.synchronize()

# ========== BOOLEAN OPERATIONS ==========
# Cut the two holes from the plate
cut_result = gmsh.model.occ.cut([(2, plate)], [(2, hole1), (2, hole2)])
new_plate_tag = cut_result[0][0][1]
print("[GEOMETRY] Cut operation completed")

gmsh.model.occ.synchronize()

# ========== ADAPTIVE MESH REFINEMENT ==========
"""
Strategy: Create a mesh size field that varies with distance to ellipse curves
- Fine mesh (SizeMin) near hole boundaries for accurate stress capture
- Coarse mesh (SizeMax) far from holes to reduce total element count
- Smooth transition between fine and coarse regions
"""

# Field 1: Distance from ellipse 1
# Calculates distance from each point in domain to curve e1_c
field1 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumber(field1, "Sampling", 100)    # 100 sample points on curve
gmsh.model.mesh.field.setNumbers(field1, "EdgesList", [e1_c])
print("[MESH] Distance field 1 created for ellipse 1")

# Field 2: Distance from ellipse 2
field2 = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumber(field2, "Sampling", 100)
gmsh.model.mesh.field.setNumbers(field2, "EdgesList", [e2_c])
print("[MESH] Distance field 2 created for ellipse 2")

# Field 3: Minimum distance to either ellipse
# For each point: min_distance = min(distance_to_e1, distance_to_e2)
field3 = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(field3, "FieldsList", [field1, field2])
print("[MESH] Min field 3 created (combines both distances)")

# Field 4: Threshold - controls element size based on distance
# Creates smooth transition from fine to coarse mesh
field4 = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(field4, "InField", field3)
gmsh.model.mesh.field.setNumber(field4, "SizeMin", 0.1)   # Fine mesh size [mm]
gmsh.model.mesh.field.setNumber(field4, "SizeMax", 0.8)   # Coarse mesh size [mm]
gmsh.model.mesh.field.setNumber(field4, "DistMin", 0.0)   # Transition start distance [mm]
gmsh.model.mesh.field.setNumber(field4, "DistMax", 5.0)   # Transition end distance [mm]
print("[MESH] Threshold field 4 created")

# Set field4 as background mesh (master size field)
gmsh.model.mesh.field.setAsBackgroundMesh(field4)

# ========== MESH GENERATION ==========
# Global element size as fallback
all_points = gmsh.model.getEntities(0)
gmsh.model.mesh.setSize(all_points, 0.8)

# Generate 2D triangular mesh with 2nd order elements (CPS6)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.setOrder(2)
print("[MESH] 2D mesh generated with adaptive refinement")

# Optimize mesh quality
gmsh.model.mesh.optimize("Netgen")    
gmsh.model.mesh.optimize("Gmsh")

# Print mesh quality statistics
element_data = gmsh.model.mesh.getElements(dim=2)
element_tags = element_data[1][0]
element_qualities = gmsh.model.mesh.getElementQualities(elementTags=element_tags)

print("\n[Mesh Quality Distribution]")
for lower in np.arange(0, 1, 0.1):
    upper = lower + 0.1
    count = np.sum((element_qualities >= lower) & (element_qualities < upper))
    pct = 100 * count / len(element_qualities)
    print(f"  {lower:.2f}-{upper:.2f}: {count:4d} ({pct:5.1f}%)")

# ========== BOUNDARY NODE IDENTIFICATION ==========
"""
Find nodes on:
- Left edge (x ≈ 0): Fixed support (displacement constraints)
- Right edge (x ≈ width): Applied prescribed displacement (tension)
"""

TOL = 1e-4  # Tolerance for boundary detection [mm]
left_nodes, right_nodes = [], []

# Extract node coordinates from mesh
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
node_coords = node_coords.reshape(-1, 3)  # Convert to Nx3 array
 
for i, tag in enumerate(node_tags):
    x = node_coords[i][0]  # X-coordinate
    
    # Fixed boundary: left edge (x ≈ 0)
    if abs(x) < TOL:
        left_nodes.append(int(tag))
    
    # Load boundary: right edge (x ≈ width)
    if abs(x - width) < TOL:
        right_nodes.append(int(tag))

print(f"[BOUNDARIES] Fixed nodes: {len(left_nodes)}  |  Load nodes: {len(right_nodes)}")

# ========== PHYSICAL GROUPS ==========
# Define material region for section assignment
gmsh.model.addPhysicalGroup(2, [new_plate_tag], tag=3, name="Plate")

# ========== MESH FILE OUTPUT ==========
gmsh.write("tensile_FEA_mesh.inp")
print("[GMSH] Mesh file saved: tensile_FEA_mesh.inp")

# Display mesh in Gmsh GUI
gmsh.fltk.run()
gmsh.finalize()

# ========== NODE SETS FOR CALCULIX ==========
"""
CalculiX requires node sets for applying boundary conditions
Format: *NSET, NSET=SetName
         nodeID1, nodeID2, nodeID3, ...
"""

# Create fixed support node set
nset_fixed = "*NSET, NSET=Fixed\n"
for i, node in enumerate(left_nodes):
    if i > 0 and i % 16 == 0:  
        nset_fixed += "\n"
    nset_fixed += f"{node}, "
nset_fixed = nset_fixed.rstrip(", ") + "\n"

# Create load node set
nset_load = "*NSET, NSET=Load\n"
for i, node in enumerate(right_nodes):
    if i > 0 and i % 16 == 0:
        nset_load += "\n"
    nset_load += f"{node}, "
nset_load = nset_load.rstrip(", ") + "\n"

# ========== READ MESH FILE ==========
# Read generated mesh to embed in CalculiX input
with open("tensile_FEA_mesh.inp", "r") as f:
    mesh_content = f.read()

# Remove temporary mesh file
os.remove("tensile_FEA_mesh.inp")

# ========== CREATE CALCULIX INPUT FILE ==========
"""
Complete CalculiX input file containing:
- Mesh (nodes and elements from Gmsh)
- Node sets for boundary conditions
- Material properties (Steel: E=210 GPa, ν=0.3)
- Load step definition
- Boundary conditions (fixed support + prescribed displacement)
- Output requests (stress only for .dat processing)
"""

calculix_input = f"""{mesh_content}

{nset_fixed}

{nset_load}

*MATERIAL, NAME=Steel
*ELASTIC
210000, 0.3

*SOLID SECTION, ELSET=Plate, MATERIAL=Steel
1.0

*STEP, NLGEOM=NO
*STATIC

** BOUNDARY CONDITIONS
*BOUNDARY
Fixed, 1, 3, 0.0

** APPLIED LOAD (Prescribed displacement)
*BOUNDARY
Load, 1, 1, {elongation}

** OUTPUT REQUESTS
*EL PRINT, ELSET=Plate
S

*END STEP
"""

# Write CalculiX input file
with open("tensile_FEA.inp", "w") as f:
    f.write(calculix_input)

print("[CalculiX] Input file created: tensile_FEA.inp")

# ========== RUN CALCULIX SOLVER ==========
"""
Execute CalculiX solver with error handling:
- Timeout: 60 seconds (abort if analysis runs too long)
- Capture stdout/stderr for diagnostics
"""

try:
    print("[CalculiX] Starting structural analysis...")
    result = subprocess.run([calculix_path, "tensile_FEA"],
                            capture_output=True,
                            text=True,
                            timeout=60)
    
    if result.returncode == 0:
        print("[CalculiX] Analysis completed successfully")
        print("[OUTPUT] Results: tensile_FEA.dat (stress data for homogenization)")
    else:
        print("[CalculiX] Error in analysis:")
        print(f"{result.stderr}")
        
except FileNotFoundError:
    print(f"[CalculiX] Error: Executable not found at: {calculix_path}")
except subprocess.TimeoutExpired:
    print("[CalculiX] Error: Analysis timeout - exceeded 60 seconds")
except Exception as e:
    print(f"[CalculiX] Unexpected error: {e}")

# ========== POST-PROCESSING ==========
print("\n[Post-processing] Extracting results and computing homogenized properties...")
data = calculate_E(
    inp_path="tensile_FEA.inp",
    dat_path="tensile_FEA.dat",
    output="results.csv"
)

print(f"\n[Complete] Results saved: {len(data)} elements processed")
print(data.head())
