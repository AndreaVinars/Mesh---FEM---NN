"""
TENSION PLATE FEA - GMSH MESH GENERATION & CALCULIX ANALYSIS
Single Geometry with Two Elliptical Holes

This script:
1. Creates a rectangular plate geometry with two elliptical holes
2. Generates an adaptive FE mesh using distance-based refinement
3. Identifies boundary nodes for constraints and loads
4. Generates a CalculiX input file with material and boundary conditions
5. Runs the structural analysis using CalculiX solver


Author: Andrea Vinarš
"""

import gmsh
import os
import numpy as np
import subprocess

# ========== WORKING DIRECTORY ==========
os.chdir("-")
calculix_path = r"-/ccx_static.exe"

# ========== GEOMETRY PARAMETERS ==========
# Define elliptical hole positions and dimensions [mm]

# Hole 1 parameters
x1, y1 = 6, 7              # Center coordinates [mm]
radX1, radY1 = 1.2, 0.8    # Semi-major and semi-minor axes [mm]
angle1 = 45                 # Rotation angle [degrees]

# Hole 2 parameters
x2, y2 = 3, 2              # Center coordinates [mm]
radX2, radY2 = 1.3, 0.7    # Semi-major and semi-minor axes [mm]
angle2 = -45                # Rotation angle [degrees]

# ========== GEOMETRY CREATION ==========
gmsh.initialize()
gmsh.model.add("tension_plate_FEA")

# Create base rectangular plate
# Plate dimensions: 10mm x 10mm
plate = gmsh.model.occ.addRectangle(0, 0, 0, 15, 15)

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

gmsh.model.occ.synchronize()

# ========== ADAPTIVE MESH REFINEMENT USING DISTANCE FIELDS ==========
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
# Example: Point at 0.5mm from e1 and 1.2mm from e2 → field3 = 0.5
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
gmsh.model.mesh.field.setNumber(field4, "DistMax", 3.0)   # Transition end distance [mm]
print("[MESH] Threshold field 4 created")

# Set field4 as background mesh (master size field)
gmsh.model.mesh.field.setAsBackgroundMesh(field4)
print("[MESH] Field 4 set as background mesh")

# ========== MESH GENERATION ==========
# Global element size as fallback
all_points = gmsh.model.getEntities(0)
gmsh.model.mesh.setSize(all_points, 0.8)

# Generate 2D triangular mesh
gmsh.model.mesh.generate(2)
print("[MESH] 2D mesh generated with adaptive refinement")

# ========== BOUNDARY NODE IDENTIFICATION ==========
"""
Find nodes on:
- Left edge (x ≈ 0): Fixed support (displacement constraints)
- Right edge (x ≈ 10): Applied load (tension)
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
    
    # Load boundary: right edge (x ≈ 10)
    if abs(x - 10) < TOL:
        right_nodes.append(int(tag))

print(f"[BOUNDARIES] Fixed nodes: {len(left_nodes)}  |  Load nodes: {len(right_nodes)}")

# ========== PHYSICAL GROUPS ==========
# Define material region for section assignment
gmsh.model.addPhysicalGroup(2, [new_plate_tag], tag=3, name="Plate")

# ========== MESH FILE OUTPUT & VISUALIZATION ==========
gmsh.write("tension_FEA_mesh.inp")
print("[GMSH] Mesh file saved: tension_FEA_mesh.inp")

# Display mesh in Gmsh GUI
gmsh.fltk.run()

# ========== NODE SETS FOR CALCULIX ==========
"""
CalculiX requires node sets for applying boundary conditions
Format: *NSET, NSET=SetName
         nodeID1, nodeID2, nodeID3, ...
"""

# Create fixed support node set
nset_fixed = "*NSET, NSET=Fixed\n"
for i, node in enumerate(left_nodes):
    if i > 0 and i % 16 == 0:  # Line break every 16 nodes for readability
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
with open("tension_FEA_mesh.inp", "r") as f:
    mesh_content = f.read()

# ========== CREATE CALCULIX INPUT FILE ==========
"""
Complete CalculiX input file containing:
- Mesh (nodes and elements from Gmsh)
- Node sets for boundary conditions
- Material properties (Steel: E=210 GPa, ν=0.3)
- Load step definition
- Boundary conditions (fixed support + tension load)
- Output requests (displacement, stress, strain)
"""

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

**FIXED SUPPORT (left edge)
*BOUNDARY
Fixed, 1, 3, 0.0

**UNIAXIAL TENSION LOAD (right edge, x-direction)
*BOUNDARY
Load, 1, 1, 0.1

**OUTPUT REQUESTS
*NODE FILE
U, RF

*EL FILE
S, E

*END STEP
"""

# Write CalculiX input file
with open("tension_FEA.inp", "w") as f:
    f.write(calculix_input)

print("[CalculiX] Input file created: tension_FEA.inp")
gmsh.finalize()

# ========== RUN CALCULIX SOLVER ==========
"""
Execute CalculiX solver with error handling:
- Timeout: 60 seconds (abort if analysis runs too long)
- Capture stdout/stderr for diagnostics
"""


try:
    print("[CalculiX] Starting structural analysis...")
    result = subprocess.run([calculix_path, "tension_FEA"],
                            capture_output=True,
                            text=True,
                            timeout=60)
    
    if result.returncode == 0:
        print("Analysis completed successfully!")
        print("[Output files: tension_FEA.frd (results), tension_FEA.dat (stress))")
    else:
        print("X Analysis error:")
        print(f"{result.stderr}")
        
except FileNotFoundError:
    print(f"X CalculiX not found at: {calculix_path}")
except subprocess.TimeoutExpired:
    print("X Analysis timeout - exceeded 60 seconds")
