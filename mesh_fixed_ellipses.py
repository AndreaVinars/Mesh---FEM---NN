"""
2D Tensile Plate with Two Fixed Elliptical Holes - CalculiX Mesh Generator
Created on Mon Nov 10 18:55:23 2025
@author: Andrea Vinarš

Description:
This script generates a 2D finite element mesh of a tensile test plate 
(10mm x 10mm) with two parametrically defined elliptical holes. 
The mesh is generated using GMSH and exported to CalculiX format (.inp).

Fixed and load boundaries are automatically identified for FEM setup.
"""

import gmsh
import numpy as np
import os

# ============================================================================
# STEP 1: DEFINE ELLIPSE PARAMETERS (FIXED VALUES - NON-PARAMETRIC)
# ============================================================================
# These parameters define the two elliptical holes in the plate
# x, y = center coordinates (mm)
# radX = semi-major axis (mm)
# radY = semi-minor axis (mm)
# angle = rotation angle (degrees)

x1, y1, radX1, radY1, angle1 = 6, 7, 1.2, 0.8, 30      # Ellipse 1
x2, y2, radX2, radY2, angle2 = 2, 3, 1.3, 0.7, -45     # Ellipse 2

# ============================================================================
# STEP 2: INITIALIZE GMSH AND CREATE MODEL
# ============================================================================

gmsh.initialize()
# gmsh.model.add(name) - Creates a new model in GMSH
gmsh.model.add("TENSILE_PLATE")

# ============================================================================
# STEP 3: CREATE BASE GEOMETRY - RECTANGULAR PLATE
# ============================================================================
# addRectangle(x_min, y_min, z, width, height)
# Returns the tag (ID) of the created rectangle
# Plate dimensions: 10mm x 10mm in XY plane (z=0)

plate = gmsh.model.occ.addRectangle(0, 0, 0, 10, 10)

# ============================================================================
# STEP 4: CREATE FIRST ELLIPTICAL HOLE
# ============================================================================

# addEllipse(center_x, center_y, z, semi_major, semi_minor)
# Returns the curve (1D entity) representing the ellipse
e1_c = gmsh.model.occ.addEllipse(x1, y1, 0, radX1, radY1)

# rotate(entities, center_x, center_y, center_z, axis_x, axis_y, axis_z, angle_rad)
# (1, e1_c) = entity type 1 (curve) with tag e1_c
# Rotation axis (0, 0, 1) = Z-axis (rotation in XY plane)
# angle1*np.pi/180 converts degrees to radians
gmsh.model.occ.rotate([(1, e1_c)], x1, y1, 0, 0, 0, 1, angle1*np.pi/180)

# addCurveLoop(curves) - Creates a closed loop from one or more curves
# Returns tag of the curve loop (needed to create a surface)
e1_cl = gmsh.model.occ.addCurveLoop([e1_c])

# addPlaneSurface(curve_loops) - Creates a 2D surface from curve loop(s)
# This surface will be "subtracted" from the plate (Boolean cut)
hole1 = gmsh.model.occ.addPlaneSurface([e1_cl])

# ============================================================================
# STEP 5: CREATE SECOND ELLIPTICAL HOLE
# ============================================================================
# Same logic as hole1

e2_c = gmsh.model.occ.addEllipse(x2, y2, 0, radX2, radY2)
gmsh.model.occ.rotate([(1, e2_c)], x2, y2, 0, 0, 0, 1, angle2*np.pi/180)
e2_cl = gmsh.model.occ.addCurveLoop([e2_c])
hole2 = gmsh.model.occ.addPlaneSurface([e2_cl])

# ============================================================================
# STEP 6: SYNCHRONIZE OCC MODEL
# ============================================================================
# synchronize() must be called after OCC operations and before further processing
# It transfers geometry from OCC kernel to GMSH model
gmsh.model.occ.synchronize()

# ============================================================================
# STEP 7: EXTRACT BOUNDARY EDGES OF HOLES (FOR MESH REFINEMENT)
# ============================================================================
# getBoundary(entities, recursive) - Returns edges (boundaries) of surfaces
# (2, hole1/hole2) = entity type 2 (surface) with tag hole1/hole2
# recursive=True includes all sub-boundaries

hole1_edges = gmsh.model.getBoundary([(2, hole1)], recursive=True)
hole2_edges = gmsh.model.getBoundary([(2, hole2)], recursive=True)

# ============================================================================
# STEP 8: BOOLEAN CUT - CREATE PLATE WITH HOLES
# ============================================================================
# cut(object, tools) - Subtracts tool surfaces from object surface
# Returns: [[dim, tags], [dim, tags_map]]
# We need to extract the resulting plate tag

cut_result = gmsh.model.occ.cut([(2, plate)], [(2, hole1), (2, hole2)])
# cut_result[0] = list of resulting entities
# cut_result[0][0] = (dim, tag) of first resulting entity
# cut_result[0][0][1] = tag of the plate with holes
new_plate_tag = cut_result[0][0][1]

# ============================================================================
# STEP 9: SYNCHRONIZE AFTER BOOLEAN OPERATION
# ============================================================================
gmsh.model.occ.synchronize()

# ============================================================================
# STEP 10: MESH SIZE DEFINITION AND REFINEMENT
# ============================================================================
# getEntities(dim) - Returns all entities of dimension dim
# dim: 0=points, 1=lines, 2=surfaces, 3=volumes

# Set global mesh size to 0.8 mm for all points
all_points = gmsh.model.getEntities(0)
gmsh.model.mesh.setSize(all_points, 0.8)

# Refine mesh at hole boundaries (smaller elements = more accurate stress concentration)
gmsh.model.mesh.setSize(hole1_edges, 0.2)
gmsh.model.mesh.setSize(hole2_edges, 0.2)

# generate(dim) - Generates mesh of dimension dim
# dim=2 generates 2D mesh (triangles)
gmsh.model.mesh.generate(2)

# ============================================================================
# STEP 11: IDENTIFY BOUNDARY EDGES FOR FEM BOUNDARY CONDITIONS
# ============================================================================
# Goal: Find left edge (x=0) for fixed BC and right edge (x=10) for load BC
# We iterate through all edges and check their coordinates

TOL = 1e-3  # Tolerance for floating-point comparison

all_edges = gmsh.model.getEntities(1)  # Get all 1D entities (edges)
left_line, right_line = [], []

for dim, tag in all_edges:
    # Get boundary points (nodes) of each edge
    edge_nodes = gmsh.model.getBoundary([(1, tag)], recursive=False)
    
    # Get coordinates of boundary points
    coords = [gmsh.model.getValue(d_n, t_n, []) for d_n, t_n in edge_nodes]
    
    # Only process edges with 2 endpoints (straight edges on plate boundary)
    if len(coords) == 2:
        x_vals = [c[0] for c in coords]  # Extract x-coordinates
        y_vals = [c[1] for c in coords]  # Extract y-coordinates
        
        # Check if edge is vertical (x-coordinates same, y-coordinates different)
        if abs(x_vals[0] - x_vals[1]) < TOL and abs(y_vals[0] - y_vals[1]) > TOL:
            
            # Check if edge is on left boundary (x ≈ 0)
            if all(abs(x) < TOL for x in x_vals):
                left_line.append(tag)
            
            # Check if edge is on right boundary (x ≈ 10)
            elif all(abs(x - 10.0) < TOL for x in x_vals):
                right_line.append(tag)

# ============================================================================
# STEP 12: CREATE PHYSICAL GROUPS FOR CALCULIX
# ============================================================================
# Physical groups define regions and boundaries for FEM solver
# tag: unique identifier for the physical group
# name: descriptive name
# First argument: list of entity tags

# Surface group (the plate with holes)
gmsh.model.addPhysicalGroup(2, [new_plate_tag], tag=3, name="Plate")

# Line groups (boundaries for boundary conditions)
gmsh.model.addPhysicalGroup(1, left_line, tag=1, name="Fixed")    # Fixed BC
gmsh.model.addPhysicalGroup(1, right_line, tag=2, name="Load")    # Load BC

# ============================================================================
# STEP 13: PRINT SUMMARY AND SAVE
# ============================================================================

print("=" * 60)
print("MESH GENERATION SUMMARY")
print("=" * 60)
print(f"Fixed boundary edges found: {len(left_line)}")
print(f"Load boundary edges found:  {len(right_line)}")
print("=" * 60)

# Create output directory if it doesn't exist
os.makedirs("meshes", exist_ok=True)

# Save mesh in CalculiX format (.inp)
gmsh.write("meshes/TENSILE_ANALYSIS.inp")

# Visualize mesh in GMSH GUI
gmsh.fltk.run()

# Finalize GMSH and clean up
gmsh.finalize()
