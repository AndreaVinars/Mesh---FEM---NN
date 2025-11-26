"""
Created on Tue Nov 11 17:12:31 2025

@author: Andrea Vinarš

Description:
This script generates a parametric 2D mesh (finite element grid) of a plate with two rotatable elliptical holes.
All parameters (position, axes, rotation) can be randomized for each run.
Physical groups are defined for easy boundary condition assignment in FEM solvers.
"""

import gmsh
import sys
import random
import numpy as np

# Set global plate (domain) dimensions
w, h = 10, 10  # Plate width and height in mm

# ========= GMSH FUNCTION TO EXPORT .INP FILE =========
def generate_inp(params, output_path):
    """
    Generates a .inp mesh file for CalculiX (FEM) given ellipse parameters.
    Args:
        params (dict): Contains keys 'hole1' and 'hole2' (each with x, y, rx, ry, angle)
        output_path (str): Output file path for the .inp mesh
    """
    x1, y1, rx1, ry1, angle1 = params['hole1']['x'], params['hole1']['y'], params['hole1']['rx'], params['hole1']['ry'], params['hole1']['angle']
    x2, y2, rx2, ry2, angle2 = params['hole2']['x'], params['hole2']['y'], params['hole2']['rx'], params['hole2']['ry'], params['hole2']['angle']

    gmsh.initialize()
    gmsh.model.add("TensilePlate")

    # --- Base Rectangle (Plate) ---
    plate = gmsh.model.occ.addRectangle(0, 0, 0, w, h)

    # --- Ellipse 1 with Rotation ---
    e1_c = gmsh.model.occ.addEllipse(x1, y1, 0, rx1, ry1)
    gmsh.model.occ.rotate([(1, e1_c)], x1, y1, 0, 0, 0, 1, angle1 * np.pi / 180)
    e1_cl = gmsh.model.occ.addCurveLoop([e1_c])
    hole1 = gmsh.model.occ.addPlaneSurface([e1_cl])

    # --- Ellipse 2 with Rotation ---
    e2_c = gmsh.model.occ.addEllipse(x2, y2, 0, rx2, ry2)
    gmsh.model.occ.rotate([(1, e2_c)], x2, y2, 0, 0, 0, 1, angle2 * np.pi / 180)
    e2_cl = gmsh.model.occ.addCurveLoop([e2_c])
    hole2 = gmsh.model.occ.addPlaneSurface([e2_cl])

    gmsh.model.occ.synchronize()

    # --- Refined Mesh at Ellipse Borders ---
    hole1_points = gmsh.model.getBoundary([(2, hole1)], recursive=True)
    hole2_points = gmsh.model.getBoundary([(2, hole2)], recursive=True)

    # --- Subtract Ellipses (cut holes out) ---
    cut_result = gmsh.model.occ.cut([(2, plate)], [(2, hole1), (2, hole2)])
    new_plate_tag = cut_result[0][0][1]
    gmsh.model.occ.synchronize()

    # --- Mesh Size ---
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.8)  # global mesh
    # Finer mesh near holes
    gmsh.model.mesh.setSize(hole1_points, 0.15 if rx1 <= 0.7 else 0.2)
    gmsh.model.mesh.setSize(hole2_points, 0.15 if rx2 <= 0.7 else 0.2)

    gmsh.model.mesh.generate(2)  # Generate 2D mesh (triangles/quads)

    # --- Tagging Physical Groups for FEM BCs ---
    all_edges = gmsh.model.getEntities(1)
    left_line, right_line = [], []
    TOL = 1e-3  # tolerance for edge comparison

    for dim, tag in all_edges:
        edge_nodes = gmsh.model.getBoundary([(1, tag)], recursive=False)
        coords = [gmsh.model.getValue(d_n, t_n, []) for d_n, t_n in edge_nodes]
        if len(coords) == 2:
            x_vals = [c[0] for c in coords]
            y_vals = [c[1] for c in coords]
            # Select vertical edges (boundary)
            if abs(x_vals[0] - x_vals[1]) < TOL and abs(y_vals[0] - y_vals[1]) > TOL:
                if all(abs(x) < TOL for x in x_vals):
                    left_line.append(tag)
                elif all(abs(x - w) < TOL for x in x_vals):
                    right_line.append(tag)

    gmsh.model.addPhysicalGroup(2, [new_plate_tag], tag=3, name="Plate")
    gmsh.model.addPhysicalGroup(1, left_line, tag=1, name="Fixed")
    gmsh.model.addPhysicalGroup(1, right_line, tag=2, name="Load")

    gmsh.write(output_path)
    gmsh.finalize()


def visualize_geometry(params, save_image_path=None):
    """
    Generate and view mesh for the given params.
    """
    x1, y1, rx1, ry1, angle1 = params['hole1']['x'], params['hole1']['y'], params['hole1']['rx'], params['hole1']['ry'], params['hole1']['angle']
    x2, y2, rx2, ry2, angle2 = params['hole2']['x'], params['hole2']['y'], params['hole2']['rx'], params['hole2']['ry'], params['hole2']['angle']

    gmsh.initialize()
    gmsh.model.add("Visualization")

    plate = gmsh.model.occ.addRectangle(0, 0, 0, w, h)
    e1_c = gmsh.model.occ.addEllipse(x1, y1, 0, rx1, ry1)
    gmsh.model.occ.rotate([(1, e1_c)], x1, y1, 0, 0, 0, 1, angle1 * np.pi / 180)
    e1_cl = gmsh.model.occ.addCurveLoop([e1_c])
    hole1 = gmsh.model.occ.addPlaneSurface([e1_cl])
    e2_c = gmsh.model.occ.addEllipse(x2, y2, 0, rx2, ry2)
    gmsh.model.occ.rotate([(1, e2_c)], x2, y2, 0, 0, 0, 1, angle2 * np.pi / 180)
    e2_cl = gmsh.model.occ.addCurveLoop([e2_c])
    hole2 = gmsh.model.occ.addPlaneSurface([e2_cl])

    gmsh.model.occ.synchronize()

    hole1_points = gmsh.model.getBoundary([(2, hole1)], recursive=True)
    hole2_points = gmsh.model.getBoundary([(2, hole2)], recursive=True)
    gmsh.model.occ.cut([(2, plate)], [(2, hole1), (2, hole2)])
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.8)
    gmsh.model.mesh.setSize(hole1_points, 0.15 if rx1 <= 0.7 else 0.2)
    gmsh.model.mesh.setSize(hole2_points, 0.15 if rx2 <= 0.7 else 0.2)
    gmsh.model.mesh.generate(2)

    if save_image_path:
        gmsh.write(save_image_path)
    else:
        gmsh.fltk.run()
    gmsh.finalize()

def generate_random_params():
    """
    Generate random parameters for both ellipses within the domain.
    Returns:
        dict: parameters for hole1 and hole2
    """
    def generate_hole():
        min_dist = 1.5  # Minimal distance from edge for hole center
        x = random.uniform(min_dist, w - min_dist)
        y = random.uniform(min_dist, h - min_dist)
        rx = random.uniform(0.4, 1.6)
        ry = random.uniform(0.3, rx)
        angle = random.uniform(0, 360)
        
        return {'x': x, 'y': y, 'rx': rx, 'ry': ry, 'angle': angle}
    
    hole1 = generate_hole()
    hole2 = generate_hole()
    
    return {'hole1': hole1, 'hole2': hole2}

def run_single_simulation(i):
    """
    Runs a single geometry/mesh generation simulation. (With visualization for the first one.)
    """
    try:
        params = generate_random_params()
        # Visualize only the FIRST sample
        if i == 0:
            print("Visualizing first plate geometry...")
            visualize_geometry(params)
            response = input("\nContinue with meshing? (y/n): ").strip().lower()
            if response != 'y':
                print("Exiting. Simulation stopped by user.")
                sys.exit(0)
        return params
    except Exception as e:
        print(f"Error in simulation {i}: {e}")
        return None

# ========== MAIN PROGRAM ==========
if __name__ == "__main__":
    print("Starting mesh generation pipeline...")
    results = []
    for i in range(5):  # Generate 5 sample plates as a test
        result = run_single_simulation(i)
        if result:
            print(f"Simulation {i+1}/5 finished successfully.")
            results.append(result)
    print(f"\nPipeline finished! {len(results)} mesh simulations completed.")
