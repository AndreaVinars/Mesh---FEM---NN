"""
Parametric Tensile Test Plate Mesh Generator

This module generates 2D finite element meshes for tensile test plates with
parametric elliptical holes using Gmsh. It includes mesh generation, visualization,
and parameter tracking for machine learning applications.

Author: Andrea Vinarš
Date: Thu 14 Nov 2025
"""

import gmsh
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Plate dimensions (mm)
PLATE_WIDTH = 10
PLATE_HEIGHT = 10


def generate_mesh(params, output_path, mesh_size=0.8):
    """
    Generate a 2D mesh for a tensile plate with two elliptical holes.
    
    This function creates a rectangular plate with two parametrically-defined
    elliptical holes. The mesh density is adaptively refined around the holes
    based on their size.
    
    Parameters
    ----------
    params : dict
        Dictionary containing hole parameters:
        - 'hole1' : dict with keys 'x', 'y', 'rx', 'ry', 'angle'
        - 'hole2' : dict with keys 'x', 'y', 'rx', 'ry', 'angle'
        Where:
        - x, y : hole center coordinates (mm)
        - rx, ry : semi-major and semi-minor axes (mm)
        - angle : rotation angle (degrees)
    
    output_path : str
        Path where the .msh file will be saved
    
    mesh_size : float, optional
        Default mesh element size (mm). Default is 0.8.
    
    Returns
    -------
    tuple
        (num_elements, num_nodes) : Number of mesh elements and nodes
    """
    
    # Extract parameters for hole 1
    x1, y1 = params['hole1']['x'], params['hole1']['y']
    rx1, ry1 = params['hole1']['rx'], params['hole1']['ry']
    angle1 = params['hole1']['angle']
    
    # Extract parameters for hole 2
    x2, y2 = params['hole2']['x'], params['hole2']['y']
    rx2, ry2 = params['hole2']['rx'], params['hole2']['ry']
    angle2 = params['hole2']['angle']
    
    # Initialize Gmsh and create a new model
    gmsh.initialize()
    gmsh.model.add("Tensile_Plate")
    
    # Create plate (rectangle)
    plate = gmsh.model.occ.addRectangle(0, 0, 0, PLATE_WIDTH, PLATE_HEIGHT)
    
    # Create first elliptical hole
    e1_c = gmsh.model.occ.addEllipse(x1, y1, 0, rx1, ry1)
    gmsh.model.occ.rotate([(1, e1_c)], x1, y1, 0, 0, 0, 1, angle1 * np.pi / 180)
    e1_cl = gmsh.model.occ.addCurveLoop([e1_c])
    hole1 = gmsh.model.occ.addPlaneSurface([e1_cl])
    
    # Create second elliptical hole
    e2_c = gmsh.model.occ.addEllipse(x2, y2, 0, rx2, ry2)
    gmsh.model.occ.rotate([(1, e2_c)], x2, y2, 0, 0, 0, 1, angle2 * np.pi / 180)
    e2_cl = gmsh.model.occ.addCurveLoop([e2_c])
    hole2 = gmsh.model.occ.addPlaneSurface([e2_cl])
    
    # Apply geometry changes
    gmsh.model.occ.synchronize()
    
    # Get boundary edges of holes for mesh refinement
    hole1_edges = gmsh.model.getBoundary([(2, hole1)], recursive=True)
    hole2_edges = gmsh.model.getBoundary([(2, hole2)], recursive=True)
    
    # Subtract holes from plate
    gmsh.model.occ.cut([(2, plate)], [(2, hole1), (2, hole2)])
    gmsh.model.occ.synchronize()
    
    # Set default mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    
    # Adaptive mesh refinement: smaller holes get finer mesh
    if rx1 <= 0.7:
        gmsh.model.mesh.setSize(hole1_edges, 0.15)
    else:
        gmsh.model.mesh.setSize(hole1_edges, 0.2)
    
    if rx2 <= 0.7:
        gmsh.model.mesh.setSize(hole2_edges, 0.15)
    else:
        gmsh.model.mesh.setSize(hole2_edges, 0.2)
    
    # Generate 2D mesh (triangular elements)
    gmsh.model.mesh.generate(2)
    
    # Get mesh node information
    tags, coords, parametric = gmsh.model.mesh.getNodes()
    num_nodes = len(tags)
    
    # Count total number of elements
    elements = gmsh.model.mesh.getElements()
    num_elements = 0
    for elem_block in elements[1]:
        num_elements += len(elem_block)
    
    # Save mesh to file
    gmsh.write(output_path)
    gmsh.finalize()
    
    return num_elements, num_nodes


def visualize_geometry(params):
    """
    Visualize the geometry and mesh using Gmsh interactive window.
    
    This function creates and displays a mesh for visual inspection before
    running simulations. Useful for verifying geometry and mesh quality.
    
    Parameters
    ----------
    params : dict
        Dictionary containing hole parameters (see generate_mesh for format)
    """
    
    # Extract parameters
    x1, y1 = params['hole1']['x'], params['hole1']['y']
    rx1, ry1 = params['hole1']['rx'], params['hole1']['ry']
    angle1 = params['hole1']['angle']
    
    x2, y2 = params['hole2']['x'], params['hole2']['y']
    rx2, ry2 = params['hole2']['rx'], params['hole2']['ry']
    angle2 = params['hole2']['angle']
    
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("Visualization")
    
    # Create plate
    plate = gmsh.model.occ.addRectangle(0, 0, 0, PLATE_WIDTH, PLATE_HEIGHT)
    
    # Create first elliptical hole
    e1_c = gmsh.model.occ.addEllipse(x1, y1, 0, rx1, ry1)
    gmsh.model.occ.rotate([(1, e1_c)], x1, y1, 0, 0, 0, 1, angle1 * np.pi / 180)
    e1_cl = gmsh.model.occ.addCurveLoop([e1_c])
    hole1 = gmsh.model.occ.addPlaneSurface([e1_cl])
    
    # Create second elliptical hole
    e2_c = gmsh.model.occ.addEllipse(x2, y2, 0, rx2, ry2)
    gmsh.model.occ.rotate([(1, e2_c)], x2, y2, 0, 0, 0, 1, angle2 * np.pi / 180)
    e2_cl = gmsh.model.occ.addCurveLoop([e2_c])
    hole2 = gmsh.model.occ.addPlaneSurface([e2_cl])
    
    # Apply geometry changes
    gmsh.model.occ.synchronize()
    
    # Get boundary edges
    hole1_edges = gmsh.model.getBoundary([(2, hole1)], recursive=True)
    hole2_edges = gmsh.model.getBoundary([(2, hole2)], recursive=True)
    
    # Subtract holes from plate
    gmsh.model.occ.cut([(2, plate)], [(2, hole1), (2, hole2)])
    gmsh.model.occ.synchronize()
    
    # Set mesh sizes
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.8)
    
    if rx1 <= 0.7:
        gmsh.model.mesh.setSize(hole1_edges, 0.15)
    else:
        gmsh.model.mesh.setSize(hole1_edges, 0.2)
    
    if rx2 <= 0.7:
        gmsh.model.mesh.setSize(hole2_edges, 0.15)
    else:
        gmsh.model.mesh.setSize(hole2_edges, 0.2)
    
    # Generate mesh
    gmsh.model.mesh.generate(2)
    
    # Launch interactive Gmsh GUI
    gmsh.fltk.run()
    gmsh.finalize()


def generate_random_params():
    """
    Generate random parameters for two elliptical holes.
    
    This function creates random but valid hole configurations, ensuring:
    - Holes are sufficiently far from plate edges (min_dist = 1.5 mm)
    - Semi-major axis (rx) ranges from 0.4 to 1.4 mm
    - Semi-minor axis (ry) is smaller than semi-major axis
    - Rotation angles are uniformly distributed
    
    Returns
    -------
    dict
        Dictionary with keys 'hole1' and 'hole2', each containing:
        - 'x' : float, center x-coordinate (mm)
        - 'y' : float, center y-coordinate (mm)
        - 'rx' : float, semi-major axis (mm)
        - 'ry' : float, semi-minor axis (mm)
        - 'angle' : float, rotation angle (degrees)
    """
    
    def generate_hole():
        """Generate a single random hole with valid parameters."""
        
        # Minimum distance from edges to keep holes away from boundaries
        min_dist = 1.5
        
        # Random center coordinates
        x = round(random.uniform(min_dist, PLATE_WIDTH - min_dist), 2)
        y = round(random.uniform(min_dist, PLATE_HEIGHT - min_dist), 2)
        
        # Random semi-axes (semi-minor axis <= semi-major axis)
        rx = round(random.uniform(0.4, 1.4), 2)
        ry = round(random.uniform(0.3, rx), 2)
        
        # Random rotation angle
        angle = round(random.uniform(0, 360), 1)
        
        return {'x': x, 'y': y, 'rx': rx, 'ry': ry, 'angle': angle}
    
    # Generate two holes
    hole1 = generate_hole()
    hole2 = generate_hole()
    
    return {
        'hole1': hole1,
        'hole2': hole2
    }


if __name__ == "__main__":
    print("Generating meshes...\n")
    
    # Create output directories
    os.makedirs("meshes", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # List to store all generated parameters
    all_params = []
    
    # Generate 5 example meshes
    for i in range(5):
        print(f"Mesh {i + 1}/5...")
        
        # Generate random parameters
        params = generate_random_params()
        all_params.append(params)
        
        # Generate mesh
        mesh_file = f"meshes/mesh_{i:04d}.msh"
        num_elem, num_nodes = generate_mesh(params, mesh_file)
        
        # Print hole 1 parameters
        print("  Hole 1: x={x}, y={y}, rx={rx}, ry={ry}, angle={angle}°".format(
            x=params['hole1']['x'],
            y=params['hole1']['y'],
            rx=params['hole1']['rx'],
            ry=params['hole1']['ry'],
            angle=params['hole1']['angle']
        ))
        
        # Print hole 2 parameters
        print("  Hole 2: x={x}, y={y}, rx={rx}, ry={ry}, angle={angle}°".format(
            x=params['hole2']['x'],
            y=params['hole2']['y'],
            rx=params['hole2']['rx'],
            ry=params['hole2']['ry'],
            angle=params['hole2']['angle']
        ))
        
        # Print mesh statistics
        print(f"  Elements: {num_elem}, Nodes: {num_nodes}")
        print(f"  Saved: {mesh_file}\n")
    
    # Export parameters to CSV file
    data = []
    for i, params in enumerate(all_params):
        data.append({
            'mesh_id': i,
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
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    csv_file = "data/mesh_parameters.csv"
    df.to_csv(csv_file, index=False, sep=";")
    
    print("Done!")
    print(f"Saved parameters: {csv_file}")
    
    # Create visualization: scatter plot of ellipse centers and sizes
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot first set of ellipses with color-coded rotation angle
    ax.scatter(df["x1"], df["y1"], s=60 * df["rx1"], c=df["angle1"], 
               cmap="cool", alpha=0.7, label="Ellipse 1")
    
    # Plot second set of ellipses with different marker and color
    ax.scatter(df["x2"], df["y2"], s=60 * df["rx2"], c=df["angle2"], 
               cmap="autumn", alpha=0.7, label="Ellipse 2", marker="x")
    
    # Configure plot
    ax.set_title("Ellipse Centers and Dimensions on Plate")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.legend()
    
    # Save and display plot
    plt.tight_layout()
    plt.savefig("data/ellipse_scatter.png")
    plt.show()
    
    # Optional: visualize first mesh
    response = input("\nVisualize first mesh? (y/n): ").strip().lower()
    if response == 'y':
        print("Launching Gmsh...")
        visualize_geometry(all_params[0])
