


"""
FEA POST-PROCESSING & HOMOGENIZATION
Data Processing Module for Parametric Simulation Results

This module:
1. Parses CalculiX mesh files (.inp) to extract nodal coordinates and elements
2. Calculates triangular element areas from mesh geometry
3. Parses stress results (.dat) and computes area-weighted homogenization
4. Calculates effective Young's modulus for each simulation case
5. Generates ML-ready CSV datasets (geometric parameters → effective modulus)
6. Creates visualization histograms for parameter distribution analysis
7. Saves rejected/failed simulation data for debugging

Author: Andrea Vinarš

"""

import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def calculate_youngs_modulus(
    width: float, 
    elongation: float, 
    inp_path: str, 
    dat_path: str) -> Optional[float]:
    
    """
    Calculate the effective Young's modulus from FEA simulation results.
    
    This function parses the CalculiX input file (.inp) to extract mesh geometry
    and the output file (.dat) to extract stress results. It computes the 
    homogenized stress using area-weighted averaging and divides by the 
    applied strain to obtain the effective Young's modulus.
    
    Args:
        width: Plate width in mm (used to calculate strain)
        elongation: Applied elongation in mm
        inp_path: Path to the CalculiX input file (.inp)
        dat_path: Path to the CalculiX output file (.dat)
    
    Returns:
        Effective Young's modulus in MPa, or None if stress data is unavailable
    """
    
    def triangle_area(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """
        Calculate the area of a triangle using the shoelace formula.
        
        Args:
            p1, p2, p3: Vertex coordinates as (x, y) tuples
        
        Returns:
            Triangle area (always positive)
        """
        return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
    
    # Dictionaries to store mesh data
    nodes: Dict[int, Tuple[float, float]] = {}      # node_id -> (x, y)
    elements: List[Dict] = []                       # List of element dictionaries
    
    
    # Extract nodes and elements from CalculiX .inp file
   
    with open(inp_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Parse nodes: ID, X, Y coordinates
        if line.startswith('*NODE'):
            i += 1
            while i < len(lines) and not lines[i].startswith('*'):
                parts = lines[i].split(',')
                if len(parts) >= 4:
                    node_id = int(parts[0])
                    x_coord = float(parts[1])
                    y_coord = float(parts[2])
                    nodes[node_id] = (x_coord, y_coord)
                i += 1
            continue
        
        # Parse elements: ID + 6 nodes (only 3 corner nodes needed for area)
        if line.startswith('*ELEMENT'):
            i += 1
            data = []
            while i < len(lines) and not lines[i].startswith('*'):
                parts = [x for x in lines[i].split(',') if x.strip()]
                if parts:
                    data.extend(parts)
                    # Each element has ID + 6 nodes; we only need first 3 for triangular area
                    if len(data) >= 7:
                        elements.append({
                            'id': int(data[0]),
                            'corners': [int(data[1]), int(data[2]), int(data[3])]
                        })
                        data = []
                i += 1
            continue
        
        i += 1
    
    
    # Calculate area for each element
    
    areas: List[Dict] = []
    
    for elem in elements:
        try:
            corners = elem['corners']
            p1 = nodes[corners[0]]
            p2 = nodes[corners[1]]
            p3 = nodes[corners[2]]
            areas.append({
                'element_id': elem['id'],
                'Area_mm2': triangle_area(p1, p2, p3)
            })
        except (KeyError, IndexError):
            # Skip elements with missing nodes
            pass
    
    areas_df = pd.DataFrame(areas)
    
    
    # Parse the .dat file to extract stress results (Sxx component)
   
    stress_data: List[Dict] = []
    reading = False
    
    with open(dat_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Start reading when stress section is found
            if line.startswith('stresses (elem'):
                reading = True
                continue
            
            # Stop reading at next asterisk-delimited section
            if reading and line.startswith('*'):
                break
            
            # Parse stress data lines: element_id, int_pt, Sxx, Syy, Szz, Sxy, Sxz, Syz
            if reading and line:
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        element_id = int(parts[0])
                        sxx = float(parts[2])  # Sxx is the 3rd column (index 2)
                        
                        stress_data.append({
                            'element_id': element_id,
                            'Sxx_MPa': sxx
                        })
                    except (ValueError, IndexError):
                        continue
    
    stress_df = pd.DataFrame(stress_data)
    
    # Return None if no stress data was found
    if stress_df.empty:
        logger.warning(f"No stress data found in {dat_path}")
        return None
    
    # Average stress values over element integration points
    stress_df = stress_df.groupby('element_id')['Sxx_MPa'].mean().reset_index()
    
    
    # Compute homogenized Young's modulus using area-weighted averaging
    
    ml_df = areas_df.merge(stress_df, on='element_id', how='inner')
    
    total_area = ml_df['Area_mm2'].sum()
    weighted_stress = (ml_df['Sxx_MPa'] * ml_df['Area_mm2']).sum()
    homogenized_stress = weighted_stress / total_area

    applied_strain = elongation / width
    
    # Effective Young's modulus = stress / strain
    E_eff = homogenized_stress / applied_strain
    
    return E_eff


def params_csv_histograms(
        
    all_params: List[Dict], 
    output_dir: str) -> None:
    
    """
    Generate CSV dataset and histogram plots from simulation parameters.
    
    Creates a comprehensive dataset with all geometric parameters and their
    trigonometric transformations, plus visualization histograms for each
    parameter distribution.
    
    Args:
        all_params: List of parameter dictionaries from successful simulations
        output_dir: Directory path for saving CSV and PNG files
    """
    
    os.makedirs(output_dir, exist_ok=True)
    data: List[Dict] = []

    # Build dataset with all geometric parameters
    for i, params in enumerate(all_params):
        data.append({
            "SIMULATION": i,
            'x1': params['hole1']['x'],
            'y1': params['hole1']['y'],
            'rx1': params['hole1']['rx'],
            'ry1': params['hole1']['ry'],
            'angle1': params['hole1']['angle'],
            'angle1_sin': round(np.sin(np.radians(params['hole1']['angle'])), 4),
            'angle1_cos': round(np.cos(np.radians(params['hole1']['angle'])), 4),
            'x2': params['hole2']['x'],
            'y2': params['hole2']['y'],
            'rx2': params['hole2']['rx'],
            'ry2': params['hole2']['ry'],
            'angle2': params['hole2']['angle'],
            'angle2_sin': round(np.sin(np.radians(params['hole2']['angle'])), 4),
            'angle2_cos': round(np.cos(np.radians(params['hole2']['angle'])), 4),
            'E_eff': params['E_eff'],
            'avg_quality': params['avg_quality']
        })

    df = pd.DataFrame(data)
    csv_file = os.path.join(output_dir, "ml_data.csv")
    
    # Save to CSV 
    df.to_csv(csv_file, index=False, sep=";", decimal=",")

   
    # Create histogram visualization for all geometric parameters
   
    fig, axes = plt.subplots(2, 5, figsize=(15, 10))
    fig.suptitle("GEOMETRICAL PARAMETERS", fontsize=16, fontweight="bold")

    # Define parameters to plot: (column_name, row, col, color)
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
        ('angle2', 1, 4, 'green')
    ]

    for param_name, row, col, color in params_to_plot:
        axes[row, col].hist(df[param_name], bins=30, color=color, alpha=0.7, edgecolor="black")
        axes[row, col].set_title(f"{param_name} distribution", fontweight="bold")

        # Set appropriate units based on parameter type
        if param_name.startswith("angle"):
            axes[row, col].set_xlabel(f"{param_name} [°]")
        else:
            axes[row, col].set_xlabel(f"{param_name} [mm]")

        axes[row, col].set_ylabel("n")
        axes[row, col].grid(True, alpha=0.5)

    plt.tight_layout()
    stat_file = os.path.join(output_dir, "param_histograms.png")
    plt.savefig(stat_file, dpi=200, bbox_inches="tight")
    logger.info(f"Statistics saved: {stat_file}")

    plt.close()
    

def save_rejected_csv(
        
    rejected_results: List[Dict], 
    output_dir: str) -> None:
    
    """
    Save rejected (skipped/failed/error) simulations to CSV for analysis.
    
    This function enables debugging of problematic geometries and facilitates
    potential re-simulation of failed cases.
    
    Args:
        rejected_results: List of result dictionaries with failed/skipped status
        output_dir: Directory path for saving the rejected simulations CSV
    """
    
    if not rejected_results:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    data: List[Dict] = []

    for result in rejected_results:
        params = result.get('params')
        if not params:
            continue
            
        row = {
            "SIMULATION": result['index'],
            "STATUS": result['status'],
            "REASON": result.get('reason', result.get('error', 'Unknown')),
            'x1': params['hole1']['x'],
            'y1': params['hole1']['y'],
            'rx1': params['hole1']['rx'],
            'ry1': params['hole1']['ry'],
            'angle1': params['hole1']['angle'],
            'angle1_sin': round(np.sin(np.radians(params['hole1']['angle'])), 4),
            'angle1_cos': round(np.cos(np.radians(params['hole1']['angle'])), 4),
            'x2': params['hole2']['x'],
            'y2': params['hole2']['y'],
            'rx2': params['hole2']['rx'],
            'ry2': params['hole2']['ry'],
            'angle2': params['hole2']['angle'],
            'angle2_sin': round(np.sin(np.radians(params['hole2']['angle'])), 4),
            'angle2_cos': round(np.cos(np.radians(params['hole2']['angle'])), 4),
            'avg_quality': params['avg_quality']
        }
        data.append(row)

    if data:
        df = pd.DataFrame(data)
        csv_file = os.path.join(output_dir, "rejected_simulations.csv")
        df.to_csv(csv_file, index=False, sep=";", decimal=",")
        
        logger.info(f"Rejected simulations saved: {csv_file}")
        logger.info(f"  - Total rejected: {len(data)}")
        
        # Print summary by status category
        status_counts = df['STATUS'].value_counts()
        for status, count in status_counts.items():
            logger.info(f"    * {status}: {count}")
    
    plt.close()