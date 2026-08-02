



"""Validate the periodic homogenization and FNN prediction for one geometry.

The script creates a rectangular plate with two fixed elliptical holes,
registers periodic edge mappings, and generates a second-order Gmsh mesh. The
same mesh is reused for the EX, EY, and XY CalculiX load cases. Volume-averaged
stresses form the homogenized in-plane stiffness matrix, whose inverse yields
``E_x``, ``E_y``, ``G_xy``, and ``NU_xy``. The final section compares these FEM
properties with the trained feed-forward neural-network surrogate.

Author: Andrea Vinarš
"""

import gmsh
import os
import numpy as np
import subprocess
import pandas as pd
from pathlib import Path

from predict_fixed import load_surrogate_model, predict_custom


def set_periodic_edges_before_mesh(surface_tag: int,
                   height: int,
                   width: int,
                   tol = 1e-6):
    """Identify opposite outer edges and register their Gmsh periodic maps.

    This function must run before mesh generation so that Gmsh creates matching
    node pairs on the left-right and bottom-top boundaries.
    """

    import gmsh

    edges = gmsh.model.getBoundary([(2, surface_tag)],
                                   oriented=False,
                                   recursive=False)

    left_edge = None
    right_edge = None
    top_edge = None
    bottom_edge = None

    for dim, tag in edges:
        if dim != 1:
            continue

        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, tag)

        if abs(xmin) < tol and abs(xmax) < tol:
            left_edge = tag

        if abs(xmin - width) < tol and abs(xmax - width) < tol:
            right_edge = tag

        if abs(ymin) < tol and abs(ymax) < tol:
            bottom_edge = tag

        if abs(ymin - height) < tol and abs(ymax - height) < tol:
            top_edge = tag

    if None in [left_edge, right_edge, bottom_edge, top_edge]:
        raise RuntimeError(
        print(
            f"Could not find all outer edges:\n"
            f"left={left_edge}, right={right_edge}, "
            f"bottom={bottom_edge}, top={top_edge}"))


    aff_lr = [
              1, 0, 0, width,
              0, 1, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 1,
              ]

    aff_tb = [
              1, 0, 0, 0,
              0, 1, 0, height,
              0, 0, 1, 0,
              0, 0, 0, 1,
              ]

    gmsh.model.mesh.setPeriodic(1, [right_edge], [left_edge], aff_lr)
    gmsh.model.mesh.setPeriodic(1, [top_edge], [bottom_edge], aff_tb)

    return {
        "left_edge": left_edge,
        "right_edge": right_edge,
        "bottom_edge": bottom_edge,
        "top_edge": top_edge,
    }


def build_equation_block_after_mesh(periodic_edges: dict,
                                    height: float,
                                    width: float,
                                    tol=1e-6):
    """Build CalculiX periodic equations from the final Gmsh node pairs.

    Returns the serialized ``*EQUATION`` block followed by the bottom-left,
    bottom-right, top-right, and top-left corner-node tags.
    """

    import gmsh

    right_edge = periodic_edges["right_edge"]
    top_edge = periodic_edges["top_edge"]

    tagM_lr, right_nodes, left_nodes, _ = gmsh.model.mesh.getPeriodicNodes(1, right_edge)
    tagM_tb, top_nodes, bottom_nodes, _ = gmsh.model.mesh.getPeriodicNodes(1, top_edge)

    print(f"LR periodic master: {tagM_lr}, number of pairs: {len(right_nodes)}")
    print(f"TB periodic master: {tagM_tb}, number of pairs: {len(top_nodes)}")

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords.reshape(-1, 3)

    node_xy = {}

    for i, tag in enumerate(node_tags):
        node_xy[int(tag)] = (float(node_coords[i, 0]),
                             float(node_coords[i, 1]))

    def find_node_at(x_target, y_target, node_xy, tol=1e-6):
        """Return the node located at the requested coordinates."""

        for node, (x, y) in node_xy.items():
            if abs(x - x_target) < tol and abs(y - y_target) < tol:
                return node

        msg = f"Could not find node at ({x_target}, {y_target})"
        print(msg)
        raise RuntimeError(msg)

    corner_1 = find_node_at(0.0, 0.0, node_xy)
    corner_2 = find_node_at(width, 0.0, node_xy)
    corner_3 = find_node_at(width, height, node_xy)
    corner_4 = find_node_at(0.0, height, node_xy)

    print(f"[NODES] corner_1: {corner_1}")
    print(f"[NODES] corner_2: {corner_2}")
    print(f"[NODES] corner_3: {corner_3}")
    print(f"[NODES] corner_4: {corner_4}")

    """
    Periodic boundary conditions for a rectangular 2D RVE.

    Corner convention:
        corner_1 = bottom-left
        corner_2 = bottom-right
        corner_3 = top-right
        corner_4 = top-left

    Periodic constraints:
        uR = uL + u2 - u1
        uT = uB + u4 - u1

    The macroscopic deformation is prescribed separately through
    the displacements of corner_2 and corner_4.
    """

    corner_1 = int(corner_1)
    corner_2 = int(corner_2)
    corner_3 = int(corner_3)
    corner_4 = int(corner_4)

    eq_lines = []

    def add_equation(terms):
        """Append one CalculiX equation from node, DOF, and coefficient terms."""

        eq_lines.append("*EQUATION")
        eq_lines.append(str(len(terms)))
        eq_lines.append(", ".join(
            f"{int(node)},{int(dof)},{float(coef)}"
            for node, dof, coef in terms
        ))

    # ------------------------------------------------------------
    # Right-left periodicity:
    #
    # uR = uL + u2 - u1
    # therefore:
    # uR - uL - u2 + u1 = 0
    # ------------------------------------------------------------

    corner_lr_pairs = {
        (corner_2, corner_1),
        (corner_3, corner_4),
    }

    for nr, nl in zip(right_nodes, left_nodes):
        nr = int(nr)
        nl = int(nl)

        if (nr, nl) in corner_lr_pairs:
            continue

        # ux_R - ux_L - ux_2 + ux_1 = 0
        add_equation([
            (nr,       1,  1.0),
            (nl,       1, -1.0),
            (corner_2, 1, -1.0),
            (corner_1, 1,  1.0),
        ])

        # uy_R - uy_L - uy_2 + uy_1 = 0
        add_equation([
            (nr,       2,  1.0),
            (nl,       2, -1.0),
            (corner_2, 2, -1.0),
            (corner_1, 2,  1.0),
        ])

    # ------------------------------------------------------------
    # Top-bottom periodicity:
    #
    # uT = uB + u4 - u1
    # therefore:
    # uT - uB - u4 + u1 = 0
    # ------------------------------------------------------------

    corner_tb_pairs = {
        (corner_4, corner_1),
    }

    for nt, nb in zip(top_nodes, bottom_nodes):
        nt = int(nt)
        nb = int(nb)

        if (nt, nb) in corner_tb_pairs:
            continue

        # ux_T - ux_B - ux_4 + ux_1 = 0
        add_equation([
            (nt,       1,  1.0),
            (nb,       1, -1.0),
            (corner_4, 1, -1.0),
            (corner_1, 1,  1.0),
        ])

        # uy_T - uy_B - uy_4 + uy_1 = 0
        add_equation([
            (nt,       2,  1.0),
            (nb,       2, -1.0),
            (corner_4, 2, -1.0),
            (corner_1, 2,  1.0),
        ])

    return "\n".join(eq_lines), corner_1, corner_2, corner_3, corner_4

# ========== RUNTIME CONFIGURATION ==========
# Keep all generated files next to this fixed-geometry validation script.
DEFAULT_WORKDIR = Path(__file__).resolve().parent

# Set CALCULIX_CMD when the CalculiX executable is not available on PATH.
# PowerShell example: $env:CALCULIX_CMD = 'C:/path/to/ccx_static.exe'
DEFAULT_CALCULIX_CMD = os.environ.get("CALCULIX_CMD") or ("ccx_static.exe" if os.name == "nt" else "ccx")

# Interactive viewers remain disabled during normal batch execution.
SHOW_GUI = False
SHOW_CGX = False

# ========== MACROSCOPIC STRAINS ==========
load_case = "EX" # Supported load-case labels are EX, EY, and XY.
eps0 = 0.001
gamma0 = 0.001

# ========== GEOMETRY PARAMETERS ==========
# Plate dimensions [mm].
width, height = 12, 12

# Ellipse centers and semi-axes are specified in millimetres.

# Hole 1.
x1, y1 = 3.93, 5.37        # Center coordinates [mm]
radX1, radY1 = 1.56, 1.27  # Semi-major and semi-minor axes [mm]
angle1 = 130               # Rotation angle [degrees]

# Hole 2.
x2, y2 = 4.55, 5.44           # Center coordinates [mm]
radX2, radY2 =1.64, 1.07  # Semi-major and semi-minor axes [mm]
angle2 = 237            # Rotation angle [degrees]


"""
Optional example for loading one fixed geometry from a rejected-simulation CSV:

df = pd.read_csv("data/rejected_simulations.csv", sep = ";", decimal = ",")
X = df[['x1', 'y1', 'rx1', 'ry1', 'angle1',
        'x2', 'y2', 'rx2', 'ry2', 'angle2']]

row = X.iloc[1]
x1, y1, radX1, radY1 = row['x1'], row['y1'], row['rx1'], row['ry1']
x2, y2, radX2, radY2 = row['x2'], row['y2'], row['rx2'], row['ry2']
angle1 = row['angle1']
angle2 = row['angle2']
"""

# ========== CALCULIX POST-PROCESSING ==========
def _parse_calculix_float(value):
    """Parse standard and Fortran-style scientific notation."""

    return float(value.replace("D", "E").replace("d", "e"))


def calculate_C(dat_path, output="results.csv"):
    """
    Read the latest CalculiX result increment and return homogenized stress.

        sigma_hom = [sigma_x, sigma_y, tau_xy]

    Integration-point stresses are averaged for each element and weighted by
    the corresponding element volume. The merged element data are written to
    ``output`` for inspection.
    """

    stress_data = []
    volume_data = []

    reading_stress = False
    reading_volume = False

    with open(dat_path, "r", encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            lower = line.lower()

            if "stresses" in lower and "(elem" in lower:
                # Keep only the latest result increment in the file.
                stress_data = []
                volume_data = []
                reading_stress = True
                reading_volume = False
                continue

            if "volume" in lower and "(element" in lower:
                volume_data = []
                reading_stress = False
                reading_volume = True
                continue

            if " for set " in lower and " and time " in lower:
                # Another CalculiX result header ends the active section.
                reading_stress = False
                reading_volume = False
                continue

            if not line:
                continue

            if line.startswith("*"):
                reading_stress = False
                reading_volume = False
                continue

            if reading_stress:
                parts = line.split()

                try:
                    element_id = int(parts[0])
                except (IndexError, ValueError):
                    continue

                if len(parts) < 8:
                    raise RuntimeError(
                        f"Invalid stress data in {dat_path}, line {line_number}: {line}"
                    )

                try:
                    sxx = _parse_calculix_float(parts[2])
                    syy = _parse_calculix_float(parts[3])
                    sxy = _parse_calculix_float(parts[5])
                except ValueError as exc:
                    raise RuntimeError(
                        f"Invalid stress data in {dat_path}, line {line_number}: {line}"
                    ) from exc

                stress_data.append(
                    {
                        "element_id": element_id,
                        "Sxx_MPa": sxx,
                        "Syy_MPa": syy,
                        "Sxy_MPa": sxy,
                    }
                )

            if reading_volume:
                parts = line.split()

                try:
                    element_id = int(parts[0])
                except (IndexError, ValueError):
                    continue

                if len(parts) != 2:
                    raise RuntimeError(
                        f"Invalid volume data in {dat_path}, line {line_number}: {line}"
                    )

                try:
                    elem_volume = _parse_calculix_float(parts[1])
                except ValueError as exc:
                    raise RuntimeError(
                        f"Invalid volume data in {dat_path}, line {line_number}: {line}"
                    ) from exc

                volume_data.append(
                    {
                        "element_id": element_id,
                        "element_volume": elem_volume,
                    }
                )

    stress_df = pd.DataFrame(stress_data)
    volume_df = pd.DataFrame(volume_data)

    if stress_df.empty:
        raise RuntimeError(f"No stress data found in {dat_path}")

    if volume_df.empty:
        raise RuntimeError(f"No volume data found in {dat_path}")

    stress_columns = ["Sxx_MPa", "Syy_MPa", "Sxy_MPa"]

    if not np.isfinite(stress_df[stress_columns].to_numpy()).all():
        raise RuntimeError(f"Stress data contains non-finite values in {dat_path}")

    if not np.isfinite(volume_df["element_volume"].to_numpy()).all():
        raise RuntimeError(f"Volume data contains non-finite values in {dat_path}")

    if volume_df["element_id"].duplicated().any():
        raise RuntimeError(f"Duplicate element volumes found in {dat_path}")

    # Average integration-point stresses before applying element-volume weights.
    stress_df = stress_df.groupby("element_id")[stress_columns].mean().reset_index()

    stress_ids = set(stress_df["element_id"])
    volume_ids = set(volume_df["element_id"])

    if stress_ids != volume_ids:
        raise RuntimeError(
            f"Inconsistent stress and volume data in {dat_path}: "
            f"{len(stress_ids - volume_ids)} elements without volume and "
            f"{len(volume_ids - stress_ids)} elements without stress"
        )

    df = volume_df.merge(stress_df, on="element_id", how="inner")

    if df.empty:
        raise RuntimeError("Stress and volume data could not be merged.")

    total_volume = df["element_volume"].sum()

    if total_volume <= 0:
        raise RuntimeError("Total element volume is zero or negative.")

    sigma_x = (df["Sxx_MPa"] * df["element_volume"]).sum() / total_volume
    sigma_y = (df["Syy_MPa"] * df["element_volume"]).sum() / total_volume
    tau_xy = (df["Sxy_MPa"] * df["element_volume"]).sum() / total_volume

    sigma_hom = np.array([sigma_x, sigma_y, tau_xy], dtype=float)

    df.to_csv(output, index=False, sep=";", decimal=",")

    return sigma_hom

def geometry():

    """Build the fixed geometry and return its reusable periodic mesh data."""

    os.chdir(DEFAULT_WORKDIR)
    calculix_path = DEFAULT_CALCULIX_CMD
    print(f"[CONFIG] Working directory: {DEFAULT_WORKDIR}")
    print(f"[CONFIG] CalculiX command: {calculix_path}")

    try:
        # ========== GMSH INITIALIZATION ==========
        gmsh.initialize()
        gmsh.model.add("tensile_plate_fea")

        # ========== GEOMETRY CREATION ==========
        # Construct the rectangular RVE boundary.
        model = gmsh.model.occ

        n1 = model.addPoint(x = 0, y = 0, z = 0)
        n2 = model.addPoint(x = width, y = 0, z = 0)
        n3 = model.addPoint(x = width, y = height, z = 0)
        n4 = model.addPoint(x = 0, y = height, z = 0)

        l1 = model.addLine(n1, n2)
        l2 = model.addLine(n2, n3)
        l3 = model.addLine(n3, n4)
        l4 = model.addLine(n4, n1)

        loop = model.addCurveLoop([l1, l2, l3, l4])
        plate = model.addPlaneSurface([loop])

        model.synchronize()

        # Create and rotate the first elliptical hole.
        e1_c = model.addEllipse(x1, y1, 0, radX1, radY1)
        model.rotate([(1, e1_c)], x1, y1, 0, 0, 0, 1, angle1*np.pi/180)
        e1_cl = model.addCurveLoop([e1_c])
        hole1 = model.addPlaneSurface([e1_cl])

        # Create and rotate the second elliptical hole.
        e2_c = model.addEllipse(x2, y2, 0, radX2, radY2)
        model.rotate([(1, e2_c)], x2, y2, 0, 0, 0, 1, angle2*np.pi/180)
        e2_cl = model.addCurveLoop([e2_c])
        hole2 = model.addPlaneSurface([e2_cl])

        # Auxiliary center points control distance-based mesh refinement.
        center1 = model.addPoint(x1, y1, 0)
        center2 = model.addPoint(x2, y2, 0)
        print(f"[MESH] Center points added: {center1}, {center2}")

        gmsh.model.occ.synchronize()

        # Subtract both holes from the rectangular plate.
        cut_result = model.cut([(2, plate)], [(2, hole1), (2, hole2)])
        new_plate_tag = cut_result[0][0][1]
        print("[GEOMETRY] Cut operation completed")

        gmsh.model.occ.synchronize()

        # Register opposite-edge periodic mappings before generating nodes.
        periodic_edges = set_periodic_edges_before_mesh(surface_tag=new_plate_tag,
                                                        height=height,
                                                        width=width,
                                                        tol=1e-6)

        # ========== ADAPTIVE MESH REFINEMENT ==========

        """
        Strategy: Create a mesh size field that varies with distance to ellipse curves
        - Fine mesh (SizeMin) near hole boundaries for accurate stress capture
        - Coarse mesh (SizeMax) far from holes to reduce total element count
        - Smooth transition between fine and coarse regions
        """

        # Distance from the first ellipse center.
        field1 = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field1, "PointsList", [center1])
        print("[MESH] Distance field 1 created for ellipse 1 center")

        # Distance from the second ellipse center.
        field2 = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field2, "PointsList", [center2])
        print("[MESH] Distance field 2 created for ellipse 2 center")

        # Use the smaller distance to either center.
        field3 = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(field3, "FieldsList", [field1, field2])
        print("[MESH] Min field 3 created (combines both distances)")

        min_radius = min(radY1, radY2)
        max_radius = max(radX1, radX2)

        # Convert the distance field into near-hole and far-field element sizes.
        field4 = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field4, "InField", field3)
        gmsh.model.mesh.field.setNumber(field4, "SizeMin", 0.1)   # Fine mesh size [mm]
        gmsh.model.mesh.field.setNumber(field4, "SizeMax", 0.5)   # Coarse mesh size [mm]
        gmsh.model.mesh.field.setNumber(field4, "DistMin", min_radius)   # Transition start distance [mm]
        gmsh.model.mesh.field.setNumber(field4, "DistMax", max_radius * 3)   # Transition end distance [mm]
        print("[MESH] Threshold field 4 created")

        # Apply the threshold field as the global background sizing field.
        gmsh.model.mesh.field.setAsBackgroundMesh(field4)

        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # algorithm id is version-dependent in Gmsh
        gmsh.option.setNumber("Mesh.Smoothing", 3)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)  # avoid fighting the sizing field

        # Generate and smooth the 2D mesh.
        gmsh.model.mesh.generate(2)

        gmsh.model.mesh.optimize("Laplace2D")

        # Convert the mesh to second-order triangular elements.
        gmsh.model.mesh.setOrder(2)

        equation_block, corner_1, corner_2, corner_3, corner_4 = build_equation_block_after_mesh(periodic_edges=periodic_edges,
                                                                                                 height=height,
                                                                                                 width=width,
                                                                                                 tol=1e-6)

        # Name the plate surface so Gmsh exports the CalculiX element set Plate.
        gmsh.model.addPhysicalGroup(2, [new_plate_tag], tag=3, name="Plate")

        # Export the accepted mesh in CalculiX input format.
        gmsh.write("tensile_FEA_mesh.inp")
        print("[GMSH] Mesh file saved: tensile_FEA_mesh.inp")

        # Open the Gmsh viewer only during interactive inspection.
        if SHOW_GUI:
            gmsh.fltk.run()

        # ========== MESH EXPORT POST-PROCESSING ==========

        """
        Read mesh file written by Gmsh and patch it to include NSET=NALL
        so that CalculiX writes nodal results for all nodes.
        """

        with open("tensile_FEA_mesh.inp", "r", encoding="utf-8", errors="ignore") as f:
            mesh_content = f.read()

        mesh_content = mesh_content.replace("*NODE", "*NODE, NSET=NALL", 1)

        # The mesh content is now held in memory, so remove the temporary file.
        if os.path.exists("tensile_FEA_mesh.inp"):
            os.remove("tensile_FEA_mesh.inp")

        return mesh_content, corner_1, corner_2, corner_3, corner_4, equation_block

    except FileExistsError:
        print("File not found")


def make_input_file(base_data, load_case):
    """Create a CalculiX input deck with periodic constraints for one load case."""

    mesh_content, corner_1, corner_2, corner_3, corner_4, eq_block = base_data

    job_name = f"tensile_FEA_{load_case}"
    inp_path = f"{job_name}.inp"
    dat_path = f"{job_name}.dat"

    u2x = 0.0
    u2y = 0.0
    u4y = 0.0

    if load_case == "EX":

        u2x = eps0 * width

    elif load_case == "EY":

        u4y = eps0 * height

    elif load_case == "XY":

        u2y = gamma0 * width


    calculix_input = f"""{mesh_content}

{eq_block}

*MATERIAL, NAME=Steel
*ELASTIC
210000, 0.3

*SOLID SECTION, ELSET=Plate, MATERIAL=Steel
1.0

*STEP, NLGEOM=NO
*STATIC

** BOUNDARY CONDITIONS
*BOUNDARY
{corner_1}, 1, 3, 0.0
{corner_4}, 1, 1, 0.0

{corner_2}, 1, 1, {u2x}
{corner_2}, 2, 2, {u2y}

{corner_4}, 2, 2, {u4y}


** OUTPUT REQUESTS
*EL PRINT, ELSET=Plate, GLOBAL=YES
S, EVOL

*EL FILE
S, E

*NODE FILE, NSET=NALL
U

*END STEP
"""

    with open(inp_path, "w") as f:
        f.write(calculix_input)

    print(f"[CalculiX] Input file created: {inp_path}")

    return job_name, inp_path, dat_path, load_case


def run_simulation(base_model, load_case):
    """Run one CalculiX load case and return its homogenized stress vector.

    Args:
        base_model: Mesh data and periodic equations returned by ``geometry``.
        load_case: Macroscopic deformation case: ``EX``, ``EY``, or ``XY``.

    Returns:
        The homogenized stress vector ``[sigma_x, sigma_y, tau_xy]``.
    """

    # Create a dedicated input deck while reusing the same periodic mesh.
    job_name, inp_path, dat_path, load_case = make_input_file(base_model, load_case)


    print(f"\n[RUN] Load case: {load_case}")
    print(f"[RUN] Job name: {job_name}")
    print(f"[RUN] Input file: {inp_path}")

    try:
        result = subprocess.run(
            [DEFAULT_CALCULIX_CMD, job_name],
            capture_output=True,
            text=True,
            timeout=60)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"CalculiX executable not found: {DEFAULT_CALCULIX_CMD}"
        )

    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"CalculiX simulation timed out for {job_name}"
        )

    if result.returncode != 0:
        print("\n[CalculiX ERROR]")
        print("Return code:", result.returncode)

        print("\n--- STDOUT ---")
        print(result.stdout)

        print("\n--- STDERR ---")
        print(result.stderr)

    print("[CalculiX] Analysis completed successfully")

    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"Expected result file not found: {dat_path}")

    sigma_hom = calculate_C(dat_path=dat_path, output=f"{job_name}_results.csv")

    print(f"[HOMOGENIZATION] {load_case}: sigma_hom = {sigma_hom}")

    if SHOW_CGX:
        frd_path = f"{job_name}.frd"
        CGX_PATH = r"C:/Users/andrea/Desktop/calculix_2.23_4win/calculix_2.23_4win/cgx_STATIC.exe"

        if os.path.exists(frd_path):
            subprocess.run([CGX_PATH, frd_path])

        else:
            print(f"[CGX] FRD file not found: {frd_path}")

    return sigma_hom

def compute_C_matrix():
    """Compute the homogenized stiffness matrix and engineering constants.

    Load cases:
        ``EX`` produces the first column of ``C``.
        ``EY`` produces the second column of ``C``.
        ``XY`` produces the third column of ``C``.

    Returns:
        The 3x3 stiffness matrix in MPa and a dictionary of effective
        engineering constants for comparison with the surrogate model.
    """

    print("\n[START] Computing homogenized stiffness matrix C")

    # Generate one mesh and preserve identical discretization across load cases.
    base_data = geometry()

    # Solve the three independent macroscopic deformation states.
    sigma_EX = run_simulation(base_data, "EX")
    sigma_EY = run_simulation(base_data, "EY")
    sigma_xy = run_simulation(base_data, "XY")

    # Normalize each stress response to assemble one stiffness-matrix column.
    C = np.column_stack([
        sigma_EX / eps0,
        sigma_EY / eps0,
        sigma_xy / gamma0
    ])

    C_df = pd.DataFrame(
        C,
        index=["sigma_x", "sigma_y", "tau_xy"],
        columns=["epsilon_x", "epsilon_y", "gamma_xy"]
    )

    print("="*70)
    print("\n[HOMOGENIZED CONSTITUTIVE MATRIX C] [MPa] \n")
    print(C_df)

    # Save the assembled stiffness matrix for inspection or external use.
    C_df.to_csv(
        "homogenized_C_matrix.csv",
        sep=";",
        decimal=","
    )

    # Small off-diagonal differences quantify numerical symmetry error.
    print("\n[SYMMETRY CHECK]")
    print(f"C12 - C21 = {C[0, 1] - C[1, 0]:.2f}")
    print(f"C16 - C61 = {C[0, 2] - C[2, 0]:.2f}")
    print(f"C26 - C62 = {C[1, 2] - C[2, 1]:.2f}")

    S = np.linalg.inv(C)

    E_x = 1 / S[0, 0]
    E_y = 1 / S[1, 1]
    G_xy = 1 / S[2, 2]
    NU_xy = - S[0, 1] / S[0, 0]

    print(f"Young's modulus in the x-direction: {E_x :.2f} MPa")
    print(f"Young's modulus in the y-direction: {E_y :.2f} MPa")
    print(f"Shear modulus in the xy-plane: {G_xy :.2f} MPa")
    print(f"Poisson's ratio: {NU_xy :.3f}")
    print("")

    fem_result = {
        "E_x [GPa]": E_x / 1000.0,
        "E_y [GPa]": E_y / 1000.0,
        "G_xy [GPa]": G_xy / 1000.0,
        "NU_xy [-]": NU_xy,
    }

    return C, fem_result

if __name__ == "__main__":

    C, fem_result = compute_C_matrix()

    model, scaler_X, scaler_y, device = load_surrogate_model()

    surrogate_result = predict_custom(
        model=model,
        scaler_X=scaler_X,
        scaler_y=scaler_y,

        x1=x1,
        y1=y1,
        rx1=radX1,
        ry1=radY1,
        angle1_deg=angle1,

        x2=x2,
        y2=y2,
        rx2=radX2,
        ry2=radY2,
        angle2_deg=angle2,

        device=device
    )

    print("=" * 70)
    print("\n[COMPARISON: FEM vs SURROGATE MODEL]\n")

    for key in ["E_x [GPa]", "E_y [GPa]", "G_xy [GPa]", "NU_xy [-]"]:
        fem_value = fem_result[key]
        pred_value = surrogate_result[key]

        error = pred_value - fem_value
        rel_error = abs(error) / abs(fem_value) * 100

        print(
            f"{key:10s} | FEM = {fem_value:12.6f} | "
            f"Prediction = {pred_value:12.6f} | "
            f"Error = {error:12.6f} | "
            f"Rel. error = {rel_error:8.3f} %"
        )
