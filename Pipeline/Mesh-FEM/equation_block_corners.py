"""Periodic boundary-condition utilities for a rectangular 2D RVE.

The edge mapping is registered with Gmsh before mesh generation. After the
final mesh is accepted, the resulting periodic node pairs are converted to
CalculiX ``*EQUATION`` constraints. Corner-node displacements carry the
prescribed macroscopic deformation for the EX, EY, and XY load cases.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple


def set_periodic_edges_before_mesh(
    surface_tag: int,
    height: float,
    width: float,
    logger: logging.Logger,
    tol: float = 1e-6,
) -> Dict[str, int]:
    """Identify opposite outer edges and register their periodic mapping.

    The rectangular RVE is assumed to span ``[0, width] x [0, height]``.
    Gmsh maps the right edge to the left edge and the top edge to the bottom
    edge using pure translations. The mapping survives mesh regeneration, but
    node-based equations must be rebuilt after the final mesh is generated.

    Args:
        surface_tag: Gmsh tag of the perforated plate surface.
        height: RVE height.
        width: RVE width.
        logger: Logger used for diagnostics.
        tol: Coordinate tolerance used to identify the outer edges.

    Returns:
        Tags of the left, right, bottom, and top edges.
    """

    import gmsh

    edges = gmsh.model.getBoundary(
        [(2, surface_tag)],
        oriented=False,
        recursive=False,
    )

    left_edge = None
    right_edge = None
    top_edge = None
    bottom_edge = None

    for dim, tag in edges:
        if dim != 1:
            continue

        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, tag)

        if abs(xmin) < tol and abs(xmax) < tol:
            left_edge = tag

        if abs(xmin - width) < tol and abs(xmax - width) < tol:
            right_edge = tag

        if abs(ymin) < tol and abs(ymax) < tol:
            bottom_edge = tag

        if abs(ymin - height) < tol and abs(ymax - height) < tol:
            top_edge = tag

    if None in [left_edge, right_edge, bottom_edge, top_edge]:
        message = (
            f"Could not find all outer edges:\n"
            f"left={left_edge}, right={right_edge}, "
            f"bottom={bottom_edge}, top={top_edge}"
        )

        logger.error(message)
        raise RuntimeError(message)

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


def build_equation_block_after_mesh(
    periodic_edges: Dict[str, int],
    height: float,
    width: float,
    logger: logging.Logger,
    tol: float = 1e-6,
) -> Tuple[str, int, int, int, int]:
    """Build CalculiX periodic equations from the final Gmsh mesh.

    Corner convention::

        corner_4 -------- corner_3
           |                  |
           |                  |
        corner_1 -------- corner_2

    Opposite-edge fluctuations are constrained periodically. The corner
    differences ``u2 - u1`` and ``u4 - u1`` represent the macroscopic
    displacement jumps and are prescribed later for each load case.

    High-order edge nodes are included so the equations remain complete for
    second-order meshes. This function must therefore be called only after the
    final mesh, including any fallback remeshing, has been generated.

    Returns:
        The CalculiX equation block followed by corner tags 1 through 4.
    """

    import gmsh

    right_edge = periodic_edges["right_edge"]
    top_edge = periodic_edges["top_edge"]

    # The third argument includes midside nodes on second-order boundary edges.
    tagM_lr, right_nodes, left_nodes, _ = gmsh.model.mesh.getPeriodicNodes(
        1, right_edge, True
    )
    tagM_tb, top_nodes, bottom_nodes, _ = gmsh.model.mesh.getPeriodicNodes(
        1, top_edge, True
    )

    logger.info(f"LR periodic master: {tagM_lr}, number of pairs: {len(right_nodes)}")
    logger.info(f"TB periodic master: {tagM_tb}, number of pairs: {len(top_nodes)}")

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords.reshape(-1, 3)

    node_xy: Dict[int, Tuple[float, float]] = {}

    for i, tag in enumerate(node_tags):
        node_xy[int(tag)] = (float(node_coords[i, 0]), float(node_coords[i, 1]))

    def find_node_at(
        x_target: float,
        y_target: float,
        coordinates: Dict[int, Tuple[float, float]],
    ) -> int:
        for node, (x, y) in coordinates.items():
            if abs(x - x_target) < tol and abs(y - y_target) < tol:
                return node

        message = f"Could not find node at ({x_target}, {y_target})"
        logger.error(message)
        raise RuntimeError(message)

    corner_1 = find_node_at(0.0, 0.0, node_xy)
    corner_2 = find_node_at(width, 0.0, node_xy)
    corner_3 = find_node_at(width, height, node_xy)
    corner_4 = find_node_at(0.0, height, node_xy)

    logger.info(f"[NODES] corner_1: {corner_1}")
    logger.info(f"[NODES] corner_2: {corner_2}")
    logger.info(f"[NODES] corner_3: {corner_3}")
    logger.info(f"[NODES] corner_4: {corner_4}")

    corner_1 = int(corner_1)
    corner_2 = int(corner_2)
    corner_3 = int(corner_3)
    corner_4 = int(corner_4)

    eq_lines = []

    def add_equation(terms):
        eq_lines.append("*EQUATION")
        eq_lines.append(str(len(terms)))
        eq_lines.append(
            ", ".join(
                f"{int(node)},{int(dof)},{float(coef)}" for node, dof, coef in terms
            )
        )

    # Right-left periodicity: uR - uL - u2 + u1 = 0.

    corner_lr_pairs = {
        (corner_2, corner_1),
        (corner_3, corner_4),
    }

    for right_node, left_node in zip(right_nodes, left_nodes, strict=True):
        nr = int(right_node)
        nl = int(left_node)

        if (nr, nl) in corner_lr_pairs:
            continue

        add_equation(
            [
                (nr, 1, 1.0),
                (nl, 1, -1.0),
                (corner_2, 1, -1.0),
                (corner_1, 1, 1.0),
            ]
        )

        add_equation(
            [
                (nr, 2, 1.0),
                (nl, 2, -1.0),
                (corner_2, 2, -1.0),
                (corner_1, 2, 1.0),
            ]
        )

    # Top-bottom periodicity: uT - uB - u4 + u1 = 0.

    corner_tb_pairs = {
        (corner_4, corner_1),
    }

    for top_node, bottom_node in zip(top_nodes, bottom_nodes, strict=True):
        nt = int(top_node)
        nb = int(bottom_node)

        if (nt, nb) in corner_tb_pairs:
            continue

        add_equation(
            [
                (nt, 1, 1.0),
                (nb, 1, -1.0),
                (corner_4, 1, -1.0),
                (corner_1, 1, 1.0),
            ]
        )

        add_equation(
            [
                (nt, 2, 1.0),
                (nb, 2, -1.0),
                (corner_4, 2, -1.0),
                (corner_1, 2, 1.0),
            ]
        )

    return "\n".join(eq_lines), corner_1, corner_2, corner_3, corner_4
