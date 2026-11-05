"""Wrap the extruded front frame around the face.

The wrap is a shear along Z only:

    (x, y, z) -> (x, y, z + f(x)),   f(x) = -depth * (x / half_width)^2

the same curve glasses_3d.py uses for its lenses. `depth` is how far the outer
edges sit behind the centre, in cm. A shear suits this job because:
  - X and Y don't change, so the front view is exactly the outline from the photo;
  - it is one-to-one, so it cannot make the surface pass through itself;
  - front and rear surfaces stay `thickness` apart along Z.

A mesh only bends at its vertices, so faces are first split at evenly spaced
planes of constant X, close enough together to keep every chord within
CHORD_TOLERANCE_CM of the curve.
"""
import math

import numpy as np
import shapely
import trimesh

from image_to_3d.errors import MeshBuildError

CHORD_TOLERANCE_CM = 0.005    # largest gap between a face and the true curve
ON_PLANE_TOLERANCE_CM = 1e-3  # closer than this counts as on a plane, avoiding slivers


def wrap_offset(x, frame_width, depth):
    """Z offset of the frame surface at x; 0 at the centre, -depth at the outer edges."""
    return -depth * (2 * np.asarray(x) / frame_width) ** 2


def split_faces_at_x(vertices, faces, plane_x):
    """Split faces crossing the plane x = plane_x, keeping winding and shared edges."""
    side = np.sign(vertices[:, 0] - plane_x)
    side[np.abs(vertices[:, 0] - plane_x) < ON_PLANE_TOLERANCE_CM] = 0
    face_sides = side[faces]
    crossing = (face_sides.max(axis=1) > 0) & (face_sides.min(axis=1) < 0)
    if not crossing.any():
        return vertices, faces

    new_vertices = []
    edge_points = {}  # keyed by undirected edge, so neighbouring faces share the new vertex

    def point_on_edge(a, b):
        key = (min(a, b), max(a, b))
        if key not in edge_points:
            start, end = vertices[key[0]], vertices[key[1]]
            point = start + (end - start) * (plane_x - start[0]) / (end[0] - start[0])
            point[0] = plane_x
            edge_points[key] = len(vertices) + len(new_vertices)
            new_vertices.append(point)
        return edge_points[key]

    new_faces = []
    for a, b, c in faces[crossing].tolist():
        sides = (side[a], side[b], side[c])
        if 0 in sides:  # one vertex on the plane: two faces
            on_plane = sides.index(0)
            a, b, c = (a, b, c)[on_plane:] + (a, b, c)[:on_plane]
            middle = point_on_edge(b, c)
            new_faces += [(a, b, middle), (a, middle, c)]
        else:  # one vertex alone on its side: three faces
            alone = next(i for i in range(3)
                         if sides[i] != sides[i - 1] and sides[i] != sides[i - 2])
            a, b, c = (a, b, c)[alone:] + (a, b, c)[:alone]
            ab, ca = point_on_edge(a, b), point_on_edge(c, a)
            new_faces += [(a, ab, ca), (ab, b, c), (ab, c, ca)]
    return (np.vstack((vertices, new_vertices)),
            np.vstack((faces[~crossing], new_faces)))


def _split_positions(frame_width, depth):
    """Planes close enough that a chord of the parabola sags at most the tolerance.

    A chord of width w on z = -a x^2 sags a w^2 / 4 at its middle, wherever it is.
    """
    curve = depth / (frame_width / 2) ** 2
    spacing = math.sqrt(4 * CHORD_TOLERANCE_CM / curve)
    count = math.ceil(frame_width / spacing)
    return [frame_width * (i / count - 0.5) for i in range(1, count)]


def _extrude_polygon(polygon, thickness):
    """Prism from Z = 0 to Z = thickness, with Delaunay caps.

    Traced outlines have many vertices on the same pixel rows. Earcut can join three
    of those into a zero-area triangle; constrained Delaunay avoids that and slivers.
    """
    triangles = shapely.constrained_delaunay_triangles(polygon)
    corners = shapely.get_coordinates(triangles).reshape(-1, 4, 2)[:, :3]  # rings repeat a point
    vertices, indices = np.unique(corners.reshape(-1, 2), axis=0, return_inverse=True)
    return trimesh.creation.extrude_triangulation(vertices, indices.reshape(-1, 3), thickness)


def build_curved_frame(polygons_cm, thickness, depth):
    """Front frame mesh: front surface at Z = f(x), rear surface `thickness` behind it."""
    mesh = trimesh.util.concatenate([_extrude_polygon(polygon, thickness)
                                     for polygon in polygons_cm])
    vertices = mesh.vertices.copy()
    vertices[:, 2] -= thickness
    faces = mesh.faces
    frame_width = np.ptp(vertices[:, 0])
    if depth > 0:
        for plane_x in _split_positions(frame_width, depth):
            vertices, faces = split_faces_at_x(vertices, faces, plane_x)
        vertices[:, 2] += wrap_offset(vertices[:, 0], frame_width, depth)

    curved = trimesh.Trimesh(vertices, faces)
    if not curved.is_volume:
        raise MeshBuildError("The curved front frame is not a closed, consistent volume")
    return curved
