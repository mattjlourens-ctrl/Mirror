"""Check generated meshes: topology of each component and contact between components.

Written with NumPy alone because Trimesh's collision and ray helpers need extra
native packages. Intersection here means surfaces crossing: a mesh wholly inside
another doesn't count.
"""
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

DEGENERATE_AREA_CM2 = 1e-9
HIT_EPSILON = 1e-9
EDGE_BATCH = 128  # edges tested at once, which bounds the memory of the pair matrix


def _segments_hit_triangles(starts, ends, triangles):
    """Moller-Trumbore, pairwise: does each open segment cross its triangle's interior?"""
    directions = ends - starts
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(directions, edge_2)
    determinant = (edge_1 * h).sum(axis=1)
    usable = np.abs(determinant) > 1e-14  # parallel segments don't count as crossing
    inverse = 1.0 / determinant[usable]
    from_corner = starts[usable] - triangles[usable, 0]
    u = inverse * (from_corner * h[usable]).sum(axis=1)
    q = np.cross(from_corner, edge_1[usable])
    v = inverse * (q * directions[usable]).sum(axis=1)
    t = inverse * (q * edge_2[usable]).sum(axis=1)
    return ((u > HIT_EPSILON) & (v > HIT_EPSILON) & (u + v < 1 - HIT_EPSILON)
            & (t > HIT_EPSILON) & (t < 1 - HIT_EPSILON))


def _edges_hit(mesh, other):
    """Does any edge of mesh pass through a face of other (sharing no vertex if same)?"""
    same = mesh is other
    triangles = other.triangles
    tri_min, tri_max = triangles.min(axis=1), triangles.max(axis=1)
    edges = mesh.edges_unique
    ends = mesh.vertices[edges]
    edge_min, edge_max = ends.min(axis=1), ends.max(axis=1)
    near = np.flatnonzero(((edge_max >= other.bounds[0]) & (edge_min <= other.bounds[1])).all(axis=1))
    # Sorted along X, each batch is spatially tight, so it rules out most faces at once.
    near = near[np.argsort(edge_min[near, 0], kind="stable")]
    for first in range(0, len(near), EDGE_BATCH):
        batch = near[first:first + EDGE_BATCH]
        nearby = np.flatnonzero(((edge_max[batch].max(axis=0) >= tri_min)
                                 & (edge_min[batch].min(axis=0) <= tri_max)).all(axis=1))
        # Keep only edge-face pairs whose bounding boxes overlap.
        close = ((edge_max[batch, None] >= tri_min[nearby])
                 & (edge_min[batch, None] <= tri_max[nearby])).all(axis=2)
        edge_index, face_index = np.nonzero(close)
        face_index = nearby[face_index]
        edge_index = batch[edge_index]
        if same:
            faces = other.faces[face_index]
            shares_vertex = (faces[:, :, None] == edges[edge_index][:, None]).any(axis=(1, 2))
            edge_index, face_index = edge_index[~shares_vertex], face_index[~shares_vertex]
        if _segments_hit_triangles(ends[edge_index, 0], ends[edge_index, 1],
                                   triangles[face_index]).any():
            return True
    return False


def meshes_intersect(a, b):
    """Whether two meshes' surfaces cross."""
    if ((a.bounds[1] < b.bounds[0]) | (b.bounds[1] < a.bounds[0])).any():
        return False
    return _edges_hit(a, b) or _edges_hit(b, a)


def self_intersects(mesh):
    """Whether a mesh's surface crosses itself, ignoring faces sharing the tested edge's ends."""
    return _edges_hit(mesh, mesh)


@dataclass(frozen=True)
class ComponentReport:
    name: str
    vertices: int
    faces: int
    degenerate_faces: int = 0
    non_manifold_edges: int = 0
    problems: list = field(default_factory=list)


def inspect_component(name, mesh):
    """Face normals, degenerate triangles and non-manifold edges of one component."""
    problems = []
    if not np.isfinite(mesh.vertices).all():
        problems.append(f"{name}: non-finite vertex positions")
        return ComponentReport(name, len(mesh.vertices), len(mesh.faces), problems=problems)

    degenerate = int((mesh.area_faces < DEGENERATE_AREA_CM2).sum())
    _, uses = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
    non_manifold = int((uses != 2).sum())
    if degenerate:
        problems.append(f"{name}: {degenerate} degenerate faces")
    if non_manifold:
        problems.append(f"{name}: {non_manifold} non-manifold edges")
    if not mesh.is_winding_consistent:
        problems.append(f"{name}: inconsistent winding")
    elif not non_manifold and mesh.volume <= 0:
        problems.append(f"{name}: inward-facing normals")
    return ComponentReport(name, len(mesh.vertices), len(mesh.faces), degenerate, non_manifold,
                           problems)


@dataclass(frozen=True)
class ValidationReport:
    components: list
    unintended_intersections: list
    missing_contacts: list
    self_intersecting: list

    @property
    def problems(self):
        found = [problem for component in self.components for problem in component.problems]
        found += [f"{a} and {b} intersect unintentionally"
                  for a, b in self.unintended_intersections]
        found += [f"{a} and {b} should touch but don't" for a, b in self.missing_contacts]
        found += [f"{name} intersects itself" for name in self.self_intersecting]
        return found

    @property
    def ok(self):
        return not self.problems


def validate_components(components, contacts, check_self=()):
    """Inspect every component; declared contacts must touch, all other pairs must not.

    components -- {name: mesh}
    contacts   -- (name, name) pairs that are meant to overlap, like a hinge in a frame
    check_self -- names of single-shell components to test for self-intersection
    """
    declared = {frozenset(pair) for pair in contacts}
    unintended, missing = [], []
    for a, b in combinations(components, 2):
        touching = meshes_intersect(components[a], components[b])
        if frozenset((a, b)) in declared:
            if not touching:
                missing.append((a, b))
        elif touching:
            unintended.append((a, b))
    return ValidationReport(
        components=[inspect_component(name, mesh) for name, mesh in components.items()],
        unintended_intersections=unintended,
        missing_contacts=missing,
        self_intersecting=[name for name in check_self if self_intersects(components[name])])
