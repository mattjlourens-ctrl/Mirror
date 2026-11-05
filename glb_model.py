"""Load a generated glasses .glb for the live renderer.

Each mesh node stays a separate named part (front_frame, left_temple, ...) with
its own material. The glTF axes already match Mirror's accessory axes (X right,
Y up, Z forward), so loading only converts metres to centimetres.

Models are static: a file is parsed once, and render_data is built once per
model, however many frames use it.
"""
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np
import trimesh

METRES_TO_CM = 100.0
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "image_to_3d" / "output" / "example_3_photo.glb"
DEFAULT_COLOR_BGR = (40, 40, 40)

# Largest dimension of anything plausibly worn on a head, in cm. Catches files
# exported in the wrong unit, which would otherwise fill or vanish from the view.
MIN_MODEL_SIZE_CM = 1.0
MAX_MODEL_SIZE_CM = 100.0

# Neighbouring faces closer than this share a smooth normal. Curved rims and
# temples shade smoothly, and the square edges of the extrusion stay sharp.
CREASE_ANGLE_DEGREES = 35.0


class ModelLoadError(Exception):
    """A model file could not be used; the message says why."""


@dataclass(frozen=True, eq=False)
class MeshPart:
    name: str
    vertices: np.ndarray  # (N, 3) cm
    faces: np.ndarray     # (F, 3) vertex indices, counter-clockwise from outside
    color_bgr: tuple = DEFAULT_COLOR_BGR
    gloss: float = 0.0    # 0 = matte, 1 = strong highlight


@dataclass(frozen=True, eq=False)
class RenderData:
    """One row per face, ready for the renderer. Never changed after building."""
    corners: np.ndarray  # (F, 3, 3) cm
    normals: np.ndarray  # (F, 3, 3) unit normal at each corner
    colors: np.ndarray   # (F, 3) BGR
    gloss: np.ndarray    # (F,)


@dataclass(frozen=True, eq=False)
class MeshModel:
    name: str
    parts: tuple

    def __post_init__(self):
        if not self.parts:
            raise ModelLoadError(f"{self.name}: the model has no usable meshes")
        for part in self.parts:
            _validate_part(part)

    @property
    def part_names(self):
        return tuple(part.name for part in self.parts)

    def part(self, name):
        for part in self.parts:
            if part.name == name:
                return part
        raise KeyError(name)

    @cached_property
    def render_data(self):
        corners, normals, colors, gloss = [], [], [], []
        for part in self.parts:
            faces = part.faces[_non_degenerate(part)]
            corners.append(part.vertices[faces])
            normals.append(_corner_normals(part.vertices, faces))
            colors.append(np.tile(np.asarray(part.color_bgr, dtype=np.float32), (len(faces), 1)))
            gloss.append(np.full(len(faces), part.gloss, dtype=np.float32))
        return RenderData(np.concatenate(corners), np.concatenate(normals),
                          np.concatenate(colors), np.concatenate(gloss))


def _validate_part(part):
    if len(part.vertices) == 0 or len(part.faces) == 0:
        raise ModelLoadError(f"{part.name}: the mesh is empty")
    if not np.isfinite(part.vertices).all():
        raise ModelLoadError(f"{part.name}: the mesh has non-finite vertices")
    if part.faces.min() < 0 or part.faces.max() >= len(part.vertices):
        raise ModelLoadError(f"{part.name}: a face refers to a missing vertex")


def _face_normals(vertices, faces):
    """(F, 3) vectors whose length is twice the face's area."""
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    return np.cross(b - a, c - a)


def _non_degenerate(part):
    return np.linalg.norm(_face_normals(part.vertices, part.faces), axis=1) > 1e-12


def _corner_normals(vertices, faces):
    """(F, 3, 3) area-weighted normals, smoothed across edges below the crease angle."""
    weighted = _face_normals(vertices, faces)
    unit = weighted / np.linalg.norm(weighted, axis=1, keepdims=True)
    limit = np.cos(np.radians(CREASE_ANGLE_DEGREES))

    # Vertices at the same position count as one, in case a file duplicates them.
    _, welded = np.unique(np.round(vertices, 6), axis=0, return_inverse=True)
    corner_vertex = welded.ravel()[faces].ravel()
    corner_face = np.repeat(np.arange(len(faces)), 3)
    order = np.argsort(corner_vertex, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(corner_vertex[order])) + 1)

    normals = np.empty((len(faces) * 3, 3))
    for corners in groups:
        touching = corner_face[corners]
        similar = unit[touching] @ unit[touching].T >= limit
        normals[corners] = similar @ weighted[touching]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals.reshape(-1, 3, 3)


def _material_of(mesh):
    """(BGR colour, gloss) from a glTF material; a missing roughness means matte."""
    material = getattr(mesh.visual, "material", None)
    color = getattr(material, "baseColorFactor", None)
    roughness = getattr(material, "roughnessFactor", None)
    gloss = 0.0 if roughness is None else float(np.clip(1.0 - roughness, 0.0, 1.0))
    if color is None:
        return DEFAULT_COLOR_BGR, gloss
    red, green, blue = (int(channel) for channel in np.asarray(color)[:3])
    return (blue, green, red), gloss


def _parts_of(scene):
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name]
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        vertices = trimesh.transform_points(mesh.vertices, transform) * METRES_TO_CM
        color, gloss = _material_of(mesh)
        parts.append(MeshPart(node, np.asarray(vertices, dtype=np.float64),
                              np.asarray(mesh.faces, dtype=np.int64), color, gloss))
    return tuple(parts)


def _check_size(path, parts):
    points = np.concatenate([part.vertices for part in parts])
    size = np.ptp(points, axis=0).max()
    if not MIN_MODEL_SIZE_CM <= size <= MAX_MODEL_SIZE_CM:
        raise ModelLoadError(f"{path}: implausible size of {size:.3g} cm when read as metres; "
                             "expected glasses exported in metres")


@lru_cache(maxsize=8)
def _load_file(path, modified_ns, size_bytes):
    try:
        scene = trimesh.load_scene(path, file_type="glb")
        parts = _parts_of(scene)
    except ModelLoadError:
        raise
    except Exception as error:  # the parsers raise many unrelated types on bad data
        raise ModelLoadError(f"{path}: not a readable GLB file ({error})") from error
    if not parts:
        raise ModelLoadError(f"{path}: the file has no meshes")
    model = MeshModel(Path(path).stem, parts)
    _check_size(path, parts)
    if len(model.render_data.corners) == 0:  # built now, not during the first frame
        raise ModelLoadError(f"{path}: every face has zero area")
    return model


def load_glasses_model(path=None):
    """MeshModel from a .glb (default: DEFAULT_MODEL_PATH), or ModelLoadError.

    The result is cached until the file changes, so calling this again is cheap
    and returns the same object.
    """
    path = Path(DEFAULT_MODEL_PATH if path is None else path)
    if path.suffix.lower() != ".glb":
        raise ModelLoadError(f"{path}: expected a .glb file")
    try:
        status = path.stat()
    except OSError:
        raise ModelLoadError(f"{path}: file not found") from None
    if not path.is_file():
        raise ModelLoadError(f"{path}: not a file")
    if status.st_size == 0:
        raise ModelLoadError(f"{path}: the file is empty")
    return _load_file(str(path.resolve()), status.st_mtime_ns, status.st_size)


load_glasses_model.cache_clear = _load_file.cache_clear
