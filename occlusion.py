"""Depth-buffer occlusion: hide accessory geometry that is behind the head.

How it works each frame
-----------------------
1. The head occluder mesh (face-local cm) is moved into camera space with
   the same HeadPose.transform the accessory uses, so both share one
   coordinate system.
2. Its triangles are rasterized into a small depth buffer: for every pixel,
   the camera-space Z (cm) of the nearest head surface, or +inf where there
   is no head. Depth is interpolated perspective-correctly (1/Z is linear in
   screen space, Z is not).
3. Accessory lines are sampled every few millimetres in 3D. A sample is drawn
   only if it is in front of the camera's near plane and no more than
   DEPTH_TOLERANCE_CM behind the head surface at its pixel.

The occluder
------------
MediaPipe's canonical face mesh (468 vertices, 898 triangles) only covers the
front of the face; it stops at the face oval, roughly in front of the ears.
Temple arms run back past that, so we add a simple, clearly approximate
"head proxy": the face oval extruded backwards to an ear-plane ring, then to a
narrower ring at the back of the head, closed with a cap. It has no ears or
hair, so arm tips are hidden by the proxy surface, not by a real ear.
"""
from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np

from accessory_transform import MIN_DEPTH_CM, project_camera_points, transform_points

TASK_FILE = Path(__file__).resolve().parent / "face_landmarker.task"
GEOMETRY_ENTRY = "geometry_pipeline_metadata_landmarks.binarypb"

# Face oval, in order around the face (matches FACEMESH_FACE_OVAL).
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]

# Head proxy rings behind the face oval (face-local cm; the face oval's sides
# are at about z = -2.4). Widths are fractions of the face oval's width.
EAR_PLANE_Z = -6.0
EAR_PLANE_WIDTH = 1.0
BACK_OF_HEAD_Z = -12.0
BACK_OF_HEAD_WIDTH = 0.75
BACK_CAP_Z = -13.5

# The depth buffer is rendered at reduced resolution for speed.
DEPTH_BUFFER_WIDTH = 320

# Accessory samples up to this far behind the head surface still count as
# visible. Absorbs the gap between the canonical face and a real face, parts
# that rest on the skin (nose pads), and depth-buffer pixel size.
DEPTH_TOLERANCE_CM = 0.5

# Safety limit on rasterization work (e.g. a face almost touching the lens).
MAX_RASTER_SAMPLES = 2_000_000


@dataclass(frozen=True)
class HeadOccluder:
    vertices: np.ndarray   # (N, 3) face-local cm
    triangles: np.ndarray  # (T, 3) int vertex indices


@dataclass(frozen=True)
class DepthBuffer:
    depth: np.ndarray  # (H, W) float32 camera-space Z in cm, inf = no head
    scale: float       # buffer pixels per image pixel


@dataclass(frozen=True)
class LineSamples:
    """Points along a line model, sampled once in model space."""
    points: np.ndarray  # (S, 3) model units
    pairs: np.ndarray   # (P, 2) indices of neighbouring samples on the same edge


def load_canonical_face_mesh(task_file=TASK_FILE):
    """(vertices cm, triangles) of MediaPipe's canonical face, or None."""
    try:
        from mediapipe.tasks.cc.vision.face_geometry.proto import (
            geometry_pipeline_metadata_pb2)
        with zipfile.ZipFile(task_file) as bundle:
            data = bundle.read(GEOMETRY_ENTRY)
    except (OSError, KeyError, ImportError, zipfile.BadZipFile):
        return None
    metadata = geometry_pipeline_metadata_pb2.GeometryPipelineMetadata()
    metadata.ParseFromString(data)
    mesh = metadata.canonical_mesh
    # Each vertex is x, y, z, u, v; keep the position.
    vertices = np.array(mesh.vertex_buffer, dtype=np.float64).reshape(-1, 5)[:, :3]
    triangles = np.array(mesh.index_buffer, dtype=np.int32).reshape(-1, 3)
    return np.ascontiguousarray(vertices), triangles


def build_head_occluder(face_vertices, face_triangles):
    """Canonical face plus the approximate head proxy behind it."""
    oval = np.array(FACE_OVAL)
    oval_points = face_vertices[oval]
    ring_size = len(oval)

    ear_ring = oval_points * (EAR_PLANE_WIDTH, 1.0, 0.0) + (0.0, 0.0, EAR_PLANE_Z)
    back_ring = oval_points * (BACK_OF_HEAD_WIDTH, 0.9, 0.0) + (0.0, 0.0, BACK_OF_HEAD_Z)
    cap = np.array([[0.0, oval_points[:, 1].mean(), BACK_CAP_Z]])

    first_ear = len(face_vertices)
    first_back = first_ear + ring_size
    cap_index = first_back + ring_size
    vertices = np.vstack([face_vertices, ear_ring, back_ring, cap])

    def band(front, back):
        """Two triangles per quad between two rings of vertex indices."""
        tris = []
        for i in range(ring_size):
            j = (i + 1) % ring_size
            tris += [(front[i], back[i], back[j]), (front[i], back[j], front[j])]
        return tris

    ear_indices = np.arange(first_ear, first_ear + ring_size)
    back_indices = np.arange(first_back, first_back + ring_size)
    proxy = band(oval, ear_indices) + band(ear_indices, back_indices)
    proxy += [(back_indices[i], cap_index, back_indices[(i + 1) % ring_size])
              for i in range(ring_size)]

    proxy = _wind_outwards(vertices, np.array(proxy, dtype=np.int32))
    triangles = np.vstack([face_triangles, proxy])
    return HeadOccluder(vertices, triangles)


def _wind_outwards(vertices, triangles):
    """Flip triangles so their normals point away from the head's centre,
    matching the canonical face (counter-clockwise seen from outside)."""
    center = vertices.mean(axis=0)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    normals = np.cross(b - a, c - a)
    inward = np.einsum("ij,ij->i", normals, (a + b + c) / 3 - center) < 0
    triangles = triangles.copy()
    triangles[inward] = triangles[inward][:, [0, 2, 1]]
    return triangles


def load_head_occluder(task_file=TASK_FILE):
    """Build the occluder from the bundled model file, or None if unavailable."""
    mesh = load_canonical_face_mesh(task_file)
    if mesh is None:
        return None
    return build_head_occluder(*mesh)


def rasterize_depth(points_xy, depths, triangles, width, height, cull_back_faces=False):
    """Z-buffer: nearest depth per pixel over all triangles, inf elsewhere.

    points_xy -- (N, 2) vertex positions in buffer pixels
    depths    -- (N,) camera-space Z of each vertex (must be > 0)
    cull_back_faces -- skip triangles facing away from the camera. Only valid
        for a closed, outward-wound mesh, where the nearest surface always
        faces the camera. Outward (counter-clockwise from outside) triangles
        appear clockwise in image coordinates because image y points down,
        which gives them a negative signed area here.
    Pixel (x, y) is sampled at integer coordinates.
    """
    depth = np.full(width * height, np.inf, dtype=np.float32)
    if len(triangles) == 0:
        return depth.reshape(height, width)

    tri_xy = points_xy[triangles]  # (T, 3, 2)
    tri_z = depths[triangles]      # (T, 3)

    # Bounding box of each triangle, clipped to the buffer.
    x0 = np.clip(np.ceil(tri_xy[..., 0].min(axis=1)), 0, width).astype(np.int64)
    x1 = np.clip(np.floor(tri_xy[..., 0].max(axis=1)), -1, width - 1).astype(np.int64)
    y0 = np.clip(np.ceil(tri_xy[..., 1].min(axis=1)), 0, height).astype(np.int64)
    y1 = np.clip(np.floor(tri_xy[..., 1].max(axis=1)), -1, height - 1).astype(np.int64)
    a, b, c = tri_xy[:, 0], tri_xy[:, 1], tri_xy[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])

    keep = (x1 >= x0) & (y1 >= y0) & (np.abs(area) > 1e-9)
    if cull_back_faces:
        keep &= area < 0
    box_w = (x1 - x0 + 1)[keep]
    counts = box_w * (y1 - y0 + 1)[keep]
    total = int(counts.sum())
    if total == 0 or total > MAX_RASTER_SAMPLES:
        return None if total else depth.reshape(height, width)

    # One row per (triangle, candidate pixel) pair, all at once.
    tri = np.repeat(np.flatnonzero(keep), counts)
    local = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    widths = np.repeat(box_w, counts)
    px = x0[tri] + local % widths
    py = y0[tri] + local // widths

    # Barycentric weights of each pixel within its triangle.
    ax, ay = a[tri, 0], a[tri, 1]
    w_b = ((px - ax) * (c[tri, 1] - ay) - (c[tri, 0] - ax) * (py - ay)) / area[tri]
    w_c = ((b[tri, 0] - ax) * (py - ay) - (px - ax) * (b[tri, 1] - ay)) / area[tri]
    w_a = 1.0 - w_b - w_c
    eps = -1e-7
    inside = (w_a >= eps) & (w_b >= eps) & (w_c >= eps)

    # Perspective-correct depth: interpolate 1/Z, then invert.
    z = tri_z[tri]
    inverse_z = w_a / z[:, 0] + w_b / z[:, 1] + w_c / z[:, 2]
    pixel_z = 1.0 / inverse_z[inside]

    np.minimum.at(depth, (py * width + px)[inside], pixel_z.astype(np.float32))
    return depth.reshape(height, width)


def render_depth_buffer(occluder, pose, image_width, image_height,
                        buffer_width=DEPTH_BUFFER_WIDTH):
    """Head depth buffer for this frame, or None without a usable pose/occluder."""
    if occluder is None or pose is None:
        return None
    scale = min(1.0, buffer_width / image_width)
    width = max(1, round(image_width * scale))
    height = max(1, round(image_height * scale))

    camera_points = transform_points(pose.transform, occluder.vertices)
    pixels = project_camera_points(camera_points, pose.camera_matrix) * scale

    # Near-plane handling: drop triangles with any vertex too close/behind.
    vertex_ok = np.isfinite(pixels).all(axis=1)
    triangles = occluder.triangles[vertex_ok[occluder.triangles].all(axis=1)]

    depth = rasterize_depth(np.nan_to_num(pixels), camera_points[:, 2], triangles,
                            width, height, cull_back_faces=True)
    if depth is None:
        return None
    return DepthBuffer(depth, scale)


def head_depth_at(depth_buffer, pixels):
    """Head depth (cm) at image pixels; inf off-buffer or for invalid pixels."""
    result = np.full(len(pixels), np.inf)
    buffer_xy = np.round(pixels * depth_buffer.scale)
    height, width = depth_buffer.depth.shape
    valid = np.isfinite(buffer_xy).all(axis=1)
    valid &= (buffer_xy[:, 0] >= 0) & (buffer_xy[:, 0] < width)
    valid &= (buffer_xy[:, 1] >= 0) & (buffer_xy[:, 1] < height)
    x = buffer_xy[valid, 0].astype(np.int64)
    y = buffer_xy[valid, 1].astype(np.int64)
    result[valid] = depth_buffer.depth[y, x]
    return result


def visible_mask(depth_buffer, pixels, depths, tolerance=DEPTH_TOLERANCE_CM):
    """True where a point (image pixel + camera Z) is not hidden by the head.

    Points with invalid pixels or depths (NaN, behind the near plane) are
    never visible.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    drawable = np.isfinite(pixels).all(axis=1) & np.isfinite(depths) & (depths >= MIN_DEPTH_CM)
    if depth_buffer is None:
        return drawable
    return drawable & (depths <= head_depth_at(depth_buffer, pixels) + tolerance)


def sample_lines(vertices, edges, spacing):
    """Split every edge into pieces no longer than `spacing` (model units)."""
    points, pairs = [], []
    count = 0
    for start, end in edges:
        length = np.linalg.norm(vertices[end] - vertices[start])
        pieces = max(1, int(np.ceil(length / spacing)))
        t = np.linspace(0.0, 1.0, pieces + 1)[:, None]
        points.append(vertices[start] * (1 - t) + vertices[end] * t)
        pairs += [(count + i, count + i + 1) for i in range(pieces)]
        count += pieces + 1
    return LineSamples(np.vstack(points), np.array(pairs, dtype=np.int64).reshape(-1, 2))


def visible_line_segments(samples, model_to_camera, camera_matrix, depth_buffer):
    """(K, 2, 2) pixel segments between neighbouring samples that are both visible.

    Sampling in 3D then projecting each sample keeps perspective exact, and
    samples behind the near plane are simply dropped, which clips lines there.
    """
    camera_points = transform_points(model_to_camera, samples.points)
    pixels = project_camera_points(camera_points, camera_matrix)
    visible = visible_mask(depth_buffer, pixels, camera_points[:, 2])
    pairs = samples.pairs[visible[samples.pairs].all(axis=1)]
    return pixels[pairs]
