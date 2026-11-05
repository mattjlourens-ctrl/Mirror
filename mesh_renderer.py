"""Draw a MeshModel as filled, shaded triangles over the camera frame.

Each frame, using the same transforms and camera matrix as the wireframe:

1. Corners and normals move to camera space. Triangles crossing the near plane
   are clipped, not dropped, so the model doesn't lose pieces close to the lens.
2. Triangles are projected and those facing away are culled. The rest are
   rasterized into a depth buffer covering just the model's bounding box, with
   occlusion.py's rasterizer.
3. Pixels behind the head depth buffer are discarded.
4. Only the pixels that remain are shaded, so hidden surfaces cost nothing.
   Normals are interpolated per pixel.
5. Coverage and colour pass through a small filter, softening edges, and are
   blended in. Measured at 720p, this costs a small fraction of supersampling.

Cost follows the number of pixels the model covers, which grows with the frame
size and as the face nears the camera. So a model wider than MAX_RENDER_WIDTH on
screen is rasterized at that width and scaled up. It looks slightly softer but
keeps the frame rate steady, and geometry and depth testing are unaffected.

Pixels the model doesn't cover are left untouched.
"""
import cv2
import numpy as np

from accessory_transform import MIN_DEPTH_CM, project_camera_points
from occlusion import (DEPTH_TOLERANCE_CM, MAX_RASTER_SAMPLES, head_depth_at, triangle_boxes,
                       triangle_fragments)

# Fixed in camera space (X right, Y down, Z forward), as room lighting would be:
# the light is above and slightly to the side of the camera.
_light = np.array((0.25, -0.6, -0.75), dtype=np.float32)
LIGHT_DIRECTION = _light / np.linalg.norm(_light)
# Halfway to the viewer. The viewer direction is treated as constant, which is
# close enough across an object the size of a pair of glasses.
_half = LIGHT_DIRECTION + (0.0, 0.0, -1.0)
HALF_VECTOR = _half / np.linalg.norm(_half)

AMBIENT = 0.45
DIFFUSE = 0.7
# Kept low: a flat frame front turns to face the highlight all at once, and a
# strong one would wash the whole front out at that angle.
SPECULAR = 0.35
SHININESS_SQUARINGS = 5  # highlight falloff exponent 2**5 = 32

MAX_RENDER_WIDTH = 420  # pixels across the model's bounding box

# Weights of the edge-softening filter.
EDGE_FILTER = np.array((0.25, 0.5, 0.25), dtype=np.float32)


def _lerp_to_near(start, end, near_z):
    """Where the edges from `start` to `end` cross the near plane, and how far along."""
    t = ((near_z - start[:, 2]) / (end[:, 2] - start[:, 2]))[:, None]
    point = start + (end - start) * t
    point[:, 2] = near_z  # exactly, so rounding can't put it behind the plane
    return point, t


def clip_triangles_to_near_plane(corners, normals, colors, gloss, near_z=MIN_DEPTH_CM):
    """Keep the part of each triangle with Z >= near_z, preserving winding.

    A triangle with one corner in front stays one triangle; with two it becomes
    a quad, split into two triangles.
    """
    in_front = corners[:, :, 2] >= near_z
    count = in_front.sum(axis=1)
    if count.min(initial=3) == 3:
        return corners, normals, colors, gloss

    whole = np.flatnonzero(count == 3)
    results = [(corners[whole], normals[whole], colors[whole], gloss[whole])]

    def rolled(rows, first):
        """Rows rotated so corner `first` comes first; rotation keeps the winding."""
        order = (first[:, None] + np.arange(3)) % 3
        index = rows[:, None]
        return corners[index, order], normals[index, order]

    def cut(c_from, c_to, n_from, n_to):
        point, t = _lerp_to_near(c_from, c_to, near_z)
        return point, n_from + (n_to - n_from) * t

    single = np.flatnonzero(count == 1)
    if len(single):
        c, n = rolled(single, in_front[single].argmax(axis=1))
        c1, n1 = cut(c[:, 0], c[:, 1], n[:, 0], n[:, 1])
        c2, n2 = cut(c[:, 0], c[:, 2], n[:, 0], n[:, 2])
        results.append((np.stack((c[:, 0], c1, c2), axis=1), np.stack((n[:, 0], n1, n2), axis=1),
                        colors[single], gloss[single]))

    double = np.flatnonzero(count == 2)
    if len(double):
        # Rotate so corner 2 is the one behind the plane.
        c, n = rolled(double, (in_front[double].argmin(axis=1) + 1) % 3)
        c1, n1 = cut(c[:, 1], c[:, 2], n[:, 1], n[:, 2])
        c2, n2 = cut(c[:, 0], c[:, 2], n[:, 0], n[:, 2])
        results.append((np.stack((c[:, 0], c[:, 1], c1), axis=1), np.stack((n[:, 0], n[:, 1], n1), axis=1),
                        colors[double], gloss[double]))
        results.append((np.stack((c[:, 0], c1, c2), axis=1), np.stack((n[:, 0], n1, n2), axis=1),
                        colors[double], gloss[double]))
    return tuple(np.concatenate(arrays) for arrays in zip(*results))


def _batches(counts, limit=MAX_RASTER_SAMPLES):
    """Split triangle indices so each batch stays near `limit` candidate samples."""
    start = 0
    total = 0
    for i, count in enumerate(counts):
        if total and total + count > limit:
            yield start, i
            start, total = i, 0
        total += count
    if start < len(counts):
        yield start, len(counts)


def _resolve_surface(tri_xy, tri_z, width, height):
    """Depth buffer of the nearest surface: (depth, triangle per sample, weights)."""
    depth = np.full(width * height, np.inf, dtype=np.float32)
    triangle = np.full(width * height, -1, dtype=np.int32)
    weights = np.zeros((width * height, 3), dtype=np.float32)

    boxes = triangle_boxes(tri_xy, width, height)
    for start, end in _batches(boxes.counts.tolist()):
        fragments = triangle_fragments(tri_xy, boxes, start, end)
        if fragments is None:
            continue
        tri, x, y, barycentric = fragments
        # Perspective-correct: 1/Z is linear on screen, and so is attribute/Z.
        over_z = barycentric / tri_z[boxes.triangles[tri]]
        z = (1.0 / over_z.sum(axis=1)).astype(np.float32)
        index = y * width + x
        np.minimum.at(depth, index, z)
        nearest = z == depth[index]
        chosen = index[nearest]
        triangle[chosen] = boxes.triangles[tri[nearest]]
        weights[chosen] = over_z[nearest] * z[nearest, None]
    return depth, triangle, weights


def _shade(normals, base_colors, gloss):
    """Lit BGR colours (float32, 0-255) for unit normals."""
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)
    diffuse = np.clip(normals @ LIGHT_DIRECTION, 0.0, 1.0)
    facing = np.clip(normals @ HALF_VECTOR, 0.0, 1.0)
    # Repeated squaring: per-pixel pow() was the largest cost in shading.
    for _ in range(SHININESS_SQUARINGS):
        facing = facing * facing
    highlight = gloss * SPECULAR * facing
    return (base_colors * (AMBIENT + DIFFUSE * diffuse)[:, None]
            + 255.0 * highlight[:, None]).astype(np.float32)


def render_mesh(frame, model, accessory_to_face, pose, depth_buffer=None):
    """Draw the model onto frame in place; True if any pixel was drawn."""
    if pose is None:
        return False
    data = model.render_data
    to_camera = pose.transform @ accessory_to_face
    linear, offset = to_camera[:3, :3], to_camera[:3, 3]
    corners, normals, colors, gloss = clip_triangles_to_near_plane(
        data.corners @ linear.T + offset,
        data.normals.astype(np.float32) @ linear.T.astype(np.float32), data.colors, data.gloss)
    if len(corners) == 0:
        return False

    pixels = project_camera_points(corners.reshape(-1, 3), pose.camera_matrix).reshape(-1, 3, 2)
    image_h, image_w = frame.shape[:2]

    # Outward triangles appear clockwise in image coordinates (see rasterize_depth).
    a, b, c = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    facing = area < 0
    if not facing.any():
        return False
    low = pixels[facing].min(axis=(0, 1))
    high = pixels[facing].max(axis=(0, 1))
    # One pixel of margin leaves room for the softened edge.
    x0, y0 = max(0, int(np.floor(low[0])) - 1), max(0, int(np.floor(low[1])) - 1)
    x1, y1 = min(image_w, int(np.ceil(high[0])) + 2), min(image_h, int(np.ceil(high[1])) + 2)
    if x1 <= x0 or y1 <= y0:
        return False

    # Pixel centres line up at any scale: image x = x0 + (raster x + 0.5) / scale - 0.5,
    # which is the mapping cv2.resize uses.
    size = (x1 - x0, y1 - y0)
    scale = min(1.0, MAX_RENDER_WIDTH / size[0])
    width, height = max(1, round(size[0] * scale)), max(1, round(size[1] * scale))
    scale_xy = np.array((width / size[0], height / size[1]))
    tri_xy = (pixels - (x0, y0) + 0.5) * scale_xy - 0.5
    tri_xy[~facing] = np.nan  # triangle_boxes skips these
    depth, triangle, weights = _resolve_surface(tri_xy, corners[:, :, 2], width, height)

    covered = np.flatnonzero(triangle >= 0)
    if depth_buffer is not None and len(covered):
        raster_xy = np.stack((covered % width, covered // width), axis=1)
        image_xy = (raster_xy + 0.5) / scale_xy - 0.5 + (x0, y0)
        head = head_depth_at(depth_buffer, image_xy)
        covered = covered[depth[covered] <= head + DEPTH_TOLERANCE_CM]
    if len(covered) == 0:
        return False

    hit = triangle[covered]
    pixel_normals = (normals[hit] * weights[covered, :, None]).sum(axis=1)
    color = np.zeros((height * width, 3), dtype=np.float32)
    color[covered] = _shade(pixel_normals, colors[hit], gloss[hit])
    alpha = np.zeros(height * width, dtype=np.float32)
    alpha[covered] = 1.0

    # Filtering colour (zero where uncovered) and coverage alike keeps the result
    # premultiplied. Replicating the border keeps edges solid at the image's side.
    color = cv2.sepFilter2D(color.reshape(height, width, 3), -1, EDGE_FILTER, EDGE_FILTER,
                            borderType=cv2.BORDER_REPLICATE)
    alpha = cv2.sepFilter2D(alpha.reshape(height, width), -1, EDGE_FILTER, EDGE_FILTER,
                            borderType=cv2.BORDER_REPLICATE)
    if (width, height) != size:
        color = cv2.resize(color, size, interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_LINEAR)

    region = frame[y0:y1, x0:x1]
    touched = alpha > 0
    blended = region[touched] * (1.0 - alpha[touched, None]) + color[touched]
    region[touched] = np.clip(np.round(blended), 0, 255).astype(np.uint8)
    return True
