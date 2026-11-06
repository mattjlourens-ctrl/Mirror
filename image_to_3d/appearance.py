"""Texture and materials for the complete model, taken from the product photo.

What comes from the photo, and what is estimated
------------------------------------------------
- The front of the frame shows the photo itself: every point on the front
  surface samples the photo pixel it was traced from.
- Side walls and the back can't be seen in a front view. They sample the same
  place in a copy of the photo with highlights flattened, so a pattern that runs
  through the material continues around the edge, without the photographer's
  reflections stretched across surfaces they never lit.
- Temples take the typical colour of the frame's end pieces, where they attach.
  Hinges match the frame unless metal is chosen; a front view can't tell.
- Roughness and metalness are settings, not measurements.

The texture
-----------
One image holds two halves, stacked: the photo crop on top, the flattened copy
below. Frame pixels next to the silhouette are a blend of frame and background,
so the texture only trusts pixels a little inside the frame, and every other
texel takes the colour of the nearest trusted one. Texture filtering near the
outline then never reaches white background.

UV coordinates follow trimesh: (0, 0) is the bottom-left of the image.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from image_to_3d.errors import SegmentationError
from image_to_3d.mesh import working_to_cm, working_to_image

TEXTURE_MARGIN_PX = 8      # around the frame, so filtering stays on frame colours
MAX_TEXTURE_SIZE_PX = 2048
UNTRUSTED_EDGE_PX = 2      # outline pixels blended with the background
HIGHLIGHT_PERCENTILE = 75  # edges are no lighter than this share of frame pixels
END_PIECE_FRACTION = 0.12  # of the frame's width, at each side, for the temple colour
FRONT_NORMAL_Z = 0.5       # faces turned further from the viewer use the edge half

METAL_COLOR = (170, 170, 175)
PAD_COLOR = (225, 225, 215)


@dataclass(frozen=True)
class MaterialSettings:
    """Material properties the photo can't measure."""
    frame_roughness: float = 0.6  # 0 = mirror-like, 1 = fully matte
    hinge_material: str = "match"  # "match" the frame colour, or "metal"
    metal_roughness: float = 0.3
    pad_roughness: float = 0.5

    def validate(self):
        for name in ("frame_roughness", "metal_roughness", "pad_roughness"):
            value = getattr(self, name)
            if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name.replace('_', ' ')} must be between 0 and 1")
        if self.hinge_material not in ("match", "metal"):
            raise ValueError("hinge material must be 'match' or 'metal'")


@dataclass(frozen=True, eq=False)
class ImageMapping:
    """Front-view centimetres (X right, Y up) <-> photo pixels (centres at integers)."""
    image_scale: np.ndarray    # (2,) image = image_scale * working + image_offset
    image_offset: np.ndarray
    cm_per_working: float      # cm = cm_per_working * (working - working_center), Y flipped
    working_center: np.ndarray

    def cm_to_image(self, points_cm):
        points_cm = np.asarray(points_cm, dtype=np.float64)
        working = points_cm * (1.0, -1.0) / self.cm_per_working + self.working_center
        return working * self.image_scale + self.image_offset

    def image_to_cm(self, pixels):
        working = (np.asarray(pixels, dtype=np.float64) - self.image_offset) / self.image_scale
        return (working - self.working_center) * self.cm_per_working * (1.0, -1.0)


@dataclass(frozen=True, eq=False)
class FrameAppearance:
    texture: np.ndarray    # (2H, W, 3) uint8 RGB: photo above, flattened copy below
    crop_origin: np.ndarray  # photo pixel (x, y) at the texture's top-left corner
    crop_size: np.ndarray    # (width, height) of that crop in photo pixels
    mapping: ImageMapping
    frame_color: tuple     # RGB
    temple_color: tuple
    settings: MaterialSettings

    def uv_at(self, points_cm, front=True):
        """(N, 2) UVs for front-view positions, in the photo or the edge half."""
        pixels = self.mapping.cm_to_image(points_cm)
        u = (pixels[:, 0] - self.crop_origin[0] + 0.5) / self.crop_size[0]
        row = (pixels[:, 1] - self.crop_origin[1] + 0.5) / self.crop_size[1]
        v = (1.0 - row / 2) if front else (0.5 - row / 2)
        return np.stack((u, v), axis=1)

    def materials(self):
        """glTF PBR values per role ("front", "frame", "temple", "hinge", "pad")."""
        settings = self.settings
        frame = dict(baseColorFactor=(*self.frame_color, 255), metallicFactor=0.0,
                     roughnessFactor=settings.frame_roughness)
        if settings.hinge_material == "metal":
            hinge = dict(baseColorFactor=(*METAL_COLOR, 255), metallicFactor=0.9,
                         roughnessFactor=settings.metal_roughness)
        else:
            hinge = frame
        return {
            "front": dict(baseColorFactor=(255, 255, 255, 255), metallicFactor=0.0,
                          roughnessFactor=settings.frame_roughness),
            "frame": frame,
            "temple": dict(frame, baseColorFactor=(*self.temple_color, 255)),
            "hinge": hinge,
            "pad": dict(baseColorFactor=(*PAD_COLOR, 255), metallicFactor=0.0,
                        roughnessFactor=settings.pad_roughness),
        }


def _nearest_trusted(rgb, trusted):
    """Every pixel takes the colour of the nearest trusted pixel."""
    if trusted.all():
        return rgb.copy()
    _, labels = cv2.distanceTransformWithLabels((~trusted).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    # Labels number the trusted pixels in raster order, starting at 1.
    return rgb[trusted][labels - 1]


def _flatten_highlights(rgb, trusted):
    """Clamp lightness to a high percentile of the frame's, keeping hue and chroma."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    limit = np.percentile(lab[..., 0][trusted], HIGHLIGHT_PERCENTILE)
    lab[..., 0] = np.minimum(lab[..., 0], np.uint8(round(limit)))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _median_color(rgb, where):
    return tuple(int(round(value)) for value in np.median(rgb[where], axis=0))


def extract_appearance(image, mask, polygons, frame_width, settings=MaterialSettings()):
    """Texture, colours and UV mapping for a model built from `polygons`.

    image    -- the photo as loaded (BGR or BGRA, uint8)
    mask     -- the mask the polygons were traced from
    polygons -- mask_to_polygons(mask)
    """
    settings.validate()
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("expected a colour image (BGR or BGRA)")
    if image.shape[:2] != mask.shape:
        raise ValueError("the image and the mask must be the same size")
    frame = mask > 0
    if not frame.any():
        raise SegmentationError("No glasses found: the mask is empty")

    image_scale, image_offset = working_to_image(mask)
    cm_per_working, working_center = working_to_cm(polygons, frame_width)
    mapping = ImageMapping(image_scale, image_offset, cm_per_working, working_center)

    ys, xs = np.nonzero(frame)
    height, width = mask.shape
    left = max(0, xs.min() - TEXTURE_MARGIN_PX)
    top = max(0, ys.min() - TEXTURE_MARGIN_PX)
    right = min(width, xs.max() + 1 + TEXTURE_MARGIN_PX)
    bottom = min(height, ys.max() + 1 + TEXTURE_MARGIN_PX)

    crop_frame = frame[top:bottom, left:right]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    trusted = cv2.erode(crop_frame.astype(np.uint8), kernel, iterations=UNTRUSTED_EDGE_PX) > 0
    if not trusted.any():  # a frame thinner than the untrusted band
        trusted = crop_frame
    rgb = cv2.cvtColor(np.ascontiguousarray(image[top:bottom, left:right, :3]), cv2.COLOR_BGR2RGB)
    photo = _nearest_trusted(rgb, trusted)
    edges = _flatten_highlights(photo, trusted)

    columns = np.arange(left, right)[None, :].repeat(bottom - top, axis=0)
    end_width = END_PIECE_FRACTION * (xs.max() + 1 - xs.min())
    ends = trusted & ((columns < xs.min() + end_width) | (columns >= xs.max() + 1 - end_width))
    frame_color = _median_color(edges, trusted)
    temple_color = _median_color(edges, ends) if ends.any() else frame_color

    texture = np.vstack((photo, edges))
    longest = max(texture.shape[:2])
    if longest > MAX_TEXTURE_SIZE_PX:
        factor = MAX_TEXTURE_SIZE_PX / longest
        size = (max(1, round(texture.shape[1] * factor)), max(2, 2 * round(photo.shape[0] * factor)))
        texture = cv2.resize(texture, size, interpolation=cv2.INTER_AREA)

    return FrameAppearance(texture=texture, crop_origin=np.array((left, top)),
                           crop_size=np.array((right - left, bottom - top)), mapping=mapping,
                           frame_color=frame_color, temple_color=temple_color, settings=settings)


def frame_uvs(appearance, front_mesh):
    """(F, 3, 2) UVs per face corner: faces towards the viewer use the photo half."""
    facing = front_mesh.face_normals[:, 2] > FRONT_NORMAL_Z
    corners = front_mesh.vertices[front_mesh.faces][..., :2].reshape(-1, 2)
    on_front = np.repeat(facing, 3)
    uvs = np.where(on_front[:, None], appearance.uv_at(corners, True),
                   appearance.uv_at(corners, False))
    return uvs.reshape(-1, 3, 2)


def textured_frame_mesh(front_mesh, appearance):
    """The front frame with UVs and the texture; geometry unchanged.

    A vertex on the border between the front and an edge is split in two, one per
    half of the texture. Everywhere else vertices stay shared, so the surface has
    no seams.
    """
    import trimesh
    from PIL import Image

    uvs = frame_uvs(appearance, front_mesh).reshape(-1, 2)
    facing = front_mesh.face_normals[:, 2] > FRONT_NORMAL_Z
    keys = np.stack((front_mesh.faces.ravel(), np.repeat(facing, 3)), axis=1)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    vertex_uvs = np.zeros((len(unique), 2))
    vertex_uvs[inverse] = uvs
    mesh = trimesh.Trimesh(front_mesh.vertices[unique[:, 0]], inverse.reshape(-1, 3), process=False)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(appearance.texture), **appearance.materials()["front"])
    mesh.visual = trimesh.visual.TextureVisuals(uv=vertex_uvs, material=material)
    return mesh
