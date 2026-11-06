"""Validating source photos and judging whether they show glasses from the front.

The front-view check is plain geometry on the same segmentation the generator
uses: a clean background, two lens openings, and a wide, short silhouette.
"""
import hashlib
import io
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.errors import InvalidImage
from image_to_3d.errors import ImageToMeshError
from image_to_3d.mesh import mask_to_polygons
from image_to_3d.pipeline import DEFAULT_OPENING_CLEANUP
from image_to_3d.segmentation import SegmentationConfig, remove_opening_intrusions, segment_glasses

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
MIN_SIDE = 64

# Front-view thresholds, from typical retailer packshots.
MIN_BACKGROUND_BORDER = 0.9  # share of the image border that is background
MIN_ASPECT, MAX_ASPECT = 1.6, 4.5  # silhouette width / height
MIN_WIDTH_SHARE = 0.3        # silhouette width / image width
SEGMENTATION = SegmentationConfig(adaptive_white=True)


@dataclass(frozen=True)
class SourceImage:
    data: bytes        # PNG or JPEG bytes the generator can read
    media_type: str    # image/png or image/jpeg
    width: int
    height: int

    @property
    def extension(self):
        return ".png" if self.media_type == "image/png" else ".jpg"

    @property
    def sha256(self):
        return hashlib.sha256(self.data).hexdigest()


def validate_image(data, max_bytes, max_pixels, allow_webp=False):
    """SourceImage from PNG or JPEG bytes (and WebP, converted to PNG, if allowed)."""
    if not data:
        raise InvalidImage("The image is empty.", code="empty_image")
    if len(data) > max_bytes:
        raise InvalidImage(f"Images must be {max_bytes // (1024 * 1024)} MB or smaller.",
                           code="image_too_large", status=413)
    is_webp = data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    if not (data.startswith((PNG_SIGNATURE, JPEG_SIGNATURE)) or (allow_webp and is_webp)):
        raise InvalidImage("Choose a PNG or JPG image.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data))
            width, height = image.size  # read from the header, before decoding
            if width * height > max_pixels:
                raise InvalidImage("The image has too many pixels.", code="image_too_large", status=413)
            if min(width, height) < MIN_SIDE:
                raise InvalidImage(f"The image is too small; use one at least {MIN_SIDE} px on each side.",
                                   code="image_too_small")
            image.load()
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        raise InvalidImage("The image couldn't be read; it may be damaged.", code="corrupt_image") from None

    if is_webp:
        mode = "RGBA" if "A" in image.getbands() else "RGB"
        buffer = io.BytesIO()
        image.convert(mode).save(buffer, format="PNG")
        return SourceImage(buffer.getvalue(), "image/png", width, height)
    media_type = "image/png" if data.startswith(PNG_SIGNATURE) else "image/jpeg"
    # OpenCV must also be able to decode it, since the generator uses OpenCV.
    if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED) is None:
        raise InvalidImage("The image couldn't be read; it may be damaged.", code="corrupt_image")
    return SourceImage(data, media_type, width, height)


@dataclass(frozen=True)
class FrontViewAssessment:
    suitable: bool
    score: float      # higher is a better front view; only for ranking
    openings: int
    aspect: float
    background: float


UNSUITABLE = FrontViewAssessment(False, 0.0, 0, 0.0, 0.0)


def _background_share(image):
    """Share of the outermost 2% border that counts as background (light or transparent)."""
    height, width = image.shape[:2]
    band = max(1, round(0.02 * min(height, width)))
    border = np.ones((height, width), bool)
    border[band:-band, band:-band] = False
    pixels = image[border]
    if image.ndim == 2:
        light = pixels >= SEGMENTATION.white_threshold - 10
    else:
        light = (pixels[:, :3] >= SEGMENTATION.white_threshold - 10).all(axis=1)
        if image.shape[2] == 4:
            light |= pixels[:, 3] < SEGMENTATION.alpha_threshold
    return float(light.mean())


def assess_front_view(source):
    """Whether a photo looks like a frontal packshot the generator can use."""
    image = cv2.imdecode(np.frombuffer(source.data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        return UNSUITABLE
    if image.dtype == np.uint16:
        image = (image / 257).round().astype(np.uint8)
    background = _background_share(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    try:
        mask = remove_opening_intrusions(segment_glasses(image, SEGMENTATION), DEFAULT_OPENING_CLEANUP)
        polygons = mask_to_polygons(mask)
    except (ImageToMeshError, ValueError):
        return FrontViewAssessment(False, 0.0, 0, 0.0, background)
    if not polygons:
        return FrontViewAssessment(False, 0.0, 0, 0.0, background)

    openings = sum(len(polygon.interiors) for polygon in polygons)
    ys, xs = np.nonzero(mask)
    box_width = xs.max() - xs.min() + 1
    box_height = ys.max() - ys.min() + 1
    aspect = float(box_width / box_height)
    width_share = float(box_width / mask.shape[1])

    suitable = bool(background >= MIN_BACKGROUND_BORDER and openings >= 2
                and MIN_ASPECT <= aspect <= MAX_ASPECT and width_share >= MIN_WIDTH_SHARE)
    # Typical front views are about 2.8 times wider than tall.
    score = (2.0 * (openings == 2) + background + max(0.0, 1.0 - abs(aspect - 2.8) / 2.8)
             + min(width_share, 0.9))
    return FrontViewAssessment(suitable, round(score, 4), openings, round(aspect, 3), round(background, 3))
