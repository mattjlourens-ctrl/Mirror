"""Load a product image and separate the glasses from its background."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from image_to_3d.errors import ImageLoadError, SegmentationError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"

# Whatever the background is, it makes up most of the image edge.
MIN_BORDER_BACKGROUND_FRACTION = 0.5
# Specks and pinholes are small relative to the frame itself.
MIN_FEATURE_FRACTION = 0.002
MAX_INTRUSION_FRACTION = 0.2


@dataclass(frozen=True)
class SegmentationConfig:
    white_threshold: int = 240  # a pixel with every channel at or above this is background
    alpha_threshold: int = 200  # a pixel with alpha below this is background (tinted lenses)
    # Lower the white threshold to halfway between the frame and white, which is
    # where a real edge lies; faint drop shadows then fall on the background side.
    adaptive_white: bool = False

    def validate(self):
        for threshold in (self.white_threshold, self.alpha_threshold):
            if not 0 < threshold <= 255:
                raise ValueError("segmentation thresholds must be in 1..255")


def _looks_complete(data):
    """OpenCV decodes cut-off files into partly blank images, so look for the end marker.

    It isn't always last: cameras may append metadata after a JPEG's.
    """
    return (b"IEND" if data.startswith(PNG_SIGNATURE) else b"\xff\xd9") in data


def load_image(path):
    """Read a PNG or JPEG into an 8-bit BGR or BGRA array."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise ImageLoadError(f"Image not found: {path}") from None
    except OSError as error:
        raise ImageLoadError(f"Could not read {path}: {error}") from error

    if not data:
        raise ImageLoadError(f"Image file is empty: {path}")
    if not data.startswith((PNG_SIGNATURE, JPEG_SIGNATURE)):
        raise ImageLoadError(f"Not a PNG or JPEG image: {path}")
    if not _looks_complete(data):
        raise ImageLoadError(f"Image file is truncated: {path}")

    try:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as error:
        raise ImageLoadError(f"Could not decode {path}: {error}") from error
    if image is None or image.size == 0:
        raise ImageLoadError(f"Could not decode {path}")

    if image.dtype == np.uint16:
        image = (image / 257).round().astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _border(array):
    return np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]))


def _foreground_from_white(bgr, config):
    darkest_channel = bgr.min(axis=2)
    foreground = darkest_channel < config.white_threshold
    if (~_border(foreground)).mean() < MIN_BORDER_BACKGROUND_FRACTION:
        raise SegmentationError("The image background is not white or transparent; "
                                "this version cannot isolate glasses from it")
    if config.adaptive_white and foreground.any():
        midpoint = (np.median(darkest_channel[foreground]) + 255) / 2
        foreground = darkest_channel < min(midpoint, config.white_threshold)
    return foreground


def _remove_specks(mask):
    """Drop connected regions far smaller than the largest one."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    if count <= 1:
        return mask
    keep = areas >= areas[1:].max() * MIN_FEATURE_FRACTION
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _fill_pinholes(mask):
    """Fill tiny enclosed gaps such as highlights, leaving lens openings alone."""
    limit = np.count_nonzero(mask) * MIN_FEATURE_FRACTION
    _, labels, stats, _ = cv2.connectedComponentsWithStats(255 - mask, connectivity=4)
    pinhole = stats[:, cv2.CC_STAT_AREA] < limit
    pinhole[np.unique(_border(labels))] = False
    filled = mask.copy()
    filled[pinhole[labels]] = 255
    return filled


def segment_glasses(image, config=SegmentationConfig()):
    """Return a uint8 mask (255 = glasses, 0 = background) the size of the image.

    Enclosed background, such as clear lens openings, stays 0.
    """
    config.validate()
    opaque = image[..., 3] >= config.alpha_threshold if image.shape[2] == 4 else None
    if opaque is not None and (~_border(opaque)).mean() >= MIN_BORDER_BACKGROUND_FRACTION:
        foreground = opaque
    else:
        # No alpha, or a white image with only incidental transparency (rounded
        # corners). Colours under transparent pixels are meaningless, so ignore them.
        white_on = image[..., :3] if opaque is None else np.where(opaque[..., None],
                                                                  image[..., :3], 255)
        foreground = _foreground_from_white(white_on, config)
    if not foreground.any():
        raise SegmentationError("No glasses found: the image has no foreground pixels")

    mask = foreground.astype(np.uint8) * 255
    return _fill_pinholes(_remove_specks(mask))


def remove_opening_intrusions(mask, width_fraction):
    """Open up strips that poke into lens openings, such as temples seen through a lens.

    Each opening is morphologically closed on its own with a disk this fraction of
    the frame's width across. Closing only fills gaps bounded by that opening on
    both sides, so strips narrower than the disk go, while rims survive however
    thin they are: the far side of a rim isn't part of the opening.
    """
    if not 0 <= width_fraction <= MAX_INTRUSION_FRACTION:
        raise ValueError(f"opening cleanup must be in 0..{MAX_INTRUSION_FRACTION}")
    columns = np.flatnonzero(mask.any(axis=0))
    diameter = round(width_fraction * (columns[-1] - columns[0] + 1)) if len(columns) else 0
    if diameter < 2:
        return mask.copy()
    diameter |= 1  # an even kernel has no centre pixel, so closing would shift the shape

    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(255 - mask, connectivity=4)
    # Label 0 is the frame itself; labels on the image edge are the outside.
    openings = set(range(1, len(stats))) - set(np.unique(_border(labels)).tolist())
    cleaned = mask.copy()
    for label in sorted(openings):
        left, top, width, height, _ = stats[label]
        # Padded so the closing never meets the edge of its working window.
        window = np.pad(labels[top:top + height, left:left + width] == label, diameter)
        closed = cv2.morphologyEx(window.astype(np.uint8), cv2.MORPH_CLOSE, disk)
        y_min, x_min = top - diameter, left - diameter
        ys, xs = np.nonzero(closed)
        cleaned[ys + y_min, xs + x_min] = 0
    return cleaned
