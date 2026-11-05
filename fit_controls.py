"""Live fit adjustments: where a loaded model sits on the face.

A fit is applied as a matrix when rendering and never changes the model's mesh.
Offsets are in face-local cm (X screen-right, Y up, Z out of the face). Scale and
rotation pivot on the model's origin, the centre of the frame's front.
"""
from dataclasses import dataclass, replace

import cv2
import numpy as np

from accessory_transform import attachment_matrix, linear_matrix

MAX_OFFSET_CM = 5.0
MAX_ROTATION_DEGREES = 30.0
MIN_SCALE = 0.5
MAX_SCALE = 2.0

OFFSET_STEP_CM = 0.1
SCALE_STEP = 0.01
ROTATION_STEP_DEGREES = 1.0

# key: (field, change per press)
FIT_KEYS = {
    "l": ("x", OFFSET_STEP_CM), "j": ("x", -OFFSET_STEP_CM),
    "i": ("y", OFFSET_STEP_CM), "k": ("y", -OFFSET_STEP_CM),
    "u": ("z", OFFSET_STEP_CM), "n": ("z", -OFFSET_STEP_CM),
    "+": ("scale", SCALE_STEP), "=": ("scale", SCALE_STEP), "-": ("scale", -SCALE_STEP),
    "'": ("pitch", ROTATION_STEP_DEGREES), ";": ("pitch", -ROTATION_STEP_DEGREES),
    "]": ("yaw", ROTATION_STEP_DEGREES), "[": ("yaw", -ROTATION_STEP_DEGREES),
    ".": ("roll", ROTATION_STEP_DEGREES), ",": ("roll", -ROTATION_STEP_DEGREES),
}
RESET_KEY = "r"


@dataclass(frozen=True)
class FitAdjustment:
    scale: float = 1.0
    x: float = 0.0      # cm
    y: float = 0.0
    z: float = 0.0
    pitch: float = 0.0  # degrees
    yaw: float = 0.0
    roll: float = 0.0

    def validate(self):
        if not MIN_SCALE <= self.scale <= MAX_SCALE:
            raise ValueError(f"fit scale must be {MIN_SCALE} to {MAX_SCALE}, got {self.scale}")
        for name in ("x", "y", "z"):
            _check_limit(name, getattr(self, name), MAX_OFFSET_CM)
        for name in ("pitch", "yaw", "roll"):
            _check_limit(name, getattr(self, name), MAX_ROTATION_DEGREES)


def _check_limit(name, value, limit):
    # Written so that NaN fails too.
    if not -limit <= value <= limit:
        raise ValueError(f"fit {name} must be within +/-{limit}, got {value}")


def _limit_of(field):
    if field == "scale":
        return MIN_SCALE, MAX_SCALE
    limit = MAX_OFFSET_CM if field in ("x", "y", "z") else MAX_ROTATION_DEGREES
    return -limit, limit


def fit_matrix(attachment, fit):
    """Accessory-local (cm) -> face-local 4x4 matrix with the fit applied."""
    yaw, _ = cv2.Rodrigues(np.radians((0.0, fit.yaw, 0.0)))
    pitch, _ = cv2.Rodrigues(np.radians((fit.pitch, 0.0, 0.0)))
    roll, _ = cv2.Rodrigues(np.radians((0.0, 0.0, fit.roll)))
    moved = replace(attachment, offset_cm=tuple(np.add(attachment.offset_cm, (fit.x, fit.y, fit.z))))
    return attachment_matrix(moved) @ linear_matrix(fit.scale * (yaw @ pitch @ roll))


def apply_fit_key(fit, key):
    """The new fit after a key press, or None if the key isn't a fit control."""
    if not 0 <= key < 128:
        return None
    character = chr(key).lower()
    if character == RESET_KEY:
        return FitAdjustment()
    if character not in FIT_KEYS:
        return None
    field, step = FIT_KEYS[character]
    low, high = _limit_of(field)
    # Rounding stops repeated steps collecting float noise such as 0.30000000000000004.
    value = round(min(high, max(low, getattr(fit, field) + step)), 6)
    return replace(fit, **{field: value})


def fit_flags(fit):
    """Command line flags reproducing this fit; empty for the default."""
    default = FitAdjustment()
    return " ".join(f"--fit-{name} {getattr(fit, name):g}"
                    for name in ("scale", "x", "y", "z", "pitch", "yaw", "roll")
                    if getattr(fit, name) != getattr(default, name))
