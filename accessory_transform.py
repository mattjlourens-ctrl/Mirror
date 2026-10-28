"""Place an accessory model on the face and project it into the image.

Pipeline (each arrow is one 4x4 matrix, then the final projection):

    accessory-local   model units, model axes (described by the model)
        |  attachment matrix: model units -> cm, model axes -> face axes,
        v                     then move to the anchor point on the face
    face-local        cm, canonical face axes: X screen-right, Y up,
        |             Z out of the face (see head_pose.py)
        |  HeadPose.transform: R_head, t_head from solvePnP
        v
    camera            cm, OpenCV axes: X right, Y down, Z forward (depth)
        |  pinhole projection with the HeadPose's own camera matrix
        v
    image             pixels, x right, y down (mirrored frame)

The attachment matrix depends only on the model and its calibration, so it is
built once. Only the head pose changes from frame to frame.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from head_pose import FACE_ORIGIN, NO_DISTORTION

# Points closer than this to the camera plane (or behind it) are not projected.
MIN_DEPTH_CM = 1.0

ZERO_VECTOR = np.zeros(3)


@dataclass(frozen=True)
class Attachment:
    """Where an accessory's local origin sits on the face, in face-local cm.

    anchor_cm -- a fixed point on the canonical face (default: nose bridge,
                 landmark 168, the same origin as the pose axes)
    offset_cm -- per-model calibration from that anchor:
                 x = sideways, y = vertical (+ up), z = depth (+ away from face)
    """
    anchor_cm: tuple = tuple(FACE_ORIGIN)
    offset_cm: tuple = (0.0, 0.0, 0.0)


def translation_matrix(offset):
    """4x4 matrix that moves points by offset (x, y, z)."""
    matrix = np.eye(4)
    matrix[:3, 3] = offset
    return matrix


def linear_matrix(linear):
    """4x4 matrix that applies a 3x3 rotation/scale and no translation."""
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    return matrix


def attachment_matrix(attachment, units_to_cm=1.0, orientation=np.eye(3)):
    """Accessory-local -> face-local 4x4 matrix.

    orientation -- 3x3 rotation from the model's axes to Mirror's accessory
                   axes (X right, Y up, Z forward); identity if already matching
    units_to_cm -- e.g. 1.0 for a model in cm, 100.0 for a model in metres
    """
    position = np.add(attachment.anchor_cm, attachment.offset_cm)
    scale_and_orient = units_to_cm * np.asarray(orientation, dtype=np.float64)
    return translation_matrix(position) @ linear_matrix(scale_and_orient)


def transform_points(matrix, points):
    """Apply a 4x4 matrix to an (N, 3) array of points.

    Same result as appending w = 1, multiplying, and dropping w, but without
    building the (N, 4) array every frame.
    """
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def project_camera_points(camera_points, camera_matrix):
    """(N, 2) pixel positions; rows are NaN for points not in front of the camera."""
    pixels = np.full((len(camera_points), 2), np.nan)
    in_front = camera_points[:, 2] >= MIN_DEPTH_CM
    if in_front.any():
        # The points are already in camera space, so no extra rotation/translation.
        projected, _ = cv2.projectPoints(camera_points[in_front], ZERO_VECTOR,
                                         ZERO_VECTOR, camera_matrix, NO_DISTORTION)
        pixels[in_front] = projected.reshape(-1, 2)
    return pixels


def project_accessory(vertices, accessory_to_face, pose):
    """Accessory-local vertices -> (N, 2) pixels, or None without a valid pose."""
    if pose is None:
        return None
    accessory_to_camera = pose.transform @ accessory_to_face
    camera_points = transform_points(accessory_to_camera, vertices)
    return project_camera_points(camera_points, pose.camera_matrix)
