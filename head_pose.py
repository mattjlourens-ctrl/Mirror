"""3D head pose estimation with cv2.solvePnP.

Coordinate systems
------------------
1. Image coordinates (pixels)
   Origin at the top-left of the DISPLAYED (already mirrored) frame.
   x points right, y points DOWN. MediaPipe gives landmarks normalized to
   0..1 in this same frame, so pixel = normalized * (width, height).
   MediaPipe's landmark z is a relative depth with no fixed unit; we never
   use it as a measurement.

2. Camera coordinates (centimetres, OpenCV convention)
   Origin at the camera's optical centre. X right, Y DOWN, Z forward
   (into the scene), so Z is depth and a visible face has Z > 0.
   The units come from the face model below, so distances are only as
   accurate as the face model and the guessed focal length.

3. Face / model coordinates (centimetres)
   MediaPipe's canonical face model. Right-handed, X towards the screen-right
   side of the face (landmark 263), Y UP, Z OUT of the face towards the
   camera. This matches the glTF convention (+Y up, +Z front), which is
   what a future 3D glasses model should be authored in.
   The face-aligned frame we draw uses these same axes with its origin
   moved to the nose bridge (landmark 168).

Mirroring
---------
The frame is flipped BEFORE MediaPipe sees it, and every calculation here
uses that flipped frame. MediaPipe labels the face it sees, so landmark 263
is always on screen-right and agrees with the model's +X. The pose we
estimate is the pose of the face as it appears on screen, which is what we
draw on. Flipping does not change the camera matrix because cx = width / 2.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np

# Stable, well-spread landmarks. The mouth, chin and eyelids are left out
# because they move with expressions, blinking and talking.
# Model points are copied from MediaPipe's canonical face mesh
# (geometry_pipeline_metadata_landmarks.binarypb inside face_landmarker.task).
POSE_LANDMARKS = {
    33: (-4.4459, 2.6640, 3.1734),   # outer eye corner, screen-left
    263: (4.4459, 2.6640, 3.1734),   # outer eye corner, screen-right
    133: (-1.8564, 2.5852, 3.7579),  # inner eye corner, screen-left
    362: (1.8564, 2.5852, 3.7579),   # inner eye corner, screen-right
    168: (0.0, 3.2710, 5.2360),      # nose bridge (glasses anchor)
    6: (0.0, 2.4733, 5.7886),        # upper nose
    1: (0.0, -1.1269, 7.4756),       # nose tip
    98: (-1.4056, -1.7142, 5.2411),  # nostril, screen-left
    327: (1.4056, -1.7142, 5.2411),  # nostril, screen-right
    151: (0.0, 6.5454, 5.0273),      # mid forehead
}
POSE_LANDMARK_IDS = list(POSE_LANDMARKS)
MODEL_POINTS = np.array(list(POSE_LANDMARKS.values()), dtype=np.float64)

# Face-aligned frame: origin at the nose bridge, axes along the model axes.
FACE_ORIGIN = np.array(POSE_LANDMARKS[168], dtype=np.float64)
AXIS_LENGTH_CM = 4.0
AXIS_POINTS = FACE_ORIGIN + np.array([
    [0.0, 0.0, 0.0],
    [AXIS_LENGTH_CM, 0.0, 0.0],  # X: across the face, towards screen-right
    [0.0, AXIS_LENGTH_CM, 0.0],  # Y: up the face
    [0.0, 0.0, AXIS_LENGTH_CM],  # Z: out of the face
])

# Assumed focal length, in pixels, as a multiple of the frame width.
# 1.0 means a horizontal field of view of about 53 degrees. Real webcams
# differ, so translation (especially depth) is approximate.
FOCAL_LENGTH_SCALE = 1.0

# Reject poses whose average reprojection error is larger than this fraction
# of the face's size in the image.
MAX_REPROJECTION_ERROR_RATIO = 0.1

# Model -> camera rotation for a face looking straight at the camera.
# The model is Y-up/Z-towards-camera and the camera is Y-down/Z-forward,
# so a frontal face is a 180 degree turn about X.
FRONTAL_ROTATION = np.diag([1.0, -1.0, -1.0])

NO_DISTORTION = np.zeros(4)

RED = (0, 0, 255)  # BGR
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


@dataclass(frozen=True)
class HeadPose:
    """Transform from face-model coordinates (cm) to camera coordinates (cm).

    point_camera = rotation_matrix @ point_model + translation_vector

    A failed estimate is represented by None, not by a HeadPose, so a pose
    object always holds valid numbers.
    """
    rotation_vector: np.ndarray     # (3, 1) axis * angle in radians
    translation_vector: np.ndarray  # (3, 1) model origin in camera space, cm
    rotation_matrix: np.ndarray     # (3, 3)
    camera_matrix: np.ndarray       # (3, 3) intrinsics used for the estimate

    @property
    def transform(self):
        """4x4 model -> camera matrix, the form a 3D renderer usually wants."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation_matrix
        matrix[:3, 3] = self.translation_vector.ravel()
        return matrix


def camera_matrix(width, height):
    """Approximate pinhole intrinsics for an uncalibrated webcam."""
    focal = FOCAL_LENGTH_SCALE * width
    return np.array([
        [focal, 0.0, width / 2],
        [0.0, focal, height / 2],
        [0.0, 0.0, 1.0],
    ])


def image_points_from_landmarks(landmarks, width, height):
    """Pixel positions (float, N x 2) of POSE_LANDMARK_IDS, or None.

    Floats are kept (not rounded like landmark_to_pixel) because sub-pixel
    accuracy helps solvePnP.
    """
    if len(landmarks) <= max(POSE_LANDMARK_IDS):
        return None
    return np.array(
        [(landmarks[i].x * width, landmarks[i].y * height) for i in POSE_LANDMARK_IDS],
        dtype=np.float64,
    )


def estimate_head_pose(image_points, camera):
    """Solve for the head pose, or return None if it can't be trusted."""
    if image_points is None:
        return None
    image_points = np.asarray(image_points, dtype=np.float64)
    if image_points.shape != (len(MODEL_POINTS), 2) or not np.isfinite(image_points).all():
        return None

    try:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS, image_points, camera, NO_DISTORTION,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        return None

    if not success or not np.isfinite(rotation_vector).all() \
            or not np.isfinite(translation_vector).all():
        return None
    if translation_vector[2, 0] <= 0:  # face must be in front of the camera
        return None

    projected, _ = cv2.projectPoints(MODEL_POINTS, rotation_vector,
                                     translation_vector, camera, NO_DISTORTION)
    error = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1).mean()
    face_size = np.linalg.norm(image_points.max(axis=0) - image_points.min(axis=0))
    if face_size == 0 or error > MAX_REPROJECTION_ERROR_RATIO * face_size:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return HeadPose(rotation_vector, translation_vector, rotation_matrix, camera)


def euler_angles(rotation_matrix):
    """(yaw, pitch, roll) in degrees, relative to looking straight at the camera.

    Signs, as seen on screen:
      yaw   > 0  nose turns towards screen-right
      pitch > 0  nose turns up
      roll  > 0  head tilts clockwise (same sign as the 2D tilt)

    The head's rotation relative to frontal is expressed in viewer axes
    (X right, Y up, Z towards the viewer) and decomposed in Y-X-Z order:
        R = Ry(yaw) @ Rx(-pitch) @ Rz(-roll)
    Yaw is applied last, about the vertical axis. This is reliable while
    |pitch| < 90 degrees, which a real head never reaches.
    """
    # FRONTAL_ROTATION also converts camera axes (Y down, Z forward) to
    # viewer axes (Y up, Z towards the viewer); it is its own inverse.
    r = FRONTAL_ROTATION @ rotation_matrix
    yaw = math.atan2(r[0, 2], r[2, 2])
    pitch = math.asin(np.clip(r[1, 2], -1.0, 1.0))  # = -asin(-r[1, 2])
    roll = -math.atan2(r[1, 0], r[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def project_face_axes(pose):
    """Pixel positions (4 x 2) of the face frame's origin, X, Y and Z ends."""
    projected, _ = cv2.projectPoints(AXIS_POINTS, pose.rotation_vector,
                                     pose.translation_vector, pose.camera_matrix,
                                     NO_DISTORTION)
    return projected.reshape(-1, 2)


def draw_pose_axes(frame, pose):
    """Draw the face-aligned frame: X red, Y green, Z blue."""
    origin, x_end, y_end, z_end = np.round(project_face_axes(pose)).astype(int)
    origin = tuple(origin)
    cv2.line(frame, origin, tuple(z_end), BLUE, 3, cv2.LINE_AA)
    cv2.line(frame, origin, tuple(y_end), GREEN, 3, cv2.LINE_AA)
    cv2.line(frame, origin, tuple(x_end), RED, 3, cv2.LINE_AA)
