import argparse
from dataclasses import fields
import math
import sys
import time

import cv2
import mediapipe as mp

from accessory_transform import attachment_matrix
from fit_controls import FitAdjustment, apply_fit_key, fit_flags, fit_matrix
from geometry import distance, midpoint, tilt_angle, transform_point
from glasses import draw_glasses
from glasses_3d import DEFAULT_GLASSES_ATTACHMENT, build_glasses_model, draw_wireframe
from glb_model import ModelLoadError, load_glasses_model
from head_pose import (camera_matrix, draw_pose_axes, estimate_head_pose,
                       euler_angles, image_points_from_landmarks)
from mesh_renderer import render_mesh
from occlusion import load_head_occluder, render_depth_buffer

# Glasses attachment landmarks (MediaPipe Face Mesh IDs).
# The frame is flipped before it reaches MediaPipe, so "left" and "right"
# describe where each point appears on screen, not which of your eyes it is:
#   33  -> outer eye corner on the LEFT of the screen  (left_point)
#   263 -> outer eye corner on the RIGHT of the screen (right_point)
#   168 -> nose bridge, between the eyes               (bridge_point)
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
NOSE_BRIDGE = 168

BLUE = (255, 0, 0)  # OpenCV colours are BGR, not RGB
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Glasses frames are wider than the distance between the outer eye corners.
# frame width (px) = eye distance (px) * GLASSES_WIDTH_SCALE
GLASSES_WIDTH_SCALE = 1.5

# Weight of the newest frame in the displayed frame rate, which keeps it readable.
FPS_SMOOTHING = 0.1

# Shift from the bridge landmark to the glasses' centre, in glasses-local
# units (1.0 = frame width). Positive y moves the glasses down the face.
# Because it is in local units, the offset scales and tilts with the face.
GLASSES_OFFSET = (0.0, 0.05)


def landmark_to_pixel(landmark, width, height):
    """Convert a normalized MediaPipe landmark (0.0-1.0) to (x, y) pixels."""
    return int(landmark.x * width), int(landmark.y * height)


def draw_debug(frame, left_point, right_point, bridge_point, eye_midpoint,
               eye_distance, angle, glasses_center, pose, glasses_status):
    """Draw tracked landmarks and measurements for debugging."""
    cv2.circle(frame, left_point, 4, BLUE, -1)
    cv2.circle(frame, right_point, 4, BLUE, -1)
    cv2.circle(frame, bridge_point, 4, GREEN, -1)
    cv2.circle(frame, (round(eye_midpoint[0]), round(eye_midpoint[1])), 4, WHITE, 1)

    lines = [
        f"eye distance: {eye_distance:.1f} px",
        f"tilt: {math.degrees(angle):+.1f} deg",
        f"glasses center: ({glasses_center[0]:.0f}, {glasses_center[1]:.0f})",
        *glasses_status.split("\n"),
    ]
    if pose is None:
        lines.append("3D pose: not available")
    else:
        yaw, pitch, roll = euler_angles(pose.rotation_matrix)
        tx, ty, tz = pose.translation_vector.ravel()
        lines.append(f"yaw {yaw:+.0f}  pitch {pitch:+.0f}  roll {roll:+.0f} deg")
        lines.append(f"translation: ({tx:.1f}, {ty:.1f}, {tz:.1f}) cm*")
    cv2.rectangle(frame, (8, 8), (530, 16 + 22 * len(lines)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (16, 30 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, WHITE, 1, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser(description="Mirror: real-time virtual eyewear try-on.")
    parser.add_argument("--model", metavar="PATH",
                        help="start with this generated .glb glasses model (M switches models)")
    for field in fields(FitAdjustment):
        parser.add_argument(f"--fit-{field.name}", type=float, default=field.default,
                            help="starting fit, as printed by the F key (default: %(default)s)")
    args = parser.parse_args()
    fit = FitAdjustment(**{field.name: getattr(args, f"fit_{field.name}")
                           for field in fields(FitAdjustment)})
    try:
        fit.validate()
    except ValueError as error:
        parser.error(str(error))
    return args.model, fit


def load_mesh_model(path):
    """The model at path (None: the default model), or None with a message on failure."""
    try:
        return load_glasses_model(path)
    except ModelLoadError as error:
        print(f"Could not load the glasses model: {error}\n"
              "Using the wireframe glasses. Create a model with:\n"
              "  uv run python image_to_3d/mirror_image_to_3d.py image_to_3d/glasses.jpg "
              "--complete -o image_to_3d/output/example_3_photo.glb", file=sys.stderr)
        return None


model_path, fit = parse_args()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Static 3D model and its attachment are built once, not every frame.
glasses_model = build_glasses_model()
glasses_to_face = attachment_matrix(DEFAULT_GLASSES_ATTACHMENT,
                                    glasses_model.units_to_cm,
                                    glasses_model.orientation)
head_occluder = load_head_occluder()  # None if the model file is missing
mesh_model = load_mesh_model(model_path) if model_path else None
mesh_to_face = fit_matrix(DEFAULT_GLASSES_ATTACHMENT, fit)

camera = cv2.VideoCapture(0)
# Asking for a model means asking for 3D glasses, even if it then fails to load.
show_glasses_2d = model_path is None   # G
show_glasses_3d = model_path is not None  # H
use_mesh_model = mesh_model is not None  # M: filled model instead of the wireframe
show_occlusion = True    # O: hide 3D glasses parts that are behind the head
show_pose_axes = False  # P
debug_mode = False    # D
camera_intrinsics = None
intrinsics_size = None  # rebuild the camera matrix only if the frame size changes
required_landmarks = max(LEFT_EYE_OUTER, RIGHT_EYE_OUTER, NOSE_BRIDGE) + 1
fps = None  # measured over the whole loop, shown in the debug panel
last_frame_time = None

while True:
    success, frame = camera.read()

    if not success:
        break

    now = time.perf_counter()
    if last_frame_time is not None and now > last_frame_time:
        latest = 1.0 / (now - last_frame_time)
        fps = latest if fps is None else fps + FPS_SMOOTHING * (latest - fps)
    last_frame_time = now

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    if intrinsics_size != (width, height):
        camera_intrinsics = camera_matrix(width, height)
        intrinsics_size = (width, height)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    landmarks = []
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

    if len(landmarks) >= required_landmarks:
        left_point = landmark_to_pixel(landmarks[LEFT_EYE_OUTER], width, height)
        right_point = landmark_to_pixel(landmarks[RIGHT_EYE_OUTER], width, height)
        bridge_point = landmark_to_pixel(landmarks[NOSE_BRIDGE], width, height)

        eye_distance = distance(left_point, right_point)
        eye_midpoint = midpoint(left_point, right_point)  # reference only
        angle = tilt_angle(left_point, right_point)

        # Only run the 3D solver when something will use the result.
        pose = None
        if show_pose_axes or debug_mode or show_glasses_3d:
            image_points = image_points_from_landmarks(landmarks, width, height)
            pose = estimate_head_pose(image_points, camera_intrinsics)

        # A zero distance would collapse the glasses to a single point.
        if eye_distance > 0:
            glasses_width = eye_distance * GLASSES_WIDTH_SCALE
            glasses_center = transform_point(GLASSES_OFFSET, glasses_width, angle,
                                             bridge_point)
            if show_glasses_2d:
                draw_glasses(frame, glasses_center, glasses_width, angle)

        if show_glasses_3d:
            depth_buffer = None
            if show_occlusion:
                depth_buffer = render_depth_buffer(head_occluder, pose, width, height)
            if use_mesh_model:
                render_mesh(frame, mesh_model, mesh_to_face, pose, depth_buffer)
            else:
                draw_wireframe(frame, glasses_model, glasses_to_face, pose, depth_buffer)

        if debug_mode and eye_distance > 0:
            status = (f"2D glasses [G]: {'on' if show_glasses_2d else 'off'}  "
                      f"3D glasses [H]: {'on' if show_glasses_3d else 'off'}  "
                      f"occlusion [O]: {'on' if show_occlusion else 'off'}\n"
                      f"model [M]: {mesh_model.name if use_mesh_model else 'wireframe'}  "
                      f"{width}x{height}  "
                      f"{'measuring fps' if fps is None else f'{fps:.0f} fps'}\n"
                      f"fit: {fit_flags(fit) or 'default'}")
            draw_debug(frame, left_point, right_point, bridge_point,
                       eye_midpoint, eye_distance, angle, glasses_center, pose, status)

        if show_pose_axes and pose is not None:
            draw_pose_axes(frame, pose)

    cv2.imshow("Mirror", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q")):
        break
    if key in (ord("d"), ord("D")):
        debug_mode = not debug_mode
    if key in (ord("g"), ord("G")):
        show_glasses_2d = not show_glasses_2d
    if key in (ord("h"), ord("H")):
        show_glasses_3d = not show_glasses_3d
    if key in (ord("o"), ord("O")):
        show_occlusion = not show_occlusion
    if key in (ord("p"), ord("P")):
        show_pose_axes = not show_pose_axes
    if key in (ord("m"), ord("M")):
        want_mesh = not (show_glasses_3d and use_mesh_model)
        show_glasses_3d = True
        if want_mesh and mesh_model is None:
            mesh_model = load_mesh_model(None)
        use_mesh_model = want_mesh and mesh_model is not None
    if key in (ord("f"), ord("F")):
        print(f"Current fit: {fit_flags(fit) or 'default (no flags needed)'}")
    new_fit = apply_fit_key(fit, key)
    if new_fit is not None:
        fit = new_fit
        mesh_to_face = fit_matrix(DEFAULT_GLASSES_ATTACHMENT, fit)

camera.release()
cv2.destroyAllWindows()