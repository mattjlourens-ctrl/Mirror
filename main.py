import math

import cv2
import mediapipe as mp

from geometry import distance, midpoint, tilt_angle, transform_point
from glasses import draw_glasses

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

# Shift from the bridge landmark to the glasses' centre, in glasses-local
# units (1.0 = frame width). Positive y moves the glasses down the face.
# Because it is in local units, the offset scales and tilts with the face.
GLASSES_OFFSET = (0.0, 0.05)


def landmark_to_pixel(landmark, width, height):
    """Convert a normalized MediaPipe landmark (0.0-1.0) to (x, y) pixels."""
    return int(landmark.x * width), int(landmark.y * height)


def draw_debug(frame, left_point, right_point, bridge_point, eye_midpoint,
               eye_distance, angle, glasses_center):
    """Draw tracked landmarks and measurements for debugging."""
    cv2.circle(frame, left_point, 4, BLUE, -1)
    cv2.circle(frame, right_point, 4, BLUE, -1)
    cv2.circle(frame, bridge_point, 4, GREEN, -1)
    cv2.circle(frame, (round(eye_midpoint[0]), round(eye_midpoint[1])), 4, WHITE, 1)

    lines = [
        f"eye distance: {eye_distance:.1f} px",
        f"tilt: {math.degrees(angle):+.1f} deg",
        f"glasses center: ({glasses_center[0]:.0f}, {glasses_center[1]:.0f})",
    ]
    cv2.rectangle(frame, (8, 8), (270, 16 + 22 * len(lines)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (16, 30 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, WHITE, 1, cv2.LINE_AA)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

camera = cv2.VideoCapture(0)
debug_mode = False
required_landmarks = max(LEFT_EYE_OUTER, RIGHT_EYE_OUTER, NOSE_BRIDGE) + 1

while True:
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

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

        # A zero distance would collapse the glasses to a single point.
        if eye_distance > 0:
            glasses_width = eye_distance * GLASSES_WIDTH_SCALE
            glasses_center = transform_point(GLASSES_OFFSET, glasses_width, angle,
                                             bridge_point)
            draw_glasses(frame, glasses_center, glasses_width, angle)

            if debug_mode:
                draw_debug(frame, left_point, right_point, bridge_point,
                           eye_midpoint, eye_distance, angle, glasses_center)

    cv2.imshow("Mirror", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q")):
        break
    if key in (ord("d"), ord("D")):
        debug_mode = not debug_mode

camera.release()
cv2.destroyAllWindows()