import cv2
import mediapipe as mp

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


def landmark_to_pixel(landmark, width, height):
    """Convert a normalized MediaPipe landmark (0.0-1.0) to (x, y) pixels."""
    return int(landmark.x * width), int(landmark.y * height)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        landmarks = face_landmarks.landmark

        left_point = landmark_to_pixel(landmarks[LEFT_EYE_OUTER], width, height)
        right_point = landmark_to_pixel(landmarks[RIGHT_EYE_OUTER], width, height)
        bridge_point = landmark_to_pixel(landmarks[NOSE_BRIDGE], width, height)

        cv2.circle(frame, left_point, 6, BLUE, -1)
        cv2.circle(frame, right_point, 6, BLUE, -1)
        cv2.circle(frame, bridge_point, 6, GREEN, -1)

    cv2.imshow("Mirror", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()