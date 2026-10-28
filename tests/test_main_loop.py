"""Run the real main.py loop with a fake camera, fake key presses and fake
MediaPipe output, so the whole app can be checked without a webcam."""
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import mediapipe as mp
import numpy as np

from head_pose import MODEL_POINTS, NO_DISTORTION, POSE_LANDMARK_IDS
from tests.test_head_pose import CAMERA, HEIGHT, WIDTH, FRONTAL_ROTATION, make_pose

MAIN = str(Path(__file__).resolve().parent.parent / "main.py")
NO_KEY = 255


def fake_face_landmarks():
    """478 normalized landmarks of the canonical face, frontal at 45 cm."""
    pose = make_pose(FRONTAL_ROTATION, (0.0, 0.0, 45.0))
    pixels, _ = cv2.projectPoints(MODEL_POINTS, pose.rotation_vector,
                                  pose.translation_vector, CAMERA, NO_DISTORTION)
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    for landmark_id, (u, v) in zip(POSE_LANDMARK_IDS, pixels.reshape(-1, 2)):
        landmarks[landmark_id] = SimpleNamespace(x=u / WIDTH, y=v / HEIGHT)
    return landmarks


def run_app(keys, face=None):
    """Show one grey frame per key press; returns (main's globals, shown frames)."""
    frames = iter([np.full((HEIGHT, WIDTH, 3), 128, np.uint8) for _ in keys])
    key_presses = iter(ord(k) if isinstance(k, str) else k for k in keys)
    shown = []

    class FakeCamera:
        def __init__(self, *args):
            pass

        def read(self):
            frame = next(frames, None)
            return frame is not None, frame

        def release(self):
            pass

    class FakeFaceMesh:
        def __init__(self, **kwargs):
            pass

        def process(self, rgb_frame):
            faces = [SimpleNamespace(landmark=face)] if face else None
            return SimpleNamespace(multi_face_landmarks=faces)

    with mock.patch.object(cv2, "VideoCapture", FakeCamera), \
            mock.patch.object(cv2, "imshow", lambda name, frame: shown.append(frame.copy())), \
            mock.patch.object(cv2, "waitKey", lambda delay: next(key_presses, NO_KEY)), \
            mock.patch.object(cv2, "destroyAllWindows", lambda: None), \
            mock.patch.object(mp.solutions.face_mesh, "FaceMesh", FakeFaceMesh):
        app_globals = runpy.run_path(MAIN, run_name="__main__")
    return app_globals, shown


def has_2d_glasses(frame):
    return (frame < 70).all(axis=2).sum() > 50      # dark grey frame lines


def has_3d_glasses(frame):
    b, g, r = frame[..., 0], frame[..., 1], frame[..., 2]
    return ((b > 200) & (r > 200) & (g < 80)).sum() > 50  # magenta lines


class MainLoopTests(unittest.TestCase):
    def test_no_face_does_not_crash(self):
        _, shown = run_app([NO_KEY, "d", "p", "h", "o", NO_KEY])
        self.assertEqual(len(shown), 6)
        self.assertFalse(has_3d_glasses(shown[-1]))

    def test_q_quits_immediately(self):
        _, shown = run_app(["q", NO_KEY, NO_KEY])
        self.assertEqual(len(shown), 1)

    def test_toggle_keys(self):
        app_globals, _ = run_app(["d", "p", "g", "H", "O"])
        self.assertFalse(app_globals["show_occlusion"])
        self.assertTrue(app_globals["debug_mode"])
        self.assertTrue(app_globals["show_pose_axes"])
        self.assertFalse(app_globals["show_glasses_2d"])
        self.assertTrue(app_globals["show_glasses_3d"])

    def test_default_shows_only_2d_glasses(self):
        app_globals, _ = run_app([NO_KEY])
        self.assertTrue(app_globals["show_glasses_2d"])
        self.assertFalse(app_globals["show_glasses_3d"])
        self.assertTrue(app_globals["show_occlusion"])

    def test_occlusion_hides_part_of_the_3d_glasses(self):
        # 3D glasses only, occlusion on, then off.
        _, shown = run_app(["g", "h", "o", NO_KEY], face=fake_face_landmarks())
        occluded, unoccluded = shown[2], shown[3]
        self.assertTrue(has_3d_glasses(occluded))
        magenta = lambda f: ((f[..., 0] > 200) & (f[..., 2] > 200) & (f[..., 1] < 80)).sum()
        self.assertLess(magenta(occluded), magenta(unoccluded))

    def test_g_and_h_toggle_independently_with_a_face(self):
        # A key takes effect on the frame AFTER the one it was pressed on.
        _, shown = run_app(["h", "g", "h", "g", NO_KEY], face=fake_face_landmarks())
        only_2d, both, only_3d, neither, only_2d_again = shown

        self.assertTrue(has_2d_glasses(only_2d) and not has_3d_glasses(only_2d))
        self.assertTrue(has_2d_glasses(both) and has_3d_glasses(both))
        self.assertTrue(has_3d_glasses(only_3d) and not has_2d_glasses(only_3d))
        self.assertFalse(has_2d_glasses(neither) or has_3d_glasses(neither))
        self.assertTrue(has_2d_glasses(only_2d_again) and not has_3d_glasses(only_2d_again))

    def test_debug_and_axes_with_a_face(self):
        _, shown = run_app(["d", "p", NO_KEY], face=fake_face_landmarks())
        self.assertTrue(np.array_equal(shown[2][10, 10], (0, 0, 0)))  # debug panel
        red_axis = (shown[2][..., 2] > 200) & (shown[2][..., 1] < 60) & (shown[2][..., 0] < 60)
        self.assertGreater(red_axis.sum(), 20)


if __name__ == "__main__":
    unittest.main()
