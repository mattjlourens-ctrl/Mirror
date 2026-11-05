"""Run the real main.py loop with a fake camera, fake key presses and fake
MediaPipe output, so the whole app can be checked without a webcam."""
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import mediapipe as mp
import numpy as np

from glb_model import load_glasses_model
from head_pose import MODEL_POINTS, NO_DISTORTION, POSE_LANDMARK_IDS
from image_to_3d.assembly import GlassesParameters, export_model_glb
from tests.test_glb_model import generated_model
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


def run_app(keys, face=None, argv=()):
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
            mock.patch.object(mp.solutions.face_mesh, "FaceMesh", FakeFaceMesh), \
            mock.patch.object(sys, "argv", ["main.py", *map(str, argv)]):
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


def has_filled_glasses(frame):
    return (np.abs(frame.astype(int) - 128).sum(axis=2) > 60).sum() > 500


class GeneratedModelTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.dir = Path(directory.name)
        cls.glb = export_model_glb(generated_model(GlassesParameters()), cls.dir / "model.glb")


class ModelSelectionTests(GeneratedModelTestCase):
    def test_model_flag_starts_with_only_the_filled_glasses(self):
        app_globals, shown = run_app([NO_KEY], fake_face_landmarks(), ["--model", self.glb])
        self.assertTrue(app_globals["show_glasses_3d"])
        self.assertFalse(app_globals["show_glasses_2d"])
        self.assertTrue(app_globals["use_mesh_model"])
        self.assertTrue(has_filled_glasses(shown[0]))
        self.assertFalse(has_3d_glasses(shown[0]))  # no magenta wireframe

    def test_m_switches_to_the_wireframe_and_back(self):
        _, shown = run_app(["m", "m", NO_KEY], fake_face_landmarks(), ["--model", self.glb])
        mesh, wireframe, mesh_again = shown
        self.assertFalse(has_3d_glasses(mesh))
        self.assertTrue(has_3d_glasses(wireframe))
        self.assertFalse(has_3d_glasses(mesh_again))
        np.testing.assert_array_equal(mesh, mesh_again)

    def test_h_hides_the_generated_model_too(self):
        _, shown = run_app(["h", NO_KEY], fake_face_landmarks(), ["--model", self.glb])
        self.assertTrue(has_filled_glasses(shown[0]))
        self.assertFalse(has_filled_glasses(shown[1]))

    def test_model_is_loaded_once_not_every_frame(self):
        load_glasses_model.cache_clear()
        with mock.patch("glb_model.trimesh.load_scene", wraps=__import__("trimesh").load_scene) as parse:
            run_app([NO_KEY] * 5 + ["m", "m", NO_KEY], fake_face_landmarks(), ["--model", self.glb])
        self.assertEqual(parse.call_count, 1)

    def test_occlusion_key_affects_the_generated_model(self):
        _, shown = run_app(["o", NO_KEY], fake_face_landmarks(), ["--model", self.glb])
        self.assertFalse(np.array_equal(shown[0], shown[1]))

    def test_no_arguments_keep_the_original_behaviour(self):
        app_globals, shown = run_app([NO_KEY], fake_face_landmarks())
        self.assertFalse(app_globals["use_mesh_model"])
        self.assertFalse(app_globals["show_glasses_3d"])
        self.assertTrue(has_2d_glasses(shown[0]))

    def test_wireframe_mode_is_pixel_identical_with_or_without_a_loaded_model(self):
        _, plain = run_app(["g", "h", NO_KEY], fake_face_landmarks())
        _, with_model = run_app(["m", NO_KEY], fake_face_landmarks(), ["--model", self.glb])
        np.testing.assert_array_equal(plain[2], with_model[1])

    def test_missing_model_falls_back_to_the_wireframe_without_crashing(self):
        app_globals, shown = run_app([NO_KEY], fake_face_landmarks(),
                                     ["--model", self.dir / "absent.glb"])
        self.assertFalse(app_globals["use_mesh_model"])
        self.assertTrue(app_globals["show_glasses_3d"])
        self.assertTrue(has_3d_glasses(shown[0]))

    def test_invalid_model_falls_back_to_the_wireframe_without_crashing(self):
        bad = self.dir / "bad.glb"
        bad.write_bytes(b"not a model")
        app_globals, shown = run_app([NO_KEY], fake_face_landmarks(), ["--model", bad])
        self.assertFalse(app_globals["use_mesh_model"])
        self.assertTrue(has_3d_glasses(shown[0]))

    def test_m_without_a_usable_model_stays_on_the_wireframe(self):
        with mock.patch("glb_model.DEFAULT_MODEL_PATH", self.dir / "absent.glb"):
            app_globals, shown = run_app(["g", "m", NO_KEY], fake_face_landmarks())
        self.assertFalse(app_globals["use_mesh_model"])
        self.assertTrue(has_3d_glasses(shown[2]))

    def test_m_without_a_flag_loads_the_default_model(self):
        with mock.patch("glb_model.DEFAULT_MODEL_PATH", self.glb):
            app_globals, shown = run_app(["g", "m", NO_KEY], fake_face_landmarks())
        self.assertTrue(app_globals["use_mesh_model"])
        self.assertTrue(has_filled_glasses(shown[2]))
        self.assertFalse(has_3d_glasses(shown[2]))


class FitControlTests(GeneratedModelTestCase):
    def test_fit_keys_move_the_model_while_running(self):
        app_globals, shown = run_app(["i", NO_KEY], fake_face_landmarks(), ["--model", self.glb])
        self.assertAlmostEqual(app_globals["fit"].y, 0.1)
        self.assertFalse(np.array_equal(shown[0], shown[1]))

    def test_reset_key_restores_the_first_frame(self):
        _, shown = run_app(["i", "+", "r", NO_KEY], fake_face_landmarks(), ["--model", self.glb])
        np.testing.assert_array_equal(shown[0], shown[3])

    def test_fit_flags_set_the_starting_fit(self):
        app_globals, _ = run_app([NO_KEY], fake_face_landmarks(),
                                 ["--model", self.glb, "--fit-scale", 1.1, "--fit-y", -0.4,
                                  "--fit-yaw", 3])
        self.assertEqual(app_globals["fit"].scale, 1.1)
        self.assertEqual(app_globals["fit"].y, -0.4)
        self.assertEqual(app_globals["fit"].yaw, 3.0)

    def test_fit_flags_are_used_from_the_first_frame(self):
        _, plain = run_app([NO_KEY], fake_face_landmarks(), ["--model", self.glb])
        _, fitted = run_app([NO_KEY], fake_face_landmarks(), ["--model", self.glb, "--fit-y", 1.0])
        self.assertFalse(np.array_equal(plain[0], fitted[0]))

    def test_invalid_fit_flag_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as stop:
            run_app([NO_KEY], fake_face_landmarks(), ["--fit-scale", 0])
        self.assertEqual(stop.exception.code, 2)

    def test_fit_does_not_change_the_wireframe(self):
        _, plain = run_app(["g", "h", NO_KEY], fake_face_landmarks())
        _, moved = run_app(["g", "h", "i", NO_KEY], fake_face_landmarks())
        np.testing.assert_array_equal(plain[2], moved[3])

    def test_debug_panel_with_the_generated_model_does_not_crash(self):
        _, shown = run_app(["d", NO_KEY], fake_face_landmarks(), ["--model", self.glb])
        self.assertTrue(np.array_equal(shown[1][10, 10], (0, 0, 0)))

    def test_frame_rate_is_measured_from_the_loop(self):
        clock = iter(float(i) / 20 for i in range(100))  # one frame every 50 ms
        with mock.patch("time.perf_counter", lambda: next(clock)):
            app_globals, _ = run_app([NO_KEY] * 6)
        self.assertAlmostEqual(app_globals["fps"], 20.0, places=3)

    def test_frame_rate_starts_unknown(self):
        app_globals, _ = run_app(["q"])
        self.assertIsNone(app_globals["fps"])

    def test_f_prints_the_current_fit_flags(self):
        with mock.patch("builtins.print") as printed:
            run_app(["i", "f", NO_KEY], fake_face_landmarks(), ["--model", self.glb])
        self.assertTrue(any("--fit-y 0.1" in str(call) for call in printed.call_args_list))


if __name__ == "__main__":
    unittest.main()
