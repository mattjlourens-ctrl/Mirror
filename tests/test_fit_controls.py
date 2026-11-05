import unittest

import numpy as np

from accessory_transform import attachment_matrix, transform_points
from fit_controls import FitAdjustment, apply_fit_key, fit_flags, fit_matrix
from glasses_3d import DEFAULT_GLASSES_ATTACHMENT
from head_pose import FACE_ORIGIN


class FitMatrixTests(unittest.TestCase):
    ORIGIN = np.zeros((1, 3))

    def transform(self, fit, point=(0.0, 0.0, 0.0)):
        matrix = fit_matrix(DEFAULT_GLASSES_ATTACHMENT, fit)
        return transform_points(matrix, np.array([point], float))[0]

    def test_default_fit_equals_the_plain_attachment(self):
        np.testing.assert_allclose(fit_matrix(DEFAULT_GLASSES_ATTACHMENT, FitAdjustment()),
                                   attachment_matrix(DEFAULT_GLASSES_ATTACHMENT))

    def test_translations_move_along_face_axes(self):
        base = self.transform(FitAdjustment())
        np.testing.assert_allclose(self.transform(FitAdjustment(x=1.0)) - base, (1, 0, 0))
        np.testing.assert_allclose(self.transform(FitAdjustment(y=-0.5)) - base, (0, -0.5, 0))
        np.testing.assert_allclose(self.transform(FitAdjustment(z=0.25)) - base, (0, 0, 0.25))

    def test_scale_grows_about_the_model_origin(self):
        base = self.transform(FitAdjustment())
        scaled = self.transform(FitAdjustment(scale=1.5), (2.0, 0.0, 0.0))
        np.testing.assert_allclose(scaled - base, (3.0, 0.0, 0.0))
        # The origin itself stays on the anchor.
        np.testing.assert_allclose(self.transform(FitAdjustment(scale=1.5)), base)

    def test_yaw_turns_the_right_side_backwards(self):
        base = self.transform(FitAdjustment())
        moved = self.transform(FitAdjustment(yaw=90.0), (1.0, 0.0, 0.0)) - base
        np.testing.assert_allclose(moved, (0, 0, -1), atol=1e-9)

    def test_pitch_turns_the_top_towards_the_viewer(self):
        base = self.transform(FitAdjustment())
        moved = self.transform(FitAdjustment(pitch=90.0), (0.0, 1.0, 0.0)) - base
        np.testing.assert_allclose(moved, (0, 0, 1), atol=1e-9)

    def test_roll_turns_the_right_side_upwards(self):
        base = self.transform(FitAdjustment())
        moved = self.transform(FitAdjustment(roll=90.0), (1.0, 0.0, 0.0)) - base
        np.testing.assert_allclose(moved, (0, 1, 0), atol=1e-9)

    def test_anchor_stays_on_the_nose_bridge(self):
        np.testing.assert_allclose(self.transform(FitAdjustment()),
                                   FACE_ORIGIN + DEFAULT_GLASSES_ATTACHMENT.offset_cm)


class ValidationTests(unittest.TestCase):
    def test_bad_values_rejected(self):
        for bad in (dict(scale=0), dict(scale=-1), dict(scale=10), dict(x=float("nan")),
                    dict(y=float("inf")), dict(z=50), dict(yaw=200), dict(pitch=float("nan")),
                    dict(roll=-90)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    FitAdjustment(**bad).validate()

    def test_defaults_are_valid(self):
        FitAdjustment().validate()


class KeyTests(unittest.TestCase):
    def test_each_key_changes_only_its_own_value(self):
        cases = {"i": ("y", 0.1), "k": ("y", -0.1), "l": ("x", 0.1), "j": ("x", -0.1),
                 "u": ("z", 0.1), "n": ("z", -0.1), "+": ("scale", 1.01), "=": ("scale", 1.01),
                 "-": ("scale", 0.99), "]": ("yaw", 1.0), "[": ("yaw", -1.0),
                 ";": ("pitch", -1.0), "'": ("pitch", 1.0), ",": ("roll", -1.0), ".": ("roll", 1.0)}
        for key, (field, value) in cases.items():
            with self.subTest(key=key):
                fit = apply_fit_key(FitAdjustment(), ord(key))
                self.assertAlmostEqual(getattr(fit, field), value)
                others = {name: getattr(fit, name) for name in ("x", "y", "z", "pitch", "yaw", "roll")
                          if name != field}
                self.assertTrue(all(v == 0 for v in others.values()))
                if field != "scale":
                    self.assertEqual(fit.scale, 1.0)

    def test_uppercase_works(self):
        self.assertAlmostEqual(apply_fit_key(FitAdjustment(), ord("I")).y, 0.1)

    def test_other_keys_return_none(self):
        for key in (ord("q"), ord("d"), ord("g"), ord("h"), ord("o"), ord("p"), ord("m"), ord("f"), 255):
            self.assertIsNone(apply_fit_key(FitAdjustment(), key))

    def test_reset_key_restores_defaults(self):
        self.assertEqual(apply_fit_key(FitAdjustment(x=2.0, scale=1.2, yaw=4.0), ord("r")),
                         FitAdjustment())

    def test_values_clamp_at_their_limits_and_stay_valid(self):
        fit = FitAdjustment()
        for _ in range(500):
            fit = apply_fit_key(fit, ord("+"))
            fit = apply_fit_key(fit, ord("l"))
            fit = apply_fit_key(fit, ord("]"))
        fit.validate()
        self.assertLessEqual(fit.scale, 2.0)
        self.assertLessEqual(fit.x, 5.0)
        self.assertLessEqual(fit.yaw, 30.0)

    def test_repeated_steps_do_not_accumulate_float_noise(self):
        fit = FitAdjustment()
        for _ in range(3):
            fit = apply_fit_key(fit, ord("l"))
        self.assertEqual(fit.x, 0.3)

    def test_keys_return_a_new_fit_and_leave_the_old_one_alone(self):
        original = FitAdjustment()
        apply_fit_key(original, ord("l"))
        self.assertEqual(original.x, 0.0)


class FlagTests(unittest.TestCase):
    def test_default_fit_needs_no_flags(self):
        self.assertEqual(fit_flags(FitAdjustment()), "")

    def test_only_changed_values_are_listed(self):
        self.assertEqual(fit_flags(FitAdjustment(y=-0.3, scale=1.05)),
                         "--fit-scale 1.05 --fit-y -0.3")


if __name__ == "__main__":
    unittest.main()
