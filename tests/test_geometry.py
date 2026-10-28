import math
import unittest

import numpy as np

from geometry import distance, midpoint, tilt_angle, transform_point, transform_points
from glasses import draw_glasses


class DistanceTests(unittest.TestCase):
    def test_horizontal_points(self):
        self.assertEqual(distance((100, 200), (160, 200)), 60)

    def test_diagonal_points(self):
        # 3-4-5 right triangle
        self.assertEqual(distance((0, 0), (3, 4)), 5)

    def test_order_does_not_matter(self):
        self.assertEqual(distance((10, 20), (40, 60)), distance((40, 60), (10, 20)))


class MidpointTests(unittest.TestCase):
    def test_midpoint(self):
        self.assertEqual(midpoint((100, 200), (200, 240)), (150, 220))

    def test_midpoint_can_be_fractional(self):
        self.assertEqual(midpoint((0, 0), (5, 3)), (2.5, 1.5))


class TiltAngleTests(unittest.TestCase):
    def test_level_eyes_give_zero(self):
        self.assertEqual(tilt_angle((100, 200), (200, 200)), 0)

    def test_right_point_lower_is_positive(self):
        # y grows downward, so this is a clockwise tilt on screen
        angle = tilt_angle((100, 200), (200, 300))
        self.assertAlmostEqual(math.degrees(angle), 45)

    def test_right_point_higher_is_negative(self):
        angle = tilt_angle((100, 200), (200, 100))
        self.assertAlmostEqual(math.degrees(angle), -45)

    def test_swapped_points_do_not_flip_180_degrees(self):
        self.assertAlmostEqual(tilt_angle((200, 300), (100, 200)),
                               tilt_angle((100, 200), (200, 300)))


class TransformTests(unittest.TestCase):
    def assertPointAlmostEqual(self, actual, expected):
        self.assertAlmostEqual(actual[0], expected[0])
        self.assertAlmostEqual(actual[1], expected[1])

    def test_zero_rotation_only_scales_and_translates(self):
        result = transform_point((0.5, -0.1), 200, 0, (320, 240))
        self.assertPointAlmostEqual(result, (420, 220))

    def test_positive_rotation_turns_clockwise_on_screen(self):
        # Pointing right, rotated +90 degrees, now points down (+y).
        result = transform_point((1, 0), 1, math.radians(90), (0, 0))
        self.assertPointAlmostEqual(result, (0, 1))

    def test_negative_rotation_turns_counterclockwise_on_screen(self):
        result = transform_point((1, 0), 1, math.radians(-90), (0, 0))
        self.assertPointAlmostEqual(result, (0, -1))

    def test_origin_maps_to_translation(self):
        result = transform_point((0, 0), 300, math.radians(30), (50, 60))
        self.assertPointAlmostEqual(result, (50, 60))

    def test_transform_points_keeps_order(self):
        result = transform_points([(0, 0), (1, 0)], 10, 0, (5, 5))
        self.assertEqual(result, [(5, 5), (15, 5)])


class DrawGlassesSmokeTest(unittest.TestCase):
    def test_draws_on_blank_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_glasses(frame, (320, 240), 200, math.radians(10))
        self.assertTrue(frame.any())


if __name__ == "__main__":
    unittest.main()
