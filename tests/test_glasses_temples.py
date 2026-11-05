import math
import unittest

import numpy as np

from image_to_3d.sweep import (cubic_bezier, flatten_cubics, polyline_length, section_ring,
                               sweep_section)
from image_to_3d.temples import TempleParameters, build_temple, temple_centerline
from image_to_3d.validation import self_intersects


def straight_params(**overrides):
    values = dict(splay=0.0, curvature=0.0, ear_drop=0.0)
    values.update(overrides)
    return TempleParameters(**values)


class BezierTests(unittest.TestCase):
    CONTROL = np.array([(0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 1, 0)], float)

    def test_curve_starts_and_ends_on_its_end_points(self):
        np.testing.assert_allclose(cubic_bezier(self.CONTROL, 0.0), self.CONTROL[0])
        np.testing.assert_allclose(cubic_bezier(self.CONTROL, 1.0), self.CONTROL[3])

    def test_midpoint_matches_the_bernstein_weights(self):
        # (P0 + 3 P1 + 3 P2 + P3) / 8
        np.testing.assert_allclose(cubic_bezier(self.CONTROL, 0.5), (1.5, 0.5, 0.0))

    def test_flattening_a_straight_cubic_gives_two_points(self):
        control = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], float)
        self.assertEqual(len(flatten_cubics([control], 0.01)), 2)

    def test_flattening_stays_within_tolerance(self):
        tolerance = 0.005
        points = flatten_cubics([self.CONTROL], tolerance)
        samples = cubic_bezier(self.CONTROL, np.linspace(0, 1, 400))
        starts, ends = points[:-1], points[1:]
        for sample in samples:
            along = np.clip(((sample - starts) * (ends - starts)).sum(axis=1)
                            / ((ends - starts) ** 2).sum(axis=1), 0, 1)
            nearest = starts + along[:, None] * (ends - starts)
            self.assertLess(np.linalg.norm(sample - nearest, axis=1).min(), tolerance * 1.01)

    def test_tighter_tolerance_uses_more_points(self):
        coarse = flatten_cubics([self.CONTROL], 0.05)
        fine = flatten_cubics([self.CONTROL], 0.001)
        self.assertLess(len(coarse), len(fine))

    def test_joined_curves_do_not_repeat_the_join(self):
        second = self.CONTROL + (3, 1, 0)
        points = flatten_cubics([self.CONTROL, second], 0.01)
        self.assertGreater(np.linalg.norm(np.diff(points, axis=0), axis=1).min(), 0)

    def test_polyline_length(self):
        self.assertAlmostEqual(polyline_length(np.array([(0, 0, 0), (3, 4, 0), (3, 4, 2)], float)),
                               7.0)


class SweepTests(unittest.TestCase):
    def tube(self, path, scales=None):
        return sweep_section(np.array(path, float), section_ring(0.5, 0.25), scales)

    def test_section_ring_has_the_requested_half_sizes(self):
        ring = section_ring(0.5, 0.25)
        self.assertAlmostEqual(ring[:, 0].max(), 0.5)
        self.assertAlmostEqual(ring[:, 1].max(), 0.25)

    def test_straight_tube_is_a_closed_volume_with_outward_normals(self):
        tube = self.tube([(0, 0, 0), (0, 0, -5)])
        self.assertTrue(tube.is_volume)
        self.assertGreater(tube.volume, 0)

    def test_first_section_axis_points_up(self):
        tube = self.tube([(0, 0, 0), (0, 0, -5)])
        self.assertAlmostEqual(tube.extents[1], 1.0)   # first axis (+-0.5) along Y
        self.assertAlmostEqual(tube.extents[0], 0.5)   # second axis (+-0.25) along X

    def test_bent_tube_keeps_its_section_without_twisting(self):
        theta = np.linspace(0, math.pi / 2, 12)
        path = np.stack((3 * np.cos(theta), np.zeros(12), -3 * np.sin(theta)), axis=1)
        tube = self.tube(path)
        self.assertTrue(tube.is_volume)
        # A bend within the XZ plane must keep the section's height along Y everywhere.
        self.assertAlmostEqual(tube.extents[1], 1.0, places=6)

    def test_path_turning_straight_down_stays_a_valid_untwisted_tube(self):
        # Heading along -Y, there's no "up" left to orient against, only the last frame.
        theta = np.linspace(0, math.pi / 2, 12)
        path = np.stack((np.zeros(12), 3 * np.cos(theta) - 3, -3 * np.sin(theta)), axis=1)
        tube = self.tube(path)
        self.assertTrue(np.isfinite(tube.vertices).all())
        self.assertTrue(tube.is_volume)
        # Bending within the YZ plane must keep the section's depth along X everywhere.
        self.assertAlmostEqual(tube.extents[0], 0.5, places=6)

    def test_path_passing_through_vertical_does_not_flip_its_section(self):
        # A U-turn in the YZ plane. Re-deriving "up" at each ring would turn the section
        # 180 degrees at the vertical, joining opposite sides of neighbouring rings.
        theta = np.linspace(0, math.pi, 24)
        path = np.stack((np.zeros(24), 3 * np.cos(theta), -3 * np.sin(theta)), axis=1)
        tube = self.tube(path)
        self.assertLess(tube.edges_unique_length.max(), 0.8)

    def test_scales_shrink_rings(self):
        tube = self.tube([(0, 0, 0), (0, 0, -5)], scales=[1.0, 0.5])
        at_end = tube.vertices[np.isclose(tube.vertices[:, 2], -5.0)]
        self.assertAlmostEqual(np.ptp(at_end[:, 1]), 0.5)

    def test_path_with_too_few_points_is_rejected(self):
        with self.assertRaises(ValueError):
            self.tube([(0, 0, 0)])


class TempleCenterlineTests(unittest.TestCase):
    def test_starts_at_the_hinge_heading_backwards(self):
        path = temple_centerline(TempleParameters())
        np.testing.assert_allclose(path[0], (0, 0, 0))
        self.assertLess(path[1][2], 0)
        self.assertTrue(np.all(np.diff(path[:, 2]) < 0))

    def test_arc_length_equals_the_temple_length(self):
        for length in (10.0, 14.0, 16.5):
            for params in (straight_params(length=length),
                           TempleParameters(length=length),
                           TempleParameters(length=length, curvature=1.5, ear_drop=2.5)):
                with self.subTest(params=params):
                    self.assertAlmostEqual(polyline_length(temple_centerline(params)),
                                           length, delta=0.01)

    def test_straight_parameters_give_a_straight_line(self):
        path = temple_centerline(straight_params(length=12.0))
        self.assertEqual(len(path), 2)
        np.testing.assert_allclose(path[-1], (0, 0, -12.0), atol=1e-6)

    def test_ear_drop_lowers_only_the_end_of_the_arm(self):
        path = temple_centerline(TempleParameters(ear_drop=1.5, bend_fraction=0.8))
        self.assertAlmostEqual(path[-1][1], -1.5)
        ahead_of_bend = path[np.cumsum(np.r_[0, np.linalg.norm(np.diff(path, axis=0), axis=1)])
                             < 0.75 * 14.0]
        np.testing.assert_allclose(ahead_of_bend[:, 1], 0, atol=1e-9)

    def test_tip_x_is_splay_minus_curvature(self):
        path = temple_centerline(TempleParameters(splay=1.6, curvature=0.5))
        self.assertAlmostEqual(path[-1][0], 1.1)

    def test_more_curvature_moves_the_tip_inward(self):
        tips = [temple_centerline(TempleParameters(curvature=c))[-1][0] for c in (0, 0.5, 1, 2)]
        self.assertTrue(np.all(np.diff(tips) < 0))

    def test_splay_moves_the_arm_outward(self):
        self.assertGreater(temple_centerline(TempleParameters(splay=2.0))[-1][0],
                           temple_centerline(TempleParameters(splay=1.0))[-1][0])

    def test_path_is_smooth_at_the_ear_bend(self):
        path = temple_centerline(TempleParameters())
        directions = np.diff(path, axis=0)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        turn = np.arccos(np.clip((directions[:-1] * directions[1:]).sum(axis=1), -1, 1))
        self.assertLess(math.degrees(turn.max()), 15)

    def test_density_is_modest(self):
        self.assertLess(len(temple_centerline(TempleParameters())), 40)


class BuildTempleTests(unittest.TestCase):
    HINGE = np.array((6.8, 0.9, -0.8))

    def test_temple_is_a_closed_volume(self):
        for side in (-1, 1):
            for params in (TempleParameters(), straight_params(),
                           TempleParameters(width=0.9, thickness=0.5, curvature=1.5, ear_drop=3)):
                with self.subTest(side=side, params=params):
                    temple = build_temple(self.HINGE * (side, 1, 1), side, params)
                    self.assertTrue(temple.is_volume)
                    self.assertGreater(temple.volume, 0)

    def test_right_temple_stays_right_and_left_stays_left(self):
        right = build_temple(self.HINGE, 1, TempleParameters())
        left = build_temple(self.HINGE * (-1, 1, 1), -1, TempleParameters())
        self.assertGreater(right.bounds[0][0], 0)
        self.assertLess(left.bounds[1][0], 0)

    def test_left_is_the_exact_mirror_of_right(self):
        right = build_temple(self.HINGE, 1, TempleParameters())
        left = build_temple(self.HINGE * (-1, 1, 1), -1, TempleParameters())
        np.testing.assert_allclose(left.vertices, right.vertices * (-1, 1, 1))

    def test_arms_extend_backwards_from_the_hinge(self):
        temple = build_temple(self.HINGE, 1, TempleParameters())
        self.assertLess(temple.bounds[0][2], self.HINGE[2] - 10)
        self.assertLess(temple.bounds[1][2], self.HINGE[2] + 0.3)

    def test_width_is_the_height_at_the_hinge_and_thickness_is_sideways(self):
        temple = build_temple(np.zeros(3), 1, straight_params(width=0.8, thickness=0.2))
        self.assertAlmostEqual(temple.extents[1], 0.8, places=6)
        self.assertAlmostEqual(temple.extents[0], 0.2, places=6)

    def test_ear_end_is_rounded_off(self):
        temple = build_temple(np.zeros(3), 1, straight_params(width=0.8, length=10.0))
        end = temple.vertices[np.isclose(temple.vertices[:, 2], temple.bounds[0][2])]
        self.assertLess(np.ptp(end[:, 1]), 0.6)

    def test_rounding_does_not_shorten_the_arm(self):
        temple = build_temple(np.zeros(3), 1, straight_params(length=10.0))
        self.assertAlmostEqual(temple.bounds[0][2], -10.0, places=6)

    def test_no_length_makes_the_arm_fold_through_itself(self):
        # Where tip rounding starts relative to the path's points depends on length.
        for length in np.arange(8.0, 20.0, 0.25):
            for params in (TempleParameters(length=length),
                           TempleParameters(length=length, curvature=1.0, ear_drop=2.5)):
                with self.subTest(params=params):
                    self.assertFalse(self_intersects(build_temple(self.HINGE, 1, params)))

    def test_no_degenerate_triangles(self):
        for params in (TempleParameters(), straight_params()):
            temple = build_temple(self.HINGE, 1, params)
            self.assertGreater(temple.area_faces.min(), 1e-6)


class TempleValidationTests(unittest.TestCase):
    def test_bad_values_are_rejected(self):
        for bad in (dict(length=0), dict(length=-3), dict(width=0), dict(thickness=-1),
                    dict(curvature=-0.1), dict(ear_drop=-1), dict(length=float("nan")),
                    dict(splay=float("inf")), dict(bend_fraction=0.1), dict(bend_fraction=1.0),
                    dict(curvature=20.0), dict(ear_drop=20.0), dict(splay=20.0),
                    dict(width=8.0), dict(thickness=8.0)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    TempleParameters(**bad).validate()

    def test_defaults_are_valid(self):
        TempleParameters().validate()

    def test_build_validates(self):
        with self.assertRaises(ValueError):
            build_temple(np.zeros(3), 1, TempleParameters(length=-1))

    def test_side_must_be_plus_or_minus_one(self):
        with self.assertRaises(ValueError):
            build_temple(np.zeros(3), 0, TempleParameters())


if __name__ == "__main__":
    unittest.main()
