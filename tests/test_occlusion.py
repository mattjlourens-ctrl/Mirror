from collections import Counter
from pathlib import Path
import unittest

import mediapipe as mp
import numpy as np

from accessory_transform import attachment_matrix, project_camera_points, transform_points
from glasses_3d import DEFAULT_GLASSES_ATTACHMENT, build_glasses_model, draw_wireframe
from occlusion import (DEPTH_TOLERANCE_CM, FACE_OVAL, TASK_FILE, DepthBuffer,
                       load_canonical_face_mesh, load_head_occluder, rasterize_depth,
                       render_depth_buffer, sample_lines, visible_line_segments,
                       visible_mask)
from tests.test_head_pose import CAMERA, HEIGHT, WIDTH, head_rotation, make_pose

IDENTITY = np.eye(4)


def depth_buffer_for(triangles_cm, cull_back_faces=False):
    """Full-resolution depth buffer of camera-space triangles (T, 3, 3) in cm."""
    points = np.asarray(triangles_cm, dtype=np.float64).reshape(-1, 3)
    pixels = project_camera_points(points, CAMERA)
    triangles = np.arange(len(points)).reshape(-1, 3)
    depth = rasterize_depth(pixels, points[:, 2], triangles, WIDTH, HEIGHT, cull_back_faces)
    return DepthBuffer(depth, 1.0)


def wall(z, x_range=(-10, 10), y_range=(-10, 10)):
    """Two triangles forming a rectangle facing the camera at depth z.

    Wound so each normal, cross(b - a, c - a), points to -Z: towards the camera.
    """
    (x0, x1), (y0, y1) = x_range, y_range
    return [[(x0, y0, z), (x1, y1, z), (x1, y0, z)],
            [(x0, y0, z), (x0, y1, z), (x1, y1, z)]]


def pixel_of(point):
    return project_camera_points(np.array([point], dtype=np.float64), CAMERA)[0]


class DepthBufferTests(unittest.TestCase):
    def test_wall_depth_inside_and_inf_outside(self):
        buffer = depth_buffer_for(wall(50))
        x, y = np.round(pixel_of((0, 0, 50))).astype(int)
        self.assertAlmostEqual(buffer.depth[y, x], 50, places=4)
        self.assertEqual(buffer.depth[5, 5], np.inf)

    def test_depth_is_perspective_correct_on_a_tilted_triangle(self):
        corners = np.array([(-10, -8, 40), (12, -6, 70), (0, 10, 55)], dtype=np.float64)
        buffer = depth_buffer_for([corners])
        normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        for u, v in [(300, 240), (330, 225), (315, 260)]:
            # Exact depth: where the pixel's viewing ray meets the triangle's plane.
            ray = np.array([(u - 320) / 640, (v - 240) / 640, 1.0])
            exact_z = normal @ corners[0] / (normal @ ray)
            self.assertAlmostEqual(float(buffer.depth[v, u]), exact_z, places=3)

    def test_nearest_surface_wins_in_any_order(self):
        for triangles in (wall(50) + wall(40), wall(40) + wall(50)):
            buffer = depth_buffer_for(triangles)
            self.assertAlmostEqual(buffer.depth[240, 320], 40, places=4)

    def test_back_face_culling(self):
        front, back = wall(50)[0], wall(50)[0][::-1]  # reversed winding
        x, y = np.round(pixel_of((5, -5, 50))).astype(int)
        self.assertEqual(depth_buffer_for([front], True).depth[y, x], 50)
        self.assertEqual(depth_buffer_for([back], True).depth[y, x], np.inf)
        self.assertEqual(depth_buffer_for([back], False).depth[y, x], 50)


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.buffer = depth_buffer_for(wall(50, x_range=(0, 10)))  # right half only

    def visible(self, *points):
        points = np.array(points, dtype=np.float64)
        return visible_mask(self.buffer, project_camera_points(points, CAMERA), points[:, 2])

    def test_in_front_is_visible(self):
        self.assertTrue(self.visible((5, 0, 45)).all())

    def test_behind_is_hidden(self):
        self.assertFalse(self.visible((5, 0, 55)).any())

    def test_tolerance_keeps_points_just_behind_the_surface(self):
        just_behind = 50 + DEPTH_TOLERANCE_CM / 2
        too_far_behind = 50 + DEPTH_TOLERANCE_CM * 2
        np.testing.assert_array_equal(self.visible((5, 0, just_behind), (5, 0, too_far_behind)),
                                      [True, False])

    def test_no_occluder_at_that_pixel_is_visible(self):
        self.assertTrue(self.visible((-5, 0, 100)).all())

    def test_invalid_points_are_never_visible(self):
        pixels = np.array([[np.nan, 240.0], [320.0, 240.0], [320.0, 240.0]])
        depths = np.array([45.0, np.nan, 0.5])  # 0.5 cm is in front of the near plane
        self.assertFalse(visible_mask(self.buffer, pixels, depths).any())
        self.assertFalse(visible_mask(None, pixels, depths).any())

    def test_off_screen_pixels_are_not_occluded(self):
        self.assertTrue(visible_mask(self.buffer, np.array([[-500.0, -500.0]]),
                                     np.array([45.0])).all())


class LineClippingTests(unittest.TestCase):
    def test_partially_hidden_line_is_cut_at_the_wall_edge(self):
        # The line is at z = 55; the wall at z = 50 covers only x >= 0.
        buffer = depth_buffer_for(wall(50, x_range=(0, 10)))
        samples = sample_lines(np.array([[-10.0, 0, 55], [10.0, 0, 55]]),
                               np.array([[0, 1]]), spacing=0.25)
        segments = visible_line_segments(samples, IDENTITY, CAMERA, buffer)
        self.assertGreater(len(segments), 0)
        self.assertLess(len(segments), len(samples.pairs))
        wall_edge_u = pixel_of((0, 0, 55))[0]
        self.assertAlmostEqual(segments[..., 0].max(), wall_edge_u, delta=4)
        self.assertLess(segments[..., 0].min(), wall_edge_u - 50)

    def test_line_in_front_of_the_wall_is_fully_visible(self):
        buffer = depth_buffer_for(wall(50))
        samples = sample_lines(np.array([[-10.0, 0, 45], [10.0, 0, 45]]),
                               np.array([[0, 1]]), spacing=0.25)
        self.assertEqual(len(visible_line_segments(samples, IDENTITY, CAMERA, buffer)),
                         len(samples.pairs))

    def test_line_through_the_near_plane_is_clipped_not_dropped(self):
        samples = sample_lines(np.array([[1.0, 0, -10], [1.0, 0, 10]]),
                               np.array([[0, 1]]), spacing=0.25)
        segments = visible_line_segments(samples, IDENTITY, CAMERA, None)
        self.assertGreater(len(segments), 0)
        self.assertTrue(np.isfinite(segments).all())

    def test_samples_cover_every_edge_end_to_end(self):
        vertices = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 2.0, 0]])
        samples = sample_lines(vertices, np.array([[0, 1], [1, 2]]), spacing=0.25)
        self.assertEqual(len(samples.pairs), 4 + 8)
        steps = np.linalg.norm(samples.points[samples.pairs[:, 1]]
                               - samples.points[samples.pairs[:, 0]], axis=1)
        self.assertTrue((steps <= 0.25 + 1e-12).all())


class HeadOccluderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.occluder = load_head_occluder()
        if cls.occluder is None:
            raise unittest.SkipTest("face_landmarker.task not available")

    def test_uses_the_canonical_face_triangles(self):
        vertices, triangles = load_canonical_face_mesh()
        self.assertEqual(len(triangles), 898)
        np.testing.assert_array_equal(self.occluder.vertices[:468], vertices)
        np.testing.assert_array_equal(self.occluder.triangles[:898], triangles)

    def test_face_oval_matches_mediapipe(self):
        loop = {frozenset((FACE_OVAL[i], FACE_OVAL[(i + 1) % len(FACE_OVAL)]))
                for i in range(len(FACE_OVAL))}
        expected = {frozenset(edge)
                    for edge in mp.solutions.face_mesh_connections.FACEMESH_FACE_OVAL}
        self.assertEqual(loop, expected)

    def test_occluder_is_a_closed_surface(self):
        # Back-face culling is only safe on a closed mesh: every edge must be
        # shared by exactly two triangles.
        edges = Counter(frozenset(pair) for tri in self.occluder.triangles
                        for pair in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])))
        self.assertEqual(set(edges.values()), {2})

    def test_missing_model_file(self):
        self.assertIsNone(load_head_occluder(Path("does_not_exist.task")))

    def test_missing_pose_or_occluder(self):
        self.assertIsNone(render_depth_buffer(self.occluder, None, WIDTH, HEIGHT))
        pose = make_pose(head_rotation(), (0, 0, 50))
        self.assertIsNone(render_depth_buffer(None, pose, WIDTH, HEIGHT))

    def test_head_too_close_does_not_crash(self):
        pose = make_pose(head_rotation(yaw=30), (0, 0, 3))
        render_depth_buffer(self.occluder, pose, WIDTH, HEIGHT)


@unittest.skipUnless(TASK_FILE.exists(), "face_landmarker.task not available")
class GlassesOcclusionTests(unittest.TestCase):
    """The default glasses on the canonical head, checked part by part."""

    @classmethod
    def setUpClass(cls):
        cls.occluder = load_head_occluder()
        cls.model = build_glasses_model()
        cls.attach = attachment_matrix(DEFAULT_GLASSES_ATTACHMENT)
        n = 24  # lens_segments
        edges = cls.model.edges
        cls.parts = {
            "lenses": edges[:2 * n],
            "bridge": edges[2 * n:2 * n + 2],
            "left temple": edges[-4:-2],   # screen-left arm
            "right temple": edges[-2:],    # screen-right arm
        }

    def visible_fraction(self, part, **angles):
        pose = make_pose(head_rotation(**angles), (0, 0, 50))
        buffer = render_depth_buffer(self.occluder, pose, WIDTH, HEIGHT)
        samples = sample_lines(self.model.vertices, self.parts[part], 0.25)
        camera_points = transform_points(pose.transform @ self.attach, samples.points)
        pixels = project_camera_points(camera_points, CAMERA)
        return visible_mask(buffer, pixels, camera_points[:, 2]).mean()

    def test_frontal_lenses_and_bridge_stay_visible(self):
        self.assertGreater(self.visible_fraction("lenses"), 0.97)
        self.assertEqual(self.visible_fraction("bridge"), 1.0)

    def test_turning_hides_the_far_temple_but_not_the_near_one(self):
        # Positive yaw turns the screen-right side of the head away.
        self.assertLess(self.visible_fraction("right temple", yaw=60), 0.25)
        self.assertGreater(self.visible_fraction("left temple", yaw=60), 0.9)
        self.assertLess(self.visible_fraction("left temple", yaw=-60), 0.25)
        self.assertGreater(self.visible_fraction("right temple", yaw=-60), 0.9)

    def test_far_temple_disappears_gradually_not_all_at_once(self):
        fractions = [self.visible_fraction("right temple", yaw=yaw) for yaw in (0, 20, 40, 60)]
        partly = [f for f in fractions if 0.05 < f < 0.95]
        self.assertTrue(partly, f"no partially hidden pose found: {fractions}")

    def test_draw_with_and_without_occlusion(self):
        pose = make_pose(head_rotation(yaw=50), (0, 0, 50))
        buffer = render_depth_buffer(self.occluder, pose, WIDTH, HEIGHT)
        occluded = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        unoccluded = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        self.assertTrue(draw_wireframe(occluded, self.model, self.attach, pose, buffer))
        self.assertTrue(draw_wireframe(unoccluded, self.model, self.attach, pose))
        self.assertLess(occluded.any(axis=2).sum(), unoccluded.any(axis=2).sum())


if __name__ == "__main__":
    unittest.main()
