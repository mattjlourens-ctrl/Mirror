from collections import deque
import unittest

import cv2
import numpy as np

from accessory_transform import (MIN_DEPTH_CM, Attachment, attachment_matrix,
                                 project_accessory, project_camera_points,
                                 transform_points, translation_matrix)
from glasses_3d import (DEFAULT_GLASSES_ATTACHMENT, GlassesDimensions, build_glasses_model,
                        draw_wireframe, projected_segments)
from head_pose import FACE_ORIGIN, FRONTAL_ROTATION, project_face_axes
from tests.test_head_pose import CAMERA, axis_rotation, head_rotation, make_pose

MODEL = build_glasses_model()
DIMS = GlassesDimensions()
N = DIMS.lens_segments
LEFT_LENS = slice(0, N)        # screen-left lens vertices
RIGHT_LENS = slice(N, 2 * N)   # screen-right lens vertices
ATTACH = attachment_matrix(DEFAULT_GLASSES_ATTACHMENT)


def project(pose, attach=ATTACH):
    return project_accessory(MODEL.vertices, attach, pose)


def camera_space(pose, attach=ATTACH):
    return transform_points(pose.transform @ attach, MODEL.vertices)


def projected_width(pixels):
    return pixels[:, 0].max() - pixels[:, 0].min()


class ModelDefinitionTests(unittest.TestCase):
    def test_vertices_are_finite_3d_points(self):
        self.assertEqual(MODEL.vertices.ndim, 2)
        self.assertEqual(MODEL.vertices.shape[1], 3)
        self.assertTrue(np.isfinite(MODEL.vertices).all())

    def test_edges_reference_existing_distinct_vertices(self):
        self.assertEqual(MODEL.edges.shape[1], 2)
        self.assertTrue((MODEL.edges >= 0).all())
        self.assertTrue((MODEL.edges < len(MODEL.vertices)).all())
        self.assertTrue((MODEL.edges[:, 0] != MODEL.edges[:, 1]).all())

    def test_wireframe_is_one_connected_piece(self):
        neighbours = {i: set() for i in range(len(MODEL.vertices))}
        for a, b in MODEL.edges:
            neighbours[a].add(b)
            neighbours[b].add(a)
        seen, queue = {0}, deque([0])
        while queue:
            for nxt in neighbours[queue.popleft()] - seen:
                seen.add(nxt)
                queue.append(nxt)
        self.assertEqual(len(seen), len(MODEL.vertices))

    def test_model_is_genuinely_3d(self):
        depth = MODEL.vertices[:, 2].max() - MODEL.vertices[:, 2].min()
        self.assertGreater(depth, 10)  # temples reach well behind the lenses


class DimensionTests(unittest.TestCase):
    def test_default_dimensions(self):
        v = MODEL.vertices
        lenses = v[:2 * N]
        self.assertAlmostEqual(lenses[:, 0].max() - lenses[:, 0].min(), DIMS.frame_width)
        lens_y = v[LEFT_LENS, 1]
        self.assertAlmostEqual(lens_y.max() - lens_y.min(), DIMS.lens_height)
        self.assertAlmostEqual(v[RIGHT_LENS, 0].min() - v[LEFT_LENS, 0].max(),
                               DIMS.bridge_width)

    def test_temples_extend_backwards(self):
        # Temples go towards -Z (behind the face), never in front of it.
        self.assertAlmostEqual(MODEL.vertices[:, 2].max(), 0.0, places=6)
        reach = -MODEL.vertices[:, 2].min()
        self.assertGreater(reach, DIMS.temple_length)
        self.assertLessEqual(reach, DIMS.temple_length + DIMS.wrap_depth)

    def test_custom_dimensions_change_the_model(self):
        dims = GlassesDimensions(frame_width=12.0, lens_height=3.0, temple_length=10.0)
        v = build_glasses_model(dims).vertices
        lenses = v[:2 * dims.lens_segments]
        self.assertAlmostEqual(lenses[:, 0].max() - lenses[:, 0].min(), 12.0)
        self.assertAlmostEqual(np.ptp(v[:dims.lens_segments, 1]), 3.0)
        self.assertLess(v[:, 2].min(), -10.0)

    def test_temples_splay_outwards(self):
        # The last four vertices are each temple's ear bend and tip.
        hinge_x = MODEL.vertices[MODEL.edges[MODEL.edges[:, 1] == len(MODEL.vertices) - 4][0, 0], 0]
        left_bend_x, right_bend_x = MODEL.vertices[[-4, -2], 0]
        self.assertAlmostEqual(left_bend_x, hinge_x - DIMS.temple_spread)
        self.assertAlmostEqual(right_bend_x, -left_bend_x)

    def test_invalid_dimensions_are_rejected(self):
        for bad in (GlassesDimensions(frame_width=0), GlassesDimensions(bridge_width=20),
                    GlassesDimensions(temple_length=-1), GlassesDimensions(lens_segments=3)):
            with self.assertRaises(ValueError):
                build_glasses_model(bad)


class TransformTests(unittest.TestCase):
    def test_identity_transform(self):
        matrix = attachment_matrix(Attachment(anchor_cm=(0, 0, 0)))
        np.testing.assert_allclose(matrix, np.eye(4))
        np.testing.assert_allclose(transform_points(matrix, MODEL.vertices), MODEL.vertices)

    def test_translation_without_rotation(self):
        moved = transform_points(translation_matrix((1.0, -2.0, 3.0)), MODEL.vertices)
        np.testing.assert_allclose(moved - MODEL.vertices,
                                   np.tile((1.0, -2.0, 3.0), (len(MODEL.vertices), 1)))

    def test_matches_homogeneous_multiplication(self):
        matrix = make_pose(head_rotation(yaw=20, pitch=5), (1, 2, 40)).transform @ ATTACH
        homogeneous = np.hstack([MODEL.vertices, np.ones((len(MODEL.vertices), 1))])
        np.testing.assert_allclose(transform_points(matrix, MODEL.vertices),
                                   (homogeneous @ matrix.T)[:, :3])

    def test_attachment_offset(self):
        attachment = Attachment(offset_cm=(0.0, -0.6, 0.5))
        origin = transform_points(attachment_matrix(attachment), np.zeros((1, 3)))[0]
        np.testing.assert_allclose(origin, FACE_ORIGIN + (0.0, -0.6, 0.5))

    def test_model_units_are_converted_to_cm(self):
        # A model authored in metres: 0.07 m along X must become 7 cm.
        matrix = attachment_matrix(Attachment(anchor_cm=(0, 0, 0)), units_to_cm=100.0)
        np.testing.assert_allclose(transform_points(matrix, np.array([[0.07, 0, 0]])),
                                   [[7.0, 0, 0]])

    def test_model_orientation_is_applied(self):
        # A Z-up model (as Blender uses) needs -90 degrees about X so its up is +Y.
        z_up_to_y_up = axis_rotation("x", -90)
        matrix = attachment_matrix(Attachment(anchor_cm=(0, 0, 0)), orientation=z_up_to_y_up)
        np.testing.assert_allclose(transform_points(matrix, np.array([[0, 0, 1.0]])),
                                   [[0, 1.0, 0]], atol=1e-12)


class HeadRotationTests(unittest.TestCase):
    def test_frontal_lenses_are_symmetric(self):
        pixels = project(make_pose(FRONTAL_ROTATION, (0, 0, 50)))
        self.assertAlmostEqual(projected_width(pixels[LEFT_LENS]),
                               projected_width(pixels[RIGHT_LENS]), places=6)

    def test_yaw_makes_the_far_lens_smaller(self):
        # Positive yaw turns the nose screen-right, so the screen-right lens
        # swings away from the camera.
        pose = make_pose(head_rotation(yaw=30), (0, 0, 50))
        depth = camera_space(pose)[:, 2]
        self.assertGreater(depth[RIGHT_LENS].mean(), depth[LEFT_LENS].mean() + 2)
        pixels = project(pose)
        self.assertLess(projected_width(pixels[RIGHT_LENS]),
                        projected_width(pixels[LEFT_LENS]))

    def test_yaw_swings_temples_sideways(self):
        hinge_to_tip = self.temple_vectors(make_pose(head_rotation(yaw=30), (0, 0, 50)))
        self.assertTrue((hinge_to_tip[:, 0] < -20).all())  # backwards = screen-left

    def test_pitch_up_points_temples_down(self):
        hinge_to_tip = self.temple_vectors(make_pose(head_rotation(pitch=20), (0, 0, 50)))
        self.assertTrue((hinge_to_tip[:, 1] > 20).all())  # image y grows downwards

    def test_roll_tilts_the_frame_clockwise(self):
        pixels = project(make_pose(head_rotation(roll=20), (0, 0, 50)))
        left_y = pixels[LEFT_LENS, 1].mean()
        right_y = pixels[RIGHT_LENS, 1].mean()
        self.assertGreater(right_y, left_y + 10)

    def temple_vectors(self, pose):
        """Pixel vectors along each temple, from the hinge to the ear bend."""
        pixels = project(pose)
        bends = [len(MODEL.vertices) - 4, len(MODEL.vertices) - 2]
        hinges = [MODEL.edges[MODEL.edges[:, 1] == bend][0, 0] for bend in bends]
        return pixels[bends] - pixels[hinges]


class ProjectionTests(unittest.TestCase):
    def test_known_projection(self):
        pixels = project_camera_points(np.array([[10.0, 5.0, 100.0]]), CAMERA)
        np.testing.assert_allclose(pixels, [[384, 272]])

    def test_twice_as_far_looks_half_as_big(self):
        near = project_camera_points(np.array([[-5.0, 0, 50], [5.0, 0, 50]]), CAMERA)
        far = project_camera_points(np.array([[-5.0, 0, 100], [5.0, 0, 100]]), CAMERA)
        self.assertAlmostEqual(projected_width(near), 2 * projected_width(far))

    def test_moving_closer_makes_the_glasses_bigger(self):
        near = project(make_pose(FRONTAL_ROTATION, (0, 0, 40)))
        far = project(make_pose(FRONTAL_ROTATION, (0, 0, 80)))
        self.assertGreater(projected_width(near[:2 * N]), 1.8 * projected_width(far[:2 * N]))

    def test_points_behind_or_at_the_camera_are_rejected(self):
        points = np.array([[0, 0, 50.0], [0, 0, -5.0], [0, 0, 0.0], [0, 0, MIN_DEPTH_CM / 2]])
        pixels = project_camera_points(points, CAMERA)
        self.assertTrue(np.isfinite(pixels[0]).all())
        self.assertTrue(np.isnan(pixels[1:]).all())

    def test_edges_touching_rejected_points_are_not_drawn(self):
        # The head is so close that the temples end up behind the camera.
        pose = make_pose(FRONTAL_ROTATION, (0, 0, 5))
        segments = projected_segments(MODEL, ATTACH, pose)
        self.assertLess(len(segments), len(MODEL.edges))
        frame = np.zeros((480, 640, 3), np.uint8)
        draw_wireframe(frame, MODEL, ATTACH, pose)  # must not raise


class MissingPoseTests(unittest.TestCase):
    def test_project_without_pose(self):
        self.assertIsNone(project_accessory(MODEL.vertices, ATTACH, None))

    def test_draw_without_pose_draws_nothing(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        self.assertFalse(draw_wireframe(frame, MODEL, ATTACH, None))
        self.assertFalse(frame.any())

    def test_draw_with_pose_draws(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        self.assertTrue(draw_wireframe(frame, MODEL, ATTACH,
                                       make_pose(FRONTAL_ROTATION, (0, 0, 50))))
        self.assertTrue(frame.any())


class IntegrationTests(unittest.TestCase):
    def test_glasses_origin_sits_on_the_pose_axes_origin(self):
        # With no offset, the glasses' origin is the anchor that Step 11's axes use.
        pose = make_pose(head_rotation(yaw=20, pitch=-10, roll=5), (3, -2, 55))
        no_offset = attachment_matrix(Attachment())
        origin_pixel = project_accessory(np.zeros((1, 3)), no_offset, pose)[0]
        np.testing.assert_allclose(origin_pixel, project_face_axes(pose)[0], atol=1e-6)

    def test_matches_an_independent_opencv_projection(self):
        # Our chain (attachment, then HeadPose.transform, then projection) must equal
        # OpenCV projecting face-local points with rvec/tvec directly. This would
        # catch the rotation being applied twice or in the wrong order.
        pose = make_pose(head_rotation(yaw=-25, pitch=15, roll=-10), (-4, 1, 60))
        face_points = transform_points(ATTACH, MODEL.vertices)
        expected, _ = cv2.projectPoints(face_points, pose.rotation_vector,
                                        pose.translation_vector, CAMERA, np.zeros(4))
        np.testing.assert_allclose(project(pose), expected.reshape(-1, 2), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
