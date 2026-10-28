import math
from pathlib import Path
import unittest
import zipfile

import cv2
import numpy as np

from head_pose import (FRONTAL_ROTATION, MODEL_POINTS, NO_DISTORTION, POSE_LANDMARKS,
                       HeadPose, camera_matrix, estimate_head_pose, euler_angles,
                       image_points_from_landmarks, project_face_axes)

WIDTH, HEIGHT = 640, 480
CAMERA = camera_matrix(WIDTH, HEIGHT)
TASK_FILE = Path(__file__).resolve().parent.parent / "face_landmarker.task"


def axis_rotation(axis, degrees):
    """Rotation matrix about a single axis, built independently via Rodrigues."""
    rotation_vector = np.zeros(3)
    rotation_vector["xyz".index(axis)] = math.radians(degrees)
    matrix, _ = cv2.Rodrigues(rotation_vector)
    return matrix


def head_rotation(yaw=0.0, pitch=0.0, roll=0.0):
    """Model -> camera rotation for a head turned by yaw/pitch/roll (degrees)."""
    in_viewer_axes = (axis_rotation("y", yaw) @ axis_rotation("x", -pitch)
                      @ axis_rotation("z", -roll))
    return FRONTAL_ROTATION @ in_viewer_axes


def make_pose(rotation_matrix, translation=(0.0, 0.0, 50.0)):
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    translation_vector = np.array(translation, dtype=np.float64).reshape(3, 1)
    return HeadPose(rotation_vector, translation_vector, rotation_matrix, CAMERA)


def synthetic_image_points(pose):
    projected, _ = cv2.projectPoints(MODEL_POINTS, pose.rotation_vector,
                                     pose.translation_vector, CAMERA, NO_DISTORTION)
    return projected.reshape(-1, 2)


def rotation_difference_degrees(a, b):
    """Angle of the rotation that takes a to b. Comparing matrices avoids the
    ambiguity of rotation vectors (e.g. +180 and -180 degrees are the same)."""
    cos_angle = (np.trace(a.T @ b) - 1) / 2
    return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))


class RodriguesTests(unittest.TestCase):
    def test_zero_vector_is_identity(self):
        matrix, _ = cv2.Rodrigues(np.zeros(3))
        np.testing.assert_allclose(matrix, np.eye(3), atol=1e-12)

    def test_quarter_turn_about_z(self):
        matrix, _ = cv2.Rodrigues(np.array([0.0, 0.0, math.pi / 2]))
        expected = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        np.testing.assert_allclose(matrix, expected, atol=1e-12)

    def test_round_trip(self):
        rotation = head_rotation(yaw=25, pitch=-10, roll=5)
        vector, _ = cv2.Rodrigues(rotation)
        back, _ = cv2.Rodrigues(vector)
        np.testing.assert_allclose(back, rotation, atol=1e-12)


class ProjectionTests(unittest.TestCase):
    def test_point_on_optical_axis_hits_principal_point(self):
        projected, _ = cv2.projectPoints(np.array([[0.0, 0.0, 100.0]]), np.zeros(3),
                                         np.zeros(3), CAMERA, NO_DISTORTION)
        np.testing.assert_allclose(projected.ravel(), [320, 240])

    def test_known_point(self):
        # x_px = f * X / Z + cx = 640 * 10 / 100 + 320 = 384
        projected, _ = cv2.projectPoints(np.array([[10.0, 5.0, 100.0]]), np.zeros(3),
                                         np.zeros(3), CAMERA, NO_DISTORTION)
        np.testing.assert_allclose(projected.ravel(), [384, 272])

    def test_transform_matches_rotation_and_translation(self):
        pose = make_pose(head_rotation(yaw=30), (1.0, 2.0, 40.0))
        point = np.array([1.0, 2.0, 3.0])
        expected = pose.rotation_matrix @ point + pose.translation_vector.ravel()
        np.testing.assert_allclose((pose.transform @ np.append(point, 1))[:3], expected)


class EulerAngleTests(unittest.TestCase):
    def assertAngles(self, rotation, expected):
        np.testing.assert_allclose(euler_angles(rotation), expected, atol=1e-9)

    def test_frontal_face_is_all_zero(self):
        self.assertAngles(FRONTAL_ROTATION, (0, 0, 0))

    def test_yaw_only(self):
        self.assertAngles(head_rotation(yaw=20), (20, 0, 0))

    def test_pitch_only(self):
        self.assertAngles(head_rotation(pitch=-15), (0, -15, 0))

    def test_roll_only(self):
        self.assertAngles(head_rotation(roll=10), (0, 0, 10))

    def test_combined_rotation(self):
        self.assertAngles(head_rotation(yaw=-30, pitch=12, roll=-8), (-30, 12, -8))


class AxisDirectionTests(unittest.TestCase):
    """Check the on-screen meaning of each sign, independent of the formulas.
    Remember image y grows DOWN."""

    def axes(self, **angles):
        return project_face_axes(make_pose(head_rotation(**angles)))

    def test_frontal_axes(self):
        origin, x_end, y_end, _ = self.axes()
        self.assertGreater(x_end[0], origin[0] + 10)  # X points screen-right
        self.assertLess(y_end[1], origin[1] - 10)     # Y points screen-up

    def test_positive_yaw_turns_nose_screen_right(self):
        origin, _, _, z_end = self.axes(yaw=30)
        self.assertGreater(z_end[0], origin[0] + 5)

    def test_positive_pitch_turns_nose_up(self):
        origin, _, _, z_end = self.axes(pitch=30)
        self.assertLess(z_end[1], origin[1] - 5)

    def test_positive_roll_tilts_clockwise(self):
        origin, x_end, _, _ = self.axes(roll=20)
        self.assertGreater(x_end[1], origin[1] + 5)  # X end drops below origin


class EstimateHeadPoseTests(unittest.TestCase):
    def assertRecovers(self, true_pose, image_points, max_degrees, max_cm):
        estimate = estimate_head_pose(image_points, CAMERA)
        self.assertIsNotNone(estimate)
        self.assertLess(rotation_difference_degrees(estimate.rotation_matrix,
                                                    true_pose.rotation_matrix), max_degrees)
        np.testing.assert_allclose(estimate.translation_vector,
                                   true_pose.translation_vector, atol=max_cm)

    def test_recovers_frontal_pose(self):
        pose = make_pose(FRONTAL_ROTATION, (0.0, 0.0, 50.0))
        self.assertRecovers(pose, synthetic_image_points(pose), 0.01, 0.01)

    def test_recovers_rotated_and_shifted_pose(self):
        pose = make_pose(head_rotation(yaw=25, pitch=-10, roll=8), (4.0, -3.0, 60.0))
        self.assertRecovers(pose, synthetic_image_points(pose), 0.01, 0.01)
        estimate = estimate_head_pose(synthetic_image_points(pose), CAMERA)
        np.testing.assert_allclose(euler_angles(estimate.rotation_matrix),
                                   (25, -10, 8), atol=0.01)

    def test_tolerates_pixel_noise(self):
        pose = make_pose(head_rotation(yaw=-20, pitch=10), (0.0, 0.0, 55.0))
        noise = np.random.default_rng(0).uniform(-1, 1, (len(MODEL_POINTS), 2))
        self.assertRecovers(pose, synthetic_image_points(pose) + noise, 3.0, 3.0)

    def test_none_input(self):
        self.assertIsNone(estimate_head_pose(None, CAMERA))

    def test_too_few_points(self):
        pose = make_pose(FRONTAL_ROTATION)
        self.assertIsNone(estimate_head_pose(synthetic_image_points(pose)[:5], CAMERA))

    def test_non_finite_points(self):
        points = synthetic_image_points(make_pose(FRONTAL_ROTATION))
        points[3] = (np.nan, 100.0)
        self.assertIsNone(estimate_head_pose(points, CAMERA))

    def test_degenerate_points_fail_safely(self):
        points = np.full((len(MODEL_POINTS), 2), 200.0)
        self.assertIsNone(estimate_head_pose(points, CAMERA))

    def test_random_points_are_rejected(self):
        points = np.random.default_rng(1).uniform(0, 480, (len(MODEL_POINTS), 2))
        self.assertIsNone(estimate_head_pose(points, CAMERA))

    def test_recovers_after_a_failure(self):
        self.assertIsNone(estimate_head_pose(np.full((len(MODEL_POINTS), 2), 0.0), CAMERA))
        pose = make_pose(head_rotation(yaw=10), (0.0, 0.0, 50.0))
        self.assertRecovers(pose, synthetic_image_points(pose), 0.01, 0.01)


class LandmarkInputTests(unittest.TestCase):
    class FakeLandmark:
        def __init__(self, x, y):
            self.x, self.y = x, y

    def test_too_few_landmarks(self):
        landmarks = [self.FakeLandmark(0.5, 0.5)] * 100
        self.assertIsNone(image_points_from_landmarks(landmarks, WIDTH, HEIGHT))

    def test_converts_to_float_pixels(self):
        landmarks = [self.FakeLandmark(0.5, 0.25)] * 478
        points = image_points_from_landmarks(landmarks, WIDTH, HEIGHT)
        self.assertEqual(points.shape, (len(POSE_LANDMARKS), 2))
        np.testing.assert_allclose(points[0], (320.0, 120.0))


@unittest.skipUnless(TASK_FILE.exists(), "face_landmarker.task not found")
class CanonicalModelTests(unittest.TestCase):
    def test_model_points_match_mediapipe_canonical_mesh(self):
        from mediapipe.tasks.cc.vision.face_geometry.proto import (
            geometry_pipeline_metadata_pb2)
        metadata = geometry_pipeline_metadata_pb2.GeometryPipelineMetadata()
        with zipfile.ZipFile(TASK_FILE) as bundle:
            metadata.ParseFromString(
                bundle.read("geometry_pipeline_metadata_landmarks.binarypb"))
        # Each vertex is x, y, z, u, v.
        vertices = np.array(metadata.canonical_mesh.vertex_buffer).reshape(-1, 5)[:, :3]
        for landmark_id, model_point in POSE_LANDMARKS.items():
            np.testing.assert_allclose(vertices[landmark_id], model_point, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
