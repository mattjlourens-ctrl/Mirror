from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import trimesh

from accessory_transform import attachment_matrix, project_camera_points, transform_points
from fit_controls import FitAdjustment, fit_matrix
from glasses_3d import DEFAULT_GLASSES_ATTACHMENT
from glb_model import MeshModel, MeshPart, load_glasses_model
from head_pose import (FACE_ORIGIN, FRONTAL_ROTATION, NO_DISTORTION, POSE_LANDMARKS, HeadPose,
                       camera_matrix)
from image_to_3d.assembly import (FRONT_FRAME, LEFT_TEMPLE, RIGHT_TEMPLE, GlassesParameters,
                                  export_model_glb)
import mesh_renderer
from mesh_renderer import HALF_VECTOR, clip_triangles_to_near_plane, render_mesh
from occlusion import (DepthBuffer, load_head_occluder, render_depth_buffer, triangle_boxes,
                       triangle_fragments)
from tests.test_glb_model import generated_model
from tests.test_head_pose import CAMERA, HEIGHT, WIDTH, head_rotation, make_pose

REPO_ROOT = Path(__file__).resolve().parent.parent
PHOTO_MODEL = REPO_ROOT / "image_to_3d" / "output" / "example_3_photo.glb"
GREY = 128
RED = (0, 0, 220)
GREEN = (0, 220, 0)
# Identity head pose with the camera at the origin: model = camera space.
FACE_FREE_POSE = make_pose(np.eye(3), (0.0, 0.0, 0.0))
IDENTITY = np.eye(4)
FACET_STEP_LIMIT = 4


def blank():
    return np.full((HEIGHT, WIDTH, 3), GREY, np.uint8)


def changed(frame):
    return (frame != GREY).any(axis=2)


def quad(x_range, y_range, z, color=RED, gloss=0.0, name="quad"):
    """Rectangle in camera space facing the camera (normals towards -Z)."""
    (x0, x1), (y0, y1) = x_range, y_range
    vertices = np.array([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], float)
    # Counter-clockwise when seen from the camera at the origin looking along +Z,
    # with camera Y pointing down: cross(b - a, c - a) points to -Z.
    faces = np.array([(0, 2, 1), (0, 3, 2)])
    return MeshPart(name, vertices, faces, color, gloss)


def model_of(*parts):
    return MeshModel("test", tuple(parts))


def render(model, pose=FACE_FREE_POSE, attach=IDENTITY, depth_buffer=None):
    frame = blank()
    drawn = render_mesh(frame, model, attach, pose, depth_buffer)
    return frame, drawn


def pixel(point):
    return np.round(project_camera_points(np.array([point], float), CAMERA)[0]).astype(int)


class ProjectionTests(unittest.TestCase):
    def test_quad_lands_where_the_pinhole_model_says(self):
        # f = 640 at 640 wide: x px = 320 + 640 * X / Z.
        frame, drawn = render(model_of(quad((-5, 5), (-5, 5), 50)))
        self.assertTrue(drawn)
        mask = changed(frame)
        solid = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        ys, xs = np.nonzero(solid)
        self.assertAlmostEqual(xs.min(), 320 - 64, delta=3)
        self.assertAlmostEqual(xs.max(), 320 + 64, delta=3)
        self.assertAlmostEqual(ys.min(), 240 - 64, delta=3)
        self.assertAlmostEqual(ys.max(), 240 + 64, delta=3)

    def test_farther_objects_look_smaller(self):
        near, _ = render(model_of(quad((-5, 5), (-5, 5), 40)))
        far, _ = render(model_of(quad((-5, 5), (-5, 5), 80)))
        self.assertAlmostEqual(changed(near).sum() / changed(far).sum(), 4.0, delta=0.2)

    def test_uses_the_pose_camera_matrix(self):
        narrow = CAMERA.copy()
        narrow[0, 0] = narrow[1, 1] = 320.0
        pose = make_pose(np.eye(3), (0.0, 0.0, 0.0))
        object.__setattr__(pose, "camera_matrix", narrow)
        wide_view, _ = render(model_of(quad((-5, 5), (-5, 5), 50)))
        narrow_view, _ = render(model_of(quad((-5, 5), (-5, 5), 50)), pose=pose)
        self.assertLess(changed(narrow_view).sum(), changed(wide_view).sum() / 2)


    def test_projection_matches_opencv_exactly(self):
        points = np.random.default_rng(1).uniform((-30, -30, 1), (30, 30, 200), (200, 3))
        expected, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), CAMERA, NO_DISTORTION)
        np.testing.assert_allclose(project_camera_points(points, CAMERA), expected.reshape(-1, 2),
                                   atol=1e-9)

    def test_points_behind_the_near_plane_project_to_nan(self):
        pixels = project_camera_points(np.array([(0.0, 0.0, 0.5), (0.0, 0.0, 5.0)]), CAMERA)
        self.assertTrue(np.isnan(pixels[0]).all())
        self.assertTrue(np.isfinite(pixels[1]).all())


class FragmentTests(unittest.TestCase):
    """The rasterizer must find exactly the pixels a test of every pixel would."""
    WIDTH, HEIGHT = 60, 40

    def fragments_of(self, triangles):
        tri_xy = np.array(triangles, dtype=np.float64)
        boxes = triangle_boxes(tri_xy, self.WIDTH, self.HEIGHT)
        found = triangle_fragments(tri_xy, boxes, 0, len(boxes.triangles))
        if found is None:
            return set(), boxes
        tri, x, y, _ = found
        return set(zip(boxes.triangles[tri].tolist(), x.tolist(), y.tolist())), boxes

    def every_pixel_inside(self, index, triangle):
        (ax, ay), (bx, by), (cx, cy) = triangle
        area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if abs(area) <= 1e-9:
            return set()
        result = set()
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                w_b = ((x - ax) * (cy - ay) - (cx - ax) * (y - ay)) / area
                w_c = ((bx - ax) * (y - ay) - (x - ax) * (by - ay)) / area
                if min(w_b, w_c, 1 - w_b - w_c) >= -1e-7:  # the rasterizer's edge tolerance
                    result.add((index, x, y))
        return result

    def assert_matches_every_pixel_test(self, triangles):
        found, _ = self.fragments_of(triangles)
        expected = set()
        for index, triangle in enumerate(triangles):
            expected |= self.every_pixel_inside(index, triangle)
        self.assertEqual(found, expected)

    def test_random_triangles(self):
        rng = np.random.default_rng(7)
        triangles = rng.uniform((-20, -20), (80, 60), (150, 3, 2)).tolist()
        self.assert_matches_every_pixel_test(triangles)

    def test_slivers(self):
        self.assert_matches_every_pixel_test([[(0.5, 0.5), (55.5, 30.25), (55.75, 30.5)],
                                              [(3.2, 39.0), (3.4, 1.0), (3.6, 39.0)]])

    def test_corners_exactly_on_pixels_and_flat_edges(self):
        self.assert_matches_every_pixel_test([[(10, 10), (30, 10), (30, 25)],
                                              [(10, 10), (30, 25), (10, 25)],
                                              [(5, 5), (5, 30), (20, 5)],
                                              [(40, 5), (40, 5), (50, 20)],
                                              [(0, 0), (59, 0), (59, 39)]])

    def test_triangles_larger_than_the_buffer(self):
        self.assert_matches_every_pixel_test([[(-500, -500), (900, -400), (100, 800)],
                                              [(-100, 20), (200, 20.5), (50, 21)]])


class RasterizationTests(unittest.TestCase):
    def test_filled_not_outlined(self):
        frame, _ = render(model_of(quad((-5, 5), (-5, 5), 50)))
        self.assertTrue(changed(frame)[240, 320])

    def test_shared_edges_leave_no_holes(self):
        frame, _ = render(model_of(quad((-5, 5), (-5, 5), 50)))
        inside = changed(frame)[240 - 60:240 + 60, 320 - 60:320 + 60]
        self.assertTrue(inside.all())

    def test_closed_mesh_has_no_holes(self):
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
        sphere.apply_translation((0, 0, 50))
        frame, _ = render(model_of(MeshPart("s", sphere.vertices, sphere.faces, RED)))
        inside = changed(frame)[240 - 45:240 + 45, 320 - 45:320 + 45]
        yy, xx = np.mgrid[-45:45, -45:45]
        self.assertTrue(inside[xx ** 2 + yy ** 2 < 40 ** 2].all())

    def test_pixels_outside_the_model_are_untouched(self):
        frame, _ = render(model_of(quad((-5, 5), (-5, 5), 50)))
        self.assertTrue((frame[:100] == GREY).all())
        self.assertTrue((frame[:, :200] == GREY).all())

    def test_back_faces_are_culled(self):
        part = quad((-5, 5), (-5, 5), 50)
        flipped = MeshPart("flipped", part.vertices, part.faces[:, ::-1], RED)
        frame, _ = render(model_of(flipped))
        self.assertFalse(changed(frame).any())

    def test_partly_off_screen_model_is_drawn_without_error(self):
        frame, drawn = render(model_of(quad((-100, 0), (-5, 5), 50)))
        self.assertTrue(drawn)
        self.assertTrue(changed(frame)[240, 200])
        self.assertFalse(changed(frame)[240, 400])

    def test_fully_off_screen_model_changes_nothing(self):
        frame, drawn = render(model_of(quad((500, 600), (-5, 5), 50)))
        self.assertFalse(drawn)
        self.assertFalse(changed(frame).any())

    def test_screen_filling_mesh_near_the_camera_still_renders(self):
        # Many huge triangles need far more candidate pixels than one batch holds.
        parts = [quad((-50, 50), (-50, 50), 5.0 + i * 0.01, name=f"q{i}") for i in range(40)]
        frame, drawn = render(model_of(*parts))
        self.assertTrue(drawn)
        self.assertTrue(changed(frame).all())

    def test_same_input_renders_identically(self):
        model = generated_model_mesh()
        attach = attachment_matrix(DEFAULT_GLASSES_ATTACHMENT)
        pose = make_pose(head_rotation(yaw=20), (1.0, -2.0, 45.0))
        first, _ = render(model, pose, attach)
        second, _ = render(model, pose, attach)
        np.testing.assert_array_equal(first, second)


class RenderResolutionLimitTests(unittest.TestCase):
    """A model filling a large frame is rasterized smaller, then scaled up in place."""
    WIDTH, HEIGHT = 1920, 1080

    def render_large(self, depth_buffer=None):
        camera = camera_matrix(self.WIDTH, self.HEIGHT)
        pose = HeadPose(np.zeros((3, 1)), np.zeros((3, 1)), np.eye(3), camera)
        frame = np.full((self.HEIGHT, self.WIDTH, 3), GREY, np.uint8)
        # At f = 1920 and Z = 20, X = +/-5 cm spans 960 pixels, centred at 960.
        render_mesh(frame, model_of(quad((-5, 5), (-2, 2), 20)), IDENTITY, pose, depth_buffer)
        return frame

    def test_rasterized_width_is_limited(self):
        sizes = []
        real = mesh_renderer._resolve_surface

        def spy(tri_xy, tri_z, width, height):
            sizes.append(width)
            return real(tri_xy, tri_z, width, height)

        with mock.patch.object(mesh_renderer, "_resolve_surface", spy):
            self.render_large()
        self.assertLessEqual(sizes[0], mesh_renderer.MAX_RENDER_WIDTH + 1)

    def test_scaled_model_stays_where_it_belongs(self):
        mask = changed(self.render_large())
        ys, xs = np.nonzero(mask)
        self.assertAlmostEqual(xs.min(), 960 - 480, delta=3)
        self.assertAlmostEqual(xs.max(), 960 + 480, delta=3)
        self.assertAlmostEqual(ys.min(), 540 - 192, delta=3)
        self.assertAlmostEqual(ys.max(), 540 + 192, delta=3)

    def test_scaled_model_is_centred_with_no_half_pixel_drift(self):
        ys, xs = np.nonzero(changed(self.render_large()))
        self.assertAlmostEqual((xs.min() + xs.max()) / 2, 960, delta=1)
        self.assertAlmostEqual((ys.min() + ys.max()) / 2, 540, delta=1)

    def test_scaled_model_is_solid(self):
        self.assertTrue(changed(self.render_large())[540 - 180:540 + 180, 960 - 470:960 + 470].all())

    def test_head_occlusion_lands_at_the_right_image_position(self):
        depth = np.full((self.HEIGHT, self.WIDTH), np.inf, np.float32)
        depth[:, :960] = 10  # the head covers the left half of the image
        mask = changed(self.render_large(DepthBuffer(depth, 1.0)))
        self.assertAlmostEqual(np.nonzero(mask)[1].min(), 960, delta=4)

    def test_small_models_are_not_scaled(self):
        sizes = []
        real = mesh_renderer._resolve_surface

        def spy(tri_xy, tri_z, width, height):
            sizes.append(width)
            return real(tri_xy, tri_z, width, height)

        with mock.patch.object(mesh_renderer, "_resolve_surface", spy):
            render(model_of(quad((-5, 5), (-5, 5), 50)))
        self.assertGreaterEqual(sizes[0], 128)  # its full width on screen


class BatchingTests(unittest.TestCase):
    """Huge triangles are rasterized in batches so memory stays bounded."""

    def batches(self, counts, limit):
        return list(mesh_renderer._batches(counts, limit))

    def test_every_triangle_is_in_exactly_one_batch(self):
        self.assertEqual(self.batches([3, 3, 3, 3, 3], 7), [(0, 2), (2, 4), (4, 5)])

    def test_batches_stay_within_the_limit(self):
        counts = [5, 1, 9, 2, 2, 2, 8]
        for start, end in self.batches(counts, 10):
            self.assertLessEqual(sum(counts[start:end]), 10)

    def test_a_triangle_over_the_limit_gets_its_own_batch(self):
        self.assertEqual(self.batches([2, 50, 2], 10), [(0, 1), (1, 2), (2, 3)])

    def test_no_triangles_means_no_batches(self):
        self.assertEqual(self.batches([], 10), [])


class DepthBufferingTests(unittest.TestCase):
    def assert_center_is_red(self, parts):
        frame, _ = render(model_of(*parts))
        b, g, r = frame[240, 320]
        self.assertGreater(int(r), int(g) + 30)

    def test_nearer_surface_hides_farther_surface_in_either_order(self):
        near = quad((-5, 5), (-5, 5), 40, RED, name="near")
        far = quad((-5, 5), (-5, 5), 60, GREEN, name="far")
        self.assert_center_is_red([near, far])
        self.assert_center_is_red([far, near])

    def test_interpenetrating_surfaces_cross_at_the_right_pixel(self):
        # Red tilts from Z = 40 at the left to Z = 60 at the right; green is flat at 50.
        tilted = np.array([(-5, -5, 40), (5, -5, 60), (5, 5, 60), (-5, 5, 40)], float)
        red = MeshPart("red", tilted, np.array([(0, 2, 1), (0, 3, 2)]), RED)
        green = quad((-5, 5), (-5, 5), 50, GREEN, name="green")
        frame, _ = render(model_of(red, green))
        left, right = frame[240, 320 - 40], frame[240, 320 + 40]
        self.assertGreater(int(left[2]), int(left[1]))   # red is nearer on the left
        self.assertGreater(int(right[1]), int(right[2]))  # green is nearer on the right
        # X = 0 at Z = 50 is the crossing, which projects to the image centre.
        row = frame[240]
        crossing = np.flatnonzero(row[:, 1] > row[:, 2]).min()
        self.assertAlmostEqual(crossing, 320, delta=2)


class NearPlaneClippingTests(unittest.TestCase):
    NEAR = 1.0

    def clip(self, triangle, values=None):
        corners = np.array([triangle], float)
        values = np.tile((0.0, 0.0, -1.0), (1, 3, 1)) if values is None else np.array([values], float)
        return clip_triangles_to_near_plane(corners, values, self.NEAR)

    def test_triangle_in_front_is_unchanged(self):
        corners, _, faces = self.clip([(0, 0, 5), (1, 0, 5), (0, 1, 5)])
        np.testing.assert_array_equal(corners, [[(0, 0, 5), (1, 0, 5), (0, 1, 5)]])
        np.testing.assert_array_equal(faces, [0])

    def test_triangle_behind_is_removed(self):
        corners, _, faces = self.clip([(0, 0, -5), (1, 0, -5), (0, 1, 0.5)])
        self.assertEqual(len(corners), 0)
        self.assertEqual(len(faces), 0)

    def test_one_corner_in_front_leaves_one_triangle(self):
        corners, _, _ = self.clip([(0, 0, 3), (2, 0, -1), (0, 2, -1)])
        self.assertEqual(len(corners), 1)
        self.assertTrue((corners[..., 2] >= self.NEAR - 1e-9).all())
        # (0,0,3) -> (2,0,-1) meets Z = 1 halfway.
        self.assertTrue(np.isclose(corners[0], (1, 0, 1)).all(axis=1).any())

    def test_two_corners_in_front_leave_two_triangles(self):
        corners, _, _ = self.clip([(0, 0, 3), (2, 0, 3), (0, 2, -1)])
        self.assertEqual(len(corners), 2)
        self.assertTrue((corners[..., 2] >= self.NEAR - 1e-9).all())

    def test_winding_is_preserved(self):
        for triangle in ([(0, 0, 3), (2, 0, -1), (0, 2, -1)], [(0, 0, 3), (2, 0, 3), (0, 2, -1)],
                         [(2, 0, -1), (0, 0, 3), (0, 2, -1)], [(0, 2, -1), (0, 0, 3), (2, 0, 3)]):
            with self.subTest(triangle=triangle):
                original = np.cross(np.subtract(triangle[1], triangle[0]),
                                    np.subtract(triangle[2], triangle[0]))
                corners, _, _ = self.clip(triangle)
                for a, b, c in corners:
                    self.assertGreater(np.cross(b - a, c - a) @ original, 0)

    def test_pieces_point_back_to_their_original_face(self):
        corners = np.array([[(0, 0, 5), (1, 0, 5), (0, 1, 5)],
                            [(0, 0, 3), (2, 0, 3), (0, 2, -1)],
                            [(0, 0, -3), (2, 0, -3), (0, 2, -1)],
                            [(0, 0, 3), (2, 0, -1), (0, 2, -1)]], float)
        values = np.zeros((4, 3, 1))
        _, _, faces = clip_triangles_to_near_plane(corners, values, self.NEAR)
        self.assertEqual(sorted(faces.tolist()), [0, 1, 1, 3])

    def test_corner_values_interpolate_along_cut_edges(self):
        # Normals and texture coordinates alike: (nx, ny, nz, u, v).
        values = [(1, 0, 0, 0.0, 0.0), (0, 1, 0, 1.0, 0.0), (1, 0, 0, 0.0, 1.0)]
        clipped, new_values, _ = self.clip([(0, 0, 3), (2, 0, -1), (0, 2, 3)], values)
        at_cut = np.isclose(clipped, (1, 0, 1)).all(axis=2)
        np.testing.assert_allclose(new_values[at_cut][0], (0.5, 0.5, 0, 0.5, 0.0))

    def test_triangle_crossing_the_camera_renders_only_its_front_part(self):
        vertices = np.array([(-3, 3, 10), (3, 3, 10), (0, 5, -10)], float)
        part = MeshPart("through", vertices, np.array([(0, 2, 1)]), RED)
        frame, drawn = render(model_of(part))
        self.assertTrue(drawn)
        self.assertTrue(changed(frame).any())
        # Unclipped, the corner behind the camera would mirror across the image.
        self.assertEqual(changed(frame)[:100].sum(), 0)


def textured_quad(x_range, y_range, z_values, texture, name="tex", uv_box=((0, 0), (1, 1))):
    """Quad facing the camera, textured with u along X and v up the image (towards -Y).

    z_values -- Z at the four corners: (x0, y0), (x1, y0), (x1, y1), (x0, y1)
    """
    (x0, x1), (y0, y1) = x_range, y_range
    vertices = np.array([(x0, y0, z_values[0]), (x1, y0, z_values[1]),
                         (x1, y1, z_values[2]), (x0, y1, z_values[3])], float)
    (u0, v0), (u1, v1) = uv_box
    # Camera Y points down, so the top of the image (v1) goes at y0.
    uvs = np.array([(u0, v1), (u1, v1), (u1, v0), (u0, v0)], float)
    return MeshPart(name, vertices, np.array([(0, 2, 1), (0, 3, 2)]), (255, 255, 255), 0.0,
                    uvs=uvs, texture=texture)


def quadrant_texture(size=64):
    """Top-left red, top-right green, bottom-left blue, bottom-right yellow (BGR)."""
    texture = np.zeros((size, size, 3), np.uint8)
    half = size // 2
    texture[:half, :half] = (0, 0, 230)
    texture[:half, half:] = (0, 230, 0)
    texture[half:, :half] = (230, 0, 0)
    texture[half:, half:] = (0, 230, 230)
    return texture


def dominant(pixel):
    b, g, r = (int(value) for value in pixel)
    if r > 120 and g > 120 and b < 80:
        return "yellow"
    return ("blue", "green", "red")[int(np.argmax((b, g, r)))]


class TextureTests(unittest.TestCase):
    def test_texture_lands_the_right_way_up_and_round(self):
        frame, drawn = render(model_of(textured_quad((-5, 5), (-5, 5), (50,) * 4, quadrant_texture())))
        self.assertTrue(drawn)
        self.assertEqual(dominant(frame[240 - 30, 320 - 30]), "red")
        self.assertEqual(dominant(frame[240 - 30, 320 + 30]), "green")
        self.assertEqual(dominant(frame[240 + 30, 320 - 30]), "blue")
        self.assertEqual(dominant(frame[240 + 30, 320 + 30]), "yellow")

    def test_photo_colours_survive_the_lighting_when_facing_the_camera(self):
        texture = np.full((8, 8, 3), (60, 120, 180), np.uint8)
        frame, _ = render(model_of(textured_quad((-5, 5), (-5, 5), (50,) * 4, texture)))
        np.testing.assert_allclose(frame[240, 320].astype(int), (60, 120, 180), atol=8)

    def test_material_colour_tints_the_texture_as_in_gltf(self):
        texture = np.full((8, 8, 3), (60, 120, 180), np.uint8)
        part = textured_quad((-5, 5), (-5, 5), (50,) * 4, texture)
        tinted = MeshPart(part.name, part.vertices, part.faces, (128, 128, 128), 0.0,
                          uvs=part.uvs, texture=part.texture)
        plain, _ = render(model_of(part))
        half, _ = render(model_of(tinted))
        np.testing.assert_allclose(half[240, 320].astype(int), plain[240, 320] / 2, atol=3)

    def test_texture_interpolation_is_perspective_correct(self):
        # Left edge at Z = 40, right edge at Z = 80. The texture's middle is at X = 0,
        # Z = 60, which projects to the image centre. Affine interpolation would put it
        # at x = 300, halfway between the edges on screen (240 and 360).
        texture = np.zeros((4, 64, 3), np.uint8)
        texture[:, :32] = (0, 0, 230)
        texture[:, 32:] = (0, 230, 0)
        quad_part = textured_quad((-5, 5), (-5, 5), (40, 80, 80, 40), texture)
        frame, _ = render(model_of(quad_part))
        row = frame[240].astype(int)
        boundary = np.flatnonzero(row[:, 1] > row[:, 2]).min()
        self.assertAlmostEqual(boundary, 320, delta=2)

    def test_near_plane_clipping_keeps_the_texture_in_place(self):
        # The quad's top runs behind the camera (Z = 4Y + 5, so Z < 1 above Y = -1); the
        # colour boundary must stay at X = 0, which projects to x = 320 on every row.
        texture = np.zeros((4, 64, 3), np.uint8)
        texture[:, :32] = (0, 0, 230)
        texture[:, 32:] = (0, 230, 0)
        quad_part = textured_quad((-5, 5), (-5, 5), (-15, -15, 25, 25), texture)
        frame, drawn = render(model_of(quad_part))
        self.assertTrue(drawn)
        for y in (60, 200, 360):
            row = frame[y].astype(int)
            boundary = np.flatnonzero(row[:, 1] > row[:, 2]).min()
            self.assertAlmostEqual(boundary, 320, delta=2, msg=f"row {y}")

    def test_fine_detail_far_away_averages_instead_of_sparkling(self):
        texture = np.zeros((256, 256, 3), np.uint8)
        texture[(np.arange(256)[:, None] + np.arange(256)[None, :]) % 2 == 0] = 255
        # At Z = 90 the quad is 71 px across, about 3.6 texels per pixel. Sampling only
        # the full-size texture measured a standard deviation of 16.8 here; some other
        # distances put every sample between squares, which hides the difference.
        frame, _ = render(model_of(textured_quad((-5, 5), (-5, 5), (90,) * 4, texture)))
        inside = frame[240 - 20:240 + 20, 320 - 20:320 + 20, 1].astype(int)
        self.assertLess(inside.std(), 6)
        self.assertAlmostEqual(inside.mean(), 125, delta=15)

    def test_levels_blend_so_detail_fades_without_a_jump(self):
        from glb_model import _pyramid
        texture = _pyramid(quadrant_texture(64))
        uvs = np.array([(0.49, 0.75), (0.2, 0.3), (0.51, 0.24)])
        finer = mesh_renderer._sample_texture(texture, uvs, 1.0)
        coarser = mesh_renderer._sample_texture(texture, uvs, 2.0)
        self.assertGreater(np.abs(finer - coarser).max(), 5)  # the levels really differ here
        np.testing.assert_allclose(mesh_renderer._sample_texture(texture, uvs, 1.5),
                                   (finer + coarser) / 2, atol=1e-3)

    def test_texture_detail_is_kept_up_close(self):
        frame, _ = render(model_of(textured_quad((-5, 5), (-5, 5), (50,) * 4, quadrant_texture())))
        # Sharp quadrant boundary: a few pixels from red to green.
        row = frame[240 - 30, 320 - 6:320 + 6].astype(int)
        self.assertEqual(dominant(row[0]), "red")
        self.assertEqual(dominant(row[-1]), "green")

    def test_textured_surfaces_are_depth_tested(self):
        near = textured_quad((-5, 5), (-5, 5), (40,) * 4, quadrant_texture(), "near")
        far = quad((-5, 5), (-5, 5), 60, (128, 0, 128), name="far")
        for parts in ((near, far), (far, near)):
            frame, _ = render(model_of(*parts))
            self.assertEqual(dominant(frame[240 - 30, 320 - 30]), "red")

    def test_textured_surfaces_are_hidden_by_the_head(self):
        depth = np.full((HEIGHT, WIDTH), np.inf, np.float32)
        depth[:, :320] = 40
        part = textured_quad((-5, 5), (-5, 5), (50,) * 4, quadrant_texture())
        frame, _ = render(model_of(part), depth_buffer=DepthBuffer(depth, 1.0))
        self.assertFalse(changed(frame)[240 - 30, 300])
        self.assertEqual(dominant(frame[240 - 30, 320 + 30]), "green")

    def test_textured_and_plain_parts_together(self):
        textured = textured_quad((-12, -2), (-5, 5), (50,) * 4, quadrant_texture(), "tex")
        plain = quad((2, 12), (-5, 5), 50, (200, 0, 0), name="plain")
        frame, _ = render(model_of(textured, plain))
        self.assertEqual(dominant(frame[240 - 30, 320 - 130]), "red")
        self.assertEqual(dominant(frame[240, 320 + 90]), "blue")


class ShadingTests(unittest.TestCase):
    def test_material_colour_shows_through(self):
        frame, _ = render(model_of(quad((-5, 5), (-5, 5), 50, (0, 0, 200))))
        b, g, r = frame[240, 320]
        self.assertGreater(int(r), int(b) + 50)
        self.assertGreater(int(r), int(g) + 50)

    def test_curved_surface_shades_visibly(self):
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
        sphere.apply_translation((0, 0, 50))
        frame, _ = render(model_of(MeshPart("s", sphere.vertices, sphere.faces, (120, 120, 120))))
        inside = frame[240 - 40:240 + 40, 320 - 40:320 + 40]
        self.assertGreater(int(inside.max()) - int(inside.min()), 40)

    def test_light_comes_from_above(self):
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
        sphere.apply_translation((0, 0, 50))
        frame, _ = render(model_of(MeshPart("s", sphere.vertices, sphere.faces, (120, 120, 120))))
        self.assertGreater(int(frame[240 - 30, 320].sum()), int(frame[240 + 30, 320].sum()))

    def test_smooth_surface_has_no_visible_facet_steps(self):
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
        sphere.apply_translation((0, 0, 50))
        frame, _ = render(model_of(MeshPart("s", sphere.vertices, sphere.faces, (120, 120, 120))))
        row = frame[240, 320 - 40:320 + 40, 1].astype(int)
        self.assertLess(np.abs(np.diff(row)).max(), 12)

    def test_normals_are_interpolated_across_large_faces(self):
        # Measured steps on this sphere: 2 with interpolation, 10 with flat shading.
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
        sphere.apply_translation((0, 0, 50))
        frame, _ = render(model_of(MeshPart("s", sphere.vertices, sphere.faces, (120, 120, 120))))
        column = frame[240 - 40:240 + 40, 320, 1].astype(int)
        self.assertLessEqual(np.abs(np.diff(column)).max(), FACET_STEP_LIMIT)

    def test_gloss_adds_a_highlight(self):
        def brightest(gloss):
            sphere = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
            sphere.apply_translation((0, 0, 50))
            frame, _ = render(model_of(MeshPart("s", sphere.vertices, sphere.faces, (40, 40, 40), gloss)))
            return int(frame[240 - 40:240 + 40, 320 - 40:320 + 40].max())  # inside the sphere
        self.assertGreater(brightest(0.9), brightest(0.0) + 30)

    def test_flat_glossy_black_frame_does_not_wash_out_when_facing_the_light(self):
        # A flat front at the angle that mirrors the light must still read as a dark frame.
        half = HALF_VECTOR
        u = np.cross(half, (1.0, 0.0, 0.0))
        u /= np.linalg.norm(u)
        v = np.cross(half, u)
        center = np.array((0.0, 0.0, 50.0))
        vertices = np.array([center + a * u + b * v for a, b in ((-5, -5), (5, -5), (5, 5), (-5, 5))])
        faces = np.array([(0, 1, 2), (0, 2, 3)])
        if np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]) @ half < 0:
            faces = faces[:, ::-1]
        part = MeshPart("front", vertices, faces, (30, 30, 34), 0.65)
        frame, _ = render(model_of(part))
        self.assertLess(int(frame[240, 320].max()), 110)

    def test_edges_are_antialiased(self):
        frame, _ = render(model_of(quad((-5, 5), (-5, 5), 50, (0, 0, 0))))
        values = np.unique(frame[240, 320 - 80:320 - 50, 1])
        self.assertGreater(len(values), 2)  # more than just background and model


class HeadWallOcclusionTests(unittest.TestCase):
    """A depth buffer holding a flat wall stands in for the head."""

    def wall(self, z, x0=0, x1=WIDTH):
        depth = np.full((HEIGHT, WIDTH), np.inf, np.float32)
        depth[:, x0:x1] = z
        return DepthBuffer(depth, 1.0)

    def test_wall_in_front_hides_the_model(self):
        frame, _ = render(model_of(quad((-5, 5), (-5, 5), 50)), depth_buffer=self.wall(40))
        self.assertFalse(changed(frame).any())

    def test_wall_behind_changes_nothing(self):
        plain, _ = render(model_of(quad((-5, 5), (-5, 5), 50)))
        occluded, _ = render(model_of(quad((-5, 5), (-5, 5), 50)), depth_buffer=self.wall(60))
        np.testing.assert_array_equal(plain, occluded)

    def test_wall_hides_only_what_it_covers(self):
        frame, _ = render(model_of(quad((-5, 5), (-5, 5), 50)), depth_buffer=self.wall(40, 0, 320))
        self.assertFalse(changed(frame)[240, 300])
        self.assertTrue(changed(frame)[240, 340])

    def test_small_gap_is_tolerated_like_the_wireframe(self):
        # Within DEPTH_TOLERANCE_CM of the head surface still counts as visible.
        frame, _ = render(model_of(quad((-5, 5), (-5, 5), 50)), depth_buffer=self.wall(49.8))
        self.assertTrue(changed(frame)[240, 320])

    def test_reduced_resolution_buffer_is_scaled(self):
        depth = np.full((HEIGHT // 2, WIDTH // 2), np.inf, np.float32)
        depth[:, :160] = 40  # left half of the image
        frame, _ = render(model_of(quad((-5, 5), (-5, 5), 50)), depth_buffer=DepthBuffer(depth, 0.5))
        self.assertFalse(changed(frame)[240, 300])
        self.assertTrue(changed(frame)[240, 340])


class GuardTests(unittest.TestCase):
    def test_no_pose_draws_nothing(self):
        frame = blank()
        self.assertFalse(render_mesh(frame, model_of(quad((-5, 5), (-5, 5), 50)), IDENTITY, None))
        self.assertFalse(changed(frame).any())

    def test_model_behind_the_camera_draws_nothing(self):
        frame, drawn = render(model_of(quad((-5, 5), (-5, 5), -50)))
        self.assertFalse(drawn)
        self.assertFalse(changed(frame).any())


_cached = {}


def generated_model_mesh():
    if "model" not in _cached:
        directory = tempfile.TemporaryDirectory()
        _cached["dir"] = directory
        path = export_model_glb(generated_model(GlassesParameters()), Path(directory.name) / "m.glb")
        _cached["model"] = load_glasses_model(path)
    return _cached["model"]


def only(model, *names):
    return MeshModel(model.name, tuple(model.part(name) for name in names))


class FaceAttachmentTests(unittest.TestCase):
    """The generated model on Mirror's real attachment and pose transforms."""
    ATTACH = attachment_matrix(DEFAULT_GLASSES_ATTACHMENT)

    @classmethod
    def setUpClass(cls):
        cls.model = generated_model_mesh()
        cls.front = only(cls.model, FRONT_FRAME)

    def center_of(self, pose, model=None):
        frame, _ = render(model or self.front, pose, self.ATTACH)
        ys, xs = np.nonzero(changed(frame))
        return (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2

    def landmark_pixel(self, landmark_id, pose):
        point = np.array([POSE_LANDMARKS[landmark_id]], float)
        return project_camera_points(transform_points(pose.transform, point), CAMERA)[0]

    def test_frame_is_centred_on_the_nose_bridge(self):
        pose = make_pose(FRONTAL_ROTATION, (0.0, 0.0, 45.0))
        x, _ = self.center_of(pose)
        self.assertAlmostEqual(x, self.landmark_pixel(168, pose)[0], delta=2)

    def test_frame_sits_around_the_eyes_not_on_the_forehead_or_cheeks(self):
        pose = make_pose(FRONTAL_ROTATION, (0.0, 0.0, 45.0))
        frame, _ = render(self.front, pose, self.ATTACH)
        ys, _ = np.nonzero(changed(frame))
        for eye in (33, 263, 133, 362):
            eye_y = self.landmark_pixel(eye, pose)[1]
            self.assertGreater(eye_y, ys.min())
            self.assertLess(eye_y, ys.max())

    def test_frame_is_wider_than_the_eyes(self):
        pose = make_pose(FRONTAL_ROTATION, (0.0, 0.0, 45.0))
        frame, _ = render(self.front, pose, self.ATTACH)
        _, xs = np.nonzero(changed(frame))
        self.assertLess(xs.min(), self.landmark_pixel(33, pose)[0])
        self.assertGreater(xs.max(), self.landmark_pixel(263, pose)[0])

    def test_translation_moves_the_model_with_the_head(self):
        base = make_pose(FRONTAL_ROTATION, (0.0, 0.0, 45.0))
        # The frame's front is 5.7 cm in front of the head's origin, so 39.3 cm from the camera.
        pixels = 5 * 640 / (45 - (FACE_ORIGIN[2] + DEFAULT_GLASSES_ATTACHMENT.offset_cm[2]))
        for offset, expected in (((5.0, 0.0), (pixels, 0)), ((0.0, 5.0), (0, pixels))):
            moved = make_pose(FRONTAL_ROTATION, (offset[0], offset[1], 45.0))
            x0, y0 = self.center_of(base)
            x1, y1 = self.center_of(moved)
            self.assertAlmostEqual(x1 - x0, expected[0], delta=3)
            self.assertAlmostEqual(y1 - y0, expected[1], delta=3)

    def test_moving_away_shrinks_the_model(self):
        def width(z):
            frame, _ = render(self.front, make_pose(FRONTAL_ROTATION, (0.0, 0.0, z)), self.ATTACH)
            _, xs = np.nonzero(changed(frame))
            return np.ptp(xs)
        self.assertAlmostEqual(width(40) / width(80), 2.0, delta=0.15)

    def test_right_lens_is_on_screen_right_so_nothing_is_mirrored(self):
        right = only(self.model, RIGHT_TEMPLE)
        left = only(self.model, LEFT_TEMPLE)
        pose = make_pose(FRONTAL_ROTATION, (0.0, 0.0, 45.0))
        self.assertGreater(self.center_of(pose, right)[0], 320)
        self.assertLess(self.center_of(pose, left)[0], 320)

    def test_roll_follows_the_head_clockwise(self):
        # Clockwise on screen: the screen-right side goes down, like the 2D tilt.
        pose = make_pose(head_rotation(roll=20), (0.0, 0.0, 45.0))
        right = only(self.model, RIGHT_TEMPLE)
        left = only(self.model, LEFT_TEMPLE)
        self.assertGreater(self.center_of(pose, right)[1], self.center_of(pose, left)[1])

    def test_yaw_towards_screen_right_moves_the_frame_centre_that_way(self):
        pose = make_pose(head_rotation(yaw=30), (0.0, 0.0, 45.0))
        origin = np.array([[0.0, 0.0, 0.0]])
        camera = transform_points(pose.transform @ self.ATTACH, origin)
        self.assertGreater(project_camera_points(camera, CAMERA)[0, 0], 320)
        # The nose bridge landmark moves with it, within a few pixels.
        self.assertGreater(self.landmark_pixel(168, pose)[0], 320)

    def test_pitch_up_raises_the_frame(self):
        pose = make_pose(head_rotation(pitch=20), (0.0, 0.0, 45.0))
        base = make_pose(FRONTAL_ROTATION, (0.0, 0.0, 45.0))
        self.assertLess(self.center_of(pose)[1], self.center_of(base)[1])

    def test_fit_moves_the_rendered_model_without_changing_the_mesh(self):
        pose = make_pose(FRONTAL_ROTATION, (0.0, 0.0, 45.0))
        vertices_before = self.model.part(FRONT_FRAME).vertices.copy()
        base = self.center_of(pose)
        frame, _ = render(self.front, pose,
                          fit_matrix(DEFAULT_GLASSES_ATTACHMENT, FitAdjustment(y=1.0)))
        ys, _ = np.nonzero(changed(frame))
        self.assertLess((ys.min() + ys.max()) / 2, base[1] - 10)
        np.testing.assert_array_equal(self.model.part(FRONT_FRAME).vertices, vertices_before)

    def test_fit_scale_changes_the_rendered_size(self):
        pose = make_pose(FRONTAL_ROTATION, (0.0, 0.0, 45.0))
        plain, _ = render(self.front, pose, self.ATTACH)
        scaled, _ = render(self.front, pose,
                           fit_matrix(DEFAULT_GLASSES_ATTACHMENT, FitAdjustment(scale=1.2)))
        self.assertAlmostEqual(np.ptp(np.nonzero(changed(scaled))[1])
                               / np.ptp(np.nonzero(changed(plain))[1]), 1.2, delta=0.03)


class HeadOcclusionTests(unittest.TestCase):
    """Real head occluder and real generated model at several head angles."""
    ATTACH = attachment_matrix(DEFAULT_GLASSES_ATTACHMENT)

    @classmethod
    def setUpClass(cls):
        cls.model = generated_model_mesh()
        cls.occluder = load_head_occluder()

    def visible(self, part, yaw=0.0, occlude=True, pitch=0.0):
        pose = make_pose(head_rotation(yaw=yaw, pitch=pitch), (0.0, 0.0, 50.0))
        buffer = render_depth_buffer(self.occluder, pose, WIDTH, HEIGHT) if occlude else None
        frame, _ = render(only(self.model, part), pose, self.ATTACH, buffer)
        return int(changed(frame).sum())

    def fraction(self, part, yaw=0.0):
        return self.visible(part, yaw) / max(1, self.visible(part, yaw, occlude=False))

    def test_frontal_front_frame_is_in_front_of_the_face(self):
        self.assertGreater(self.fraction(FRONT_FRAME), 0.95)

    def test_frontal_temples_disappear_behind_the_head_towards_the_ears(self):
        for part in (LEFT_TEMPLE, RIGHT_TEMPLE):
            with self.subTest(part=part):
                self.assertLess(self.visible(part), self.visible(part, occlude=False))

    def test_three_quarter_view_hides_the_far_temple_and_keeps_the_near_one(self):
        # Nose towards screen-right: the head's screen-left side faces the camera.
        # The far temple's first stretch, by the hinge, really is visible past the face.
        self.assertGreater(self.fraction(LEFT_TEMPLE, 40), 0.8)
        self.assertLess(self.fraction(RIGHT_TEMPLE, 40), 0.4)

    def test_opposite_yaw_swaps_which_temple_is_hidden(self):
        self.assertGreater(self.fraction(RIGHT_TEMPLE, -40), 0.8)
        self.assertLess(self.fraction(LEFT_TEMPLE, -40), 0.4)

    def test_side_view_keeps_the_near_temple_and_hides_the_far_one(self):
        self.assertGreater(self.fraction(LEFT_TEMPLE, 75), 0.9)
        self.assertLess(self.fraction(RIGHT_TEMPLE, 75), 0.05)

    def test_nose_hides_part_of_the_frame_when_the_head_turns(self):
        self.assertGreater(self.fraction(FRONT_FRAME, 0), self.fraction(FRONT_FRAME, 60) + 0.02)

    def test_missing_depth_buffer_draws_everything(self):
        self.assertGreater(self.visible(RIGHT_TEMPLE, 40, occlude=False),
                           self.visible(RIGHT_TEMPLE, 40))


@unittest.skipUnless(PHOTO_MODEL.exists(), "generated example model is not available")
class PhotoModelTests(unittest.TestCase):
    """The initial test model from the brief."""
    ATTACH = attachment_matrix(DEFAULT_GLASSES_ATTACHMENT)

    @classmethod
    def setUpClass(cls):
        cls.model = load_glasses_model(PHOTO_MODEL)
        cls.occluder = load_head_occluder()

    def test_eyes_look_through_the_lens_openings(self):
        pose = make_pose(FRONTAL_ROTATION, (0.0, 0.0, 45.0))
        frame, _ = render(only(self.model, FRONT_FRAME), pose, self.ATTACH)
        for inner, outer in ((33, 133), (263, 362)):
            middle = (np.array(POSE_LANDMARKS[inner]) + np.array(POSE_LANDMARKS[outer])) / 2
            x, y = np.round(project_camera_points(
                transform_points(pose.transform, middle[None]), CAMERA)[0]).astype(int)
            self.assertFalse(changed(frame)[y, x], f"eye at {(x, y)} is covered by the frame")

    def test_bridge_crosses_above_the_nose_centre(self):
        pose = make_pose(FRONTAL_ROTATION, (0.0, 0.0, 45.0))
        frame, _ = render(only(self.model, FRONT_FRAME), pose, self.ATTACH)
        column = changed(frame)[:, 320]
        self.assertTrue(column.any())
        bridge_y = np.flatnonzero(column).mean()
        nose_bridge = project_camera_points(
            transform_points(pose.transform, np.array([POSE_LANDMARKS[168]], float)), CAMERA)[0, 1]
        self.assertAlmostEqual(bridge_y, nose_bridge, delta=25)


if __name__ == "__main__":
    unittest.main()
