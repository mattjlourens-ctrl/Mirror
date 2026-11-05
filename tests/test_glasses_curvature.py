import unittest

import cv2
import numpy as np
from shapely.geometry import Polygon

from image_to_3d.curvature import (CHORD_TOLERANCE_CM, build_curved_frame, split_faces_at_x,
                                   wrap_offset)
from image_to_3d.mesh import (FrameDimensions, mask_to_polygons, polygons_to_mesh,
                              scale_polygons_to_cm)
from image_to_3d.segmentation import (SegmentationConfig, load_image, remove_opening_intrusions,
                                      segment_glasses)
from tests.test_image_to_3d import (FRAME_COLOR, LEFT_OPENING_CENTER, LEFT_RIM, REPO_ROOT, RIGHT_RIM,
                                    RIM_THICKNESS, covers_xy, pixel_to_mesh_xy, white_image)

WIDTH = 14.0
PHOTO = REPO_ROOT / "image_to_3d" / "glasses.jpg"


def rim_polygons_cm(width=WIDTH):
    return scale_polygons_to_cm(mask_to_polygons(segment_glasses(white_image())), width)


def rims_with_temple_ghost():
    """Rims with a thin strip seen through the left lens, joined to the rim's top."""
    image = white_image()
    cv2.line(image, (70, 50), (120, 130), FRAME_COLOR, 6)
    return image


class WrapOffsetTests(unittest.TestCase):
    def test_centre_stays_on_the_front_plane(self):
        self.assertEqual(wrap_offset(0.0, WIDTH, 0.5), 0.0)

    def test_outer_edges_sag_back_by_the_depth(self):
        np.testing.assert_allclose(wrap_offset(np.array([-7.0, 7.0]), WIDTH, 0.5), [-0.5, -0.5])

    def test_matches_the_quadratic_at_a_hand_computed_point(self):
        # -0.8 * (3.5 / 7)^2 = -0.2
        self.assertAlmostEqual(wrap_offset(3.5, WIDTH, 0.8), -0.2)

    def test_zero_depth_is_flat(self):
        np.testing.assert_array_equal(wrap_offset(np.array([-7.0, 2.0, 7.0]), WIDTH, 0.0),
                                      [0.0, 0.0, 0.0])


class SplitFacesTests(unittest.TestCase):
    def split(self, vertices, faces, x):
        vertices = np.array(vertices, dtype=float)
        faces = np.array(faces)
        return split_faces_at_x(vertices, faces, x)

    @staticmethod
    def areas(vertices, faces):
        tris = vertices[faces]
        return np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) / 2

    def test_face_away_from_the_plane_is_untouched(self):
        vertices, faces = self.split([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)], 5.0)
        self.assertEqual(len(faces), 1)
        self.assertEqual(len(vertices), 3)

    def test_lone_vertex_case_makes_three_faces_and_keeps_area(self):
        vertices, faces = self.split([(0, 0, 0), (2, 0, 0), (0, 2, 0)], [(0, 1, 2)], 1.0)
        self.assertEqual(len(faces), 3)
        self.assertAlmostEqual(self.areas(vertices, faces).sum(), 2.0)

    def test_vertex_on_plane_case_makes_two_faces(self):
        vertices, faces = self.split([(0, 0, 0), (2, 0, 0), (1, 2, 0)], [(0, 1, 2)], 1.0)
        self.assertEqual(len(faces), 2)
        self.assertAlmostEqual(self.areas(vertices, faces).sum(), 2.0)

    def test_no_resulting_face_crosses_the_plane(self):
        vertices, faces = self.split([(0, 0, 0), (2, 0, 0), (0, 2, 0)], [(0, 1, 2)], 0.7)
        xs = vertices[faces][:, :, 0]
        self.assertTrue(np.all((xs.max(axis=1) <= 0.7 + 1e-9) | (xs.min(axis=1) >= 0.7 - 1e-9)))

    def test_winding_is_preserved(self):
        vertices, faces = self.split([(0, 0, 0), (2, 0, 0), (0, 2, 0)], [(0, 1, 2)], 1.0)
        tris = vertices[faces]
        normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        self.assertTrue(np.all(normals[:, 2] > 0))

    def test_faces_sharing_an_edge_share_the_new_vertex(self):
        vertices, _ = self.split([(0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 2, 0)],
                                     [(0, 1, 2), (1, 3, 2)], 1.0)
        # Two crossed outer edges and one shared diagonal: three new vertices, not four.
        self.assertEqual(len(vertices), 4 + 3)

    def test_vertex_a_hair_from_the_plane_makes_no_sliver(self):
        vertices, faces = self.split([(0, 0, 0), (2, 0, 0), (1 + 1e-9, 2, 0)], [(0, 1, 2)], 1.0)
        self.assertGreater(self.areas(vertices, faces).min(), 0.1)


class CurvedFrameTests(unittest.TestCase):
    def test_mesh_is_a_valid_closed_volume(self):
        for depth in (0.0, 0.3, 1.0, 3.0):
            with self.subTest(depth=depth):
                self.assertTrue(build_curved_frame(rim_polygons_cm(), 0.4, depth).is_volume)

    def test_front_outline_is_preserved_exactly(self):
        polygons = rim_polygons_cm()
        flat = build_curved_frame(polygons, 0.4, 0.0)
        curved = build_curved_frame(polygons, 0.4, 1.0)
        np.testing.assert_allclose(curved.bounds[:, :2], flat.bounds[:, :2], atol=1e-9)

        def footprint(mesh):
            front = mesh.triangles[mesh.face_normals[:, 2] > 0][:, :, :2]
            return sum(Polygon(t).area for t in front)

        self.assertAlmostEqual(footprint(curved), footprint(flat), places=6)

    def test_openings_stay_open(self):
        mask = segment_glasses(white_image())
        mesh = build_curved_frame(rim_polygons_cm(), 0.4, 1.0)
        self.assertFalse(covers_xy(mesh, pixel_to_mesh_xy(LEFT_OPENING_CENTER, mask, WIDTH)))

    def test_surfaces_follow_the_curve_within_tolerance(self):
        depth, thickness = 1.5, 0.4
        mesh = build_curved_frame(rim_polygons_cm(), thickness, depth)
        # Test face centres: they are where chords stray from the curve.
        centers = mesh.triangles_center
        caps = np.abs(mesh.face_normals[:, 2]) > 0.5
        expected = wrap_offset(centers[caps, 0], WIDTH, depth)
        offsets = centers[caps, 2] - expected
        front = offsets > -thickness / 2
        self.assertLess(np.abs(offsets[front]).max(), CHORD_TOLERANCE_CM)
        self.assertLess(np.abs(offsets[~front] + thickness).max(), CHORD_TOLERANCE_CM)

    def test_centre_of_the_front_surface_is_the_most_forward_point(self):
        mesh = build_curved_frame(rim_polygons_cm(), 0.4, 1.0)
        self.assertAlmostEqual(mesh.bounds[1][2], wrap_offset(mesh.vertices[np.argmax(
            mesh.vertices[:, 2]), 0], WIDTH, 1.0))
        self.assertLessEqual(mesh.bounds[1][2], 0.0)

    def test_outer_edge_sits_back_by_the_curvature(self):
        mesh = build_curved_frame(rim_polygons_cm(), 0.4, 1.0)
        outer = np.abs(mesh.vertices[:, 0]) > 7.0 - 1e-6
        self.assertAlmostEqual(mesh.vertices[outer, 2].max(), -1.0, places=6)
        self.assertAlmostEqual(mesh.vertices[outer, 2].min(), -1.4, places=6)

    def test_thickness_is_preserved_along_z(self):
        mesh = build_curved_frame(rim_polygons_cm(), 0.7, 0.0)
        self.assertAlmostEqual(mesh.extents[2], 0.7)

    def test_density_grows_only_as_needed(self):
        flat = len(build_curved_frame(rim_polygons_cm(), 0.4, 0.0).faces)
        gentle = len(build_curved_frame(rim_polygons_cm(), 0.4, 0.5).faces)
        strong = len(build_curved_frame(rim_polygons_cm(), 0.4, 3.0).faces)
        self.assertLess(flat, gentle)
        self.assertLess(gentle, strong)
        self.assertLess(gentle, 2000)  # a few simple rims at the default curvature

    def test_no_degenerate_triangles(self):
        for depth in (0.0, 0.5, 3.0):
            with self.subTest(depth=depth):
                mesh = build_curved_frame(rim_polygons_cm(), 0.4, depth)
                self.assertGreater(mesh.area_faces.min(), 1e-8)

    def test_outline_with_collinear_points_makes_no_degenerate_triangles(self):
        outer = [(-7, -2), (-3, -2), (0, -2), (3, -2), (7, -2), (7, 2), (0, 2), (-7, 2)]
        hole = [(-5, -1), (-2, -1), (-2, 1), (-3.5, 1), (-5, 1)]
        mesh = build_curved_frame([Polygon(outer, [hole])], 0.4, 0.5)
        self.assertTrue(mesh.is_volume)
        self.assertGreater(mesh.area_faces.min(), 1e-8)

    @unittest.skipUnless(PHOTO.exists(), "image_to_3d/glasses.jpg is not available")
    def test_real_outline_with_collinear_points_makes_no_degenerate_triangles(self):
        # Earcut turns this traced outline into zero-area faces.
        polygons = scale_polygons_to_cm(mask_to_polygons(segment_glasses(load_image(PHOTO))), WIDTH)
        for depth in (0.0, 0.5):
            with self.subTest(depth=depth):
                mesh = build_curved_frame(polygons, 0.4, depth)
                self.assertGreater(mesh.area_faces.min(), 1e-9)

    def test_flat_case_matches_the_existing_flat_model(self):
        mask = segment_glasses(white_image())
        legacy = polygons_to_mesh(mask_to_polygons(mask), FrameDimensions(WIDTH, 0.4))
        flat = build_curved_frame(rim_polygons_cm(), 0.4, 0.0)
        np.testing.assert_allclose(flat.bounds, legacy.bounds, atol=1e-9)
        self.assertAlmostEqual(flat.volume, legacy.volume, places=6)

    def test_disconnected_rims_stay_valid(self):
        image = np.full((200, 400, 3), 255, np.uint8)
        cv2.rectangle(image, *LEFT_RIM, FRAME_COLOR, RIM_THICKNESS)
        cv2.rectangle(image, *RIGHT_RIM, FRAME_COLOR, RIM_THICKNESS)
        polygons = scale_polygons_to_cm(mask_to_polygons(segment_glasses(image)), WIDTH)
        mesh = build_curved_frame(polygons, 0.4, 1.0)
        self.assertTrue(mesh.is_volume)
        self.assertEqual(len(mesh.split(only_watertight=True)), 2)


class OpeningIntrusionTests(unittest.TestCase):
    def test_thin_strip_inside_an_opening_is_removed(self):
        mask = segment_glasses(rims_with_temple_ghost())
        self.assertEqual(mask[90, 95], 255)  # on the strip
        cleaned = remove_opening_intrusions(mask, 0.04)
        self.assertEqual(cleaned[90, 95], 0)

    def test_rims_and_bridge_are_untouched(self):
        mask = segment_glasses(white_image())
        cleaned = remove_opening_intrusions(mask, 0.04)
        np.testing.assert_array_equal(cleaned, mask)

    def test_thin_rim_survives_even_when_thinner_than_the_kernel(self):
        mask = np.zeros((200, 400), np.uint8)
        cv2.rectangle(mask, (40, 50), (170, 150), 255, 3)
        cv2.rectangle(mask, (230, 50), (360, 150), 255, 3)
        cv2.line(mask, (170, 80), (230, 80), 255, 3)
        cleaned = remove_opening_intrusions(mask, 0.1)
        np.testing.assert_array_equal(cleaned, mask)

    def test_wide_intrusion_is_kept(self):
        mask = segment_glasses(white_image())
        mask[50:110, 90:120] = 255  # 30 px wide, wider than a 4% (13 px) kernel
        cleaned = remove_opening_intrusions(mask, 0.04)
        self.assertEqual(cleaned[90, 105], 255)

    def test_zero_fraction_disables_the_cleanup(self):
        mask = segment_glasses(rims_with_temple_ghost())
        np.testing.assert_array_equal(remove_opening_intrusions(mask, 0.0), mask)

    def test_cleanup_does_not_add_foreground(self):
        mask = segment_glasses(rims_with_temple_ghost())
        cleaned = remove_opening_intrusions(mask, 0.04)
        self.assertFalse(np.any((cleaned > 0) & (mask == 0)))

    def test_invalid_fraction_is_rejected(self):
        mask = segment_glasses(white_image())
        for bad in (-0.1, 0.5, float("nan")):
            with self.assertRaises(ValueError):
                remove_opening_intrusions(mask, bad)


class AdaptiveWhiteThresholdTests(unittest.TestCase):
    def shadowed_image(self):
        image = white_image()
        image[152:160, 30:180] = (232, 232, 232)  # faint drop shadow under the rim
        return image

    def test_fixed_threshold_keeps_the_faint_shadow(self):
        self.assertEqual(segment_glasses(self.shadowed_image())[156, 100], 255)

    def test_adaptive_threshold_drops_the_faint_shadow(self):
        mask = segment_glasses(self.shadowed_image(), SegmentationConfig(adaptive_white=True))
        self.assertEqual(mask[156, 100], 0)
        self.assertEqual(mask[100, 40], 255)

    def test_adaptive_threshold_keeps_a_pale_frame(self):
        image = np.full((200, 400, 3), 255, np.uint8)
        cv2.rectangle(image, *LEFT_RIM, (205, 205, 205), RIM_THICKNESS)
        cv2.rectangle(image, *RIGHT_RIM, (205, 205, 205), RIM_THICKNESS)
        mask = segment_glasses(image, SegmentationConfig(adaptive_white=True))
        self.assertEqual(mask[100, 40], 255)

    def test_adaptive_threshold_never_exceeds_the_configured_one(self):
        # A very pale frame at 238: the midpoint with white would be 246.
        image = np.full((200, 400, 3), 255, np.uint8)
        cv2.rectangle(image, *LEFT_RIM, (238, 238, 238), RIM_THICKNESS)
        image[0:4, 0:4] = 244  # below white, above the configured threshold
        mask = segment_glasses(image, SegmentationConfig(adaptive_white=True))
        self.assertEqual(mask[1, 1], 0)


if __name__ == "__main__":
    unittest.main()
