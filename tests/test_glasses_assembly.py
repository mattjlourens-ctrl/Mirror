from pathlib import Path
import contextlib
import hashlib
import io
import itertools
import os
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
from shapely.geometry import Point, Polygon
import trimesh

from accessory_transform import attachment_matrix, transform_points
from glasses_3d import (DEFAULT_GLASSES_ATTACHMENT, GlassesDimensions, WireframeModel,
                        projected_segments)
from head_pose import FACE_ORIGIN, FRONTAL_ROTATION
from image_to_3d.assembly import (BRIDGE, FRONT_FRAME, LEFT_HINGE, LEFT_NOSE_PAD, LEFT_TEMPLE,
                                  RIGHT_HINGE, RIGHT_NOSE_PAD, RIGHT_TEMPLE, GlassesParameters,
                                  assemble_glasses, export_model_glb)
from image_to_3d.curvature import wrap_offset
from image_to_3d.errors import MeshBuildError
from image_to_3d.hinges import (HingeParameters, build_fallback_bridge, build_hinge,
                                build_nose_pad, estimate_hinge_anchor, estimate_nose_pad_anchor)
from image_to_3d.mesh import FrameDimensions, mask_to_polygons, polygons_to_mesh, scale_polygons_to_cm
from image_to_3d.mirror_image_to_3d import main
from image_to_3d.pipeline import convert_image_to_glasses
from image_to_3d.segmentation import segment_glasses
from image_to_3d.temples import TempleParameters
from image_to_3d.validation import (inspect_component, meshes_intersect, self_intersects,
                                    validate_components)
from occlusion import EAR_PLANE_WIDTH, EAR_PLANE_Z, FACE_OVAL, load_canonical_face_mesh
from tests.test_head_pose import make_pose
from tests.test_image_to_3d import (FRAME_COLOR, LEFT_OPENING_CENTER, LEFT_RIM, RIGHT_OPENING_CENTER,
                                    RIGHT_RIM, RIM_THICKNESS, REPO_ROOT, covers_xy,
                                    pixel_to_mesh_xy, white_image)

WIDTH = 14.0
PHOTO = REPO_ROOT / "image_to_3d" / "glasses.jpg"


def rim_pixels():
    return mask_to_polygons(segment_glasses(white_image()))


def rim_polygons_cm(width=WIDTH):
    return scale_polygons_to_cm(rim_pixels(), width)


def disconnected_pixels():
    image = np.full((200, 400, 3), 255, np.uint8)
    cv2.rectangle(image, *LEFT_RIM, FRAME_COLOR, RIM_THICKNESS)
    cv2.rectangle(image, *RIGHT_RIM, FRAME_COLOR, RIM_THICKNESS)
    return mask_to_polygons(segment_glasses(image))


def uniform_temples(**overrides):
    params = TempleParameters(**overrides)
    return dict(left_temple=params, right_temple=params)


class HingeAnchorTests(unittest.TestCase):
    def test_anchors_sit_on_the_matching_outer_edge(self):
        polygons = rim_polygons_cm()
        right = estimate_hinge_anchor(polygons, 1)
        left = estimate_hinge_anchor(polygons, -1)
        self.assertGreater(right[0], 6.0)
        self.assertLess(left[0], -6.0)

    def test_anchor_is_on_frame_material(self):
        polygons = rim_polygons_cm()
        for side in (-1, 1):
            anchor = estimate_hinge_anchor(polygons, side)
            self.assertLess(polygons[0].distance(Point(anchor)), 1e-6)

    def test_symmetric_frame_gives_symmetric_anchors(self):
        polygons = rim_polygons_cm()
        np.testing.assert_allclose(estimate_hinge_anchor(polygons, 1) * np.array((-1, 1)),
                                   estimate_hinge_anchor(polygons, -1), atol=0.01)

    def test_anchor_height_follows_the_outline(self):
        # An L-shaped end: only the top 1 cm reaches the outermost edge.
        outer = [(-7, -2), (6, -2), (6, 1), (7, 1), (7, 2), (-7, 2)]
        anchor = estimate_hinge_anchor([Polygon(outer)], 1)
        self.assertAlmostEqual(anchor[1], 1.5)

    def test_anchor_stays_on_material_when_the_end_is_narrow(self):
        # A pointed end: straight in from the tip is the only material.
        outer = [(-7, -2), (5, -2), (7, 0), (5, 2), (-7, 2)]
        polygon = Polygon(outer)
        anchor = estimate_hinge_anchor([polygon], 1)
        self.assertLess(polygon.distance(Point(anchor)), 1e-6)

    def test_anchor_moves_onto_material_when_mid_height_is_a_gap(self):
        # A forked end: two prongs reach the outer edge with a gap between them.
        outer = [(-7, -2), (7, -2), (7, -1), (5, -1), (5, 1), (7, 1), (7, 2), (-7, 2)]
        polygon = Polygon(outer)
        self.assertFalse(polygon.contains(Point(6.8, 0.0)))
        anchor = estimate_hinge_anchor([polygon], 1)
        self.assertLess(polygon.distance(Point(anchor)), 1e-6)

    def test_each_side_uses_its_own_part_when_rims_are_separate(self):
        polygons = scale_polygons_to_cm(disconnected_pixels(), WIDTH)
        self.assertGreater(estimate_hinge_anchor(polygons, 1)[0], 0)
        self.assertLess(estimate_hinge_anchor(polygons, -1)[0], 0)


class HingeGeometryTests(unittest.TestCase):
    def hinge(self, side=1, curvature=0.5, x=6.8, y=0.5):
        return build_hinge(np.array((side * x, y)), side, WIDTH, 0.4, curvature)

    def test_pivot_is_behind_the_rear_surface(self):
        for curvature in (0.0, 0.5, 3.0):
            with self.subTest(curvature=curvature):
                hinge = self.hinge(curvature=curvature)
                rear = wrap_offset(WIDTH / 2, WIDTH, curvature) - 0.4
                self.assertLess(hinge.pivot[2], rear)

    def test_pivot_keeps_the_anchor_height(self):
        self.assertAlmostEqual(self.hinge(y=0.7).pivot[1], 0.7)

    def test_barrel_axis_is_vertical(self):
        hinge = self.hinge()
        self.assertAlmostEqual(hinge.barrel.extents[1], HingeParameters().barrel_height, places=6)
        self.assertAlmostEqual(hinge.barrel.extents[0], 2 * HingeParameters().barrel_radius,
                               places=2)
        np.testing.assert_allclose(hinge.barrel.bounds.mean(axis=0), hinge.pivot, atol=1e-6)

    def test_block_runs_along_the_frame_normal(self):
        # z = -a x^2 has the forward normal (2 a x, 0, 1); here a = 1/49, x = 6.8.
        hinge = self.hinge(curvature=1.0)
        expected = -np.array((2 * 6.8 / 49, 0, 1.0))
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(hinge.block_axis, expected, atol=1e-9)

    def test_flat_frame_block_runs_straight_back(self):
        np.testing.assert_allclose(self.hinge(curvature=0.0).block_axis, (0, 0, -1), atol=1e-9)

    def test_block_stays_upright(self):
        self.assertAlmostEqual(self.hinge(curvature=3.0).block.extents[1],
                               HingeParameters().block_height, places=6)

    def test_block_reaches_into_the_frame_and_to_the_pivot(self):
        hinge = self.hinge()
        rear = wrap_offset(6.8, WIDTH, 0.5) - 0.4
        self.assertGreater(hinge.block.bounds[1][2], rear)
        self.assertLessEqual(hinge.block.bounds[0][2], hinge.pivot[2] + 1e-6)

    def test_sides_mirror_each_other(self):
        right, left = self.hinge(side=1), self.hinge(side=-1)
        np.testing.assert_allclose(left.pivot, right.pivot * (-1, 1, 1))

    def test_parts_are_closed_with_outward_normals(self):
        hinge = self.hinge(curvature=2.0)
        for part in (hinge.barrel, hinge.block):
            self.assertTrue(part.is_volume)
            self.assertGreater(part.volume, 0)

    def test_invalid_hinge_parameters_are_rejected(self):
        for bad in (dict(barrel_radius=0), dict(barrel_height=-1), dict(block_width=float("nan")),
                    dict(block_height=0), dict(anchor_inset=-0.1)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    HingeParameters(**bad).validate()


class NoseSupportTests(unittest.TestCase):
    def test_anchors_are_on_the_nasal_side_of_each_opening(self):
        polygons = rim_polygons_cm()
        opening_x = {1: [], -1: []}
        for ring in polygons[0].interiors:
            xs = np.array(ring.coords)[:, 0]
            opening_x[1 if xs.mean() > 0 else -1] = xs
        for side in (-1, 1):
            anchor = estimate_nose_pad_anchor(polygons, side)
            self.assertLess(abs(anchor[0]), np.abs(opening_x[side]).min() + 1e-6)
            self.assertGreater(anchor[0] * side, 0)
            self.assertLess(polygons[0].distance(Point(anchor)), 1e-6)

    def test_anchors_sit_below_the_opening_centre(self):
        polygons = rim_polygons_cm()
        for ring in polygons[0].interiors:
            center_y = np.array(ring.coords)[:, 1].mean()
            side = 1 if np.array(ring.coords)[:, 0].mean() > 0 else -1
            self.assertLess(estimate_nose_pad_anchor(polygons, side)[1], center_y)

    def test_frame_without_openings_still_gets_anchors_on_material(self):
        polygon = Polygon([(-7, -2), (7, -2), (7, 2), (-7, 2)])
        for side in (-1, 1):
            anchor = estimate_nose_pad_anchor([polygon], side)
            self.assertGreater(anchor[0] * side, 0)
            self.assertLess(polygon.distance(Point(anchor)), 1e-6)

    def test_pad_sits_behind_the_frame_and_towards_the_nose(self):
        anchor = np.array((1.5, -0.5))
        pad = build_nose_pad(anchor, 1, WIDTH, 0.4, 0.5)
        rear = wrap_offset(1.5, WIDTH, 0.5) - 0.4
        self.assertLess(pad.bounds[0][2], rear)
        self.assertLess(pad.bounds[:, 0].mean(), anchor[0] - 0.1)

    def test_pads_mirror_each_other(self):
        right = build_nose_pad(np.array((1.5, -0.5)), 1, WIDTH, 0.4, 0.5)
        left = build_nose_pad(np.array((-1.5, -0.5)), -1, WIDTH, 0.4, 0.5)
        np.testing.assert_allclose(left.bounds[:, 0][::-1], -right.bounds[:, 0], atol=1e-9)
        np.testing.assert_allclose(left.bounds[:, 1:], right.bounds[:, 1:], atol=1e-9)

    def test_bridge_is_not_generated_when_the_photo_has_one(self):
        self.assertIsNone(build_fallback_bridge(rim_polygons_cm(), 0.4, 0.5))

    def test_bridge_is_not_generated_for_stray_parts_beside_a_central_frame(self):
        def square(x):
            return Polygon([(x - 0.5, -0.5), (x + 0.5, -0.5), (x + 0.5, 0.5), (x - 0.5, 0.5)])

        central = Polygon([(-4, -2), (4, -2), (4, 2), (-4, 2)])
        self.assertIsNone(build_fallback_bridge([square(-6.5), central, square(6.5)], 0.4, 0.5))

    def test_bridge_spans_the_gap_between_separate_rims(self):
        polygons = scale_polygons_to_cm(disconnected_pixels(), WIDTH)
        bridge = build_fallback_bridge(polygons, 0.4, 0.5)
        left_edge = polygons[0].bounds[2]
        right_edge = polygons[1].bounds[0]
        self.assertTrue(bridge.is_volume)
        self.assertLess(bridge.bounds[0][0], left_edge)
        self.assertGreater(bridge.bounds[1][0], right_edge)

    def test_bridge_follows_the_frame_curve(self):
        polygons = scale_polygons_to_cm(disconnected_pixels(), WIDTH)
        flat = build_fallback_bridge(polygons, 0.4, 0.0)
        curved = build_fallback_bridge(polygons, 0.4, 2.0)
        self.assertLess(curved.vertices[:, 2].min(), flat.vertices[:, 2].min())


class ValidationTests(unittest.TestCase):
    def test_overlapping_boxes_intersect(self):
        a = trimesh.creation.box((1, 1, 1))
        b = trimesh.creation.box((1, 1, 1))
        b.apply_translation((0.5, 0.2, 0.1))
        self.assertTrue(meshes_intersect(a, b))

    def test_separate_boxes_do_not_intersect(self):
        a = trimesh.creation.box((1, 1, 1))
        b = trimesh.creation.box((1, 1, 1))
        b.apply_translation((3, 0, 0))
        self.assertFalse(meshes_intersect(a, b))

    def test_box_poking_through_a_face_without_vertices_inside_intersects(self):
        plate = trimesh.creation.box((4, 4, 0.1))
        pin = trimesh.creation.box((0.2, 0.2, 3))
        self.assertTrue(meshes_intersect(plate, pin))

    def test_box_entirely_inside_another_is_not_a_surface_intersection(self):
        outer = trimesh.creation.box((4, 4, 4))
        inner = trimesh.creation.box((1, 1, 1))
        self.assertFalse(meshes_intersect(outer, inner))

    def test_clean_box_has_no_self_intersection(self):
        self.assertFalse(self_intersects(trimesh.creation.box((1, 1, 1))))

    def test_two_crossing_shells_self_intersect(self):
        a = trimesh.creation.box((2, 0.2, 0.2))
        b = trimesh.creation.box((0.2, 2, 0.2))
        self.assertTrue(self_intersects(trimesh.util.concatenate([a, b])))

    def test_clean_box_reports_no_problems(self):
        self.assertEqual(inspect_component("box", trimesh.creation.box((1, 1, 1))).problems, [])

    def test_degenerate_triangle_is_reported(self):
        mesh = trimesh.Trimesh([(0, 0, 0), (1, 0, 0), (2, 0, 0)], [(0, 1, 2)], process=False)
        report = inspect_component("bad", mesh)
        self.assertEqual(report.degenerate_faces, 1)
        self.assertTrue(any("degenerate" in problem for problem in report.problems))

    def test_open_mesh_is_reported_as_non_manifold(self):
        box = trimesh.creation.box((1, 1, 1))
        holed = trimesh.Trimesh(box.vertices, box.faces[1:], process=False)
        self.assertGreater(inspect_component("holed", holed).non_manifold_edges, 0)

    def test_edge_shared_by_three_faces_is_reported(self):
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
        mesh = trimesh.Trimesh(vertices, [(0, 1, 2), (1, 0, 3), (0, 1, 4)], process=False)
        self.assertGreater(inspect_component("fin", mesh).non_manifold_edges, 0)

    def test_inside_out_mesh_is_reported(self):
        box = trimesh.creation.box((1, 1, 1))
        box.invert()
        report = inspect_component("inverted", box)
        self.assertTrue(any("inward" in problem for problem in report.problems))

    def test_inconsistent_winding_is_reported(self):
        box = trimesh.creation.box((1, 1, 1))
        faces = box.faces.copy()
        faces[0] = faces[0][::-1]
        report = inspect_component("flip", trimesh.Trimesh(box.vertices, faces, process=False))
        self.assertTrue(any("winding" in problem for problem in report.problems))

    def test_non_finite_vertices_are_reported(self):
        box = trimesh.creation.box((1, 1, 1))
        vertices = box.vertices.copy()
        vertices[0, 0] = np.nan
        report = inspect_component("nan", trimesh.Trimesh(vertices, box.faces, process=False))
        self.assertTrue(any("non-finite" in problem for problem in report.problems))

    def test_unintended_intersection_is_reported(self):
        a = trimesh.creation.box((1, 1, 1))
        b = a.copy()
        b.apply_translation((0.5, 0.2, 0.1))
        report = validate_components({"a": a, "b": b}, contacts=[])
        self.assertEqual(report.unintended_intersections, [("a", "b")])
        self.assertFalse(report.ok)

    def test_declared_contact_may_intersect(self):
        a = trimesh.creation.box((1, 1, 1))
        b = a.copy()
        b.apply_translation((0.5, 0.2, 0.1))
        self.assertTrue(validate_components({"a": a, "b": b}, contacts=[("a", "b")]).ok)

    def test_declared_contact_that_misses_is_reported(self):
        a = trimesh.creation.box((1, 1, 1))
        b = a.copy()
        b.apply_translation((5, 0, 0))
        report = validate_components({"a": a, "b": b}, contacts=[("a", "b")])
        self.assertEqual(report.missing_contacts, [("a", "b")])
        self.assertFalse(report.ok)


    def test_self_intersection_is_reported(self):
        a = trimesh.creation.box((2, 0.2, 0.2))
        b = trimesh.creation.box((0.2, 2, 0.2))
        crossed = trimesh.util.concatenate([a, b])
        report = validate_components({"crossed": crossed}, contacts=[], check_self=["crossed"])
        self.assertEqual(report.self_intersecting, ["crossed"])
        self.assertFalse(report.ok)


class AssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = assemble_glasses(rim_pixels())

    def test_contains_the_required_components(self):
        self.assertEqual(set(self.model.components),
                         {FRONT_FRAME, LEFT_TEMPLE, RIGHT_TEMPLE, LEFT_HINGE, RIGHT_HINGE})

    def test_nose_pads_are_off_by_default_and_optional(self):
        self.assertNotIn(LEFT_NOSE_PAD, self.model.components)
        with_pads = assemble_glasses(rim_pixels(), GlassesParameters(nose_pads=True))
        self.assertIn(LEFT_NOSE_PAD, with_pads.components)
        self.assertIn(RIGHT_NOSE_PAD, with_pads.components)

    def test_validation_passes(self):
        self.assertEqual(self.model.report.problems, [])

    def test_all_components_share_one_coordinate_system(self):
        front = self.model.components[FRONT_FRAME]
        np.testing.assert_allclose(front.bounds[:, :2].mean(axis=0), 0, atol=1e-6)
        self.assertLessEqual(front.bounds[1][2], 1e-9)
        for name in (LEFT_TEMPLE, RIGHT_TEMPLE):
            self.assertLess(self.model.components[name].bounds[0][2], -10)

    def test_frame_width_is_exact_and_excludes_the_temples(self):
        model = assemble_glasses(rim_pixels(), GlassesParameters(frame_width=12.0))
        self.assertAlmostEqual(model.components[FRONT_FRAME].extents[0], 12.0, places=6)

    def test_left_temple_is_on_the_left_and_right_on_the_right(self):
        self.assertLess(self.model.components[LEFT_TEMPLE].bounds[1][0], 0)
        self.assertGreater(self.model.components[RIGHT_TEMPLE].bounds[0][0], 0)

    def test_temples_are_approximately_symmetric(self):
        left = self.model.components[LEFT_TEMPLE]
        right = self.model.components[RIGHT_TEMPLE]
        np.testing.assert_allclose(left.bounds[:, 0][::-1], -right.bounds[:, 0], atol=0.05)
        np.testing.assert_allclose(left.bounds[:, 1:], right.bounds[:, 1:], atol=0.05)

    def test_sides_can_be_adjusted_separately(self):
        params = GlassesParameters(left_temple=TempleParameters(length=11.0),
                                   right_temple=TempleParameters(length=15.0, curvature=1.0))
        model = assemble_glasses(rim_pixels(), params)
        self.assertGreater(model.components[LEFT_TEMPLE].bounds[0][2],
                           model.components[RIGHT_TEMPLE].bounds[0][2] + 3)

    def test_temples_start_at_the_hinge_pivots(self):
        for name, pivot in ((LEFT_TEMPLE, self.model.left_pivot),
                            (RIGHT_TEMPLE, self.model.right_pivot)):
            temple = self.model.components[name]
            distance = np.linalg.norm(temple.vertices - pivot, axis=1).min()
            self.assertLess(distance, 0.5)

    def test_curved_frame_moves_hinges_back(self):
        flat = assemble_glasses(rim_pixels(), GlassesParameters(frame_curvature=0.0))
        curved = assemble_glasses(rim_pixels(), GlassesParameters(frame_curvature=1.5))
        self.assertLess(curved.right_pivot[2], flat.right_pivot[2] - 1.0)

    def test_opening_stays_open_with_everything_assembled(self):
        mask = segment_glasses(white_image())
        front = self.model.components[FRONT_FRAME]
        for pixel in (LEFT_OPENING_CENTER, RIGHT_OPENING_CENTER):
            self.assertFalse(covers_xy(front, pixel_to_mesh_xy(pixel, mask, WIDTH)))
        self.assertEqual(self.model.openings, 2)

    def test_photo_bridge_reported_and_no_extra_bridge_added(self):
        self.assertNotIn(BRIDGE, self.model.components)
        self.assertEqual(self.model.nose_support, "bridge from the photo")

    def test_separate_rims_get_a_generated_bridge_that_touches_both(self):
        model = assemble_glasses(disconnected_pixels())
        self.assertIn(BRIDGE, model.components)
        self.assertEqual(model.nose_support, "generated bridge")
        self.assertEqual(model.report.problems, [])

    def test_parameter_combinations_all_validate(self):
        arms = (dict(length=10.0, curvature=0.0, ear_drop=0.0),
                dict(length=16.0, curvature=1.2, ear_drop=2.0))
        for width, thickness, curvature, arm, pads in itertools.product(
                (11.0, 16.0), (0.2, 0.6), (0.0, 2.5), arms, (False, True)):
            params = GlassesParameters(frame_width=width, frame_thickness=thickness,
                                       frame_curvature=curvature, nose_pads=pads,
                                       **uniform_temples(**arm))
            with self.subTest(params=params):
                self.assertEqual(assemble_glasses(rim_pixels(), params).report.problems, [])

    def test_temples_that_collide_are_refused(self):
        temples = uniform_temples(splay=0.0, curvature=7.0, ear_drop=0.0)
        with self.assertRaisesRegex(MeshBuildError, "intersect"):
            assemble_glasses(rim_pixels(), GlassesParameters(**temples))

    def test_empty_polygons_are_refused(self):
        with self.assertRaises(MeshBuildError):
            assemble_glasses([])


class ParameterValidationTests(unittest.TestCase):
    def test_bad_frame_values_are_rejected(self):
        for bad in (dict(frame_width=0), dict(frame_width=-1), dict(frame_width=float("inf")),
                    dict(frame_thickness=0), dict(frame_thickness=float("nan")),
                    dict(frame_curvature=-0.1), dict(frame_curvature=float("nan")),
                    dict(frame_curvature=4.0)):  # over a quarter of the 14 cm width
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    GlassesParameters(**bad).validate()

    def test_each_temple_is_validated(self):
        for side in ("left_temple", "right_temple"):
            with self.assertRaises(ValueError):
                GlassesParameters(**{side: TempleParameters(length=-1)}).validate()

    def test_assembly_validates_before_building(self):
        with self.assertRaises(ValueError):
            assemble_glasses(rim_pixels(), GlassesParameters(frame_width=-1))


class ExportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)

    def export(self, params=GlassesParameters()):
        return export_model_glb(assemble_glasses(rim_pixels(), params), self.dir / "model.glb")

    def test_reloads_with_named_separable_components(self):
        params = GlassesParameters(nose_pads=True)
        scene = trimesh.load_scene(self.export(params))
        self.assertEqual(set(scene.geometry),
                         {FRONT_FRAME, LEFT_TEMPLE, RIGHT_TEMPLE, LEFT_HINGE, RIGHT_HINGE,
                          LEFT_NOSE_PAD, RIGHT_NOSE_PAD})

    def test_reloaded_dimensions_are_in_metres(self):
        scene = trimesh.load_scene(self.export())
        self.assertAlmostEqual(scene.geometry[FRONT_FRAME].extents[0], 0.14, places=6)
        length = np.ptp(scene.geometry[LEFT_TEMPLE].vertices[:, 2])
        self.assertGreater(length, 0.12)
        self.assertLess(length, 0.15)

    def test_reloaded_components_keep_valid_topology(self):
        scene = trimesh.load_scene(self.export())
        for name in (FRONT_FRAME, LEFT_TEMPLE, RIGHT_TEMPLE):
            self.assertTrue(scene.geometry[name].is_volume, name)

    def test_export_does_not_touch_the_model(self):
        model = assemble_glasses(rim_pixels())
        before = model.components[FRONT_FRAME].extents.copy()
        export_model_glb(model, self.dir / "model.glb")
        np.testing.assert_array_equal(model.components[FRONT_FRAME].extents, before)

    def test_output_must_be_named_glb(self):
        with self.assertRaisesRegex(ValueError, r"\.glb"):
            export_model_glb(assemble_glasses(rim_pixels()), self.dir / "model.obj")


class DeterminismTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)

    def test_same_input_gives_identical_bytes_in_one_process(self):
        params = GlassesParameters(frame_curvature=0.8, nose_pads=True)
        a = export_model_glb(assemble_glasses(rim_pixels(), params), self.dir / "a.glb")
        b = export_model_glb(assemble_glasses(rim_pixels(), params), self.dir / "b.glb")
        self.assertEqual(a.read_bytes(), b.read_bytes())

    def test_same_input_gives_identical_bytes_across_processes(self):
        digests = set()
        for index, seed in enumerate(("1", "4242")):
            out = self.dir / f"{index}.glb"
            done = subprocess.run(
                [sys.executable, "image_to_3d/mirror_image_to_3d.py", "--demo", "--complete",
                 "--nose-pads", "-o", str(out)],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
                env={**os.environ, "PYTHONHASHSEED": seed})
            self.assertEqual(done.returncode, 0, done.stderr)
            digests.add(hashlib.sha256(out.read_bytes()).hexdigest())
        self.assertEqual(len(digests), 1)

    def test_changing_a_parameter_changes_the_output(self):
        a = export_model_glb(assemble_glasses(rim_pixels()), self.dir / "a.glb")
        b = export_model_glb(assemble_glasses(rim_pixels(), GlassesParameters(frame_curvature=1.0)),
                             self.dir / "b.glb")
        self.assertNotEqual(a.read_bytes(), b.read_bytes())


class MirrorCompatibilityTests(unittest.TestCase):
    """Run the GLB's geometry through Mirror's real transforms, unmodified."""

    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        path = export_model_glb(assemble_glasses(rim_pixels()), Path(directory.name) / "m.glb")
        cls.scene = trimesh.load_scene(path)

    def to_face(self, name):
        matrix = attachment_matrix(DEFAULT_GLASSES_ATTACHMENT, units_to_cm=100.0)
        return transform_points(matrix, np.asarray(self.scene.geometry[name].vertices))

    def test_frame_lands_on_the_nose_bridge_anchor(self):
        points = self.to_face(FRONT_FRAME)
        expected = FACE_ORIGIN + DEFAULT_GLASSES_ATTACHMENT.offset_cm
        np.testing.assert_allclose((points.min(axis=0) + points.max(axis=0))[:2] / 2,
                                   expected[:2], atol=1e-5)
        self.assertAlmostEqual(points[:, 2].max(), expected[2], places=4)

    def test_temples_reach_back_past_the_ears(self):
        self.assertLess(self.to_face(RIGHT_TEMPLE)[:, 2].min(), EAR_PLANE_Z)

    def test_default_temples_clear_the_head_proxy_at_the_ear(self):
        vertices, _ = load_canonical_face_mesh()
        head_half_width = np.abs(vertices[FACE_OVAL, 0]).max() * EAR_PLANE_WIDTH
        points = self.to_face(RIGHT_TEMPLE)
        near_ear = np.abs(points[:, 2] - EAR_PLANE_Z) < 1.0
        self.assertGreater(points[near_ear, 0].min(), head_half_width)

    def test_defaults_agree_with_the_wireframe_model(self):
        dims = GlassesDimensions()
        self.assertEqual(GlassesParameters().frame_width, dims.frame_width)
        self.assertEqual(GlassesParameters().frame_curvature, dims.wrap_depth)
        self.assertEqual(TempleParameters().length, dims.temple_length)
        self.assertEqual(TempleParameters().ear_drop, dims.ear_drop)
        self.assertEqual(TempleParameters().splay, dims.temple_spread)

    def test_frontal_pose_puts_the_right_temple_on_screen_right(self):
        pose = make_pose(FRONTAL_ROTATION, (0.0, 0.0, 40.0))
        attach = attachment_matrix(DEFAULT_GLASSES_ATTACHMENT, units_to_cm=100.0)
        vertices = np.asarray(self.scene.geometry[RIGHT_TEMPLE].vertices)
        camera = transform_points(pose.transform @ attach, vertices)
        self.assertTrue(np.all(camera[:, 0] > 0))
        # Further from the camera towards the ear: Z in camera space grows.
        self.assertGreater(camera[:, 2].max(), 40.0)

    def test_existing_wireframe_projection_accepts_the_mesh(self):
        mesh = self.scene.geometry[RIGHT_TEMPLE]
        model = WireframeModel(name="from glb", vertices=np.asarray(mesh.vertices),
                               edges=np.asarray(mesh.edges_unique), units_to_cm=100.0)
        pose = make_pose(FRONTAL_ROTATION, (0.0, 0.0, 40.0))
        attach = attachment_matrix(DEFAULT_GLASSES_ATTACHMENT, units_to_cm=model.units_to_cm)
        self.assertGreater(len(projected_segments(model, attach, pose)), 0)


class SourceImageTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        self.source = self.dir / "a.png"
        self.source.write_bytes(cv2.imencode(".png", white_image())[1].tobytes())

    def run_main(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = main([str(arg) for arg in argv])
            except SystemExit as stop:
                code = stop.code
        return code, stdout.getvalue(), stderr.getvalue()


class FlatRegressionTests(SourceImageTestCase):
    """The original shallow model must not change."""

    def test_flat_mesh_is_still_one_volume_with_the_same_shape(self):
        mesh = polygons_to_mesh(rim_pixels(), FrameDimensions(14.0, 0.4))
        self.assertTrue(mesh.is_volume)
        self.assertAlmostEqual(mesh.extents[0], 14.0)
        self.assertAlmostEqual(mesh.extents[2], 0.4)
        self.assertEqual(mesh.bounds[1][2], 0.0)

    def test_default_command_still_writes_the_flat_mesh_alone(self):
        out = self.dir / "flat.glb"
        code, _, stderr = self.run_main(self.source, "-o", out)
        self.assertEqual(code, 0, stderr)
        scene = trimesh.load_scene(out)
        self.assertEqual(len(scene.geometry), 1)
        self.assertAlmostEqual(trimesh.load_mesh(out).extents[2], 0.004, places=6)


class CommandLineTests(SourceImageTestCase):
    def complete(self, *argv):
        out = self.dir / "complete.glb"
        code, stdout, stderr = self.run_main(self.source, "--complete", "-o", out, *argv)
        self.assertEqual(code, 0, stderr)
        return trimesh.load_scene(out), stdout

    def test_complete_writes_all_components(self):
        scene, stdout = self.complete()
        self.assertIn(LEFT_TEMPLE, scene.geometry)
        self.assertIn("left_temple", stdout)
        self.assertIn("openings: 2", stdout)

    def test_frame_flags_apply(self):
        scene, _ = self.complete("--frame-width", 12, "--frame-thickness", 0.5,
                                 "--frame-curvature", 0.0)
        front = scene.geometry[FRONT_FRAME]
        self.assertAlmostEqual(front.extents[0], 0.12, places=6)
        self.assertAlmostEqual(front.extents[2], 0.005, places=6)

    def test_old_thickness_flag_still_works_in_complete_mode(self):
        scene, _ = self.complete("--thickness", 0.5, "--frame-curvature", 0.0)
        self.assertAlmostEqual(scene.geometry[FRONT_FRAME].extents[2], 0.005, places=6)

    def test_new_thickness_name_works_in_flat_mode(self):
        out = self.dir / "flat.glb"
        code, _, stderr = self.run_main(self.source, "-o", out, "--frame-thickness", 0.5)
        self.assertEqual(code, 0, stderr)
        self.assertAlmostEqual(trimesh.load_mesh(out).extents[2], 0.005, places=6)

    def test_temple_flags_apply_to_both_sides(self):
        scene, _ = self.complete("--temple-length", 11, "--temple-curvature", 0,
                                 "--temple-splay", 0, "--temple-ear-drop", 0,
                                 "--temple-width", 0.7, "--temple-thickness", 0.3)
        for name in (LEFT_TEMPLE, RIGHT_TEMPLE):
            temple = scene.geometry[name]
            self.assertAlmostEqual(np.ptp(temple.vertices[:, 2]), 0.11, delta=0.001)
            self.assertAlmostEqual(temple.extents[1], 0.007, places=6)
            self.assertAlmostEqual(temple.extents[0], 0.003, places=6)

    def test_side_flags_override_the_shared_value(self):
        scene, _ = self.complete("--temple-length", 12, "--right-temple-length", 15)
        left = np.ptp(scene.geometry[LEFT_TEMPLE].vertices[:, 2])
        right = np.ptp(scene.geometry[RIGHT_TEMPLE].vertices[:, 2])
        self.assertLess(left, 0.125)
        self.assertGreater(right, 0.14)

    def test_nose_pads_flag(self):
        scene, _ = self.complete()
        self.assertNotIn(LEFT_NOSE_PAD, scene.geometry)
        scene, _ = self.complete("--nose-pads")
        self.assertIn(LEFT_NOSE_PAD, scene.geometry)

    def test_invalid_numbers_fail_cleanly(self):
        for flags in (("--frame-curvature", -1), ("--temple-length", 0), ("--frame-width", "nan"),
                      ("--temple-width", -2), ("--temple-curvature", 100),
                      ("--left-temple-length", -5), ("--opening-cleanup", 0.9)):
            with self.subTest(flags=flags):
                out = self.dir / "bad.glb"
                code, _, stderr = self.run_main(self.source, "--complete", "-o", out, *flags)
                self.assertEqual(code, 1)
                self.assertIn("error:", stderr)
                self.assertFalse(out.exists())

    def test_non_numeric_value_is_a_usage_error(self):
        code, _, _ = self.run_main(self.source, "--complete", "--temple-length", "long")
        self.assertEqual(code, 2)

    def test_procedural_flags_without_complete_are_a_usage_error(self):
        for flags in (("--temple-length", 12), ("--frame-curvature", 1), ("--nose-pads",),
                      ("--left-temple-length", 12)):
            with self.subTest(flags=flags):
                code, _, stderr = self.run_main(self.source, *flags)
                self.assertEqual(code, 2)
                self.assertIn("--complete", stderr)

    def test_demo_complete_works(self):
        out = self.dir / "demo.glb"
        code, _, stderr = self.run_main("--demo", "--complete", "-o", out)
        self.assertEqual(code, 0, stderr)
        self.assertIn(RIGHT_TEMPLE, trimesh.load_scene(out).geometry)

    def test_failed_validation_fails_cleanly_and_writes_nothing(self):
        out = self.dir / "bad.glb"
        code, _, stderr = self.run_main(self.source, "--complete", "-o", out,
                                        "--temple-splay", 0, "--temple-curvature", 7,
                                        "--temple-ear-drop", 0)
        self.assertEqual(code, 1)
        self.assertIn("intersect", stderr)
        self.assertFalse(out.exists())


@unittest.skipUnless(PHOTO.exists(), "image_to_3d/glasses.jpg is not available")
class RealPhotoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.result = convert_image_to_glasses(PHOTO, Path(directory.name) / "photo.glb")

    def test_validates(self):
        self.assertEqual(self.result.model.report.problems, [])

    def test_two_openings(self):
        self.assertEqual(self.result.model.openings, 2)

    def test_temple_seen_through_the_lens_is_not_part_of_the_frame(self):
        # Measured on the photo: at y = 350 the arm behind the lens spans x = 130..154.
        mask = self.result.mask
        self.assertEqual(mask[350, 142], 0)
        front = self.result.model.components[FRONT_FRAME]
        self.assertFalse(covers_xy(front, pixel_to_mesh_xy((142, 350), mask, WIDTH)))

    def test_rim_stays_solid(self):
        # At y = 400 the left rim spans x = 67..93.
        mask = self.result.mask
        self.assertEqual(mask[400, 80], 255)
        front = self.result.model.components[FRONT_FRAME]
        self.assertTrue(covers_xy(front, pixel_to_mesh_xy((80, 400), mask, WIDTH)))

    def test_drop_shadow_is_not_part_of_the_frame(self):
        # At y = 541 only the faint shadow below the rim remains.
        self.assertEqual(self.result.mask[541, 280], 0)

    def test_front_view_matches_the_traced_silhouette(self):
        mask = self.result.mask
        ys, xs = np.nonzero(mask)
        left, right, top, bottom = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        scale = (right - left) / WIDTH
        front = self.result.model.components[FRONT_FRAME]
        triangles = front.triangles[front.face_normals[:, 2] > 0][:, :, :2]
        pixels = np.stack(((left + right) / 2 + triangles[..., 0] * scale,
                           (top + bottom) / 2 - triangles[..., 1] * scale), axis=-1)
        drawn = np.zeros(mask.shape, np.uint8)
        for triangle in pixels:
            cv2.fillConvexPoly(drawn, np.round(triangle * 16).astype(np.int32), 255, cv2.LINE_8, 4)
        overlap = np.count_nonzero((drawn > 0) & (mask > 0)) / np.count_nonzero((drawn > 0) | (mask > 0))
        self.assertGreater(overlap, 0.93)

    def test_hinges_sit_near_the_top_of_the_outer_edge(self):
        model = self.result.model
        self.assertGreater(model.right_pivot[1], 0.5)
        self.assertGreater(model.left_pivot[1], 0.5)
        self.assertAlmostEqual(model.right_pivot[1], model.left_pivot[1], delta=0.2)


if __name__ == "__main__":
    unittest.main()
