from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh

from accessory_transform import attachment_matrix, project_accessory
from glasses_3d import DEFAULT_GLASSES_ATTACHMENT, GlassesDimensions
from glb_model import (MeshModel, MeshPart, ModelLoadError, load_glasses_model)
from image_to_3d.assembly import (FRONT_FRAME, LEFT_HINGE, LEFT_TEMPLE, RIGHT_HINGE, RIGHT_TEMPLE,
                                  GlassesParameters, assemble_glasses, export_model_glb)
from image_to_3d.demo import make_demo_image
from image_to_3d.mesh import mask_to_polygons
from image_to_3d.segmentation import segment_glasses
from tests.test_head_pose import FRONTAL_ROTATION, make_pose


def generated_model(params=GlassesParameters()):
    return assemble_glasses(mask_to_polygons(segment_glasses(make_demo_image())), params)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        load_glasses_model.cache_clear()
        self.addCleanup(load_glasses_model.cache_clear)


class LoadGeneratedModelTests(TempDirTestCase):
    def load(self, params=GlassesParameters()):
        source = generated_model(params)
        path = export_model_glb(source, self.dir / "model.glb")
        return source, load_glasses_model(path)

    def test_components_are_preserved_by_name(self):
        _, model = self.load()
        self.assertEqual(set(model.part_names),
                         {FRONT_FRAME, LEFT_TEMPLE, RIGHT_TEMPLE, LEFT_HINGE, RIGHT_HINGE})

    def test_metres_become_centimetres(self):
        _, model = self.load(GlassesParameters(frame_width=13.0))
        self.assertAlmostEqual(np.ptp(model.part(FRONT_FRAME).vertices[:, 0]), 13.0, places=4)

    def test_geometry_matches_the_generator_exactly_up_to_float_precision(self):
        source, model = self.load()
        for name in (FRONT_FRAME, LEFT_TEMPLE, LEFT_HINGE):
            np.testing.assert_allclose(model.part(name).vertices.min(axis=0),
                                       source.components[name].bounds[0], atol=1e-4)
            np.testing.assert_allclose(model.part(name).vertices.max(axis=0),
                                       source.components[name].bounds[1], atol=1e-4)
            self.assertEqual(len(model.part(name).faces), len(source.components[name].faces))

    def test_proportions_are_preserved(self):
        source, model = self.load()
        expected = source.components[FRONT_FRAME].extents
        actual = np.ptp(model.part(FRONT_FRAME).vertices, axis=0)
        self.assertAlmostEqual(actual[1] / actual[0], expected[1] / expected[0], places=5)

    def test_axes_stay_x_right_y_up_z_forward(self):
        _, model = self.load()
        self.assertGreater(model.part(RIGHT_TEMPLE).vertices[:, 0].min(), 0)
        self.assertLess(model.part(LEFT_TEMPLE).vertices[:, 0].max(), 0)
        self.assertLess(model.part(RIGHT_TEMPLE).vertices[:, 2].min(), -10)
        self.assertLessEqual(model.part(FRONT_FRAME).vertices[:, 2].max(), 1e-5)

    def test_materials_are_read_as_bgr(self):
        path = self.dir / "colored.glb"
        mesh = trimesh.creation.box((0.1, 0.1, 0.1))
        material = trimesh.visual.material.PBRMaterial(baseColorFactor=(200, 100, 50, 255),
                                                       metallicFactor=0.0, roughnessFactor=0.5)
        mesh.visual = trimesh.visual.TextureVisuals(material=material)
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name="box", geom_name="box")
        path.write_bytes(scene.export(file_type="glb"))
        self.assertEqual(tuple(load_glasses_model(path).part("box").color_bgr), (50, 100, 200))

    def test_node_transforms_are_applied(self):
        path = self.dir / "moved.glb"
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.box((0.14, 0.05, 0.01)), node_name="front", geom_name="front",
                           transform=trimesh.transformations.translation_matrix((0.02, 0, 0)))
        path.write_bytes(scene.export(file_type="glb"))
        self.assertAlmostEqual(load_glasses_model(path).part("front").vertices[:, 0].mean(), 2.0,
                               places=4)

    def test_model_works_with_the_existing_accessory_transforms(self):
        _, model = self.load()
        attach = attachment_matrix(DEFAULT_GLASSES_ATTACHMENT, 1.0)
        pose = make_pose(FRONTAL_ROTATION, (0.0, 0.0, 45.0))
        pixels = project_accessory(model.part(FRONT_FRAME).vertices, attach, pose)
        self.assertTrue(np.isfinite(pixels).all())

    def test_default_size_agrees_with_the_wireframe_model(self):
        _, model = self.load()
        self.assertAlmostEqual(np.ptp(model.part(FRONT_FRAME).vertices[:, 0]),
                               GlassesDimensions().frame_width, places=4)


class CachingTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.path = export_model_glb(generated_model(), self.dir / "model.glb")

    def test_second_load_reuses_the_same_object_and_does_not_reparse(self):
        first = load_glasses_model(self.path)
        with mock.patch("glb_model.trimesh.load_scene") as parse:
            second = load_glasses_model(self.path)
        parse.assert_not_called()
        self.assertIs(first, second)

    def test_changed_file_is_reloaded(self):
        first = load_glasses_model(self.path)
        export_model_glb(generated_model(GlassesParameters(frame_width=12.0)), self.path)
        second = load_glasses_model(self.path)
        self.assertIsNot(first, second)
        self.assertAlmostEqual(np.ptp(second.part(FRONT_FRAME).vertices[:, 0]), 12.0, places=4)

    def test_render_data_is_built_once(self):
        model = load_glasses_model(self.path)
        self.assertIs(model.render_data, model.render_data)


class InvalidFileTests(TempDirTestCase):
    def test_missing_file(self):
        with self.assertRaisesRegex(ModelLoadError, "not found"):
            load_glasses_model(self.dir / "absent.glb")

    def test_directory(self):
        with self.assertRaises(ModelLoadError):
            load_glasses_model(self.dir)

    def test_empty_file(self):
        path = self.dir / "empty.glb"
        path.write_bytes(b"")
        with self.assertRaisesRegex(ModelLoadError, "empty"):
            load_glasses_model(path)

    def test_garbage_file(self):
        path = self.dir / "garbage.glb"
        path.write_bytes(b"definitely not a glb file at all")
        with self.assertRaises(ModelLoadError):
            load_glasses_model(path)

    def test_truncated_file(self):
        whole = export_model_glb(generated_model(), self.dir / "whole.glb").read_bytes()
        path = self.dir / "truncated.glb"
        path.write_bytes(whole[:len(whole) // 2])
        with self.assertRaises(ModelLoadError):
            load_glasses_model(path)

    def test_wrong_extension_is_rejected(self):
        path = self.dir / "model.obj"
        path.write_bytes(b"v 0 0 0\n")
        with self.assertRaisesRegex(ModelLoadError, r"\.glb"):
            load_glasses_model(path)

    def test_scene_without_meshes(self):
        path = self.dir / "none.glb"
        # A valid glTF container holding no meshes at all.
        import json
        import struct
        document = json.dumps({"asset": {"version": "2.0"}}).encode()
        document += b" " * (-len(document) % 4)
        path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(document))
                         + struct.pack("<I4s", len(document), b"JSON") + document)
        with self.assertRaisesRegex(ModelLoadError, "no meshes|no usable"):
            load_glasses_model(path)

    def test_model_in_centimetres_is_rejected_by_its_implausible_size(self):
        path = self.dir / "centimetres.glb"
        # 14 "metres" wide: someone exported centimetre values without scaling.
        path.write_bytes(trimesh.creation.box((14.0, 5.0, 0.4)).export(file_type="glb"))
        with self.assertRaisesRegex(ModelLoadError, "size"):
            load_glasses_model(path)

    def test_mesh_with_only_zero_area_faces_is_rejected_when_loading(self):
        path = self.dir / "flat.glb"
        vertices = np.array([(0, 0, 0), (0.1, 0, 0), (0.2, 0, 0), (0, 0.1, 0)], float)
        mesh = trimesh.Trimesh(vertices, np.array([(0, 1, 2)]), process=False, validate=False)
        path.write_bytes(trimesh.Scene({"flat": mesh}).export(file_type="glb"))
        with self.assertRaises(ModelLoadError):
            load_glasses_model(path)

    def test_render_data_is_prepared_during_loading(self):
        path = export_model_glb(generated_model(), self.dir / "ready.glb")
        self.assertIn("render_data", vars(load_glasses_model(path)))

    def test_failure_is_not_cached(self):
        path = self.dir / "later.glb"
        with self.assertRaises(ModelLoadError):
            load_glasses_model(path)
        export_model_glb(generated_model(), path)
        self.assertGreater(len(load_glasses_model(path).part_names), 0)


class MeshPartValidationTests(unittest.TestCase):
    TRIANGLE = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], float)

    def test_non_finite_vertices_rejected(self):
        bad = self.TRIANGLE.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ModelLoadError):
            MeshModel("m", (MeshPart("p", bad, np.array([[0, 1, 2]])),))

    def test_out_of_range_faces_rejected(self):
        with self.assertRaises(ModelLoadError):
            MeshModel("m", (MeshPart("p", self.TRIANGLE, np.array([[0, 1, 9]])),))

    def test_no_parts_rejected(self):
        with self.assertRaises(ModelLoadError):
            MeshModel("m", ())

    def test_unknown_part_name_raises_key_error(self):
        model = MeshModel("m", (MeshPart("p", self.TRIANGLE, np.array([[0, 1, 2]])),))
        with self.assertRaises(KeyError):
            model.part("nope")


class RenderDataTests(unittest.TestCase):
    def model_of(self, mesh, color=(10, 20, 30)):
        return MeshModel("m", (MeshPart("p", mesh.vertices, mesh.faces, color),))

    def test_one_row_per_face(self):
        box = trimesh.creation.box((1, 1, 1))
        data = self.model_of(box).render_data
        self.assertEqual(data.corners.shape, (12, 3, 3))
        self.assertEqual(data.normals.shape, (12, 3, 3))
        self.assertEqual(data.colors.shape, (12, 3))

    def test_sharp_edges_keep_flat_normals(self):
        box = trimesh.creation.box((1, 1, 1))
        data = self.model_of(box).render_data
        # Each face's three corner normals agree, because a box has only 90 degree edges.
        np.testing.assert_allclose(data.normals[:, 0], data.normals[:, 1], atol=1e-9)
        np.testing.assert_allclose(data.normals[:, 0], data.normals[:, 2], atol=1e-9)

    def test_gentle_curves_get_smooth_normals(self):
        sphere = trimesh.creation.icosphere(subdivisions=3)
        data = self.model_of(sphere).render_data
        # On a sphere the smoothed normal at a corner is nearly the position itself.
        # A flat face normal is about 5 degrees off at this subdivision: a dot of 0.996.
        dots = (data.normals * data.corners).sum(axis=2)
        self.assertGreater(dots.min(), 0.999)

    def test_normals_are_unit_length_and_point_outward(self):
        box = trimesh.creation.box((1, 1, 1))
        data = self.model_of(box).render_data
        np.testing.assert_allclose(np.linalg.norm(data.normals, axis=2), 1.0)
        self.assertTrue(((data.normals * data.corners).sum(axis=2) > 0).all())

    def test_colors_follow_their_parts(self):
        a = trimesh.creation.box((1, 1, 1))
        b = trimesh.creation.box((1, 1, 1))
        model = MeshModel("m", (MeshPart("a", a.vertices, a.faces, (1, 2, 3)),
                                MeshPart("b", b.vertices, b.faces, (4, 5, 6))))
        colors = model.render_data.colors
        np.testing.assert_array_equal(colors[0], (1, 2, 3))
        np.testing.assert_array_equal(colors[-1], (4, 5, 6))


if __name__ == "__main__":
    unittest.main()
