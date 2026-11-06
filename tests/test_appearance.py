"""Texture and materials taken from the product photo."""
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import cv2
import numpy as np
from shapely.geometry import Point
import trimesh

from image_to_3d.appearance import (MaterialSettings, extract_appearance, frame_uvs,
                                    textured_frame_mesh)
from image_to_3d.assembly import (FRONT_FRAME, LEFT_HINGE, LEFT_TEMPLE, RIGHT_TEMPLE,
                                  GlassesParameters, assemble_glasses, export_model_glb)
from image_to_3d.errors import SegmentationError
from image_to_3d.mesh import mask_to_polygons, scale_polygons_to_cm
from image_to_3d.mirror_image_to_3d import main
from image_to_3d.segmentation import load_image, segment_glasses

WIDTH_CM = 14.0
# BGR, as OpenCV draws them.
RED = (40, 40, 200)
BLUE = (200, 60, 40)
GREEN = (40, 180, 40)
YELLOW = (0, 200, 200)
WHITE = (255, 255, 255)
LEFT_TOP_BAND = (155, 60)   # a pixel well inside the left rim's top band
RIGHT_TOP_BAND = (445, 60)
LEFT_BOTTOM_BAND = (155, 200)


def frame_image(left=RED, right=BLUE, bridge=GREEN, top=None, ends=None, channels=3,
                background=WHITE, antialias=False):
    """Two rectangular rims and a bridge; optional differently coloured top bands and ends."""
    line = cv2.LINE_AA if antialias else cv2.LINE_8
    image = np.zeros((260, 600, channels), np.uint8)
    image[..., :3] = background
    alpha = (255,) if channels == 4 else ()
    cv2.rectangle(image, (40, 60), (270, 200), (*left, *alpha), 24, line)
    cv2.rectangle(image, (330, 60), (560, 200), (*right, *alpha), 24, line)
    cv2.line(image, (270, 90), (330, 90), (*bridge, *alpha), 20, line)
    if top is not None:
        cv2.line(image, (40, 60), (270, 60), (*top, *alpha), 24, line)
        cv2.line(image, (330, 60), (560, 60), (*top, *alpha), 24, line)
    if ends is not None:
        frame = image[..., :3].min(axis=2) < 230
        columns = np.arange(image.shape[1])
        end = frame & ((columns < 100) | (columns > 500))[None, :]
        image[end, :3] = ends
    return image


def appearance_of(image, mask=None, settings=MaterialSettings()):
    mask = segment_glasses(image) if mask is None else mask
    polygons = mask_to_polygons(mask)
    return extract_appearance(image, mask, polygons, WIDTH_CM, settings), polygons


def texture_rgb(appearance, uv):
    """Nearest texel (RGB) at trimesh-convention UVs (v up)."""
    height, width = appearance.texture.shape[:2]
    uv = np.atleast_2d(uv)
    col = np.clip(np.floor(uv[:, 0] * width), 0, width - 1).astype(int)
    row = np.clip(np.floor((1 - uv[:, 1]) * height), 0, height - 1).astype(int)
    return appearance.texture[row, col].astype(int)


def rgb(bgr):
    return np.array(bgr[::-1], int)


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.appearance, polygons = appearance_of(frame_image())
        self.mask = segment_glasses(frame_image())
        self.polygons_cm = scale_polygons_to_cm(polygons, WIDTH_CM)

    def test_points_inside_the_model_land_on_the_frame_in_the_photo(self):
        points = []
        for polygon in self.polygons_cm:
            x0, y0, x1, y1 = polygon.bounds
            for x in np.linspace(x0, x1, 40):
                for y in np.linspace(y0, y1, 20):
                    if polygon.buffer(-0.05).contains(Point(x, y)):
                        points.append((x, y))
        pixels = np.round(self.appearance.mapping.cm_to_image(np.array(points))).astype(int)
        on_frame = self.mask[pixels[:, 1], pixels[:, 0]] > 0
        self.assertGreater(len(points), 100)
        self.assertTrue(on_frame.all())

    def test_outer_edges_land_on_the_photo_edges(self):
        ys, xs = np.nonzero(self.mask)
        (left, top), = self.appearance.mapping.cm_to_image(np.array([[-WIDTH_CM / 2, 0.0]]))
        self.assertAlmostEqual(left, xs.min() - 0.5, delta=1.0)
        (right, _), = self.appearance.mapping.cm_to_image(np.array([[WIDTH_CM / 2, 0.0]]))
        self.assertAlmostEqual(right, xs.max() + 0.5, delta=1.0)

    def test_up_in_the_model_is_up_in_the_photo(self):
        (_, high), (_, low) = self.appearance.mapping.cm_to_image(np.array([[0.0, 1.0], [0.0, -1.0]]))
        self.assertLess(high, low)

    def test_no_half_pixel_drift_on_large_photos(self):
        # Each working pixel spans about 3.4 photo pixels here, so dropping the
        # half-pixel term would move both edges by about 1.7 pixels the same way.
        image = cv2.resize(frame_image(), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        appearance, _ = appearance_of(image)
        ys, xs = np.nonzero(segment_glasses(image))
        (left, _), (right, _) = appearance.mapping.cm_to_image(
            np.array([[-WIDTH_CM / 2, 0.0], [WIDTH_CM / 2, 0.0]]))
        self.assertAlmostEqual((left + right) / 2, (xs.min() + xs.max()) / 2, delta=0.6)

    def test_inverse(self):
        points = np.array([[-6.0, 1.5], [0.0, 0.0], [3.2, -2.0]])
        mapping = self.appearance.mapping
        np.testing.assert_allclose(mapping.image_to_cm(mapping.cm_to_image(points)), points,
                                   atol=1e-9)

    def test_any_image_resolution(self):
        for factor in (0.5, 3.0):
            with self.subTest(factor=factor):
                image = cv2.resize(frame_image(), None, fx=factor, fy=factor,
                                   interpolation=cv2.INTER_NEAREST)
                appearance, _ = appearance_of(image)
                (x, y), = appearance.mapping.cm_to_image(
                    self.appearance.mapping.image_to_cm(np.array([LEFT_TOP_BAND], float)))
                self.assertAlmostEqual(x, LEFT_TOP_BAND[0] * factor, delta=2 * factor + 1)
                self.assertAlmostEqual(y, LEFT_TOP_BAND[1] * factor, delta=2 * factor + 1)


class ExtractionTests(unittest.TestCase):
    def uv_of_pixel(self, appearance, pixel, front=True):
        cm = appearance.mapping.image_to_cm(np.array([pixel], float))
        return appearance.uv_at(cm, front)

    def test_colours_are_preserved_where_they_are(self):
        appearance, _ = appearance_of(frame_image())
        for pixel, color in ((LEFT_TOP_BAND, RED), (RIGHT_TOP_BAND, BLUE), ((300, 90), GREEN)):
            with self.subTest(pixel=pixel):
                np.testing.assert_allclose(texture_rgb(appearance, self.uv_of_pixel(appearance, pixel))[0],
                                           rgb(color), atol=2)

    def test_not_a_single_average_colour(self):
        appearance, _ = appearance_of(frame_image())
        left = texture_rgb(appearance, self.uv_of_pixel(appearance, LEFT_TOP_BAND))[0]
        right = texture_rgb(appearance, self.uv_of_pixel(appearance, RIGHT_TOP_BAND))[0]
        self.assertGreater(np.abs(left - right).sum(), 200)

    def test_white_background_never_enters_the_texture(self):
        appearance, _ = appearance_of(frame_image(antialias=True))
        # Every frame colour has a channel at or below 60; white and the blended edge don't.
        self.assertLessEqual(appearance.texture.min(axis=2).max(), 60)

    def test_transparent_background_colour_never_enters_the_texture(self):
        image = frame_image(bridge=RED, channels=4, background=(0, 255, 0))
        image[..., 3] = np.where((image[..., :3] == (0, 255, 0)).all(axis=2), 0, 255)
        appearance, _ = appearance_of(image)
        self.assertLess(appearance.texture[..., 1].max(), 100)

    def test_png_and_jpg_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix, tolerance in ((".png", 2), (".jpg", 14)):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / f"glasses{suffix}"
                    path.write_bytes(cv2.imencode(suffix, frame_image())[1].tobytes())
                    appearance, _ = appearance_of(load_image(path))
                    np.testing.assert_allclose(
                        texture_rgb(appearance, self.uv_of_pixel(appearance, LEFT_TOP_BAND))[0],
                        rgb(RED), atol=tolerance)

    def test_highlights_stay_on_the_front_and_are_flattened_for_edges(self):
        plain = frame_image()
        lit = plain.copy()
        cv2.line(lit, (90, 60), (220, 60), (235, 235, 235), 4)
        appearance, _ = appearance_of(lit, mask=segment_glasses(plain))
        front = texture_rgb(appearance, self.uv_of_pixel(appearance, LEFT_TOP_BAND))[0]
        edge = texture_rgb(appearance, self.uv_of_pixel(appearance, LEFT_TOP_BAND, front=False))[0]
        self.assertGreater(front.min(), 220)
        self.assertLess(edge.sum(), front.sum() - 300)
        # Away from the highlight, the edge colours are the photo's.
        elsewhere = self.uv_of_pixel(appearance, LEFT_BOTTOM_BAND, front=False)
        np.testing.assert_allclose(texture_rgb(appearance, elsewhere)[0], rgb(RED), atol=3)

    def test_temple_colour_comes_from_the_end_pieces(self):
        appearance, _ = appearance_of(frame_image(ends=YELLOW))
        np.testing.assert_allclose(appearance.temple_color, rgb(YELLOW), atol=3)
        self.assertGreater(np.abs(np.subtract(appearance.frame_color, rgb(YELLOW))).sum(), 100)

    def test_frame_colour_is_the_typical_frame_pixel(self):
        appearance, _ = appearance_of(frame_image(left=RED, right=RED, bridge=RED))
        np.testing.assert_allclose(appearance.frame_color, rgb(RED), atol=2)

    def test_same_input_same_texture(self):
        first, _ = appearance_of(frame_image(antialias=True))
        second, _ = appearance_of(frame_image(antialias=True))
        np.testing.assert_array_equal(first.texture, second.texture)

    def test_large_photos_are_capped(self):
        image = cv2.resize(frame_image(), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        appearance, _ = appearance_of(image)
        self.assertLessEqual(max(appearance.texture.shape[:2]), 2048)

    def test_invalid_inputs(self):
        image = frame_image()
        mask = segment_glasses(image)
        polygons = mask_to_polygons(mask)
        with self.assertRaises(ValueError):
            extract_appearance(image[:100], mask, polygons, WIDTH_CM)
        with self.assertRaises(ValueError):
            extract_appearance(image[..., 0], mask, polygons, WIDTH_CM)
        with self.assertRaises(SegmentationError):
            extract_appearance(image, np.zeros_like(mask), polygons, WIDTH_CM)
        for bad in (dict(frame_roughness=1.5), dict(frame_roughness=float("nan")),
                    dict(hinge_material="gold")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    MaterialSettings(**bad).validate()


class FrameUvTests(unittest.TestCase):
    def build(self, image, curvature=0.5):
        appearance, polygons = appearance_of(image)
        model = assemble_glasses(polygons, GlassesParameters(frame_curvature=curvature))
        return appearance, model.components[FRONT_FRAME]

    def front_colors(self, appearance, front):
        uvs = frame_uvs(appearance, front)
        facing = front.face_normals[:, 2] > 0.5
        return (front.triangles_center[facing], texture_rgb(appearance, uvs[facing].mean(axis=1)))

    def test_front_faces_show_the_photo_at_their_position(self):
        for curvature in (0.0, 1.0):
            with self.subTest(curvature=curvature):
                image = frame_image()
                appearance, front = self.build(image, curvature)
                centers, colors = self.front_colors(appearance, front)
                pixels = np.round(appearance.mapping.cm_to_image(centers[:, :2])).astype(int)
                expected = image[pixels[:, 1], pixels[:, 0]][:, ::-1].astype(int)
                inside = segment_glasses(image)[pixels[:, 1], pixels[:, 0]] > 0
                close = np.abs(colors - expected).max(axis=1) <= 3
                self.assertGreater(close[inside].mean(), 0.97)

    def test_left_stays_left_and_top_stays_top(self):
        appearance, front = self.build(frame_image(top=YELLOW))
        centers, colors = self.front_colors(appearance, front)
        is_blue = np.abs(colors - rgb(BLUE)).max(axis=1) < 10
        is_red = np.abs(colors - rgb(RED)).max(axis=1) < 10
        is_yellow = np.abs(colors - rgb(YELLOW)).max(axis=1) < 10
        self.assertTrue((centers[is_blue, 0] > 0).all())
        self.assertTrue((centers[is_red, 0] < 0).all())
        self.assertGreater(centers[is_yellow, 1].min(), centers[is_red | is_blue, 1].min())
        self.assertGreater(centers[is_yellow, 1].mean(), centers[is_red | is_blue, 1].mean())

    def test_sides_and_back_use_the_edge_half(self):
        appearance, front = self.build(frame_image())
        uvs = frame_uvs(appearance, front)
        facing = front.face_normals[:, 2] > 0.5
        self.assertTrue((uvs[facing][..., 1] >= 0.5).all())
        self.assertTrue((uvs[~facing][..., 1] <= 0.5).all())

    def test_textured_mesh_keeps_the_geometry(self):
        appearance, front = self.build(frame_image())
        textured = textured_frame_mesh(front, appearance)
        np.testing.assert_allclose(textured.bounds, front.bounds)
        welded = trimesh.Trimesh(textured.vertices, textured.faces)  # merges by position
        self.assertEqual(len(welded.vertices), len(front.vertices))
        self.assertTrue(welded.is_volume)

    def test_each_face_keeps_its_half_after_vertices_are_split(self):
        appearance, front = self.build(frame_image())
        textured = textured_frame_mesh(front, appearance)
        corner_v = textured.visual.uv[textured.faces][..., 1]
        facing = textured.face_normals[:, 2] > 0.5
        self.assertTrue((corner_v[facing] >= 0.5).all())
        self.assertTrue((corner_v[~facing] <= 0.5).all())

    def test_no_seams_inside_the_front_surface(self):
        appearance, front = self.build(frame_image())
        textured = textured_frame_mesh(front, appearance)
        uv = textured.visual.uv
        # A position appears at most twice: once on the front, once on the sides and back.
        _, counts = np.unique(np.round(textured.vertices, 9), axis=0, return_counts=True)
        self.assertLessEqual(counts.max(), 2)
        self.assertEqual(len(uv), len(textured.vertices))


def glb_json(path):
    data = Path(path).read_bytes()
    length = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + length])


class ExportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        self.appearance, polygons = appearance_of(frame_image(ends=YELLOW))
        self.model = assemble_glasses(polygons)

    def export(self, name="model.glb", appearance="default"):
        appearance = self.appearance if appearance == "default" else appearance
        return export_model_glb(self.model, self.dir / name, appearance)

    def materials_by_node(self, path):
        document = glb_json(path)
        result = {}
        for node in document["nodes"]:
            if "mesh" in node:
                primitive = document["meshes"][node["mesh"]]["primitives"][0]
                result[node["name"]] = document["materials"][primitive["material"]]
        return document, result

    def test_texture_is_embedded_not_linked(self):
        document, _ = self.materials_by_node(self.export())
        self.assertEqual(len(document["images"]), 1)
        self.assertNotIn("uri", document["images"][0])
        self.assertEqual(document["images"][0]["mimeType"], "image/png")

    def test_front_is_textured_and_other_parts_are_coloured(self):
        _, materials = self.materials_by_node(self.export())
        self.assertIn("baseColorTexture", materials[FRONT_FRAME]["pbrMetallicRoughness"])
        temple = materials[LEFT_TEMPLE]["pbrMetallicRoughness"]
        self.assertNotIn("baseColorTexture", temple)
        np.testing.assert_allclose(np.array(temple["baseColorFactor"][:3]) * 255,
                                   rgb(YELLOW), atol=3)
        self.assertEqual(materials[LEFT_TEMPLE]["pbrMetallicRoughness"],
                         materials[RIGHT_TEMPLE]["pbrMetallicRoughness"])

    def test_hinges_match_the_frame_unless_metal_is_asked_for(self):
        _, materials = self.materials_by_node(self.export())
        hinge = materials[LEFT_HINGE]["pbrMetallicRoughness"]
        np.testing.assert_allclose(np.array(hinge["baseColorFactor"][:3]) * 255,
                                   self.appearance.frame_color, atol=2)
        self.assertEqual(hinge.get("metallicFactor", 1.0), 0.0)
        appearance, _ = appearance_of(frame_image(), settings=MaterialSettings(hinge_material="metal"))
        _, materials = self.materials_by_node(self.export("metal.glb", appearance))
        self.assertGreater(materials[LEFT_HINGE]["pbrMetallicRoughness"]["metallicFactor"], 0.5)

    def test_texture_survives_a_reload(self):
        scene = trimesh.load_scene(self.export())
        front = scene.geometry[FRONT_FRAME]
        reloaded = np.asarray(front.visual.material.baseColorTexture.convert("RGB"))
        np.testing.assert_array_equal(reloaded, self.appearance.texture)
        self.assertEqual(len(front.visual.uv), len(front.vertices))

    def test_dimensions_match_the_untextured_model(self):
        textured = trimesh.load_scene(self.export("textured.glb"))
        plain = trimesh.load_scene(self.export("plain.glb", None))
        for name in self.model.components:
            np.testing.assert_allclose(textured.geometry[name].bounds, plain.geometry[name].bounds,
                                       atol=1e-9)

    def test_identical_bytes_for_identical_input(self):
        self.assertEqual(self.export("a.glb").read_bytes(), self.export("b.glb").read_bytes())

    def test_untextured_export_is_unchanged(self):
        document, materials = self.materials_by_node(self.export("plain.glb", None))
        self.assertNotIn("images", document)
        np.testing.assert_allclose(materials[FRONT_FRAME]["pbrMetallicRoughness"]["baseColorFactor"],
                                   np.array((30, 30, 34, 255)) / 255, atol=1e-6)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        self.source = self.dir / "glasses.png"
        self.source.write_bytes(cv2.imencode(".png", frame_image())[1].tobytes())

    def run_main(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = main([str(arg) for arg in argv])
            except SystemExit as stop:
                code = stop.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_complete_models_are_textured_by_default(self):
        out = self.dir / "model.glb"
        code, stdout, stderr = self.run_main(self.source, "--complete", "-o", out)
        self.assertEqual(code, 0, stderr)
        self.assertIn("images", glb_json(out))
        self.assertIn("texture", stdout)

    def test_no_texture_flag(self):
        out = self.dir / "model.glb"
        code, _, stderr = self.run_main(self.source, "--complete", "--no-texture", "-o", out)
        self.assertEqual(code, 0, stderr)
        self.assertNotIn("images", glb_json(out))

    def test_texture_flags_need_complete(self):
        for flags in (["--no-texture"], ["--hinge-material", "metal"], ["--frame-roughness", "0.5"]):
            with self.subTest(flags=flags):
                code, _, stderr = self.run_main(self.source, *flags, "-o", self.dir / "x.glb")
                self.assertEqual(code, 2)
                self.assertIn("--complete", stderr)

    def test_model_works_after_the_photo_is_gone(self):
        from glb_model import load_glasses_model
        out = self.dir / "elsewhere" / "model.glb"
        code, _, stderr = self.run_main(self.source, "--complete", "-o", out)
        self.assertEqual(code, 0, stderr)
        self.source.unlink()
        moved = self.dir / "moved.glb"
        out.rename(moved)
        load_glasses_model.cache_clear()
        self.assertIsNotNone(load_glasses_model(moved).part(FRONT_FRAME).texture)


if __name__ == "__main__":
    unittest.main()
