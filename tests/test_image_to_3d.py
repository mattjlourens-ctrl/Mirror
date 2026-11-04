from pathlib import Path
import contextlib
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import trimesh

from image_to_3d import (FrameDimensions, ImageLoadError, MeshBuildError, SegmentationConfig,
                         SegmentationError, build_frame_mesh, convert_image, export_glb,
                         load_image, make_demo_image, mask_to_polygons, segment_glasses,
                         write_demo_image)
from image_to_3d.mirror_image_to_3d import main

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAME_COLOR = (30, 30, 30)  # BGR

# Two rectangular rims joined by a bridge, on a 400x200 canvas.
LEFT_RIM = ((40, 50), (170, 150))
RIGHT_RIM = ((230, 50), (360, 150))
RIM_THICKNESS = 12
LEFT_OPENING_CENTER = (105, 100)
RIGHT_OPENING_CENTER = (295, 100)
LEFT_RIM_POINT = (40, 100)     # on the rim's outer edge band
BRIDGE_POINT = (200, 80)


def draw_rims(image, color):
    cv2.rectangle(image, *LEFT_RIM, color, RIM_THICKNESS)
    cv2.rectangle(image, *RIGHT_RIM, color, RIM_THICKNESS)
    cv2.line(image, (170, 80), (230, 80), color, RIM_THICKNESS)
    return image


def white_image():
    return draw_rims(np.full((200, 400, 3), 255, np.uint8), FRAME_COLOR)


def transparent_image():
    """White RGB under the alpha channel, so only alpha can reveal the frame."""
    image = np.zeros((200, 400, 4), np.uint8)
    image[..., :3] = 255
    return draw_rims(image, (*FRAME_COLOR, 255))


def rims_mask():
    return segment_glasses(white_image())


def covers_xy(mesh, xy):
    """Whether any triangle of the mesh covers (x, y) when looking along Z."""
    triangles = mesh.triangles[:, :, :2]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    def side(p, q):
        return ((q[:, 0] - p[:, 0]) * (xy[1] - p[:, 1])
                - (q[:, 1] - p[:, 1]) * (xy[0] - p[:, 0]))

    sides = np.stack([side(a, b), side(b, c), side(c, a)])
    return bool((np.all(sides >= 0, axis=0) | np.all(sides <= 0, axis=0)).any())


def pixel_to_mesh_xy(pixel, mask, frame_width):
    """Where an image pixel lands in mesh X/Y (image centred, Y flipped, scaled)."""
    ys, xs = np.nonzero(mask)
    left, right = xs.min(), xs.max() + 1
    top, bottom = ys.min(), ys.max() + 1
    scale = frame_width / (right - left)
    return (scale * (pixel[0] + 0.5 - (left + right) / 2),
            scale * ((top + bottom) / 2 - pixel[1] - 0.5))


class WithTempDir(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)

    def encoded_file(self, name, image, *params):
        ok, buffer = cv2.imencode(Path(name).suffix, image, list(params))
        self.assertTrue(ok)
        path = self.dir / name
        path.write_bytes(buffer.tobytes())
        return path


class LoadImageTests(WithTempDir):
    def test_png_loads(self):
        image = load_image(self.encoded_file("a.png", white_image()))
        self.assertEqual(image.shape, (200, 400, 3))

    def test_jpg_loads(self):
        image = load_image(self.encoded_file("a.jpg", white_image()))
        self.assertEqual(image.shape, (200, 400, 3))

    def test_png_alpha_channel_is_kept(self):
        image = load_image(self.encoded_file("a.png", transparent_image()))
        self.assertEqual(image.shape, (200, 400, 4))

    def test_grayscale_png_becomes_three_channels(self):
        gray = cv2.cvtColor(white_image(), cv2.COLOR_BGR2GRAY)
        image = load_image(self.encoded_file("a.png", gray))
        self.assertEqual(image.shape, (200, 400, 3))

    def test_sixteen_bit_png_becomes_eight_bit(self):
        deep = np.full((200, 400, 3), 0x2000, np.uint16)  # low byte 0: a plain cast gives 0
        deep[0, 0] = 0xFFFF
        image = load_image(self.encoded_file("a.png", deep))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image[0, 0].tolist(), [255, 255, 255])
        self.assertEqual(image[100, 100].tolist(), [32, 32, 32])

    def test_jpg_with_data_after_the_end_marker_loads(self):
        whole = self.encoded_file("whole.jpg", white_image()).read_bytes()
        path = self.dir / "trailer.jpg"
        path.write_bytes(whole + b"\x00" * 500)  # some cameras append metadata
        self.assertEqual(load_image(path).shape, (200, 400, 3))

    def test_missing_file_is_a_load_error(self):
        with self.assertRaisesRegex(ImageLoadError, "not found"):
            load_image(self.dir / "absent.png")

    def test_directory_is_a_load_error(self):
        with self.assertRaises(ImageLoadError):
            load_image(self.dir)

    def test_empty_file_is_a_load_error(self):
        path = self.dir / "empty.png"
        path.write_bytes(b"")
        with self.assertRaisesRegex(ImageLoadError, "empty"):
            load_image(path)

    def test_non_image_content_is_a_load_error(self):
        path = self.dir / "fake.png"
        path.write_bytes(b"this is definitely not an image")
        with self.assertRaisesRegex(ImageLoadError, "PNG or JPEG"):
            load_image(path)

    def test_other_image_format_is_a_load_error(self):
        path = self.encoded_file("a.bmp", white_image())
        with self.assertRaisesRegex(ImageLoadError, "PNG or JPEG"):
            load_image(path)

    def test_corrupt_png_body_is_a_load_error(self):
        path = self.dir / "corrupt.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        with self.assertRaises(ImageLoadError):
            load_image(path)

    def test_truncated_png_is_a_load_error(self):
        whole = self.encoded_file("whole.png", white_image()).read_bytes()
        path = self.dir / "truncated.png"
        path.write_bytes(whole[:len(whole) // 2])
        with self.assertRaises(ImageLoadError):
            load_image(path)

    def test_truncated_jpg_is_a_load_error(self):
        whole = self.encoded_file("whole.jpg", white_image()).read_bytes()
        path = self.dir / "truncated.jpg"
        path.write_bytes(whole[:len(whole) // 2])
        with self.assertRaises(ImageLoadError):
            load_image(path)


class SegmentationTests(unittest.TestCase):
    def assert_rims_segmented(self, mask):
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(set(np.unique(mask)), {0, 255})
        for x, y in (LEFT_RIM_POINT, BRIDGE_POINT):
            self.assertEqual(mask[y, x], 255, f"frame pixel {(x, y)}")
        for x, y in (LEFT_OPENING_CENTER, RIGHT_OPENING_CENTER, (5, 5)):
            self.assertEqual(mask[y, x], 0, f"background pixel {(x, y)}")

    def test_white_background_is_removed(self):
        self.assert_rims_segmented(segment_glasses(white_image()))

    def test_transparent_background_is_removed(self):
        self.assert_rims_segmented(segment_glasses(transparent_image()))

    def test_white_frame_on_transparent_background_is_kept(self):
        image = np.zeros((200, 400, 4), np.uint8)
        self.assert_rims_segmented(segment_glasses(draw_rims(image, (255, 255, 255, 255))))

    def test_semi_transparent_lenses_count_as_openings(self):
        image = transparent_image()
        cv2.rectangle(image, (46, 56), (164, 144), (80, 80, 80, 90), -1)
        self.assert_rims_segmented(segment_glasses(image))

    def test_opaque_png_on_white_uses_the_white_background(self):
        image = cv2.cvtColor(white_image(), cv2.COLOR_BGR2BGRA)
        self.assert_rims_segmented(segment_glasses(image))

    def test_white_card_with_transparent_rounded_corners_uses_the_white_background(self):
        image = cv2.cvtColor(white_image(), cv2.COLOR_BGR2BGRA)
        for rows in (slice(0, 20), slice(-20, None)):
            for cols in (slice(0, 20), slice(-20, None)):
                image[rows, cols] = (0, 0, 0, 0)  # black under the transparency
        self.assert_rims_segmented(segment_glasses(image))

    def test_jpeg_artifacts_do_not_close_openings_or_add_specks(self):
        _, buffer = cv2.imencode(".jpg", white_image(), [cv2.IMWRITE_JPEG_QUALITY, 40])
        mask = segment_glasses(cv2.imdecode(buffer, cv2.IMREAD_COLOR))
        self.assert_rims_segmented(mask)
        count, _ = cv2.connectedComponents(mask, connectivity=8)
        self.assertEqual(count - 1, 1)

    def test_isolated_specks_are_removed(self):
        image = white_image()
        image[10:12, 380:382] = FRAME_COLOR
        mask = segment_glasses(image)
        self.assertEqual(mask[10:12, 380:382].max(), 0)

    def test_pinholes_inside_the_frame_are_filled(self):
        image = white_image()
        image[99:101, 39:41] = 255  # a specular highlight on the rim
        mask = segment_glasses(image)
        self.assertEqual(mask[99:101, 39:41].min(), 255)

    def test_white_threshold_controls_what_counts_as_background(self):
        image = white_image()
        cv2.rectangle(image, (46, 56), (164, 144), (200, 200, 200), -1)  # pale lens tint
        self.assertEqual(segment_glasses(image)[100, 105], 255)
        relaxed = segment_glasses(image, SegmentationConfig(white_threshold=180))
        self.assertEqual(relaxed[100, 105], 0)

    def test_blank_white_image_is_a_segmentation_error(self):
        with self.assertRaisesRegex(SegmentationError, "No glasses"):
            segment_glasses(np.full((100, 100, 3), 255, np.uint8))

    def test_fully_transparent_image_is_a_segmentation_error(self):
        with self.assertRaisesRegex(SegmentationError, "No glasses"):
            segment_glasses(np.zeros((100, 100, 4), np.uint8))

    def test_non_white_background_is_a_segmentation_error(self):
        image = np.full((200, 400, 3), 90, np.uint8)
        with self.assertRaisesRegex(SegmentationError, "white or transparent"):
            segment_glasses(draw_rims(image, FRAME_COLOR))

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            segment_glasses(white_image(), SegmentationConfig(white_threshold=300))


class MaskToPolygonsTests(unittest.TestCase):
    def test_rim_openings_become_polygon_holes(self):
        polygons = mask_to_polygons(rims_mask())
        self.assertEqual(len(polygons), 1)
        self.assertEqual(len(polygons[0].interiors), 2)

    def test_disconnected_lenses_stay_separate_polygons(self):
        image = np.full((200, 400, 3), 255, np.uint8)
        cv2.rectangle(image, *LEFT_RIM, FRAME_COLOR, RIM_THICKNESS)
        cv2.rectangle(image, *RIGHT_RIM, FRAME_COLOR, RIM_THICKNESS)
        polygons = mask_to_polygons(segment_glasses(image))
        self.assertEqual(len(polygons), 2)
        self.assertEqual([len(p.interiors) for p in polygons], [1, 1])

    def test_polygons_are_valid(self):
        for polygon in mask_to_polygons(rims_mask()):
            self.assertTrue(polygon.is_valid)

    def test_outlines_match_the_mask_area(self):
        mask = rims_mask()
        ys, xs = np.nonzero(mask)
        polygons = mask_to_polygons(mask)
        bounds = polygons[0].bounds
        # Polygon area in mask pixels, undoing the working-resolution scale.
        scale = (bounds[2] - bounds[0]) / (xs.max() - xs.min() + 1)
        area = sum(p.area for p in polygons) / scale ** 2
        self.assertAlmostEqual(area / np.count_nonzero(mask), 1.0, delta=0.03)

    def test_one_pixel_rim_survives(self):
        mask = np.zeros((100, 200), np.uint8)
        cv2.rectangle(mask, (20, 20), (180, 80), 255, 1)
        polygons = mask_to_polygons(mask)
        self.assertEqual(len(polygons), 1)
        self.assertEqual(len(polygons[0].interiors), 1)

    def test_thin_rim_on_a_huge_image_survives(self):
        mask = np.zeros((1500, 6000), np.uint8)
        cv2.rectangle(mask, (100, 100), (2900, 1400), 255, 2)
        cv2.rectangle(mask, (3100, 100), (5900, 1400), 255, 2)
        cv2.line(mask, (2900, 500), (3100, 500), 255, 2)
        polygons = mask_to_polygons(mask)
        self.assertEqual(len(polygons), 1)
        self.assertEqual(len(polygons[0].interiors), 2)

    def test_single_pixel_on_a_huge_image_does_not_crash(self):
        mask = np.zeros((1500, 6000), np.uint8)
        mask[10, 10] = 255
        mask[1400, 5900] = 255
        self.assertIsInstance(mask_to_polygons(mask), list)

    def test_empty_mask_gives_no_polygons(self):
        self.assertEqual(mask_to_polygons(np.zeros((50, 50), np.uint8)), [])


class BuildFrameMeshTests(unittest.TestCase):
    def test_mesh_is_a_valid_closed_volume(self):
        mesh = build_frame_mesh(rims_mask())
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertTrue(mesh.is_volume)
        self.assertGreater(mesh.volume, 0)

    def test_frame_width_sets_the_x_extent(self):
        mesh = build_frame_mesh(rims_mask(), FrameDimensions(frame_width=10.0))
        self.assertAlmostEqual(mesh.extents[0], 10.0, places=6)

    def test_thickness_sets_the_z_extent(self):
        mesh = build_frame_mesh(rims_mask(), FrameDimensions(thickness=0.7))
        self.assertAlmostEqual(mesh.extents[2], 0.7, places=6)

    def test_aspect_ratio_follows_the_silhouette(self):
        mask = rims_mask()
        ys, xs = np.nonzero(mask)
        expected = (ys.max() - ys.min() + 1) / (xs.max() - xs.min() + 1)
        mesh = build_frame_mesh(mask)
        self.assertAlmostEqual(mesh.extents[1] / mesh.extents[0], expected, delta=0.01)

    def test_front_face_is_at_z_zero_and_depth_runs_backwards(self):
        mesh = build_frame_mesh(rims_mask(), FrameDimensions(thickness=0.5))
        self.assertAlmostEqual(mesh.bounds[1][2], 0.0, places=6)
        self.assertAlmostEqual(mesh.bounds[0][2], -0.5, places=6)

    def test_mesh_is_centred_in_x_and_y(self):
        mesh = build_frame_mesh(rims_mask())
        np.testing.assert_allclose(mesh.bounds[:, :2].mean(axis=0), 0, atol=1e-6)

    def test_openings_stay_open_and_the_frame_stays_solid(self):
        mask = rims_mask()
        dims = FrameDimensions()
        mesh = build_frame_mesh(mask, dims)
        for pixel in (LEFT_OPENING_CENTER, RIGHT_OPENING_CENTER):
            self.assertFalse(covers_xy(mesh, pixel_to_mesh_xy(pixel, mask, dims.frame_width)))
        for pixel in (LEFT_RIM_POINT, BRIDGE_POINT):
            self.assertTrue(covers_xy(mesh, pixel_to_mesh_xy(pixel, mask, dims.frame_width)))

    def test_image_up_is_mesh_plus_y(self):
        mask = np.zeros((200, 200), np.uint8)
        mask[20:60, 20:180] = 255   # wide bar high in the image
        mask[60:180, 90:110] = 255  # narrow stem below it
        mesh = build_frame_mesh(mask)
        top_row_width = np.ptp(mesh.vertices[mesh.vertices[:, 1] > mesh.bounds[1][1] - 1e-6, 0])
        self.assertAlmostEqual(top_row_width, mesh.extents[0], places=6)

    def test_disconnected_parts_are_each_kept_as_valid_volume(self):
        image = np.full((200, 400, 3), 255, np.uint8)
        cv2.rectangle(image, *LEFT_RIM, FRAME_COLOR, RIM_THICKNESS)
        cv2.rectangle(image, *RIGHT_RIM, FRAME_COLOR, RIM_THICKNESS)
        mesh = build_frame_mesh(segment_glasses(image))
        self.assertTrue(mesh.is_volume)
        self.assertEqual(len(mesh.split(only_watertight=True)), 2)

    def test_empty_mask_is_a_mesh_build_error(self):
        with self.assertRaises(MeshBuildError):
            build_frame_mesh(np.zeros((50, 50), np.uint8))

    def test_invalid_dimensions_are_rejected(self):
        for bad in (FrameDimensions(frame_width=0), FrameDimensions(thickness=-1),
                    FrameDimensions(thickness=float("nan")),
                    FrameDimensions(frame_width=float("inf"))):
            with self.assertRaises(ValueError):
                build_frame_mesh(rims_mask(), bad)


class ExportTests(WithTempDir):
    def export_rims(self, dims=FrameDimensions()):
        path = self.dir / "frame.glb"
        export_glb(build_frame_mesh(rims_mask(), dims), path)
        return path

    def test_export_writes_a_binary_gltf_file(self):
        data = self.export_rims().read_bytes()
        magic, version, declared_length = struct.unpack("<4sII", data[:12])
        self.assertEqual(magic, b"glTF")
        self.assertEqual(version, 2)
        self.assertEqual(declared_length, len(data))

    def test_export_reloads_as_a_valid_mesh(self):
        reloaded = trimesh.load_mesh(self.export_rims())
        self.assertGreater(len(reloaded.faces), 0)
        self.assertTrue(reloaded.is_volume)

    def test_export_is_in_metres_per_the_gltf_spec(self):
        dims = FrameDimensions(frame_width=14.0, thickness=0.4)
        reloaded = trimesh.load_mesh(self.export_rims(dims))
        self.assertAlmostEqual(reloaded.extents[0], 0.14, places=5)
        self.assertAlmostEqual(reloaded.extents[2], 0.004, places=5)

    def test_export_leaves_the_mesh_in_centimetres(self):
        mesh = build_frame_mesh(rims_mask())
        before = mesh.extents.copy()
        export_glb(mesh, self.dir / "frame.glb")
        np.testing.assert_array_equal(mesh.extents, before)

    def test_export_creates_missing_parent_directories(self):
        path = self.dir / "new" / "dir" / "frame.glb"
        export_glb(build_frame_mesh(rims_mask()), path)
        self.assertGreater(path.stat().st_size, 0)

    def test_output_must_be_named_glb(self):
        path = self.dir / "frame.png"
        with self.assertRaisesRegex(ValueError, r"\.glb"):
            export_glb(build_frame_mesh(rims_mask()), path)
        self.assertFalse(path.exists())

    def test_empty_mesh_is_not_exported(self):
        path = self.dir / "frame.glb"
        with self.assertRaises(MeshBuildError):
            export_glb(trimesh.Trimesh(), path)
        self.assertFalse(path.exists())


class DemoImageTests(WithTempDir):
    def test_demo_image_is_deterministic(self):
        np.testing.assert_array_equal(make_demo_image(), make_demo_image())

    def test_demo_image_segments_into_a_frame_with_two_openings(self):
        polygons = mask_to_polygons(segment_glasses(make_demo_image()))
        self.assertEqual(len(polygons), 1)
        self.assertEqual(len(polygons[0].interiors), 2)

    def test_written_demo_image_round_trips(self):
        path = write_demo_image(self.dir / "sub" / "demo.png")
        np.testing.assert_array_equal(load_image(path), make_demo_image())


class ConvertImageTests(WithTempDir):
    def convert(self, source, **kwargs):
        return convert_image(source, self.dir / "out.glb", **kwargs)

    def test_png_on_white_converts(self):
        result = self.convert(self.encoded_file("a.png", white_image()))
        self.assertTrue(result.mesh.is_volume)
        self.assertEqual(result.openings, 2)
        self.assertGreater(result.output_path.stat().st_size, 0)

    def test_transparent_png_converts(self):
        result = self.convert(self.encoded_file("a.png", transparent_image()))
        self.assertEqual(result.openings, 2)

    def test_jpg_converts(self):
        result = self.convert(self.encoded_file("a.jpg", white_image()))
        self.assertEqual(result.openings, 2)

    def test_dimensions_reach_the_mesh(self):
        result = self.convert(self.encoded_file("a.png", white_image()),
                              dimensions=FrameDimensions(frame_width=12.0, thickness=0.6))
        self.assertAlmostEqual(result.mesh.extents[0], 12.0, places=6)
        self.assertAlmostEqual(result.mesh.extents[2], 0.6, places=6)

    def test_failed_conversion_writes_no_output(self):
        blank = self.encoded_file("blank.png", np.full((50, 50, 3), 255, np.uint8))
        with self.assertRaises(SegmentationError):
            self.convert(blank)
        self.assertFalse((self.dir / "out.glb").exists())


class CommandLineTests(WithTempDir):
    def run_main(self, *argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                code = main([str(arg) for arg in argv])
            except SystemExit as stop:  # argparse usage errors
                code = stop.code
        return code, stdout.getvalue(), stderr.getvalue()

    def test_demo_writes_sample_image_and_nonempty_glb(self):
        code, stdout, stderr = self.run_main("--demo", "--output", self.dir / "demo.glb")
        self.assertEqual(code, 0, stderr)
        self.assertGreater((self.dir / "demo.glb").stat().st_size, 0)
        self.assertEqual(load_image(self.dir / "demo.png").shape, make_demo_image().shape)
        self.assertTrue(trimesh.load_mesh(self.dir / "demo.glb").is_volume)
        self.assertIn("openings: 2", stdout)

    def test_output_defaults_to_the_image_path_with_glb_suffix(self):
        source = self.encoded_file("product.png", white_image())
        code, _, stderr = self.run_main(source)
        self.assertEqual(code, 0, stderr)
        self.assertTrue((self.dir / "product.glb").exists())

    def test_frame_width_and_thickness_flags_apply(self):
        source = self.encoded_file("a.png", white_image())
        out = self.dir / "a.glb"
        code, _, stderr = self.run_main(source, "-o", out,
                                        "--frame-width", 10, "--thickness", 0.5)
        self.assertEqual(code, 0, stderr)
        extents = trimesh.load_mesh(out).extents
        self.assertAlmostEqual(extents[0], 0.10, places=5)
        self.assertAlmostEqual(extents[2], 0.005, places=5)

    def test_empty_image_file_fails_cleanly(self):
        source = self.dir / "empty.png"
        source.write_bytes(b"")
        code, _, stderr = self.run_main(source)
        self.assertEqual(code, 1)
        self.assertIn("error:", stderr)
        self.assertIn("empty", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_invalid_image_file_fails_cleanly(self):
        source = self.dir / "garbage.jpg"
        source.write_bytes(b"not a jpeg")
        code, _, stderr = self.run_main(source)
        self.assertEqual(code, 1)
        self.assertIn("error:", stderr)

    def test_blank_image_fails_cleanly(self):
        source = self.encoded_file("blank.png", np.full((50, 50, 3), 255, np.uint8))
        code, _, stderr = self.run_main(source)
        self.assertEqual(code, 1)
        self.assertIn("No glasses", stderr)

    def test_invalid_dimension_fails_cleanly(self):
        source = self.encoded_file("a.png", white_image())
        code, _, stderr = self.run_main(source, "--thickness", 0)
        self.assertEqual(code, 1)
        self.assertIn("error:", stderr)

    def test_non_glb_output_fails_cleanly(self):
        source = self.encoded_file("a.png", white_image())
        code, _, stderr = self.run_main(source, "-o", self.dir / "a.obj")
        self.assertEqual(code, 1)
        self.assertIn(".glb", stderr)

    def test_unwritable_output_fails_cleanly(self):
        source = self.encoded_file("a.png", white_image())
        (self.dir / "taken.glb").mkdir()
        code, _, stderr = self.run_main(source, "-o", self.dir / "taken.glb")
        self.assertEqual(code, 1)
        self.assertIn("error:", stderr)

    def test_warns_when_no_openings_are_found(self):
        image = np.full((200, 400, 3), 255, np.uint8)
        cv2.rectangle(image, *LEFT_RIM, FRAME_COLOR, -1)  # opaque, sunglasses-like lens
        code, _, stderr = self.run_main(self.encoded_file("a.png", image))
        self.assertEqual(code, 0)
        self.assertIn("warning:", stderr)

    def test_image_and_demo_together_are_a_usage_error(self):
        code, _, _ = self.run_main("x.png", "--demo")
        self.assertEqual(code, 2)

    def test_no_arguments_are_a_usage_error(self):
        code, _, _ = self.run_main()
        self.assertEqual(code, 2)


class StandaloneScriptTests(WithTempDir):
    """Run in a real interpreter: covers the entry points and import isolation."""

    def run_python(self, *args):
        return subprocess.run([sys.executable, *map(str, args)], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=120)

    def test_script_runs_directly_by_path(self):
        out = self.dir / "demo.glb"
        done = self.run_python("image_to_3d/mirror_image_to_3d.py", "--demo", "-o", out)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertGreater(out.stat().st_size, 0)

    def test_script_runs_as_a_module(self):
        out = self.dir / "demo.glb"
        done = self.run_python("-m", "image_to_3d.mirror_image_to_3d", "--demo", "-o", out)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertGreater(out.stat().st_size, 0)

    def test_package_does_not_load_the_ar_pipeline(self):
        code = ("import json, sys, image_to_3d; "
                "print(json.dumps(sorted(m for m in ('mediapipe', 'main', 'head_pose', "
                "'glasses_3d', 'occlusion', 'accessory_transform') if m in sys.modules)))")
        done = self.run_python("-c", code)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout), [])


if __name__ == "__main__":
    unittest.main()
