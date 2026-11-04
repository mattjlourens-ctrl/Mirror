"""Command line: convert a glasses image to a shallow GLB mesh.

    uv run python image_to_3d/mirror_image_to_3d.py glasses.png -o glasses.glb
    uv run python image_to_3d/mirror_image_to_3d.py --demo

The result is an approximate extrusion of the front silhouette, not a physically
accurate reconstruction of the glasses.
"""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):  # run by path: make the package importable
    sys.path[0] = str(Path(__file__).resolve().parent.parent)

from image_to_3d.demo import write_demo_image
from image_to_3d.errors import ImageToMeshError
from image_to_3d.mesh import FrameDimensions
from image_to_3d.pipeline import convert_image
from image_to_3d.segmentation import SegmentationConfig

DEFAULT_DEMO_OUTPUT = Path(__file__).resolve().parent / "output" / "demo_glasses.glb"


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Convert a front-facing PNG or JPG of glasses on a white or transparent "
                    "background into a shallow 3D mesh (.glb).")
    parser.add_argument("image", nargs="?", type=Path, help="PNG or JPG image of glasses")
    parser.add_argument("--demo", action="store_true",
                        help="generate a sample glasses image and convert it")
    parser.add_argument("-o", "--output", type=Path,
                        help="output .glb path (default: the image path with a .glb suffix; "
                             "with --demo, a sample image is written next to it)")
    parser.add_argument("--frame-width", type=float, default=FrameDimensions.frame_width,
                        help="frame width, outer edge to outer edge, in cm (default: %(default)s)")
    parser.add_argument("--thickness", type=float, default=FrameDimensions.thickness,
                        help="front to back depth in cm (default: %(default)s)")
    parser.add_argument("--white-threshold", type=int, default=SegmentationConfig.white_threshold,
                        help="pixels with every channel at or above this are background "
                             "(default: %(default)s)")
    parser.add_argument("--alpha-threshold", type=int, default=SegmentationConfig.alpha_threshold,
                        help="pixels with alpha below this are background (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.demo == (args.image is not None):
        parser.error("give either an image or --demo")
    return args


def run(args):
    dimensions = FrameDimensions(frame_width=args.frame_width, thickness=args.thickness)
    segmentation = SegmentationConfig(white_threshold=args.white_threshold,
                                      alpha_threshold=args.alpha_threshold)
    if args.demo:
        output = args.output or DEFAULT_DEMO_OUTPUT
        source = write_demo_image(output.with_suffix(".png"))
        print(f"Sample image: {source}")
    else:
        source = args.image
        output = args.output or source.with_suffix(".glb")

    result = convert_image(source, output, dimensions, segmentation)

    mesh = result.mesh
    width, height, depth = mesh.extents
    print(f"Wrote {result.output_path} ({result.output_path.stat().st_size} bytes)")
    print(f"  vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}, "
          f"valid volume: {mesh.is_volume}, openings: {result.openings}")
    print(f"  size: {width:.2f} x {height:.2f} x {depth:.2f} cm")
    print("  Approximate shallow silhouette extrusion, not an accurate reconstruction.")
    if result.openings == 0:
        print("warning: no enclosed openings found; lenses may have been treated as solid "
              "(try a lower --white-threshold or --alpha-threshold)", file=sys.stderr)


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (ImageToMeshError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
