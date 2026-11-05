"""Command line: convert a glasses image to a 3D mesh (.glb). All lengths are in cm.

    uv run python image_to_3d/mirror_image_to_3d.py glasses.png -o glasses.glb
    uv run python image_to_3d/mirror_image_to_3d.py glasses.png --complete --nose-pads
    uv run python image_to_3d/mirror_image_to_3d.py --demo --complete

By default this writes the shallow front frame alone. --complete adds a curved
front, temples and hinges. Only the front outline comes from the image; everything
else is estimated from parameters, so the result is an approximation and not a
physically accurate reconstruction of the glasses.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import sys

if __package__ in (None, ""):  # run by path: make the package importable
    sys.path[0] = str(Path(__file__).resolve().parent.parent)

from image_to_3d.assembly import GlassesParameters
from image_to_3d.demo import write_demo_image
from image_to_3d.errors import ImageToMeshError
from image_to_3d.mesh import FrameDimensions
from image_to_3d.pipeline import DEFAULT_OPENING_CLEANUP, convert_image, convert_image_to_glasses
from image_to_3d.segmentation import SegmentationConfig
from image_to_3d.temples import TempleParameters

DEFAULT_DEMO_OUTPUT = Path(__file__).resolve().parent / "output" / "demo_glasses.glb"

# (flag suffix and TempleParameters field, help text)
TEMPLE_OPTIONS = (
    ("length", "length along the arm, hinge to tip, in cm"),
    ("width", "arm height in cm"),
    ("thickness", "arm depth across its side, in cm"),
    ("curvature", "how far the tip curves in from the ear bend, in cm"),
    ("splay", "outward offset at the ear bend, to clear the head, in cm"),
    ("ear_drop", "how far the tip falls below the hinge, in cm"),
)


def _add_complete_options(parser):
    group = parser.add_argument_group(
        "complete model (need --complete)",
        "These can't be measured from a front view, so the defaults are typical values.")
    group.add_argument("--complete", action="store_true",
                       help="build the curved front, temples and hinges")
    group.add_argument("--frame-curvature", type=float,
                       help="how far the outer edges sit behind the centre, in cm "
                            f"(default: {GlassesParameters.frame_curvature})")
    group.add_argument("--nose-pads", action="store_true", help="add separate nose pads")
    group.add_argument("--opening-cleanup", type=float,
                       help="remove strips narrower than this fraction of the frame width from "
                            "lens openings, such as temples seen through a lens; 0 disables "
                            f"(default: {DEFAULT_OPENING_CLEANUP})")
    for suffix, description in TEMPLE_OPTIONS:
        flag = suffix.replace("_", "-")
        default = getattr(TempleParameters, suffix)
        group.add_argument(f"--temple-{flag}", type=float,
                           help=f"both arms: {description} (default: {default})")
        group.add_argument(f"--left-temple-{flag}", type=float, help="screen-left arm only")
        group.add_argument(f"--right-temple-{flag}", type=float, help="screen-right arm only")


def _complete_only_flags(args):
    """Options given that only mean something with --complete."""
    flags = [f"--{name.replace('_', '-')}" for name, value in vars(args).items()
             if value is not None and (name in ("frame_curvature", "opening_cleanup")
                                       or "temple_" in name)]
    return flags + (["--nose-pads"] if args.nose_pads else [])


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Convert a front-facing PNG or JPG of glasses on a white or transparent "
                    "background into a 3D mesh (.glb).")
    parser.add_argument("image", nargs="?", type=Path, help="PNG or JPG image of glasses")
    parser.add_argument("--demo", action="store_true",
                        help="generate a sample glasses image and convert it")
    parser.add_argument("-o", "--output", type=Path,
                        help="output .glb path (default: the image path with a .glb suffix; "
                             "with --demo, a sample image is written next to it)")
    parser.add_argument("--frame-width", type=float, default=FrameDimensions.frame_width,
                        help="frame width, outer edge to outer edge, in cm (default: %(default)s)")
    parser.add_argument("--frame-thickness", "--thickness", dest="thickness", type=float,
                        default=FrameDimensions.thickness,
                        help="front to back depth in cm (default: %(default)s)")
    parser.add_argument("--white-threshold", type=int,
                        help="pixels with every channel at or above this are background "
                             f"(default: {SegmentationConfig.white_threshold}; with --complete, "
                             "halfway between the frame and white, at most that)")
    parser.add_argument("--alpha-threshold", type=int, default=SegmentationConfig.alpha_threshold,
                        help="pixels with alpha below this are background (default: %(default)s)")
    _add_complete_options(parser)
    args = parser.parse_args(argv)
    if args.demo == (args.image is not None):
        parser.error("give either an image or --demo")
    if not args.complete and (flags := _complete_only_flags(args)):
        parser.error(f"{', '.join(flags)} only apply with --complete")
    return args


def _temple_parameters(args, side):
    """Shared --temple-* values, overridden by this side's own."""
    overrides = {}
    for suffix, _ in TEMPLE_OPTIONS:
        for name in (f"temple_{suffix}", f"{side}_temple_{suffix}"):
            if (value := getattr(args, name)) is not None:
                overrides[suffix] = value
    return replace(TempleParameters(), **overrides)


def _glasses_parameters(args):
    params = GlassesParameters(frame_width=args.frame_width, frame_thickness=args.thickness,
                               left_temple=_temple_parameters(args, "left"),
                               right_temple=_temple_parameters(args, "right"),
                               nose_pads=args.nose_pads)
    if args.frame_curvature is not None:
        params = replace(params, frame_curvature=args.frame_curvature)
    return params


def _warn_if_no_openings(openings):
    if openings == 0:
        print("warning: no enclosed openings found; lenses may have been treated as solid "
              "(try a lower --white-threshold or --alpha-threshold)", file=sys.stderr)


def _run_complete(args, source, output):
    # An explicit threshold is taken literally; the default adapts to the frame.
    adaptive = args.white_threshold is None
    threshold = SegmentationConfig.white_threshold if adaptive else args.white_threshold
    segmentation = SegmentationConfig(white_threshold=threshold,
                                      alpha_threshold=args.alpha_threshold,
                                      adaptive_white=adaptive)
    cleanup = DEFAULT_OPENING_CLEANUP if args.opening_cleanup is None else args.opening_cleanup
    result = convert_image_to_glasses(source, output, _glasses_parameters(args), segmentation,
                                      cleanup)

    model = result.model
    print(f"Wrote {result.output_path} ({result.output_path.stat().st_size} bytes)")
    for component in model.report.components:
        print(f"  {component.name}: {component.vertices} vertices, {component.faces} faces")
    print(f"  validation passed; openings: {model.openings}, nose support: {model.nose_support}")
    print("  The front outline comes from the image. Depth, temples, hinges and nose pads are "
          "estimated, not reconstructed.")
    _warn_if_no_openings(model.openings)


def _run_flat(args, source, output):
    dimensions = FrameDimensions(frame_width=args.frame_width, thickness=args.thickness)
    threshold = (SegmentationConfig.white_threshold if args.white_threshold is None
                 else args.white_threshold)
    segmentation = SegmentationConfig(white_threshold=threshold,
                                      alpha_threshold=args.alpha_threshold)
    result = convert_image(source, output, dimensions, segmentation)

    mesh = result.mesh
    width, height, depth = mesh.extents
    print(f"Wrote {result.output_path} ({result.output_path.stat().st_size} bytes)")
    print(f"  vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)}, "
          f"valid volume: {mesh.is_volume}, openings: {result.openings}")
    print(f"  size: {width:.2f} x {height:.2f} x {depth:.2f} cm")
    print("  Approximate shallow silhouette extrusion, not an accurate reconstruction.")
    _warn_if_no_openings(result.openings)


def run(args):
    if args.demo:
        output = args.output or DEFAULT_DEMO_OUTPUT
        source = write_demo_image(output.with_suffix(".png"))
        print(f"Sample image: {source}")
    else:
        source = args.image
        output = args.output or source.with_suffix(".glb")
    (_run_complete if args.complete else _run_flat)(args, source, output)


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
