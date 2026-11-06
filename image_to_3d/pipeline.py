"""The full image-to-GLB conversions: the flat front only, or a complete model."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from image_to_3d.appearance import FrameAppearance, MaterialSettings, extract_appearance
from image_to_3d.assembly import GlassesModel, GlassesParameters, assemble_glasses, export_model_glb
from image_to_3d.mesh import FrameDimensions, export_glb, mask_to_polygons, polygons_to_mesh
from image_to_3d.segmentation import (SegmentationConfig, load_image, remove_opening_intrusions,
                                      segment_glasses)

# Temples behind the lenses can show through them in a product photo. Their width
# in images measured so far is under 3% of the frame's width.
DEFAULT_OPENING_CLEANUP = 0.04


@dataclass(frozen=True)
class ConversionResult:
    mesh: trimesh.Trimesh  # in centimetres
    openings: int          # enclosed gaps preserved in the silhouette
    output_path: Path


def convert_image(source, output, dimensions=FrameDimensions(),
                  segmentation=SegmentationConfig()):
    """Convert a glasses image to a shallow GLB mesh; writes nothing on failure."""
    dimensions.validate()
    mask = segment_glasses(load_image(source), segmentation)
    polygons = mask_to_polygons(mask)
    mesh = polygons_to_mesh(polygons, dimensions)
    return ConversionResult(mesh=mesh,
                            openings=sum(len(polygon.interiors) for polygon in polygons),
                            output_path=export_glb(mesh, output))


@dataclass(frozen=True)
class GlassesConversionResult:
    model: GlassesModel
    mask: np.ndarray  # the cleaned silhouette the frame was traced from
    output_path: Path
    appearance: FrameAppearance = None  # None when texturing was turned off


def convert_image_to_glasses(source, output, params=GlassesParameters(),
                             segmentation=SegmentationConfig(adaptive_white=True),
                             opening_cleanup=DEFAULT_OPENING_CLEANUP, texture=True,
                             materials=MaterialSettings()):
    """Convert a glasses image to a complete model; writes nothing on failure.

    texture -- give the front the photo's appearance and the other parts colours
               estimated from it; False keeps the fixed default materials
    """
    params.validate()
    materials.validate()
    image = load_image(source)
    mask = remove_opening_intrusions(segment_glasses(image, segmentation), opening_cleanup)
    polygons = mask_to_polygons(mask)
    model = assemble_glasses(polygons, params)
    appearance = (extract_appearance(image, mask, polygons, params.frame_width, materials)
                  if texture else None)
    return GlassesConversionResult(model=model, mask=mask, appearance=appearance,
                                   output_path=export_model_glb(model, output, appearance))
