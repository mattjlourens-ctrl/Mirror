"""The full image-to-GLB conversion."""
from dataclasses import dataclass
from pathlib import Path

import trimesh

from image_to_3d.mesh import FrameDimensions, export_glb, mask_to_polygons, polygons_to_mesh
from image_to_3d.segmentation import SegmentationConfig, load_image, segment_glasses


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
