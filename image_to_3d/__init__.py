"""Convert a front-facing image of glasses into an approximate shallow 3D mesh.

Independent of the AR face-tracking pipeline: nothing here imports it, and it
imports nothing from here.
"""
from image_to_3d.demo import make_demo_image, write_demo_image
from image_to_3d.errors import (ImageLoadError, ImageToMeshError, MeshBuildError,
                                SegmentationError)
from image_to_3d.mesh import (FrameDimensions, build_frame_mesh, export_glb, mask_to_polygons,
                              polygons_to_mesh)
from image_to_3d.pipeline import ConversionResult, convert_image
from image_to_3d.segmentation import SegmentationConfig, load_image, segment_glasses
