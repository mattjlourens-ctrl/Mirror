"""Convert a front-facing image of glasses into an approximate 3D model.

The front outline comes from the image; depth, temples, hinges and nose pads are
generated from parameters. Everything is deterministic geometry, with no learned
models or external services.

Independent of the AR face-tracking pipeline: nothing here imports it, and it
imports nothing from here.
"""
from image_to_3d.assembly import (GlassesModel, GlassesParameters, assemble_glasses,
                                  export_model_glb)
from image_to_3d.demo import make_demo_image, write_demo_image
from image_to_3d.errors import (ImageLoadError, ImageToMeshError, MeshBuildError,
                                SegmentationError)
from image_to_3d.mesh import (FrameDimensions, build_frame_mesh, export_glb, mask_to_polygons,
                              polygons_to_mesh)
from image_to_3d.pipeline import (ConversionResult, GlassesConversionResult, convert_image,
                                  convert_image_to_glasses)
from image_to_3d.segmentation import (SegmentationConfig, load_image, remove_opening_intrusions,
                                      segment_glasses)
from image_to_3d.temples import TempleParameters
