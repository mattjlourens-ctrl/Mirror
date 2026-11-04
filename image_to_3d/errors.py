"""Expected failures of the image-to-mesh pipeline."""


class ImageToMeshError(Exception):
    """Base class; the command line reports these without a traceback."""


class ImageLoadError(ImageToMeshError):
    """The file is missing, empty, corrupt, or not a PNG/JPEG."""


class SegmentationError(ImageToMeshError):
    """No frame could be separated from the image background."""


class MeshBuildError(ImageToMeshError):
    """The silhouette could not be turned into a valid, exportable mesh."""
