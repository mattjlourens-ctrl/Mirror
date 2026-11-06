"""A bounded queue that turns stored source photos into stored glasses models."""
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from backend.errors import Busy
from shapely.geometry import Polygon

from glb_model import ModelLoadError, load_glasses_model
from image_to_3d.assembly import GlassesParameters
from image_to_3d.errors import ImageLoadError, MeshBuildError, SegmentationError
from image_to_3d.mesh import mask_to_polygons, scale_polygons_to_cm
from image_to_3d.pipeline import convert_image_to_glasses

log = logging.getLogger(__name__)

FAILURES = {
    SegmentationError: ("no_glasses_found",
                        "Mirror couldn't separate the glasses from the background. "
                        "Try a front-facing photo on a plain white background."),
    ImageLoadError: ("unreadable_image", "The photo couldn't be read. Try another image."),
    MeshBuildError: ("reconstruction_failed",
                     "Mirror couldn't build a 3D model from this photo. "
                     "Try a sharper, front-facing photo on a plain white background."),
    ModelLoadError: ("reconstruction_failed",
                     "The generated model didn't pass Mirror's checks. Try another photo."),
}


def reconstruct(source_path, output_path):
    """Generate a textured model with the image_to_3d pipeline, then load it the
    way the live try-on does to prove it's usable.

    Returns (part names, fit hints for the browser try-on).
    """
    result = convert_image_to_glasses(source_path, output_path)
    parts = load_glasses_model(output_path).part_names
    # The same outline, in the model's centimetres (the front frame is built from it).
    polygons_cm = scale_polygons_to_cm(mask_to_polygons(result.mask), GlassesParameters().frame_width)
    return parts, fit_hints(polygons_cm)


def fit_hints(polygons_cm):
    """Centres of the two largest lens openings (model cm, X right, Y up), left first.

    The browser uses them to put the lenses in front of the eyes; None when the
    frame doesn't have two openings.
    """
    openings = [Polygon(ring) for polygon in polygons_cm for ring in polygon.interiors]
    openings = sorted(openings, key=lambda opening: opening.area, reverse=True)[:2]
    if len(openings) < 2:
        return None
    centres = sorted((round(opening.centroid.x, 3), round(opening.centroid.y, 3)) for opening in openings)
    return {"lens_centers_cm": [list(centre) for centre in centres]}


@dataclass(frozen=True)
class Job:
    id: str
    status: str                 # queued, processing, succeeded or failed
    source_image_id: str
    model_id: str | None = None
    parts: tuple = ()
    fit: dict | None = None
    error_code: str | None = None
    error_message: str | None = None
    created: float = 0.0


class JobQueue:
    def __init__(self, store, workers=1, max_pending=4, reconstruct=reconstruct, clock=time.monotonic):
        self.store = store
        self.max_pending = max_pending
        self.reconstruct = reconstruct
        self.clock = clock
        self._jobs = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mirror-job")

    def submit(self, source_image_id):
        with self._lock:
            self._forget_expired()
            pending = sum(job.status in ("queued", "processing") for job in self._jobs.values())
            if pending >= self.max_pending:
                raise Busy("Mirror is busy preparing other glasses. Please try again in a moment.")
            job = Job(uuid.uuid4().hex, "queued", source_image_id, created=self.clock())
            self._jobs[job.id] = job
        self._executor.submit(self._run, job.id)
        return job

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def _update(self, job_id, **changes):
        with self._lock:
            self._jobs[job_id] = replace(self._jobs[job_id], **changes)

    def _forget_expired(self):
        for job_id in [job.id for job in self._jobs.values()
                       if job.status in ("succeeded", "failed")
                       and self.clock() - job.created > self.store.ttl]:
            del self._jobs[job_id]

    def _run(self, job_id):
        self._update(job_id, status="processing")
        source = self.store.get("image", self.get(job_id).source_image_id)
        if source is None:
            self._update(job_id, status="failed", error_code="source_expired",
                         error_message="The photo expired before it could be processed. Please add it again.")
            return
        try:
            with self.store.work_dir() as work:
                output = work / "glasses.glb"
                parts, fit = self.reconstruct(source.path, output)
                model_id = self.store.put_model(output)
        except tuple(FAILURES) as error:
            code, message = next(FAILURES[kind] for kind in FAILURES if isinstance(error, kind))
            log.info("job %s failed: %s", job_id, error)
            self._update(job_id, status="failed", error_code=code, error_message=message)
        except Exception:
            log.exception("job %s crashed", job_id)
            self._update(job_id, status="failed", error_code="internal_error",
                         error_message="Something went wrong while preparing your glasses. Please try again.")
        else:
            self._update(job_id, status="succeeded", model_id=model_id, parts=tuple(parts), fit=fit)

    def shutdown(self):
        """Cancel queued jobs and wait for the running one, so storage can be removed safely."""
        self._executor.shutdown(wait=True, cancel_futures=True)
