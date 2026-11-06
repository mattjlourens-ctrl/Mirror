"""Mirror's HTTP API.

Run locally with:  uv run uvicorn backend.app:app --port 8000

POST /api/jobs/from-image       multipart "image" -> 202 job
POST /api/sources/from-url      {"url"} -> front-view photos found; a job starts when there is exactly one
POST /api/jobs/from-candidate   {"candidate_id"} -> 202 job
GET  /api/jobs/{id}             job status; includes model_url once succeeded
GET  /api/images/{id}           a stored source or candidate photo
GET  /api/models/{id}           a generated model (.glb), for the browser renderer only
"""
import logging
import re
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.config import LOCAL_ORIGIN_PATTERN, Settings
from backend.errors import Busy, FetchFailed, NoProductImages, NotFound, ServiceError
from backend.images import assess_front_view, validate_image
from backend.jobs import JobQueue
from backend.product_page import check_robots, find_product_images
from backend.safe_fetch import SafeFetcher
from backend.storage import TempStore

log = logging.getLogger(__name__)

ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
HTML_TYPES = ("text/html", "application/xhtml+xml")
MAX_CONCURRENT_PAGE_LOOKUPS = 4


class UrlRequest(BaseModel):
    url: str


class CandidateRequest(BaseModel):
    candidate_id: str


def job_json(job):
    return {
        "id": job.id,
        "status": job.status,
        "source_image_url": f"/api/images/{job.source_image_id}",
        "model_url": f"/api/models/{job.model_id}" if job.model_id else None,
        "parts": list(job.parts),
        "fit": job.fit,
        "error": ({"code": job.error_code, "message": job.error_message} if job.error_code else None),
    }


class BodySizeLimit:
    """Reject request bodies over a limit while they stream in, not after."""

    def __init__(self, app, max_bytes):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        declared = headers.get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            return await _too_large(send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await _too_large(send)


class _BodyTooLarge(Exception):
    pass


async def _too_large(send):
    response = JSONResponse({"detail": {"code": "image_too_large", "message": "The upload is too large."}},
                            status_code=413)
    await send({"type": "http.response.start", "status": 413, "headers": response.raw_headers})
    await send({"type": "http.response.body", "body": response.body})


def create_app(settings=None, fetcher=None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        app.state.store = TempStore(settings.storage_dir, settings.storage_ttl_seconds, settings.max_stored_items)
        app.state.jobs = JobQueue(app.state.store, settings.workers, settings.max_pending_jobs)
        app.state.fetcher = fetcher or SafeFetcher(settings.user_agent)
        app.state.page_lookups = threading.BoundedSemaphore(MAX_CONCURRENT_PAGE_LOOKUPS)
        try:
            yield
        finally:
            app.state.jobs.shutdown()
            if fetcher is None:
                app.state.fetcher.close()
            app.state.store.close()

    app = FastAPI(title="Mirror API", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins),
                       allow_origin_regex=LOCAL_ORIGIN_PATTERN if settings.allow_local_origins else None,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    # Multipart framing adds a little to the image itself.
    app.add_middleware(BodySizeLimit, max_bytes=settings.max_image_bytes + 64 * 1024)

    @app.exception_handler(ServiceError)
    async def service_error(_request, error):
        return JSONResponse({"detail": {"code": error.code, "message": error.message}}, status_code=error.status)

    def stored(request, kind, file_id):
        item = ID_PATTERN.match(file_id) and request.app.state.store.get(kind, file_id)
        if not item:
            raise NotFound("That file has expired or doesn't exist.")
        return item

    def start_job(request, image_id):
        return request.app.state.jobs.submit(image_id)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/jobs/from-image", status_code=202)
    def job_from_image(request: Request, image: UploadFile = File(...)):
        data = image.file.read(settings.max_image_bytes + 1)
        source = validate_image(data, settings.max_image_bytes, settings.max_image_pixels)
        image_id = request.app.state.store.put_image(source)
        return job_json(start_job(request, image_id))

    @app.post("/api/sources/from-url")
    def source_from_url(request: Request, body: UrlRequest):
        state = request.app.state
        if not state.page_lookups.acquire(blocking=False):
            raise Busy("Mirror is busy looking up other pages. Please try again in a moment.")
        try:
            candidates = _find_candidates(state, settings, body.url)
        finally:
            state.page_lookups.release()

        job = start_job(request, candidates[0]["id"]) if len(candidates) == 1 else None
        return {
            "status": "processing" if job else "choose",
            "job": job_json(job) if job else None,
            "candidates": candidates,
        }

    @app.post("/api/jobs/from-candidate", status_code=202)
    def job_from_candidate(request: Request, body: CandidateRequest):
        stored(request, "image", body.candidate_id)
        return job_json(start_job(request, body.candidate_id))

    @app.get("/api/jobs/{job_id}")
    def job_status(request: Request, job_id: str):
        job = ID_PATTERN.match(job_id) and request.app.state.jobs.get(job_id)
        if not job:
            raise NotFound("That job has expired or doesn't exist.")
        return job_json(job)

    @app.get("/api/images/{image_id}")
    def get_image(request: Request, image_id: str):
        item = stored(request, "image", image_id)
        return FileResponse(item.path, media_type=item.media_type,
                            headers={"Cache-Control": "private, max-age=3600",
                                     "X-Content-Type-Options": "nosniff"})

    @app.get("/api/models/{model_id}")
    def get_model(request: Request, model_id: str):
        item = stored(request, "model", model_id)
        return FileResponse(item.path, media_type=item.media_type,
                            headers={"Cache-Control": "private, max-age=3600",
                                     "X-Content-Type-Options": "nosniff"})

    return app


def _find_candidates(state, settings, url):
    """The front-view product photos on a page (or the image a URL points to), best first.

    Photos that fail the front-view check are dropped rather than offered, since
    the generator can't use them; the person is pointed to uploading instead.
    """
    fetcher = state.fetcher
    check_robots(fetcher, url, settings.user_agent)
    page = fetcher.get(url, max_bytes=settings.max_page_bytes,
                       accept="text/html,application/xhtml+xml,image/png,image/jpeg,image/webp;q=0.9")
    if page.status in (401, 403, 429):
        raise FetchFailed("The store blocked automatic access to this page. "
                          "Save the product photo and upload it instead.", code="access_restricted")
    if page.status == 404:
        raise FetchFailed("That page wasn't found. Check the link and try again.", code="page_not_found")
    if page.status != 200:
        raise FetchFailed("The store's page couldn't be loaded. Try again later, or upload a photo.",
                          code="page_unavailable")

    if page.content_type.startswith("image/"):
        references = [page.url]
        preloaded = {page.url: page.body}
    elif page.content_type in HTML_TYPES or not page.content_type:
        html = page.body.decode(page.charset or "utf-8", "replace")
        references = [ref.url for ref in find_product_images(html, page.url, settings.max_candidates + 4)]
        preloaded = {}
    else:
        raise FetchFailed("That link isn't a product page or a photo.", code="not_a_page")

    candidates, rejected = [], 0
    for image_url in references:
        if len(candidates) >= settings.max_candidates:
            break
        try:
            data = preloaded.get(image_url)
            if data is None:
                result = fetcher.get(image_url, max_bytes=settings.max_image_bytes,
                                     accept="image/png,image/jpeg,image/webp")
                if result.status != 200:
                    continue
                data = result.body
            source = validate_image(data, settings.max_image_bytes, settings.max_image_pixels, allow_webp=True)
        except ServiceError as error:
            log.info("skipping candidate %s: %s", image_url, error.message)
            continue
        assessment = assess_front_view(source)
        if not assessment.suitable:
            rejected += 1
            continue
        image_id = state.store.put_image(source)
        candidates.append({"id": image_id, "image_url": f"/api/images/{image_id}",
                           "width": source.width, "height": source.height,
                           "score": assessment.score, "source_url": image_url})

    if not candidates and rejected:
        raise NoProductImages("Mirror found photos on that page, but none shows the glasses from the front "
                              "on a plain background. Save a front-view photo and upload it instead.",
                              code="no_front_view")
    if not candidates:
        raise NoProductImages("Mirror couldn't find a product photo on that page. "
                              "Save the photo and upload it instead.")
    # Best first; page order breaks ties, so results are repeatable.
    order = {candidate["id"]: index for index, candidate in enumerate(candidates)}
    candidates.sort(key=lambda candidate: (-candidate["score"], order[candidate["id"]]))
    return candidates


app = create_app()
