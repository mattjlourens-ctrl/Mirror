"""Backend settings, read from MIRROR_* environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path

MB = 1024 * 1024


def _int(name, default):
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


# The website's dev server, by either local name.
DEFAULT_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")
# Any port on this machine, so a dev server that moved to :3001 still works.
# Matched against the whole Origin header.
LOCAL_ORIGIN_PATTERN = r"http://(localhost|127\.0\.0\.1)(:\d{1,5})?"


@dataclass(frozen=True)
class Settings:
    allowed_origins: tuple = DEFAULT_ORIGINS
    allow_local_origins: bool = True  # also allow LOCAL_ORIGIN_PATTERN; turn off in production
    max_image_bytes: int = 10 * MB    # uploads and downloaded product photos
    max_page_bytes: int = 3 * MB      # product page HTML
    max_image_pixels: int = 40_000_000
    storage_dir: Path | None = None   # parent of the per-process temp folder; None = system temp
    storage_ttl_seconds: int = 3600   # images and models are deleted after this
    max_stored_items: int = 300
    max_pending_jobs: int = 4         # queued plus running; further submissions get 503
    workers: int = 1
    max_candidates: int = 6           # product photos downloaded and checked per page
    user_agent: str = "MirrorBot/0.1 (virtual try-on; respects robots.txt)"

    @classmethod
    def from_env(cls):
        origins = os.environ.get("MIRROR_ALLOWED_ORIGINS", ",".join(DEFAULT_ORIGINS))
        storage_dir = os.environ.get("MIRROR_STORAGE_DIR", "").strip()
        return cls(
            allowed_origins=tuple(origin.strip() for origin in origins.split(",") if origin.strip()),
            allow_local_origins=os.environ.get("MIRROR_ALLOW_LOCAL_ORIGINS", "1").strip() != "0",
            max_image_bytes=_int("MIRROR_MAX_IMAGE_MB", 10) * MB,
            storage_dir=Path(storage_dir) if storage_dir else None,
            storage_ttl_seconds=_int("MIRROR_STORAGE_TTL_SECONDS", 3600),
            max_pending_jobs=_int("MIRROR_MAX_PENDING_JOBS", 4),
            workers=_int("MIRROR_WORKERS", 1),
        )
