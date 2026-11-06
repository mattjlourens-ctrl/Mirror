"""Short-lived storage for source photos and generated models.

Everything lives in one private temporary folder per process, is deleted after
a time limit or when the store is over capacity, and the whole folder is removed
on shutdown. Files are found by random IDs only; no client-supplied name ever
becomes a path.
"""
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoredFile:
    id: str
    kind: str          # "image" or "model"
    path: Path
    media_type: str
    created: float


class TempStore:
    def __init__(self, parent=None, ttl_seconds=3600, max_items=300, clock=time.monotonic):
        self.root = Path(tempfile.mkdtemp(prefix="mirror-", dir=parent))
        for name in ("images", "models", "work"):
            (self.root / name).mkdir()
        self.ttl = ttl_seconds
        self.max_items = max_items
        self.clock = clock
        self._items = {}
        self._lock = threading.Lock()

    def _put(self, kind, write, suffix, media_type):
        self.sweep()
        file_id = uuid.uuid4().hex
        path = self.root / f"{kind}s" / f"{file_id}{suffix}"
        write(path)
        with self._lock:
            self._items[file_id] = StoredFile(file_id, kind, path, media_type, self.clock())
        return file_id

    def put_image(self, image):
        return self._put("image", lambda path: path.write_bytes(image.data), image.extension, image.media_type)

    def put_model(self, source_path):
        """Move a finished .glb into the store."""
        return self._put("model", lambda path: shutil.move(source_path, path), ".glb", "model/gltf-binary")

    def get(self, kind, file_id):
        with self._lock:
            item = self._items.get(file_id)
        if item is None or item.kind != kind:
            return None
        if self.clock() - item.created > self.ttl or not item.path.exists():
            self._delete(file_id)
            return None
        return item

    def _delete(self, file_id):
        with self._lock:
            item = self._items.pop(file_id, None)
        if item is not None:
            item.path.unlink(missing_ok=True)

    def sweep(self):
        """Delete expired files, then the oldest ones beyond capacity."""
        now = self.clock()
        with self._lock:
            ordered = sorted(self._items.values(), key=lambda item: item.created)
        expired = [item.id for item in ordered if now - item.created > self.ttl]
        live = [item.id for item in ordered if now - item.created <= self.ttl]
        excess = live[:max(0, len(live) - self.max_items + 1)]
        for file_id in expired + excess:
            self._delete(file_id)

    @contextmanager
    def work_dir(self):
        """A private scratch folder, removed afterwards whatever happens."""
        with tempfile.TemporaryDirectory(dir=self.root / "work") as path:
            yield Path(path)

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)
        with self._lock:
            self._items.clear()
