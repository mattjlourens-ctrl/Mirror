"""A fake internet for backend tests: canned DNS answers and HTTP responses."""
from pathlib import Path

import cv2
import httpx
import numpy as np

from backend.errors import FetchFailed
from backend.safe_fetch import SafeFetcher

PHOTO_PATH = Path(__file__).resolve().parent.parent / "image_to_3d" / "glasses.jpg"
PUBLIC_IP = "93.184.216.34"


def photo_bytes():
    return PHOTO_PATH.read_bytes()


def mirrored_photo_bytes():
    """A second, different front view: the sample photo flipped left to right."""
    return cv2.imencode(".jpg", cv2.flip(cv2.imread(str(PHOTO_PATH)), 1))[1].tobytes()


def side_view_bytes():
    """A long thin bar on white, like an arm seen from the side: not a front view."""
    image = np.full((300, 900, 3), 255, np.uint8)
    cv2.rectangle(image, (50, 140), (850, 160), (30, 30, 30), -1)
    return cv2.imencode(".png", image)[1].tobytes()


def blank_png_bytes(size=300):
    return cv2.imencode(".png", np.full((size, size, 3), 255, np.uint8))[1].tobytes()


class FakeWeb:
    """Routes keyed by (host, path); DNS answers keyed by host."""

    def __init__(self):
        self.dns = {}
        self.routes = {}
        self.requests = []

    def add(self, url, body=b"", status=200, content_type="text/html", headers=None, ip=PUBLIC_IP):
        request_url = httpx.URL(url)
        self.dns.setdefault(request_url.host, [ip])
        self.routes[(request_url.host, request_url.raw_path.decode())] = (status, body, content_type, headers or {})

    def resolve(self, host, port):
        if host not in self.dns:
            raise FetchFailed("That website's address couldn't be found.", code="unknown_host")
        return self.dns[host]

    def handle(self, request):
        host = request.headers["host"].split(":")[0]
        self.requests.append((host, request.url.host, request.url.raw_path.decode(), request.extensions))
        route = self.routes.get((host, request.url.raw_path.decode()))
        if route is None:
            return httpx.Response(404, text="not found")
        status, body, content_type, headers = route
        if isinstance(body, str):
            body = body.encode()
        return httpx.Response(status, content=body, headers={"content-type": content_type, **headers})

    def fetcher(self, **kwargs):
        return SafeFetcher("MirrorBot/test", resolver=self.resolve,
                           transport=httpx.MockTransport(self.handle), **kwargs)
