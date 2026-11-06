"""End-to-end API tests: real reconstruction, a fake internet for product pages."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import Settings
from glb_model import load_glasses_model
from image_to_3d.assembly import FRONT_FRAME, LEFT_TEMPLE, RIGHT_TEMPLE
from image_to_3d.errors import MeshBuildError
from tests.backend_fakes import (FakeWeb, blank_png_bytes, mirrored_photo_bytes, photo_bytes,
                                 side_view_bytes)

ORIGIN = "http://localhost:3000"
PAGE = "https://shop.test/frames/aria"


class ApiTestCase(unittest.TestCase):
    settings = Settings(allowed_origins=(ORIGIN,), max_pending_jobs=2)

    def setUp(self):
        self.web = FakeWeb()
        self.fetcher = self.web.fetcher()
        self.addCleanup(self.fetcher.close)
        self.client = TestClient(create_app(self.settings, fetcher=self.fetcher))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def upload(self, data, name="glasses.jpg", content_type="image/jpeg"):
        return self.client.post("/api/jobs/from-image", files={"image": (name, data, content_type)})

    def wait_for(self, job, timeout=60):
        deadline = time.monotonic() + timeout
        while job["status"] in ("queued", "processing"):
            self.assertLess(time.monotonic(), deadline, "job did not finish")
            time.sleep(0.1)
            response = self.client.get(f"/api/jobs/{job['id']}")
            self.assertEqual(response.status_code, 200)
            job = response.json()
        return job

    def error_code(self, response):
        return response.json()["detail"]["code"]


class UploadTests(ApiTestCase):
    def test_upload_produces_a_model_mirror_can_load(self):
        response = self.upload(photo_bytes())
        self.assertEqual(response.status_code, 202)
        job = self.wait_for(response.json())
        self.assertEqual(job["status"], "succeeded", job)
        self.assertIsNone(job["error"])
        self.assertIn(FRONT_FRAME, job["parts"])

        source = self.client.get(job["source_image_url"])
        self.assertEqual((source.status_code, source.headers["content-type"]), (200, "image/jpeg"))
        self.assertEqual(source.content, photo_bytes())

        model = self.client.get(job["model_url"])
        self.assertEqual(model.status_code, 200)
        self.assertEqual(model.headers["content-type"], "model/gltf-binary")
        self.assertTrue(model.content.startswith(b"glTF"))

        # Load it the way the desktop try-on does and check Mirror's conventions:
        # centimetres, X across the frame (default 14 cm), temples running back along -Z.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "downloaded.glb"
            path.write_bytes(model.content)
            loaded = load_glasses_model(path)
        parts = {part.name: part for part in loaded.parts}
        front = parts[FRONT_FRAME].vertices
        self.assertAlmostEqual(front[:, 0].max() - front[:, 0].min(), 14.0, delta=0.2)
        self.assertLess(parts[LEFT_TEMPLE].vertices[:, 2].min(), -10)
        self.assertLess(parts[LEFT_TEMPLE].vertices[:, 0].mean(), 0)
        self.assertGreater(parts[RIGHT_TEMPLE].vertices[:, 0].mean(), 0)
        self.assertIsNotNone(parts[FRONT_FRAME].texture)

        # Lens centres for the browser's automatic fit: one each side, inside the frame.
        (left_x, left_y), (right_x, right_y) = job["fit"]["lens_centers_cm"]
        self.assertTrue(-7 < left_x < -1 and 1 < right_x < 7)
        self.assertAlmostEqual(left_x, -right_x, delta=0.5)
        self.assertTrue(abs(left_y) < 2 and abs(right_y) < 2)

    def test_same_photo_gives_the_same_model(self):
        first = self.wait_for(self.upload(photo_bytes()).json())
        second = self.wait_for(self.upload(photo_bytes()).json())
        self.assertNotEqual(first["model_url"], second["model_url"])
        self.assertEqual(self.client.get(first["model_url"]).content, self.client.get(second["model_url"]).content)

    def test_unsupported_and_damaged_images_are_rejected_immediately(self):
        cases = [(b"GIF89a" + b"\0" * 100, "unsupported_image", 415), (b"", "empty_image", 415),
                 (photo_bytes()[:3000], "corrupt_image", 415)]
        for data, code, status in cases:
            with self.subTest(code=code):
                response = self.upload(data)
                self.assertEqual(response.status_code, status)
                self.assertEqual(self.error_code(response), code)

    def test_oversized_uploads_are_rejected(self):
        small = Settings(allowed_origins=(ORIGIN,), max_image_bytes=20_000)
        with TestClient(create_app(small, fetcher=self.fetcher)) as client:
            response = client.post("/api/jobs/from-image", files={"image": ("big.jpg", photo_bytes(), "image/jpeg")})
        self.assertEqual(response.status_code, 413)

    def test_photos_without_glasses_fail_with_a_clear_reason(self):
        job = self.wait_for(self.upload(blank_png_bytes(), "blank.png", "image/png").json())
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error"]["code"], "no_glasses_found")
        self.assertIsNone(job["model_url"])
        self.assertIsNone(job["fit"])

    def test_reconstruction_failures_are_reported_and_leave_no_files(self):
        with mock.patch("backend.jobs.convert_image_to_glasses", side_effect=MeshBuildError("bad")):
            job = self.wait_for(self.upload(photo_bytes()).json())
        self.assertEqual((job["status"], job["error"]["code"]), ("failed", "reconstruction_failed"))
        store = self.client.app.state.store
        self.assertEqual(list((store.root / "models").iterdir()), [])
        self.assertEqual(list((store.root / "work").iterdir()), [])

    def test_unexpected_errors_are_hidden(self):
        with mock.patch("backend.jobs.convert_image_to_glasses", side_effect=RuntimeError("secret path /x")), \
                self.assertLogs("backend.jobs", "ERROR"):
            job = self.wait_for(self.upload(photo_bytes()).json())
        self.assertEqual(job["error"]["code"], "internal_error")
        self.assertNotIn("secret", job["error"]["message"])

    def test_queue_is_bounded(self):
        release = __import__("threading").Event()

        def slow(source, output):
            release.wait(10)
            raise MeshBuildError("stopped")

        with mock.patch("backend.jobs.convert_image_to_glasses", side_effect=slow):
            self.assertEqual(self.upload(photo_bytes()).status_code, 202)
            self.assertEqual(self.upload(photo_bytes()).status_code, 202)
            response = self.upload(photo_bytes())
            release.set()
        self.assertEqual((response.status_code, self.error_code(response)), (503, "busy"))


class RetrievalTests(ApiTestCase):
    def test_unknown_and_malformed_ids_are_not_found(self):
        for path in ["/api/jobs/" + "0" * 32, "/api/models/" + "0" * 32, "/api/images/" + "0" * 32,
                     "/api/models/..%2F..%2Fetc%2Fpasswd", "/api/images/abc"]:
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)

    def test_expired_models_are_gone(self):
        job = self.wait_for(self.upload(photo_bytes()).json())
        store = self.client.app.state.store
        with mock.patch.object(store, "clock", return_value=store.clock() + store.ttl + 1):
            self.assertEqual(self.client.get(job["model_url"]).status_code, 404)

    def test_cors_allows_the_website_only(self):
        allowed = self.client.options("/api/jobs/from-image", headers={
            "Origin": ORIGIN, "Access-Control-Request-Method": "POST"})
        self.assertEqual(allowed.headers.get("access-control-allow-origin"), ORIGIN)
        other = self.client.options("/api/jobs/from-image", headers={
            "Origin": "https://evil.test", "Access-Control-Request-Method": "POST"})
        self.assertNotIn("access-control-allow-origin", other.headers)


class ProductUrlTests(ApiTestCase):
    def page(self, body, url=PAGE):
        self.web.add(url, body)

    def lookup(self, url=PAGE):
        return self.client.post("/api/sources/from-url", json={"url": url})

    def test_one_front_view_starts_a_job_automatically(self):
        self.page("""<meta property="og:image" content="https://cdn.shop.test/aria-front.jpg">
                     <img src="/img/aria-side.png" alt="Side"><img src="/brand-logo.png">""")
        self.web.add("https://cdn.shop.test/aria-front.jpg", photo_bytes(), content_type="image/jpeg")
        self.web.add("https://shop.test/img/aria-side.png", side_view_bytes(), content_type="image/png")

        response = self.lookup()
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["status"], "processing")
        self.assertEqual([candidate["source_url"] for candidate in body["candidates"]],
                         ["https://cdn.shop.test/aria-front.jpg"])
        job = self.wait_for(body["job"])
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["source_image_url"], body["candidates"][0]["image_url"])
        self.assertNotIn("/brand-logo.png", [path for *_, path, _ in self.web.requests])

    def test_several_front_views_let_the_user_choose(self):
        self.page('<img src="/a.jpg" alt="Aria black"><img src="/b.jpg" alt="Aria black, other angle">')
        self.web.add("https://shop.test/a.jpg", photo_bytes(), content_type="image/jpeg")
        self.web.add("https://shop.test/b.jpg", mirrored_photo_bytes(), content_type="image/jpeg")

        body = self.lookup().json()
        self.assertEqual(body["status"], "choose")
        self.assertIsNone(body["job"])
        self.assertEqual(len(body["candidates"]), 2)
        chosen = body["candidates"][1]
        image = self.client.get(chosen["image_url"])
        self.assertEqual(image.status_code, 200)

        response = self.client.post("/api/jobs/from-candidate", json={"candidate_id": chosen["id"]})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.wait_for(response.json())["status"], "succeeded")

    def test_a_direct_image_link_works(self):
        self.web.add("https://cdn.shop.test/aria.jpg", photo_bytes(), content_type="image/jpeg")
        body = self.lookup("https://cdn.shop.test/aria.jpg").json()
        self.assertEqual(body["status"], "processing")

    def test_pages_with_only_unusable_photos_suggest_uploading(self):
        self.page('<img src="/side.png"><img src="/blank.png">')
        self.web.add("https://shop.test/side.png", side_view_bytes(), content_type="image/png")
        self.web.add("https://shop.test/blank.png", blank_png_bytes(), content_type="image/png")
        response = self.lookup()
        self.assertEqual((response.status_code, self.error_code(response)), (422, "no_front_view"))
        self.assertIn("upload", response.json()["detail"]["message"].lower())
        self.assertEqual(list((self.client.app.state.store.root / "images").iterdir()), [])

    def test_pages_without_photos_suggest_uploading(self):
        self.page("<html><body>No images here <img src='/missing.jpg'></body></html>")
        response = self.lookup()
        self.assertEqual((response.status_code, self.error_code(response)), (422, "no_product_images"))
        self.assertIn("upload", response.json()["detail"]["message"].lower())

    def test_invalid_urls_are_rejected(self):
        for url in ["ftp://shop.test/x", "not a url", "javascript:alert(1)", "https://user:pw@shop.test/"]:
            with self.subTest(url=url):
                response = self.lookup(url)
                self.assertEqual((response.status_code, self.error_code(response)), (422, "invalid_url"))

    def test_internal_destinations_are_blocked(self):
        self.web.add("https://shop.test/sneaky", status=302, headers={"location": "http://169.254.169.254/"})
        self.web.dns["intranet.test"] = ["10.0.0.7"]
        for url in ["http://127.0.0.1/", "http://intranet.test/", "https://shop.test/sneaky"]:
            with self.subTest(url=url):
                response = self.lookup(url)
                self.assertEqual(self.error_code(response), "blocked_destination")
        self.assertNotIn("intranet.test", [host for host, *_ in self.web.requests])

    def test_internal_image_urls_on_a_public_page_are_skipped(self):
        self.page('<meta property="og:image" content="http://127.0.0.1/admin.jpg">')
        response = self.lookup()
        self.assertEqual(self.error_code(response), "no_product_images")

    def test_robots_txt_is_respected(self):
        self.web.add("https://shop.test/robots.txt", "User-agent: *\nDisallow: /frames/", content_type="text/plain")
        self.page('<img src="/a.jpg">')
        response = self.lookup()
        self.assertEqual(self.error_code(response), "access_restricted")
        self.assertNotIn("/frames/aria", [path for *_, path, _ in self.web.requests])

    def test_blocked_and_missing_pages(self):
        self.web.add(PAGE, status=403)
        self.assertEqual(self.error_code(self.lookup()), "access_restricted")
        self.web.add(PAGE, status=404)
        self.assertEqual(self.error_code(self.lookup()), "page_not_found")

    def test_oversized_and_unsupported_candidate_images_are_skipped(self):
        small = Settings(allowed_origins=(ORIGIN,), max_image_bytes=50_000)
        self.page('<img src="/huge.jpg"><img src="/anim.gif.jpg"><img src="/ok.jpg">')
        self.web.add("https://shop.test/huge.jpg", photo_bytes() + b"\0" * 60_000, content_type="image/jpeg")
        self.web.add("https://shop.test/anim.gif.jpg", b"GIF89a" + b"\0" * 100, content_type="image/gif")
        self.web.add("https://shop.test/ok.jpg", photo_bytes(), content_type="image/jpeg")
        with TestClient(create_app(small, fetcher=self.fetcher)) as client:
            body = client.post("/api/sources/from-url", json={"url": PAGE}).json()
        self.assertEqual([candidate["source_url"] for candidate in body["candidates"]], ["https://shop.test/ok.jpg"])

    def test_unknown_candidates_are_not_found(self):
        response = self.client.post("/api/jobs/from-candidate", json={"candidate_id": "f" * 32})
        self.assertEqual(response.status_code, 404)


class DefaultCorsTests(unittest.TestCase):
    def preflight(self, settings, origin):
        with TestClient(create_app(settings)) as client:
            return client.options("/api/jobs/from-image", headers={
                "Origin": origin, "Access-Control-Request-Method": "POST"})

    def default_settings(self, **env):
        with mock.patch.dict("os.environ", env, clear=True):
            return Settings.from_env()

    def test_local_dev_servers_allowed_on_any_port(self):
        settings = self.default_settings()
        for origin in ("http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001",
                       "http://127.0.0.1:5173", "http://localhost"):
            response = self.preflight(settings, origin)
            self.assertEqual(response.status_code, 200, origin)
            self.assertEqual(response.headers["access-control-allow-origin"], origin)

    def test_other_origins_stay_blocked(self):
        settings = self.default_settings()
        for origin in ("https://evil.example", "http://localhost.evil.example", "http://evil.example/localhost:3000",
                       "http://10.0.102.2:3000", "http://localhost:3000.evil.example", "null"):
            response = self.preflight(settings, origin)
            self.assertNotIn("access-control-allow-origin", response.headers, origin)

    def test_local_origins_can_be_disabled(self):
        settings = self.default_settings(MIRROR_ALLOW_LOCAL_ORIGINS="0",
                                         MIRROR_ALLOWED_ORIGINS="https://mirror.example")
        self.assertNotIn("access-control-allow-origin", self.preflight(settings, "http://localhost:3001").headers)
        allowed = self.preflight(settings, "https://mirror.example")
        self.assertEqual(allowed.headers["access-control-allow-origin"], "https://mirror.example")


class RouteAvailabilityTests(unittest.TestCase):
    """The website depends on these exact paths."""

    def test_routes_the_website_uses_exist(self):
        with TestClient(create_app(Settings())) as client:
            self.assertEqual(client.get("/api/health").json(), {"status": "ok"})
            paths = set(client.get("/openapi.json").json()["paths"])
        self.assertLessEqual({"/api/health", "/api/jobs/from-image", "/api/sources/from-url",
                              "/api/jobs/from-candidate", "/api/jobs/{job_id}", "/api/images/{image_id}",
                              "/api/models/{model_id}"}, paths)

    def test_missing_upload_field_is_a_validation_error(self):
        with TestClient(create_app(Settings())) as client:
            response = client.post("/api/jobs/from-image", files={"photo": ("a.png", b"x", "image/png")})
        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"]
        self.assertIsInstance(detail, list)
        self.assertEqual(detail[0]["loc"][-1], "image")

if __name__ == "__main__":
    unittest.main()
