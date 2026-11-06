import assert from "node:assert/strict";
import { test } from "node:test";
import { httpError, networkFailureMessage } from "./api-errors.ts";

test("Mirror's own errors keep their message and code", () => {
  assert.deepEqual(httpError(422, { detail: { code: "invalid_image", message: "That file isn't a PNG or JPG image." } }), {
    message: "That file isn't a PNG or JPG image.",
    code: "invalid_image",
  });
});

test("FastAPI validation errors name the field and the problem", () => {
  const body = { detail: [{ type: "missing", loc: ["body", "image"], msg: "Field required", input: null }] };
  assert.deepEqual(httpError(422, body), {
    message: "Mirror rejected the request (422): image: Field required.",
    code: "validation_error",
  });
  const two = { detail: [{ loc: ["body", "url"], msg: "Field required" }, { loc: ["query", "limit"], msg: "Input should be a valid integer" }] };
  assert.equal(httpError(422, two).message, "Mirror rejected the request (422): url: Field required; limit: Input should be a valid integer.");
});

test("an unknown path says the endpoint is missing, not that the backend is down", () => {
  const { message, code } = httpError(404, { detail: "Not Found" }, "http://localhost:8000/api/jobs/from-image");
  assert.equal(code, "endpoint_not_found");
  assert.match(message, /404/);
  assert.match(message, /http:\/\/localhost:8000\/api\/jobs\/from-image/);
});

test("server errors are reported as server errors", () => {
  assert.match(httpError(500, null).message, /internal error \(500\)/);
  assert.match(httpError(502, "<html>").message, /\(502\)/);
});

test("a running backend that rejects the page's origin is not reported as down", () => {
  const blocked = networkFailureMessage({ backendResponds: true, apiUrl: "http://localhost:8000", pageOrigin: "http://10.0.0.5:3000", dev: true });
  assert.match(blocked, /running/);
  assert.match(blocked, /http:\/\/10\.0\.0\.5:3000/);
  assert.match(blocked, /MIRROR_ALLOWED_ORIGINS/);
  assert.doesNotMatch(blocked, /Start it with/);
  assert.match(blocked, /Open the site at http:\/\/localhost:3000/);
  const onDefault = networkFailureMessage({ backendResponds: true, apiUrl: "http://localhost:8000", pageOrigin: "http://localhost:3000", dev: true });
  assert.doesNotMatch(onDefault, /Open the site at/);
  assert.match(onDefault, /add http:\/\/localhost:3000 to MIRROR_ALLOWED_ORIGINS/i);
});

test("a backend that doesn't answer is reported as not running, with the command in development", () => {
  const down = networkFailureMessage({ backendResponds: false, apiUrl: "http://localhost:8000", pageOrigin: "http://localhost:3000", dev: true });
  assert.match(down, /Couldn't reach the Mirror backend at http:\/\/localhost:8000/);
  assert.match(down, /uv run uvicorn backend\.app:app --port 8000/);
  const prod = networkFailureMessage({ backendResponds: false, apiUrl: "https://api.example", pageOrigin: "https://mirror.example", dev: false });
  assert.doesNotMatch(prod, /uvicorn/);
});
