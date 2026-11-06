import { test } from "node:test";
import assert from "node:assert/strict";
import { MAX_IMAGE_BYTES, formatBytes, validateImageFile, validateProductUrl } from "./validation.ts";

test("accepts http and https product links", () => {
  assert.deepEqual(validateProductUrl("https://shop.example.com/frames/42?c=black"), {
    ok: true,
    value: "https://shop.example.com/frames/42?c=black",
  });
  assert.equal(validateProductUrl("http://example.com").ok, true);
  assert.equal(validateProductUrl("  https://example.com/a  ").ok, true);
});

test("adds https when the scheme is missing", () => {
  assert.deepEqual(validateProductUrl("shop.example.com/frames"), {
    ok: true,
    value: "https://shop.example.com/frames",
  });
});

test("rejects empty, malformed and non-web links", () => {
  for (const input of ["", "   ", "not a url", "glasses", "ftp://example.com/x",
    "javascript:alert(1)", "mailto:a@b.com", "https://", "https://exa mple.com"]) {
    assert.equal(validateProductUrl(input).ok, false, input);
  }
});

test("accepts PNG and JPG files", () => {
  assert.equal(validateImageFile({ name: "a.png", type: "image/png", size: 1000 }).ok, true);
  assert.equal(validateImageFile({ name: "a.JPG", type: "image/jpeg", size: 1000 }).ok, true);
  assert.equal(validateImageFile({ name: "a.jpeg", type: "", size: 1000 }).ok, true);
});

test("rejects other formats, empty files and oversized files", () => {
  assert.equal(validateImageFile({ name: "a.gif", type: "image/gif", size: 1000 }).ok, false);
  assert.equal(validateImageFile({ name: "a.webp", type: "image/webp", size: 1000 }).ok, false);
  assert.equal(validateImageFile({ name: "a.png", type: "text/plain", size: 1000 }).ok, false);
  assert.equal(validateImageFile({ name: "a.png", type: "image/png", size: 0 }).ok, false);
  assert.equal(validateImageFile({ name: "a.png", type: "image/png", size: MAX_IMAGE_BYTES }).ok, true);
  assert.equal(validateImageFile({ name: "a.png", type: "image/png", size: MAX_IMAGE_BYTES + 1 }).ok, false);
});

test("formats sizes", () => {
  assert.equal(formatBytes(500), "500 B");
  assert.equal(formatBytes(2048), "2 KB");
  assert.equal(formatBytes(MAX_IMAGE_BYTES), "10 MB");
  assert.equal(formatBytes(1.5 * 1024 * 1024), "1.5 MB");
});
