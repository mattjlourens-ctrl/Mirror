import assert from "node:assert/strict";
import { test } from "node:test";
import { resolveApiUrl } from "./api-config.ts";

test("uses NEXT_PUBLIC_MIRROR_API_URL without a trailing slash", () => {
  assert.equal(resolveApiUrl(" https://api.example.com/ ", "production"), "https://api.example.com");
});

test("falls back to the local backend in development", () => {
  assert.equal(resolveApiUrl(undefined, "development"), "http://localhost:8000");
  assert.equal(resolveApiUrl("", "development"), "http://localhost:8000");
});

test("stays unconfigured in production builds without a URL", () => {
  assert.equal(resolveApiUrl(undefined, "production"), "");
});
