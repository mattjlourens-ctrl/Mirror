// Copies MediaPipe's WebAssembly runtime and the face landmark model into
// public/mediapipe, so face tracking loads from this site and never sends
// camera frames anywhere. Runs before `npm run dev` and `npm run build`.
import { copyFileSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const target = join(root, "public", "mediapipe");
const wasm = join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm");
// The same model file the desktop app uses, from the repository root.
const model = join(root, "..", "face_landmarker.task");

mkdirSync(join(target, "wasm"), { recursive: true });
for (const file of readdirSync(wasm)) copyFileSync(join(wasm, file), join(target, "wasm", file));
if (!existsSync(model)) {
  console.error(`Missing ${model}; it ships with the Mirror repository.`);
  process.exit(1);
}
copyFileSync(model, join(target, "face_landmarker.task"));
console.log("Copied MediaPipe runtime and face model to public/mediapipe");
