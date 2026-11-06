/**
 * MediaPipe Face Landmarker for the web (the same model and version as the
 * desktop app). It runs entirely in the browser from files this site serves
 * (public/mediapipe); no camera frames leave the device.
 *
 * MediaPipe Face Landmarker is a pretrained machine-learning model. Only face
 * tracking uses it; glasses reconstruction is deterministic geometry.
 */
import type { FaceLandmarker, NormalizedLandmark } from "@mediapipe/tasks-vision";

const WASM_PATH = "/mediapipe/wasm";
const MODEL_PATH = "/mediapipe/face_landmarker.task";

let landmarker: Promise<FaceLandmarker> | null = null;

/** The shared landmarker, created on first use (GPU if available, else CPU). */
export function loadFaceLandmarker(): Promise<FaceLandmarker> {
  if (!landmarker) {
    landmarker = (async () => {
      const { FaceLandmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
      const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
      const options = (delegate: "GPU" | "CPU") => ({
        baseOptions: { modelAssetPath: MODEL_PATH, delegate },
        runningMode: "VIDEO" as const,
        numFaces: 1,
      });
      try {
        return await FaceLandmarker.createFromOptions(fileset, options("GPU"));
      } catch {
        return await FaceLandmarker.createFromOptions(fileset, options("CPU"));
      }
    })();
    landmarker.catch(() => {
      landmarker = null; // let a later attempt retry
    });
  }
  return landmarker;
}

/** Landmarks of the first face in a frame, or null. Timestamps must increase. */
export function detectFace(
  tracker: FaceLandmarker,
  frame: HTMLCanvasElement,
  timestamp: number,
): NormalizedLandmark[] | null {
  const result = tracker.detectForVideo(frame, timestamp);
  return result.faceLandmarks[0] ?? null;
}
