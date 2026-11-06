/**
 * Live try-on: tracks the face in each new camera frame and draws the
 * generated glasses on it.
 *
 * Per frame: draw the frame mirrored (the desktop app also flips before
 * tracking, so landmark 263 is on screen-right like the canonical face) →
 * MediaPipe landmarks → head pose → two matrices → render. Work runs only when
 * the camera delivers a new frame, and stops with the session.
 */
import type { FaceLandmarker } from "@mediapipe/tasks-vision";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { ArScene, disposeObject } from "./ar-scene.ts";
import { autoFit, combineFits, type FitHints } from "./auto-fit.ts";
import { detectFace, loadFaceLandmarker } from "./face-tracker.ts";
import { cameraIntrinsics, estimateHeadPose, imagePointsFromLandmarks, type HeadPose, type Intrinsics } from "./pose.ts";
import { NO_FIT, type Fit } from "./transforms.ts";

// Tracking runs on a smaller copy of the frame; landmarks are normalised, so
// pose still uses full-resolution pixel coordinates.
const TRACKING_WIDTH = 640;
const MAX_RENDER_WIDTH = 1280;
// Frames without a face before the glasses are hidden (rides out a missed frame).
const MISSES_BEFORE_LOST = 3;
// Loaded models kept for switching back without reloading.
const MODEL_CACHE_SIZE = 3;

export type TrackerState = "loading" | "ready" | "error";
export type ModelState = "none" | "loading" | "ready" | "error";
export type SessionState = { tracker: TrackerState; model: ModelState; faceDetected: boolean };
export type SessionStats = { fps: number; trackingMs: number; poseMs: number; renderMs: number; frames: number };
export type TryOnGlasses = { modelUrl: string; fitHints: FitHints | null };

type CachedModel = { promise: Promise<THREE.Object3D>; object: THREE.Object3D | null };

export class TryOnSession {
  private readonly scene: ArScene;
  private readonly frame = document.createElement("canvas");
  private readonly frameContext: CanvasRenderingContext2D;
  private tracker: FaceLandmarker | null = null;
  private intrinsics: Intrinsics | null = null;
  private previousPose: HeadPose | null = null;
  private misses = MISSES_BEFORE_LOST;
  private lastTimestamp = 0;
  private callbackId = 0;
  private callbackIsVideoFrame = false;
  private disposed = false;
  private autoFitValue: Fit = NO_FIT;
  private manualFit: Fit = NO_FIT;
  private currentUrl: string | null = null;
  private readonly models = new Map<string, CachedModel>();
  private state: SessionState = { tracker: "loading", model: "none", faceDetected: false };
  private stats = { frames: 0, windowStart: 0, windowFrames: 0, tracking: 0, pose: 0, render: 0 };

  constructor(
    private readonly video: HTMLVideoElement,
    canvas: HTMLCanvasElement,
    private readonly onState: (state: SessionState) => void,
    private readonly onStats?: (stats: SessionStats) => void,
  ) {
    this.scene = new ArScene(canvas);
    this.frameContext = this.frame.getContext("2d", { alpha: false })!;
  }

  async start() {
    try {
      this.tracker = await loadFaceLandmarker();
    } catch {
      if (!this.disposed) this.setState({ tracker: "error" });
      return;
    }
    if (this.disposed) return;
    this.setState({ tracker: "ready" });
    this.stats.windowStart = performance.now();
    this.schedule();
  }

  /** Show different glasses without restarting the camera; null shows none. */
  setGlasses(glasses: TryOnGlasses | null) {
    this.autoFitValue = autoFit(glasses?.fitHints ?? null);
    const url = glasses?.modelUrl ?? null;
    if (url === this.currentUrl) return;
    this.currentUrl = url;
    if (!url) {
      this.scene.setModel(null);
      this.setState({ model: "none" });
      return;
    }
    let cached = this.models.get(url);
    if (!cached) {
      const entry: CachedModel = { object: null, promise: new GLTFLoader().loadAsync(url).then((gltf) => gltf.scene) };
      entry.promise.then((object) => (entry.object = object), () => this.models.delete(url));
      this.models.set(url, entry);
      cached = entry;
      this.evictModels();
    }
    if (cached.object) {
      this.scene.setModel(cached.object);
      this.setState({ model: "ready" });
      return;
    }
    this.scene.setModel(null);
    this.setState({ model: "loading" });
    cached.promise.then(
      (object) => {
        if (this.disposed || this.currentUrl !== url) return;
        this.scene.setModel(object);
        this.setState({ model: "ready" });
      },
      () => {
        if (!this.disposed && this.currentUrl === url) this.setState({ model: "error" });
      },
    );
  }

  setFit(manual: Fit) {
    this.manualFit = manual;
  }

  dispose() {
    this.disposed = true;
    if (this.callbackIsVideoFrame) this.video.cancelVideoFrameCallback(this.callbackId);
    else cancelAnimationFrame(this.callbackId);
    this.scene.dispose();
    for (const { object, promise } of this.models.values()) {
      if (object) disposeObject(object);
      else promise.then(disposeObject, () => undefined);
    }
    this.models.clear();
    this.frame.width = this.frame.height = 0;
  }

  private evictModels() {
    while (this.models.size > MODEL_CACHE_SIZE) {
      const oldest = [...this.models.keys()].find((key) => key !== this.currentUrl)!;
      const { object, promise } = this.models.get(oldest)!;
      this.models.delete(oldest);
      if (object) disposeObject(object);
      else promise.then(disposeObject, () => undefined);
    }
  }

  private setState(change: Partial<SessionState>) {
    const next = { ...this.state, ...change };
    if (next.tracker === this.state.tracker && next.model === this.state.model && next.faceDetected === this.state.faceDetected) return;
    this.state = next;
    this.onState(next);
  }

  private schedule() {
    if (this.disposed) return;
    // Only process frames the camera actually delivers.
    this.callbackIsVideoFrame = "requestVideoFrameCallback" in this.video;
    if (this.callbackIsVideoFrame) {
      this.callbackId = this.video.requestVideoFrameCallback(() => this.tick());
    } else {
      this.callbackId = requestAnimationFrame(() => this.tick());
    }
  }

  private tick() {
    if (this.disposed) return;
    const { videoWidth: width, videoHeight: height } = this.video;
    if (this.tracker && width > 0 && height > 0 && this.video.readyState >= 2) this.process(width, height);
    this.schedule();
  }

  private process(width: number, height: number) {
    if (!this.intrinsics || this.intrinsics.width !== width || this.intrinsics.height !== height) {
      this.intrinsics = cameraIntrinsics(width, height);
      const renderWidth = Math.min(width, MAX_RENDER_WIDTH);
      this.scene.setViewport(renderWidth, Math.round((renderWidth * height) / width), this.intrinsics);
      this.frame.width = Math.min(width, TRACKING_WIDTH);
      this.frame.height = Math.round((this.frame.width * height) / width);
      this.previousPose = null;
    }

    const started = performance.now();
    const ctx = this.frameContext;
    ctx.setTransform(-1, 0, 0, 1, this.frame.width, 0); // mirror, as displayed
    ctx.drawImage(this.video, 0, 0, this.frame.width, this.frame.height);
    const timestamp = Math.max(started, this.lastTimestamp + 1);
    this.lastTimestamp = timestamp;
    let landmarks = null;
    try {
      landmarks = detectFace(this.tracker!, this.frame, timestamp);
    } catch {
      landmarks = null;
    }
    const tracked = performance.now();

    const pose = landmarks
      ? estimateHeadPose(imagePointsFromLandmarks(landmarks, width, height), this.intrinsics, this.previousPose)
      : null;
    if (pose) {
      this.previousPose = pose;
      this.misses = 0;
    } else {
      this.misses = Math.min(this.misses + 1, MISSES_BEFORE_LOST);
      if (this.misses >= MISSES_BEFORE_LOST) this.previousPose = null;
    }
    const posed = performance.now();

    // Between a lost frame and losing the face, keep the last pose on screen.
    this.scene.update(pose ?? (this.misses < MISSES_BEFORE_LOST ? this.previousPose : null),
      combineFits(this.autoFitValue, this.manualFit));
    this.scene.render();
    const rendered = performance.now();
    this.setState({ faceDetected: this.misses < MISSES_BEFORE_LOST });
    this.record(started, tracked, posed, rendered);
  }

  private record(started: number, tracked: number, posed: number, rendered: number) {
    const s = this.stats;
    s.frames++;
    s.windowFrames++;
    s.tracking += tracked - started;
    s.pose += posed - tracked;
    s.render += rendered - posed;
    const elapsed = rendered - s.windowStart;
    if (elapsed >= 1000) {
      this.onStats?.({
        fps: (s.windowFrames * 1000) / elapsed,
        trackingMs: s.tracking / s.windowFrames,
        poseMs: s.pose / s.windowFrames,
        renderMs: s.render / s.windowFrames,
        frames: s.frames,
      });
      Object.assign(s, { windowStart: rendered, windowFrames: 0, tracking: 0, pose: 0, render: 0 });
    }
  }
}
