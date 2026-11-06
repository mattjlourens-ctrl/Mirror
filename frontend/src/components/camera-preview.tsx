import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowDown, ArrowUp, Camera, CameraOff, CircleCheck, LoaderCircle, Minus, Plus, RotateCcw } from "lucide-react";
import type { ReadyGlasses } from "@/lib/api";
import type { SessionState, SessionStats, TryOnSession } from "@/lib/ar/try-on-session";
import { NO_FIT, clampFit, type Fit } from "@/lib/ar/transforms";
import { ModelViewer } from "./model-viewer";
import type { JobPhase } from "./use-glasses-job";

type CameraState =
  | { kind: "idle" }
  | { kind: "starting" }
  | { kind: "live" }
  | { kind: "error"; title: string; detail: string };

/** Maps getUserMedia failures to messages a visitor can act on. */
function describeCameraError(error: unknown): { title: string; detail: string } {
  const name = error instanceof DOMException ? error.name : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return {
        title: "Camera permission required",
        detail: "Allow camera access for this site in your browser's address bar or settings, then try again.",
      };
    case "NotFoundError":
    case "OverconstrainedError":
      return { title: "No camera found", detail: "Connect a camera and try again." };
    case "NotReadableError":
    case "AbortError":
      return {
        title: "The camera is busy",
        detail: "Another app may be using it. Close that app and try again.",
      };
    default:
      return { title: "The camera couldn't start", detail: "Please try again." };
  }
}

type Tone = "busy" | "ok" | "info" | "error";

/** One short line saying what the try-on is doing. */
function tryOnMessage(session: SessionState, glasses: ReadyGlasses | null, jobPhase: JobPhase): { text: string; tone: Tone } {
  if (session.tracker === "loading") return { text: "Starting face tracking…", tone: "busy" };
  if (session.tracker === "error") return { text: "Face tracking couldn't start. Reload to try again.", tone: "error" };
  if (jobPhase === "uploading") return { text: "Uploading image…", tone: "busy" };
  if (jobPhase === "preparing") return { text: "Generating glasses…", tone: "busy" };
  if (!glasses) {
    return jobPhase === "failed"
      ? { text: "Unable to process this image", tone: "error" }
      : { text: "Add glasses above to try them on", tone: "info" };
  }
  if (session.model === "loading") return { text: "Loading model…", tone: "busy" };
  if (session.model === "error") return { text: "These glasses couldn't be loaded", tone: "error" };
  if (!session.faceDetected) return { text: "Face not detected", tone: "info" };
  return { text: "Ready to try on", tone: "ok" };
}

const INITIAL_SESSION: SessionState = { tracker: "loading", model: "none", faceDetected: false };
const FIT_STEP_CM = 0.2;
const FIT_SCALE_STEP = 0.03;
const FIT_BUTTONS = [
  { label: "Move glasses up", icon: ArrowUp, change: { y: FIT_STEP_CM } },
  { label: "Move glasses down", icon: ArrowDown, change: { y: -FIT_STEP_CM } },
  { label: "Make glasses smaller", icon: Minus, change: { scale: 1 - FIT_SCALE_STEP } },
  { label: "Make glasses larger", icon: Plus, change: { scale: 1 + FIT_SCALE_STEP } },
];

export function CameraPreview({ glasses, jobPhase }: { glasses: ReadyGlasses | null; jobPhase: JobPhase }) {
  const video = useRef<HTMLVideoElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const session = useRef<TryOnSession | null>(null);
  const [state, setState] = useState<CameraState>({ kind: "idle" });
  // Each camera session gets a fresh canvas: a released WebGL context can't be reused.
  const [canvasKey, setCanvasKey] = useState(0);
  const [sessionState, setSessionState] = useState<SessionState>(INITIAL_SESSION);
  const [stats, setStats] = useState<SessionStats | null>(null);
  // Manual adjustments belong to one pair; a new pair starts from its automatic fit.
  const [manual, setManual] = useState<{ modelUrl: string | null; fit: Fit }>({ modelUrl: null, fit: NO_FIT });
  const fit = manual.modelUrl === (glasses?.modelUrl ?? null) ? manual.fit : NO_FIT;
  // ?debug shows live performance numbers.
  const [debug] = useState(() => typeof window !== "undefined" && new URLSearchParams(window.location.search).has("debug"));

  const release = useCallback(() => {
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
  }, []);

  // Stop the camera when the visitor leaves the page.
  useEffect(() => release, [release]);

  const live = state.kind === "live";

  // Face tracking and drawing run only while the camera is live.
  useEffect(() => {
    if (!live) return;
    let active: TryOnSession | null = null;
    let cancelled = false;
    void import("@/lib/ar/try-on-session").then(({ TryOnSession }) => {
      if (cancelled || !video.current || !overlay.current) return;
      try {
        active = new TryOnSession(video.current, overlay.current, setSessionState, setStats);
      } catch {
        setSessionState({ ...INITIAL_SESSION, tracker: "error" }); // e.g. WebGL unavailable
        return;
      }
      session.current = active;
      void active.start();
    }, () => setSessionState({ ...INITIAL_SESSION, tracker: "error" }));
    return () => {
      cancelled = true;
      active?.dispose();
      session.current = null;
      setSessionState(INITIAL_SESSION);
      setStats(null);
    };
  }, [live]);

  // New glasses replace the old ones on the face; the camera keeps running.
  // (Re-run once tracking is ready, since the session is created asynchronously.)
  useEffect(() => {
    session.current?.setGlasses(glasses ? { modelUrl: glasses.modelUrl, fitHints: glasses.fitHints } : null);
  }, [glasses, sessionState.tracker]);

  useEffect(() => {
    session.current?.setFit(fit);
  }, [fit, sessionState.tracker]);

  useEffect(() => {
    if (debug) (window as { __mirrorTryOn?: unknown }).__mirrorTryOn = { state: sessionState, stats, fit };
  }, [debug, sessionState, stats, fit]);

  async function start() {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      setState({
        kind: "error",
        title: "Camera not available",
        detail: window.isSecureContext
          ? "This browser doesn't support camera access."
          : "The camera only works over HTTPS or on localhost.",
      });
      return;
    }

    setState({ kind: "starting" });
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      release();
      stream.current = media;
      // If the camera is unplugged or revoked, fall back to the idle view.
      media.getVideoTracks().forEach((track) =>
        track.addEventListener("ended", () => {
          if (stream.current !== media) return;
          release();
          setState({ kind: "error", title: "The camera stopped", detail: "It may have been disconnected." });
          setCanvasKey((key) => key + 1);
        }),
      );
      if (video.current) {
        video.current.srcObject = media;
        await video.current.play().catch(() => undefined);
      }
      setState({ kind: "live" });
    } catch (error) {
      release();
      setState({ kind: "error", ...describeCameraError(error) });
    }
  }

  function stop() {
    release();
    setState({ kind: "idle" });
    setCanvasKey((key) => key + 1);
  }

  function adjust(change: { y?: number; scale?: number }) {
    setManual({
      modelUrl: glasses?.modelUrl ?? null,
      fit: clampFit({ ...fit, y: fit.y + (change.y ?? 0), scale: fit.scale * (change.scale ?? 1) }),
    });
  }

  const message = tryOnMessage(sessionState, glasses, jobPhase);
  const canAdjust = live && glasses !== null && sessionState.model === "ready";
  return (
    <section aria-labelledby="camera-heading" className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 id="camera-heading" className="text-base font-medium text-neutral-900">
            Live preview
          </h2>
          <p className="text-sm text-neutral-500">Your camera stays on this device; nothing is uploaded.</p>
        </div>
        {live ? (
          <button
            type="button"
            onClick={stop}
            className="inline-flex h-10 shrink-0 items-center gap-2 rounded-lg border border-neutral-200 px-4 text-sm font-medium text-neutral-800 hover:bg-neutral-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-950"
          >
            <CameraOff aria-hidden className="size-4" />
            Stop camera
          </button>
        ) : (
          <button
            type="button"
            onClick={start}
            disabled={state.kind === "starting"}
            className="inline-flex h-10 shrink-0 items-center gap-2 rounded-lg bg-neutral-950 px-4 text-sm font-medium text-white hover:bg-neutral-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-950 disabled:bg-neutral-300"
          >
            {state.kind === "starting" ? (
              <LoaderCircle aria-hidden className="size-4 animate-spin" />
            ) : (
              <Camera aria-hidden className="size-4" />
            )}
            {state.kind === "error" ? "Try again" : "Start camera"}
          </button>
        )}
      </div>

      <div className="relative aspect-[4/3] w-full overflow-hidden rounded-2xl border border-neutral-200 bg-neutral-50 sm:aspect-video">
        {/* Mirrored like a real mirror, matching the desktop app. */}
        <video
          ref={video}
          muted
          playsInline
          aria-label="Your camera"
          className={`h-full w-full -scale-x-100 object-cover ${live ? "" : "invisible"}`}
        />
        {/* Glasses, drawn in the mirrored frame's pixels (so not flipped again) and cropped like the video. */}
        <canvas
          key={canvasKey}
          ref={overlay}
          aria-hidden
          data-testid="try-on-overlay"
          className={`pointer-events-none absolute inset-0 h-full w-full object-cover ${live ? "" : "invisible"}`}
        />
        {live && (
          <p
            role="status"
            aria-live="polite"
            data-tone={message.tone}
            className="absolute top-3 left-3 inline-flex max-w-[calc(100%-1.5rem)] items-center gap-2 rounded-full bg-white/90 px-3 py-1.5 text-sm font-medium text-neutral-900 shadow-xs"
          >
            {message.tone === "busy" && <LoaderCircle aria-hidden className="size-4 animate-spin text-neutral-500" />}
            {message.tone === "ok" && <CircleCheck aria-hidden className="size-4 text-emerald-700" />}
            {message.tone === "error" && <span aria-hidden className="size-2 rounded-full bg-red-500" />}
            {message.tone === "info" && <span aria-hidden className="size-2 rounded-full bg-neutral-400" />}
            {message.text}
          </p>
        )}
        {live && debug && stats && (
          <p className="absolute right-3 bottom-3 rounded-md bg-black/70 px-2 py-1 font-mono text-xs text-white">
            {stats.fps.toFixed(1)} fps · track {stats.trackingMs.toFixed(1)} ms · pose {stats.poseMs.toFixed(2)} ms · draw{" "}
            {stats.renderMs.toFixed(1)} ms
          </p>
        )}
        {!live && (
          <div role="status" aria-live="polite" className="absolute inset-0 flex flex-col items-center justify-center gap-2 px-6 text-center">
            {state.kind === "error" ? (
              <>
                <CameraOff aria-hidden className="size-8 text-neutral-400" />
                <p className="text-base font-medium text-neutral-900">{state.title}</p>
                <p className="max-w-sm text-sm text-neutral-500">{state.detail}</p>
              </>
            ) : state.kind === "starting" ? (
              <>
                <LoaderCircle aria-hidden className="size-8 animate-spin text-neutral-400" />
                <p className="text-sm text-neutral-500">Waiting for camera permission…</p>
              </>
            ) : (
              <>
                <Camera aria-hidden className="size-8 text-neutral-400" />
                <p className="text-base font-medium text-neutral-900">Camera is off</p>
                <p className="max-w-sm text-sm text-neutral-500">
                  {glasses
                    ? "Your glasses are ready. Start your camera to try them on."
                    : "Start your camera, then add glasses above to try them on."}
                </p>
              </>
            )}
          </div>
        )}
      </div>

      {canAdjust && (
        <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Adjust fit">
          <span className="mr-1 text-sm text-neutral-500">Adjust fit</span>
          {FIT_BUTTONS.map(({ label, icon: Icon, change }) => (
            <button
              key={label}
              type="button"
              aria-label={label}
              title={label}
              onClick={() => adjust(change)}
              className="inline-flex size-9 items-center justify-center rounded-lg border border-neutral-200 text-neutral-700 hover:bg-neutral-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-950"
            >
              <Icon aria-hidden className="size-4" />
            </button>
          ))}
          <button
            type="button"
            onClick={() => setManual({ modelUrl: null, fit: NO_FIT })}
            className="inline-flex h-9 items-center gap-2 rounded-lg border border-neutral-200 px-3 text-sm text-neutral-700 hover:bg-neutral-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-neutral-950"
          >
            <RotateCcw aria-hidden className="size-4" />
            Reset
          </button>
        </div>
      )}

      {glasses && !live && (
        <div
          data-model-url={glasses.modelUrl}
          className="flex flex-col gap-4 rounded-2xl border border-neutral-200 p-4 sm:flex-row sm:items-center"
        >
          <div className="h-36 w-full shrink-0 rounded-xl bg-neutral-50 sm:w-56">
            <ModelViewer key={glasses.modelUrl} modelUrl={glasses.modelUrl} label="3D model of your glasses, generated from the photo" />
          </div>
          <div className="flex flex-col gap-1">
            <p className="flex items-center gap-2 text-base font-medium text-neutral-900">
              <CircleCheck aria-hidden className="size-4 text-emerald-700" />
              Your glasses are ready
            </p>
            <p className="text-sm text-neutral-500">Start your camera to see them on your face.</p>
          </div>
        </div>
      )}
    </section>
  );
}
