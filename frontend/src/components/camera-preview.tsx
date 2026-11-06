import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, CameraOff, CircleCheck, LoaderCircle } from "lucide-react";
import type { ReadyGlasses } from "@/lib/api";
import { ModelViewer } from "./model-viewer";

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
        title: "Camera access was blocked",
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

export function CameraPreview({ glasses }: { glasses: ReadyGlasses | null }) {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const [state, setState] = useState<CameraState>({ kind: "idle" });

  const release = useCallback(() => {
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
  }, []);

  // Stop the camera when the visitor leaves the page.
  useEffect(() => release, [release]);

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
  }

  const live = state.kind === "live";
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
                  Start your camera to see the live preview. Glasses will appear here once try-on is connected.
                </p>
              </>
            )}
          </div>
        )}
      </div>

      {glasses && (
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
            <p className="text-sm text-neutral-500">
              This is the 3D model Mirror built from your photo. Showing it on your face in the live preview is the next
              step and isn&apos;t available in the browser yet.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
