# Mirror Website

Paste a glasses product link or upload a photo. Mirror's backend finds the product photo and builds a 3D model with the same deterministic pipeline as the desktop app. The site then puts the glasses on your face through your camera. Nobody handles model files.

Built with Next.js (App Router), TypeScript, Tailwind CSS, Lucide icons, three.js and MediaPipe Tasks for the web. The backend is in [`../backend`](../backend).

## Getting started

Requires Node.js 20.9+ and, for the backend, Python 3.11 with uv.

```bash
# Terminal 1, from the repository root
uv run uvicorn backend.app:app --port 8000

# Terminal 2
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open http://localhost:3000. The camera works only on `localhost` or over HTTPS.

`npm run dev` talks to the backend at `NEXT_PUBLIC_MIRROR_API_URL`, or `http://localhost:8000` if that's unset. A production build (`npm run build`) needs the variable set when it builds. After editing `.env.local`, restart `npm run dev`. If the backend isn't running, the site says so, with the command to start it.

| Command | Purpose |
|---|---|
| `npm run dev` | Development server |
| `npm run build` / `npm start` | Production build and server |
| `npm run lint` | ESLint |
| `npm test` | URL and image validation, and try-on maths checked against the Python app |

`npm run dev` and `npm run build` first copy the MediaPipe runtime and `../face_landmarker.task` into `public/mediapipe/`; that folder is git-ignored.

## How it works

```
Product link ─► backend fetches page safely ─► finds product photos ─► front-view check ─┐
Uploaded photo ─► validated ─────────────────────────────────────────────────────────────┤
                                                                                         ▼
           live try-on ◄─── GLB (temporary) ◄─ texture ◄─ geometry ◄─ segmentation ◄─ job queue
```

- **Product links.** The backend fetches the page with ordinary HTTP. It looks for photos in schema.org Product data, Open Graph and Twitter tags, `<img>` elements, and image URLs embedded in the page's scripts. Each photo is checked with OpenCV for a plain background, two lens openings and a wide silhouette. With one front view, generation starts straight away; with several, you choose; with none, you're asked to upload a photo instead. No machine learning is involved.
- **Safety.** Only http(s) on standard ports. Hosts must resolve to public addresses, and the connection goes to the address that was checked. Every redirect is re-checked, downloads are size-limited, and `robots.txt` is respected. There is no browser-automation fallback; blocked or script-only pages fall back to upload.
- **Jobs.** Generation runs in a bounded queue (503 when full). The site polls the job and shows "Preparing your glasses", then the model or a clear error with a retry. Photos and models live in a private temporary folder, expire after an hour, and are deleted on shutdown.

## Live try-on

Once the camera is on and a model is ready, the glasses appear on your face. You can switch to a different pair while the camera keeps running.

- **Tracking.** MediaPipe Face Landmarker, a pretrained machine-learning model, runs in the browser (WebAssembly; GPU when available). It uses the same model and version (0.10.21) as the desktop app, and frames are mirrored before tracking, as on desktop. Tracking runs only while the camera is on, and only on new camera frames. No frame is sent anywhere.
- **Pose and placement.** A TypeScript port of the desktop maths (`src/lib/ar/`): the same 10 landmarks and canonical face, the same camera model, and a Levenberg–Marquardt PnP solve in place of OpenCV's `solvePnP`. It uses the same reprojection-error rejection and the same anchor, offset and fit matrices. Poses facing away from the camera are rejected, which catches mirrored input.
- **Rendering.** three.js draws the textured GLB on a transparent canvas over the video. The projection is built from the same intrinsics, so each point lands on the pixel the desktop app would use.
- **Occlusion.** A depth-only copy of the desktop app's head occluder (face mesh plus head proxy). Its vertices are pushed 0.5 cm back along their view rays, which reproduces the desktop rule exactly: hide a point only if it is more than 0.5 cm behind the head.
- **Fit.** Automatic: the backend returns the lens-opening centres, and the browser centres the lenses on the eyes. Manual: up/down, smaller/larger and reset.
- **Statuses.** Starting face tracking, Preparing glasses, Loading model, Face not detected, Ready to try on, Unable to process this image, and Camera permission required.
- **Debugging.** Add `?debug=1` to the URL to show fps and per-stage timings. The same numbers are on `window.__mirrorTryOn`.

The shared face model, constants and test fixtures come from Python: run `uv run python tools/export_web_ar_data.py` from the repository root after changing `head_pose.py`, `accessory_transform.py` or `occlusion.py`. A Python test fails while the export is stale.

**Measured** with Chrome's fake camera (1280×720, 30 fps) on the development Mac (Intel Iris Plus 655 GPU): about 27–29 fps end to end, against a 30 fps source. Per frame, tracking took about 18 ms, pose under 1 ms and drawing 0.2 ms. With software WebGL (no GPU), it ran at about 3 fps.

**Still needs a real webcam.** Automated tests used a still photo, moved and rotated in the image plane. Check by hand, with the fit controls at reset:

1. The glasses sit on the nose bridge, with the lenses over your eyes. They should fit, not float.
2. Turn left and right to about 45°. The glasses turn with you, and the far temple disappears behind your head.
3. Turn fully to the side. The far lens hides behind your nose and face.
4. Nod and tilt. Nothing is mirrored, and moving right moves the glasses right on screen.
5. Leave the frame and come back. You see "Face not detected", then the glasses return.
6. Watch the `?debug=1` fps on your own machine.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/jobs/from-image` | Multipart `image` (PNG/JPG, up to 10 MB) → 202 job |
| `POST /api/sources/from-url` | `{"url"}` → `{"status": "processing", "job", "candidates"}` or `{"status": "choose", "candidates"}` |
| `POST /api/jobs/from-candidate` | `{"candidate_id"}` → 202 job |
| `GET /api/jobs/{id}` | `status` (`queued`, `processing`, `succeeded`, `failed`), `source_image_url`, `model_url`, `fit` (`lens_centers_cm`), `error` |
| `GET /api/images/{id}`, `GET /api/models/{id}` | Stored photo or generated `.glb` (temporary) |

Errors are `{"detail": {"code", "message"}}`; messages are written for the person using the site.

Backend settings, all optional: `MIRROR_ALLOWED_ORIGINS` (comma-separated, default `http://localhost:3000,http://127.0.0.1:3000`), `MIRROR_ALLOW_LOCAL_ORIGINS` (default `1`: also accept `http://localhost` and `http://127.0.0.1` on any port, for local development; set `0` in production), `MIRROR_MAX_IMAGE_MB`, `MIRROR_STORAGE_TTL_SECONDS`, `MIRROR_MAX_PENDING_JOBS`, `MIRROR_WORKERS`, `MIRROR_STORAGE_DIR`.

## Deploying

The website deploys to Vercel: import the repository, set **Root Directory** to `frontend`, set `NEXT_PUBLIC_MIRROR_API_URL` to the backend's public HTTPS URL, and add a custom domain under **Settings → Domains**. The variable is built in, so redeploy after changing it.

The backend is a long-running Python process with local temporary storage. It needs a host that runs one (not Vercel functions). Set `MIRROR_ALLOWED_ORIGINS` to the website's domain.
