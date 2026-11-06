# Mirror Website

Paste a glasses product link or upload a photo. Mirror's backend finds the product photo, builds a 3D model with the same deterministic pipeline as the desktop app, and the site shows the model in the browser. Nobody handles model files.

Built with Next.js (App Router), TypeScript, Tailwind CSS, Lucide icons and three.js. The backend is in [`../backend`](../backend).

> **Not built yet:** placing the glasses on your face in the browser. The site shows the generated model next to the live camera preview and says so. Live try-on currently runs in the desktop app (`uv run python main.py`).

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

| Command | Purpose |
|---|---|
| `npm run dev` | Development server |
| `npm run build` / `npm start` | Production build and server |
| `npm run lint` | ESLint |
| `npm test` | URL and image validation tests |

## How it works

```
Product link ─► backend fetches page safely ─► finds product photos ─► front-view check ─┐
Uploaded photo ─► validated ─────────────────────────────────────────────────────────────┤
                                                                                         ▼
           browser preview ◄─ GLB (temporary) ◄─ texture ◄─ geometry ◄─ segmentation ◄─ job queue
```

- **Product links.** The backend fetches the page with ordinary HTTP. It looks for photos in schema.org Product data, Open Graph and Twitter tags, `<img>` elements, and image URLs embedded in the page's scripts. Each photo is checked with OpenCV for a plain background, two lens openings and a wide silhouette. With one front view, generation starts straight away; with several, you choose; with none, you're asked to upload a photo instead. No machine learning is involved.
- **Safety.** Only http(s) on standard ports. Hosts must resolve to public addresses, and the connection goes to the address that was checked. Every redirect is re-checked, downloads are size-limited, and `robots.txt` is respected. There is no browser-automation fallback; blocked or script-only pages fall back to upload.
- **Jobs.** Generation runs in a bounded queue (503 when full). The site polls the job and shows "Preparing your glasses", then the model or a clear error with a retry. Photos and models live in a private temporary folder, expire after an hour, and are deleted on shutdown.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/jobs/from-image` | Multipart `image` (PNG/JPG, up to 10 MB) → 202 job |
| `POST /api/sources/from-url` | `{"url"}` → `{"status": "processing", "job", "candidates"}` or `{"status": "choose", "candidates"}` |
| `POST /api/jobs/from-candidate` | `{"candidate_id"}` → 202 job |
| `GET /api/jobs/{id}` | `status` (`queued`, `processing`, `succeeded`, `failed`), `source_image_url`, `model_url`, `error` |
| `GET /api/images/{id}`, `GET /api/models/{id}` | Stored photo or generated `.glb` (temporary) |

Errors are `{"detail": {"code", "message"}}`; messages are written for the person using the site.

Backend settings, all optional: `MIRROR_ALLOWED_ORIGINS` (comma-separated, default `http://localhost:3000`), `MIRROR_MAX_IMAGE_MB`, `MIRROR_STORAGE_TTL_SECONDS`, `MIRROR_MAX_PENDING_JOBS`, `MIRROR_WORKERS`, `MIRROR_STORAGE_DIR`.

## Deploying

The website deploys to Vercel: import the repository, set **Root Directory** to `frontend`, set `NEXT_PUBLIC_MIRROR_API_URL` to the backend's public HTTPS URL, and add a custom domain under **Settings → Domains**. The variable is built in, so redeploy after changing it.

The backend is a long-running Python process with local temporary storage. It needs a host that runs one (not Vercel functions). Set `MIRROR_ALLOWED_ORIGINS` to the website's domain.
