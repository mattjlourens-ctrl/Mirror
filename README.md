# Mirror — AR Virtual Try-On

Mirror turns a single product photo of glasses into a 3D model and lets you try it on live through your webcam. The glasses follow your head as you move and turn, and disappear behind your head where they should.

Glasses reconstruction uses classical computer vision and geometry, with no machine learning or generative AI, and the desktop 3D rendering pipeline is written from scratch in NumPy. That includes projection, rasterization, depth testing, texturing and occlusion; no game engine or graphics API is used. Face tracking uses MediaPipe, which is a machine-learning model, so Mirror as a whole is not ML-free.

A website (Next.js + FastAPI) turns a product link or uploaded photo into a model automatically and puts it on your face through the browser camera, using the same pose, placement and occlusion maths as the desktop app. Camera frames never leave the browser.

> **Status:** working prototype, under active development.

## Key Features

- **Real-time face tracking:** MediaPipe Face Mesh follows 478 facial landmarks from a webcam.
- **3D head-pose estimation:** OpenCV `solvePnP` against a canonical face model gives head rotation and position, and rejects unreliable poses.
- **Glasses that follow your head:** the model stays aligned through turning, tilting, nodding and moving closer or further away.
- **Image-to-3D generation:** a frontal product photo becomes a complete model with a curved front, temples and hinges, exported as `.glb`.
- **Website:** paste a product link or upload a photo; the backend finds the product photo and builds the model, and the browser shows it live on your face, with no files to manage.
- **No AI in reconstruction:** geometry comes from segmentation, contour tracing, Delaunay triangulation, Bézier curves and swept surfaces, so the same photo always gives the same model.
- **Photo-based appearance:** the frame front is textured with the original photo, and unseen surfaces get colours estimated from it.
- **Face occlusion:** a per-frame depth buffer of the head hides the temples as they pass behind it.

## How It Works

```
Product photo ──► Outline and openings ──► 3D geometry ──► Photo texture ──► .glb model
                                                                              │
Webcam ──► Face landmarks ──► Head pose ──► Place model on face ──► Render with depth and occlusion
```

1. **Segment the photo:** the glasses are separated from a white or transparent background.
2. **Trace the outline:** the frame outline and lens openings are traced into polygons; temples visible through the lenses are removed.
3. **Generate geometry:** the outline is triangulated, extruded and curved to wrap around the face. Temples are swept along Bézier curves, and hinges are added. Every part is validated before export.
4. **Transfer appearance:** each point on the front maps back to its pixel in the photo. Sides, back and temples get estimated colours.
5. **Track the face:** each webcam frame gives landmarks, which give a head pose.
6. **Render:** the model is projected with perspective, rasterized with a depth buffer and lit. Anything behind the head is hidden.

Algorithm details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Application language |
| MediaPipe | Facial landmark tracking and the canonical 3D face model |
| OpenCV | Camera capture, segmentation, contours, pose estimation, image processing |
| NumPy | Vectorized transforms, rasterization, depth buffering and shading |
| Trimesh | Mesh construction and GLB import/export |
| Shapely | Polygon processing and constrained Delaunay triangulation |
| Pillow | Embedding textures in GLB files |
| FastAPI, httpx | Web backend and SSRF-safe product-page fetching |
| Next.js, TypeScript, Tailwind CSS, three.js, MediaPipe Tasks (web) | Website and live in-browser try-on |
| uv, npm | Dependency management |

## Installation and Usage

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), and a webcam.

```bash
git clone https://github.com/mattjlourens-ctrl/Mirror.git
cd Mirror
uv sync
```

### Run the webcam app

```bash
uv run python main.py
```

The app starts with a simple 2D overlay; press **H** for 3D glasses. On macOS, allow camera access for your terminal or IDE under *System Settings → Privacy & Security → Camera* if no window appears.

### Generate a 3D model from a photo

```bash
uv run python image_to_3d/mirror_image_to_3d.py image_to_3d/glasses.jpg --complete \
    -o image_to_3d/output/example_3_photo.glb
```

Useful options:

| Option | Default | Purpose |
|---|---|---|
| `--frame-width` | 14.0 cm | Real frame width (a photo has no absolute scale) |
| `--frame-curvature` | 0.5 cm | How far the frame wraps back at the edges |
| `--temple-length` | 14.0 cm | Arm length from hinge to tip |
| `--nose-pads` | off | Add nose pads |
| `--no-texture` | off | Plain materials instead of the photo texture |
| `--hinge-material` | `match` | Hinges match the frame colour, or are `metal` |

Without `--complete`, only a flat front frame is generated. Run with `--help` for every option; the full list is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#generator-options).

### Try on the generated model

```bash
uv run python main.py --model image_to_3d/output/example_3_photo.glb
```

If the file is missing or invalid, Mirror prints the reason and shows the wireframe glasses instead. Saving to `example_3_photo.glb`, as above, also lets **M** load the model without `--model`.

### Run the website

Requires Node.js 20.9+. In one terminal, start the backend:

```bash
uv run uvicorn backend.app:app --port 8000
```

In another, start the website and open http://localhost:3000:

```bash
cd frontend
npm install
cp .env.example .env.local   # NEXT_PUBLIC_MIRROR_API_URL; npm run dev defaults to http://localhost:8000
npm run dev
```

See [frontend/README.md](frontend/README.md) for the API, configuration and deployment.

### Keyboard controls

| Key | Action |
|---|---|
| `H` | Show or hide 3D glasses |
| `M` | Switch between the generated model and the wireframe |
| `G` | Show or hide the 2D overlay |
| `O` | Toggle occlusion behind the head |
| `P` | Show 3D face axes |
| `D` | Debug panel: fit, resolution and FPS |
| `Q` | Quit |

**Fit adjustments:**

| Key | Adjustment |
|---|---|
| `J` / `L` | Left / right |
| `I` / `K` | Up / down |
| `U` / `N` | Forward / back |
| `+` / `-` | Size |
| `[` / `]` | Yaw |
| `;` / `'` | Pitch |
| `,` / `.` | Roll |
| `R` | Reset |
| `F` | Print the current fit as flags, such as `--fit-y -0.3`, to reuse with `main.py` |

## Project Architecture

```
Mirror/
├── main.py                   # App loop, command line and keyboard controls
├── head_pose.py              # Camera model and head-pose estimation
├── accessory_transform.py    # Placing accessories on the face and projecting them
├── occlusion.py              # Head depth buffer and the shared NumPy rasterizer
├── glb_model.py              # Loading, validating and caching .glb models
├── mesh_renderer.py          # Filled, lit, textured triangle renderer
├── fit_controls.py           # Live fit adjustments
├── glasses_3d.py             # Procedural wireframe glasses (fallback)
├── glasses.py, geometry.py   # 2D overlay and 2D helpers
├── validate_appearance.py    # Compares a generated model with its source photo
├── backend/                  # FastAPI service: uploads, product pages, jobs, model storage
├── tools/export_web_ar_data.py # Exports the face model and test fixtures for browser try-on
├── frontend/                 # Next.js website
├── image_to_3d/              # Photo → 3D model generator (independent of the live app)
│   ├── segmentation.py       #   Background removal and opening cleanup
│   ├── mesh.py, curvature.py #   Outline tracing, extrusion and frame curvature
│   ├── temples.py, hinges.py #   Temples, hinges and nose supports
│   ├── appearance.py         #   Texture, UV mapping and estimated materials
│   ├── assembly.py           #   Assembly and GLB export
│   └── mirror_image_to_3d.py #   Command line
├── tests/                    # Unit and integration tests
└── docs/ARCHITECTURE.md      # Technical reference
```

## Testing

```bash
uv run python -m unittest discover -s tests
cd frontend && npm run lint && npm test && npm run build
```

The Python suite has 610 tests, and all pass. The frontend tests check the browser try-on maths against the Python implementation. Python and the browser share one source: the face model, constants and fixtures are exported by `uv run python tools/export_web_ar_data.py`, and a test fails if the export is out of date. The Python tests cover:

- **Live app:** pose estimation, coordinate transforms, rasterization, depth testing, near-plane clipping, texturing, head occlusion at frontal, three-quarter and side angles, GLB loading and fit controls.
- **Generator:** segmentation, geometry generation and validation, texture extraction, UV mapping, GLB export and determinism.
- **Backend:** uploads and image validation, product-page photo extraction, SSRF protection (private addresses, redirects, DNS pinning), robots.txt, size limits, the job queue, temporary storage, and end-to-end reconstruction checked with the desktop app's own model loader.
- **Integration:** the real `main.py` loop runs with a simulated camera, simulated key presses and synthetic face landmarks, so the whole app is tested without a webcam.

## Limitations and Future Work

**Current limitations**

- **Estimated geometry:** only the frame outline and front appearance come from the photo. Depth, temples and hinges are parametric estimates.
- **Hidden surfaces:** one frontal photo can't show the sides, back or temples, so their colours are approximations. Lighting from the original photo stays baked into the texture.
- **Approximate fit:** the camera focal length is estimated, pose is solved against an average face, and there is no smoothing between frames.
- **Simplified occlusion:** the head shape behind the face has no ears or hair.
- **Website:** many retailers block automated access or build their pages with JavaScript, so product links often fall back to uploading a photo. Browser try-on has been tested with a simulated camera only; real-webcam testing is still needed (see [frontend/README.md](frontend/README.md#live-try-on)).
- **Rendering and performance:** about 26 fps at 640×480 and 16 fps at 1280×720 in testing on a busy machine. There's no lens or transparency rendering yet.

**Possible improvements** (not yet implemented)

- Saving a calibrated fit with each model
- Pose smoothing and faster rendering at higher resolutions
- Lens rendering and richer materials
- Support for photos with non-white backgrounds
- A glasses catalogue
