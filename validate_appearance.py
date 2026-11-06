"""Render a generated model next to its source photo and measure how well they match.

    uv run python validate_appearance.py [image] [--out DIR]

Writes, under DIR (default image_to_3d/output/inspection/appearance, which git ignores):
  textured.glb, untextured.glb  identical geometry, with and without the photo texture
  front_*.png, three_quarter_*.png, side_*.png, comparison.png
and prints the front-view metrics:
  silhouette IoU  -- overlap of the rendered frame with the photo's segmentation mask
  colour ΔE76     -- CIELAB distance between rendered and photo pixels, over frame pixels
                     at least 3 px inside both outlines (so edge blending doesn't count)
These measure agreement with one photo under the renderer's lighting; they are not
a measure of photorealism. The front view is rendered without the live app's
render-width limit, so differences come from the model rather than that speed setting.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from glb_model import load_glasses_model
from head_pose import FRONTAL_ROTATION, HeadPose, camera_matrix
from image_to_3d.pipeline import convert_image_to_glasses
import mesh_renderer
from mesh_renderer import render_mesh
from occlusion import load_head_occluder, render_depth_buffer
from accessory_transform import attachment_matrix
from glasses_3d import DEFAULT_GLASSES_ATTACHMENT

REPO = Path(__file__).resolve().parent
FRONT_DISTANCE_CM = 2000.0  # far enough that the front view is effectively orthographic
VIEW_SIZE = (960, 720)
EDGE_MARGIN_PX = 3


def pose(rotation, translation, camera):
    vector, _ = cv2.Rodrigues(rotation)
    return HeadPose(vector, np.array(translation, float).reshape(3, 1), rotation, camera)


def head_rotation(yaw=0.0, pitch=0.0):
    turn, _ = cv2.Rodrigues(np.radians((0.0, yaw, 0.0)))
    tilt, _ = cv2.Rodrigues(np.radians((-pitch, 0.0, 0.0)))
    return FRONTAL_ROTATION @ turn @ tilt


def render_with_alpha(model, size, model_to_face, head_pose, depth_buffer=None):
    """(on white, coverage 0..1): rendering on black and white separates colour and coverage."""
    width, height = size
    white = np.full((height, width, 3), 255, np.uint8)
    black = np.zeros((height, width, 3), np.uint8)
    render_mesh(white, model, model_to_face, head_pose, depth_buffer)
    render_mesh(black, model, model_to_face, head_pose, depth_buffer)
    alpha = 1.0 - (white.astype(np.float32) - black.astype(np.float32)).mean(axis=2) / 255.0
    return white, np.clip(alpha, 0.0, 1.0)


def front_view(model, appearance, image_shape):
    """The model seen exactly as the photo was taken: same scale, same position."""
    mapping = appearance.mapping
    pixels_per_cm = mapping.image_scale / mapping.cm_per_working
    (center_x, center_y), = mapping.cm_to_image(np.zeros((1, 2)))
    camera = np.array([[pixels_per_cm[0] * FRONT_DISTANCE_CM, 0, center_x],
                       [0, pixels_per_cm[1] * FRONT_DISTANCE_CM, center_y], [0, 0, 1]])
    head_pose = pose(FRONTAL_ROTATION, (0, 0, FRONT_DISTANCE_CM), camera)
    height, width = image_shape[:2]
    limit = mesh_renderer.MAX_RENDER_WIDTH
    mesh_renderer.MAX_RENDER_WIDTH = width
    try:
        return render_with_alpha(model, (width, height), np.eye(4), head_pose)
    finally:
        mesh_renderer.MAX_RENDER_WIDTH = limit


def delta_e(a_bgr, b_bgr):
    to_lab = lambda image: cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    return np.linalg.norm(to_lab(a_bgr) - to_lab(b_bgr), axis=-1)


def front_metrics(photo, mask, rendered, alpha):
    drawn = alpha > 0.5
    frame = mask > 0
    iou = np.count_nonzero(drawn & frame) / np.count_nonzero(drawn | frame)
    kernel = np.ones((2 * EDGE_MARGIN_PX + 1,) * 2, np.uint8)
    inner = cv2.erode((drawn & frame).astype(np.uint8), kernel) > 0
    error = delta_e(rendered, photo[..., :3])[inner]
    return iou, error


def head_image(occluder, head_pose, size):
    """The head proxy, shaded by slope, as a stand-in face."""
    width, height = size
    depth = render_depth_buffer(occluder, head_pose, width, height, buffer_width=width).depth
    inside = np.isfinite(depth)
    filled = np.where(inside, depth, depth[inside].max()).astype(np.float32)
    gx = cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3)
    shade = np.clip(185 - 4 * gx - 6 * gy, 80, 235)
    image = np.full((height, width, 3), 236, np.uint8)
    image[inside] = np.stack((shade * 0.75, shade * 0.85, shade), axis=-1)[inside].astype(np.uint8)
    return image


def on_head(model, occluder, yaw, pitch=0.0):
    camera = camera_matrix(*VIEW_SIZE)
    head_pose = pose(head_rotation(yaw, pitch), (0, 0, 38.0), camera)
    image = head_image(occluder, head_pose, VIEW_SIZE)
    attach = attachment_matrix(DEFAULT_GLASSES_ATTACHMENT)
    render_mesh(image, model, attach, head_pose,
                render_depth_buffer(occluder, head_pose, *VIEW_SIZE))
    return image


def crop_to(image, content, margin=20):
    ys, xs = np.nonzero(content)
    return image[max(0, ys.min() - margin):ys.max() + margin, max(0, xs.min() - margin):xs.max() + margin]


def labelled(image, text, width=640):
    scaled = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])),
                        interpolation=cv2.INTER_AREA)
    cv2.putText(scaled, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 190), 2, cv2.LINE_AA)
    return scaled


def stack_rows(rows):
    width = max(sum(tile.shape[1] for tile in row) for row in rows)
    out = []
    for row in rows:
        height = max(tile.shape[0] for tile in row)
        line = np.full((height, width, 3), 255, np.uint8)
        x = 0
        for tile in row:
            line[:tile.shape[0], x:x + tile.shape[1]] = tile
            x += tile.shape[1]
        out.append(line)
    return np.vstack(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", nargs="?", type=Path, default=REPO / "image_to_3d" / "glasses.jpg")
    parser.add_argument("--out", type=Path,
                        default=REPO / "image_to_3d" / "output" / "inspection" / "appearance")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    textured = convert_image_to_glasses(args.image, args.out / "textured.glb")
    convert_image_to_glasses(args.image, args.out / "untextured.glb", texture=False)
    load_glasses_model.cache_clear()
    models = {"textured": load_glasses_model(args.out / "textured.glb"),
              "untextured": load_glasses_model(args.out / "untextured.glb")}
    photo = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    occluder = load_head_occluder()

    print(f"Front view against {args.image.name} (frame pixels {EDGE_MARGIN_PX}+ px inside both outlines):")
    tiles = {"photo": crop_to(photo, textured.mask)}
    for name, model in models.items():
        rendered, alpha = front_view(model, textured.appearance, photo.shape)
        iou, error = front_metrics(photo, textured.mask, rendered, alpha)
        print(f"  {name:10s} silhouette IoU {iou:.4f}   ΔE76 mean {error.mean():5.2f}  "
              f"median {np.median(error):5.2f}  95th percentile {np.percentile(error, 95):5.2f}  "
              f"({error.size} px)")
        tiles[f"front {name}"] = crop_to(rendered, textured.mask)
        cv2.imwrite(str(args.out / f"front_{name}.png"), rendered)
        for view, yaw in (("three_quarter", 35.0), ("side", 80.0)):
            image = on_head(model, occluder, yaw)
            cv2.imwrite(str(args.out / f"{view}_{name}.png"), image)
            tiles[f"{view.replace('_', '-')} {name}"] = image[90:560, 140:820]

    comparison = stack_rows([
        [labelled(tiles["photo"], "source photo"), labelled(tiles["front textured"], "front, textured"),
         labelled(tiles["front untextured"], "front, untextured")],
        [labelled(tiles["three-quarter textured"], "three-quarter, textured"),
         labelled(tiles["side textured"], "side, textured"),
         labelled(tiles["three-quarter untextured"], "three-quarter, untextured")]])
    cv2.imwrite(str(args.out / "comparison.png"), comparison)
    print(f"Images and models written to {args.out}")


if __name__ == "__main__":
    main()
