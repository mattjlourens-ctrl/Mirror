/**
 * Placing a glasses model on the face: TypeScript port of
 * accessory_transform.py and fit_controls.fit_matrix, plus the conversion to
 * three.js. Matrices are 4x4, row-major (three's Matrix4.set order).
 *
 *   glTF model (m, X right, Y up, Z forward)
 *     | × 100, then fit matrix (anchor at nose bridge + offset, scale, yaw·pitch·roll)
 *   face-local (cm, canonical face axes)
 *     | head pose [R | t]
 *   camera (cm, OpenCV: X right, Y down, Z forward)
 *     | CV_TO_GL = diag(1, −1, −1): three's camera looks down −Z with Y up
 *   three.js view space, projected with the same pinhole intrinsics
 */
import { ANCHOR_CM, DEFAULT_OFFSET_CM, FIT_LIMITS, MIN_DEPTH_CM } from "./generated/web-ar-data.ts";
import type { HeadPose, Intrinsics, Point2, Vec3 } from "./pose.ts";

export type Mat4 = number[];

/** Manual adjustments on top of the attachment, as in fit_controls.FitAdjustment. */
export type Fit = { scale: number; x: number; y: number; z: number; pitch: number; yaw: number; roll: number };
export const NO_FIT: Fit = { scale: 1, x: 0, y: 0, z: 0, pitch: 0, yaw: 0, roll: 0 };

export const GLTF_TO_CM = 100;
export const CV_TO_GL: Mat4 = [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1];

export function identity(): Mat4 {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

export function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Array(16).fill(0);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++)
      for (let k = 0; k < 4; k++) out[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
  return out;
}

export function translation([x, y, z]: Vec3): Mat4 {
  return [1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1];
}

export function scaling(s: number): Mat4 {
  return [s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1];
}

function rotationAbout(axis: 0 | 1 | 2, degrees: number): Mat4 {
  const a = (degrees * Math.PI) / 180;
  const c = Math.cos(a);
  const s = Math.sin(a);
  if (axis === 0) return [1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1];
  if (axis === 1) return [c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1];
  return [c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

const clamp = (value: number, low: number, high: number) => Math.min(high, Math.max(low, value));

/** Keep a fit inside fit_controls' limits. */
export function clampFit(fit: Fit): Fit {
  const offset = FIT_LIMITS.maxOffsetCm;
  const angle = FIT_LIMITS.maxRotationDegrees;
  return {
    scale: clamp(fit.scale, FIT_LIMITS.minScale, FIT_LIMITS.maxScale),
    x: clamp(fit.x, -offset, offset),
    y: clamp(fit.y, -offset, offset),
    z: clamp(fit.z, -offset, offset),
    pitch: clamp(fit.pitch, -angle, angle),
    yaw: clamp(fit.yaw, -angle, angle),
    roll: clamp(fit.roll, -angle, angle),
  };
}

/** Model cm -> face-local cm (fit_controls.fit_matrix with the default glasses attachment). */
export function fitMatrix(fit: Fit): Mat4 {
  const position: Vec3 = [0, 1, 2].map((i) => ANCHOR_CM[i] + DEFAULT_OFFSET_CM[i] + [fit.x, fit.y, fit.z][i]) as Vec3;
  const rotation = multiply(multiply(rotationAbout(1, fit.yaw), rotationAbout(0, fit.pitch)), rotationAbout(2, fit.roll));
  return multiply(translation(position), multiply(scaling(fit.scale), rotation));
}

/** glTF model (metres) -> face-local cm. */
export function modelToFace(fit: Fit): Mat4 {
  return multiply(fitMatrix(fit), scaling(GLTF_TO_CM));
}

/** Face-local cm -> camera cm (HeadPose.transform). */
export function poseMatrix(pose: HeadPose): Mat4 {
  const r = pose.rotation;
  const t = pose.translation;
  return [r[0], r[1], r[2], t[0], r[3], r[4], r[5], t[1], r[6], r[7], r[8], t[2], 0, 0, 0, 1];
}

/** Object matrix for three.js: glTF model -> three view space. */
export function glassesViewMatrix(pose: HeadPose, fit: Fit): Mat4 {
  return multiply(CV_TO_GL, multiply(poseMatrix(pose), modelToFace(fit)));
}

/** Object matrix for three.js: face-local cm (head occluder) -> three view space. */
export function faceViewMatrix(pose: HeadPose): Mat4 {
  return multiply(CV_TO_GL, poseMatrix(pose));
}

export function transformPoint(m: Mat4, [x, y, z]: Vec3): Vec3 {
  return [
    m[0] * x + m[1] * y + m[2] * z + m[3],
    m[4] * x + m[5] * y + m[6] * z + m[7],
    m[8] * x + m[9] * y + m[10] * z + m[11],
  ];
}

/** Pinhole projection of a camera-space point (cm), or null within MIN_DEPTH_CM of the camera. */
export function projectCameraPoint([x, y, z]: Vec3, k: Intrinsics): Point2 | null {
  if (z < MIN_DEPTH_CM) return null;
  return [k.fx * (x / z) + k.cx, k.fy * (y / z) + k.cy];
}

/**
 * OpenGL projection matrix (row-major) equivalent to the pinhole intrinsics:
 * a view-space point lands on the same pixel the OpenCV projection gives.
 */
export function projectionMatrix(k: Intrinsics, near: number, far: number): Mat4 {
  const { fx, fy, cx, cy, width: w, height: h } = k;
  return [
    (2 * fx) / w, 0, 1 - (2 * cx) / w, 0,
    0, (2 * fy) / h, (2 * cy) / h - 1, 0,
    0, 0, -(far + near) / (far - near), (-2 * far * near) / (far - near),
    0, 0, -1, 0,
  ];
}
