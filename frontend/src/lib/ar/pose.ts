/**
 * Head pose from face landmarks: a TypeScript port of head_pose.py.
 *
 * Same conventions as the desktop app:
 * - Image: pixels of the DISPLAYED (mirrored) frame, x right, y down.
 * - Camera: OpenCV axes, cm. X right, Y down, Z forward (depth).
 * - Face model: MediaPipe's canonical face, cm. X towards screen-right, Y up,
 *   Z out of the face. point_camera = R · point_model + t.
 * Same ten landmarks, same uncalibrated pinhole camera (focal = frame width),
 * same reprojection-error rejection. OpenCV's solvePnP is replaced by a
 * Levenberg–Marquardt fit of the reprojection error, started from the
 * previous frame's pose and from a few frontal guesses.
 */
import {
  FOCAL_LENGTH_SCALE,
  MAX_REPROJECTION_ERROR_RATIO,
  MODEL_POINTS,
  POSE_LANDMARK_IDS,
} from "./generated/web-ar-data.ts";

export type Vec3 = [number, number, number];
/** 3x3 matrix, row-major. */
export type Mat3 = number[];
export type Point2 = [number, number];

export type Intrinsics = { fx: number; fy: number; cx: number; cy: number; width: number; height: number };

export type HeadPose = {
  rotation: Mat3; // model -> camera
  translation: Vec3; // model origin in camera space, cm
  intrinsics: Intrinsics;
  /** Mean reprojection error, pixels. */
  error: number;
};

const MODEL: Vec3[] = [];
for (let i = 0; i < MODEL_POINTS.length; i += 3) MODEL.push([MODEL_POINTS[i], MODEL_POINTS[i + 1], MODEL_POINTS[i + 2]]);

/** Approximate pinhole intrinsics for an uncalibrated webcam (head_pose.camera_matrix). */
export function cameraIntrinsics(width: number, height: number): Intrinsics {
  const focal = FOCAL_LENGTH_SCALE * width;
  return { fx: focal, fy: focal, cx: width / 2, cy: height / 2, width, height };
}

/** Pixel positions of the pose landmarks, or null if the face has too few landmarks. */
export function imagePointsFromLandmarks(
  landmarks: ReadonlyArray<{ x: number; y: number }>,
  width: number,
  height: number,
): Point2[] | null {
  if (landmarks.length <= Math.max(...POSE_LANDMARK_IDS)) return null;
  return POSE_LANDMARK_IDS.map((id) => [landmarks[id].x * width, landmarks[id].y * height] as Point2);
}

/** Rotation matrix from an axis-angle vector (cv2.Rodrigues). */
export function rodrigues(v: Vec3): Mat3 {
  const theta = Math.hypot(v[0], v[1], v[2]);
  if (theta < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const [x, y, z] = [v[0] / theta, v[1] / theta, v[2] / theta];
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const k = 1 - c;
  return [
    c + x * x * k, x * y * k - z * s, x * z * k + y * s,
    y * x * k + z * s, c + y * y * k, y * z * k - x * s,
    z * x * k - y * s, z * y * k + x * s, c + z * z * k,
  ];
}

/** Axis-angle vector from a rotation matrix (inverse of rodrigues). */
export function rotationVector(r: Mat3): Vec3 {
  const cos = Math.min(1, Math.max(-1, (r[0] + r[4] + r[8] - 1) / 2));
  const theta = Math.acos(cos);
  if (theta < 1e-9) return [0, 0, 0];
  if (Math.PI - theta < 1e-6) {
    // Near 180 degrees: take the axis from the diagonal.
    const x = Math.sqrt(Math.max(0, (r[0] + 1) / 2));
    const y = Math.sqrt(Math.max(0, (r[4] + 1) / 2)) * (r[1] >= 0 ? 1 : -1);
    const z = Math.sqrt(Math.max(0, (r[8] + 1) / 2)) * (r[2] >= 0 ? 1 : -1);
    return [x * theta, y * theta, z * theta];
  }
  const f = theta / (2 * Math.sin(theta));
  return [(r[7] - r[5]) * f, (r[2] - r[6]) * f, (r[3] - r[1]) * f];
}

export function multiply3(a: Mat3, b: Mat3): Mat3 {
  const out = new Array(9).fill(0);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) out[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  return out;
}

function project(r: Mat3, t: Vec3, p: Vec3, k: Intrinsics): Point2 {
  const x = r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0];
  const y = r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1];
  const z = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2];
  return [k.fx * (x / z) + k.cx, k.fy * (y / z) + k.cy];
}

function residuals(params: number[], points: Point2[], k: Intrinsics): number[] {
  const r = rodrigues([params[0], params[1], params[2]]);
  const t: Vec3 = [params[3], params[4], params[5]];
  const out: number[] = [];
  MODEL.forEach((p, i) => {
    const [u, v] = project(r, t, p, k);
    out.push(u - points[i][0], v - points[i][1]);
  });
  return out;
}

/** Translation that best fits the points for a fixed rotation (linear least squares). */
function translationFor(r: Mat3, points: Point2[], k: Intrinsics): Vec3 {
  // fx·tx − (u−cx)·tz = (u−cx)·(Rp)z − fx·(Rp)x, and the same for y.
  const ata = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  const atb = [0, 0, 0];
  MODEL.forEach((p, i) => {
    const rp = [0, 1, 2].map((row) => r[row * 3] * p[0] + r[row * 3 + 1] * p[1] + r[row * 3 + 2] * p[2]);
    const du = points[i][0] - k.cx;
    const dv = points[i][1] - k.cy;
    const rows: [number[], number][] = [
      [[k.fx, 0, -du], du * rp[2] - k.fx * rp[0]],
      [[0, k.fy, -dv], dv * rp[2] - k.fy * rp[1]],
    ];
    for (const [a, b] of rows) {
      for (let m = 0; m < 3; m++) {
        atb[m] += a[m] * b;
        for (let n = 0; n < 3; n++) ata[m * 3 + n] += a[m] * a[n];
      }
    }
  });
  return solve(ata, atb, 3) as Vec3;
}

/** Solve a small dense system by Gaussian elimination with partial pivoting. */
function solve(a: number[], b: number[], n: number): number[] {
  const m = a.slice();
  const x = b.slice();
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) if (Math.abs(m[row * n + col]) > Math.abs(m[pivot * n + col])) pivot = row;
    if (Math.abs(m[pivot * n + col]) < 1e-12) return new Array(n).fill(NaN);
    if (pivot !== col) {
      for (let k = 0; k < n; k++) [m[col * n + k], m[pivot * n + k]] = [m[pivot * n + k], m[col * n + k]];
      [x[col], x[pivot]] = [x[pivot], x[col]];
    }
    for (let row = col + 1; row < n; row++) {
      const f = m[row * n + col] / m[col * n + col];
      for (let k = col; k < n; k++) m[row * n + k] -= f * m[col * n + k];
      x[row] -= f * x[col];
    }
  }
  for (let row = n - 1; row >= 0; row--) {
    let sum = x[row];
    for (let k = row + 1; k < n; k++) sum -= m[row * n + k] * x[k];
    x[row] = sum / m[row * n + row];
  }
  return x;
}

const sumSquares = (values: number[]) => values.reduce((s, v) => s + v * v, 0);

/** Levenberg–Marquardt on [rotation vector, translation]. */
function refine(initial: number[], points: Point2[], k: Intrinsics): number[] {
  let params = initial.slice();
  let res = residuals(params, points, k);
  let cost = sumSquares(res);
  let lambda = 1e-3;
  for (let iteration = 0; iteration < 50; iteration++) {
    // Numeric Jacobian (2N x 6); ten points make this cheap.
    const jac: number[][] = [];
    for (let j = 0; j < 6; j++) {
      const step = j < 3 ? 1e-6 : 1e-4;
      const shifted = params.slice();
      shifted[j] += step;
      jac.push(residuals(shifted, points, k).map((value, i) => (value - res[i]) / step));
    }
    const jtj = new Array(36).fill(0);
    const jtr = new Array(6).fill(0);
    for (let a = 0; a < 6; a++) {
      for (let i = 0; i < res.length; i++) jtr[a] += jac[a][i] * res[i];
      for (let b = 0; b < 6; b++) for (let i = 0; i < res.length; i++) jtj[a * 6 + b] += jac[a][i] * jac[b][i];
    }
    let improved = false;
    while (lambda < 1e10) {
      const damped = jtj.slice();
      for (let d = 0; d < 6; d++) damped[d * 7] *= 1 + lambda;
      const delta = solve(damped, jtr, 6);
      const candidate = params.map((p, i) => p - delta[i]);
      const candidateRes = residuals(candidate, points, k);
      const candidateCost = sumSquares(candidateRes);
      if (Number.isFinite(candidateCost) && candidateCost < cost) {
        const change = Math.hypot(...delta);
        params = candidate;
        res = candidateRes;
        lambda = Math.max(lambda / 10, 1e-9);
        improved = cost - candidateCost > 1e-10 * cost && change > 1e-10;
        cost = candidateCost;
        break;
      }
      lambda *= 10;
    }
    if (!improved) break;
  }
  return params;
}

// Model -> camera rotation for a face looking straight at the camera (180° about X).
export const FRONTAL_ROTATION: Mat3 = [1, 0, 0, 0, -1, 0, 0, 0, -1];
// Starting yaws for the search when there's no previous pose (degrees).
const START_YAWS = [0, -40, 40];

function yawRotation(degrees: number): Mat3 {
  const a = (degrees * Math.PI) / 180;
  return multiply3(FRONTAL_ROTATION, [Math.cos(a), 0, Math.sin(a), 0, 1, 0, -Math.sin(a), 0, Math.cos(a)]);
}

/**
 * Solve for the head pose, or null if it can't be trusted (estimate_head_pose).
 * `previous` (last frame's pose) speeds up and stabilises tracking.
 */
export function estimateHeadPose(
  points: Point2[] | null,
  intrinsics: Intrinsics,
  previous: HeadPose | null = null,
): HeadPose | null {
  if (!points || points.length !== MODEL.length || !points.every((p) => Number.isFinite(p[0]) && Number.isFinite(p[1]))) {
    return null;
  }
  const starts: number[][] = [];
  if (previous) starts.push([...rotationVector(previous.rotation), ...previous.translation]);
  for (const yaw of START_YAWS) {
    const r = yawRotation(yaw);
    starts.push([...rotationVector(r), ...translationFor(r, points, intrinsics)]);
  }

  let best: { params: number[]; cost: number } | null = null;
  for (const start of starts) {
    if (!start.every(Number.isFinite)) continue;
    const params = refine(start, points, intrinsics);
    const cost = sumSquares(residuals(params, points, intrinsics));
    if (params[5] > 0 && Number.isFinite(cost) && (!best || cost < best.cost)) best = { params, cost };
    // A tracked pose that still fits well needs no other starts.
    if (previous && best && meanError(best.params, points, intrinsics) < 1) break;
  }
  if (!best) return null;

  const error = meanError(best.params, points, intrinsics);
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const faceSize = Math.hypot(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  if (faceSize === 0 || error > MAX_REPROJECTION_ERROR_RATIO * faceSize) return null;

  const rotation = rodrigues([best.params[0], best.params[1], best.params[2]]);
  // A face turned away from the camera can't be seen; such a fit means the
  // landmarks are mirrored or mislabelled, so drawing it would be inverted.
  if (multiply3(FRONTAL_ROTATION, rotation)[8] <= 0) return null;

  return {
    rotation,
    translation: [best.params[3], best.params[4], best.params[5]],
    intrinsics,
    error,
  };
}

function meanError(params: number[], points: Point2[], k: Intrinsics): number {
  const res = residuals(params, points, k);
  let total = 0;
  for (let i = 0; i < res.length; i += 2) total += Math.hypot(res[i], res[i + 1]);
  return total / (res.length / 2);
}

/**
 * (yaw, pitch, roll) in degrees relative to looking straight at the camera,
 * with head_pose.euler_angles' signs: yaw > 0 nose towards screen-right,
 * pitch > 0 nose up, roll > 0 head tilts clockwise.
 */
export function eulerAngles(rotation: Mat3): [number, number, number] {
  const r = multiply3(FRONTAL_ROTATION, rotation);
  const deg = 180 / Math.PI;
  const yaw = Math.atan2(r[2], r[8]);
  const pitch = Math.asin(Math.min(1, Math.max(-1, r[5])));
  const roll = -Math.atan2(r[3], r[4]);
  return [yaw * deg, pitch * deg, roll * deg];
}
