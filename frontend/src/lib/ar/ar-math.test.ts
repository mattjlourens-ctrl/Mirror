import { test } from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { CASES, FRAME_SIZE, OCCLUSION_CASES, OCCLUSION_POINTS_M, SAMPLE_MODEL_POINTS_M } from "./generated/pose-fixtures.ts";
import { ANCHOR_CM, DEFAULT_OFFSET_CM, DEPTH_TOLERANCE_CM, OCCLUDER_VERTICES } from "./generated/web-ar-data.ts";
import {
  FRONTAL_ROTATION,
  cameraIntrinsics,
  estimateHeadPose,
  eulerAngles,
  imagePointsFromLandmarks,
  multiply3,
  rodrigues,
  rotationVector,
  type HeadPose,
  type Mat3,
  type Point2,
  type Vec3,
} from "./pose.ts";
import {
  CV_TO_GL,
  NO_FIT,
  faceViewMatrix,
  glassesViewMatrix,
  modelToFace,
  multiply,
  poseMatrix,
  projectCameraPoint,
  projectionMatrix,
  transformPoint,
  type Fit,
} from "./transforms.ts";
import { autoFit, combineFits } from "./auto-fit.ts";
import { buildOccluder, pushBack } from "./occluder.ts";

const K = cameraIntrinsics(FRAME_SIZE[0], FRAME_SIZE[1]);
const points = (flat: readonly number[]): Point2[] => {
  const out: Point2[] = [];
  for (let i = 0; i < flat.length; i += 2) out.push([flat[i], flat[i + 1]]);
  return out;
};
/** Angle between two rotations, degrees. */
function rotationDifference(a: Mat3, b: readonly number[]): number {
  const bt = [b[0], b[3], b[6], b[1], b[4], b[7], b[2], b[5], b[8]];
  const v = rotationVector(multiply3(a, bt));
  return (Math.hypot(...v) * 180) / Math.PI;
}
function rotationFromEuler(yaw: number, pitch: number, roll: number): Mat3 {
  const d = Math.PI / 180;
  const ry = rodrigues([0, yaw * d, 0]);
  const rx = rodrigues([-pitch * d, 0, 0]);
  const rz = rodrigues([0, 0, -roll * d]);
  return multiply3(FRONTAL_ROTATION, multiply3(ry, multiply3(rx, rz)));
}
const poseOf = (rotation: Mat3, translation: Vec3): HeadPose => ({ rotation, translation, intrinsics: K, error: 0 });

test("solves the same poses as OpenCV's solvePnP in head_pose.py", () => {
  let solved = 0;
  for (const c of CASES) {
    const pose = estimateHeadPose(points(c.imagePoints), K);
    if (c.pose === null) {
      assert.equal(pose, null, "a pose Python rejects must be rejected");
      continue;
    }
    assert.ok(pose, `pose for ${JSON.stringify(c.truth)}`);
    solved++;
    assert.ok(rotationDifference(pose.rotation, c.pose.rotation) < 0.25, "rotation within 0.25°");
    pose.translation.forEach((value, i) => assert.ok(Math.abs(value - c.pose!.translation[i]) < 0.25, `t[${i}]`));
    eulerAngles(pose.rotation).forEach((angle, i) => assert.ok(Math.abs(angle - c.pose!.euler[i]) < 0.3, `euler ${i}`));
  }
  assert.equal(solved, 8);
});

test("recovers yaw, pitch and roll with head_pose.py's signs", () => {
  for (const c of CASES) {
    if (!c.truth || !c.pose) continue;
    const [yaw, pitch, roll] = eulerAngles(estimateHeadPose(points(c.imagePoints), K)!.rotation);
    assert.ok(Math.abs(yaw - c.truth.yaw) < 2.5, `yaw ${yaw} vs ${c.truth.yaw}`);
    assert.ok(Math.abs(pitch - c.truth.pitch) < 2.5, `pitch ${pitch} vs ${c.truth.pitch}`);
    assert.ok(Math.abs(roll - c.truth.roll) < 2.5, `roll ${roll} vs ${c.truth.roll}`);
  }
});

test("projects glasses points to the same pixels as the desktop app, with and without a fit", () => {
  for (const c of CASES) {
    if (!c.pose) continue;
    const pose = estimateHeadPose(points(c.imagePoints), K)!;
    for (const projection of c.pose.projections) {
      const toCamera = multiply(poseMatrix(pose), modelToFace(projection.fit as Fit));
      SAMPLE_MODEL_POINTS_M.forEach((p, i) => {
        const expected = projection.pixels[i];
        const actual = projectCameraPoint(transformPoint(toCamera, p as unknown as Vec3), K);
        if (expected === null) return assert.equal(actual, null);
        assert.ok(actual, "point projected");
        assert.ok(Math.hypot(actual[0] - expected[0], actual[1] - expected[1]) < 1.5,
          `pixel ${actual} vs ${expected}`);
      });
    }
  }
});

test("uses the previous pose to track, reaching the same answer", () => {
  const c = CASES.find((item) => item.truth?.yaw === 45)!;
  const fresh = estimateHeadPose(points(c.imagePoints), K)!;
  const nudged = poseOf(multiply3(rodrigues([0.05, -0.04, 0.02]), fresh.rotation), [fresh.translation[0] + 1, fresh.translation[1], fresh.translation[2] - 2]);
  const tracked = estimateHeadPose(points(c.imagePoints), K, nudged)!;
  assert.ok(rotationDifference(tracked.rotation, fresh.rotation) < 0.05);
});

test("the three.js projection matrix lands on the pinhole pixel", () => {
  const projection = new THREE.Matrix4().set(...(projectionMatrix(K, 1, 1000) as [
    number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number]));
  const toGl = new THREE.Matrix4().set(...(CV_TO_GL as [
    number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number]));
  for (const p of [[0, 0, 50], [10, -5, 40], [-12, 8, 80], [3, 3, 5]] as Vec3[]) {
    const ndc = new THREE.Vector3(...p).applyMatrix4(toGl).applyMatrix4(projection);
    const pixel = [(ndc.x + 1) / 2 * K.width, (1 - ndc.y) / 2 * K.height];
    const expected = projectCameraPoint(p, K)!;
    assert.ok(Math.hypot(pixel[0] - expected[0], pixel[1] - expected[1]) < 1e-6);
    assert.ok(ndc.z > -1 && ndc.z < 1, "inside the depth range");
  }
});

test("a frontal glTF model keeps its axes on the face: right is screen-right, up is up, front faces the camera", () => {
  const pose = poseOf(FRONTAL_ROTATION, [0, 0, 50]);
  const view = glassesViewMatrix(pose, NO_FIT);
  const origin = transformPoint(view, [0, 0, 0]);
  const right = transformPoint(view, [0.05, 0, 0]);
  const up = transformPoint(view, [0, 0.05, 0]);
  const front = transformPoint(view, [0, 0, 0.05]);
  assert.ok(right[0] > origin[0], "model +X is screen-right (three view +X)");
  assert.ok(up[1] > origin[1], "model +Y is up");
  assert.ok(front[2] > origin[2], "model +Z points at the camera (three view +Z)");
  // Origin sits at the nose bridge plus the calibrated offset, 50 cm away.
  const face = [0, 1, 2].map((i) => ANCHOR_CM[i] + DEFAULT_OFFSET_CM[i]);
  assert.ok(Math.abs(origin[0] - face[0]) < 1e-9 && Math.abs(origin[1] - face[1]) < 1e-9);
  assert.ok(Math.abs(origin[2] - (-(50 - face[2]))) < 1e-9);
  // glTF metres become centimetres: a 14 cm frame stays 14 cm.
  assert.ok(Math.abs(transformPoint(modelToFace(NO_FIT), [0.07, 0, 0])[0] - transformPoint(modelToFace(NO_FIT), [-0.07, 0, 0])[0] - 14) < 1e-9);
});

test("mirroring: landmarks from an unmirrored frame are rejected, not drawn inverted", () => {
  const frontal = CASES.find((c) => c.truth?.yaw === 0 && c.truth.pitch === 0)!;
  const pts = points(frontal.imagePoints);
  const screenRightEye = pts[1][0]; // landmark 263
  const screenLeftEye = pts[0][0]; // landmark 33
  assert.ok(screenRightEye > screenLeftEye, "in the mirrored frame landmark 263 is on screen-right");
  const unmirrored = pts.map(([x, y]) => [K.width - x, y] as Point2);
  assert.equal(estimateHeadPose(unmirrored, K), null);
});

test("turning the head towards screen-right is positive yaw", () => {
  const turned = CASES.find((c) => c.truth?.yaw === 25)!;
  const pose = estimateHeadPose(points(turned.imagePoints), K)!;
  assert.ok(eulerAngles(pose.rotation)[0] > 20);
  // ...and the nose tip (landmark 1) moves screen-right of the nose bridge (168).
  const pts = points(turned.imagePoints);
  assert.ok(pts[6][0] > pts[4][0]);
});

test("landmarks are read in the pose order and scaled to pixels", () => {
  const landmarks = Array.from({ length: 478 }, (_, i) => ({ x: i / 1000, y: 0.5 }));
  const result = imagePointsFromLandmarks(landmarks, 1000, 600)!;
  assert.deepEqual(result[0], [33, 300]);
  assert.deepEqual(result[1], [263, 300]);
  assert.equal(imagePointsFromLandmarks(landmarks.slice(0, 300), 1000, 600), null);
});

test("garbage and incomplete input is rejected", () => {
  assert.equal(estimateHeadPose(null, K), null);
  assert.equal(estimateHeadPose([[1, 2]], K), null);
  assert.equal(estimateHeadPose(Array.from({ length: 10 }, () => [NaN, 1] as Point2), K), null);
  assert.equal(estimateHeadPose(Array.from({ length: 10 }, () => [300, 300] as Point2), K), null);
});

test("auto-fit moves the lens centres to where the calibrated lenses sit", () => {
  assert.deepEqual(autoFit(null), NO_FIT);
  const fit = autoFit({ lensCentersCm: [[-3.3, 0.4], [3.3, 0.4]] });
  assert.ok(Math.abs(fit.y + 0.4) < 1e-9 && Math.abs(fit.x) < 1e-9 && fit.scale === 1);
  const limited = autoFit({ lensCentersCm: [[-3, 9], [3, 9]] });
  assert.equal(limited.y, -1.5);
  const combined = combineFits(fit, { ...NO_FIT, scale: 1.1, y: 0.2 });
  assert.ok(Math.abs(combined.y + 0.2) < 1e-9 && Math.abs(combined.scale - 1.1) < 1e-9);
  assert.equal(combineFits(NO_FIT, { ...NO_FIT, scale: 9 }).scale, 2);
});

test("the occluder is the Python head proxy, pushed back by the tolerance without moving on screen", () => {
  const { positions, indices } = buildOccluder();
  assert.deepEqual(Array.from(positions), OCCLUDER_VERTICES.map((v) => Math.fround(v)));
  assert.equal(indices.length % 3, 0);
  for (const p of [[3, -2, -40], [-10, 5, -55], [0, 0, -8]] as Vec3[]) {
    const moved = pushBack(p);
    assert.ok(Math.abs(-moved[2] - (-p[2] + DEPTH_TOLERANCE_CM)) < 1e-9, "depth grows by the tolerance");
    assert.ok(Math.abs(moved[0] / moved[2] - p[0] / p[2]) < 1e-12 && Math.abs(moved[1] / moved[2] - p[1] / p[2]) < 1e-12,
      "same pixel");
  }
});

// Occlusion, checked the way the GPU depth test works: the occluder's nearest
// front-facing surface along the ray to a glasses point, pushed back by the
// tolerance as the vertex shader does, must be nearer than the point to hide it.
const OCCLUDER = buildOccluder();
function occlusionAt(pose: HeadPose, modelPointM: Vec3): { hidden: boolean; marginCm: number } {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(OCCLUDER.positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(OCCLUDER.indices, 1));
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ side: THREE.FrontSide }));
  mesh.matrixAutoUpdate = false;
  mesh.matrix.set(...(faceViewMatrix(pose) as [
    number, number, number, number, number, number, number, number, number, number, number, number, number, number, number, number]));
  mesh.matrixWorld.copy(mesh.matrix);
  const target = new THREE.Vector3(...transformPoint(glassesViewMatrix(pose, NO_FIT), modelPointM));
  const cast = (point: THREE.Vector3) => new THREE.Raycaster(new THREE.Vector3(0, 0, 0), point.clone().normalize()).intersectObject(mesh);
  // Hidden if the surface on the ray, pushed back by the tolerance, is nearer than the point's depth.
  const hiddenAlong = (point: THREE.Vector3) => {
    const hit = cast(point)[0];
    return hit !== undefined && -pushBack([hit.point.x, hit.point.y, hit.point.z])[2] < -target.z;
  };
  const hidden = hiddenAlong(target);
  // Python samples a 320 px depth buffer (4 image pixels per cell) at the
  // truncated pixel, so where the verdict changes within one cell (outlines,
  // surfaces seen edge-on) either answer is right. Those points are borderline.
  const cell = (4 / K.fx) * -target.z;
  const borderline = [[cell, 0], [-cell, 0], [0, cell], [0, -cell]].some(
    ([dx, dy]) => hiddenAlong(target.clone().add(new THREE.Vector3(dx, dy, 0))) !== hidden);
  return { hidden, marginCm: borderline ? 0 : Infinity };
}

test("occlusion matches occlusion.py at frontal, three-quarter and side views", () => {
  let compared = 0;
  for (const { yaw, hidden } of OCCLUSION_CASES) {
    const pose = poseOf(rotationFromEuler(yaw, 0, 0), [0, 0, 50]);
    for (const [name, point] of Object.entries(OCCLUSION_POINTS_M)) {
      const result = occlusionAt(pose, point as unknown as Vec3);
      // Within a couple of millimetres of the surface the answer depends on
      // depth-buffer resolution (320 px in Python), so only clear cases count.
      if (result.marginCm < 0.2) continue;
      compared++;
      assert.equal(result.hidden, hidden[name as keyof typeof hidden], `${name} at yaw ${yaw} (margin ${result.marginCm.toFixed(2)} cm)`);
    }
  }
  assert.ok(compared >= 38, `compared ${compared} of 45 points`);
});

test("occlusion behaves as expected on the face", () => {
  const frontal = poseOf(rotationFromEuler(0, 0, 0), [0, 0, 50]);
  assert.equal(occlusionAt(frontal, [0, 0.015, 0.002]).hidden, false, "bridge visible");
  assert.equal(occlusionAt(frontal, [-0.072, 0, -0.13]).hidden, true, "temple tip behind the head");
  const side = poseOf(rotationFromEuler(75, 0, 0), [0, 0, 50]);
  assert.equal(occlusionAt(side, [0.04, -0.02, 0]).hidden, true, "far lens behind the nose");
  assert.equal(occlusionAt(side, [-0.072, 0, -0.06]).hidden, false, "near temple visible");
});
