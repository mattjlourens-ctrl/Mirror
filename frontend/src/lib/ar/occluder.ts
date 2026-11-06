/**
 * The head occluder from occlusion.py (canonical face + head proxy), for a
 * depth-only pass in WebGL.
 *
 * occlusion.py hides an accessory point only if it is more than
 * DEPTH_TOLERANCE_CM behind the head surface at the same pixel. The GPU
 * reproduces that rule exactly by sliding each occluder vertex that much
 * further from the camera along its own view ray: its pixel is unchanged and
 * its depth grows by the tolerance (OCCLUDER_VERTEX_SHADER; pushBack is the
 * same formula in TypeScript, for tests).
 */
import { DEPTH_TOLERANCE_CM, OCCLUDER_TRIANGLES, OCCLUDER_VERTICES } from "./generated/web-ar-data.ts";
import type { Vec3 } from "./pose.ts";

export type OccluderMesh = { positions: Float32Array; indices: Uint32Array };

/** Occluder in face-local cm. */
export function buildOccluder(): OccluderMesh {
  return { positions: Float32Array.from(OCCLUDER_VERTICES), indices: Uint32Array.from(OCCLUDER_TRIANGLES) };
}

/** A view-space point (three.js axes, camera looking down −Z) moved `tolerance` cm deeper along its ray. */
export function pushBack([x, y, z]: Vec3, tolerance: number = DEPTH_TOLERANCE_CM): Vec3 {
  const depth = -z;
  if (depth <= 0) return [x, y, z];
  const factor = (depth + tolerance) / depth;
  return [x * factor, y * factor, z * factor];
}

export const OCCLUDER_VERTEX_SHADER = /* glsl */ `
uniform float tolerance;
void main() {
  vec4 view = modelViewMatrix * vec4(position, 1.0);
  float depth = -view.z;
  if (depth > 0.0) view.xyz *= (depth + tolerance) / depth;
  gl_Position = projectionMatrix * view;
}
`;

export const OCCLUDER_FRAGMENT_SHADER = /* glsl */ `
void main() {
  gl_FragColor = vec4(0.0);
}
`;
