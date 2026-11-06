/**
 * Initial fit for a generated model, from where its lenses are.
 *
 * The desktop attachment (nose bridge, 0.6 cm down, 0.5 cm forward) was
 * calibrated on a face with the wireframe glasses, whose lens centres sit at
 * the model's origin height. Generated models are centred on their bounding
 * box instead, so their lenses can sit higher or lower than that. Shifting by
 * the lens centres puts every model's lenses where the calibrated ones were.
 * Size needs no fitting: models are generated at a real frame width and the
 * face is the canonical face, so both are in true centimetres.
 */
import { NO_FIT, clampFit, type Fit } from "./transforms.ts";

export type FitHints = {
  /** Lens opening centres in model cm (X right, Y up), left then right; null if not found. */
  lensCentersCm: [number, number][] | null;
};

// Largest shift auto-fit applies, cm; beyond this the hints are probably wrong.
const MAX_AUTO_SHIFT_CM = 1.5;

export function autoFit(hints: FitHints | null): Fit {
  const centers = hints?.lensCentersCm;
  if (!centers || centers.length !== 2) return NO_FIT;
  const midX = (centers[0][0] + centers[1][0]) / 2;
  const midY = (centers[0][1] + centers[1][1]) / 2;
  if (!Number.isFinite(midX) || !Number.isFinite(midY)) return NO_FIT;
  const limit = (value: number) => Math.min(MAX_AUTO_SHIFT_CM, Math.max(-MAX_AUTO_SHIFT_CM, value));
  return clampFit({ ...NO_FIT, x: limit(-midX), y: limit(-midY) });
}

/** A person's manual adjustment applied on top of the automatic fit. */
export function combineFits(base: Fit, manual: Fit): Fit {
  return clampFit({
    scale: base.scale * manual.scale,
    x: base.x + manual.x,
    y: base.y + manual.y,
    z: base.z + manual.z,
    pitch: base.pitch + manual.pitch,
    yaw: base.yaw + manual.yaw,
    roll: base.roll + manual.roll,
  });
}
