/**
 * The three.js scene drawn over the camera video: generated glasses posed on
 * the face, with the head occluder hiding what is behind the head.
 *
 * The camera sits at the origin with projection built from the same pinhole
 * intrinsics as pose estimation, so a point lands on the pixel the desktop
 * app would draw it on. Objects carry full view-space matrices (see
 * transforms.ts); nothing is rebuilt per frame except two matrices.
 */
import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { DEPTH_TOLERANCE_CM } from "./generated/web-ar-data.ts";
import { OCCLUDER_FRAGMENT_SHADER, OCCLUDER_VERTEX_SHADER, buildOccluder } from "./occluder.ts";
import type { HeadPose, Intrinsics } from "./pose.ts";
import { faceViewMatrix, glassesViewMatrix, projectionMatrix, type Fit, type Mat4 } from "./transforms.ts";

const NEAR_CM = 1;
const FAR_CM = 1000;

type Matrix16 = [number, number, number, number, number, number, number, number,
  number, number, number, number, number, number, number, number];
const setRowMajor = (target: THREE.Matrix4, m: Mat4) => target.set(...(m as Matrix16));

export function disposeObject(root: THREE.Object3D) {
  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    object.geometry.dispose();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      for (const value of Object.values(material)) if (value instanceof THREE.Texture) value.dispose();
      material.dispose();
    }
  });
}

export class ArScene {
  readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.Camera();
  private readonly occluder: THREE.Mesh;
  private readonly glasses = new THREE.Group();
  private readonly environment: THREE.Texture;
  private model: THREE.Object3D | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "high-performance" });
    this.renderer.setClearColor(0x000000, 0);

    // Soft studio light from a neutral room, plus a key light above the camera.
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const room = new RoomEnvironment();
    this.environment = pmrem.fromScene(room, 0.04).texture;
    room.traverse((object) => object instanceof THREE.Mesh && object.geometry.dispose());
    pmrem.dispose();
    this.scene.environment = this.environment;
    this.scene.environmentIntensity = 0.8;
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(0, 1, 0.6);
    this.scene.add(key, new THREE.HemisphereLight(0xffffff, 0x8a8a8a, 0.9));

    // Depth-only head: written before the glasses so they're tested against it.
    const { positions, indices } = buildOccluder();
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    const material = new THREE.ShaderMaterial({
      vertexShader: OCCLUDER_VERTEX_SHADER,
      fragmentShader: OCCLUDER_FRAGMENT_SHADER,
      uniforms: { tolerance: { value: DEPTH_TOLERANCE_CM } },
      colorWrite: false,
      side: THREE.FrontSide,
    });
    this.occluder = new THREE.Mesh(geometry, material);
    this.occluder.renderOrder = -1;
    this.occluder.frustumCulled = false;

    for (const object of [this.occluder, this.glasses]) {
      object.matrixAutoUpdate = false;
      object.visible = false;
      this.scene.add(object);
    }
  }

  setViewport(renderWidth: number, renderHeight: number, intrinsics: Intrinsics) {
    // CSS sizes the canvas; this only sets the drawing buffer.
    this.renderer.setSize(renderWidth, renderHeight, false);
    setRowMajor(this.camera.projectionMatrix, projectionMatrix(intrinsics, NEAR_CM, FAR_CM));
    this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
  }

  /** Show a loaded model (or none). The previous model is detached, not disposed. */
  setModel(model: THREE.Object3D | null) {
    if (this.model) this.glasses.remove(this.model);
    this.model = model;
    if (model) {
      model.traverse((object) => {
        object.frustumCulled = false; // posed by our matrices, not three's bounds
      });
      this.glasses.add(model);
    }
  }

  /** Pose everything for this frame; with no pose, nothing is drawn. */
  update(pose: HeadPose | null, fit: Fit) {
    const visible = pose !== null && this.model !== null;
    this.glasses.visible = visible;
    this.occluder.visible = visible;
    if (!pose || !visible) return;
    setRowMajor(this.glasses.matrix, glassesViewMatrix(pose, fit));
    setRowMajor(this.occluder.matrix, faceViewMatrix(pose));
    this.glasses.matrixWorldNeedsUpdate = true;
    this.occluder.matrixWorldNeedsUpdate = true;
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.setModel(null);
    disposeObject(this.occluder);
    this.environment.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
  }
}
