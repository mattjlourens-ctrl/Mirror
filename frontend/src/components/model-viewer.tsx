import { useEffect, useRef, useState } from "react";

type ViewerState = "loading" | "ready" | "error";

/**
 * Draws a generated glasses model (.glb) with three.js, loaded on demand.
 * glTF and Mirror share axes (X right, Y up, Z towards the viewer), so the
 * model faces the camera as generated. This is a model preview, not try-on.
 * Give it a `key` of the model URL so a new model starts from a fresh viewer.
 */
export function ModelViewer({ modelUrl, label }: { modelUrl: string; label: string }) {
  const container = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<ViewerState>("loading");

  useEffect(() => {
    let disposed = false;
    let frame = 0;
    let cleanup = () => {};

    (async () => {
      const THREE = await import("three");
      const { GLTFLoader } = await import("three/examples/jsm/loaders/GLTFLoader.js");
      const element = container.current;
      if (disposed || !element) return;

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      const scene = new THREE.Scene();
      scene.add(new THREE.HemisphereLight(0xffffff, 0x888888, 2.2));
      const light = new THREE.DirectionalLight(0xffffff, 1.6);
      light.position.set(0.3, 0.6, 1);
      scene.add(light);
      const camera = new THREE.PerspectiveCamera(28, 1, 0.001, 10);

      const resize = () => {
        const { width, height } = element.getBoundingClientRect();
        renderer.setSize(width, height);
        camera.aspect = width / Math.max(height, 1);
        camera.updateProjectionMatrix();
      };

      let gltf;
      try {
        gltf = await new GLTFLoader().loadAsync(modelUrl);
      } catch {
        renderer.dispose();
        if (!disposed) setState("error");
        return;
      }
      if (disposed) {
        renderer.dispose();
        return;
      }

      const model = gltf.scene;
      const box = new THREE.Box3().setFromObject(model);
      const size = box.getSize(new THREE.Vector3());
      model.position.sub(box.getCenter(new THREE.Vector3()));
      const pivot = new THREE.Group();
      pivot.add(model);
      scene.add(pivot);
      const radius = Math.max(size.x, size.y, size.z) / 2;
      camera.position.set(0, radius * 0.25, (radius / Math.tan((camera.fov * Math.PI) / 360)) * 1.15);
      camera.lookAt(0, 0, 0);

      element.appendChild(renderer.domElement);
      renderer.domElement.setAttribute("aria-hidden", "true");
      resize();
      const observer = new ResizeObserver(resize);
      observer.observe(element);

      // A gentle turn shows depth; reduced-motion users get a still three-quarter view.
      const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const started = performance.now();
      const render = () => {
        pivot.rotation.y = still ? -0.5 : Math.sin((performance.now() - started) / 1800) * 0.6;
        renderer.render(scene, camera);
        if (!still) frame = requestAnimationFrame(render);
      };
      render();
      setState("ready");

      cleanup = () => {
        cancelAnimationFrame(frame);
        observer.disconnect();
        model.traverse((object) => {
          if (object instanceof THREE.Mesh) {
            object.geometry.dispose();
            const materials = Array.isArray(object.material) ? object.material : [object.material];
            materials.forEach((material) => {
              Object.values(material).forEach((value) => value instanceof THREE.Texture && value.dispose());
              material.dispose();
            });
          }
        });
        renderer.dispose();
        renderer.domElement.remove();
      };
    })();

    return () => {
      disposed = true;
      cleanup();
    };
  }, [modelUrl]);

  return (
    <div
      ref={container}
      role="img"
      aria-label={label}
      data-state={state}
      className="relative h-full w-full overflow-hidden"
    >
      {state !== "ready" && (
        <p className="absolute inset-0 flex items-center justify-center text-xs text-neutral-500">
          {state === "loading" ? "Loading 3D model…" : "The 3D model couldn't be displayed."}
        </p>
      )}
    </div>
  );
}
