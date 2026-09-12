import * as THREE from "three";
import URDFLoader from "urdf-loader";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { applyUrdfMaterials } from "./hand-materials.js";

export function disposeObject(root) {
  const geometries = new Set(),
    materials = new Set(),
    textures = new Set();
  root?.traverse((object) => {
    if (object.geometry) geometries.add(object.geometry);
    [object.material]
      .flat()
      .filter(Boolean)
      .forEach((material) => materials.add(material));
  });
  materials.forEach((material) => {
    Object.values(material).forEach((v) => {
      if (v?.isTexture) textures.add(v);
    });
    material.dispose();
  });
  geometries.forEach((g) => g.dispose());
  textures.forEach((t) => t.dispose());
}

export async function loadUrdfModel(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(30000) });
  if (!response.ok) throw new Error("Could not load the hand description");
  const source = await response.text();
  const manager = new THREE.LoadingManager();
  let robot, failure, timer;
  const ready = new Promise((resolve) => {
    manager.onLoad = resolve;
    manager.onError = (path) => {
      failure = new Error(
        "Could not load model asset: " + path.split("/").pop(),
      );
    };
  });
  const loader = new URDFLoader(manager);
  loader.parseCollision = false;
  const defaultLoad = loader.loadMeshCb;
  loader.loadMeshCb = (path, mgr, done) => {
    if (/\.glb$/i.test(path))
      new GLTFLoader(mgr).load(
        path,
        (g) => done(g.scene),
        undefined,
        (e) => {
          failure = e;
          done(null, e);
        },
      );
    else if (/\.(stl|dae)$/i.test(path))
      defaultLoad(path, mgr, (object, e) => {
        if (e) failure = e;
        // URDF owns the mesh coordinate frame. ColladaLoader adds a Z-up to
        // Y-up rotation to its wrapper scene; applying it here rotates only
        // some links (e.g. Shadow PST fingertips) away from their joint axes.
        // Authored node transforms and the asset's unit scale stay intact.
        if (object && /\.dae$/i.test(path)) object.quaternion.identity();
        done(object, e);
      });
    else {
      failure = new Error("Unsupported visual mesh format");
      done(null, failure);
    }
  };
  manager.itemStart("description");
  try {
    try {
      robot = loader.parse(source, "");
    } finally {
      manager.itemEnd("description");
    }
    await Promise.race([
      ready,
      new Promise((_, reject) => {
        timer = setTimeout(
          () => reject(new Error("Hand model loading timed out")),
          30000,
        );
      }),
    ]);
    if (failure) throw failure;
    applyUrdfMaterials(robot, source);
    return robot;
  } catch (error) {
    disposeObject(robot);
    ready.then(() => disposeObject(robot));
    throw error;
  } finally {
    clearTimeout(timer);
  }
}
