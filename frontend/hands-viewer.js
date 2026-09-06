import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { applyUrdfMaterials } from "./hand-materials.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import URDFLoader from "urdf-loader";

function dispose(root) {
  const geometries = new Set(),
    materials = new Set(),
    textures = new Set();
  root?.traverse((node) => {
    if (node.geometry) geometries.add(node.geometry);
    for (const material of [node.material].flat().filter(Boolean))
      materials.add(material);
  });
  for (const material of materials) {
    for (const v of Object.values(material)) if (v?.isTexture) textures.add(v);
    material.dispose();
  }
  geometries.forEach((g) => g.dispose());
  textures.forEach((t) => t.dispose());
}

export class HandsViewer {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0xf1f3f4);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.tabIndex = 0;
    this.renderer.domElement.setAttribute(
      "aria-label",
      "Interactive hand model. Drag to rotate, scroll to zoom, or use arrow keys to pan.",
    );
    this.renderer.domElement.setAttribute("role", "img");
    container.append(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.001, 100);
    this.camera.up.set(0, 0, 1);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x7e8899, 2.2));
    const key = new THREE.DirectionalLight(0xffffff, 3);
    key.position.set(1, -2, 3);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 1.3);
    fill.position.set(-2, 1, 1);
    this.scene.add(fill);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.listenToKeyEvents(this.renderer.domElement);
    this.controls.addEventListener("change", () => this.render());
    this.resize = new ResizeObserver(() => this.render());
    this.resize.observe(container);
    this.generation = 0;
  }
  render() {
    if (
      !this.container.offsetWidth ||
      !this.container.offsetHeight ||
      document.hidden ||
      this.container.closest("[hidden]")
    )
      return;
    const w = this.container.clientWidth,
      h = this.container.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.render(this.scene, this.camera);
  }
  async load(url, metadata) {
    const generation = ++this.generation;
    if (this.robot) {
      this.scene.remove(this.robot);
      dispose(this.robot);
      this.robot = null;
    }
    this.render();
    const manager = new THREE.LoadingManager();
    let robot,
      failure,
      timedOut = false,
      timer;
    const ready = new Promise((resolve) => {
      manager.onLoad = () => resolve();
      manager.onError = (url) => {
        failure = new Error(
          `Could not load model asset: ${url.split("/").pop()}`,
        );
      };
    });
    const loader = new URDFLoader(manager);
    loader.parseCollision = false;
    const defaultMeshLoader = loader.loadMeshCb;
    loader.loadMeshCb = (path, mgr, done) => {
      if (/\.glb$/i.test(path))
        new GLTFLoader(mgr).load(
          path,
          (g) => done(g.scene),
          undefined,
          (e) => {
            failure = new Error(`Could not load ${path.split("/").pop()}`);
            done(null, e);
          },
        );
      else if (!/\.(stl|dae)$/i.test(path)) {
        failure = new Error(
          `Unsupported visual mesh format: ${path.split("/").pop()}`,
        );
        done(null, failure);
      } else
        defaultMeshLoader(path, mgr, (object, error) => {
          if (error)
            failure = new Error(`Could not load ${path.split("/").pop()}`);
          done(object, error);
        });
    };
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(30000) });
      if (!response.ok) throw new Error("Could not load the hand description");
      const source = await response.text();
      manager.itemStart("description");
      try {
        robot = loader.parse(source, "");
      } finally {
        manager.itemEnd("description");
      }
      await Promise.race([
        ready,
        new Promise((_, reject) => {
          timer = setTimeout(() => {
            timedOut = true;
            reject(new Error("Model loading timed out. Use Refresh to retry."));
          }, 30000);
        }),
      ]);
      if (failure) throw failure;
      applyUrdfMaterials(robot, source);
      if (generation !== this.generation) {
        dispose(robot);
        return false;
      }
      robot.rotation.set(...(metadata.preview_rotation || [0, 0, 0]));
      this.robot = robot;
      this.scene.add(robot);
      for (const joint of metadata.joints)
        if (!joint.mimic)
          robot.setJointValue(
            joint.name,
            Math.max(joint.lower, Math.min(joint.upper, 0)),
          );
      this.fit();
      return true;
    } catch (error) {
      dispose(robot);
      if (timedOut) ready.then(() => dispose(robot));
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }
  fit() {
    if (!this.robot) return;
    this.robot.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(this.robot);
    const center = bounds.getCenter(new THREE.Vector3());
    const radius = bounds.getSize(new THREE.Vector3()).length() / 2;
    if (!Number.isFinite(radius) || radius <= 0)
      throw new Error("The model contains no visible geometry");
    const distance =
      (radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2))) * 1.18;
    this.controls.target.copy(center);
    this.camera.position
      .copy(center)
      .add(
        new THREE.Vector3(0.65, -1, 0.5).normalize().multiplyScalar(distance),
      );
    this.camera.near = Math.max(0.0001, radius / 100);
    this.camera.far = distance * 30;
    this.controls.minDistance = radius * 0.25;
    this.controls.maxDistance = distance * 5;
    this.controls.update();
    this.render();
  }
  setJoints(values) {
    if (!this.robot) return;
    this.robot.setJointValues(values);
    this.render();
  }
  clear() {
    ++this.generation;
    if (this.robot) {
      this.scene.remove(this.robot);
      dispose(this.robot);
      this.robot = null;
    }
    this.render();
  }
}
