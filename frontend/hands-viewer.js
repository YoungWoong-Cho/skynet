import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { loadUrdfModel, disposeObject as dispose } from "./robot-visuals.js";

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
    const robot = await loadUrdfModel(url);
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
  dispose() {
    this.clear();
    this.resize.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
