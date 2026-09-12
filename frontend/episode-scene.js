import * as THREE from "three";
import {OrbitControls} from "three/examples/jsm/controls/OrbitControls.js";

const colors = {actual: 0x16835c, prediction: 0xe68b22, demonstration: 0x7860cf};

export class EpisodeScene {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({antialias:true});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0xf1f3f4);
    container.append(this.renderer.domElement);
    this.renderer.domElement.tabIndex = 0;
    this.renderer.domElement.setAttribute("aria-label", "Recorded hand in 3D. Drag to rotate; scroll to zoom.");
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.001, 100);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.listenToKeyEvents(this.renderer.domElement);
    this.controls.addEventListener("change", () => this.render());
    this.groups = {};
    this.fitted = false;
    this.observer = new ResizeObserver(() => this.render());
    this.observer.observe(container);
  }

  update(frame, edges, enabled, keepHand=false) {
    for (const key of ["actual", "prediction", "demonstration"]) {
      const points = frame?.[key];
      let group = this.groups[key];
      if (!group) {
        const material = new THREE.LineBasicMaterial({color:colors[key], transparent:true, opacity:0.9});
        const lines = new THREE.LineSegments(new THREE.BufferGeometry(), material);
        const dots = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({color:colors[key], size:0.006, sizeAttenuation:true}));
        group = this.groups[key] = {lines, dots};
        this.scene.add(lines, dots);
      }
      // A neutral actual skeleton remains as the interactive hand when markers are off.
      group.lines.visible = Boolean(points && ((keepHand && key === "actual") || enabled.has(key)));
      group.dots.visible = Boolean(points && enabled.has(key));
      group.lines.material.color.setHex(key === "actual" && !enabled.has(key) ? 0x77847c : colors[key]);
      if (!points) continue;
      this.positions(group.dots.geometry, points.flat());
      this.positions(group.lines.geometry, edges.flatMap(([a,b]) => [...points[a], ...points[b]]));
      group.dots.geometry.computeBoundingSphere();
      group.lines.geometry.computeBoundingSphere();
      if (!this.fitted && key === "actual") {
        const box = new THREE.Box3().setFromArray(points.flat());
        const center = box.getCenter(new THREE.Vector3());
        const radius = Math.max(box.getSize(new THREE.Vector3()).length(), 0.15);
        this.controls.target.copy(center);
        this.camera.position.copy(center).add(new THREE.Vector3(0.8, -1.4, 0.9).multiplyScalar(radius));
        this.controls.update();
        this.fitted = true;
      }
    }
    this.render();
  }

  positions(geometry, values) {
    const attribute=geometry.getAttribute("position");
    if(attribute && attribute.array.length===values.length) {attribute.array.set(values);attribute.needsUpdate=true;}
    else {geometry.dispose();geometry.setAttribute("position",new THREE.Float32BufferAttribute(values,3));}
  }

  render() {
    const {clientWidth:w, clientHeight:h} = this.container;
    if (!w || !h || this.container.hidden) return;
    this.renderer.setSize(w,h,false);
    this.camera.aspect=w/h;
    this.camera.updateProjectionMatrix();
    this.renderer.render(this.scene,this.camera);
  }

  dispose() {
    this.observer.disconnect();
    this.controls.dispose();
    for (const group of Object.values(this.groups)) for (const object of Object.values(group)) {
      object.geometry.dispose(); object.material.dispose();
    }
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
