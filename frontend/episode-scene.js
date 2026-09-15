import * as THREE from "three";
import URDFLoader from "urdf-loader";
import { loadUrdfModel, disposeObject } from "./robot-visuals.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { sceneAsset } from "./scene-assets.js";
import { recordedNameCandidates } from "./recorded-hand-names.js";

const colors = {
  actual: 0x16835c,
  prediction: 0xe68b22,
  demonstration: 0x7860cf,
};

export function episodeBounds(frames = []) {
  const bounds = new THREE.Box3();
  const point = new THREE.Vector3();
  for (const frame of frames)
    for (const key of ["actual", "prediction", "demonstration"])
      for (const coordinates of frame[key] || [])
        bounds.expandByPoint(point.fromArray(coordinates));
  return bounds;
}

export class EpisodeScene {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(0xf1f3f4);
    container.append(this.renderer.domElement);
    this.renderer.domElement.tabIndex = 0;
    this.renderer.domElement.setAttribute(
      "aria-label",
      "Recorded hand in 3D. Drag to rotate; scroll to zoom.",
    );
    this.scene = new THREE.Scene();
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x758575, 2.2));
    const light = new THREE.DirectionalLight(0xffffff, 3);
    light.position.set(1, -2, 3);
    this.scene.add(light);
    this.robots = {};
    this.objects = {};
    this.generation = 0;
    this.disposed = false;
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

  async load(data, hand) {
    const generation = ++this.generation;
    this.playbackBounds = episodeBounds(data.frames);
    if (!data.kinematics_urdf || !hand) return;
    // Replay the captured kinematic tree; take visual geometry from the same Hands catalog.
    const parser = new DOMParser();
    const xml = parser.parseFromString(data.kinematics_urdf, "application/xml");
    if (xml.querySelector("parsererror") || xml.querySelector("mesh,texture"))
      throw new Error(
        "Replay geometry must contain a kinematic tree, not external asset URLs",
      );
    const loader = new URDFLoader();
    loader.parseCollision = false;
    const base = loader.parse(xml, "");
    const sides = data.robot.endsWith("_bimanual")
      ? ["left", "right"]
      : [data.robot.endsWith("_left") ? "left" : "right"];
    try {
      for (const side of sides) {
        const visual = await loadUrdfModel(
          `/api/hands/${encodeURIComponent(hand.key)}/${side}/urdf`,
        );
        if (generation !== this.generation || this.disposed) {
          disposeObject(visual);
          disposeObject(base);
          return;
        }
        for (const [name, link] of Object.entries(visual.links)) {
          const targetName = recordedNameCandidates(name, {
            robot: data.robot,
            side,
            sourceNames: data.source_names,
          }).find((candidate) => Object.hasOwn(base.links, candidate));
          const target = base.links[targetName];
          if (target)
            for (const child of [...link.children])
              if (!child.isURDFJoint && !child.isURDFLink) target.add(child);
        }
        disposeObject(visual);
      }
      let meshCount = 0;
      base.traverse((n) => {
        if (n.isMesh) meshCount++;
      });
      if (!meshCount)
        throw new Error(
          "The Hands model does not match the recorded link names",
        );
      Object.values(base.joints).forEach((j) => {
        j.ignoreLimits = true;
      });
      for (const key of ["actual", "prediction", "demonstration"]) {
        const robot = key === "actual" ? base : base.clone(true);
        if (key !== "actual")
          robot.traverse((n) => {
            if (n.material) {
              const materials = [n.material].flat().map((m) => {
                const c = m.clone();
                c.color?.setHex(colors[key]);
                c.transparent = true;
                c.opacity = 0.35;
                c.depthWrite = false;
                return c;
              });
              n.material = Array.isArray(n.material) ? materials : materials[0];
            }
          });
        this.robots[key] = robot;
        this.scene.add(robot);
      }
      this.jointNames = data.joint_names || [];
      this.render();
    } catch (error) {
      disposeObject(base);
      throw error;
    }
  }

  setObjects(definitions = []) {
    const warnings = [];
    Object.values(this.objects).forEach((object) => {
      this.scene.remove(object);
      disposeObject(object);
    });
    this.objects = {};
    for (const item of definitions) {
      try {
        const object = sceneAsset(item);
        this.objects[item.name] = object;
        this.scene.add(object);
      } catch (error) {
        warnings.push(`${item.name}: ${error.message}`);
      }
    }
    return warnings;
  }

  update(
    frame,
    edges,
    enabled,
    keepHand = false,
    geometry = { hand: true, scene: true },
  ) {
    for (const [key, robot] of Object.entries(this.robots)) {
      const pose = frame?.hand_poses?.[key];
      robot.visible = Boolean(
        geometry.hand &&
          pose &&
          ((keepHand && key === "actual") || enabled.has(key)),
      );
      if (pose) {
        robot.position.fromArray(pose.root);
        robot.quaternion.set(
          pose.root[4],
          pose.root[5],
          pose.root[6],
          pose.root[3],
        );
        robot.setJointValues(
          Object.fromEntries(
            this.jointNames.map((name, index) => [name, pose.joints[index]]),
          ),
        );
      }
    }
    for (const [name, object] of Object.entries(this.objects)) {
      const pose = frame?.objects?.[name];
      object.visible = Boolean(
        geometry.scene && (pose || !object.userData.dynamic),
      );
      if (pose) {
        object.position.fromArray(pose);
        object.quaternion.set(pose[4], pose[5], pose[6], pose[3]);
      }
    }
    for (const key of ["actual", "prediction", "demonstration"]) {
      const points = frame?.[key];
      let group = this.groups[key];
      if (!group) {
        const material = new THREE.LineBasicMaterial({
          color: colors[key],
          transparent: true,
          opacity: 0.9,
          depthTest: false,
          depthWrite: false,
        });
        const lines = new THREE.LineSegments(
          new THREE.BufferGeometry(),
          material,
        );
        const dots = new THREE.Points(
          new THREE.BufferGeometry(),
          new THREE.PointsMaterial({
            color: colors[key],
            size: 0.006,
            sizeAttenuation: true,
            depthTest: false,
            depthWrite: false,
          }),
        );
        lines.renderOrder = 2;
        dots.renderOrder = 3;
        group = this.groups[key] = { lines, dots };
        this.scene.add(lines, dots);
      }
      group.lines.visible = Boolean(points && enabled.has(key));
      group.dots.visible = Boolean(points && enabled.has(key));
      group.lines.material.color.setHex(
        key === "actual" && !enabled.has(key) ? 0x77847c : colors[key],
      );
      if (!points) continue;
      this.positions(group.dots.geometry, points.flat());
      this.positions(
        group.lines.geometry,
        edges.flatMap(([a, b]) => [...points[a], ...points[b]]),
      );
      group.dots.geometry.computeBoundingSphere();
      group.lines.geometry.computeBoundingSphere();
      if (!this.fitted && points) {
        // Frame the whole trajectory once; playback must not carry the hand
        // outside a camera fitted only to whichever frame loaded first.
        const box =
          this.playbackBounds && !this.playbackBounds.isEmpty()
            ? this.playbackBounds
            : new THREE.Box3().setFromArray(points.flat());
        const center = box.getCenter(new THREE.Vector3());
        const radius = Math.max(
          box.getSize(new THREE.Vector3()).length(),
          0.15,
        );
        this.controls.target.copy(center);
        this.camera.position
          .copy(center)
          .add(new THREE.Vector3(0.8, -1.4, 0.9).multiplyScalar(radius));
        this.controls.update();
        this.fitted = true;
      }
    }
    this.render();
  }

  positions(geometry, values) {
    const attribute = geometry.getAttribute("position");
    if (attribute && attribute.array.length === values.length) {
      attribute.array.set(values);
      attribute.needsUpdate = true;
    } else {
      geometry.dispose();
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(values, 3),
      );
    }
  }

  render() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h || this.container.hidden) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.disposed = true;
    ++this.generation;
    Object.values(this.robots).forEach(disposeObject);
    Object.values(this.objects).forEach(disposeObject);
    this.observer.disconnect();
    this.controls.dispose();
    for (const group of Object.values(this.groups))
      for (const object of Object.values(group)) {
        object.geometry.dispose();
        object.material.dispose();
      }
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
