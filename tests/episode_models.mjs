import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { JSDOM } from "jsdom";
import * as THREE from "three";
import { EpisodeScene } from "../frontend/episode-scene.js";

const dom = new JSDOM("");
Object.assign(globalThis, {
  DOMParser: dom.window.DOMParser,
  Document: dom.window.Document,
  Element: dom.window.Element,
});
const tree =
  '<robot name="hand"><link name="palm"/><link name="finger"/><link name="tip"/><joint name="bend" type="revolute"><parent link="palm"/><child link="finger"/><axis xyz="0 0 1"/><limit lower="-2" upper="2"/></joint><joint name="end" type="fixed"><parent link="finger"/><child link="tip"/><origin xyz="1 0 0"/></joint></robot>';
const visual = tree.replace(
  '<link name="finger"/>',
  '<link name="finger"><visual><geometry><box size="1 .1 .1"/></geometry></visual></link>',
);
globalThis.fetch = async () => new Response(visual);
const scene = Object.assign(Object.create(EpisodeScene.prototype), {
  scene: new THREE.Scene(),
  robots: {},
  objects: {},
  groups: {},
  generation: 0,
  disposed: false,
  fitted: true,
  render() {},
});
const pose = (angle, x) => ({ joints: [angle], root: [x, 0, 0, 1, 0, 0, 0] });
const frame = {
  actual: [[1, 0, 0]],
  prediction: [[2, 1, 0]],
  demonstration: [[3, -1, 0]],
  hand_poses: {
    actual: pose(0, 0),
    prediction: pose(Math.PI / 2, 2),
    demonstration: pose(-Math.PI / 2, 3),
  },
};
await scene.load(
  {
    robot: "test_right",
    joint_names: ["bend"],
    kinematics_urdf: tree,
    frames: [frame],
  },
  { key: "test" },
);
const all = new Set(["actual", "prediction", "demonstration"]);
scene.update(frame, [], all, false, { hand: true, scene: false });
for (const [key, expected] of Object.entries({
  actual: [1, 0, 0],
  prediction: [2, 1, 0],
  demonstration: [3, -1, 0],
})) {
  const robot = scene.robots[key];
  assert.equal(robot.visible, true, key + " hand model is visible");
  robot.updateMatrixWorld(true);
  const position = robot.links.tip.getWorldPosition(new THREE.Vector3());
  assert.ok(
    position.distanceTo(new THREE.Vector3(...expected)) < 1e-8,
    key + " follows its own pose",
  );
  assert.notEqual(
    robot,
    scene.robots[key === "actual" ? "prediction" : "actual"],
  );
}
scene.update(frame, [], all, false, { hand: false, scene: false });
assert.ok(
  Object.values(scene.robots).every((robot) => !robot.visible),
  "Hand model hides all three meshes",
);
assert.ok(
  Object.values(scene.groups).every((group) => group.dots.visible),
  "Keypoints remain independently visible",
);
scene.update(frame, [], new Set(["prediction", "demonstration"]), false, {
  hand: true,
  scene: false,
});
assert.equal(
  scene.robots.actual.visible,
  false,
  "Actual layer also controls the Actual model",
);
assert.equal(scene.robots.prediction.visible, true);
assert.equal(scene.robots.demonstration.visible, true);
scene.update(frame, [], new Set(), true, { hand: true, scene: false });
assert.equal(
  scene.robots.actual.visible,
  true,
  "Collection keypoint toggle does not hide the recorded hand",
);
scene.update(
  { ...frame, hand_poses: { demonstration: frame.hand_poses.demonstration } },
  [],
  all,
  false,
  { hand: true, scene: false },
);
assert.equal(
  scene.robots.actual.visible,
  false,
  "Missing Actual poses are never borrowed from Demonstration",
);
assert.equal(scene.robots.prediction.visible, false);
// Regression: the real WUJI Hands tree uses r_ names while the captured tree
// uses h_r_. Every visual must attach and remain attached during playback.
const wujiTree = await readFile(
  new URL("./fixtures/recorded-wuji/kinematics.urdf", import.meta.url),
  "utf8",
);
const wujiVisual = await readFile(
  new URL("./fixtures/recorded-wuji/visuals.urdf", import.meta.url),
  "utf8",
);
const makeScene = () =>
  Object.assign(Object.create(EpisodeScene.prototype), {
    scene: new THREE.Scene(),
    robots: {},
    objects: {},
    groups: {},
    generation: 0,
    disposed: false,
    fitted: true,
    render() {},
  });
const xml = new DOMParser().parseFromString(wujiVisual, "application/xml");
const names = [...xml.querySelectorAll("robot > link")].map((link) =>
  link.getAttribute("name"),
);
const sourceNames = Object.fromEntries(
  names.map((name) => [name, "h_" + name]),
);
for (const mapping of [undefined, sourceNames]) {
  globalThis.fetch = async () => new Response(wujiVisual);
  const wuji = makeScene();
  await wuji.load(
    {
      robot: "skynet_wuji_2_right",
      source_names: mapping,
      joint_names: ["h_r_index_finger_mcp_flex"],
      kinematics_urdf: wujiTree,
    },
    { key: "wuji-2" },
  );
  let count = 0;
  wuji.robots.actual.traverse((node) => {
    if (node.isMesh) count++;
  });
  assert.equal(count, 26, "all 26 WUJI visuals are attached");
  const finger = wuji.robots.actual.joints.h_r_index_finger_mcp_flex;
  for (const angle of [0, 0.6, 0]) {
    wuji.update(
      { hand_poses: { actual: pose(angle, 0.1) } },
      [],
      new Set(),
      true,
      { hand: true, scene: false },
    );
    assert.equal(wuji.robots.actual.visible, true);
    assert.equal(finger.jointValue[0], angle);
  }
  wuji.update({ hand_poses: { actual: pose(0, 0.1) } }, [], new Set(), true, {
    hand: false,
    scene: false,
  });
  assert.equal(wuji.robots.actual.visible, false);
}
// Explicit captured mappings win even when they differ from today's builder.
const renamed = wujiTree.replaceAll("h_r_", "captured_");
const renamedNames = Object.fromEntries(
  names.map((name) => [name, name.replace(/^r_/, "captured_")]),
);
const renamedScene = makeScene();
await renamedScene.load(
  {
    robot: "skynet_wuji_2_right",
    source_names: renamedNames,
    joint_names: [],
    kinematics_urdf: renamed,
  },
  { key: "wuji-2" },
);
assert.ok(renamedScene.robots.actual);
const bimanual = new DOMParser().parseFromString(
  '<robot name="both"><link name="base"/></robot>',
  "application/xml",
);
for (const side of ["left", "right"]) {
  const prefix = side[0] + "h_";
  const source = new DOMParser().parseFromString(wujiTree, "application/xml");
  for (const element of source.querySelectorAll("[name],[link],[joint]"))
    for (const attribute of ["name", "link", "joint"])
      if (element.hasAttribute(attribute))
        element.setAttribute(
          attribute,
          prefix + element.getAttribute(attribute),
        );
  for (const element of [...source.documentElement.children])
    bimanual.documentElement.appendChild(bimanual.importNode(element, true));
  const mount = new DOMParser().parseFromString(
    `<joint name="${side}_mount" type="fixed"><parent link="base"/><child link="${prefix}skynet_base"/></joint>`,
    "application/xml",
  );
  bimanual.documentElement.appendChild(
    bimanual.importNode(mount.documentElement, true),
  );
}
const both = makeScene();
await both.load(
  {
    robot: "skynet_test_bimanual",
    joint_names: [],
    kinematics_urdf: new dom.window.XMLSerializer().serializeToString(bimanual),
  },
  { key: "test" },
);
let bothCount = 0;
both.robots.actual.traverse((node) => {
  if (node.isMesh) bothCount++;
});
assert.equal(bothCount, 52, "both hands receive their own meshes");
dom.window.close();
console.log(
  "Actual, Prediction and Demonstration meshes retain separate poses and obey shared/layer toggles.",
);
