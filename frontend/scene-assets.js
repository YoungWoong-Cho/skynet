import * as THREE from "three";

const finite = (values, length) =>
  Array.isArray(values) &&
  values.length === length &&
  values.every(Number.isFinite);
const positive = (values) =>
  values.every((value) => Number.isFinite(value) && value > 0 && value < 1e6);

export function scenePart(part) {
  let geometry;
  if (part.type === "mesh") {
    const vertices = part.vertices,
      indices = part.indices;
    if (
      !Array.isArray(vertices) ||
      vertices.length < 9 ||
      vertices.length % 3 ||
      vertices.length > 360000 ||
      !vertices.every(Number.isFinite) ||
      !Array.isArray(indices) ||
      indices.length % 3 ||
      indices.length > 2000000 ||
      !indices.every(
        (i) => Number.isInteger(i) && i >= 0 && i < vertices.length / 3,
      )
    )
      throw new Error("Invalid saved mesh");
    geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(vertices, 3),
    );
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
  } else if (part.type === "box" && finite(part.size, 3) && positive(part.size))
    geometry = new THREE.BoxGeometry(...part.size);
  else if (part.type === "sphere" && positive([part.radius]))
    geometry = new THREE.SphereGeometry(part.radius, 24, 16);
  else if (
    ["cylinder", "cone", "capsule"].includes(part.type) &&
    positive([part.radius, part.height])
  ) {
    geometry =
      part.type === "capsule"
        ? new THREE.CapsuleGeometry(part.radius, part.height, 8, 24)
        : new THREE.CylinderGeometry(
            part.type === "cone" ? 0 : part.radius,
            part.radius,
            part.height,
            24,
          );
    if (part.axis === "X") geometry.rotateZ(-Math.PI / 2);
    else if (!part.axis || part.axis === "Z") geometry.rotateX(Math.PI / 2);
  } else if (
    part.type === "plane" &&
    finite(part.size, 2) &&
    positive(part.size)
  ) {
    geometry = new THREE.PlaneGeometry(...part.size);
    if (part.axis === "X") geometry.rotateY(Math.PI / 2);
    if (part.axis === "Y") geometry.rotateX(-Math.PI / 2);
  } else throw new Error("Unsupported saved scene geometry");
  const color = finite(part.color, 3) ? part.color : [0.6, 0.6, 0.6];
  const mesh = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({
      color: new THREE.Color(...color),
      roughness: 0.75,
      side: THREE.DoubleSide,
      flatShading: part.type === "mesh",
    }),
  );
  if (part.matrix) {
    if (!finite(part.matrix, 16)) {
      geometry.dispose();
      mesh.material.dispose();
      throw new Error("Invalid saved mesh transform");
    }
    mesh.applyMatrix4(new THREE.Matrix4().fromArray(part.matrix));
  }
  return mesh;
}

export function sceneAsset(definition) {
  const group = new THREE.Group();
  try {
    const parts = definition.type === "asset" ? definition.parts : [definition];
    if (!Array.isArray(parts) || parts.length > 400)
      throw new Error("Invalid scene asset");
    for (const part of parts) group.add(scenePart(part));
    group.userData.dynamic = definition.dynamic !== false;
    return group;
  } catch (error) {
    group.traverse((object) => {
      object.geometry?.dispose();
      object.material?.dispose();
    });
    throw error;
  }
}
