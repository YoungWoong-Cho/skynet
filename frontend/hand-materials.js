import * as THREE from "three";

// URDF material declarations also apply to scene-based meshes (DAE/GLB),
// not just the single Mesh returned by the STL loader. Some CAD exports
// contain fully transparent materials despite an opaque URDF declaration.
export function applyUrdfMaterials(robot, source) {
  const xml = new DOMParser().parseFromString(source, "application/xml");
  const named = new Map(
    [...xml.documentElement.children]
      .filter((n) => n.tagName === "material")
      .map((n) => [n.getAttribute("name"), n]),
  );
  const retired = new Set();
  robot.traverse((visual) => {
    if (!visual.isURDFVisual) return;
    const declaration = visual.urdfNode.querySelector("material");
    if (!declaration) return;
    const color =
      declaration.querySelector("color") ||
      named.get(declaration.getAttribute("name"))?.querySelector("color");
    if (!color) return;
    const rgba = color.getAttribute("rgba").trim().split(/\s+/).map(Number);
    if (
      rgba.length !== 4 ||
      rgba.some((v) => !Number.isFinite(v) || v < 0 || v > 1)
    )
      throw new Error("Invalid URDF material color");
    const material = new THREE.MeshPhongMaterial({
      color: new THREE.Color().setRGB(
        ...rgba.slice(0, 3),
        THREE.SRGBColorSpace,
      ),
      opacity: rgba[3],
      transparent: rgba[3] < 1,
    });
    visual.traverse((node) => {
      if (!node.isMesh) return;
      [node.material].flat().forEach((m) => retired.add(m));
      node.material = material;
    });
  });
  retired.forEach((m) => m.dispose());
}
