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
    const color =
      declaration?.querySelector("color") ||
      named.get(declaration?.getAttribute("name"))?.querySelector("color");
    if (!color) {
      // Without a URDF alpha override, keep imported shading but make fully
      // invisible CAD materials visible. Shadow PST fingertip DAE files have
      // zero opacity despite representing the solid exterior of the hand.
      visual.traverse((node) => {
        if (!node.isMesh) return;
        const materials = [node.material].flat().map((material) => {
          if (material.opacity !== 0) return material;
          const visible = material.clone();
          visible.opacity = 1;
          visible.transparent = false;
          retired.add(material);
          return visible;
        });
        node.material = Array.isArray(node.material) ? materials : materials[0];
      });
      return;
    }
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
