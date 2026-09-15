// Use captured names when available; older traces follow the published hand
// bundle naming rule. Both scene meshes and finger playback use this mapping.
export function recordedNameCandidates(
  name,
  { robot, side, sourceNames } = {},
) {
  const bimanual = robot?.endsWith("_bimanual");
  const sourceKey = bimanual ? side + ":" + name : name;
  if (sourceNames && Object.hasOwn(sourceNames, sourceKey))
    return [sourceNames[sourceKey]];
  if (robot?.startsWith("floating_shadow_"))
    return [(bimanual ? side + "_" : "") + name.replace(/^[rl]h_/, "")];
  const safe = name.replace(/[^A-Za-z0-9_]/g, "_");
  const prefix = bimanual ? side[0] + "h_" : "";
  return [prefix + "h_" + safe, prefix + safe, prefix + name];
}
