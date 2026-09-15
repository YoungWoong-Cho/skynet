import { recordedNameCandidates } from "./recorded-hand-names.js";

// Map the recorded joint order to the catalog's palm subtree. Upstream wrist
// and floating-base joints are deliberately outside this mapping.
export function fingerJointBindings(
  model,
  { palm, jointNames, robot, side, sourceNames },
) {
  const root = model.links[palm];
  if (!root) throw new Error("The hand's palm frame is unavailable.");
  const source = new Map(jointNames.map((name, index) => [name, index]));
  const bindings = [];
  root.traverse((joint) => {
    if (!joint.isURDFJoint || joint.jointType === "fixed" || joint.mimicJoint)
      return;
    const candidates = recordedNameCandidates(joint.name, {
      robot,
      side,
      sourceNames,
    });
    const name = candidates.find((candidate) => source.has(candidate));
    const index = source.get(name);
    if (index === undefined)
      throw new Error("Recorded finger joint is missing: " + joint.name);
    // Recorded physics values can lie just outside command limits. Match the
    // left replay exactly instead of clipping the observed finger motion.
    joint.ignoreLimits = true;
    bindings.push({ name: joint.name, index });
  });
  if (!bindings.length)
    throw new Error("No recorded finger joints match this hand.");
  return bindings;
}

export function fingerJointValues(bindings, pose) {
  if (!bindings?.length || !pose?.joints) return null;
  const values = {};
  for (const { name, index } of bindings) {
    const value = pose.joints[index];
    if (!Number.isFinite(value)) return null;
    values[name] = value;
  }
  return values;
}
