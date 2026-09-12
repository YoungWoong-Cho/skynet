import assert from 'node:assert/strict';
import {JSDOM} from 'jsdom';
import * as THREE from 'three';
import {applyUrdfMaterials} from '../frontend/hand-materials.js';
const dom=new JSDOM();globalThis.DOMParser=dom.window.DOMParser;
const source='<robot><material name="opaque"><color rgba="0.3 0.4 0.5 1"/></material><link><visual><material name="opaque"/></visual></link></robot>';
const xml=new DOMParser().parseFromString(source,'application/xml');
const robot=new THREE.Group(),visual=new THREE.Group();visual.isURDFVisual=true;visual.urdfNode=xml.querySelector('visual');
const nested=new THREE.Group(),mesh=new THREE.Mesh(new THREE.BoxGeometry(),new THREE.MeshPhongMaterial({opacity:0,transparent:true}));nested.add(mesh);visual.add(nested);robot.add(visual);
applyUrdfMaterials(robot,source);
assert.equal(mesh.material.opacity,1);assert.equal(mesh.material.transparent,false);
assert.ok(mesh.material.color.equals(new THREE.Color().setRGB(.3,.4,.5,THREE.SRGBColorSpace)));
mesh.geometry.dispose();mesh.material.dispose();
console.log('URDF material regression passed: nested CAD geometry is visible and uses the declared color.');

for (const {declaration,opacity,expected} of [
  {declaration:'',opacity:0,expected:1},
  {declaration:'',opacity:.4,expected:.4},
  {declaration:'<material><color rgba=".1 .2 .3 0"/></material>',opacity:1,expected:0},
]) {
  const source=`<robot><link><visual>${declaration}</visual></link></robot>`;
  const visual=new THREE.Group();visual.isURDFVisual=true;
  visual.urdfNode=new DOMParser().parseFromString(source,'application/xml').querySelector('visual');
  const original=new THREE.MeshPhongMaterial({color:0x123456,opacity,transparent:opacity<1});
  const mesh=new THREE.Mesh(new THREE.BoxGeometry(),original);visual.add(mesh);
  applyUrdfMaterials(visual,source);
  assert.equal(mesh.material.opacity,expected);
  assert.equal(mesh.material.transparent,expected<1);
  if (!declaration) assert.equal(mesh.material.color.getHex(),0x123456,'The CAD surface color is retained');
  mesh.geometry.dispose();mesh.material.dispose();
}
dom.window.close();
console.log('Invisible CAD fallback preserves partial transparency and explicit URDF alpha.');
