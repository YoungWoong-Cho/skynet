import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {JSDOM} from 'jsdom';
import * as THREE from 'three';
import URDFLoader from 'urdf-loader';
import {HandsViewer} from '../frontend/hands-viewer.js';
import {fingerJointBindings, fingerJointValues} from '../frontend/hand-playback.js';
const dom=new JSDOM('');
Object.assign(globalThis,{DOMParser:dom.window.DOMParser,Document:dom.window.Document,Element:dom.window.Element});
const xml=new DOMParser().parseFromString(await readFile(new URL('../config/hand_models/shadow/right.urdf',import.meta.url),'utf8'),'application/xml');
xml.querySelectorAll('visual,collision').forEach(n=>n.remove());
const robot=new URDFLoader().parse(xml,'');
const fingers=Object.keys(robot.joints).filter(n=>robot.joints[n].jointType==='revolute'&&!n.includes('WRJ'));
assert.equal(fingers.length,22);
for(const [name,sourceNames] of [
 ['floating_shadow_right',['x_translation_joint','z_rotation_joint',...fingers.map(n=>n.replace('rh_',''))]],
 ['skynet_shadow_right',['skynet_x','skynet_roll',...fingers.map(n=>'h_'+n)]],
 ['skynet_shadow_bimanual',[...fingers.map(n=>'lh_h_'+n), 'rh_skynet_x', 'rh_skynet_roll',...fingers.map(n=>'rh_h_'+n)]],
]){
 const viewer=Object.assign(Object.create(HandsViewer.prototype),{robot,render(){}});
 viewer.configurePlayback({palm:'rh_palm',jointNames:sourceNames,robot:name,side:'right'});
 assert.equal(viewer.playbackBindings.length,22);
 const rest=structuredClone(viewer.playbackRestPose);
 robot.position.set(2,3,4);robot.rotation.set(.1,.2,.3);
 robot.setJointValues({rh_WRJ1:.15,rh_WRJ2:.1});robot.updateMatrixWorld(true);
 const palm=robot.links.rh_palm.matrixWorld.clone();const poseRoot=robot.position.clone();const orientation=robot.quaternion.clone();
 for(const value of [.2,.6,0]){
  const joints=sourceNames.map(n=>n.startsWith('lh_')?-.8:n.includes('FFJ')||n.includes('MFJ')||n.includes('RFJ')||n.includes('LFJ')||n.includes('THJ')?value:9);
  assert.equal(viewer.setFingerPose({root:[99,88,77,0,1,0,0],joints}),true);
  robot.updateMatrixWorld(true);
  for(const finger of fingers)assert.equal(robot.joints[finger].jointValue[0],value,name+' finger matches sample');
  assert.ok(robot.links.rh_palm.matrixWorld.equals(palm),'palm and wrist never move');
  assert.ok(robot.position.equals(poseRoot));assert.ok(robot.quaternion.equals(orientation));
 }
 assert.equal(viewer.setFingerPose(null),false);assert.equal(robot.visible,true,'missing pose keeps the catalog hand visible');
 for(const finger of fingers)assert.deepEqual(robot.joints[finger].jointValue,rest[finger],'missing pose restores default fingers instead of stale playback');
 robot.updateMatrixWorld(true);assert.ok(robot.links.rh_palm.matrixWorld.equals(palm),'default pose keeps wrist and palm fixed');
 assert.ok(robot.position.equals(poseRoot));assert.ok(robot.quaternion.equals(orientation));
 assert.equal(viewer.setFingerPose({joints:sourceNames.map(()=>.3)}),true);assert.equal(robot.visible,true);
 assert.equal(fingerJointValues(viewer.playbackBindings,{joints:[NaN]}),null);
 assert.throws(()=>fingerJointBindings(robot,{palm:'rh_palm',jointNames:[],robot:name,side:'right'}),/missing/);
 assert.throws(()=>viewer.configurePlayback({palm:'rh_palm',jointNames:[],robot:name,side:'right'}),/missing/);
 assert.equal(viewer.setFingerPose(null),false);assert.equal(robot.visible,true,'unavailable joint layout still displays the default hand');
 for(const finger of fingers)assert.deepEqual(robot.joints[finger].jointValue,rest[finger]);
}
// Mimic joints are driven by their parent; they need no independent recording.
const mimic=new URDFLoader().parse('<robot name="m"><link name="palm"/><link name="a"/><link name="b"/><joint name="bend" type="revolute"><parent link="palm"/><child link="a"/><limit lower="-1" upper="1"/></joint><joint name="tip" type="revolute"><parent link="a"/><child link="b"/><limit lower="-1" upper="1"/><mimic joint="bend" multiplier="0.5"/></joint></robot>','');
const bindings=fingerJointBindings(mimic,{palm:'palm',jointNames:['h_bend'],robot:'skynet_test_right',side:'right'});
assert.equal(bindings.length,1);mimic.setJointValues(fingerJointValues(bindings,{joints:[.6]}));assert.equal(mimic.joints.tip.jointValue[0],.3);
dom.window.close();console.log('Finger playback: recorded name mapping, 22 Shadow joints, fixed palm/wrist/root, missing poses and mimic joints passed.');
