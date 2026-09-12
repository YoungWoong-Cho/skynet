import assert from 'node:assert/strict';
import {JSDOM} from 'jsdom';
import * as THREE from 'three';
import {EpisodeScene} from '../frontend/episode-scene.js';

const dom=new JSDOM('');
Object.assign(globalThis,{DOMParser:dom.window.DOMParser,Document:dom.window.Document,Element:dom.window.Element});
const tree='<robot name="hand"><link name="palm"/><link name="finger"/><link name="tip"/><joint name="bend" type="revolute"><parent link="palm"/><child link="finger"/><axis xyz="0 0 1"/><limit lower="-2" upper="2"/></joint><joint name="end" type="fixed"><parent link="finger"/><child link="tip"/><origin xyz="1 0 0"/></joint></robot>';
const visual=tree.replace('<link name="finger"/>','<link name="finger"><visual><geometry><box size="1 .1 .1"/></geometry></visual></link>');
globalThis.fetch=async()=>new Response(visual);
const scene=Object.assign(Object.create(EpisodeScene.prototype),{
  scene:new THREE.Scene(),robots:{},objects:{},groups:{},generation:0,disposed:false,fitted:true,render(){},
});
const pose=(angle,x)=>({joints:[angle],root:[x,0,0,1,0,0,0]});
const frame={actual:[[1,0,0]],prediction:[[2,1,0]],demonstration:[[3,-1,0]],hand_poses:{actual:pose(0,0),prediction:pose(Math.PI/2,2),demonstration:pose(-Math.PI/2,3)}};
await scene.load({robot:'test_right',joint_names:['bend'],kinematics_urdf:tree,frames:[frame]}, {key:'test'});
const all=new Set(['actual','prediction','demonstration']);
scene.update(frame,[],all,false,{hand:true,scene:false});
for(const [key,expected] of Object.entries({actual:[1,0,0],prediction:[2,1,0],demonstration:[3,-1,0]})){
  const robot=scene.robots[key];assert.equal(robot.visible,true,key+' hand model is visible');
  robot.updateMatrixWorld(true);
  const position=robot.links.tip.getWorldPosition(new THREE.Vector3());
  assert.ok(position.distanceTo(new THREE.Vector3(...expected))<1e-8,key+' follows its own pose');
  assert.notEqual(robot,scene.robots[key==='actual'?'prediction':'actual']);
}
scene.update(frame,[],all,false,{hand:false,scene:false});
assert.ok(Object.values(scene.robots).every(robot=>!robot.visible),'Hand model hides all three meshes');
assert.ok(Object.values(scene.groups).every(group=>group.dots.visible),'Keypoints remain independently visible');
scene.update(frame,[],new Set(['prediction','demonstration']),false,{hand:true,scene:false});
assert.equal(scene.robots.actual.visible,false,'Actual layer also controls the Actual model');
assert.equal(scene.robots.prediction.visible,true);assert.equal(scene.robots.demonstration.visible,true);
scene.update(frame,[],new Set(),true,{hand:true,scene:false});
assert.equal(scene.robots.actual.visible,true,'Collection keypoint toggle does not hide the recorded hand');
scene.update({...frame,hand_poses:{demonstration:frame.hand_poses.demonstration}},[],all,false,{hand:true,scene:false});
assert.equal(scene.robots.actual.visible,false,'Missing Actual poses are never borrowed from Demonstration');
assert.equal(scene.robots.prediction.visible,false);
dom.window.close();
console.log('Actual, Prediction and Demonstration meshes retain separate poses and obey shared/layer toggles.');
