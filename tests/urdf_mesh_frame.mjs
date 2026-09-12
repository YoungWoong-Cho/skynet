import assert from 'node:assert/strict';
import {JSDOM} from 'jsdom';
import * as THREE from 'three';
import {episodeBounds} from '../frontend/episode-scene.js';
import {loadUrdfModel, disposeObject} from '../frontend/robot-visuals.js';
const w = new JSDOM('').window;
globalThis.DOMParser=w.DOMParser;globalThis.Document=w.Document;globalThis.Element=w.Element;globalThis.ProgressEvent=w.ProgressEvent;
const dae = axis => `<COLLADA version="1.4.1" xmlns="http://www.collada.org/2005/11/COLLADASchema"><asset><unit meter="1"/><up_axis>${axis}</up_axis></asset><library_geometries><geometry id="mesh"><mesh><source id="positions"><float_array id="array" count="9">0 0 0 0 0 26 0 7 0</float_array><technique_common><accessor source="#array" count="3" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source><vertices id="verts"><input semantic="POSITION" source="#positions"/></vertices><triangles count="1"><input semantic="VERTEX" source="#verts" offset="0"/><p>0 1 2</p></triangles></mesh></geometry></library_geometries><library_visual_scenes><visual_scene id="scene"><node id="finger"><translate>1 2 3</translate><instance_geometry url="#mesh"/></node></visual_scene></library_visual_scenes><scene><instance_visual_scene url="#scene"/></scene></COLLADA>`;
for(const axis of ['Y_UP','Z_UP']) {
 globalThis.fetch=async request=>new Response(String(request.url || request).endsWith('.dae')?dae(axis):'<robot name="test"><link name="finger"><visual><origin xyz="0 0 .5"/><geometry><mesh filename="http://fixture/finger.dae" scale=".001 .001 .001"/></geometry></visual></link></robot>',{status:200,headers:{'Content-Type':'text/xml'}});
 const robot=await loadUrdfModel('http://fixture/model.urdf');
 const box=new THREE.Box3().setFromObject(robot);
 for(const [got,want] of box.min.toArray().map((v,i)=>[v,[.001,.002,.503][i]])) assert.ok(Math.abs(got-want)<1e-8,axis);
 for(const [got,want] of box.max.toArray().map((v,i)=>[v,[.001,.009,.529][i]])) assert.ok(Math.abs(got-want)<1e-8,axis);
 disposeObject(robot);
}
const bounds=episodeBounds([{actual:[[0,0,0],[.1,.1,.1]]},{actual:[[1,2,3]],prediction:[[4,5,6]]}]);
assert.deepEqual(bounds.min.toArray(),[0,0,0]);assert.deepEqual(bounds.max.toArray(),[4,5,6]);
w.close();
console.log('URDF meshes: Y-up and Z-up DAE assets preserve the same joint frame, authored offsets and millimeter scale.');
