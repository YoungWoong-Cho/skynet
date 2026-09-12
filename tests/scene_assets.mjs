import assert from 'node:assert/strict';
import * as THREE from 'three';
import {sceneAsset} from '../frontend/scene-assets.js';

const object=sceneAsset({type:'asset',dynamic:false,parts:[{type:'mesh',vertices:[0,0,0,1,0,0,0,1,0],indices:[0,1,2],color:[1,0,0],matrix:[2,0,0,0,0,3,0,0,0,0,4,0,10,20,30,1]}]});
object.updateMatrixWorld(true);
const world=new THREE.Vector3(1,0,0).applyMatrix4(object.children[0].matrixWorld);
assert.deepEqual(world.toArray(),[12,20,30],'USD transforms map into the same Three.js coordinates');
assert.equal(object.userData.dynamic,false,'Static scene assets do not require per-frame poses');
assert.throws(()=>sceneAsset({type:'asset',parts:[{type:'mesh',vertices:[0,0,0,1,0,0,0,1,0],indices:[0,1,99]}]}),/Invalid saved mesh/);
assert.throws(()=>sceneAsset({type:'asset',parts:[{type:'box',size:[1,-1,1]}]}),/Unsupported/);
for(const kind of ['cylinder','cone','capsule']) {
 const asset=sceneAsset({type:kind,radius:.1,height:.2,axis:'Z'});
 assert.ok(asset.children[0].isMesh);
 asset.children[0].geometry.dispose();asset.children[0].material.dispose();
}
object.children[0].geometry.dispose();object.children[0].material.dispose();
console.log('Scene assets: mesh transforms, static poses, primitives and invalid geometry passed.');
