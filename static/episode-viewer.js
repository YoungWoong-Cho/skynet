(() => {
  const version = new URL(document.currentScript.src).search;
  const instances = new Map();
  const colors = {actual:"#16835c", prediction:"#e68b22", demonstration:"#7860cf"};
  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text) element.textContent = text;
    if (className) element.className = className;
    return element;
  };

  // Camera metadata uses ROS optical coordinates (+x right, +y down, +z forward).
  function project(point, camera) {
    const q = camera.quaternion_world_ros, p = camera.position_world, k = camera.intrinsic_matrix;
    if (!q || !p || !k) return null;
    const norm = Math.hypot(...q);
    const [w,x,y,z] = q.map(value => value / norm);
    const r = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];
    const delta = point.map((value,i)=>value-p[i]);
    const local = [0,1,2].map(i=>delta.reduce((sum,value,j)=>sum+r[j][i]*value,0));
    if (local[2] <= 0) return null;
    return [k[0][0]*local[0]/local[2]+k[0][2], k[1][1]*local[1]/local[2]+k[1][2]];
  }

  class EpisodeViewer {
    constructor(host, video) {
      this.host=host; this.video=video; this.generation=0; this.enabled=new Set(["actual","prediction","demonstration"]);
      this.view="all";
      host.classList.add("episode-viewer");
      this.main=node("div",null,"episode-viewer-main");
      this.tabs=node("div",null,"episode-viewer-tabs"); this.tabs.setAttribute("aria-label","Episode views");
      this.layers=node("div",null,"episode-viewer-layers");
      this.canvas=node("canvas"); this.canvas.setAttribute("aria-label","Episode camera view with optional keypoints");
      this.sceneHost=node("div",null,"episode-viewer-scene"); this.sceneHost.hidden=true;
      this.sceneHelp=node("p","3D keypoints · Drag to rotate · Scroll to zoom","secondary");this.sceneHelp.hidden=true;
      this.stage=node("div",null,"episode-viewer-stage"); this.stage.append(video,this.canvas,this.sceneHost);
      video.controls=false;
      this.legend=node("p",null,"secondary");
      this.retry=node("button","Retry preview","button button-outline");this.retry.type="button";this.retry.hidden=true;
      this.retry.onclick=()=>this.load(this.base,this.options);
      this.status=node("p",null,"secondary"); this.status.setAttribute("role","status");
      this.play=node("button","Play","button button-outline");this.play.type="button";
      this.play.onclick=()=>video.paused ? video.play().catch(e=>this.message(e.message)) : video.pause();
      this.seek=node("input"); this.seek.type="range";this.seek.min="0";this.seek.step="0.001";this.seek.value="0";
      this.seek.setAttribute("aria-label","Episode playback time");
      this.seek.oninput=()=>{video.currentTime=Number(this.seek.value);this.draw();};
      this.time=node("output");
      const controls=node("div",null,"episode-viewer-playback");controls.append(this.play,this.seek,this.time);
      this.aside=node("aside",null,"episode-viewer-hand");
      this.main.append(this.tabs,this.layers,this.legend,this.stage,this.sceneHelp,controls,this.status,this.retry);
      host.append(this.main,this.aside);
      this.events={};
      for (const event of ["timeupdate","loadeddata","seeked","pause","play","ended"]) {
        this.events[event]=()=>{
          if(!video.requestVideoFrameCallback && !video.seeking)this.frameTime=video.currentTime;
          this.draw();if(event==="play")this.tick();
        };
        video.addEventListener(event,this.events[event]);
      }
      this.observer=new ResizeObserver(()=>this.draw());this.observer.observe(this.stage);
    }

    message(text) {this.status.textContent=text || "";}
    tick() {cancelAnimationFrame(this.animation);if(!this.video.paused && this.active){this.draw();this.animation=requestAnimationFrame(()=>this.tick());}}
    watchFrames() {
      if(!this.video.requestVideoFrameCallback)return;
      this.frameCallback=this.video.requestVideoFrameCallback((_, metadata)=>{
        if(!this.active)return;
        this.frameTime=metadata.mediaTime;this.draw();this.watchFrames();
      });
    }

    async json(url, options={}) {
      const response=await fetch(url,{...options,signal:AbortSignal.timeout(45000)});
      const value=await response.json();
      if(!response.ok)throw new Error(typeof value.detail==="string"?value.detail:"Episode data could not be loaded");
      return value;
    }

    async load(base,{collection=false,episode=0,robot=null}={}) {
      this.close();this.active=true;this.frameTime=null;this.retry.hidden=true;
      this.options={collection,episode,robot};this.watchFrames();
      const generation=++this.generation;this.base=base;this.collection=collection;this.episode=episode;
      this.canvas.width=this.canvas.width;this.data=null;this.view="all";this.legend.textContent="";this.aside.replaceChildren(node("h4","Hand"),node("p","Loading hand information…","secondary"));
      this.renderControls();this.message("Loading episode views…");
      if(robot)this.loadHand(robot,generation);
      const query=collection?`?episode=${episode}`:"";
      try {
        const result=await this.json(base+"/viewer"+query,collection?{method:"POST"}:{});
        if(generation!==this.generation)return;
        if(result.state==="PREPARING") {
          this.message(result.detail);this.pollTimer=setTimeout(()=>this.poll(generation),1200);return;
        }
        await this.accept(result,generation);
      } catch(error) {if(generation===this.generation){this.message(error.message);this.retry.hidden=false;}}
    }

    async poll(generation) {
      try {
        const result=await this.json(this.base+`/viewer?episode=${this.episode}`);
        if(generation!==this.generation)return;
        if(result.state==="PREPARING") {this.pollTimer=setTimeout(()=>this.poll(generation),1500);return;}
        await this.accept(result,generation);
      } catch(error) {if(generation===this.generation){this.message(error.message);this.retry.hidden=false;}}
    }

    async accept(result,generation) {
      if(result.state!=="READY") {
        this.retry.hidden=result.state!=="FAILED";
        this.message(result.detail || "This result has no saved camera or keypoint data.");
        if(result.robot)this.loadHand(result.robot,generation);
        return;
      }
      const data=result.viewer || await this.json(this.base+`/viewer/viewer.json?episode=${this.episode}`);
      if(generation!==this.generation)return;
      this.data=data;this.renderControls();this.loadHand(data.robot,generation);
      this.legend.textContent=this.collection?"": "Prediction: model target · Actual: observed hand · Demonstration: original recording";
      if(data.demonstration)this.legend.textContent+=` · Source ${data.demonstration.session_id.slice(0,8)}, recording ${data.demonstration.source_index+1}. Aligned by elapsed time; hidden when the recording ends.`;
      if(this.collection) {this.video.src=this.base+`/viewer/views.mp4?episode=${this.episode}`;this.video.load();}
      this.message((data.warnings || []).join(" "));
      this.draw();
    }

    renderControls() {
      this.tabs.replaceChildren();this.layers.replaceChildren();
      const views=this.data?.views || [];
      const choices=[{id:"all",label:views.length?"All cameras":"Video"},...views,{id:"interactive",label:"Interactive"}];
      for(const choice of choices) {
        const button=node("button",choice.label,"button button-outline");button.type="button";
        button.setAttribute("aria-pressed",String(this.view===choice.id));
        button.disabled=choice.id==="interactive" && !this.data?.frames?.some(frame=>frame.actual);
        button.onclick=async()=>{
          this.view=choice.id;this.renderControls();
          if(choice.id==="interactive" && !this.scene) {
            const generation=this.generation;
            try {const {EpisodeScene}=await import("./episode-scene.js"+version);
              if(generation===this.generation && this.active)this.scene=new EpisodeScene(this.sceneHost);
            }catch(error){this.message("Interactive view unavailable: "+error.message);}
          }
          this.draw();
        };
        this.tabs.append(button);
      }
      const labels=this.collection?{actual:"Keypoints"}:{prediction:"Prediction",actual:"Actual",demonstration:"Demonstration"};
      for(const [key,label] of Object.entries(labels)) {
        const available=Boolean(this.data?.frames?.some(frame=>frame[key]));
        const wrapper=node("label",null,"check-field");const input=node("input");input.type="checkbox";
        input.checked=available && this.enabled.has(key);input.disabled=!available;
        input.onchange=()=>{input.checked?this.enabled.add(key):this.enabled.delete(key);this.draw();};
        const text=node("span",label);text.style.color=colors[key];wrapper.append(input,text);
        if(!available)wrapper.title="No saved "+label.toLowerCase()+" keypoints for this episode";
        this.layers.append(wrapper);
      }
    }

    async loadHand(robot,generation) {
      try {
        const {hands}=await this.json("/api/hands");
        const hand=hands.find(h=>robot?.startsWith("skynet_"+h.key.replaceAll("-","_")+"_"))
          || (robot?.startsWith("floating_shadow_")?hands.find(h=>h.key==="shadow"):null);
        if(generation!==this.generation)return;
        if(!hand){this.aside.replaceChildren(node("h4","Hand"),node("p","Hand definition is not available in Hands."));return;}
        const side=robot.endsWith("_bimanual")?"both":robot.endsWith("_left")?"left":"right";
        this.aside.replaceChildren(node("h4",hand.name),node("p",side==="both"?"Both hands":side+" hand"),node("p",hand.notes));
        const dl=node("dl");for(const [key,value] of [["Source",hand.source_kind],["Revision",hand.revision?.slice(0,12)],["Wrist","6 axes"]]) {dl.append(node("dt",key),node("dd",value||"Unavailable"));}
        this.aside.append(dl);
        const link=node("a","View in Hands","text-button");link.href=`/?hand=${encodeURIComponent(hand.key)}&side=${side==="both"?"right":side}#hands`;
        this.aside.append(link);
        if(robot.startsWith("floating_shadow_"))this.aside.append(node("p","Recorded with the legacy DexVerse model. Replay uses its saved joint layout.","secondary"));
      }catch(error){if(generation===this.generation)this.aside.replaceChildren(node("h4","Hand"),node("p",error.message,"secondary"));}
    }

    draw() {
      if(!this.active)return;
      const time=this.video.currentTime || 0, duration=this.data?.duration || this.video.duration || 0;
      this.play.textContent=this.video.paused?"Play":"Pause";this.seek.max=String(Number.isFinite(duration)?duration:0);this.seek.value=String(time);
      this.time.textContent=`${time.toFixed(2)} / ${Number.isFinite(duration)?duration.toFixed(2):"—"} s`;
      this.play.disabled=this.video.readyState<2;this.seek.disabled=this.video.readyState<2;
      const renderedTime=this.view==="interactive"?time:(this.frameTime ?? 0);
      const frames=this.data?.frames || [];
      let lo=0,hi=frames.length-1;
      while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(frames[mid].time<=renderedTime+1e-6)lo=mid;else hi=mid-1;}
      const frame=frames[lo];
      this.sceneHelp.hidden=this.view!=="interactive";
      this.canvas.hidden=this.view==="interactive";this.sceneHost.hidden=this.view!=="interactive";
      if(this.view==="interactive") {this.scene?.update(frame,this.data?.edges || [],this.enabled,this.collection);return;}
      if(this.video.readyState<2 || this.frameTime===null)return;
      const views=this.data?.views || [];
      const selected=views.find(view=>view.id===this.view);
      const rect=selected?.rect || [0,0,this.video.videoWidth,this.video.videoHeight];
      if(!rect[2] || !rect[3])return;
      if(this.canvas.width!==rect[2])this.canvas.width=rect[2];if(this.canvas.height!==rect[3])this.canvas.height=rect[3];
      const context=this.canvas.getContext("2d");context.drawImage(this.video,...rect,0,0,rect[2],rect[3]);
      for(const view of selected?[selected]:views) {
        context.save();const x=selected?0:view.rect[0],y=selected?0:view.rect[1];
        context.beginPath();context.rect(x,y,view.width,view.height);context.clip();
        for(const key of this.enabled) {
          const points=frame?.[key]?.map(point=>project(point,view));if(!points)continue;
          context.strokeStyle=context.fillStyle=colors[key];context.lineWidth=1.5;
          for(const [a,b] of this.data.edges || [])if(points[a]&&points[b]){context.beginPath();context.moveTo(x+points[a][0],y+points[a][1]);context.lineTo(x+points[b][0],y+points[b][1]);context.stroke();}
          for(const p of points)if(p){context.beginPath();context.arc(x+p[0],y+p[1],2.5,0,Math.PI*2);context.fill();}
        }
        context.restore();
      }
    }

    close() {this.active=false;this.video.pause();if(this.frameCallback!==undefined)this.video.cancelVideoFrameCallback?.(this.frameCallback);++this.generation;clearTimeout(this.pollTimer);cancelAnimationFrame(this.animation);this.scene?.dispose();this.scene=null;}
  }

  window.SkynetEpisodeViewer={
    open(hostId,videoId,base,options) {let viewer=instances.get(hostId);if(!viewer){viewer=new EpisodeViewer(document.getElementById(hostId),document.getElementById(videoId));instances.set(hostId,viewer);}viewer.load(base,options);return viewer;},
    close(hostId) {instances.get(hostId)?.close();},
  };
})();
