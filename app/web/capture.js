import {scanMapHTML,frameListHTML} from './scan-map.js';
import {THUMB_W,THUMB_H,SAME_VIEW_DIFF,grayThumbnail,meanAbsDiff} from './frame-diff.js';
const $=id=>document.getElementById(id);
const [scanId,token]=location.hash.slice(1).split('/');
let scan=null,stream=null,running=false,uploadTask=null,timer=null,selectedSector=null,pollTimer=null,captureEpoch=0,manualMode=false;
const images=new Map();let cameraSize=null;
function message(text,error=false){$('scanMessage').textContent=text;$('scanMessage').classList.toggle('error',error);}
async function api(path,options={}){
 const headers={Authorization:`Bearer ${token}`,...options.headers};
 if(options.body&&!(options.body instanceof FormData)){headers['Content-Type']='application/json';options={...options,body:JSON.stringify(options.body)};}
 const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),20000);
 try{
  const response=await fetch(path,{...options,headers,cache:'no-store',signal:controller.signal});
  const value=await response.json();if(!response.ok)throw new Error(typeof value.detail==='string'?value.detail:'输入或连接有误，请重试');return value;
 }catch(error){if(error.name==='AbortError')throw new Error('电脑响应超时，请检查 Wi-Fi 连接');if(error instanceof TypeError)throw new Error('无法连接电脑，请检查 Wi-Fi 和电脑服务');throw error;}
 finally{clearTimeout(timeout);}
}
async function refresh(){scan=await api(`/api/scans/${scanId}`);render();}
function render(){
 $('scanTitle').textContent=scan.title;$('savedCount').textContent=`${scan.frame_count} 张已保存 · ${scan.good_frames} 张清晰`;
 $('coverageMap').innerHTML=scanMapHTML(scan,manualMode?selectedSector:null);$('captureFrames').innerHTML=frameListHTML(scan,selectedSector);
 $('photoHeading').textContent=selectedSector===null?'最近照片':`${selectedSector<12?'较低视角':'较高俯视'} · 第 ${selectedSector%12+1} 格`;
 $('connectionStatus').textContent='已连接电脑';$('startCapture').disabled=running||!!scan.meal_id;
 $('startCapture').textContent=scan.meal_id?'已送入电脑重建':scan.frame_count?'继续扫描':'开启摄像头并扫描';
 $('pauseCapture').disabled=!running;$('shotCapture').hidden=!manualMode;$('shotCapture').disabled=!running||selectedSector===null||shooting;$('finishCapture').disabled=!!scan.meal_id||!scan.frame_count;
 for(const image of document.querySelectorAll('[data-scan-frame-src]'))loadImage(image).catch(()=>{});
}
async function loadImage(image){
 const url=image.dataset.scanFrameSrc;
 if(!images.has(url)){
  const response=await fetch(url,{headers:{Authorization:`Bearer ${token}`},cache:'no-store'});
  if(!response.ok)return;images.set(url,URL.createObjectURL(await response.blob()));
 }
 image.src=images.get(url);
}
async function state(status){
 scan=await api(`/api/scans/${scanId}`);
 scan=await api(`/api/scans/${scanId}`,{method:'PATCH',body:{revision:scan.revision,status}});render();
}
function stopCamera(){stream?.getTracks().forEach(track=>track.stop());stream=null;$('captureVideo').srcObject=null;$('cameraPlaceholder').hidden=false;$('cameraState').textContent='摄像头已关闭';}
async function pause(quiet=false){
 captureEpoch++;running=false;clearTimeout(timer);stopCamera();
 try{await settle();await state('paused');if(!quiet)message('已暂停，照片保存在电脑上，可继续补拍。');}
 catch(error){if(!quiet)message(error.message,true);}
}
let shooting=false,lastThumb=null,skipped=0,thumbContext=null;
const inflight=new Set(),retryQueue=[],MAX_INFLIGHT=2,INTERVAL=700;
const settle=()=>Promise.allSettled([...inflight,uploadTask].filter(Boolean));
// Several uploads can overlap, so keep whichever response is newest.
function applyScan(value){if(!scan||value.revision>scan.revision)scan=value;}
async function grab(manual=null,dedupe=false){
 const video=$('captureVideo');
 if(!video.videoWidth)throw new Error('相机画面尚未就绪，请稍后重试');
 const scale=Math.min(1,1280/Math.max(video.videoWidth,video.videoHeight));
 const dimensions=[Math.round(video.videoWidth*scale),Math.round(video.videoHeight*scale)];
 if(cameraSize&&dimensions.some((v,i)=>v!==cameraSize[i]))throw new Error('拍摄方向或镜头已变化，请恢复原方向后继续');
 cameraSize=dimensions;
 thumbContext??=Object.assign(document.createElement('canvas'),{width:THUMB_W,height:THUMB_H}).getContext('2d',{willReadFrequently:true});
 const thumb=grayThumbnail(thumbContext,video);
 if(dedupe&&meanAbsDiff(thumb,lastThumb)<SAME_VIEW_DIFF){skipped++;return null;}
 const canvas=document.createElement('canvas');[canvas.width,canvas.height]=dimensions;
 canvas.getContext('2d').drawImage(video,0,0,...dimensions);
 const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.92));
 if(!blob)throw new Error('无法获取相机画面');
 lastThumb=thumb;
 return {id:crypto.randomUUID(),blob,timestamp:performance.now()/1000,manual};
}
async function upload(frame){
 const form=new FormData();form.append('file',frame.blob,'capture.jpg');form.append('frame_id',frame.id);form.append('timestamp',String(frame.timestamp));
 if(frame.manual!==null)form.append('manual_sector',String(frame.manual));
 try{
  applyScan(await api(`/api/scans/${scanId}/frames`,{method:'POST',body:form}));render();
  const latest=scan.frames.find(f=>f.id===frame.id)||scan.frames.at(-1);
  if(frame.manual!==null){message(latest.quality==='blur'?'画面模糊，这张不会计入；请停稳后重拍。':latest.quality==='exposure'?'曝光不合适，请调整光线后重拍。':latest.duplicate_view?'与已有照片重复，请换个位置再拍。':`已保存为${frame.manual<12?'较低':'较高'}视角第 ${frame.manual%12+1} 格（手动标注）。移动到下一个方向后再点拍摄。`);return;}
  message(latest.quality==='blur'?'画面模糊，请停稳一点；这张不会计入已拍角度。':latest.quality==='exposure'?'曝光不合适，请调整光线。':latest.duplicate_view?'这个视点已有照片，请继续绕行。':!latest.located?'照片已保存，角度待确认。请保持多个定位标记可见，并变换拍摄高度。':`已保存。${scan.coverage.covered}/24 个方向已覆盖，请继续补拍灰色方向。`);
 }catch(error){
  retryQueue.push(frame);running=false;clearTimeout(timer);stopCamera();
  message(`${error.message}。未确认的照片会在继续扫描时重试。`,true);if(scan)render();
 }
}
// Auto mode: capture on a fixed beat, with up to MAX_INFLIGHT uploads overlapping.
async function loop(){
 if(!running||!stream)return;
 if(inflight.size<MAX_INFLIGHT){
  try{
   const frame=retryQueue.shift()||await grab(null,true);
   if(!running)return;
   if(frame){const task=upload(frame);inflight.add(task);task.finally(()=>inflight.delete(task));}
   $('cameraState').textContent=`后置摄像头 · 自动保存中${skipped?` · 已跳过 ${skipped} 张重复画面`:''}`;
  }catch(error){running=false;clearTimeout(timer);stopCamera();message(error.message,true);if(scan)render();return;}
 }
 if(running)timer=setTimeout(loop,INTERVAL);
}
async function start(){
 if(running)return;
 if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia){message('当前页面无法调用摄像头。请安装电脑提供的 CA 证书，并通过拍摄二维码打开可信 HTTPS 页面。',true);return;}
 $('startCapture').disabled=true;
 const epoch=++captureEpoch;
 try{
  stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:960}},audio:false});
  if(epoch!==captureEpoch||document.hidden){stopCamera();return;}
  const video=$('captureVideo');video.srcObject=stream;await video.play();
  if(!video.videoWidth)await new Promise(resolve=>video.addEventListener('loadeddata',resolve,{once:true}));
  await state('capturing');running=true;$('cameraPlaceholder').hidden=true;$('cameraState').textContent='后置摄像头 · 自动保存中';
  render();
  if(manualMode){message('手动模式：先点圆环里你所在的方向格，再点“拍摄所选方向”。');}
  else{message('保持食物静止，缓慢绕行。请让定位板上的多个标记保持可见。');lastThumb=null;loop();}
 }catch(error){stopCamera();running=false;message(error.name==='NotAllowedError'?'摄像头未获授权，请在浏览器站点设置中允许相机后重试。':error.message,true);if(scan)render();}
}
$('startCapture').addEventListener('click',start);$('pauseCapture').addEventListener('click',()=>pause());
$('finishCapture').addEventListener('click',async()=>{
 try{running=false;clearTimeout(timer);stopCamera();await settle();await state('complete');message('扫描已保存。可检查下方照片、继续补拍，或回到电脑选择分割方式并生成餐盘。');}
 catch(error){message(error.message,true);}
});
$('showAllFrames').addEventListener('click',()=>{selectedSector=null;render();});
$('shotCapture').addEventListener('click',async()=>{
 if(!running||selectedSector===null||shooting)return;
 shooting=true;render();
 try{const frame=retryQueue.shift()||await grab(selectedSector);uploadTask=upload(frame);await uploadTask;}
 catch(error){message(error.message,true);}
 finally{shooting=false;if(scan)render();}
});
for(const radio of document.querySelectorAll('input[name=scanMode]'))radio.addEventListener('change',async()=>{
 if(!radio.checked)return;manualMode=radio.value==='manual';
 if(running){await pause(true);message('已切换记录方式，请重新点击“继续扫描”。');}
 selectedSector=null;if(scan)render();
});
document.addEventListener('click',async event=>{
 const sector=event.target.closest('[data-scan-sector]');if(sector){selectedSector=Number(sector.dataset.scanSector);render();return;}
 const remove=event.target.closest('[data-scan-remove]');if(remove){try{await pause(true);await refresh();scan=await api(`/api/scans/${scanId}/frames/${remove.dataset.scanRemove}?revision=${scan.revision}`,{method:'DELETE'});render();}catch(error){message(error.message,true);}return;}
 const photo=event.target.closest('[data-scan-photo]');if(photo){const image=photo.querySelector('img');if(image?.src){$('fullPhoto').src=image.src;$('photoDialog').showModal();}}
});
document.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)&&event.target.matches('[data-scan-sector]')){event.preventDefault();event.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
$('closePhoto').addEventListener('click',()=>$('photoDialog').close());
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause(true);});
window.addEventListener('pagehide',()=>{captureEpoch++;running=false;clearTimeout(timer);stopCamera();for(const url of images.values())URL.revokeObjectURL(url);images.clear();});
async function poll(){
 try{if(!running&&scan&&!document.hidden)await refresh();}catch(error){$('connectionStatus').textContent='电脑连接暂时中断';}
 pollTimer=setTimeout(poll,3000);
}
if(!/^[a-f0-9]{32}$/.test(scanId||'')||!token){message('请从电脑“手机扫描”入口扫描拍摄二维码。',true);}
else{try{await refresh();message(scan.status==='complete'?'扫描已结束，可继续补拍或回电脑生成餐盘。':'已连接电脑，点击开启摄像头开始。');poll();}catch(error){message(error.message,true);$('connectionStatus').textContent='连接失败';}}
