const $=id=>document.getElementById(id);
const runId=new URL(location).searchParams.get('run');
let meal=null,frame=null,selected=null,busy=false,ready=false,dirty=false,drag=null,suppressUntil=0;
let desired={yaw:0,pitch:0,zoom:1},cameraId=null,timer=null,epoch=0,lastSent=-Infinity,displayedKey=null;
const frameInterval=65;
const status=text=>{$('modelStatus').textContent=text;};
async function api(path,body){const response=await fetch('/api'+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const value=await response.json();if(!response.ok){const error=new Error(value.detail||'操作未完成');error.code=response.status;throw error;}return value;}
function selectionText(food){
 if(!food)return '';
 const volume=food.volume?.volume_ml;
 const basis=meal.volume_summary?.scale_source==='pansense_aruco_board_multiview'?' · 定位板估算':'';
 return `ID ${food.instance_id}：${food.name}${volume!=null?` · 约 ${volume.toFixed(1)} mL${basis}`:''}`;
}
function foodButtons(){const parent=$('foods');parent.replaceChildren();for(const food of meal.items){const button=document.createElement('button');button.className='button secondary'+(selected===food.id?' selected':'');button.textContent=selectionText(food);button.onclick=()=>{selected=food.id;$('selection').textContent=selectionText(food);foodButtons();schedule();};parent.append(button);}}
// Fixed-rate sampling, not debounce: pointer moves never postpone a frame.
// One request in flight; all subsequent gestures coalesce into the latest pose.
function schedule(minDelay=0){dirty=true;if(busy||timer!==null)return;timer=setTimeout(()=>{timer=null;render();},Math.max(minDelay,frameInterval-(performance.now()-lastSent)));}
async function render(){if(!meal?.three_d_available||busy)return;const key=JSON.stringify([epoch,desired,selected]);if(frame&&key===displayedKey){dirty=false;return;}busy=true;dirty=false;lastSent=performance.now();const currentEpoch=epoch;let retryDelay=0;const body={...desired,food_id:selected,...(frame?{base_token:frame.token}:{frame_id:cameraId})};
 try{const value=await api(`/meals/${runId}/view`,body);if(currentEpoch!==epoch)return;const image=$('modelImage'),preload=new Image();await new Promise((resolve,reject)=>{preload.onload=resolve;preload.onerror=()=>reject(new Error('三维画面未加载，请重试'));preload.src=value.image_url;});if(currentEpoch!==epoch)return;image.src=value.image_url;frame=value;displayedKey=key;image.hidden=false;$('loading').hidden=true;ready=true;
  if(!$('camera').options.length){for(const id of value.captured_frames){const option=document.createElement('option');option.value=id;option.textContent=id;option.selected=id===value.frame_id;$('camera').append(option);}}
  status('可拖动旋转 · 滚轮缩放 · 点击食物读取三维语义');
 }catch(error){status(error.message);if(error.code===409){dirty=true;retryDelay=1200;}else $('loading').textContent=error.message;}
 finally{busy=false;if(dirty)schedule(retryDelay);}
}
function reset(){epoch++;frame=null;desired={yaw:0,pitch:0,zoom:1};schedule();}
$('resetView').onclick=reset;$('camera').onchange=()=>{cameraId=$('camera').value;reset();};
function zoom(factor){desired.zoom=Math.max(.5,Math.min(2.5,desired.zoom*factor));suppressUntil=performance.now()+300;schedule();}
$('zoomIn').onclick=()=>zoom(.85);$('zoomOut').onclick=()=>zoom(1.15);
const image=$('modelImage');
image.addEventListener('pointerdown',event=>{if(!frame||event.button!==0||!event.isPrimary)return;const box=image.getBoundingClientRect();drag={id:event.pointerId,x:event.clientX,y:event.clientY,width:box.width,height:box.height,base:{...desired},moved:false};image.setPointerCapture(event.pointerId);});
image.addEventListener('pointermove',event=>{if(!drag||drag.id!==event.pointerId)return;const dx=event.clientX-drag.x,dy=event.clientY-drag.y;if(Math.hypot(dx,dy)>5)drag.moved=true;if(!drag.moved)return;event.preventDefault();$('modelShell').classList.add('dragging');desired.yaw=((drag.base.yaw-120*dx/drag.width+540)%360)-180;desired.pitch=Math.max(-75,Math.min(75,drag.base.pitch-90*dy/drag.height));schedule();});
image.addEventListener('pointerup',event=>{if(drag?.id!==event.pointerId)return;if(drag.moved){suppressUntil=performance.now()+400;schedule();}drag=null;$('modelShell').classList.remove('dragging');});
image.addEventListener('pointercancel',()=>{drag=null;$('modelShell').classList.remove('dragging');});
image.addEventListener('wheel',event=>{event.preventDefault();zoom(Math.exp(Math.max(-200,Math.min(200,event.deltaY))*.0015));},{passive:false});
image.addEventListener('click',async event=>{if(!ready||busy||dirty||!frame||performance.now()<suppressUntil)return;const box=image.getBoundingClientRect();try{const value=await api(`/meals/${runId}/pick`,{token:frame.token,x:(event.clientX-box.left)/box.width,y:(event.clientY-box.top)/box.height});selected=value.food_id;$('selection').textContent=selectionText(meal.items.find(item=>item.id===selected))||value.message;foodButtons();schedule();}catch(error){status(error.message);}});
async function load(){if(!runId){status('请从识别对比页选择一个版本');return;}try{meal=await api(`/meals/${runId}`);$('back').href='/recognition.html?session='+meal.comparison_parent;if(meal.three_d_available){foodButtons();await render();return;}const job=meal.three_d_analysis;status(job?`${job.stage} · ${job.progress}%`:(meal.three_d_reason||'请先选择版本并生成三维关联'));$('loading').textContent=job?.error||job?.stage||meal.three_d_reason||'三维模型尚未生成';if(['queued','running'].includes(job?.status))setTimeout(load,1500);}catch(error){status(error.message);}}
load();
