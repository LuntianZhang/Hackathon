import {scanMapHTML,frameListHTML} from './scan-map.js';
const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const statuses={capturing:'手机正在拍摄',paused:'已暂停，可继续',complete:'已结束，可检查照片'};
export const phoneScan={
 active:false,sessions:[],connection:null,scan:null,pairing:null,error:'',busy:false,timer:null,sector:null,
 buildMode:'llm',foodText:'',lanHost:'',boardBackground:true,
 configure(callbacks){Object.assign(this,callbacks);},
 async enter(){this.active=true;await this.refresh();this.schedule();},
 leave(){this.active=false;clearTimeout(this.timer);},
 schedule(){clearTimeout(this.timer);if(this.active)this.timer=setTimeout(async()=>{
  try{const previous=this.scan?.revision;await this.refresh();if(previous!==this.scan?.revision&&!document.activeElement?.closest('.scan-build-options'))this.redraw();}
  catch(error){this.error=error.message;this.redraw();}this.schedule();
 },1800);},
 async refresh(){
  const value=await this.api('/scans');this.sessions=value.scans;this.connection=value.connection;
  if(this.scan){const next=this.sessions.find(s=>s.id===this.scan.id);this.scan=next||null;}
 },
 html(){
  const connection=this.connection;
  return `<div class="desktop-scan-panel"><div class="eyebrow">PHONE CAMERA</div><h1 class="screen-heading">手机环绕扫描</h1><p class="tiny muted">手机只负责拍摄，照片与计算留在这台电脑。</p>
   ${this.error?`<p class="record-note">${esc(this.error)}</p>`:''}
   ${connection?.enabled?`<p class="scan-setup-note">手机拍摄服务已开启 · 手机与电脑连接同一 Wi-Fi。<br><a href="${esc(connection.setup_url)}" target="_blank" rel="noopener">安卓首次连接设置</a></p><details class="scan-pairing"><summary>首次连接安卓手机：安装此电脑的相机证书</summary><p class="tiny muted">扫码下载证书，在安卓设置中安装 CA 证书，再扫描下方拍摄二维码。仅需首次设置。</p><img src="${connection.setup_qr}" alt="安卓证书设置二维码"><a href="${esc(connection.setup_url)}" target="_blank" rel="noopener">${esc(connection.setup_url)}</a><p class="scan-fingerprint">请核对证书指纹：${esc(connection.certificate_fingerprint)}</p></details><div class="scan-controls"><button class="button primary" data-phone-action="create" ${this.busy?'disabled':''}>＋ 新建手机扫描</button><button class="button ghost" data-phone-action="disconnect" ${this.busy?'disabled':''}>关闭手机连接</button></div>`:
   `<p class="scan-setup-note">开启后，电脑会提供 HTTPS 拍摄页。手机无需安装 App，饮食日记仍只在电脑端开放。</p>${connection?.addresses?.length>1?`<div class="scan-build-options"><label for="scanLanHost">选择手机同一 Wi-Fi 的电脑地址</label><select id="scanLanHost">${connection.addresses.map(host=>`<option ${host===this.lanHost?'selected':''}>${esc(host)}</option>`).join('')}</select></div>`:''}<button class="button primary" data-phone-action="connect" ${this.busy?'disabled':''}>${this.busy?'正在开启 HTTPS 拍摄…':'开启手机连接'}</button>`}
   ${this.scan?this.selectedHTML():''}
   <div class="row"><h2 class="section-head">扫描记录</h2><span class="tiny muted">${this.sessions.length} 条</span></div>
   ${this.sessions.map(s=>`<button class="scan-session-line" data-phone-action="select" data-scan-id="${s.id}" aria-pressed="${s.id===this.scan?.id}"><strong>${esc(s.title)}</strong><small>${new Date(s.created_at).toLocaleString('zh-CN')} · ${s.frame_count} 张 · ${s.coverage.covered}/24 方向<br>${s.meal_id?'已送入重建':statuses[s.status]}</small></button>`).join('')||'<div class="empty">还没有手机扫描记录。<br>开启连接后，新建一次扫描。</div>'}
   <p class="estimate-note"><a href="/api/scans/board-20cm.png" target="_blank" rel="noopener">打印 20×20 cm 定位板</a> · <a href="/api/scans/board.svg" target="_blank" rel="noopener">A3 定位板</a> · 没有定位板时，手机页可选“手动选方向”，角度为手动标注。</p></div>`;
 },
 selectedHTML(){
  const scan=this.scan,pairing=this.pairing,report=scan.verification?.coverage||scan.coverage;
  return `<h2 class="scan-selected-heading">${esc(scan.title)}</h2><p class="tiny muted">${statuses[scan.status]} · ${scan.frame_count} 张已保存 · ${scan.good_frames} 张清晰</p>
   ${pairing?.url?`<div class="scan-pairing"><strong>用手机扫描，打开拍摄页</strong><img src="${pairing.qr}" alt="此扫描的手机拍摄二维码"><a href="${esc(pairing.url)}" target="_blank" rel="noopener noreferrer">${esc(pairing.url.split('#')[0])}</a><p class="tiny muted">二维码仅授权本次扫描。请勿转发给他人。</p></div>`:this.connection?.enabled&&!scan.meal_id?`<button class="button ghost" data-phone-action="pair" ${this.busy?'disabled':''}>获取新的拍摄二维码</button><p class="tiny muted">重新配对后，原拍摄链接将失效。</p>`:''}
   ${scanMapHTML(scan)}${scan.verification?`<p class="record-note">已注册 ${scan.verification.registered_frames}/${scan.verification.submitted_frames} 张照片，复核覆盖 ${report.covered}/24 方向。</p>`:''}
   <div class="scan-controls"><button class="button ghost" data-phone-action="pause" ${scan.status!=='capturing'||this.busy?'disabled':''}>暂停拍摄</button><button class="button ghost" data-phone-action="finish" ${!scan.frame_count||scan.meal_id||this.busy?'disabled':''}>结束并检查</button></div>
   <div class="row"><h2 class="section-head">${this.sector===null?'最近照片':`${this.sector<12?'较低视角':'较高俯视'} · 第 ${this.sector%12+1} 格`}</h2><button class="text-btn" data-phone-action="all-frames">查看最近照片</button></div><div class="scan-frames">${frameListHTML(scan,this.sector)}</div>
   ${scan.meal_id?`<button class="button primary" data-phone-action="open-meal">查看餐盘与处理进度</button>`:
   `<section class="scan-build-options"><label for="scanBuildMode">食物分割方式</label><select id="scanBuildMode">${[['list','食物清单分割'],['grounded','名称定位分割'],['manual','点击 / 框选分割'],['llm','大模型识别（推荐）'],['none','不使用语义分割']].map(([value,label])=>`<option value="${value}" ${this.buildMode===value?'selected':''}>${label}</option>`).join('')}</select>${this.buildMode==='llm'?'<p class="tiny muted">只识别白色定位板内的物体，并联合多个视角追踪；所选图片会发送至已登录 Agent 或所配置的模型服务。</p>':''}${this.buildMode==='manual'?'<p class="tiny muted">先关联扫描相机并准备固定参考图，再点击或框选食物。后续继续使用同一批相机与照片。</p>':''}${['list','grounded'].includes(this.buildMode)?`<label for="scanFoodText">有哪些食物？</label><textarea id="scanFoodText" placeholder="香蕉、青提、葡萄">${esc(this.foodText)}</textarea>`:''}<label class="board-background-control"><input id="scanBoardBackground" type="checkbox" ${this.boardBackground?'checked':''}>自动跟随定位板排除背景</label><p class="tiny muted">定位板与物体一起转动，至少露出 3 个标记。会将关键帧发给当前 Agent 使用的模型服务，再在本机跟踪完整轮廓与排除背景，需要额外时间。按原尺寸打印定位板以估算体积。</p><button class="button primary" data-phone-action="build" ${scan.status!=='complete'||scan.good_frames<20||this.busy?'disabled':''}>${this.busy?'正在提交…':this.buildMode==='manual'?'准备扫描参考图，继续点框标注':'使用扫描照片生成三维餐盘'}</button><p class="tiny muted">结束扫描后可提交，至少需要 20 张清晰且不重复的照片。电脑先用 COLMAP 复核，再进行重建。</p></section>`}`;
 },
 mount(){
  for(const image of document.querySelectorAll('.desktop-scan-panel [data-scan-frame-src]'))image.src=image.dataset.scanFrameSrc;
 },
 async action(name,id){
  if(this.busy)return;this.busy=true;this.error='';this.redraw();
  try{
   if(name==='connect'){const value=await this.api('/scans/connection',{method:'POST',body:this.lanHost?{host:this.lanHost}:{}});this.connection=value.connection;this.sessions=value.scans;}
   if(name==='disconnect'){await this.api('/scans/connection',{method:'DELETE'});this.pairing=null;await this.refresh();}
   if(name==='create'){const value=await this.api('/scans',{method:'POST',body:{title:'手机扫描 · '+new Date().toLocaleString('zh-CN',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'})}});this.scan=value.scan;this.pairing=value.pairing;this.sector=null;await this.refresh();}
   if(name==='select'){this.scan=this.sessions.find(s=>s.id===id);this.pairing=null;this.sector=null;}
   if(name==='pair'){const value=await this.api(`/scans/${this.scan.id}/pair`,{method:'POST'});this.scan=value.scan;this.pairing=value.pairing;}
   if(['pause','finish'].includes(name)){await this.refresh();this.scan=await this.api(`/scans/${this.scan.id}`,{method:'PATCH',body:{revision:this.scan.revision,status:name==='pause'?'paused':'complete'}});}
   if(name==='all-frames')this.sector=null;
   if(name==='build'){
    if(['list','grounded'].includes(this.buildMode)&&!this.foodText.trim())throw new Error('请填写食物名称');
    const meal=await this.api(`/scans/${this.scan.id}/build`,{method:'POST',body:{assistance:{mode:this.buildMode==='manual'?'auto':this.buildMode,food_text:['list','grounded'].includes(this.buildMode)?this.foodText:''},prepare:this.buildMode==='manual',board_background:this.boardBackground}});
    await this.refresh();await this.openMeal(meal.id);
   }
   if(name==='open-meal')await this.openMeal(this.scan.meal_id);
  }catch(error){this.error=error.message;}
  finally{this.busy=false;if(this.active)this.redraw();}
 }
};
document.addEventListener('click',event=>{
 if(!phoneScan.active)return;
 const button=event.target.closest('[data-phone-action]');if(button&&!button.disabled){phoneScan.action(button.dataset.phoneAction,button.dataset.scanId);return;}
 const sector=event.target.closest('.desktop-scan-panel [data-scan-sector]');if(sector){phoneScan.sector=Number(sector.dataset.scanSector);phoneScan.redraw();return;}
 const remove=event.target.closest('.desktop-scan-panel [data-scan-remove]');if(remove&&!remove.disabled&&!phoneScan.busy){
  (async()=>{try{await phoneScan.refresh();phoneScan.scan=await phoneScan.api(`/scans/${phoneScan.scan.id}/frames/${remove.dataset.scanRemove}?revision=${phoneScan.scan.revision}`,{method:'DELETE'});phoneScan.redraw();}catch(error){phoneScan.error=error.message;phoneScan.redraw();}})();return;
 }
 const photo=event.target.closest('.desktop-scan-panel [data-scan-photo]');if(photo){const frame=phoneScan.scan.frames.find(f=>f.id===photo.dataset.scanPhoto);if(frame)window.open(frame.url,'_blank','noopener');}
});
document.addEventListener('keydown',event=>{if(phoneScan.active&&['Enter',' '].includes(event.key)&&event.target.matches('.desktop-scan-panel [data-scan-sector]')){event.preventDefault();event.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
document.addEventListener('input',event=>{if(event.target.id==='scanFoodText')phoneScan.foodText=event.target.value;});
document.addEventListener('change',event=>{if(event.target.id==='scanBoardBackground')phoneScan.boardBackground=event.target.checked;if(event.target.id==='scanBuildMode'){phoneScan.buildMode=event.target.value;phoneScan.redraw();}if(event.target.id==='scanLanHost')phoneScan.lanHost=event.target.value;});
