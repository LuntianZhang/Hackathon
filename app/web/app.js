import {phoneScan} from './scan-desktop.js';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state = {page:'home',profile:null,catalog:[],categoryDensities:[],meals:[],drafts:[],summary:null,meal:null,selected:null,
  mode:'photo',date:null,file:null,fileUrl:null,fileKind:'photo',uploading:false,modal:null,
  view:null,viewBusy:false,viewDirty:false,imageReady:false,desired:{yaw:0,pitch:0,zoom:1},
  drag:null,suppressUntil:0,poll:null,stream:null,offline:false,installPrompt:null,assistMode:'grounded',foodText:'',
  manualRegions:[],seedName:'',seedTool:'box',seedPicture:null,seedPictureUrl:null,seedDrag:null,
  recognitionParent:null,recognitionChoices:[],viewEpoch:0,captureFlow:'photo',
  historyMealId:null,videoPrepared:null,videoPreparing:false,preparePoll:null,captureEpoch:0,seedSourceUrl:null,
  editPreview:'3d',editViewError:null,nutritionReturnMealId:null,interrupting:false,boardBackground:true};
const zone = () => state.profile?.timezone || 'Asia/Shanghai';
const localDay = (date = new Date()) => new Intl.DateTimeFormat('sv-SE',{timeZone:zone()}).format(date);
const kcal = n => Math.round(n || 0).toLocaleString('zh-CN');
const grams = n => Number(n || 0).toFixed(1);
const energyPending = total => !total.known_items && total.unknown_items > 0;
const energyLabel = total => energyPending(total) ? '待估算' : kcal(total.kcal);
const catalogItem = key => state.catalog.find(food => food.key === key);
const portionLabel = total => total.volume_density_items?(total.reference_items?'估重与参考份量':'体积与密度估算'):total.quantity_pending_items&&!total.reference_items?'待密度或体积':'固定参考份量';
function densityText(item){
 const d=item.density,m=item.mass_estimate;
 if(d?.status==='blocked')return d.message;
 const label=d?.status==='category_default'?'分类默认密度':d?.status==='food_reference'?'食品参考密度':'密度';
 const density=d?.value_g_ml!=null?`${label} ${d.value_g_ml.toFixed(3)} g/mL`:'密度待补充';
 return `${density} · ${m?.weight_grams!=null?`估重 ${m.weight_grams.toFixed(1)} g`:'重量待估算'}`;
}
function densitySourceHTML(entry){
 if(!entry)return '';
 let link='';try{const url=new URL(entry.source_url);if(url.protocol==='https:'&&['www.fao.org','fdc.nal.usda.gov'].includes(url.hostname))link=`<a href="${esc(url.href)}" target="_blank" rel="noopener">${esc(entry.source)}</a>`;}catch{}
 return `${esc(entry.name||'手动密度')} · ${entry.value_g_ml} g/mL<br>${esc(entry.note||entry.message||'')}${link?`<br>${link}`:entry.source?`<br>${esc(entry.source)}`:''}`;
}
function foodNutritionLabel(item){
 const n=item.nutrition;
 if(n)return `${n.portion_method==='volume_density'?'估重':'参考'} ${grams(n.reference_grams)} g · 非称重 · 约 ${kcal(n.kcal)} kcal${item.nutrition_needs_review?' · 在线参考待核对':''}`;
 if(nutritionPending(item))return '<span class="nutrition-spinner" aria-hidden="true"></span>营养数据查询中…';
 if(item.reference_nutrition)return item.mass_estimate?.status==='density_pending'?'已有营养参考，待设置密度后计算本份营养':'已有营养参考，待已标定体积后计算本份营养';
 return esc(item.nutrition_lookup?.message||'营养待补充');
}
function referenceNutritionHTML(item){
 const n=item.reference_nutrition;if(!n||!(n.reference_grams>0))return '';
 return `每 100 g 营养参考：${kcal(n.kcal*100/n.reference_grams)} kcal；本份营养等待估重。${nutritionSourceHTML(n)}`;
}
function densityFood(){
 const selected=catalogItem($('foodCatalog')?.value);if(selected)return selected;
 const name=($('foodName')?.value||'').normalize('NFKC').trim().toLocaleLowerCase();
 return state.catalog.find(food=>[food.name,food.key,food.recognition_label,...(food.aliases||[])].some(value=>String(value||'').normalize('NFKC').trim().toLocaleLowerCase()===name));
}
function densityFieldsHTML(item){
 if(['non_food','packaged_food'].includes(item?.object_kind))return '<p class="estimate-note">非食物和包装外形不用于食物密度估重。</p>';
 const food=item?catalogItem(item.catalog_key||item.density?.catalog_key):densityFood(),refs=food?.density_candidates||[],choice=item?.density_selection;
 const mode=choice?.mode==='custom'?'custom':choice?.mode==='reference'?`ref:${choice.reference_id}`:'clear';
 const automatic=food?.density_default;
 const autoLabel=automatic?.value_g_ml!=null?`自动 · ${automatic.status==='category_default'?'分类默认估值':'食品参考密度'} ${automatic.value_g_ml.toFixed(3)} g/mL`:'自动 · 待匹配食品分类';
 return `<fieldset class="density-fields"><legend>密度与估计重量</legend><label for="densityMode">密度来源</label><select id="densityMode"><option value="clear" ${mode==='clear'?'selected':''}>${esc(autoLabel)}</option>${refs.map(ref=>`<option value="ref:${ref.id}" ${mode===`ref:${ref.id}`?'selected':''}>${esc(ref.name)} · ${ref.value_g_ml} g/mL</option>`).join('')}<option value="custom" ${mode==='custom'?'selected':''}>手动填写密度</option></select><div id="customDensityFields" ${mode==='custom'?'':'hidden'}><label for="densityValue">密度（g/mL）</label><input id="densityValue" type="number" min="0.000001" max="10" step="any" value="${choice?.mode==='custom'?esc(choice.value_g_ml):''}" placeholder="填写有依据的数值"><label for="densityNote">密度依据（可选）</label><input id="densityNote" maxlength="200" value="${choice?.mode==='custom'?esc(choice.note||''):''}" placeholder="如测量方法或数据来源"></div><small id="densityDescription"></small><small id="densityWeightPreview" role="status" aria-live="polite"></small><small>优先使用食品参考密度，缺失时使用分类默认估值；手动修改优先。分类估值包含量杯与堆积状态的差异，体积须对应可食用区域。</small></fieldset>`;
}
function updateDensityPreview(){
 const mode=$('densityMode')?.value;if(!mode)return;
 $('customDensityFields').hidden=mode!=='custom';
 const food=densityFood(),entry=mode==='clear'?food?.density_default:(food?.density_candidates||[]).find(ref=>`ref:${ref.id}`===mode);
 const value=entry?.value_g_ml||(mode==='custom'?Number($('densityValue').value):null);
 $('densityDescription').innerHTML=entry?.value_g_ml!=null?densitySourceHTML(entry):mode==='custom'?'手动填写的密度仅用于本食品，系统不会把它标为已实测。':'尚未匹配食品分类，请选择参考食品或手动填写密度。';
 const item=state.meal?.items.find(item=>item.id===state.modal?.itemId),ml=item?.volume?.volume_ml;
 $('densityWeightPreview').textContent=state.meal?.volume_summary?.status==='calibrated'&&item?.volume?.status==='estimated'&&ml>0?
  value>0&&value<=10?`估重预览：${ml.toFixed(1)} mL × ${value} g/mL = ${(ml*value).toFixed(1)} g`:`当前体积约 ${ml.toFixed(1)} mL，待设置密度。`:'当前没有可用的已标定体积，密度可先保存，体积可用后再估重。';
}
function refreshDensityFields(){
 if(!$('densityFields'))return;$('densityFields').innerHTML=densityFieldsHTML(null);updateDensityPreview();
}
const nutritionPending = item => ['queued','searching'].includes(item.nutrition_lookup?.status);
let nutritionReferenceTimer=null,nutritionReferenceEpoch=0;
function nutritionSourceHTML(value){
 if(!value?.source_url)return '';
 let url;try{url=new URL(value.source_url);}catch{return '';}
 if(url.protocol!=='https:'||!['fdc.nal.usda.gov','world.openfoodfacts.org'].includes(url.hostname))return '';
 return `<br><a href="${esc(url.href)}" target="_blank" rel="noopener">${esc(value.source)}</a> · ${esc(value.reference_name)} · 在线参考，待核对`;
}
function nutritionLookupHTML(value){
 if(!value)return '';
 if(value.status==='found'&&value.nutrition){const n=value.nutrition;return `已查到：参考 ${n.reference_grams} g，约 ${kcal(n.kcal)} kcal；蛋白质 ${grams(n.protein)} g，碳水 ${grams(n.carbs)} g，脂肪 ${grams(n.fat)} g。${nutritionSourceHTML(n)}`;}
 return `${['queued','searching'].includes(value.status)?'<span class="nutrition-spinner" aria-hidden="true"></span>':''}${esc(value.message||'营养待补充')}`;
}
async function queryNutritionReference(retry=false){
 clearTimeout(nutritionReferenceTimer);const epoch=++nutritionReferenceEpoch;
 if(state.modal?.kind!=='food')return;
 const name=$('foodName')?.value.trim(),key=$('foodCatalog')?.value||null,node=$('nutritionSearchStatus');
 if(!node||!name)return;
 const editing=state.meal?.items.find(item=>item.id===state.modal.itemId);
 if(editing&&['non_food','packaged_food'].includes(editing.object_kind)){node.textContent='请先确认可食用食品；非食物对象和包装外形不自动查询营养。';return;}
 if(catalogItem(key)?.per_100g){node.innerHTML='';return;}
 node.innerHTML='<span class="nutrition-spinner" aria-hidden="true"></span>营养数据查询中…';
 try{
  let value=await api('/nutrition/lookups',{method:'POST',body:{name,catalog_key:key,retry}});
  if(epoch!==nutritionReferenceEpoch||state.modal?.kind!=='food')return;
  node.innerHTML=nutritionLookupHTML(value);
  const update=async()=>{
   try{value=await api(`/nutrition/lookups/${value.id}`);
    if(epoch!==nutritionReferenceEpoch||state.modal?.kind!=='food')return;
    node.innerHTML=nutritionLookupHTML(value);
    if(['queued','searching'].includes(value.status))nutritionReferenceTimer=setTimeout(update,1500);
   }catch(error){if(epoch===nutritionReferenceEpoch)node.textContent=error.message;}
  };
  if(['queued','searching'].includes(value.status))nutritionReferenceTimer=setTimeout(update,1500);
 }catch(error){if(epoch===nutritionReferenceEpoch)node.textContent=error.message;}
}
function historyFoodNames(){
 const seen=new Set();
 return state.meals.flatMap(meal=>meal.items).map(item=>String(item.name||'').trim()).filter(name=>{
  const key=name.toLocaleLowerCase('zh-CN');if(!name||seen.has(key))return false;seen.add(key);return true;
 });
}
const historyFoodOptions = () => [...new Set([...historyFoodNames(),...state.catalog.flatMap(food=>[food.name,...(food.aliases||[]).filter(name=>/[\u4e00-\u9fff]/.test(name))])])].map(name=>`<option value="${esc(name)}"></option>`).join('');
function foodCatalogCategories(){
 const groups=new Map(state.catalog.map(food=>[food.category||'other',food.category_label||'其他食物']));
 const order=['fruit','meat','vegetable','staple','dairy','soy','nut','drink','snack','dish','condiment','other'];
 const rank=key=>order.includes(key)?order.indexOf(key):order.length;
 return [...groups.entries()].sort((a,b)=>rank(a[0])-rank(b[0]));
}
function catalogOptions(selected='',query='',category=''){
 const term=query.trim().toLocaleLowerCase();
 const foods=state.catalog.filter(food=>(!category||food.category===category)&&(!term||[food.name,food.key,food.recognition_label,...(food.aliases||[])].some(value=>String(value||'').toLocaleLowerCase().includes(term))));
 const option=food=>`<option value="${food.key}" ${food.key===selected?'selected':''}>${esc(food.name)}${food.per_100g?'':' · 营养待补充'}</option>`;
 const current=state.catalog.find(food=>food.key===selected);
 return '<option value="">自定义名称 · 营养待补充</option>'+(current&&!foods.includes(current)?`<optgroup label="当前选择">${option(current)}</optgroup>`:'')+foodCatalogCategories().map(([key,name])=>{
  const group=foods.filter(food=>(food.category||'other')===key);return group.length?`<optgroup label="${esc(name)} · ${group.length} 项">${group.map(option).join('')}</optgroup>`:'';
 }).join('')+(!foods.length?'<option disabled>没有匹配食物，可填写自定义名称</option>':'');
}
function catalogReferenceHTML(key){
 const food=catalogItem(key);if(!food)return `已收录 ${state.catalog.length} 项食物，可按分类或名称、别名搜索。`;
 const value=food.per_100g;
 if(!value)return `${esc(food.name)}：本机暂缺营养数据，将自动联网查询；查到后显示在线参考。`;
 const item=state.meal?.items.find(item=>item.id===state.modal?.itemId);
 const portion=item?.instance_id&&state.meal?.three_d_available?'本份按体积与密度估重，见下方预览。':`本次使用固定参考 ${food.reference_grams} g，未称重。`;
 return `${esc(food.name)} · 每 100 g 可食部：${value.kcal} kcal，蛋白质 ${value.protein} g，碳水 ${value.carbs} g，脂肪 ${value.fat} g。<br>${portion}数据：USDA SR28 · ${esc(food.ndb_number)}。`;
}
function filterFoodCatalog(){
 const select=$('foodCatalog');if(!select)return;
 const selected=select.value;select.innerHTML=catalogOptions(selected,$('foodSearch')?.value||'',$('foodCategory')?.value||'');
}

const chosen = () => state.meal?.items.find(item => item.id === state.selected);
function timeLabel(value) {return new Intl.DateTimeFormat('zh-CN',{timeZone:zone(),hour:'2-digit',minute:'2-digit'}).format(new Date(value));}
function dateLabel(value) {return new Intl.DateTimeFormat('zh-CN',{timeZone:zone(),month:'long',day:'numeric',weekday:'short'}).format(new Date(value));}
const mealType = {breakfast:'早餐',lunch:'午餐',dinner:'晚餐',snack:'加餐'};

async function api(path, options={}) {
  const init = {...options,headers:{...(options.headers || {})}};
  if (options.body && !(options.body instanceof FormData)) {
    init.headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(options.body);
  }
  let response;
  try {response = await fetch(`/api${path}`,init);} catch (_) {throw new Error('未连接到本机服务，连接恢复后可以继续');}
  let value;
  try {value = await response.json();} catch (_) {throw new Error('服务暂时不可用，请稍后重试');}
  if (!response.ok) {const error = new Error(value.detail || '操作未完成，请重试');error.status=response.status;throw error;}
  return value;
}
const iconPaths = {
 home:'<path d="M3 10.5 12 3l9 7.5V21h-7v-7h-4v7H3z"/>',
 diary:'<rect x="4" y="3" width="16" height="18" rx="3"/><path d="M8 3v18M11 9h6M11 13h6"/>',
 camera:'<path d="M3 7h4l2-3h6l2 3h4v13H3z"/><circle cx="12" cy="13" r="4"/>',
 person:'<circle cx="12" cy="8" r="4"/><path d="M4 21c1-7 15-7 16 0"/>',
 arrow:'<path d="M5 12h14m-6-6 6 6-6 6"/>',back:'<path d="m15 4-8 8 8 8"/>',
 image:'<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="9" r="2"/><path d="m3 18 6-6 4 4 3-3 5 5"/>',
 leaf:'<path d="M4 17c0-7 5-11 15-12-1 10-5 14-12 14M7 19c3-5 6-8 11-11"/>',
 info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v1"/>',check:'<path d="m4 12 5 5L20 6"/>'};
const icon = name => `<svg viewBox="0 0 24 24" aria-hidden="true">${iconPaths[name] || ''}</svg>`;
function illustration() {
 return `<svg viewBox="0 0 360 270" aria-label="餐盘插画" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="181" cy="225" rx="138" ry="20" fill="#67835e" opacity=".13"/><ellipse cx="180" cy="176" rx="152" ry="70" fill="#d9dfce"/>
  <ellipse cx="180" cy="160" rx="152" ry="69" fill="#fffdf2" stroke="#eef0df" stroke-width="4"/><ellipse cx="180" cy="160" rx="130" ry="53" fill="#f5f1df"/>
  <path d="M55 148c-7-40 60-80 112-47 28 18 28 52 2 71-35 25-99 11-114-24" fill="#f7e7bd" stroke="#dbc89c" stroke-width="2"/>
  ${Array.from({length:24},(_,i)=>`<ellipse cx="${75+i%6*16}" cy="${121+Math.floor(i/6)*12}" rx="5" ry="2.3" fill="${i%2?'#fdf4d9':'#e5d6af'}" transform="rotate(${i*23%150} ${75+i%6*16} ${121+Math.floor(i/6)*12})"/>`).join('')}
  <path d="M184 125c-10-26 30-53 67-20 36 29 21 67-13 69-39 3-70-30-54-49" fill="#c99665" stroke="#b77b53" stroke-width="3"/>
  <path d="M208 112c19-13 39 8 37 27" fill="none" stroke="#ebc48a" stroke-width="10" stroke-linecap="round"/>
  ${[[276,168],[290,147],[265,148],[304,171],[285,185]].map(([x,y],i)=>`<path d="m${x} ${y+8} -4 19 8 0-2-19" stroke="#83a567" stroke-width="5"/><circle cx="${x}" cy="${y}" r="15" fill="${i%2?'#6d955c':'#87a66b'}"/><circle cx="${x-4}" cy="${y-5}" r="7" fill="#96b377"/>`).join('')}</svg>`;
}
function macro(name,value,color) {return `<div class="macro"><div class="label">${name}</div><div class="value">${value===null?'—':grams(value)} <span>g</span></div><div class="progress"><span style="width:100%;background:${color}"></span></div></div>`;}
function macros(total) {const unavailable=total.known_items===0&&total.unknown_items>0;return `<div class="macros">${macro('蛋白质',unavailable?null:total.protein,'#97aa6d')}${macro('碳水',unavailable?null:total.carbs,'#d4a876')}${macro('脂肪',unavailable?null:total.fat,'#8ab2aa')}</div>`;}
function nutritionCard(total) {
 return `<section class="surface nutrition" id="mealNutrition" tabindex="-1"><div class="nutrition-main"><div><div class="eyebrow">ESTIMATED ENERGY</div><h3>${total.known_items?'约 '+kcal(total.kcal):'待估算'} <small>${total.known_items?'kcal':''}</small></h3></div><span class="nutri-tag">${portionLabel(total)}</span></div>${macros(total)}
 ${state.meal?.items.some(nutritionPending)?`<p class="estimate-note" role="status"><span class="nutrition-spinner" aria-hidden="true"></span>${state.meal.items.filter(nutritionPending).length} 项营养数据查询中，查到后自动更新。</p>`:''}
 ${total.quantity_pending_items?`<p class="estimate-note">${total.quantity_pending_items} 项已有营养参考，等待合适的密度或已标定体积，尚未计入本餐营养。</p>`:''}
 ${total.unknown_items?`<p class="estimate-note">另有 ${total.unknown_items} 项营养尚未完整，尚未计入汇总。</p>`:''}</section>`;
}
function mealLine(meal) {
 const symbols = meal.items.map(item => catalogItem(item.catalog_key)?.emoji || '🍽️').slice(0,3).join('');
 return `<button class="meal-line" data-meal="${meal.id}"><div class="meal-art green">${meal.photo_url?`<img src="${meal.photo_url}" alt="${esc(meal.title)}的照片" loading="lazy">`:symbols || '🍽️'}</div>
 <div class="meal-copy"><strong>${esc(meal.title)}</strong><span>${mealType[meal.meal_type]} · ${timeLabel(meal.occurred_at)} · ${meal.items.length} 项食物</span></div>
 <div class="meal-kcal">${energyPending(meal.totals)?'待估算':'约 '+kcal(meal.totals.kcal)}<small>${energyPending(meal.totals)?'营养待补充':'kcal'+(meal.totals.unknown_items?' · 未完整':'')}</small></div></button>`;
}
function nav() {
 if (state.page==='onboard') return '';
 return [['home','home','首页'],['diary','diary','日记'],['scan','camera','记录'],['profile','person','我的']].map(([page,name,label]) =>
 `<button class="tab ${state.page===page || page==='home'&&['result','plate'].includes(state.page)?'active':''} ${page==='scan'?'capture':''}" data-page="${page}" aria-label="${label}">
 ${page==='scan'?`<span class="cam-circle">${icon(name)}</span>`:icon(name)}<span>${label}</span></button>`).join('');
}
function home() {
 const sum = state.summary || {count:0,totals:{},goal_kcal:null};
 const percent = sum.goal_kcal?Math.min(100,sum.totals.kcal/sum.goal_kcal*100):0;
 return `<div class="row topline"><div><div class="eyebrow">EAT WELL, FEEL GOOD</div><h1 class="screen-heading">你好，${esc(state.profile?.display_name || '记录生活')}。</h1><span class="tiny muted">${dateLabel(new Date())}</span></div><button class="avatar" data-page="profile" aria-label="个人中心">${esc((state.profile?.display_name || '盘')[0])}</button></div>
 <section class="hero"><div class="hero-content"><span class="hint">把日常饮食，轻松记下来</span><h2>今天吃了什么，<br>拍一张就知道。</h2><button class="primary-small" data-page="scan">${icon('camera')}记录这一餐${icon('arrow')}</button></div><div class="hero-platter">${illustration()}</div></section>
 <div class="row"><h2 class="section-head">今日营养概览</h2><span class="tiny muted">${sum.count} 条已保存</span></div>
 <section class="surface daily"><div class="row"><div><div class="daily-title">已记录 · 估计热量</div><div class="daily-number">${energyLabel(sum.totals)} <small>${energyPending(sum.totals)?'':'kcal'}</small></div><span class="tiny muted">${sum.goal_kcal?`参考值 ${kcal(sum.goal_kcal)} kcal`:'按照已保存的餐食累计'}</span></div>
 <div class="donut" style="--pct:${percent}%"><div class="donut-core"><strong>${sum.goal_kcal&&!energyPending(sum.totals)?Math.round(percent)+'%':sum.count}</strong><span>${sum.goal_kcal&&!energyPending(sum.totals)?'参考进度':'餐食记录'}</span></div></div></div><div class="divider"></div>${macros(sum.totals)}
 ${sum.totals.unknown_items?`<p class="estimate-note">${sum.totals.unknown_items} 项食物营养待补充。</p>`:''}</section>
 <div class="row"><h2 class="section-head">最近餐食</h2><button class="text-btn" data-page="diary">查看日记 →</button></div>
 ${state.meals.length?state.meals.slice(0,3).map(mealLine).join(''):'<div class="empty">还没有保存的餐食。<br>拍一张照片，开始自己的饮食日记。</div>'}
 ${state.drafts.length?`<h2 class="section-head">继续未保存的餐食</h2>${state.drafts.slice(0,3).map(meal=>`<button class="recent-draft" data-meal="${meal.id}">${esc(meal.title)} · ${esc(meal.analysis.stage)} →</button>`).join('')}`:''}`;
}
function onboard() {
 return `<div class="onboarding"><div class="eyebrow">WELCOME TO PANSENSE</div><h1 class="screen-heading">吃得自在，<br>记得轻松。</h1><div class="onboard-art">${illustration()}</div>
 <div class="onboard-step"><span>1</span><p><strong>拍下这一餐</strong>拍照或导入，了解食物的参考营养。</p></div><div class="onboard-step"><span>2</span><p><strong>看看，再确认</strong>名称可以纠正，三维暂不可用也能记录。</p></div><div class="onboard-step"><span>3</span><p><strong>把记录留在本机</strong>随时查看、修改、导出或删除自己的餐食。</p></div>
 <p class="estimate-note">份量与营养均为估算，无法仅凭照片判断全部油脂、酱料和隐藏配料。</p>
 <button class="button primary" data-action="onboard-goal">选择每日热量参考值</button><button class="text-btn" style="width:100%;margin-top:8px" data-action="onboard-skip">先记录，跳过设置</button></div>`;
}
function scan() {
 if(state.captureFlow==='mobile')return phoneScan.html()+`<button class="text-btn" data-capture-flow="photo">← 返回照片 / 视频记录</button>`;
 const edit=state.captureFlow==='edit',video=state.captureFlow==='video';
 const navigation=`<div class="capture-modes" role="group" aria-label="记录方式">${[['photo','照片记录'],['video','三维记录'],['mobile','手机扫描'],['edit','调整已有餐盘']].map(([value,label])=>`<button data-capture-flow="${value}" class="${state.captureFlow===value?'active':''}" aria-pressed="${state.captureFlow===value}" ${state.uploading?'disabled':''}>${label}</button>`).join('')}</div>`;
 const heading=`<div class="eyebrow">${edit?'EDIT THIS PLATE':'RECORD A MEAL'}</div><h1 class="screen-heading">${edit?'调整已有餐盘':'记录这一餐。'}</h1><p class="muted tiny">${edit?'从历史记录选择餐盘，更新食物名称与区域。':video?'环绕视频，生成可观察的三维餐盘。':'拍下完整餐盘，快速识别并核对。'}</p>${navigation}`;
 if(edit&&!state.recognitionParent)return heading+historyPlates();
 const preview = state.stream?'<video id="cameraPreview" autoplay muted playsinline></video>':(state.fileUrl||state.seedSourceUrl)?
  (state.assistMode==='manual'&&(!video||state.seedSourceUrl)?'<canvas id="mealSeedCanvas" aria-label="食物标注参考帧，点击或拖动框选"></canvas>':state.seedSourceUrl?`<img class="preview-photo" src="${state.seedSourceUrl}" alt="三维模型参考帧">`:state.fileKind==='video'?`<video src="${state.fileUrl}" controls muted playsinline></video>`:`<img class="preview-photo" src="${state.fileUrl}" alt="待分析的食物照片">`):
  `<div class="camera-placeholder">${icon('camera')}<p>拍下完整餐盘，<br>让食物都出现在画面里。</p></div>`;
 const active=!!state.file||!!state.stream;
 const parent=state.recognitionParent?`<div class="capture-parent">正在调整：${esc(state.recognitionParent.title)}<small>${state.recognitionParent.threeD?'沿用原模型，更新识别版本':'基于原照片，更新识别版本'}</small><button class="text-btn" data-action="history-plates">更换历史餐盘 →</button><button class="button ghost" data-action="existing-nutrition" ${state.uploading?'disabled':''}>不更新，直接查看这一餐营养</button></div>`:'';
 const reference=`<div class="stage camera-stage ${state.assistMode==='manual'&&(!video||state.seedSourceUrl)?'seed-stage':''}">${preview}</div>`;
 const edit3d=edit&&state.meal?.three_d_available;
 const viewer=edit3d?`<div class="segmented edit-preview-tabs" role="group" aria-label="餐盘查看方式"><button data-edit-preview="3d" class="${state.editPreview==='3d'?'selected':''}" aria-pressed="${state.editPreview==='3d'}">三维拖动查看</button><button data-edit-preview="reference" class="${state.editPreview==='reference'?'selected':''}" aria-pressed="${state.editPreview==='reference'}">参考图标注</button></div>${state.editPreview==='3d'?editablePlateViewer():reference}${state.editViewError?`<p class="record-note">${esc(state.editViewError)}<button class="text-btn" data-edit-preview="3d">重试三维预览 →</button></p>`:''}`:reference;
 return heading+parent+`<div class="capture-progress">${edit?'已选餐盘 → 调整识别 → 核对并保存':active?(video?'视频已选择 → 选择分割方式 → 生成三维餐盘':'照片已选择 → 选择分割方式 → 核对清单'):(video?'选择环绕视频 → 选择分割方式 → 生成三维餐盘':'拍照 / 相册 → 选择分割方式 → 核对并保存')}</div>
 ${viewer}<div class="hint-card">${icon('info')}<div>${video?boardCaptureTips():edit?(state.recognitionParent.threeD?'三维视图可拖动、缩放和点击查看现有食物。点框标注请切换到固定参考图，更新后才应用新的名称与区域。':'在原照片上重新标记和识别食物，完成后核对清单。'):'光线充足、食物分开摆放会更容易识别。照片提供二维识别结果。'}</div></div>
 ${captureAssistance()}
 <div class="buttons">${state.stream?`<button class="button primary" data-action="capture">${icon('camera')}拍下这一餐</button>`:state.file?
 `<button class="button primary" data-action="analyze" ${state.uploading||state.videoPreparing||video&&state.assistMode==='manual'&&!state.videoPrepared?.video_reference_ready?'disabled':''}>${icon('arrow')}${state.uploading?'正在接收文件…':state.assistMode==='none'?(edit?'直接调整食物清单':video?'仅生成三维外观':'继续填写食物清单'):edit?(state.recognitionParent.threeD?'分割并更新此三维餐盘':'重新识别照片'):video?'生成三维餐盘':'开始识别'}</button>${edit?'':`<button class="text-btn" data-action="reselect-capture" ${state.uploading?'disabled':''}>← 重新选择素材</button>`}`:
 video?`<button class="button primary" data-action="upload-video">${icon('image')}拍摄 / 导入环绕视频</button>`:`<button class="button primary" data-action="camera">${icon('camera')}拍一张照片</button><button class="button ghost" data-action="upload-photo">${icon('image')}从相册选择</button>`}</div>
 ${!active&&!edit?'<button class="text-btn" data-action="manual">直接手工记录</button>':''}
 <p class="estimate-note">${edit?'确认保存后生成新的餐食记录，原餐食保留。':video?'视频限 3 秒至 3 分钟、240 MB，三维处理需要数分钟。':'照片最多 20 MB。'}${state.assistMode==='llm'||video&&state.boardBackground?'关键帧会发送给当前 Agent 使用的模型服务，其余计算在本机进行。':'内容仅发送到本机分析服务。'}</p>
 <details class="capture-extras"><summary>示例与识别对比工具</summary><div class="semantic-entry"><button class="text-btn" data-action="semantic-reference">参考照片识别</button><button class="text-btn" data-action="recognition-results">导入已有识别结果</button><button class="text-btn" data-action="demo">已有三维示例</button><a class="text-btn" href="/recognition.html">识别对比页 →</a></div></details>`;
}
function historyPlates(){
 const meals=[...state.meals,...state.drafts].sort((a,b)=>new Date(b.occurred_at)-new Date(a.occurred_at));
 const selected=meals.find(m=>m.id===state.historyMealId);
 const ready=selected&&state.meal?.id===selected.id;
 const preview=ready&&state.meal.three_d_available?`<h2 class="section-head">${esc(state.meal.title)}</h2>${state.editViewError?`<p class="record-note">${esc(state.editViewError)}<button class="text-btn" data-edit-preview="3d">重试三维预览 →</button></p>`:editablePlateViewer()}`:'';
 return `<h2 class="section-head">选择历史餐盘 <small class="muted tiny">${meals.length} 条记录</small></h2><p class="estimate-note">已保存的餐食与未保存草稿都在这里。选择餐盘后可调整食物，或查看原有营养。</p><div class="capture-history" role="group" aria-label="历史餐盘">${meals.length?meals.map(m=>`<button class="history-plate ${selected?.id===m.id?'active':''}" data-history-meal="${m.id}" aria-pressed="${selected?.id===m.id}">${m.photo_url?`<img src="${m.photo_url}" alt="" loading="lazy">`:'<span class="history-placeholder" aria-hidden="true">🍽️</span>'}<span><strong>${esc(m.title)}</strong><small>${dateLabel(m.occurred_at)} ${timeLabel(m.occurred_at)} · ${m.saved_at?'已保存':'草稿'} · ${m.three_d_available?'三维餐盘':m.photo_url?'照片餐盘':'手工记录'}</small><small>${esc(m.items.map(i=>i.name).join('、')||'等待识别')}</small><small>${m.can_refine_semantics&&m.photo_url?'可重新分割与识别':'可调整食物名称与清单'}</small></span><b aria-hidden="true">${selected?.id===m.id?'✓':'○'}</b></button>`).join(''):'<div class="empty">还没有历史餐盘。<br>拍照、导入视频或手工记录后，会自动出现在这里。</div>'}</div>${preview}${ready&&!state.meal.can_refine_semantics?'<p class="estimate-note">这份记录暂时缺少重新分割所需的原始素材，仍可调整食物名称与清单。</p>':''}<button class="button ghost" data-action="existing-nutrition" ${ready?'':'disabled'}>不更新，直接查看这一餐营养</button><button class="button primary" data-action="edit-history" ${ready?'':'disabled'}>${ready&&!(state.meal.can_refine_semantics&&state.meal.photo_url)?'调整食物名称与清单':'调整选中的餐盘'}</button><button class="text-btn" data-page="diary">查看全部餐食日记 →</button>`;
}
function editablePlateViewer(){return mealImage(true)+`<button class="text-btn" data-action="reset-view">恢复拍摄视角</button>`+selectedCard();}
function captureMealActive(){return state.page==='scan'&&state.captureFlow==='edit'&&state.meal?.id===(state.recognitionParent?.id||state.historyMealId);}
function viewActive(){return state.meal?.three_d_available&&(state.page==='plate'||state.page==='result'&&state.mode==='3d'||captureMealActive()&&state.editPreview==='3d');}
function captureAssistance(){
 const video=state.captureFlow==='video',edit=state.captureFlow==='edit';
 const options=[['list','食物清单分割','先自动分割，再用清单辅助判断名称。'],['grounded','名称定位分割','先按输入名称寻找区域，再进行分割。'],['manual','点击 / 框选分割','在参考照片中逐个指定食物名称与区域。'],['llm','大模型识别','只识别白色定位板内的物体；有多视角素材时联合识别并追踪。'],['none','不使用语义分割',video?'只重建三维外观，不识别食物区域；食物清单可手工补充。':edit?'保留原餐盘，直接调整食物名称与清单，不重新分割。':'保留原始照片，手工填写食物清单，不运行分割模型。']];
 return `${video?boardBackgroundControl():''}<section class="surface assist-input"><label>食物语义识别 <small>四种识别方式，或不使用语义分割</small></label><div class="capture-assist-modes" role="group" aria-label="食物分割方式">${options.map(([id,label])=>`<button data-capture-assist="${id}" aria-pressed="${state.assistMode===id}" class="${state.assistMode===id?'active':''}" ${state.uploading?'disabled':''}>${label}</button>`).join('')}</div><p class="estimate-note">${options.find(([id])=>id===state.assistMode)?.[2]||'请选择食物分割方式。'}</p>
 ${['list','grounded'].includes(state.assistMode)?`<label for="foodHints">${state.assistMode==='list'?'可能有哪些食物？':'希望定位哪些食物？'}</label><textarea id="foodHints" rows="2" placeholder="苹果、香蕉、面包">${esc(state.foodText)}</textarea><p class="estimate-note">自定义名称可写成“中文名称|English name”。</p>`:''}
 ${video&&state.assistMode==='manual'&&!state.videoPrepared?.video_reference_ready?state.file?`<p class="estimate-note">先关联视频相机，取得与三维融合一致的参考帧，再点击或框选食物。此步骤需要一些时间。</p><button class="button ghost" data-action="prepare-video" ${state.uploading||state.videoPreparing?'disabled':''}>${state.videoPreparing?'正在准备参考帧…':state.videoPrepared?'重新准备参考帧':'准备标注参考帧'}</button>`:'<p class="estimate-note">请先选择环绕视频，再准备标注参考帧。</p>':''}
 ${state.videoPreparing?`<p class="record-note" role="status">${esc(state.videoPreparedStatus||'正在准备视频参考帧')}</p><button class="button ghost" data-action="interrupt-prepare" ${state.interrupting?'disabled':''}>${state.interrupting?'正在中断…':'中断参考帧准备'}</button>`:''}
 ${state.assistMode==='manual'&&(!video||state.videoPrepared?.video_reference_ready)?!state.file?'<p class="estimate-note">请先选择照片，再点击或框选食物。</p>':edit&&state.meal?.three_d_available&&state.editPreview==='3d'?'<p class="estimate-note">请在固定参考图中标记食物，已有标记会保留。</p><button class="button ghost" data-edit-preview="reference">切换到参考图标注</button>':manualSeedControls():''}
 ${edit&&state.assistMode!=='none'?`<p class="estimate-note">${state.recognitionParent.threeD?'本次分割会传播到其他视角，并更新当前模型的语义版本，无需重新训练几何。':'本次识别生成新草稿，确认后保存。'}</p>`:''}</section>`;
}
function boardCaptureTips(){
 return (state.boardBackground?'可固定相机，让定位板和物体一起转动；物体不能相对板滑动，停稳后再拍。也可固定物体、环绕拍摄。':'固定物体和定位板，缓慢环绕拍摄。')+'定位板按原尺寸打印，保持至少 3 个标记可见，以估算体积。';
}
function boardBackgroundControl(){
 return `<section class="surface"><label class="board-background-control"><input id="boardBackground" type="checkbox" ${state.boardBackground?'checked':''} ${state.uploading||state.videoPrepared?'disabled':''}>自动跟随定位板排除背景</label><p class="estimate-note">转动定位板时建议开启。定位板和物体一起转动，保留完整物体与标记；至少露出 3 个标记。会将关键帧发送给当前 Agent 使用的模型服务来识别前景，再在本机跟踪，需要额外时间。</p></section>`;
}
function backgroundPanel(){
 const value=state.meal?.background_summary;if(!value?.enabled)return '';
 return `<section class="surface semantic-summary"><strong>已跟随定位板排除背景</strong><p>${value.processed_frames} 个视角保留定位板与完整物体${value.excluded_frames?`，另有 ${value.excluded_frames} 帧因定位板不清晰被排除`:''}。${value.status==='applied_to_sfm_and_training'?'已应用于相机匹配和三维重建。':''}</p>${state.meal.background_preview_url?`<details><summary>查看背景处理预览</summary><img class="background-preview" src="${state.meal.background_preview_url}" alt="定位板及物体保留区域预览" loading="lazy"></details>`:''}</section>`;
}
function manualSeedControls(){
 return `<div class="seed-controls"><label for="seedName">当前食物名称</label><input id="seedName" maxlength="80" value="${esc(state.seedName)}" placeholder="输入名称或选择历史食物" list="seedFoodNames" autocomplete="off"><datalist id="seedFoodNames">${historyFoodOptions()}</datalist><p class="estimate-note">${historyFoodNames().length?'候选来自已保存的历史餐食，也可以直接输入新名称。':'还没有历史食物，可以直接输入新名称。'}</p>
 <div class="seed-tools"><label><input type="radio" name="seedTool" value="point" ${state.seedTool==='point'?'checked':''}>点击定位</label><label><input type="radio" name="seedTool" value="box" ${state.seedTool==='box'?'checked':''}>拖动框选</label></div>
 <p class="estimate-note">先填写名称，再在上方照片中标记食物。每块食物会获得独立 ID，名称按你的填写保留。</p><div id="seedList">${seedListHTML()}</div></div>`;
}
function seedListHTML(){return state.manualRegions.length?state.manualRegions.map((r,i)=>`<div class="seed-item"><span>食物 ${r.instance_id} · ${esc(r.name)}</span><button class="text-btn" data-action="remove-seed" data-index="${i}">移除</button></div>`).join(''):'<p class="estimate-note">还没有标记食物。</p>';}
function paintSeeds(){
 const source=state.seedSourceUrl||state.fileUrl;
 const canvas=$('mealSeedCanvas');if(!canvas||!source)return;
 if(state.seedPictureUrl!==source){const url=source,picture=new Image();state.seedPictureUrl=url;state.seedPicture=picture;picture.onload=()=>{if((state.seedSourceUrl||state.fileUrl)===url)paintSeeds();};picture.src=url;}
 const picture=state.seedPicture;if(!picture?.complete||!picture.naturalWidth)return;
 const scale=Math.min(1,960/picture.naturalWidth,960/picture.naturalHeight);canvas.width=Math.round(picture.naturalWidth*scale);canvas.height=Math.round(picture.naturalHeight*scale);
 const ctx=canvas.getContext('2d');ctx.drawImage(picture,0,0,canvas.width,canvas.height);ctx.lineWidth=3;ctx.font=`${Math.max(14,canvas.width/40)}px sans-serif`;
 for(const region of state.manualRegions){ctx.strokeStyle='#13a890';ctx.fillStyle='#32c6a530';let x,y;if(region.point){[x,y]=region.point;ctx.beginPath();ctx.arc(x*canvas.width,y*canvas.height,8,0,Math.PI*2);ctx.fill();ctx.stroke();}else{[x,y]=region.box;const w=(region.box[2]-x)*canvas.width,h=(region.box[3]-y)*canvas.height;ctx.fillRect(x*canvas.width,y*canvas.height,w,h);ctx.strokeRect(x*canvas.width,y*canvas.height,w,h);}ctx.fillStyle='#064b3a';ctx.fillText(`${region.instance_id} ${region.name}`,Math.min(x*canvas.width,canvas.width-70),Math.max(20,y*canvas.height-8));}
 if(state.seedDrag?.end){const a=state.seedDrag.start,b=state.seedDrag.end;ctx.strokeStyle='#13a890';ctx.strokeRect(a[0]*canvas.width,a[1]*canvas.height,(b[0]-a[0])*canvas.width,(b[1]-a[1])*canvas.height);}
}
function refreshSeeds(){if($('seedList'))$('seedList').innerHTML=seedListHTML();paintSeeds();}
function seedPoint(event){const box=$('mealSeedCanvas').getBoundingClientRect();return [Math.max(0,Math.min(1,(event.clientX-box.left)/box.width)),Math.max(0,Math.min(1,(event.clientY-box.top)/box.height))];}
function processing(meal) {
 const threeD=meal.three_d_analysis;
 const analysis = ['queued','running'].includes(meal.analysis.status)?meal.analysis:threeD&&['queued','running','failed','cancelled'].includes(threeD.status)?threeD:meal.analysis;
 if (['queued','running'].includes(analysis.status)) return `<div class="processing" role="status"><strong>${esc(analysis.stage)}</strong>已有清单可以核对，分析结束后自动更新三维状态。<div class="progress"><span style="width:${analysis.progress}%;background:#9db474"></span></div><div class="processing-actions"><span class="tiny">${Math.round(analysis.progress)}%</span><button class="text-btn" data-action="interrupt" ${state.interrupting?'disabled':''}>${state.interrupting?'正在中断…':'中断识别与重建'}</button></div></div>`;
 if (analysis.status==='cancelled') return `<div class="processing" role="status"><strong>${esc(analysis.stage)}</strong>原始素材和已完成的食物清单已保留，可稍后重新运行。<br><button class="text-btn" data-action="${analysis===threeD?'semantic-3d':'retry'}">${analysis===threeD?'重新生成三维关联':'重新分析'}</button></div>`;
 if (analysis.error) return `<div class="hint-card warning">${icon('info')}<div>${esc(analysis.error)}<br><button class="text-btn" data-action="${analysis===threeD?'semantic-3d':'retry'}">${analysis===threeD?'重试三维关联':'重新分析'}</button></div></div>`;
 return '';
}
function foodVolume(item) {
 const value=item.volume?.volume_ml;
 if(typeof value==='number'&&Number.isFinite(value)&&value>=0)return {value:`约 ${value.toFixed(1)} mL`,label:'外形体积',available:true};
 if(item.volume?.status==='estimated')return {value:'体积待标定',label:'尚无真实尺度',available:false};
 if(item.volume)return {value:'体积暂不可估',label:'三维采样不足',available:false};
 const analysis=state.meal.three_d_analysis||state.meal.analysis;
 if(['queued','running'].includes(analysis?.status))return {value:'体积计算中',label:'分析完成后更新',available:false};
 return {value:'体积待估算',label:state.meal.three_d_available&&!item.instance_id?'尚未关联三维区域':'暂无体积结果',available:false};
}
function foodList() {
 return state.meal.items.map(item=>{const volume=foodVolume(item);return `<button class="food-row ${state.selected===item.id?'selected':''}" data-food="${item.id}"><span class="food-dot" style="background:#edf1de">${catalogItem(item.catalog_key)?.emoji || '🍽️'}</span>
 <span class="food-text"><strong>${esc(item.name)} ${item.object_kind==='non_food'?'<span class="instance-badge">非食物对象</span>':item.object_kind==='packaged_food'?'<span class="instance-badge">包装外形</span>':''}${item.instance_id?`<span class="instance-badge">ID ${item.instance_id}</span>`:''}</strong><span class="small" ${nutritionPending(item)?'role="status"':''}>${foodNutritionLabel(item)}${item.needs_review?' · 名称待核对':''}</span><span class="small">${esc(densityText(item))}</span></span><span class="food-volume ${volume.available?'':'pending'}"><strong>${esc(volume.value)}</strong><span class="small">${esc(volume.label)}</span></span></button>
 <div class="food-actions"><button data-edit="${item.id}">修改</button>${!item.nutrition&&!item.reference_nutrition&&!nutritionPending(item)&&!['non_food','packaged_food'].includes(item.object_kind)?`<button data-retry-nutrition="${item.id}">重试营养查询</button>`:''}<button class="remove" data-remove-food="${item.id}">移除</button></div>`;}).join('');
}
function mealImage(threeD=false) {
 if (threeD) {
  if (!state.meal.three_d_available) return `<div class="empty">${esc(state.meal.three_d_reason || '三维暂时不可用')}<br>原图和食物清单仍可使用。</div>`;
  return `<div class="plate-shell ${state.drag?.moved?'dragging':''}"><span class="plate-tip">拖动旋转 · 滚轮缩放 · 轻点食物</span><span class="view-status" id="viewStatus">${state.viewBusy?'更新视角…':''}</span>
   ${state.view?`<img id="plateImage" src="${state.view.image_url}" alt="真实三维餐盘，高斯渲染" draggable="false">`:'<div class="empty">正在准备三维餐盘…</div>'}</div>`;
 }
 if (!state.meal.photo_url) return '<div class="empty">这餐没有照片。<br>食物清单仍可以正常修改和保存。</div>';
 const item = chosen();
 return `<div class="stage result-stage"><img id="mealPhoto" src="${state.mode==='semantic'&&state.meal.semantic_overlay_url?state.meal.semantic_overlay_url:state.meal.photo_url}" alt="${state.mode==='semantic'?'食物语义分割结果':state.meal.background_summary?.enabled?'排除背景后的参考照片':'这餐的原始照片'}">${item?.region_url?`<img class="photo-mask" src="${item.region_url}" alt="选中食物的视觉区域">`:''}</div>`;
}
function selectedCard() {
 const item = chosen();
 if (!item) return '';
 return `<div class="chosen-note"><strong>${esc(item.name)}</strong>${item.instance_id?` · 对象 ${item.instance_id}`:''}<br>${item.nutrition?
  `约 ${kcal(item.nutrition.kcal)} kcal · 蛋白质 ${grams(item.nutrition.protein)} g<br>碳水 ${grams(item.nutrition.carbs)} g · 脂肪 ${grams(item.nutrition.fat)} g${nutritionSourceHTML(item.nutrition)}`:item.object_kind==='non_food'?'非食物测试对象，不计入营养。':item.object_kind==='packaged_food'?'当前测量包装外形，无法据此确定包内食物体积或重量。':item.reference_nutrition?referenceNutritionHTML(item):item.nutrition_lookup?nutritionLookupHTML(item.nutrition_lookup):'尚无匹配营养数据，添加或更正食品后将自动联网查询。'}
  <br><strong>${esc(densityText(item))}</strong>${item.density?.value_g_ml!=null?`<br>${densitySourceHTML(item.density)}`:item.density?.message?`<br>${esc(item.density.message)}`:''}
  ${state.mode!=='3d'&&!item.region_available?'<br>这项食物暂未定位到照片区域。':''}
  ${(state.mode==='3d'||state.page==='plate')&&!item.instance_id?'<br>手工补充项尚未关联三维区域，已保留在清单中。':''}
  ${item.volume?`<br><strong>${item.volume.volume_ml!=null?`体积约 ${item.volume.volume_ml.toFixed(1)} mL${state.meal.volume_summary?.scale_source==='pansense_aruco_board_multiview'?'（定位板估算）':''}`:item.volume.status==='estimated'?'体积暂未完成真实尺度标定':'三维采样不足，暂不能估算体积'}</strong>${item.volume.max_span_cm?` · 最大跨度 ${item.volume.max_span_cm.toFixed(1)} cm`:''}<br>外形包围体积为粗估；密度匹配食物状态后才用于估重，仍受凹陷和遮挡影响。`:''}</div>`;
}
function volumePanel(){
 const value=state.meal.volume_summary;if(!state.meal.three_d_available||!value||value.status==='unavailable')return '';
 const board=value.scale_source==='pansense_aruco_board_multiview';
 const items=state.meal.items.filter(item=>item.volume?.volume_ml!=null);
 return `<section class="surface semantic-summary"><strong>${board?'定位板估算体积':'体积估计'}</strong><p>${value.status==='calibrated'?(board?`已用${esc(value.board_name||'定位板')}估算，${value.reference_views} 个视角、${value.reference_markers} 个标记通过校验。`:'已用实测尺寸标定。'):'暂未获得可靠定位板尺度。请按原尺寸打印定位板，保持平整静止，并拍到多个标记及不同视角。'}</p>${items.map(item=>`<div class="row"><span>${esc(item.name)}<small class="density-summary">${esc(densityText(item))}</small></span><strong>约 ${item.volume.volume_ml.toFixed(1)} mL</strong></div>`).join('')}<p class="estimate-note">mL 与 cm³ 数值相同。估算基于物体外形；包装包含内部空气，凹陷、遮挡和缺失表面也会产生误差。</p>${value.reference_image_url?'<button class="text-btn" data-action="board-reference">查看定位板识别</button>':''}</section>`;
}
function semanticPanel(){
 const meal=state.meal,summary=meal.semantic_summary||{},labels={auto:'自动分割识别',list:'食物清单辅助',grounded:'名称定位分割',manual:'点击 / 框选命名',llm:'大模型多视图识别',none:'不使用语义分割'};
 if(!meal.photo_url||meal.source_kind==='manual')return '';
 const count=meal.items.filter(i=>i.instance_id).length,active=['queued','running'].includes(meal.three_d_analysis?.status);
 return `<section class="surface semantic-summary"><div class="row">${meal.can_refine_semantics?'<button class="text-btn" data-action="refine-semantics">食物语义识别 →</button>':'<strong>食物语义识别</strong>'}<span class="nutri-tag">${labels[summary.mode]||'食物实例'}</span></div>
 <p>${summary.mode==='none'?'本餐未使用语义分割，可手工补充或修改食物清单。':count?`${count} 个独立对象区域，点击图片或清单核对名称。`:'正在准备食物区域；识别完成后可点击核对。'}</p>
 ${summary.missing_hints?.length?`<p class="estimate-note">未可靠识别：${esc(summary.missing_hints.join('、'))}。可重新标注补充。</p>`:''}
 <div class="semantic-actions">${meal.can_refine_semantics?`<button class="text-btn" data-action="refine-semantics">${meal.semantic_targets_3d?'输入食物，辅助此三维餐盘':'调整识别 / 重新标注'}</button>`:''}
 ${meal.three_d_available?'<button class="text-btn" data-page="plate">查看语义三维餐盘 →</button>':meal.reconstruction_available&&count&&!active?'<button class="text-btn" data-action="semantic-3d">生成三维食物关联 →</button>':''}</div>
 ${!meal.three_d_available&&!meal.reconstruction_available&&meal.source_kind==='photo'?'<p class="estimate-note">当前照片可用于二维语义识别；查看真实三维餐盘需要对应的多视角重建。</p>':''}</section>`;
}
function result() {
 if (!state.meal) return '<div class="empty">先拍照或导入一餐。</div>';
 const meal = state.meal;
 if(meal.video_reference_ready||meal.video_preparing)return `<div class="eyebrow">PREPARE YOUR PLATE</div><h1 class="screen-heading">准备${meal.source_kind==='scan'?'扫描':'视频'}参考帧</h1>${processing(meal)}<p class="estimate-note">先完成相机位姿估计，再在同一张参考帧上标记食物，继续重建与语义融合。草稿会保留，可以稍后继续。</p><button class="button primary" data-action="resume-video">${meal.video_reference_ready?'继续标记食物':'查看参考帧准备进度'}</button>`;
 return `<div class="backline">${state.nutritionReturnMealId===meal.id?`<button class="back" data-action="return-edit">${icon('back')}返回餐盘调整</button>`:`<button class="back" data-page="${meal.saved_at?'diary':'scan'}">${icon('back')} ${meal.saved_at?'返回日记':'返回拍摄'}</button>`}<span class="source-pill">${meal.source_kind==='demo'?'已有三维示例':meal.saved_at?'已保存':'等待确认'}</span></div>
 <div class="eyebrow">MEAL INSIGHT</div><h1 class="screen-heading">${esc(meal.title)}</h1><p class="tiny muted">${mealType[meal.meal_type]} · ${dateLabel(meal.occurred_at)} ${timeLabel(meal.occurred_at)}</p>${meal.note?`<p class="record-note">${esc(meal.note)}</p>`:''}${processing(meal)}
 ${mealImage(state.mode==='3d')}<div class="segmented"><button class="${state.mode==='photo'?'selected':''}" data-mode="photo">${meal.background_summary?.enabled?'参考照片':'原始照片'}</button>${meal.semantic_overlay_url?`<button class="${state.mode==='semantic'?'selected':''}" data-mode="semantic">语义分割</button>`:''}<button class="${state.mode==='3d'?'selected':''}" data-mode="3d">3D 餐盘</button></div>
 ${state.mode==='3d'&&meal.three_d_available?'<button class="text-btn" data-page="plate">打开完整三维餐盘 →</button>':''}${selectedCard()}${nutritionCard(meal.totals)}
 <div class="row"><h2 class="section-head" style="margin:0">食物构成</h2><button class="text-btn" data-action="add-food">＋ 补充食物</button></div>${foodList() || '<div class="empty">还没有可靠候选。<br>可以补充食物，先完成这餐记录。</div>'}
 <p class="estimate-note">已标定体积与密度用于估重，再换算营养。食品缺少专属密度时自动使用分类默认估值，可点击“修改”调整。缺少密度或营养数据的项目暂不计入营养汇总；二维记录仍标注固定参考份量。估重不是称重，请核对食物状态和可食用范围。</p>
 <button class="button primary" data-action="save">${icon(meal.saved_at?'check':'arrow')}${meal.saved_at?'修改记录信息':'确认并保存这餐'}</button>
 ${meal.saved_at?'<button class="text-btn" style="width:100%;color:#a47b68" data-action="delete-meal">删除这条记录</button>':''}
 ${backgroundPanel()}${semanticPanel()}${volumePanel()}`;
}
function plate() {
 if (!state.meal) return '<div class="empty">请先选择一条餐食。</div>';
 return `<div class="backline"><button class="back" data-page="result">${icon('back')}返回结果</button><span class="pill">3D VIEW</span></div><div class="eyebrow">SEE YOUR PLATE</div>
 <h1 class="screen-heading">从不同角度，<br>看见这一餐。</h1><p class="estimate-note">拖动视角，轻点食物查看对应营养。</p>${processing(state.meal)}${mealImage(true)}
 ${state.meal.three_d_available?'<button class="button ghost" data-action="reset-view">恢复拍摄视角</button>':'<button class="button ghost" data-mode="photo">返回原图与清单</button>'}
 ${state.meal.can_refine_semantics?'<button class="button primary" data-action="refine-semantics">输入食物，辅助此三维餐盘</button>':''}
 ${selectedCard()}${backgroundPanel()}${volumePanel()}<h2 class="section-head">餐盘包含</h2>${foodList()}<p class="estimate-note">未拍到的表面可能有重建伪影。三维体积未标定为真实重量，热量仍按参考份量估算。</p>`;
}
function diary() {
 const records = state.meals.filter(meal=>localDay(new Date(meal.occurred_at))===state.date);
 const sum = state.summary || {count:0,totals:{}};
 return `<div class="row topline"><div><div class="eyebrow">MY FOOD LOG</div><h1 class="screen-heading">饮食日记</h1></div><button class="round-btn" data-page="scan" aria-label="添加记录">${icon('camera')}</button></div>
 <div class="date-row"><button data-action="date-prev" aria-label="前一天">‹</button><input id="diaryDate" type="date" value="${state.date}" aria-label="日记日期"><button data-action="date-next" aria-label="后一天">›</button></div>
 <div class="diary-total"><div><div class="tiny" style="color:#d9e8df;font-weight:800">这天已记录 · 估计热量</div><strong>${energyLabel(sum.totals)}</strong> ${energyPending(sum.totals)?'':'kcal'}<p>共 ${records.length} 条餐食${sum.totals.unknown_items?` · ${sum.totals.unknown_items} 项营养待补充`:''}</p></div><div class="spark">✳</div></div>
 <section class="surface daily">${macros(sum.totals)}</section><h2 class="section-head">这天的餐食</h2>
 ${records.length?records.map(meal=>`<div class="diary-group">${mealType[meal.meal_type]} · ${timeLabel(meal.occurred_at)}</div>${mealLine(meal)}`).join(''):'<div class="empty">这天还没有记录。<br>拍一张，或者手工补记一餐。</div>'}
 <button class="button ghost" style="margin-top:20px" data-page="scan">＋ 再记录一餐</button>`;
}
function profile() {
 const p = state.profile || {};
 return `<div class="eyebrow">MY SPACE</div><h1 class="screen-heading">我的</h1><div class="settings-profile"><div class="avatar-large">${esc((p.display_name || '盘')[0])}</div><div><strong>${esc(p.display_name)}</strong><p>记录饮食，是为了更了解自己。</p></div></div>
 <h2 class="section-head">个人偏好</h2><div class="surface" style="overflow:hidden">
 <button class="settings-row" data-action="preferences"><span class="setting-icon">◎</span><span class="setting-text">昵称与记录偏好<small>每日参考值、时间与照片保存</small></span><span class="setting-end">›</span></button>
 <button class="settings-row" data-action="goal"><span class="setting-icon">◔</span><span class="setting-text">每日热量参考值<small>可随时修改或跳过</small></span><span class="setting-end">${p.goal_kcal?kcal(p.goal_kcal)+' kcal':'未设置'} ›</span></button>
 <button class="settings-row" data-action="privacy"><span class="setting-icon">♧</span><span class="setting-text">数据与隐私<small>导出或删除本机记录</small></span><span class="setting-end">›</span></button>
 <button class="settings-row" data-action="install"><span class="setting-icon">＋</span><span class="setting-text">添加到主屏幕<small>像 App 一样打开盘感</small></span><span class="setting-end">›</span></button>
 <button class="settings-row" data-action="about"><span class="setting-icon">✳</span><span class="setting-text">关于盘感与估算来源<small>了解识别与参考营养的局限</small></span><span class="setting-end">›</span></button></div>
 <p class="safe-note">照片由本机服务处理，记录存入本机数据库。当前为个人使用的 P0 应用，不需要注册账号。</p>`;
}
function modalHTML() {
 const modal = state.modal;
 if (!modal) return '';
 let content = '';
 if(modal.kind==='board-reference')content=`<h3>定位板体积标定</h3><p>${esc(state.meal.volume_summary.board_name||'PanSense 定位板')}按原尺寸打印。绿色轮廓为识别到的标记，系统结合多个视角检查物理尺度。</p><img src="${state.meal.volume_summary.reference_image_url}" alt="多个视角的定位板标记检测" style="width:100%;height:auto;border-radius:12px"><p>该检查用于尺度估计，不能证明物体体积没有误差。</p>`;
 if(modal.kind==='recognition-results')content=`<h3>使用已有识别结果</h3><p>将分割区域、食物名称和已有三维关联用于餐食草稿；确认保存后才计入日记。</p>${state.recognitionChoices.map(m=>`<button class="recognition-choice" data-import-run="${m.id}">${m.photo_url?`<img src="${m.photo_url}" alt="识别照片">`:''}<span><strong>${esc(m.items.map(i=>i.name).join('、'))}</strong><small>${m.items.length} 个食物${m.three_d_available?' · 已有三维关联':''}${m.comparison_imported?' · 已在餐食中':''}</small></span></button>`).join('')||'<p>还没有可用结果。可以先导入照片进行识别。</p>'}`;
 if (modal.kind==='goal') content = `<h3>每日热量参考值</h3><p>仅用于显示记录进度，由你选择，不是自动制定的饮食计划；可以不设置。</p>
 <form id="goalForm"><div class="form-row"><label for="goalValue">参考热量（kcal）</label><input id="goalValue" type="number" min="100" max="10000" value="${state.profile.goal_kcal || ''}" placeholder="例如 1900"></div><button class="button primary" type="submit">保存参考值</button><button class="text-btn" type="button" data-action="clear-goal" style="width:100%">${modal.onboard?'跳过并开始记录':'不设置参考值'}</button></form>`;
 if (modal.kind==='preferences') content = `<h3>记录偏好</h3><form id="preferencesForm"><div class="form-row"><label for="displayName">你的昵称</label><input id="displayName" maxlength="40" required value="${esc(state.profile.display_name)}"></div>
 <div class="form-row"><label for="timeZone">记录时区</label><select id="timeZone">${[...new Set([state.profile.timezone,Intl.DateTimeFormat().resolvedOptions().timeZone,'Asia/Shanghai','UTC'])].map(value=>`<option ${value===state.profile.timezone?'selected':''}>${esc(value)}</option>`).join('')}</select></div>
 <label class="checkbox-line"><input id="retainMedia" type="checkbox" ${state.profile.retain_media?'checked':''}>保存餐食照片和三维资产</label><p>关闭后新保存的餐食仅保留文字与营养记录，已有餐食不自动删除。随时可单独删除记录。</p><button class="button primary" type="submit">保存偏好</button></form>`;
 if (modal.kind==='food') {
  const item = modal.itemId?state.meal.items.find(item=>item.id===modal.itemId):null;
  content = `<h3>${item?'更正食物':'补充食物'}</h3><form id="foodForm"><div class="form-row"><label for="foodSearch">搜索食物库</label><input id="foodSearch" type="search" placeholder="如：馒头、牛奶、番茄炒蛋" autocomplete="off"><label for="foodCategory">食物分类</label><select id="foodCategory"><option value="">全部分类 · ${state.catalog.length} 项</option>${foodCatalogCategories().map(([key,name])=>`<option value="${key}">${esc(name)}</option>`).join('')}</select><label for="foodCatalog">营养参考食物</label><select id="foodCatalog">${catalogOptions(item?.catalog_key||'')}</select><small id="catalogReference">${catalogReferenceHTML(item?.catalog_key)}</small><small>请核对生熟、部位和处理方式，再选择参考食物。</small></div>
  <div class="form-row"><label for="foodName">食物名称</label><input id="foodName" maxlength="80" required value="${esc(item?.name || '')}" placeholder="输入名称或选择历史食物" list="mealFoodNames" autocomplete="off"><datalist id="mealFoodNames">${historyFoodOptions()}</datalist><small>候选包含食物库与历史食物，也可以填写自定义名称。缺少营养数据时将后台联网查询食品名称。</small><small id="nutritionSearchStatus" role="status" aria-live="polite">${item?.nutrition_lookup?nutritionLookupHTML({...item.nutrition_lookup,nutrition:item.nutrition}):''}</small><button class="text-btn" type="button" data-action="nutrition-reference">重新查询营养数据</button></div>
  <div id="densityFields">${densityFieldsHTML(item)}</div>
  ${!item&&state.meal.photo_url?`<p>可选：在照片上拖出区域，便于查看。</p><div class="region-picker"><img id="regionPhoto" class="region-draw" src="${state.meal.photo_url}" alt="拖动标记补充食物的位置"><span id="regionBox" class="region-box"></span></div>`:''}
  <button class="button primary" type="submit" style="margin-top:18px">${item?'保存更正':'添加食物'}</button></form>`;
 }
 if (modal.kind==='save') content = `<h3>${state.meal.saved_at?'修改记录信息':'保存这一餐'}</h3><form id="saveForm">
 <div class="form-row"><label for="mealTitle">餐食标题</label><input id="mealTitle" required maxlength="80" value="${esc(state.meal.title)}"></div>
 <div class="two-fields"><div class="form-row"><label for="mealType">餐次</label><select id="mealType">${Object.entries(mealType).map(([key,name])=>`<option value="${key}" ${key===state.meal.meal_type?'selected':''}>${name}</option>`).join('')}</select></div><div class="form-row"><label for="mealTime">时间</label><input id="mealTime" type="datetime-local" required value="${inputTime(state.meal.occurred_at)}"></div></div>
 <div class="form-row"><label for="mealNote">备注（可选）</label><textarea id="mealNote" maxlength="1000" placeholder="记录口味、搭配或其他想记下的事">${esc(state.meal.note)}</textarea></div>
 <p class="save-note">估算信息会随食物清单一起保存。${!state.profile.retain_media?'当前偏好仅保留文字与营养，不保存照片和三维资产。':''}</p><button class="button primary" type="submit">保存到饮食日记</button></form>`;
 if (modal.kind==='privacy') content = `<h3>数据与隐私</h3><p>照片、分析文件、三维模型和餐食记录保存在本机项目目录。开启大模型识别或自动排除背景时，所选关键帧会发送给当前 Agent 使用的模型服务；本机执行后续掩码跟踪与重建。缺少营养数据时，后台会将食品名称发送到 USDA FoodData Central 和 Open Food Facts 查询；不发送照片、餐食备注或个人信息。首次下载模型权重也需要网络。</p><p>浏览器仅缓存应用界面，不缓存你的照片、营养记录或三维画面。导出文件包含记录和偏好，不包含照片和模型。</p><a class="button ghost" href="/api/privacy/export" download>导出记录 JSON</a><button class="button danger" style="margin-top:12px" data-action="confirm-purge">删除本机全部餐食数据</button>`;
 if (modal.kind==='about') content = `<h3>关于盘感</h3><p>盘感 P0 让拍照、核对、保存和查看日记连成一个轻松的流程。视频或已有多视角模型可以生成真实三维餐盘，单张照片使用二维清单回退。</p><p>识别使用预训练 SAM 2 与 CLIP，名称可能有误。营养来自 <a href="https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/methods-and-application-of-food-composition-laboratory/mafcl-site-pages/sr11-sr28/" target="_blank" rel="noopener">USDA SR28（2016 年修订）</a>，及后台查询的 USDA FoodData Central、Open Food Facts 产品标签参考。有已标定体积时，按体积与密度估重并换算营养；其他记录使用固定参考份量，均非称重。在线结果保留来源并提示核对。具体烹饪方式、油脂和隐藏配料可能造成差异。</p><p>密度优先使用手动修改值，其次使用食品参考密度，缺失时自动使用所属分类的默认估值。分类值由 USDA 量杯份量与 FAO 菜肴样本计算，包含堆积空隙；整颗、切块及生熟状态会有差异。</p><details><summary>查看 11 类默认密度</summary><table class="density-table"><thead><tr><th>分类</th><th>默认 g/mL</th><th>样本数</th></tr></thead><tbody>${state.categoryDensities.map(d=>`<tr><td>${esc(d.category_label)}</td><td>${d.value_g_ml.toFixed(3)}</td><td>${d.sample_count}</td></tr>`).join('')}</tbody></table><p class="small">样本较少的分类仅作粗略估算；可在食品的“修改”中填写更合适的密度。</p></details><p>交互份量调整与餐前餐后对比属于 P1，本阶段未开放。</p>`;
 if (modal.kind==='install') content = `<h3>添加到主屏幕</h3><p>在支持安装的浏览器中打开菜单，选择“安装应用”或“添加到主屏幕”。iPhone 可在 Safari 的分享菜单中添加。</p><p>手机连接本机服务需要可访问的安全地址。离线时可打开应用界面，分析与日记数据仍需要连接本机服务。</p>`;
 if (modal.kind==='confirm') content = `<h3>${esc(modal.title)}</h3><p>${esc(modal.message)}</p><div class="modal-actions"><button class="button ghost" data-action="modal-close">取消</button><button class="button danger" data-action="confirm-action">${esc(modal.label || '确认删除')}</button></div>`;
 return `<div class="modal-scrim"><section class="modal-box" role="dialog" aria-modal="true" aria-label="${esc(modal.kind)}"><button class="modal-close" data-action="modal-close" aria-label="关闭弹窗">×</button>${content}</section></div>`;
}
function inputTime(value) {
 const parts = new Intl.DateTimeFormat('sv-SE',{timeZone:zone(),year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23'}).formatToParts(new Date(value));
 const obj = Object.fromEntries(parts.map(part=>[part.type,part.value]));
 return `${obj.year}-${obj.month}-${obj.day}T${obj.hour}:${obj.minute}`;
}
function zonedISO(value) {
 const [year,month,day,hour,minute] = value.match(/\d+/g).map(Number);
 const requested = Date.UTC(year,month-1,day,hour,minute);let result=requested;
 for (let i=0;i<3;i++) {
  const parts = new Intl.DateTimeFormat('sv-SE',{timeZone:zone(),year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23'}).formatToParts(new Date(result));
  const obj = Object.fromEntries(parts.map(part=>[part.type,Number(part.value)]));
  result -= Date.UTC(obj.year,obj.month-1,obj.day,obj.hour,obj.minute)-requested;
 }
 const iso = new Date(result).toISOString();if(inputTime(iso)!==value) throw new Error('请选择有效的记录时间');return iso;
}
function render(preserve=false) {
 const screen=$('screen'),scroll=screen.scrollTop;
 const pages={home,onboard,scan,result,plate,diary,profile};
 screen.innerHTML = state.offline?'<div class="offline-panel"><h2>暂未连接本机服务</h2><p>应用界面可以离线打开。连接恢复后，继续查看日记或分析食物。</p><button class="button primary" data-action="reconnect">重新连接</button></div>':pages[state.page]();
 $('tabbar').innerHTML=nav();$('modal').innerHTML=modalHTML();screen.scrollTop=preserve?scroll:0;
 if(state.stream&&$('cameraPreview')) $('cameraPreview').srcObject=state.stream;
 paintSeeds();
 if(state.page==='scan'&&state.captureFlow==='mobile')phoneScan.mount();
 if($('plateImage')) {state.imageReady=false;$('plateImage').onload=()=>{state.imageReady=true;};if($('plateImage').complete)state.imageReady=true;}
 if(viewActive()&&!state.view&&!state.viewBusy) queueMicrotask(()=>requestView(true));
}
let toastTimer;
function toast(message) {$('toast').textContent=message;$('toast').classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.remove('show'),3200);}
function stopCamera() {if(state.stream)state.stream.getTracks().forEach(track=>track.stop());state.stream=null;}
async function refreshRecords() {
 const [records,all,summary] = await Promise.all([api('/meals'),api('/meals?drafts=true'),api(`/summary?date=${state.date || localDay()}`)]);
 state.meals=records.meals;state.drafts=all.meals.filter(meal=>!meal.saved_at);state.summary=summary;
}
async function go(page, push=true) {
 if(page!=='scan')phoneScan.leave();
 if(page!=='scan')stopCamera();
 if(['result','plate'].includes(page)&&!state.meal){toast('请先拍照或选择一餐');page='scan';}
 if(page==='plate')state.mode='3d';
 if(page==='home'){state.date=localDay();await refreshRecords();}
 if(page==='diary')await refreshRecords();
 if(page==='scan'&&state.captureFlow==='edit'&&!state.recognitionParent)await refreshRecords();
 if(page==='scan'&&state.captureFlow==='mobile')await phoneScan.enter();
 state.page=page;
 if(push)history.pushState(null,'',`#${page}${['result','plate'].includes(page)?'/'+state.meal.id:''}`);
 render();
}
async function openMeal(id, platePage=false, push=true) {
 resetViewer();
 state.meal=await api(`/meals/${id}`);
 if(state.meal.items.some(item=>!item.nutrition&&!['non_food','packaged_food'].includes(item.object_kind)))state.meal=await api(`/meals/${id}/nutrition/lookup`,{method:'POST'});
 state.selected=null;state.mode=platePage?'3d':state.meal.semantic_overlay_url?'semantic':'photo';
 state.view=null;state.viewBusy=false;state.desired={yaw:0,pitch:0,zoom:1};
 await go(platePage?'plate':'result',push);pollMeal();
}
function pollMeal() {
 clearTimeout(state.poll);
 if(!state.meal||!(['queued','running'].includes(state.meal.analysis.status)||['queued','running'].includes(state.meal.three_d_analysis?.status)||state.meal.items.some(nutritionPending)))return;
 const id=state.meal.id;
 state.poll=setTimeout(async()=>{
  try{const meal=await api(`/meals/${id}`);if(state.meal?.id!==id)return;if(!state.meal.semantic_overlay_url&&meal.semantic_overlay_url&&state.mode==='photo')state.mode='semantic';
   if(!state.meal.three_d_available&&meal.three_d_available&&meal.semantic_auto_3d&&['result','plate'].includes(state.page)){state.mode='3d';state.view=null;toast('本次食物分割已应用到三维餐盘，可拖动和点击查看');}
   const nutritionFinished=state.meal.items.some(nutritionPending)&&!meal.items.some(nutritionPending);
   state.meal=meal;
   if(nutritionFinished)await refreshRecords();
   if((['result','plate'].includes(state.page)||nutritionFinished&&['home','diary'].includes(state.page))&&!state.modal)render(true);pollMeal();
  }catch(error){toast(error.message);}
 },1800);
}

// Interaction handlers and camera updates are declared below.
let lastViewSent=0,viewTimer=null,pendingViewReset=false,viewInFlight=false;
function resetViewer(){
 state.viewEpoch++;clearTimeout(viewTimer);viewTimer=null;pendingViewReset=false;
 state.view=null;state.viewBusy=false;state.viewDirty=false;state.imageReady=false;state.drag=null;
 state.selected=null;state.desired={yaw:0,pitch:0,zoom:1};
}
async function requestView(reset=false) {
 if(!viewActive())return;
 if(reset)pendingViewReset=true;
 if(viewInFlight){state.viewDirty=true;return;}
 const id=state.meal.id,epoch=state.viewEpoch,resetNow=pendingViewReset;pendingViewReset=false;
 const requested={...state.desired};const food=state.selected;
 viewInFlight=true;state.viewBusy=true;state.viewDirty=false;lastViewSent=performance.now();
 if($('viewStatus'))$('viewStatus').textContent='更新视角…';
 try {
  const body=resetNow?{food_id:food}:{base_token:state.view?.token,...requested,food_id:food};
  const view=await api(`/meals/${id}/view`,{method:'POST',body});
  const preload=new Image();await new Promise((resolve,reject)=>{preload.onload=resolve;preload.onerror=()=>reject(new Error('三维画面暂未加载，请重试'));preload.src=view.image_url;});
  if(state.meal?.id!==id||state.viewEpoch!==epoch||!viewActive())return;
  state.view=view;state.imageReady=false;
  if(resetNow)state.desired={yaw:0,pitch:0,zoom:1};
  if($('plateImage')){$('plateImage').src=view.image_url;state.imageReady=true;}
  else if(viewActive())render(true);
 }catch(error){
  if(state.meal?.id===id&&state.viewEpoch===epoch&&viewActive()){
   state.view=null;
   if(captureMealActive()){state.editPreview='reference';state.editViewError=error.message;}
   else{state.mode='photo';if(state.page==='plate'){state.page='result';history.replaceState(null,'',`#result/${id}`);}}
   render(true);toast(error.message);
  }
 }finally{
  viewInFlight=false;
  if(state.viewEpoch===epoch){state.viewBusy=false;
   if($('viewStatus'))$('viewStatus').textContent='';}
  if(state.viewDirty&&viewActive())scheduleView();
 }
}
function scheduleView() {
 state.viewDirty=true;if(state.viewBusy||viewTimer!==null)return;
 viewTimer=setTimeout(()=>{viewTimer=null;requestView();},Math.max(0,65-(performance.now()-lastViewSent)));
}
function imagePoint(image,event,clamp=false) {
 if(!image?.naturalWidth)return null;
 const rect=image.getBoundingClientRect(),scale=Math.min(rect.width/image.naturalWidth,rect.height/image.naturalHeight);
 const width=image.naturalWidth*scale,height=image.naturalHeight*scale;
 const left=rect.left+(rect.width-width)/2,top=rect.top+(rect.height-height)/2;
 let x=(event.clientX-left)/width,y=(event.clientY-top)/height;
 if(!clamp&&(x<0||x>1||y<0||y>1))return null;
 x=Math.max(0,Math.min(1,x));y=Math.max(0,Math.min(1,y));
 return {x,y,width,height,left,top};
}
async function selectFood(id) {
 state.selected=id;render(true);
 if(viewActive()){state.viewDirty=true;scheduleView();}
}
async function clickImage(event) {
 const image=event.target.closest('#plateImage,#mealPhoto');if(!image)return;
 if(performance.now()<state.suppressUntil || state.viewBusy || image.id==='plateImage'&&!state.imageReady)return;
 const point=imagePoint(image,event);if(!point)return;
 const body=image.id==='plateImage'?{token:state.view.token,x:point.x,y:point.y}:{x:point.x,y:point.y};
 const picked=await api(`/meals/${state.meal.id}/${image.id==='plateImage'?'pick':'photo-pick'}`,{method:'POST',body});
 state.selected=picked.food_id;
 if(!picked.food_id)toast(picked.message || '背景区域');
 render(true);
 if(image.id==='plateImage'){state.viewDirty=true;scheduleView();}
}
document.addEventListener('pointerdown',event=>{
 const seedCanvas=event.target.closest('#mealSeedCanvas');
 if(seedCanvas&&event.button===0&&event.isPrimary&&!state.uploading){
  const name=state.seedName.trim();if(!name){toast('先填写当前食物名称，再标记照片');$('seedName')?.focus();return;}
  if(state.manualRegions.length>=20){toast('最多标记 20 个食物');return;}
  event.preventDefault();state.seedDrag={id:event.pointerId,start:seedPoint(event),name,tool:state.seedTool};seedCanvas.setPointerCapture(event.pointerId);
 }
 const image=event.target.closest('#plateImage');
 if(image&&event.button===0&&event.isPrimary&&state.view&&state.imageReady){
  const point=imagePoint(image,event);if(!point)return;
  state.drag={id:event.pointerId,x:event.clientX,y:event.clientY,width:point.width,height:point.height,start:{...state.desired},moved:false};
 }
 const region=event.target.closest('#regionPhoto');
 if(region&&event.button===0){const point=imagePoint(region,event);if(point)state.regionDrag={id:event.pointerId,start:point};}
});
document.addEventListener('pointermove',event=>{
 if(state.seedDrag?.id===event.pointerId&&$('mealSeedCanvas')){event.preventDefault();state.seedDrag.end=seedPoint(event);paintSeeds();return;}
 if(state.drag?.id===event.pointerId){
  if(event.pointerType==='mouse'&&!(event.buttons&1)){state.drag=null;return;}
  const dx=event.clientX-state.drag.x,dy=event.clientY-state.drag.y;
  if(Math.hypot(dx,dy)>5)state.drag.moved=true;
  if(state.drag.moved){event.preventDefault();state.desired.yaw=((state.drag.start.yaw-120*dx/state.drag.width+540)%360)-180;
   state.desired.pitch=Math.max(-75,Math.min(75,state.drag.start.pitch-90*dy/state.drag.height));
   state.viewDirty=true;document.querySelector('.plate-shell')?.classList.add('dragging');scheduleView();}
 }
 if(state.regionDrag?.id===event.pointerId&&$('regionPhoto')){
  event.preventDefault();const end=imagePoint($('regionPhoto'),event,true);const start=state.regionDrag.start;
  const box=[Math.min(start.x,end.x),Math.min(start.y,end.y),Math.max(start.x,end.x),Math.max(start.y,end.y)];
  state.modal.box=box;const parent=$('regionPhoto').parentElement.getBoundingClientRect();
  Object.assign($('regionBox').style,{display:'block',left:`${end.left-parent.left+box[0]*end.width}px`,top:`${end.top-parent.top+box[1]*end.height}px`,width:`${(box[2]-box[0])*end.width}px`,height:`${(box[3]-box[1])*end.height}px`});
 }
},{passive:false});
function pointerEnd(event){
 if(state.seedDrag?.id===event.pointerId){
  const drag=state.seedDrag;state.seedDrag=null;
  if(event.type!=='pointercancel'&&$('mealSeedCanvas')){const end=seedPoint(event),id=Math.max(0,...state.manualRegions.map(r=>r.instance_id))+1;const region={instance_id:id,name:drag.name};
   if(drag.tool==='point')region.point=drag.start;
   else{region.box=[Math.min(drag.start[0],end[0]),Math.min(drag.start[1],end[1]),Math.max(drag.start[0],end[0]),Math.max(drag.start[1],end[1])];if(region.box[2]-region.box[0]<.005||region.box[3]-region.box[1]<.005){refreshSeeds();toast('请拖出一个更大的食物区域');return;}}
   state.manualRegions.push(region);toast(`已标记食物 ${id}：${region.name}`);
  }
  refreshSeeds();
 }
 if(state.drag?.id===event.pointerId){if(state.drag.moved){state.suppressUntil=performance.now()+500;state.viewDirty=true;scheduleView();}state.drag=null;document.querySelector('.plate-shell')?.classList.remove('dragging');}
 if(state.regionDrag?.id===event.pointerId)state.regionDrag=null;
}
document.addEventListener('pointerup',pointerEnd);document.addEventListener('pointercancel',pointerEnd);
document.addEventListener('dragstart',event=>{if(event.target.closest('#plateImage,#regionPhoto'))event.preventDefault();});
document.addEventListener('wheel',event=>{
 if(!event.target.closest('#plateImage')||!state.view)return;
 event.preventDefault();const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?400:1);
 state.desired.zoom=Math.max(.5,Math.min(2.5,state.desired.zoom*Math.exp(Math.max(-300,Math.min(300,delta))*.0015)));
 state.suppressUntil=performance.now()+300;state.viewDirty=true;scheduleView();
},{passive:false});

async function chooseFile(file,kind) {
 if(!file)return;
 const limit=(kind==='photo'?20:240)*1024*1024;
 if(file.size>limit){toast(kind==='photo'?'照片请小于 20 MB':'视频请小于 240 MB');return;}
 const assistance=state.assistMode,foodText=state.foodText;
 const sameFlow=state.captureFlow===(kind==='video'?'video':'photo');
 resetCapture();
 state.captureFlow=kind==='video'?'video':'photo';
 state.assistMode=sameFlow?assistance:'grounded';state.foodText=sameFlow?foodText:'';
 state.file=file;state.fileKind=kind;state.fileUrl=URL.createObjectURL(file);state.page='scan';render();
}
function resetCapture(){
 phoneScan.leave();
 resetViewer();
 stopCamera();clearTimeout(state.preparePoll);state.captureEpoch++;
 if(state.fileUrl)URL.revokeObjectURL(state.fileUrl);
 Object.assign(state,{file:null,fileUrl:null,manualRegions:[],seedDrag:null,seedPicture:null,seedPictureUrl:null,
  seedSourceUrl:null,recognitionParent:null,videoPrepared:null,videoPreparing:false,videoPreparedStatus:'',
  seedName:'',foodText:'',editPreview:'3d',editViewError:null,nutritionReturnMealId:null});
}
async function refineMeal(id){
 const parent=await api(`/meals/${id}`);
 if(!parent.can_refine_semantics||!parent.photo_url)throw new Error('这份餐盘已无可用于调整的参考照片');
 const response=await fetch(parent.photo_url);if(!response.ok)throw new Error('原照片暂不可用');
 await chooseFile(new File([await response.blob()],'重新标注.jpg',{type:'image/jpeg'}),'photo');
 state.captureFlow='edit';state.historyMealId=parent.id;
 state.meal=parent;state.editPreview=parent.three_d_available?'3d':'reference';
 state.recognitionParent={id:parent.id,revision:parent.revision,threeD:parent.semantic_targets_3d,title:parent.title};
 state.assistMode=['manual','list','grounded','llm','none'].includes(parent.assistance?.mode)?parent.assistance.mode:'grounded';
 state.foodText=parent.assistance?.foods?.length?parent.assistance.foods.map(f=>`${f.name}|${f.text}`).join('、'):parent.items.map(i=>i.name).join('、');
 state.manualRegions=parent.assistance?.mode==='manual'?parent.assistance.instances.map(r=>({...r})):[];
 history.pushState(null,'','#scan');render();
}
async function prepareVideo(){
 if(!state.file||state.fileKind!=='video'||state.uploading||state.videoPreparing)return;
 const epoch=state.captureEpoch;
 state.uploading=true;render(true);
 try{
  if(!state.videoPrepared){
   const form=new FormData();form.append('file',state.file);form.append('prepare','true');form.append('board_background',String(state.boardBackground));
   state.videoPrepared=await api('/meals/upload?kind=video',{method:'POST',body:form});
  }else{
   const latest=await api(`/meals/${state.videoPrepared.id}`);
   state.videoPrepared=['failed','cancelled'].includes(latest.analysis.status)?await api(`/meals/${latest.id}/retry`,{method:'POST'}):latest;
  }
  if(epoch!==state.captureEpoch)return;
  state.videoPreparing=true;
  await pollVideoPreparation(epoch);
 }finally{if(epoch===state.captureEpoch){state.uploading=false;if(state.page==='scan')render(true);}}
}
async function pollVideoPreparation(epoch){
 if(epoch!==state.captureEpoch||!state.videoPrepared)return;
 try{
  const meal=await api(`/meals/${state.videoPrepared.id}`);
  if(epoch!==state.captureEpoch)return;
  state.videoPrepared=meal;state.videoPreparedStatus=meal.analysis.stage;
  if(meal.video_reference_ready&&meal.photo_url){state.videoPreparing=false;state.seedSourceUrl=meal.photo_url;toast('参考帧已准备好，可以点击或框选食物');}
  else if(['failed','cancelled'].includes(meal.analysis.status)){state.videoPreparing=false;toast(meal.analysis.status==='cancelled'?'已中断参考帧准备，原始素材已保留':meal.analysis.error||'参考帧准备失败，可以重试');}
  else state.preparePoll=setTimeout(()=>pollVideoPreparation(epoch),1800);
  if(state.page==='scan')render(true);
 }catch(error){state.videoPreparing=false;toast(error.message);if(state.page==='scan')render(true);}
}
for(const [picker,kind] of [['photoPicker','photo'],['cameraPicker','photo'],['videoPicker','video']]){
 $(picker).addEventListener('change',event=>{chooseFile(event.target.files[0],kind);event.target.value='';});
}
async function launchCamera() {
 if(/Android|iPhone|iPad/i.test(navigator.userAgent)){$('cameraPicker').click();return;}
 if(!navigator.mediaDevices?.getUserMedia){toast('当前环境无法打开相机，请从相册导入');return;}
 try{stopCamera();state.stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});render();}
 catch(_){toast('无法打开相机，可以从相册导入照片');}
}
async function capture() {
 const video=$('cameraPreview');if(!video?.videoWidth){toast('相机尚未准备好');return;}
 const canvas=document.createElement('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;
 canvas.getContext('2d').drawImage(video,0,0);const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.95));
 await chooseFile(new File([blob],'meal-photo.jpg',{type:'image/jpeg'}),'photo');
}
async function analyze() {
 if(!state.file||state.uploading)return;
 if(state.recognitionParent&&state.assistMode==='none'){
  const id=state.recognitionParent.id;state.nutritionReturnMealId=id;await openMeal(id);return;
 }
 if(state.fileKind==='video'&&state.assistMode==='manual'&&!state.videoPrepared?.video_reference_ready){toast('请先准备视频标注参考帧');return;}
 if(state.videoPrepared&&!state.videoPrepared.video_reference_ready){toast('参考帧尚未准备好，请重试准备或重新选择视频');return;}
 if(state.assistMode==='manual'&&!state.manualRegions.length){toast('请先填写名称并在照片上点击或框选食物');return;}
 if(['list','grounded'].includes(state.assistMode)&&!state.foodText.trim()){toast('请填写这餐可能有哪些食物');return;}
 state.uploading=true;render(true);
 try{const assistance={mode:state.assistMode,food_text:['list','grounded'].includes(state.assistMode)?state.foodText:'',instances:state.assistMode==='manual'?state.manualRegions:[]};let meal;
  if(state.recognitionParent){meal=await api(`/meals/${state.recognitionParent.id}/recognize`,{method:'POST',body:{revision:state.recognitionParent.revision,assistance}});}
  else if(state.fileKind==='video'&&state.videoPrepared){meal=await api(`/meals/${state.videoPrepared.id}/video-start`,{method:'POST',body:{revision:state.videoPrepared.revision,assistance}});}
  else{const form=new FormData();form.append('file',state.file);form.append('assistance',JSON.stringify(assistance));form.append('board_background',String(state.boardBackground));meal=await api(`/meals/upload?kind=${state.fileKind}`,{method:'POST',body:form});}
  state.file=null;if(state.fileUrl)URL.revokeObjectURL(state.fileUrl);state.fileUrl=null;
  state.manualRegions=[];state.seedPicture=null;state.seedSourceUrl=null;state.recognitionParent=null;state.videoPrepared=null;
  await openMeal(meal.id);
 }finally{state.uploading=false;if(state.page==='scan')render(true);}
}
function openModal(modal){state.modal=modal;$('modal').innerHTML=modalHTML();const input=$('modal').querySelector('input,select,button');input?.focus();if(modal.kind==='food'){updateDensityPreview();queryNutritionReference();}}
function closeModal(){nutritionReferenceEpoch++;clearTimeout(nutritionReferenceTimer);state.modal=null;state.regionDrag=null;$('modal').innerHTML='';}
async function setGoal(value,onboard=false){
 state.profile=await api('/profile',{method:'PATCH',body:{goal_kcal:value,...(onboard?{onboarded:true}:{})}});
 closeModal();await refreshRecords();await go(onboard?'home':state.page);toast(value?'参考值已保存':'可以先记录，不设置参考值');
}
async function action(name){
 switch(name){
  case 'reconnect':await bootstrap();break;
  case 'onboard-goal':openModal({kind:'goal',onboard:true});break;
  case 'onboard-skip':state.profile=await api('/profile',{method:'PATCH',body:{onboarded:true}});await go('home');break;
  case 'goal':openModal({kind:'goal'});break;
  case 'clear-goal':await setGoal(null,state.modal.onboard);break;
  case 'preferences':openModal({kind:'preferences'});break;
  case 'privacy':openModal({kind:'privacy'});break;
  case 'about':openModal({kind:'about'});break;
  case 'board-reference':openModal({kind:'board-reference'});break;
 case 'modal-close':closeModal();break;
 case 'nutrition-reference':await queryNutritionReference(true);break;
  case 'semantic-reference':{
   const response=await fetch('/api/recognition/reference-photo');if(!response.ok)throw new Error('参考照片暂不可用');
   await chooseFile(new File([await response.blob()],'参考食物照片.jpg',{type:'image/jpeg'}),'photo');toast('已载入参考照片，可选择识别方式');break;
  }
  case 'recognition-results':state.recognitionChoices=(await api('/recognition/results')).results;openModal({kind:'recognition-results'});break;
  case 'semantic-3d':state.meal=await api(`/meals/${state.meal.id}/semantic-3d`,{method:'POST'});render(true);pollMeal();break;
  case 'interrupt':case 'interrupt-prepare':{
   if(state.interrupting)return;
   const preparing=name==='interrupt-prepare',id=preparing?state.videoPrepared?.id:state.meal?.id;if(!id)return;
   state.interrupting=true;render(true);
   try{
    const meal=await api(`/meals/${id}/interrupt`,{method:'POST'});
    if(preparing&&state.videoPrepared?.id===id){state.videoPrepared=meal;state.videoPreparing=false;state.videoPreparedStatus=meal.analysis.stage;clearTimeout(state.preparePoll);}
    if(state.meal?.id===id){state.meal=meal;pollMeal();}
    toast('已中断，原始素材和已完成结果已保留');
   }finally{state.interrupting=false;render(true);}
   break;
  }
  case 'history-plates':resetCapture();state.captureFlow='edit';await refreshRecords();render();break;
  case 'edit-history':if(state.historyMealId){
   const meal=await api(`/meals/${state.historyMealId}`);
   if(meal.can_refine_semantics&&meal.photo_url)await refineMeal(meal.id);
   else{state.nutritionReturnMealId=meal.id;await openMeal(meal.id);}
  }break;
  case 'existing-nutrition':{
   const id=state.recognitionParent?.id||state.historyMealId;if(!id)return;
   state.nutritionReturnMealId=id;await openMeal(id);
   $('mealNutrition')?.scrollIntoView({block:'start'});$('mealNutrition')?.focus({preventScroll:true});break;
  }
  case 'return-edit':state.nutritionReturnMealId=null;await go('scan');break;
  case 'reselect-capture':resetCapture();render();break;
  case 'prepare-video':await prepareVideo();break;
  case 'resume-video':{
   const meal=await api(`/meals/${state.meal.id}`);resetCapture();state.captureFlow='video';state.fileKind='video';
   state.videoPrepared=meal;state.boardBackground=!!meal.board_background;state.assistMode='manual';state.file={name:meal.source_kind==='scan'?'已上传的扫描照片':'已上传的环绕视频'};
   state.fileUrl=meal.photo_url;state.seedSourceUrl=meal.video_reference_ready?meal.photo_url:null;
   state.page='scan';history.pushState(null,'','#scan');
   if(meal.video_preparing&&['queued','running'].includes(meal.analysis.status)){state.videoPreparing=true;await pollVideoPreparation(state.captureEpoch);}
   render();break;
  }
  case 'refine-semantics':{
   await refineMeal(state.meal.id);toast('已载入餐盘参考照片，可以调整识别');break;
  }
  case 'assist-lab':{
   stopCamera();
   if(state.file&&state.fileKind==='photo'){const form=new FormData();form.append('file',state.file);const value=await api('/assist/sessions',{method:'POST',body:form});location.href='/recognition.html?session='+value.id;}
   else location.href='/recognition.html';
   break;
  }
  case 'upload-photo':stopCamera();$('photoPicker').click();break;
  case 'upload-video':stopCamera();$('videoPicker').click();break;
  case 'camera':await launchCamera();break;
  case 'capture':await capture();break;
  case 'analyze':await analyze();break;
  case 'manual':{const meal=await api('/meals/manual',{method:'POST'});await openMeal(meal.id);openModal({kind:'food'});break;}
  case 'demo':{const meal=await api('/meals/demo',{method:'POST',body:{scene:'fruit2'}});await openMeal(meal.id);toast('打开的是已重建的三维示例，可核对名称后保存');break;}
  case 'add-food':openModal({kind:'food'});break;
  case 'save':if(!state.meal.items.length){toast('请先补充至少一项食物');openModal({kind:'food'});}else openModal({kind:'save'});break;
  case 'retry':state.meal=await api(`/meals/${state.meal.id}/retry`,{method:'POST'});render(true);pollMeal();break;
  case 'reset-view':state.desired={yaw:0,pitch:0,zoom:1};state.viewDirty=false;await requestView(true);break;
  case 'date-prev':case 'date-next':{const date=new Date(state.date+'T12:00:00Z');date.setUTCDate(date.getUTCDate()+(name==='date-prev'?-1:1));state.date=date.toISOString().slice(0,10);await refreshRecords();render();break;}
  case 'delete-meal':openModal({kind:'confirm',title:'删除这条餐食？',message:'这条记录和由它上传的照片、分析资产会从本机删除，无法恢复。',verb:'delete-meal'});break;
  case 'confirm-purge':openModal({kind:'confirm',title:'删除本机全部餐食数据？',message:'所有餐食、未保存草稿和应用内上传的照片与三维资产将永久删除，偏好也会重置。原 POC 示例模型不会被删除。',verb:'purge',label:'永久删除全部数据'});break;
  case 'confirm-action':await confirmAction();break;
  case 'install':if(state.installPrompt){await state.installPrompt.prompt();await state.installPrompt.userChoice;state.installPrompt=null;}else openModal({kind:'install'});break;
 }
}
async function confirmAction(){
 const modal=state.modal;
 if(modal.verb==='remove-food'){
  state.meal=await api(`/meals/${state.meal.id}`);
  state.meal=await api(`/meals/${state.meal.id}/foods/${modal.foodId}?revision=${state.meal.revision}`,{method:'DELETE'});
  if(state.selected===modal.foodId)state.selected=null;
  closeModal();await refreshRecords();render(true);if(state.mode==='3d'){state.viewDirty=true;scheduleView();}toast('已从清单移除');
 }else if(modal.verb==='delete-meal'){
  await api(`/meals/${state.meal.id}`,{method:'DELETE'});state.meal=null;state.view=null;closeModal();await go('diary');toast('记录已删除');
 }else if(modal.verb==='purge'){
  await api('/privacy/delete',{method:'POST',body:{confirmation:'DELETE'}});closeModal();state.meal=null;state.view=null;await bootstrap();toast('本机餐食数据已删除');
 }
}
document.addEventListener('click',async event=>{
 try{
  if(event.target.closest('#plateImage,#mealPhoto')){await clickImage(event);return;}
  const button=event.target.closest('[data-page],[data-action],[data-meal],[data-food],[data-edit],[data-remove-food],[data-retry-nutrition],[data-mode],[data-import-run],[data-capture-flow],[data-capture-assist],[data-history-meal],[data-edit-preview]');
  if(!button)return;
  if(button.disabled)return;
  if(button.dataset.retryNutrition){
   const id=state.meal.id;button.disabled=true;
   state.meal=await api(`/meals/${id}/foods/${button.dataset.retryNutrition}/nutrition/retry`,{method:'POST'});
   render(true);pollMeal();return;
  }
  if(button.dataset.captureFlow){
   if(state.uploading)return;
   resetCapture();state.captureFlow=button.dataset.captureFlow;state.fileKind=state.captureFlow==='video'?'video':'photo';
   state.assistMode='grounded';state.historyMealId=null;
   if(state.captureFlow==='edit')await refreshRecords();if(state.captureFlow==='mobile')await phoneScan.enter();render();return;
  }
  if(button.dataset.editPreview){state.editPreview=button.dataset.editPreview;state.editViewError=null;render(true);return;}
  if(button.dataset.captureAssist){state.assistMode=button.dataset.captureAssist;if(state.captureFlow==='edit'&&state.assistMode==='manual')state.editPreview='reference';render(true);return;}
  if(button.dataset.historyMeal){
   const id=button.dataset.historyMeal;state.historyMealId=id;resetViewer();state.editViewError=null;state.editPreview='3d';
   const meal=await api(`/meals/${id}`);if(state.historyMealId!==id||state.captureFlow!=='edit')return;
   state.meal=meal;render(true);return;
  }
  if(button.dataset.importRun){button.disabled=true;try{const meal=await api('/meals/from-recognition',{method:'POST',body:{run_id:button.dataset.importRun}});closeModal();await openMeal(meal.id);await refreshRecords();toast('识别区域和名称已用于餐食草稿，请核对后保存');}finally{button.disabled=false;}return;}
  if(button.dataset.action==='remove-seed'){state.manualRegions.splice(Number(button.dataset.index),1);refreshSeeds();return;}
  if(button.dataset.page){await go(button.dataset.page);return;}
  if(button.dataset.meal){await openMeal(button.dataset.meal);return;}
  if(button.dataset.food){await selectFood(button.dataset.food);return;}
  if(button.dataset.edit){openModal({kind:'food',itemId:button.dataset.edit});return;}
  if(button.dataset.removeFood){const item=state.meal.items.find(item=>item.id===button.dataset.removeFood);openModal({kind:'confirm',title:`移除${item.name}？`,message:'从当前食物清单移除，营养汇总会同步更新。',verb:'remove-food',foodId:item.id});return;}
  if(button.dataset.mode){state.mode=button.dataset.mode;state.page='result';render(true);return;}
  if(button.dataset.action)await action(button.dataset.action);
 }catch(error){toast(error.message);}
});
document.addEventListener('input',event=>{if(event.target.id==='foodHints')state.foodText=event.target.value;if(event.target.id==='seedName')state.seedName=event.target.value;});
document.addEventListener('change',async event=>{
 if(event.target.id==='assistMode'){state.assistMode=event.target.value;render(true);return;}
 if(event.target.id==='boardBackground'){state.boardBackground=event.target.checked;render(true);return;}
 if(event.target.name==='seedTool'){state.seedTool=event.target.value;return;}
 try{
  if(event.target.id==='diaryDate'){state.date=event.target.value;await refreshRecords();render();}
  if(event.target.id==='foodCategory'){filterFoodCatalog();}
  if(event.target.id==='densityMode')updateDensityPreview();
  if(event.target.id==='foodCatalog'){$('foodName').value=catalogItem(event.target.value)?.name || '';$('catalogReference').innerHTML=catalogReferenceHTML(event.target.value);filterFoodCatalog();refreshDensityFields();await queryNutritionReference();}
 }catch(error){toast(error.message);}
});
document.addEventListener('submit',async event=>{
 if(!['goalForm','preferencesForm','foodForm','saveForm'].includes(event.target.id))return;
 event.preventDefault();const submit=event.target.querySelector('button[type=submit]');submit.disabled=true;
 try{
  if(event.target.id==='goalForm'){const value=$('goalValue').value;await setGoal(value?Number(value):null,state.modal.onboard);}
  if(event.target.id==='preferencesForm'){
   state.profile=await api('/profile',{method:'PATCH',body:{display_name:$('displayName').value.trim(),timezone:$('timeZone').value,retain_media:$('retainMedia').checked}});
   closeModal();state.date=localDay();await refreshRecords();render(true);toast('偏好已保存');
  }
  if(event.target.id==='foodForm'){
   const name=$('foodName').value.trim(),key=$('foodCatalog').value || null;if(!name)throw new Error('请输入食物名称');
   const modal=state.modal;
   const latest=await api(`/meals/${state.meal.id}`);
   const body={name,catalog_key:key,revision:latest.revision};
   const densityMode=$('densityMode')?.value;
   if(densityMode==='clear')body.density={mode:'clear'};
   else if(densityMode==='custom'){const value=Number($('densityValue').value);if(!(value>0&&value<=10))throw new Error('请输入大于 0 且不超过 10 的密度，单位 g/mL');body.density={mode:'custom',value_g_ml:value,note:$('densityNote').value.trim()};}
   else if(densityMode?.startsWith('ref:'))body.density={mode:'reference',reference_id:densityMode.slice(4)};
   if(!modal.itemId&&modal.box&&modal.box[2]>modal.box[0]&&modal.box[3]>modal.box[1])body.region={kind:'bbox',bbox:modal.box};
   state.meal=await api(`/meals/${latest.id}/foods${modal.itemId?'/'+modal.itemId:''}`,{method:modal.itemId?'PATCH':'POST',body});
   state.selected=modal.itemId || state.meal.items.at(-1).id;closeModal();await refreshRecords();render(true);pollMeal();if(state.mode==='3d'){state.viewDirty=true;scheduleView();}toast(state.meal.items.some(nutritionPending)?'食物已添加，营养数据查询中':'食物清单与营养已更新');
  }
  if(event.target.id==='saveForm'){
   const body={title:$('mealTitle').value.trim(),meal_type:$('mealType').value,occurred_at:zonedISO($('mealTime').value),note:$('mealNote').value};
   if(!body.title)throw new Error('请输入餐食标题');
   const latest=await api(`/meals/${state.meal.id}`);body.revision=latest.revision;
   state.meal=await api(`/meals/${latest.id}/save`,{method:'POST',body});closeModal();state.date=localDay(new Date(state.meal.occurred_at));await go('diary');toast('已保存到饮食日记');
  }
 }catch(error){toast(error.message);submit.disabled=false;}
});
document.addEventListener('keydown',event=>{if(event.key==='Escape'&&state.modal)closeModal();});
window.addEventListener('beforeinstallprompt',event=>{event.preventDefault();state.installPrompt=event;});
window.addEventListener('pagehide',stopCamera);
window.addEventListener('popstate',()=>route().catch(error=>toast(error.message)));
async function route(){
 const [page,id]=location.hash.slice(1).split('/');
 if(['result','plate'].includes(page)&&id){await openMeal(id,page==='plate',false);return;}
 if(['home','scan','diary','profile'].includes(page))await go(page,false);
}
async function bootstrap(){
 try{
  const [profile,catalog]=await Promise.all([api('/profile'),api('/catalog')]);state.profile=profile;state.catalog=catalog.foods;state.categoryDensities=catalog.category_densities||[];state.date=localDay();
  await refreshRecords();state.offline=false;
  if(!profile.onboarded){state.page='onboard';render();}else{state.page='home';await route();render();}
 }catch(error){state.offline=true;render();toast(error.message);}
}
phoneScan.configure({api,redraw:()=>{if(state.page==='scan'&&state.captureFlow==='mobile')render(true);},openMeal});
if('serviceWorker' in navigator&&window.isSecureContext)navigator.serviceWorker.register('/sw.js').catch(()=>{});
await bootstrap();

document.addEventListener('input',event=>{
 if(event.target.id==='foodSearch')filterFoodCatalog();
 if(event.target.id==='foodName')refreshDensityFields();
 if(event.target.id==='densityValue')updateDensityPreview();
 if(['foodSearch','foodName'].includes(event.target.id)){
  nutritionReferenceEpoch++;clearTimeout(nutritionReferenceTimer);
  if($('nutritionSearchStatus'))$('nutritionSearchStatus').textContent='输入完成后，缺失的营养数据将自动查询。';
 }
});
document.addEventListener('focusout',event=>{
 if(state.modal?.kind!=='food')return;
 if(event.target.id==='foodName')queryNutritionReference();
 if(event.target.id==='foodSearch'&&!$('foodCatalog').querySelector('optgroup')&&event.target.value.trim()){
  $('foodName').value=event.target.value.trim();refreshDensityFields();queryNutritionReference();
 }
});
