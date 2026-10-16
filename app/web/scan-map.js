const escapeText=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function scanMapHTML(scan,selected=null){
 const report=scan.verification?.coverage||scan.coverage;
 const point=(r,a)=>{const t=(a-90)*Math.PI/180;return [140+r*Math.cos(t),140+r*Math.sin(t)];};
 const path=(r1,r2,a,b)=>`M${point(r2,a)}A${r2},${r2} 0 0 1 ${point(r2,b)}L${point(r1,b)}A${r1},${r1} 0 0 0 ${point(r1,a)}Z`;
 const labels={covered:'已有合格照片',partial:'只有一个有效视点，继续移动',missing:'尚未拍摄',retake:'画面不合格，请补拍',manual:'手动标注，未经几何测量'};
 const cells=report.cells.map(cell=>{
  const outer=cell.level===0?117:85,inner=outer-26,p=point((outer+inner)/2,cell.direction*30);
  return `<g data-scan-sector="${cell.index}" role="button" tabindex="0" aria-label="${cell.level?'较高俯视':'较低视角'}第 ${cell.direction+1} 格：${labels[cell.status]}" class="scan-sector ${cell.status}${cell.index===selected?' selected':''}"><path d="${path(inner,outer,cell.direction*30-13,cell.direction*30+13)}"/><text x="${p[0]}" y="${p[1]+4}" text-anchor="middle">${cell.status==='covered'?'✓':cell.status==='retake'?'!':cell.status==='partial'?'◔':cell.status==='manual'?'✎':'·'}</text></g>`;
 }).join('');
 const current=report.current;const marker=current?`<path class="scan-direction" d="M-7,-5 L7,0 L-7,5 Z" transform="translate(${point(current.sector<12?131:72,(current.sector%12)*30)}) rotate(${current.sector%12*30+90})"/>`:'';
 return `<div class="scan-coverage" aria-label="已拍与未拍角度"><div class="scan-coverage-heading"><strong>${scan.verification?'几何复核覆盖':'采集角度覆盖'}</strong><span>${report.covered+(report.manual||0)} / ${report.total} 方向${report.manual?`（${report.manual} 手动）`:''}</span></div><svg viewBox="0 0 280 280" role="group" aria-label="外圈较低视角，内圈较高俯视">${cells}<circle class="scan-plate-centre" cx="140" cy="140" r="47"/><text class="scan-centre-label" x="140" y="135" text-anchor="middle">餐盘中心</text><text class="scan-centre-label small" x="140" y="157" text-anchor="middle">外低 · 内高</text>${marker}</svg><div class="scan-legend"><span class="covered">✓ 已拍</span><span class="partial">◔ 待补</span><span class="missing">· 未拍</span><span class="manual">✎ 手动标注</span><span class="retake">! 重拍</span></div><p class="scan-pose-status">${scan.verification?'已用注册照片复核':!report.calibration.ready?'角度待确认：围绕定位板变换高度，采集至少 8 个不同视点':report.current?'已定位；每格需两个不同视点':'暂时未看到足够定位标记'}${report.pending_frames.length?` · ${report.pending_frames.length} 张待定位`:''}${report.manual?' · ✎ 为你手动选择的方向，不是测量结果':''}</p></div>`;
}
export function frameLabel(frame){return frame.quality==='blur'?'画面模糊':frame.quality==='exposure'?'曝光异常':frame.duplicate_view?'重复视点':frame.located?'已定位':frame.manual?'手动标注':'角度待确认';}
export function frameListHTML(scan,sector=null){
 const frames=(sector===null?scan.frames:scan.frames.filter(f=>f.sector===sector)).slice(-12).reverse();
 return frames.map(frame=>`<div class="scan-frame"><button type="button" data-scan-photo="${escapeText(frame.id)}" aria-label="查看照片，${frameLabel(frame)}"><img data-scan-frame-src="${escapeText(frame.url)}" alt="${frameLabel(frame)}" loading="lazy"><span>${frameLabel(frame)}</span></button><button type="button" data-scan-remove="${escapeText(frame.id)}" ${scan.meal_id?'disabled':''}>移除</button></div>`).join('')||'<p class="scan-note">这个方向还没有照片。</p>';
}
