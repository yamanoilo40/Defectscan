/* DefectScan — инференс нейросети прямо в браузере (ONNX Runtime Web).
   Идея: фото режется на перекрывающиеся фрагменты, каждый классифицируется моделью,
   из результатов строится тепловая карта и общий вердикт. */
'use strict';
const $ = s => document.querySelector(s);
const NAMES = { ok:'Норма', normal:'Норма', good:'Норма', corrosion:'Коррозия', rust:'Коррозия', crack:'Трещина',
  dent:'Вмятина', scratch:'Царапина', pitting:'Питтинг', coating:'Дефект покрытия', paint:'Дефект покрытия',
  weld:'Дефект сварки', leak:'Течь', deformation:'Деформация' };
const ruName = c => NAMES[c.toLowerCase()] || c;

let cfg = null, session = null, mock = false;
const state = { mode:null, el:null, last:null, thr:0.5, running:false, timer:null, stream:null, tl:[] };
const params = new URLSearchParams(location.search);

/* ---------- загрузка модели ---------- */
async function loadModel(){
  const badge = $('#modelBadge');
  try{
    if(params.has('selftest')){          // только для отладки интерфейса, НЕ настоящая модель
      cfg = {classes:['ok','corrosion'], normal:'ok', size:64, mean:[0,0,0], std:[1,1,1], grid:4};
      mock = true; badge.textContent = 'ТЕСТОВЫЙ РЕЖИМ (не ИИ)'; badge.className='badge err'; return;
    }
    cfg = await (await fetch('model/labels.json', {cache:'no-cache'})).json();
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create('model/model.onnx', {executionProviders:['wasm'], graphOptimizationLevel:'all'});
    badge.textContent = 'Модель готова · работает офлайн'; badge.className = 'badge ok';
  }catch(e){
    console.error(e);
    badge.textContent = 'Модель не найдена'; badge.className = 'badge err';
    $('#vText').textContent = 'Положите model.onnx в папку model/';
    cfg = null;
  }
}

/* ---------- подготовка и запуск ---------- */
function makeTiles(w,h,grid){
  const cell = Math.max(w,h)/grid, cols = Math.max(1,Math.ceil(w/cell)), rows = Math.max(1,Math.ceil(h/cell));
  const side = Math.min(cell*1.4, Math.min(w,h)), tiles = [];
  for(let r=0;r<rows;r++) for(let c=0;c<cols;c++){
    const cx=(c+.5)*cell, cy=(r+.5)*cell;
    const x = Math.min(Math.max(cx-side/2,0), w-side), y = Math.min(Math.max(cy-side/2,0), h-side);
    tiles.push({x,y,s:side,c,r});
  }
  return {tiles, cols, rows, cell};
}
const work = document.createElement('canvas'); const wctx = work.getContext('2d',{willReadFrequently:true});
function toTensorData(src, rects, S){
  work.width = work.height = S; const n = rects.length, out = new Float32Array(n*3*S*S), plane=S*S;
  const {mean,std} = cfg;
  rects.forEach((q,i)=>{
    wctx.drawImage(src, q.x,q.y,q.w,q.h, 0,0,S,S);
    const d = wctx.getImageData(0,0,S,S).data, o = i*3*plane;
    for(let p=0;p<plane;p++){
      out[o+p]         = (d[p*4]  /255-mean[0])/std[0];
      out[o+plane+p]   = (d[p*4+1]/255-mean[1])/std[1];
      out[o+2*plane+p] = (d[p*4+2]/255-mean[2])/std[2];
    }
  });
  return out;
}
function softmax(a){const m=Math.max(...a);const e=a.map(v=>Math.exp(v-m));const s=e.reduce((x,y)=>x+y,0);return e.map(v=>v/s)}
async function runBatch(src, rects){
  const S = cfg.size, C = cfg.classes.length;
  if(mock){            // тестовая заглушка: «ржавость» по красноватости пикселей. Не нейросеть!
    const d = toTensorData(src,rects,S), plane=S*S, res=[];
    for(let i=0;i<rects.length;i++){let r=0,g=0,b=0;for(let p=0;p<plane;p++){r+=d[i*3*plane+p];g+=d[i*3*plane+plane+p];b+=d[i*3*plane+2*plane+p]}
      const rust=Math.max(0,Math.min(1,((r-b)/plane-.0)*9)); res.push([1-rust,rust]);}
    return res;
  }
  const data = toTensorData(src,rects,S);
  const out = await session.run({[session.inputNames[0]]: new ort.Tensor('float32',data,[rects.length,3,S,S])});
  const logits = out[session.outputNames[0]].data, res=[];
  for(let i=0;i<rects.length;i++) res.push(softmax(Array.from(logits.slice(i*C,(i+1)*C))));
  return res;
}
async function analyze(src, w, h){
  const grid = cfg.grid || 5, {tiles,cols,rows} = makeTiles(w,h,grid);
  const rects = tiles.map(t=>({x:t.x,y:t.y,w:t.s,h:t.s}));
  const probs = await runBatch(src, rects);
  const ni = Math.max(0, cfg.classes.findIndex(c=>c===cfg.normal));
  tiles.forEach((t,i)=>{ t.p=probs[i]; t.def=1-probs[i][ni]; });
  return {tiles, cols, rows, w, h, ni};
}

/* ---------- итоги ---------- */
function summarize(R, thr){
  const bad = R.tiles.filter(t=>t.def>=thr), area = bad.length/R.tiles.length;
  const maxConf = Math.max(...R.tiles.map(t=>t.def));
  const sums = cfg.classes.map((_,k)=> k===R.ni?0:R.tiles.reduce((a,t)=>a+(t.def>=thr?t.p[k]:0),0));
  const tot = sums.reduce((a,b)=>a+b,0)||1;
  const classes = cfg.classes.map((c,k)=>({c,k,share:sums[k]/tot})).filter(x=>x.k!==R.ni).sort((a,b)=>b.share-a.share);
  let level='ok', text='Дефектов не обнаружено';
  if(bad.length){ level = area<0.15 ? 'warn':'bad'; text = area<0.15 ? 'Признаки дефекта: локально':'Обнаружен дефект'; if(classes[0]&&classes[0].share>0.4) text += ' — '+ruName(classes[0].c).toLowerCase(); }
  return {area, maxConf, classes, level, text, count:bad.length};
}
function renderResult(R){
  const S = summarize(R, state.thr);
  const v=$('#verdict'); v.className='verdict '+S.level; $('#vText').textContent=S.text;
  $('#mArea').textContent = Math.round(S.area*100)+'%';
  $('#mConf').textContent = Math.round(S.maxConf*100)+'%';
  const box=$('#classes'); box.innerHTML='';
  if(!S.count) box.innerHTML='<p class="muted">Подозрительных зон нет.</p>';
  S.classes.filter(x=>x.share>0.01).forEach(x=>{
    const d=document.createElement('div'); d.className='cls';
    d.innerHTML=`<div class="row"><span>${ruName(x.c)}</span><b>${Math.round(x.share*100)}%</b></div><div class="bar"><i style="width:${x.share*100}%"></i></div>`;
    box.appendChild(d);
  });
  drawOverlay(R,S); return S;
}

/* ---------- отрисовка ---------- */
const ov = $('#overlay'), octx = ov.getContext('2d');
function heatColor(v,thr){ // 0..1 -> цвет
  const t = Math.max(0,Math.min(1,(v-thr*0.6)/(1-thr*0.6)));
  const r=255, g=Math.round(190-150*t), b=Math.round(40-30*t); return `rgba(${r},${g},${b},${0.12+0.5*t})`;
}
function drawOverlay(R,S){
  ov.width=R.w; ov.height=R.h; octx.clearRect(0,0,R.w,R.h);
  if($('#chkHeat').checked){
    const hm=document.createElement('canvas'); hm.width=R.cols; hm.height=R.rows; const hc=hm.getContext('2d');
    R.tiles.forEach(t=>{ if(t.def>=state.thr*0.6){hc.fillStyle=heatColor(t.def,state.thr);hc.fillRect(t.c,t.r,1,1);} });
    octx.imageSmoothingEnabled=true; octx.imageSmoothingQuality='high';
    const cell=Math.max(R.w,R.h)/(cfg.grid||5);
    octx.drawImage(hm,0,0,R.cols,R.rows,0,0,R.cols*cell,R.rows*cell);
  }
  if($('#chkBox').checked){
    const cell=Math.max(R.w,R.h)/(cfg.grid||5), lw=Math.max(2,R.w/400);
    octx.lineWidth=lw; const fs=Math.round(Math.max(13,R.w/45)); octx.font=`600 ${fs}px -apple-system,sans-serif`; octx.textBaseline='top';
    R.tiles.filter(t=>t.def>=state.thr).forEach(t=>{
      const x=t.c*cell+lw, y=t.r*cell+lw, w=Math.min(cell,R.w-t.c*cell)-2*lw, h=Math.min(cell,R.h-t.r*cell)-2*lw;
      octx.strokeStyle= t.def>.8?'#ff5d5d':'#ffb020'; octx.strokeRect(x,y,w,h);
      let k=0,best=-1; t.p.forEach((p,i)=>{if(i!==R.ni&&p>best){best=p;k=i}});
      const label=`${ruName(cfg.classes[k])} ${Math.round(t.def*100)}%`, tw=octx.measureText(label).width+10, th=fs+8;
      octx.fillStyle=octx.strokeStyle; octx.fillRect(x,y,Math.min(tw,w),th); octx.fillStyle='#1b1200'; octx.fillText(label,x+5,y+4,w-8);
    });
  }
}

/* ---------- фото ---------- */
function show(mode){ $('#drop').hidden = !!mode; $('#stage').hidden = !mode;
  $('#img').hidden = mode!=='photo'; $('#vid').hidden = !(mode==='video'||mode==='live'); $('#timelineBox').hidden = !(mode==='video'||mode==='live'); state.mode=mode; }
function setFrame(w,h){ $('#frame').style.aspectRatio = `${w}/${h}`; }
function resetUI(){
  stopLoop(); if(state.stream){state.stream.getTracks().forEach(t=>t.stop());state.stream=null}
  const v=$('#vid'); v.pause(); v.removeAttribute('src'); v.srcObject=null; v.load();
  octx.clearRect(0,0,ov.width,ov.height); state.last=null; state.tl=[];
  $('#verdict').className='verdict idle'; $('#vText').textContent='Ожидание изображения';
  $('#mArea').textContent=$('#mConf').textContent='—'; $('#classes').innerHTML='<p class="muted">Пока нет данных.</p>'; show(null);
}
async function onPhoto(file){
  if(!file) return; if(!cfg){alert('Модель не загружена. Проверьте файлы в папке model/.');return;}
  resetUI(); show('photo'); $('#busy').hidden=false;
  const img=$('#img'); img.src=URL.createObjectURL(file); await img.decode().catch(()=>{});
  setFrame(img.naturalWidth,img.naturalHeight); state.el=img;
  try{ const R=await analyze(img,img.naturalWidth,img.naturalHeight); state.last=R; renderResult(R); }
  catch(e){ console.error(e); $('#vText').textContent='Ошибка анализа: '+e.message; }
  $('#busy').hidden=true;
}

/* ---------- видео и живая камера ---------- */
function stopLoop(){ state.running=false; clearTimeout(state.timer); }
async function loop(){
  if(!state.running) return; const v=$('#vid');
  if(v.readyState>=2 && !v.paused){
    const t0=performance.now();
    try{ const R=await analyze(v,v.videoWidth,v.videoHeight); state.last=R; const S=renderResult(R);
      state.tl.push({t:v.currentTime||state.tl.length*.5, a:S.area, c:S.maxConf}); if(state.tl.length>300) state.tl.shift(); drawTimeline(); }
    catch(e){console.error(e)}
    state.timer=setTimeout(loop, Math.max(100, 350-(performance.now()-t0)));
  } else state.timer=setTimeout(loop,200);
}
function startLoop(){ stopLoop(); state.running=true; loop(); }
function drawTimeline(){
  const c=$('#timeline'), w=c.width=c.clientWidth*2, h=c.height=140, g=c.getContext('2d'); g.clearRect(0,0,w,h);
  const d=state.tl; if(d.length<2) return; g.lineWidth=3; g.strokeStyle='#ffb020'; g.beginPath();
  d.forEach((p,i)=>{const x=i/(d.length-1)*w, y=h-8-p.a*(h-16); i?g.lineTo(x,y):g.moveTo(x,y)}); g.stroke();
  const mx=d.reduce((a,b)=>b.a>a.a?b:a,d[0]); $('#tlInfo').textContent=`Максимум поражения: ${Math.round(mx.a*100)}% площади (на ${mx.t.toFixed(1)} с)`;
}
async function onVideo(file){
  if(!file) return; if(!cfg){alert('Модель не загружена.');return;}
  resetUI(); show('video'); const v=$('#vid'); v.src=URL.createObjectURL(file);
  await new Promise(r=>v.onloadedmetadata=r); setFrame(v.videoWidth,v.videoHeight); state.el=v;
  v.play().catch(()=>{}); startLoop();
}
async function onLive(){
  if(!cfg){alert('Модель не загружена.');return;}
  try{
    resetUI(); show('live'); const v=$('#vid'); v.controls=false;
    state.stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1280}},audio:false});
    v.srcObject=state.stream; await v.play(); setFrame(v.videoWidth||4,v.videoHeight||3); state.el=v; startLoop();
  }catch(e){ resetUI(); alert('Нет доступа к камере: '+e.message+'\nИспользуйте «Сделать снимок».'); }
}

/* ---------- сохранение ---------- */
async function save(){
  if(!state.el||!state.last) return; const R=state.last, c=document.createElement('canvas'); c.width=R.w; c.height=R.h;
  const g=c.getContext('2d'); g.drawImage(state.el,0,0,R.w,R.h); g.drawImage(ov,0,0);
  const S=summarize(R,state.thr); g.fillStyle='rgba(11,15,20,.82)'; g.fillRect(0,0,R.w,Math.max(48,R.h/14));
  g.fillStyle='#fff'; g.font=`700 ${Math.max(18,R.h/28)}px -apple-system,sans-serif`; g.textBaseline='middle';
  g.fillText(`DefectScan · ${S.text} · площадь ${Math.round(S.area*100)}%`,16,Math.max(48,R.h/14)/2);
  c.toBlob(async b=>{ const f=new File([b],'defectscan.png',{type:'image/png'});
    if(navigator.canShare&&navigator.canShare({files:[f]})){ try{await navigator.share({files:[f]});return}catch(e){} }
    const a=document.createElement('a'); a.href=URL.createObjectURL(b); a.download='defectscan.png'; a.click(); },'image/png');
}

/* ---------- события ---------- */
$('#inCamera').onchange=e=>onPhoto(e.target.files[0]);
$('#inPhoto').onchange=e=>onPhoto(e.target.files[0]);
$('#inVideo').onchange=e=>onVideo(e.target.files[0]);
['inCamera','inPhoto','inVideo'].forEach(id=>$('#'+id).addEventListener('click',e=>e.target.value=''));
$('#btnLive').onclick=onLive; $('#btnNew').onclick=resetUI; $('#btnSave').onclick=save;
$('#thr').oninput=e=>{ state.thr=e.target.value/100; $('#thrV').textContent=e.target.value+'%'; if(state.last) renderResult(state.last); };
['#chkHeat','#chkBox'].forEach(s=>$(s).onchange=()=>state.last&&renderResult(state.last));
if('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(()=>{});
loadModel();
