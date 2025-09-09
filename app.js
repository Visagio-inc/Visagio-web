// === Element references ===
const els = {
  preview: document.getElementById('preview'),
  overlay: document.getElementById('overlay'),
  detectCount: document.getElementById('detectCount'),
  symVal: document.getElementById('symVal'),
  skinVal: document.getElementById('skinVal'),
  gauge: document.getElementById('gauge'),
  scoreNum: document.getElementById('scoreNum'),
  symmetryScore: document.getElementById('symmetryScore'),
  propScore: document.getElementById('propScore'),
  featScore: document.getElementById('featScore'),
  skinScore: document.getElementById('skinScore'),
  bars: {
    sym: document.getElementById('barSym'),
    prop: document.getElementById('barProp'),
    feat: document.getElementById('barFeat'),
    skin: document.getElementById('barSkin')
  },
  tips: document.getElementById('tipsList'),
  modelStatus: document.getElementById('modelStatus'),
  file: document.getElementById('file')
};

// Footer year
document.getElementById('year').textContent = new Date().getFullYear();

// === Load models ===
async function loadModels() {
  try {
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
      faceapi.nets.faceLandmark68Net.loadFromUri('/models')
    ]);
    els.modelStatus.innerHTML = '<span class="dot-blue"></span> Models ready';
  } catch (e) {
    els.modelStatus.innerHTML = 'Using CDN models…';
    const cdn = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri(cdn),
      faceapi.nets.faceLandmark68Net.loadFromUri(cdn)
    ]);
    els.modelStatus.innerHTML = '<span class="dot-blue"></span> Models ready (CDN)';
  }
}
loadModels();

// === Drag & drop & inputs ===
const dropzone = document.getElementById('dropzone');
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', e => {
  e.preventDefault(); dropzone.classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file) loadImage(file);
});
els.file.addEventListener('change', e => {
  const f = e.target.files[0];
  if (f) loadImage(f);
});
document.getElementById('sample1').addEventListener('click', () => demo('assets/demo1.jpg'));
document.getElementById('sample2').addEventListener('click', () => demo('assets/demo2.jpg'));
document.getElementById('clear').addEventListener('click', () => { els.preview.src = ''; clearCanvas(); resetUI(); });

function demo(path){ els.preview.src = path; els.preview.onload = () => analyze(); }
function loadImage(file){ const r = new FileReader(); r.onload = e => { els.preview.src = e.target.result; }; r.readAsDataURL(file); els.preview.onload = () => analyze(); }
function clearCanvas(){ const c = els.overlay; const ctx = c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height); }
function resetUI(){ setGauge(0); els.scoreNum.textContent = '—'; els.detectCount.textContent = '0'; ['sym','prop','feat','skin'].forEach(k=>{ els.bars[k].style.width = '0%'; document.getElementById(k+"Score").textContent = '—'; }); els.tips.innerHTML = ''; }
function setGauge(val){ const v = Math.max(0, Math.min(100, Math.round(val))); els.gauge.style.setProperty('--val', v); els.scoreNum.textContent = v; }
function dist(a,b){ return Math.hypot(a.x-b.x, a.y-b.y); }
function avgPt(pts){ const x = pts.reduce((s,a)=>s+a.x,0)/pts.length, y=pts.reduce((s,a)=>s+a.y,0)/pts.length; return {x,y}; }

// === Main analysis ===
async function analyze(){
  if (!els.preview.complete || !els.preview.src) return;
  const rect = els.preview.getBoundingClientRect();
  els.overlay.width = rect.width; els.overlay.height = rect.height;

  const detection = await faceapi.detectSingleFace(els.preview).withFaceLandmarks();
  clearCanvas();
  if(!detection){ els.detectCount.textContent = '0'; resetUI(); return; }

  els.detectCount.textContent = '1';
  const resized = faceapi.resizeResults(detection, { width: rect.width, height: rect.height });
  faceapi.draw.drawFaceLandmarks(els.overlay, resized);

  const pts = resized.landmarks.positions;
  const idx = n => pts[n];

  // Face measurements
  const faceWidth = dist(idx(0), idx(16));
  const faceHeight = dist(idx(8), idx(27));
  const leftEyeCenter = avgPt([pts[36],pts[37],pts[38],pts[39]]);
  const rightEyeCenter = avgPt([pts[42],pts[43],pts[44],pts[45]]);
  const interocular = dist(leftEyeCenter, rightEyeCenter);

  // Symmetry
  const mid = idx(27);
  const pairs = [ [36,45], [39,42], [31,35], [48,54], [3,13], [5,11] ];
  let symErrors = 0;
  pairs.forEach(([l,r]) => {
    const dl = dist(idx(l), mid), dr = dist(idx(r), mid);
    symErrors += Math.abs(dl - dr);
  });
  const symNorm = 1 - Math.min(1, (symErrors / pairs.length) / (faceWidth*0.5));
  const symmetryScore = Math.round(symNorm*100);

  // Proportions
  function ratio(v,target,tol=0.1){ const d=Math.abs(v-target), t=tol*target; return Math.max(0,1-d/t); }
  const faceRatio = faceHeight/faceWidth;
  const r_face = ratio(faceRatio,1.0,0.25);
  const eyeSpacing = interocular/faceWidth;
  const r_eye = ratio(eyeSpacing,0.46,0.26);
  const propScore = Math.round(((r_face+r_eye)/2)*100);

  // Features
  const noseWidth = dist(idx(31), idx(35))/faceWidth;
  const mouthWidth = dist(idx(48), idx(54))/faceWidth;
  const eyeOpenL = (dist(idx(37),idx(41))+dist(idx(38),idx(40)))/(2*dist(idx(36),idx(39)));
  const eyeOpenR = (dist(idx(43),idx(47))+dist(idx(44),idx(46)))/(2*dist(idx(42),idx(45)));
  const r_nose = ratio(noseWidth,0.22,0.4);
  const r_mouth = ratio(mouthWidth,0.34,0.35);
  const r_eyeOpen = (ratio(eyeOpenL,0.3,0.7)+ratio(eyeOpenR,0.3,0.7))/2;
  const featScore = Math.round(((r_nose+r_mouth+r_eyeOpen)/3)*100);

  // Skin evenness
  const box = resized.detection.box;
  const skinVar = brightnessVariance(els.preview, box);
  const skinScore = Math.max(0, Math.min(100, Math.round(100-500*skinVar)));

  // Weighted total
  const overall = Math.round(symmetryScore*0.30 + propScore*0.30 + featScore*0.20 + skinScore*0.20);

  // Update UI
  setGauge(overall);
  updateBar('sym', symmetryScore);
  updateBar('prop', propScore);
  updateBar('feat', featScore);
  updateBar('skin', skinScore);
  els.skinVal.textContent = skinScore;
  renderTips({symmetryScore,propScore,featScore,skinScore},{faceRatio,eyeSpacing,noseWidth,mouthWidth,eyeOpenL,eyeOpenR});

  // Centerline
  const ctx2 = els.overlay.getContext('2d');
  ctx2.strokeStyle = '#13a8ff'; ctx2.lineWidth = 2; ctx2.setLineDash([6,4]);
  ctx2.beginPath(); ctx2.moveTo(mid.x,0); ctx2.lineTo(mid.x,rect.height); ctx2.stroke(); ctx2.setLineDash([]);

  function updateBar(k,v){ els.bars[k].style.width = v+'%'; document.getElementById(k+'Score').textContent=v; }
}
function brightnessVariance(img, box){
  const c=document.createElement('canvas'); const ctx=c.getContext('2d');
  c.width=box.width; c.height=box.height;
  ctx.drawImage(img,box.x,box.y,box.width,box.height,0,0,box.width,box.height);
  const data=ctx.getImageData(0,0,c.width,c.height).data;
  let sum=0,sum2=0,n=0;
  for(let i=0;i<data.length;i+=4){ const y=0.2126*data[i]+0.7152*data[i+1]+0.0722*data[i+2]; sum+=y; sum2+=y*y; n++; }
  const mean=sum/n; return (sum2/n - mean*mean)/(255*255);
}

// === Tips ===
function renderTips(scores, ratios){
  const tips=[];
  if (ratios.faceRatio>1.15) tips.push({t:'Balance a longer face',p:'Add width with hairstyle; avoid tall styles.'});
  if (ratios.faceRatio<0.85) tips.push({t:'Elongate a shorter face',p:'Add height on top, tighter sides.'});
  if (ratios.eyeSpacing<0.38) tips.push({t:'Broaden close-set eyes',p:'Open up inner corners; adjust brows.'});
  if (ratios.eyeSpacing>0.54) tips.push({t:'Balance wide-set eyes',p:'Define inner corners, closer brows.'});
  if (ratios.noseWidth>0.30) tips.push({t:'Balance a wider nose',p:'Glasses with bold bridge help.'});
  if (ratios.mouthWidth<0.26) tips.push({t:'Enhance lip presence',p:'Hydration + slight tint; smile in photos.'});
  if (scores.symmetryScore<70) tips.push({t:'Photo posture for symmetry',p:'Keep face straight to lens; even lighting.'});
  if (scores.skinScore<75) tips.push({t:'Even out skin tone',p:'Gentle routine: cleanse, moisturize, SPF.'});
  tips.push({t:'Mindset',p:'Confidence & kindness matter more than millimeters.'});
  els.tips.innerHTML=''; tips.slice(0,6).forEach(({t,p})=>{ const d=document.createElement('div'); d.className='tip'; d.innerHTML=`<h4>${t}</h4><p>${p}</p>`; els.tips.appendChild(d); });
}

    div.innerHTML = `<h4>${t}</h4><p>${p}</p>`;
    els.tips.appendChild(div);
  });
}
