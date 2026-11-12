"""把 Ozone shadow-debug CSV 导成多面板交互 HTML(离线,无 CDN,无外部依赖)。

与 plot_motion31_log_html.py 的区别:
  - 每个量纲组一个独立面板,直接用物理单位坐标(不做跨通道归一化叠加);
  - 不依赖 rl_active 列,也不重算固件补偿项,只做观测/动作/力矩的确定性换算;
  - 数据以原始 CSV 文本内嵌,前端解析并派生通道。

交互:滚轮以光标为中心缩放、拖动平移、双击面板体复位、双击标题栏加高、
      ←/→ 单帧步进(Shift 十倍)、左侧分组勾选、顶部预设、悬停十字线数值表。

用法:
  python scripts/plot_ozone_panels_html.py \
    --csv logs/Ozone_DataSampling_261101.csv \
    --out artifacts/2026_11_01__ozone_log_viewer/index.html
"""
import argparse
from pathlib import Path

TEMPLATE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ozone 日志数据查看 · __TITLE__</title>
<style>
:root{
  --bg:#0b0e13; --side:#0d1116; --card:#121821; --line:#222c39; --grid:#1a2230;
  --text:#e2e9f2; --muted:#8b9aac; --dim:#5d6a7a; --acc:#4a9eff;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{font:13px/1.45 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang SC","Microsoft YaHei",sans-serif;
  background:var(--bg);color:var(--text);overflow:hidden}
#wrap{display:flex;height:100%}
#side{width:268px;flex:0 0 268px;background:var(--side);border-right:1px solid var(--line);
  overflow:auto;padding:12px 12px 40px}
#side h1{font-size:13px;margin:0 0 3px;color:#8fb4d9;font-weight:600;letter-spacing:.3px}
#meta{color:var(--dim);font-size:11px;line-height:1.6;margin-bottom:10px;word-break:break-all}
#btns{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px}
button{background:#1a2330;color:#cfe0f2;border:1px solid #2a3644;border-radius:4px;
  padding:3px 8px;font-size:11px;cursor:pointer;font-family:inherit}
button:hover{background:#22303f;border-color:#3a4a5e}
button.on{background:#1d3550;border-color:#3d6b9e;color:#dcecfb}
.grp{margin:9px 0 3px;color:#7d8b9c;font-size:10px;letter-spacing:.6px;text-transform:uppercase}
label{display:flex;align-items:center;gap:6px;padding:1px 3px;border-radius:3px;cursor:pointer;white-space:nowrap}
label:hover{background:#161d26}
label input{margin:0;accent-color:var(--acc);flex:0 0 auto}
.sw{width:9px;height:9px;border-radius:2px;flex:0 0 9px}
.u{margin-left:auto;color:#55616f;font-size:10px}
#main{flex:1;min-width:0;overflow:auto;position:relative}
#panels{padding:10px 14px 60px}
.panel{position:relative;border:1px solid var(--line);border-radius:8px;background:var(--card);
  margin-bottom:8px;overflow:hidden}
.panel .hd{display:flex;align-items:center;gap:10px;padding:4px 10px;cursor:row-resize;
  background:linear-gradient(180deg,#151c26,#111720);border-bottom:1px solid var(--line);
  font-size:11px;color:var(--muted);user-select:none}
.panel .hd .ti{color:#a8bccf;font-weight:600;letter-spacing:.2px}
.panel .hd .lg{display:flex;gap:10px;flex-wrap:wrap;font-size:10.5px}
.panel .hd .lg i{display:inline-block;width:9px;height:2px;vertical-align:middle;margin-right:4px;border-radius:1px}
.panel .hd .st{margin-left:auto;color:#66737f;font-variant-numeric:tabular-nums;font-size:10.5px;white-space:nowrap}
.panel .cvwrap{position:relative;height:124px}
.panel.tall .cvwrap{height:280px}
.panel canvas{display:block;position:absolute;left:0;top:0;width:100%;height:100%}
.panel canvas.ov{cursor:crosshair}
#tip{position:fixed;pointer-events:none;display:none;z-index:20;background:rgba(8,12,18,.96);
  border:1px solid #2b3746;border-radius:5px;padding:6px 8px;font-size:11px;line-height:1.5;
  max-height:74vh;overflow:hidden;box-shadow:0 8px 24px rgba(0,0,0,.5)}
#tip .th{color:#8fb4d9;margin-bottom:3px;font-variant-numeric:tabular-nums}
#tip .row{white-space:nowrap;font-variant-numeric:tabular-nums}
#tip .row span.n{display:inline-block;min-width:74px;color:#9fb0c3}
#tip .row span.v{display:inline-block;min-width:70px;text-align:right}
#tip .row span.u{color:#55616f;padding-left:5px}
#tip .sec{color:#6b7a8b;margin-top:4px;font-size:10px;letter-spacing:.4px}
#empty{padding:30px;color:var(--dim);text-align:center}
#hint{position:fixed;right:14px;bottom:10px;color:#3f4a57;font-size:10.5px;pointer-events:none}
</style></head><body>
<div id="wrap">
  <div id="side">
    <h1>__TITLE__</h1>
    <div id="meta">__META__</div>
    <div id="btns"></div>
    <div id="list"></div>
  </div>
  <div id="main"><div id="panels"></div></div>
</div>
<div id="tip"></div>
<div id="hint">滚轮缩放 · 拖动平移 · 双击复位 · ←/→ 步进</div>
<script id="raw" type="text/csv">__CSV__</script>
<script>
"use strict";
const PAL=['#4a9eff','#ff7a5c','#3ddc97','#ffd166','#c792ea','#4dd0e1','#f06292','#aed581',
           '#ffab40','#90caf9','#ba68c8','#4db6ac','#e57373','#fff176','#81c784','#64b5f6',
           '#ff8a65','#9575cd','#4fc3f7','#a1887f'];

/* ---------- 解析 CSV ---------- */
const raw=document.getElementById('raw').textContent.replace(/^\uFEFF/,'').trim();
const lines=raw.split(/\r?\n/);
const head=lines[0].split(',');
const colOf=(name)=>{
  for(let j=0;j<head.length;j++){
    const h=head[j].replace('(g_rl_shadow_debug).','').trim();
    if(h===name) return j;
  }
  return -1;
};
const NROW=lines.length-1;
const T=new Float64Array(NROW);
const cols=new Map();
for(const name of head){
  const key=name.replace('(g_rl_shadow_debug).','').trim();
  cols.set(key,new Float64Array(NROW));
}
for(let i=0;i<NROW;i++){
  const p=lines[i+1].split(',');
  if(p.length!==head.length) continue;
  T[i]=parseFloat(p[colOf('Time')]);
  for(let j=0;j<head.length;j++){
    const key=head[j].replace('(g_rl_shadow_debug).','').trim();
    const v=parseFloat(p[j]);
    cols.get(key)[i]=Number.isFinite(v)?v:NaN;
  }
}
const g=(k)=>{const a=cols.get(k); if(!a) throw new Error('缺少列 '+k); return a;};
const has=(k)=>cols.has(k);
const D2R=Math.PI/180, R2D=180/Math.PI;

/* ---------- 派生通道定义 ---------- */
const OBS=(k)=>g('observation['+k+']');
const CONV={
  gyro:k=>{const a=OBS(k);return mul(a,4);},
  gravity:k=>OBS(k),
  pitch:()=>{const gx=OBS(3),gy=OBS(4),gz=OBS(5),o=new Float64Array(NROW);
    for(let i=0;i<NROW;i++) o[i]=Math.atan2(gz[i],-gy[i])*R2D; return o;},
  roll:()=>{const gx=OBS(3),gy=OBS(4),gz=OBS(5),o=new Float64Array(NROW);
    for(let i=0;i<NROW;i++) o[i]=Math.atan2(gx[i],-gy[i])*R2D; return o;},
  qleg:k=>mul(OBS(k),1.5*R2D),
  dq:k=>mul(OBS(k),20),
  pleg:k=>mul(OBS(k),1.5*R2D),
  pw:k=>mul(OBS(k),100),
  cmdvx:()=>mul(OBS(22),4),
  cmdyaw:()=>mul(OBS(23),4),
  cmdh:()=>mul(OBS(24),.5),
  act:k=>g('mean_action['+k+']'),
  lqrleg:k=>g('lqr_leg_torque_nm['+k+']'),
  rlleg:k=>g('rl_leg_torque_nm['+k+']'),
  lqrw:k=>g('lqr_wheel_torque_nm['+k+']'),
  rlw:k=>g('rl_wheel_torque_nm['+k+']')
};
function mul(a,s){const o=new Float64Array(a.length);for(let i=0;i<a.length;i++)o[i]=a[i]*s;return o;}

const LEGN=['RH','RA','LH','LA'];          // obs6:10 / action0..3
const CHN=['RH','RA','RW','LH','LA','LW']; // dq ch0..5
const TRQ4=['lf','lb','rf','rb'];          // 固件顺序,沿用旧工具标注

const CH=[]; let cid=0;
function add(group,label,unit,arr,opt){
  if(!arr) return;
  CH.push(Object.assign({id:'c'+(cid++),g:group,l:label,u:unit,v:arr,on:false},opt||{}));
}
/* 姿态 */
add('姿态 · 角速度','gyro x','rad/s',CONV.gyro(0));
add('姿态 · 角速度','gyro y','rad/s',CONV.gyro(1));
add('姿态 · 角速度','gyro z','rad/s',CONV.gyro(2));
add('姿态 · 倾角','pitch','deg',CONV.pitch());
add('姿态 · 倾角','roll','deg',CONV.roll());
add('姿态 · 重力分量','g x','-',CONV.gravity(3));
add('姿态 · 重力分量','g y','-',CONV.gravity(4));
add('姿态 · 重力分量','g z','-',CONV.gravity(5));
/* 关节 */
for(let k=0;k<4;k++) add('关节角','q '+LEGN[k],'deg',CONV.qleg(6+k));
for(let k=0;k<4;k++) add('关节角','prev '+LEGN[k],'deg',CONV.pleg(16+k),{dash:true});
for(let k=0;k<6;k++) add('关节速度','dq '+CHN[k],'rad/s',CONV.dq(10+k));
for(let k=0;k<2;k++) add('轮目标缓存','prev '+['RW','LW'][k],'rad/s',CONV.pw(20+k),{dash:true});
/* 命令 */
add('命令 · 速度','vx cmd','m/s',CONV.cmdvx());
add('命令 · 速度','yaw cmd','rad/s',CONV.cmdyaw());
add('命令 · 高度','height cmd','m',CONV.cmdh());
/* 动作 */
for(let k=0;k<6;k++) add('动作 a[0..5]','a'+k+' '+CHN[k],'-',CONV.act(k));
/* 力矩 */
for(let k=0;k<4;k++) add('腿力矩 · LQR','lqr '+TRQ4[k],'Nm',CONV.lqrleg(k));
for(let k=0;k<4;k++) add('腿力矩 · RL','rl '+TRQ4[k],'Nm',CONV.rlleg(k),{sat:40});
for(let k=0;k<2;k++) add('轮力矩 · LQR','lqr '+['L','R'][k],'Nm',CONV.lqrw(k));
for(let k=0;k<2;k++) add('轮力矩 · RL','rl '+['L','R'][k],'Nm',CONV.rlw(k));

CH.forEach((c,i)=>{c.color=PAL[i%PAL.length];});
const BYID=new Map(CH.map(c=>[c.id,c]));
const GROUPS=[]; CH.forEach(c=>{ if(!GROUPS.includes(c.g)) GROUPS.push(c.g); });
/* 每个面板的显示单位(混合量纲显式写清) */
const UNIT={
  '姿态 · 角速度':'rad/s','姿态 · 倾角':'deg','姿态 · 重力分量':'单位向量',
  '关节角':'deg','关节速度':'rad/s','轮目标缓存':'rad/s',
  '命令 · 速度':'m/s / rad/s','命令 · 高度':'m','动作 a[0..5]':'-',
  '腿力矩 · LQR':'Nm','腿力矩 · RL':'Nm','轮力矩 · LQR':'Nm','轮力矩 · RL':'Nm'
};

/* 统计一次(整段),用于标题最小值/最大值 */
CH.forEach(c=>{
  let mn=Infinity,mx=-Infinity,sm=0,n=0;
  for(let i=0;i<c.v.length;i++){const v=c.v[i];if(!Number.isFinite(v))continue;
    if(v<mn)mn=v; if(v>mx)mx=v; sm+=v; n++;}
  c.min=mn; c.max=mx; c.mean=n?sm/n:NaN;
});

/* ---------- 面板 ---------- */
const panelsEl=document.getElementById('panels');
const panels=[];
GROUPS.forEach(grp=>{
  const chs=CH.filter(c=>c.g===grp);
  const el=document.createElement('section');
  el.className='panel';
  el.innerHTML='<div class="hd"><span class="ti"></span><span class="lg"></span>'+
               '<span class="st"></span></div>'+
               '<div class="cvwrap"><canvas class="base"></canvas><canvas class="ov"></canvas></div>';
  panelsEl.appendChild(el);
  const cvB=el.querySelector('canvas.base'), cvO=el.querySelector('canvas.ov');
  const p={g:grp, el, cvB, cvO, ctxB:cvB.getContext('2d'), ctxO:cvO.getContext('2d'), chs,
           unit:UNIT[grp]||'-', vis:false};
  panels.push(p);
  el.addEventListener('dblclick',(e)=>{
    if(e.target.closest('.hd')) el.classList.toggle('tall');
    else { view.t0=T[0]; view.t1=T[NROW-1]; renderAll(); }
  });
});

/* ---------- 视图状态 ---------- */
const view={t0:T[0],t1:T[NROW-1],cur:-1};
let robust=false;   /* Y 轴稳健模式:0.5%/99.5% 分位,避免单点冲击压扁曲线 */
const L={left:52,right:12,top:8,bottom:16};
const PAD_X=L.left, PAD_R=L.right;

function sizeCanvas(cv,ctx){
  const r=window.devicePixelRatio||1;
  const w=cv.clientWidth||1, h=cv.clientHeight||1;
  if(cv.width!==Math.round(w*r)||cv.height!==Math.round(h*r)){
    cv.width=Math.round(w*r); cv.height=Math.round(h*r);
  }
  ctx.setTransform(r,0,0,r,0,0);
  return [w,h];
}
function px(t,w){ return PAD_X+(t-view.t0)/(view.t1-view.t0)*(w-PAD_X-PAD_R); }
function tx(x,w){ return view.t0+(x-PAD_X)/(w-PAD_X-PAD_R)*(view.t1-view.t0); }

function idxRange(){
  let a=0,b=NROW-1;
  while(a<NROW-1&&T[a]<view.t0) a++;
  while(b>0&&T[b]>view.t1) b--;
  return [Math.max(0,a-1),Math.min(NROW-1,b+1)];
}

/* ---------- 静态绘制 ---------- */
function drawPanel(p){
  const [w,h]=sizeCanvas(p.cvB,p.ctxB);
  const ctx=p.ctxB;
  ctx.clearRect(0,0,w,h);
  const x0=PAD_X,x1=w-PAD_R,y0=L.top,y1=h-L.bottom;
  const [ia,ib]=idxRange();
  const on=p.chs.filter(c=>c.on);

  if(!on.length){ p.vis=false; p.el.style.display='none'; return; }
  p.vis=true; p.el.style.display='';
  const isLast=(panels.lastIndexOf(p)===lastVisIdx);

  /* y 范围:默认全量,稳健模式取 0.5%/99.5% 分位 */
  let mn=Infinity,mx=-Infinity;
  const pool=robust?[]:null;
  for(const c of on) for(let i=ia;i<=ib;i++){const v=c.v[i];
    if(!Number.isFinite(v))continue;
    if(v<mn)mn=v; if(v>mx)mx=v; if(pool) pool.push(v);}
  if(!Number.isFinite(mn)){mn=-1;mx=1;}
  if(pool&&pool.length>20){
    pool.sort((a,b)=>a-b);
    const lo=pool[Math.floor(pool.length*.005)], hi=pool[Math.min(pool.length-1,Math.ceil(pool.length*.995))];
    if(hi>lo){ mn=lo; mx=hi; }
  }
  p.full=[mn,mx];
  if(mx-mn<1e-9){ const d=Math.max(1e-9,Math.abs(mx)*.1||1e-3); mn-=d; mx+=d; }
  const pad=(mx-mn)*.10; mn-=pad; mx+=pad;
  p.mn=mn; p.mx=mx;

  const Y=(v)=>y1-(v-mn)/(mx-mn)*(y1-y0);

  /* 网格 */
  ctx.strokeStyle='#1a2230'; ctx.lineWidth=1;
  for(let k=0;k<=3;k++){
    const y=Math.round(y0+k*(y1-y0)/3)+.5;
    ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke();
    ctx.fillStyle='#4d5a68'; ctx.font='10px ui-monospace,monospace'; ctx.textAlign='right';
    ctx.fillText(fmt(mx-(mx-mn)*k/3), x0-7, y+3);
  }
  /* 0 线 */
  if(mn<0&&mx>0){
    const y=Math.round(Y(0))+.5;
    ctx.strokeStyle='#2c3a4a'; ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke();
  }
  /* 饱和线 */
  const sat=[...new Set(on.filter(c=>c.sat).map(c=>c.sat))];
  sat.forEach(s=>{
    [-s,s].forEach(v=>{
      if(v<mn||v>mx) return;
      const y=Math.round(Y(v))+.5;
      ctx.save(); ctx.setLineDash([4,4]); ctx.strokeStyle='rgba(255,93,93,.45)';
      ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke(); ctx.restore();
    });
  });

  /* 曲线 */
  ctx.save(); ctx.beginPath(); ctx.rect(x0,y0,x1-x0,y1-y0); ctx.clip();
  const visN=Math.max(1,ib-ia+1), Wpx=Math.max(1,x1-x0);
  const envelope=visN/Wpx>1.6;
  for(const c of on){
    ctx.strokeStyle=c.color; ctx.lineWidth=1.15;
    ctx.setLineDash(c.dash?[5,4]:[]);
    if(envelope){
      const nb=Math.ceil(Wpx);
      ctx.beginPath();
      const step=visN/nb;
      for(let b=0;b<nb;b++){
        const s=ia+Math.floor(b*step), e=Math.min(ib,ia+Math.floor((b+1)*step)-1);
        let lo=Infinity,hi=-Infinity;
        for(let i=s;i<=e;i++){const v=c.v[i];if(!Number.isFinite(v))continue;
          if(v<lo)lo=v; if(v>hi)hi=v;}
        if(!Number.isFinite(lo)) continue;
        const x=x0+(b+.5)*(Wpx/nb);
        if(hi-lo<1e-12){ const yy=Y(lo); ctx.moveTo(x,yy); ctx.lineTo(x+.6,yy); }
        else { ctx.moveTo(x,Y(lo)); ctx.lineTo(x,Y(hi)); }
      }
      ctx.stroke();
    }else{
      ctx.beginPath(); let started=false;
      for(let i=ia;i<=ib;i++){
        const v=c.v[i];
        if(!Number.isFinite(v)){ started=false; continue; }
        const xx=px(T[i],w), yy=Y(v);
        if(!started){ ctx.moveTo(xx,yy); started=true; } else ctx.lineTo(xx,yy);
      }
      ctx.stroke();
    }
  }
  ctx.setLineDash([]);
  ctx.restore();

  /* 边框 + 时间刻度 */
  ctx.strokeStyle='#222c39';
  ctx.strokeRect(x0+.5,y0+.5,x1-x0,y1-y0);
  const span=view.t1-view.t0;
  let step=Math.pow(10,Math.floor(Math.log10(span/8)));
  for(const m of [1,2,5,10]){ if(span/(step*m)>=5){ step*=m; break; } }
  ctx.fillStyle='#4d5a68'; ctx.font='10px ui-monospace,monospace'; ctx.textAlign='center';
  const tS=Math.ceil(view.t0/step)*step;
  for(let t=tS;t<=view.t1;t+=step){
    const x=Math.round(px(t,w))+.5;
    ctx.strokeStyle='#161d27';
    ctx.beginPath(); ctx.moveTo(x,y0); ctx.lineTo(x,y1); ctx.stroke();
    if(isLast) ctx.fillText(t.toFixed(step<1?2:(step<10?1:0))+'s',x,y1+12);
  }

  /* 标题与图例 */
  const hd=p.el.querySelector('.hd');
  hd.querySelector('.ti').textContent=p.g;
  const lg=hd.querySelector('.lg');
  lg.innerHTML=on.map(c=>'<span><i style="background:'+c.color+'"></i>'+c.l+'</span>').join('');
  let satN=0;
  for(const c of on) if(c.sat) for(let i=ia;i<=ib;i++) if(Math.abs(c.v[i])>=c.sat-1e-6) satN++;
  hd.querySelector('.st').textContent =
    'y '+(robust?'稳健 ':'')+fmt(mn)+' … '+fmt(mx)+' '+p.unit+
    (satN?'   ｜ 饱和 '+satN:(sat? '   ｜ 饱和 0':''));
  sizeCanvas(p.cvO,p.ctxO);
  drawOverlay();
}

function fmt(v){
  if(!Number.isFinite(v)) return '—';
  const a=Math.abs(v);
  if(a>=1000) return v.toFixed(0);
  if(a>=100) return v.toFixed(1);
  if(a>=10) return v.toFixed(2);
  if(a>=1) return v.toFixed(3);
  if(a>=.01) return v.toFixed(4);
  return v.toExponential(1);
}

/* ---------- 光标层 ---------- */
function drawOverlay(){
  for(const p of panels){
    const ctx=p.ctxO, w=p.cvO.clientWidth||1, h=p.cvO.clientHeight||1;
    ctx.clearRect(0,0,w,h);
    if(!p.vis||view.cur<0) continue;
    const t=T[view.cur];
    if(t<view.t0||t>view.t1) continue;
    const x=px(t,w);
    if(x<PAD_X-1||x>w-PAD_R+1) continue;
    ctx.strokeStyle='rgba(255,255,255,.30)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(Math.round(x)+.5,L.top); ctx.lineTo(Math.round(x)+.5,h-L.bottom); ctx.stroke();
    const on=p.chs.filter(c=>c.on);
    for(const c of on){
      const v=c.v[view.cur]; if(!Number.isFinite(v)) continue;
      if(v<p.mn||v>p.mx) continue;
      const y=h-L.bottom-(v-p.mn)/(p.mx-p.mn)*(h-L.bottom-L.top);
      ctx.fillStyle=c.color;
      ctx.beginPath(); ctx.arc(x,y,2.4,0,6.2832); ctx.fill();
    }
  }
}

/* ---------- 提示框 ---------- */
const tip=document.getElementById('tip');
function showTip(){
  const i=view.cur;
  if(i<0||i>=NROW){ tip.style.display='none'; return; }
  const on=CH.filter(c=>c.on);
  if(!on.length){ tip.style.display='none'; return; }
  let html='<div class="th">t = '+T[i].toFixed(3)+' s ｜ #'+i+'</div>';
  let lastG='', shown=0;
  for(const c of on){
    if(shown>=22){ html+='<div class="sec">… 其余 '+(on.length-22)+' 项未显示</div>'; break; }
    if(c.g!==lastG){ html+='<div class="sec">'+c.g+'</div>'; lastG=c.g; }
    const v=c.v[i];
    html+='<div class="row"><span class="sw" style="background:'+c.color+'"></span>'+
          '<span class="n">'+c.l+'</span><span class="v">'+
          (Number.isFinite(v)?fmt(v):'—')+'</span><span class="u">'+c.u+'</span></div>';
    shown++;
  }
  tip.innerHTML=html;
  tip.style.display='block';
  const mx=lastMouse.x, my=lastMouse.y;
  const bw=tip.offsetWidth, bh=tip.offsetHeight;
  let left=mx+16, top=my+14;
  if(left+bw>window.innerWidth-8) left=mx-bw-16;
  if(top+bh>window.innerHeight-8) top=Math.max(8,window.innerHeight-bh-8);
  tip.style.left=left+'px'; tip.style.top=top+'px';
}

const lastMouse={x:0,y:0};
function selectAt(clientX){
  const p=panels.find(q=>q.vis)||panels[0];
  const w=p.cvO.clientWidth||1;
  const rect=p.cvO.getBoundingClientRect();
  const x=Math.min(Math.max(clientX-rect.left,PAD_X),w-PAD_R);
  const t=tx(x,w);
  let i=0,best=Infinity;
  const [ia,ib]=idxRange();
  for(let k=ia;k<=ib;k++){ const d=Math.abs(T[k]-t); if(d<best){best=d;i=k;} }
  view.cur=i;
  drawOverlay(); showTip();
}

/* ---------- 交互 ---------- */
let drag=null;
panels.forEach(p=>{
  const cv=p.cvO;
  cv.addEventListener('mousemove',e=>{
    lastMouse.x=e.clientX; lastMouse.y=e.clientY;
    if(drag){
      const w=cv.clientWidth;
      const dt=(e.clientX-drag.x)/(w-PAD_X-PAD_R)*(drag.t1-drag.t0);
      view.t0=drag.t0-dt; view.t1=drag.t1-dt; clampView(); renderStatic(); return;
    }
    selectAt(e.clientX);
  });
  cv.addEventListener('mousedown',e=>{
    drag={x:e.clientX,t0:view.t0,t1:view.t1}; cv.style.cursor='grabbing'; e.preventDefault();
  });
  cv.addEventListener('mouseleave',()=>{ if(!drag){ view.cur=-1; drawOverlay(); tip.style.display='none'; } });
  cv.addEventListener('wheel',e=>{
    e.preventDefault();
    const w=cv.clientWidth, rect=cv.getBoundingClientRect();
    const x=Math.min(Math.max(e.clientX-rect.left,PAD_X),w-PAD_R);
    const anchor=tx(x,w);
    const f=e.deltaY>0?1.18:1/1.18;
    view.t0=anchor+(view.t0-anchor)*f; view.t1=anchor+(view.t1-anchor)*f;
    clampView(); renderStatic();
  },{passive:false});
});
window.addEventListener('mouseup',()=>{
  if(drag){ drag=null; panels.forEach(p=>p.cvO.style.cursor='crosshair'); }
});
window.addEventListener('keydown',e=>{
  if(e.key!=='ArrowLeft'&&e.key!=='ArrowRight') return;
  const step=e.shiftKey?10:1;
  let i=(view.cur<0?0:view.cur)+(e.key==='ArrowRight'?step:-step);
  i=Math.min(NROW-1,Math.max(0,i));
  view.cur=i;
  const t=T[i], span=view.t1-view.t0;
  if(t<view.t0||t>view.t1){ view.t0=t-span*.5; view.t1=t+span*.5; clampView(); renderStatic(); }
  drawOverlay(); showTip(); e.preventDefault();
});

function clampView(){
  const minSpan=0.02;
  if(view.t1-view.t0<minSpan){ const c=(view.t0+view.t1)/2; view.t0=c-minSpan/2; view.t1=c+minSpan/2; }
  if(view.t0<T[0]){ const d=T[0]-view.t0; view.t0+=d; view.t1+=d; }
  if(view.t1>T[NROW-1]){ const d=view.t1-T[NROW-1]; view.t1-=d; view.t0-=d; }
  if(view.t0<T[0]) view.t0=T[0];
}

let lastVisIdx=-1;
function renderStatic(){
  lastVisIdx=-1;
  panels.forEach((p,i)=>{ if(p.chs.some(c=>c.on)) lastVisIdx=i; });
  panels.forEach(drawPanel); drawOverlay();
}
function renderAll(){ renderStatic(); if(view.cur>=0) showTip(); }

/* ---------- 侧栏 ---------- */
const list=document.getElementById('list');
GROUPS.forEach(g=>{
  const h=document.createElement('div'); h.className='grp'; h.textContent=g; list.appendChild(h);
  CH.filter(c=>c.g===g).forEach(c=>{
    const lb=document.createElement('label');
    const cb=document.createElement('input'); cb.type='checkbox'; cb.checked=c.on;
    cb.addEventListener('change',()=>{ c.on=cb.checked; syncGroupButtons(); renderStatic(); });
    const sw=document.createElement('span'); sw.className='sw'; sw.style.background=c.color;
    const nm=document.createElement('span'); nm.textContent=c.l;
    const u=document.createElement('span'); u.className='u'; u.textContent=c.u;
    lb.append(cb,sw,nm,u); list.appendChild(lb);
    c.cb=cb;
  });
});
function setOnly(ids){
  const s=new Set(ids);
  CH.forEach(c=>{ c.on=s.has(c.id); if(c.cb) c.cb.checked=c.on; });
  syncGroupButtons(); renderStatic();
}
function setGroup(list_,on){
  list_.forEach(g=>CH.filter(c=>c.g===g).forEach(c=>{c.on=on; if(c.cb)c.cb.checked=on;}));
  syncGroupButtons(); renderStatic();
}
const PRESETS=[
  ['概览',()=>{const ids=[];
    ['关节角','动作 a[0..5]','姿态 · 倾角','命令 · 速度'].forEach(g=>{
      if(g==='关节角'){ CH.filter(c=>c.g==='关节角'&&!c.dash).forEach(c=>ids.push(c.id)); }
      else CH.filter(c=>c.g===g).forEach(c=>ids.push(c.id));
    });
    CH.filter(c=>c.g==='关节速度'&&/dq (RH|RA|LH|LA)/.test(c.l)).forEach(c=>ids.push(c.id));
    setOnly(ids);}],
  ['姿态',()=>setOnly(CH.filter(c=>c.g.startsWith('姿态')).map(c=>c.id))],
  ['关节角+目标',()=>setOnly(CH.filter(c=>c.g==='关节角').map(c=>c.id))],
  ['关节速度',()=>setOnly(CH.filter(c=>c.g==='关节速度').map(c=>c.id))],
  ['动作',()=>setOnly(CH.filter(c=>c.g==='动作 a[0..5]').map(c=>c.id))],
  ['力矩对比',()=>setOnly(CH.filter(c=>c.g.startsWith('腿力矩')||c.g.startsWith('轮力矩')).map(c=>c.id))],
  ['全部',()=>setOnly(CH.map(c=>c.id))],
  ['全时段',()=>{view.t0=T[0];view.t1=T[NROW-1];renderStatic();}]
];
const btnBox=document.getElementById('btns');
let activePreset='';
PRESETS.forEach(([name,fn])=>{
  const b=document.createElement('button'); b.textContent=name;
  b.addEventListener('click',()=>{ if(name!=='全时段') activePreset=name; fn(); syncGroupButtons(); });
  btnBox.appendChild(b); b.dataset.preset=name;
});
function syncGroupButtons(){
  [...btnBox.querySelectorAll('button')].forEach(b=>{
    if(b.dataset.preset==='全时段'){ b.classList.remove('on'); return; }
    if(!b.dataset.preset) return;
    b.classList.toggle('on', b.dataset.preset===activePreset);
  });
}
/* Y 轴稳健模式开关 */
(function(){
  const b=document.createElement('button'); b.textContent='Y 稳健';
  b.title='Y 轴取 0.5%/99.5% 分位,抑制单点冲击;超出部分被裁到边框';
  b.addEventListener('click',()=>{ robust=!robust; b.classList.toggle('on',robust); renderStatic(); });
  btnBox.appendChild(b);
})();
/* 首帧定位(便于从 t=0 逐帧步进) */
(function(){
  const b=document.createElement('button'); b.textContent='回到起点';
  b.title='把窗口回到日志起点;之后可用 ← / → 逐帧(Shift 十倍)';
  b.addEventListener('click',()=>{
    const span=Math.min(view.t1-view.t0,T[NROW-1]-T[0]);
    view.t0=T[0]; view.t1=Math.min(T[NROW-1],T[0]+span);
    view.cur=0; clampView(); renderStatic(); showTip();
  });
  btnBox.appendChild(b);
})();

window.addEventListener('resize',()=>{ renderStatic(); });

/* 初始:URL hash 可指定预设(#力矩对比),否则概览 */
const hashName=decodeURIComponent((location.hash||'').replace(/^#/,''));
const initPreset=PRESETS.find(p=>p[0]===hashName);
if(initPreset&&initPreset[0]!=='全时段') activePreset=initPreset[0];
(initPreset||PRESETS[0])[1]();
syncGroupButtons();
renderStatic();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    text = args.csv.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()
    if "</script" in text:
        raise SystemExit("CSV 含 </script 字样,不能直接内嵌")
    rows = text.split("\n")
    n = len(rows) - 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    html = (TEMPLATE.replace("__CSV__", text)
            .replace("__TITLE__", args.csv.name)
            .replace("__META__", f"{n} 帧 · {len(rows[0].split(','))} 列<br>{args.csv.name}"))
    args.out.write_text(html, encoding="utf-8")
    print("写入 %s  (%.1f KB, %d 帧)" % (args.out, args.out.stat().st_size / 1024, n))


if __name__ == "__main__":
    main()
