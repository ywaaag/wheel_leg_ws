"""把 Ozone 日志导成单文件 HTML 交互图(离线可用,无 CDN)。

功能:滚轮缩放(以光标为中心)、拖动平移、双击复位、左侧分组勾选通道、
      光标十字线 + 数值表、RL 接管窗口高亮、归一化/统一坐标系切换。

用法:
  python scripts/plot_motion31_log_html.py \
    --csv logs/Ozone_DataSampling_261028.csv \
    --out artifacts/2026_10_28__ozone_log_viewer/index.html
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HIP = math.radians(90.0 - 47.1033888)
ACT = math.radians(90.0 + 47.1080837)
L1, L2, L3, L4 = .210, .250, .250, .210


def leg_length(q_rh, q_ra, q_lh, q_la):
    def fk(p1, p4):
        xb, yb = L1 * math.cos(p1), L1 * math.sin(p1)
        xd, yd = L4 * math.cos(p4), L4 * math.sin(p4)
        a0, b0 = 2 * L2 * (xd - xb), 2 * L2 * (yd - yb)
        lbd = math.hypot(xd - xb, yd - yb)
        c0 = L2 * L2 + lbd * lbd - L3 * L3
        disc = a0 * a0 + b0 * b0 - c0 * c0
        if disc < 0:
            return float("nan")
        p2 = 2 * math.atan2(b0 + math.sqrt(disc), a0 + c0)
        return math.hypot(L1 * math.cos(p1) + L2 * math.cos(p2),
                          L1 * math.sin(p1) + L2 * math.sin(p2))
    return fk(ACT + q_la, HIP + q_lh), fk(ACT + q_ra, HIP - q_rh)


def build(csv_path):
    import csv as _csv
    rows = list(_csv.reader(open(csv_path, encoding="utf-8-sig")))
    head = [h.replace("(g_rl_shadow_debug).", "") for h in rows[0]]
    d = np.array([[float(x) for x in r] for r in rows[1:] if len(r) == len(rows[0])])
    col = lambda n: head.index(n)

    t = d[:, col("Time")]
    obs = np.stack([d[:, col(f"observation[{k}]")] for k in range(25)], 1)
    action = np.stack([d[:, col(f"mean_action[{k}]")] for k in range(6)], 1)
    lqr_leg = np.stack([d[:, col(f"lqr_leg_torque_nm[{k}]")] for k in range(4)], 1)
    lqr_wheel = np.stack([d[:, col(f"lqr_wheel_torque_nm[{k}]")] for k in range(2)], 1)
    rl_leg = np.stack([d[:, col(f"rl_leg_torque_nm[{k}]")] for k in range(4)], 1)
    rl_wheel = np.stack([d[:, col(f"rl_wheel_torque_nm[{k}]")] for k in range(2)], 1)
    active = d[:, col("rl_active")]

    gx, gy, gz = obs[:, 3], obs[:, 4], obs[:, 5]
    nrm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    pitch = np.degrees(np.arcsin(np.clip(-gz / nrm, -1, 1)))
    roll = np.degrees(np.arcsin(np.clip(-gx / nrm, -1, 1)))
    gyro = obs[:, 0:3] * 4.0
    q = obs[:, 6:10] * 1.5
    dq = obs[:, 10:16] * 20.0
    prev_leg = obs[:, 16:20] * 1.5
    prev_wheel = obs[:, 20:22] * 100.0
    cmd = np.stack([obs[:, 22] * 4.0, obs[:, 23] * 4.0, obs[:, 24] * .5], 1)

    # 固件补偿项(与 rl_control.c 同式),用于把命令折成净力矩
    spread_l = (ACT + q[:, 3]) - (HIP + q[:, 2])          # 左腿 lf-lb
    spread_r = (ACT + q[:, 1]) - (HIP - q[:, 0])          # 右腿 rf-rb
    L0 = np.array([leg_length(*q[i]) for i in range(len(t))])   # 列0=左, 列1=右
    comp_l = np.clip((-83.121 * L0[:, 0] + 40.242) * .8, 16.5, 27.0) * np.abs(np.sin(spread_l))
    comp_r = np.clip((-83.121 * L0[:, 1] + 40.242) * .8, 16.5, 27.0) * np.abs(np.sin(spread_r))
    # 模型序 RH,RA,LH,LA 的补偿: +s_right, +s_right, -s_left, +s_left
    comp_model = np.stack([comp_r, comp_r, -comp_l, comp_l], 1)
    rl_leg_net = np.stack([-rl_leg[:, 3], rl_leg[:, 2], -rl_leg[:, 1], -rl_leg[:, 0]], 1) - comp_model

    ch = []

    def add(group, key, label, unit, arr):
        ch.append(dict(g=group, k=key, l=label, u=unit,
                       d=[round(float(v), 5) if np.isfinite(v) else None for v in arr]))

    add("姿态", "pitch", "pitch", "deg", pitch)
    add("姿态", "roll", "roll", "deg", roll)
    add("姿态", "gx", "gyro_x", "rad/s", gyro[:, 0])
    add("姿态", "gy", "gyro_y", "rad/s", gyro[:, 1])
    add("姿态", "gz", "gyro_z", "rad/s", gyro[:, 2])
    for i, n in enumerate("RH RA LH LA".split()):
        add("关节位置", f"q{i}", f"q_{n}", "rad", q[:, i])
    for i, n in enumerate("RH RA RW LH LA LW".split()):
        add("关节速度", f"dq{i}", f"dq_{n}", "rad/s", dq[:, i])
    add("指令", "cvx", "cmd_vx", "m/s", cmd[:, 0])
    add("指令", "cyaw", "cmd_yaw", "rad/s", cmd[:, 1])
    add("指令", "ch", "cmd_height", "m", cmd[:, 2])
    for i, n in enumerate("RH RA LH LA".split()):
        add("目标缓存", f"pv{i}", f"prev_{n}", "rad", prev_leg[:, i])
    for i, n in enumerate("RW LW".split()):
        add("目标缓存", f"pw{i}", f"prev_{n}", "rad/s", prev_wheel[:, i])
    for i, n in enumerate("lf lb rf rb".split()):
        add("RL力矩(命令)", f"rl{i}", f"rl_leg_{n}", "Nm", rl_leg[:, i])
    for i, n in enumerate("RH RA LH LA".split()):
        add("RL力矩(净)", f"rn{i}", f"net_{n}", "Nm", rl_leg_net[:, i])
    for i, n in enumerate("L R".split()):
        add("RL轮力矩", f"rw{i}", f"rl_wheel_{n}", "Nm", rl_wheel[:, i])
    for i, n in enumerate("lf lb rf rb".split()):
        add("LQR力矩", f"lq{i}", f"lqr_leg_{n}", "Nm", lqr_leg[:, i])
    for i, n in enumerate("L R".split()):
        add("LQR轮力矩", f"lw{i}", f"lqr_wheel_{n}", "Nm", lqr_wheel[:, i])
    for i, n in enumerate("L R".split()):
        add("腿长", f"L0{i}", f"L0_{n}", "m", L0[:, i])
    for i in range(6):
        add("动作", f"a{i}", f"a{i}", "-", action[:, i])
    add("状态", "act", "rl_active", "-", active)

    idx = np.where(np.diff(active) > 0)[0]
    bands = []
    if len(idx):
        s = idx[0] + 1
        e = np.where(np.diff(active) < 0)[0]
        e = int(e[0]) + 1 if len(e) else len(t) - 1
        bands.append([float(t[s]), float(t[e])])
    return dict(t=[round(float(v), 5) for v in t], ch=ch, bands=bands,
                meta=dict(csv=str(csv_path), n=len(t)))


TEMPLATE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>Ozone 日志交互图 __TITLE__</title>
<style>
*{box-sizing:border-box}
html,body{margin:0;height:100%;font:13px/1.45 -apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang SC","Microsoft YaHei",sans-serif;background:#11151c;color:#dfe6ef}
#wrap{display:flex;height:100%}
#side{width:250px;flex:0 0 250px;overflow:auto;border-right:1px solid #232b36;padding:10px 10px 40px;background:#0d1116}
#main{flex:1;position:relative;min-width:0}
canvas{display:block;width:100%;height:100%;cursor:crosshair}
h1{font-size:13px;margin:2px 0 8px;color:#8fb4d9;font-weight:600;letter-spacing:.3px}
.grp{margin:8px 0 2px;color:#7d8b9c;font-size:11px;letter-spacing:.5px;text-transform:uppercase}
label{display:flex;align-items:center;gap:6px;padding:1px 2px;border-radius:3px;cursor:pointer;white-space:nowrap}
label:hover{background:#161d26}
label input{margin:0;accent-color:#4a9eff}
.sw{width:9px;height:9px;border-radius:2px;flex:0 0 9px}
.u{margin-left:auto;color:#5f6d7e;font-size:11px}
#btns{display:flex;flex-wrap:wrap;gap:4px;margin:6px 0 10px}
button{background:#1a2330;color:#cfe0f2;border:1px solid #2a3644;border-radius:4px;padding:3px 7px;font-size:11px;cursor:pointer}
button:hover{background:#22303f}
#hud{position:absolute;left:64px;top:6px;font-size:11px;color:#93a4b8;pointer-events:none;white-space:pre}
#tip{position:absolute;pointer-events:none;background:rgba(10,14,20,.93);border:1px solid #2b3746;border-radius:4px;padding:5px 7px;font-size:11px;line-height:1.5;display:none;max-height:70%;overflow:hidden}
#tip b{color:#8fb4d9;font-weight:500}
.k{display:inline-block;min-width:86px;color:#9fb0c3}
.v{font-variant-numeric:tabular-nums}
</style></head><body>
<div id="wrap">
  <div id="side">
    <h1>通道选择</h1>
    <div id="btns">
      <button data-p="all">全选</button><button data-p="none">清空</button>
      <button data-p="att">姿态</button><button data-p="joint">关节</button>
      <button data-p="torque">力矩</button><button data-p="act">动作</button>
      <button data-p="rl">只看 RL 窗口</button><button data-p="full">全时段</button>
    </div>
    <div id="list"></div>
  </div>
  <div id="main">
    <canvas id="cv"></canvas>
    <div id="hud"></div>
    <div id="tip"></div>
  </div>
</div>
<script>
const DATA = __DATA__;
const T = DATA.t, CH = DATA.ch, N = T.length;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const tip = document.getElementById('tip'), hud = document.getElementById('hud');
const PAL = ['#4a9eff','#ff7a5c','#3ddc97','#ffd166','#c792ea','#4dd0e1','#f06292','#aed581',
             '#ffab40','#90caf9','#ba68c8','#4db6ac','#e57373','#fff176','#81c784','#64b5f6',
             '#ff8a65','#9575cd','#4fc3f7','#a1887f'];
CH.forEach((c,i)=>{c.color = PAL[i % PAL.length]; c.vis = false;});

const st = {t0: T[0], t1: T[T.length-1], cur: N-1, norm: true, drag: null};
const L = {left:64, right:14, top:16, bottom:26};

function idxRange(){
  let a = 0, b = N-1;
  while(a < N-1 && T[a] < st.t0) a++;
  while(b > 0 && T[b] > st.t1) b--;
  return [Math.max(0,a-1), Math.min(N-1,b+1)];
}
function px(t){ return L.left + (t - st.t0) / (st.t1 - st.t0) * (cv.clientWidth - L.left - L.right); }
function tx(x){ return st.t0 + (x - L.left) / (cv.clientWidth - L.left - L.right) * (st.t1 - st.t0); }

function resize(){
  const r = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * r; cv.height = cv.clientHeight * r;
  ctx.setTransform(r,0,0,r,0,0);
  draw();
}

function draw(){
  const W = cv.clientWidth, H = cv.clientHeight, x0 = L.left, x1 = W - L.right;
  const y0 = L.top, y1 = H - L.bottom;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = '#11151c'; ctx.fillRect(0,0,W,H);

  // RL 接管窗口
  DATA.bands.forEach(([a,b])=>{
    const pa = Math.max(x0, px(a)), pb = Math.min(x1, px(b));
    if(pb > pa){ ctx.fillStyle = 'rgba(74,158,255,.07)'; ctx.fillRect(pa,y0,pb-pa,y1-y0); }
  });

  const [ia, ib] = idxRange();
  const vis = CH.filter(c=>c.vis);
  // 每通道可见区间 min/max
  vis.forEach(c=>{
    let mn = Infinity, mx = -Infinity;
    for(let i=ia;i<=ib;i++){ const v = c.d[i]; if(v===null) continue; if(v<mn) mn=v; if(v>mx) mx=v; }
    c._mn = mn; c._mx = mx;
    if(!isFinite(mn)){ c._mn = 0; c._mx = 1; }
    if(c._mx - c._mn < 1e-9){ c._mx = c._mn + 1e-9; }
  });

  // 网格 + 时间轴
  ctx.strokeStyle = '#1c242f'; ctx.lineWidth = 1;
  ctx.fillStyle = '#5f6d7e'; ctx.font = '11px ui-monospace,monospace';
  const span = st.t1 - st.t0;
  let step = Math.pow(10, Math.floor(Math.log10(span/8)));
  for(const m of [1,2,5,10]) if(span/(step*m) >= 5) { step *= m; break; }
  const tStart = Math.ceil(st.t0/step)*step;
  for(let t=tStart;t<=st.t1;t+=step){
    const x = Math.round(px(t)) + .5;
    ctx.beginPath(); ctx.moveTo(x,y0); ctx.lineTo(x,y1); ctx.stroke();
    ctx.fillText(t.toFixed(step<1?2:1)+'s', x+3, y1+14);
  }
  for(let k=0;k<=4;k++){
    const y = Math.round(y0 + k*(y1-y0)/4) + .5;
    ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x1,y); ctx.stroke();
  }
  ctx.strokeStyle = '#2a3644';
  ctx.strokeRect(x0+.5, y0+.5, x1-x0, y1-y0);

  // 统一坐标系时取所有可见通道的公共范围
  let gmn = Infinity, gmx = -Infinity;
  if(!st.norm) vis.forEach(c=>{ if(c._mn<gmn) gmn=c._mn; if(c._mx>gmx) gmx=c._mx; });
  if(!st.norm && !(gmx>gmn)) { gmn = -1; gmx = 1; }

  // 曲线
  ctx.save(); ctx.beginPath(); ctx.rect(x0,y0,x1-x0,y1-y0); ctx.clip();
  vis.forEach(c=>{
    ctx.strokeStyle = c.color; ctx.lineWidth = 1.2; ctx.beginPath();
    let started = false;
    for(let i=ia;i<=ib;i++){
      const v = c.d[i]; if(v===null){ started=false; continue; }
      const yy = st.norm ? (y1 - (v - c._mn)/(c._mx - c._mn) * (y1-y0))
                         : (y1 - (v - gmn)/(gmx - gmn) * (y1-y0));
      const xx = px(T[i]);
      if(!started){ ctx.moveTo(xx,yy); started = true; } else ctx.lineTo(xx,yy);
    }
    ctx.stroke();
  });
  ctx.restore();

  // 归一化时右侧标注每通道数值范围
  if(st.norm){
    ctx.font = '10px ui-monospace,monospace';
    let ly = y0 + 11;
    vis.slice(0,24).forEach(c=>{
      ctx.fillStyle = c.color;
      ctx.fillText(c.l+' '+c._mn.toPrecision(4)+' … '+c._mx.toPrecision(4)+' '+c.u, x1-300, ly);
      ly += 11;
    });
  }

  // 光标
  if(st.cur >= 0 && st.cur < N){
    const x = px(T[st.cur]);
    if(x >= x0 && x <= x1){
      ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.beginPath();
      ctx.moveTo(Math.round(x)+.5, y0); ctx.lineTo(Math.round(x)+.5, y1); ctx.stroke();
    }
  }
  hud.textContent = '窗口 '+st.t0.toFixed(3)+' → '+st.t1.toFixed(3)+' s   ｜  滚轮缩放 · 拖动平移 · 双击复位 · 已选 '
    + vis.length + ' 项   ｜  ' + (st.norm ? '每通道归一化' : ('统一坐标 '+gmn.toPrecision(4)+' … '+gmx.toPrecision(4)));
}

function valueAt(c, i){
  const v = c.d[i];
  return v === null ? '—' : v.toPrecision(4);
}

function showTip(mx){
  const [ia,ib] = idxRange();
  let i = nearest(mx);
  st.cur = i;
  const vis = CH.filter(c=>c.vis);
  if(!vis.length || i<0){ tip.style.display='none'; draw(); return; }
  const t = T[i];
  let html = '<b>t = ' + t.toFixed(3) + ' s   #' + i + '</b>';
  const mxv = 18;
  vis.slice(0,mxv).forEach(c=>{
    html += '<div><span class="sw" style="background:'+c.color+';display:inline-block;margin-right:4px"></span>'
         + '<span class="k">'+c.l+'</span><span class="v">'+valueAt(c,i)+'</span>'
         + '<span class="u">'+c.u+'</span></div>';
  });
  if(vis.length > mxv) html += '<div style="color:#5f6d7e">… 还有 '+(vis.length-mxv)+' 项</div>';
  tip.innerHTML = html;
  tip.style.display = 'block';
  const x = px(t), W = cv.clientWidth;
  tip.style.left = (x + (x > W-260 ? -230 : 16)) + 'px';
  tip.style.top = '28px';
  draw();
}

function nearest(mx){
  const t = tx(mx);
  let i = 0, best = Infinity;
  const [ia,ib] = idxRange();
  for(let k=ia;k<=ib;k++){ const d = Math.abs(T[k]-t); if(d<best){ best=d; i=k; } }
  return i;
}

cv.addEventListener('mousemove', e=>{
  const r = cv.getBoundingClientRect(), mx = e.clientX-r.left, my = e.clientY-r.top;
  if(st.drag){
    const dt = (mx - st.drag.mx) / (cv.clientWidth-L.left-L.right) * (st.drag.t1-st.drag.t0);
    st.t0 = st.drag.t0 - dt; st.t1 = st.drag.t1 - dt; clampWindow(); draw(); return;
  }
  if(mx < L.left || mx > cv.clientWidth-L.right){ tip.style.display='none'; return; }
  showTip(mx);
});
cv.addEventListener('mousedown', e=>{
  const r = cv.getBoundingClientRect();
  st.drag = {mx: e.clientX-r.left, t0: st.t0, t1: st.t1};
  cv.style.cursor = 'grabbing';
});
window.addEventListener('mouseup', ()=>{ st.drag = null; cv.style.cursor='crosshair'; });
cv.addEventListener('mouseleave', ()=>{ tip.style.display='none'; st.cur = -1; draw(); });
cv.addEventListener('dblclick', ()=>{ st.t0 = T[0]; st.t1 = T[T.length-1]; draw(); });
cv.addEventListener('wheel', e=>{
  e.preventDefault();
  const r = cv.getBoundingClientRect(), mx = e.clientX-r.left;
  const anchor = tx(Math.min(Math.max(mx, L.left), cv.clientWidth-L.right));
  const f = e.deltaY > 0 ? 1.2 : 1/1.2;
  st.t0 = anchor + (st.t0-anchor)*f; st.t1 = anchor + (st.t1-anchor)*f;
  clampWindow(); draw();
}, {passive:false});

function clampWindow(){
  const minSpan = 0.05;
  if(st.t1 - st.t0 < minSpan){ const c=(st.t0+st.t1)/2; st.t0=c-minSpan/2; st.t1=c+minSpan/2; }
  if(st.t0 < T[0]){ const d=T[0]-st.t0; st.t0+=d; st.t1+=d; }
  if(st.t1 > T[N-1]){ const d=st.t1-T[N-1]; st.t1-=d; st.t0-=d; }
}

// 侧栏
const list = document.getElementById('list');
const groups = [];
CH.forEach(c=>{ if(!groups.includes(c.g)) groups.push(c.g); });
groups.forEach(g=>{
  const h = document.createElement('div'); h.className='grp'; h.textContent = g; list.appendChild(h);
  CH.filter(c=>c.g===g).forEach(c=>{
    const lb = document.createElement('label');
    const cb = document.createElement('input'); cb.type='checkbox'; cb.checked = c.vis;
    cb.addEventListener('change', ()=>{ c.vis = cb.checked; draw(); });
    const sw = document.createElement('span'); sw.className='sw'; sw.style.background = c.color;
    const tx_ = document.createElement('span'); tx_.textContent = c.l;
    const u = document.createElement('span'); u.className='u'; u.textContent = c.u;
    lb.append(cb, sw, tx_, u); list.appendChild(lb);
    c._cb = cb;
  });
});

function setVis(pred, group){
  CH.forEach(c=>{
    const on = group ? c.g===group : pred;
    if(on === null) return;
    c.vis = on; if(c._cb) c._cb.checked = on;
  });
  draw();
}
document.getElementById('btns').addEventListener('click', e=>{
  const p = e.target.dataset.p; if(!p) return;
  if(p==='all') setVis(true);
  else if(p==='none') setVis(false);
  else if(p==='att') setVis(true,'姿态');
  else if(p==='joint'){ setVis(true,'关节位置'); setVis(true,'关节速度'); }
  else if(p==='torque'){ setVis(true,'RL力矩(净)'); setVis(true,'RL力矩(命令)'); setVis(true,'RL轮力矩'); }
  else if(p==='act') setVis(true,'动作');
  else if(p==='rl' && DATA.bands.length){ st.t0=DATA.bands[0][0]-0.2; st.t1=DATA.bands[0][1]+0.2; clampWindow(); draw(); }
  else if(p==='full'){ st.t0=T[0]; st.t1=T[N-1]; draw(); }
  else if(p==='mode'){ st.norm = !st.norm; draw(); }
});

window.addEventListener('resize', resize);
setVis(true, '姿态'); setVis(true, '动作');
resize();
</script></body></html>"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = build(args.csv)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace(
        "__TITLE__", Path(args.csv).name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print("写入 %s  (%.1f KB, %d 帧, %d 通道)"
          % (args.out, args.out.stat().st_size / 1024, data["meta"]["n"], len(data["ch"])))


if __name__ == "__main__":
    main()
