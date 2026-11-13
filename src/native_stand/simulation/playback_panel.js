/* Web-only input adapter. Reuses the existing /command owner; no physics loop.
 * Adapted from the historical Motion panel's physical-key mapping, focus guards,
 * bounded history and dark charts. Its ramp/gamepad/skill arbitration is omitted.
 */
'use strict';
const $ = id => document.getElementById(id);
const state = {enabled:false, connected:false, paused:true, held:new Set(), pending:null,
  sending:false, resets:null, lastTime:null, scenes:'', stopping:false};
const history = [];
const aliases = {KeyW:'forward',ArrowUp:'forward',KeyS:'backward',ArrowDown:'backward',
  KeyA:'left',ArrowLeft:'left',KeyD:'right',ArrowRight:'right'};
function fmt(x, digits=3) {
  if(x === null || x === undefined || (typeof x==='number' && !Number.isFinite(x)))return '不可用';
  if(Array.isArray(x))return x.map(v=>fmt(v,digits)).join(' / ');
  return typeof x==='number'?x.toFixed(digits):String(x);
}
function report(text) {$('status').textContent=text;}
async function post(payload, keepalive=false) {
  const response=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(payload),keepalive});
  const data=await response.json();
  if(!response.ok)throw Error(data.error || response.status);
  return data;
}
// Serialize changes and coalesce unstarted requests, so key-up never overtakes
// key-down and stale commands do not accumulate behind a slow connection.
async function drain() {
  if(state.sending)return;
  state.sending=true;
  try {
    while(state.pending!==null) {
      const payload=state.pending;state.pending=null;
      try {await post(payload);report('命令已接收；实际采用值见反馈');}
      catch(error){report('发送失败：'+error.message);state.pending=null;state.held.clear();state.connected=false;paintKeys();}
    }
  } finally {state.sending=false;}
}
function enqueue(payload) {state.pending={...(state.pending||{}),...payload};void drain();}
function editable(element) {
  return !!element && (element.isContentEditable || ['INPUT','SELECT','TEXTAREA','BUTTON'].includes(element.tagName));
}
function positive(id) {
  const value=Number($(id).value);
  if(!Number.isFinite(value)||value<=0)throw Error('按键速度必须是正有限值');
  return value;
}
function paintKeys() {
  document.querySelectorAll('[data-drive]').forEach(node=>node.classList.toggle('held',
    [...state.held].some(code=>aliases[code]===node.dataset.drive)));
  $('keyboard').textContent=state.enabled?'关闭键盘控制':'启用键盘控制';
  $('keyboard').setAttribute('aria-pressed',String(state.enabled));
  $('key-state').textContent=state.enabled?
    (state.held.size?'按键：'+[...state.held].join(' + ')+(state.paused?'（已暂停）':''):'键盘已启用；按住行驶，松开归零'):
    '键盘未启用；先点击启用，再将焦点移出输入框。';
}
function movement() {
  const held=direction=>[...state.held].some(code=>aliases[code]===direction);
  return {vx:(Number(held('forward'))-Number(held('backward')))*positive('key-speed'),
          yaw:(Number(held('left'))-Number(held('right')))*positive('key-yaw')};
}
function sendHeld() {
  try {enqueue(movement());}catch(error){state.held.clear();enqueue({vx:0,yaw:0});report(error.message);}
  paintKeys();
}
function releaseKeys(force=false) {
  const active=state.held.size>0;state.held.clear();
  if(active||force)enqueue({vx:0,yaw:0});
  paintKeys();
}
function command(payload) {state.held.clear();paintKeys();enqueue(payload);}
$('keyboard').addEventListener('click',()=>{
  state.enabled=!state.enabled;releaseKeys(true);$('keyboard').blur();
});
document.addEventListener('keydown',event=>{
  if(!state.enabled||!state.connected||event.isComposing||event.ctrlKey||event.metaKey||event.altKey||editable(event.target))return;
  if(aliases[event.code]) {
    event.preventDefault();if(state.held.has(event.code))return;
    state.held.add(event.code);sendHeld();
  } else if(event.code==='Space'||event.code==='Escape'||event.code==='KeyR') {
    event.preventDefault();if(event.repeat)return;
    command(event.code==='Space'?{vx:0,yaw:0}:event.code==='Escape'?{vx:0,yaw:0,paused:true}:{vx:0,yaw:0,reset:true});
  }
});
document.addEventListener('keyup',event=>{
  if(!state.held.has(event.code))return;
  event.preventDefault();state.held.delete(event.code);sendHeld();
});
window.addEventListener('blur',()=>releaseKeys());
document.addEventListener('visibilitychange',()=>{if(document.hidden)releaseKeys();});
document.addEventListener('focusin',event=>{if(editable(event.target))releaseKeys();});
window.addEventListener('pagehide',()=>{
  if(state.held.size){state.held.clear();void post({vx:0,yaw:0},true).catch(()=>{});}
});
$('send').addEventListener('click',()=>{
  const values={vx:Number($('vx').value),yaw:Number($('yaw').value),height:Number($('height').value)};
  if(Object.values(values).some(v=>!Number.isFinite(v))){report('命令必须是有限数值');return;}
  command(values);
});
for(const [id,payload] of Object.entries({pause:{paused:true,vx:0,yaw:0},resume:{paused:false},
  stop:{vx:0,yaw:0},'pad-stop':{vx:0,yaw:0},reset:{reset:true,vx:0,yaw:0},quit:{quit:true,vx:0,yaw:0}})) {
  $(id).addEventListener('click',()=>command(payload));
}
$('spawn').addEventListener('click',()=>command({scene:$('scene').value,vx:0,yaw:0}));
document.querySelectorAll('[data-drive]').forEach(button=>button.addEventListener('click',()=>{
  try {const dir=button.dataset.drive;command({vx:dir==='forward'?positive('key-speed'):dir==='backward'?-positive('key-speed'):0,
    yaw:dir==='left'?positive('key-yaw'):dir==='right'?-positive('key-yaw'):0});}
  catch(error){report(error.message);}
}));
function draw(id, rows) {
  const canvas=$(id),scale=window.devicePixelRatio||1;
  const w=Math.max(1,Math.round(canvas.clientWidth*scale)),h=Math.max(1,Math.round(canvas.clientHeight*scale));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,w,h);
  if(!history.length)return;
  const values=history.flatMap(p=>rows.map(r=>r.get(p))).filter(Number.isFinite);
  if(!values.length)return;
  let lo=Math.min(...values),hi=Math.max(...values),pad=Math.max((hi-lo)*.15,.01);lo-=pad;hi+=pad;
  const left=48*scale,right=w-10*scale,top=12*scale,bottom=h-25*scale;
  ctx.font=`${10*scale}px system-ui`;ctx.textAlign='right';
  for(let i=0;i<=4;i++){const y=top+(bottom-top)*i/4;ctx.strokeStyle='#26364b';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillStyle='#94a6bc';ctx.fillText((hi-(hi-lo)*i/4).toFixed(2),left-5*scale,y+3*scale);}
  const end=history.at(-1).time,start=Math.max(history[0].time,end-30),span=Math.max(end-start,.5);
  ctx.textAlign='left';ctx.fillText(`${start.toFixed(1)} s`,left,h-5*scale);ctx.textAlign='right';ctx.fillText(`${end.toFixed(1)} s`,right,h-5*scale);
  for(const row of rows){ctx.strokeStyle=row.color;ctx.lineWidth=1.5*scale;ctx.setLineDash(row.dash?[5*scale,4*scale]:[]);ctx.beginPath();let open=false;
    for(const p of history){const v=row.get(p);if(!Number.isFinite(v)){open=false;continue;}const x=left+(p.time-start)/span*(right-left),y=bottom-(v-lo)/(hi-lo)*(bottom-top);if(!open)ctx.moveTo(x,y);else ctx.lineTo(x,y);open=true;}ctx.stroke();}ctx.setLineDash([]);
}
function curves(){for(const [field,index]of [['vx',0],['yaw',1],['height',2]])draw(field+'-chart',[
  {get:p=>p[field],color:'#68adff'},{get:p=>p.command?.[index],color:'#ffc477',dash:true}]);
  draw('posture-chart',[{get:p=>p.roll,color:'#68adff'},{get:p=>p.pitch,color:'#53d6a0'}]);}
function render(data) {
  const t=data.telemetry||{};
  state.paused=!!data.paused;state.connected=true;
  $('mode').textContent=data.paused?'已暂停 · 点击继续':'运行中';
  $('mode').style.color=data.paused?'var(--amber)':'var(--green)';
  $('pause').disabled=data.paused;$('resume').disabled=!data.paused;
  $('error').textContent=data.error?'故障：'+data.error:'';
  $('actual-vx').textContent=fmt(t.vx);$('actual-yaw').textContent=fmt(t.yaw);$('actual-height').textContent=fmt(t.height);
  for(const [id,index]of [['vx',0],['yaw',1],['height',2]])$('command-'+id).textContent='实际采用 '+fmt(t.adopted_command?.[index]);
  $('posture').textContent=fmt([t.roll,t.pitch]);$('terrain').textContent=t.terrain??'不可用';
  $('contacts').textContent=[t.contact_geometric_left,t.contact_geometric_right].map(v=>v===true?'接触':v===false?'离地':'不可用').join(' / ');
  const sceneKey=JSON.stringify(data.scenes);
  if(sceneKey!==state.scenes){$('scene').replaceChildren(...data.scenes.map(s=>{let o=document.createElement('option');o.value=s;o.textContent=s;return o;}));state.scenes=sceneKey;}
  if(Array.isArray(data.height_bounds)){ $('height').min=data.height_bounds[0];$('height').max=data.height_bounds[1]; }
  const rows=[['待采用 vx / yaw / height',[data.target.vx,data.target.yaw,data.height]],['策略实际采用命令',t.adopted_command],
    ['位置 X / Y m',t.position_xy],['当前地形',t.terrain],['最近重置区域',t.spawn],['仿真时间 s / 步数 / reset',[t.simulation_time_s,t.steps,t.resets]],
    ['控制归属',t.owner],['求解器profile',t.solver_profile],['保存数据',data.recording?'是':'否']];
  $('metrics').replaceChildren(...rows.map(([k,v])=>{const tr=document.createElement('tr');for(const x of[k,fmt(v)]){const td=document.createElement('td');td.textContent=x;tr.append(td);}return tr;}));
  $('detail').textContent=JSON.stringify(Object.fromEntries(['q','dq','raw','requested','adopted','motor','NET'].map(k=>[k,t[k]??null])),null,2);
  if(state.resets!==t.resets || (state.lastTime!==null&&t.simulation_time_s<state.lastTime)){history.length=0;state.lastTime=null;}
  state.resets=t.resets;
  if(Number.isFinite(t.simulation_time_s)&&state.lastTime!==t.simulation_time_s){history.push({...t,time:t.simulation_time_s,command:t.adopted_command});state.lastTime=t.simulation_time_s;while(history.length>300||history[0].time<t.simulation_time_s-30)history.shift();}
  paintKeys();curves();
}
async function poll(){try{const response=await fetch('/telemetry',{cache:'no-store'});if(!response.ok)throw Error(response.status);render(await response.json());}
  catch(error){if(state.connected)releaseKeys();state.connected=false;$('mode').textContent='连接断开';report('连接断开；最后命令可能仍被保持：'+error.message);}setTimeout(poll,200);}
window.addEventListener('resize',curves);void poll();
