"""Run the served page's real JavaScript without Isaac or a browser dependency."""
FROZEN_TEST_SCOPE = 'PORTABLE'

def test_recovery_panel_fixed_command_and_continuous_timing():
    import pytest
    from native_stand.students.motion_panel import Plan
    from native_stand.students.motion_preview import PreviewTiming
    plan=Plan(1.,0.,mode='Recovery')
    assert plan.snapshot()['mode']=='Recovery'
    for payload in ({'vx':1},{'yaw':.1},{'max_speed':2}):
        with pytest.raises(ValueError,match='fixed zero'):plan.apply(payload)
    plan.apply({'reset':True,'center':True})
    assert plan.take()['reset'] and not plan.take()['reset']
    timer=PreviewTiming(.01,100.)
    plan.publish(timer.sample(50,101.))
    assert plan.snapshot()['telemetry']['realtime_factor']==.5
    plan.apply({'reset':True});plan.take()
    plan.publish(timer.sample(100,102.))
    assert plan.snapshot()['telemetry']['simulation_time_s']==1.
    assert plan.snapshot()['telemetry']['realtime_factor_total']==.5

import json
import shutil
import subprocess

import pytest

from native_stand.students.motion_panel import HTML_PATH, Plan


def test_page_keyboard_stop_slider_and_neutral_gamepad():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed to execute the actual panel JavaScript')
    harness = r"""
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const source=fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0];
function page(pads=[]){
  const elements={},docEvents={},winEvents={},sent=[];let next,now=0;
  const get=id=>elements[id]||(elements[id]={value:'0',events:{},addEventListener(k,f){this.events[k]=f;}});
  const document={hidden:false,getElementById:get,addEventListener(k,f){docEvents[k]=f;}};
  vm.runInNewContext(source,{document,window:{addEventListener(k,f){winEvents[k]=f;}},
    navigator:{getGamepads:()=>pads},requestAnimationFrame(f){next=f;},setInterval(){},
    fetch(url,options){
      if(url==='/telemetry')return new Promise(()=>{});
      sent.push(JSON.parse(options.body));return Promise.resolve({ok:true,json:async()=>({})});
    }});
  const frame=(n=1)=>{for(let i=0;i<n;i++){now+=50;next(now);}};
  const key=(type,code,key='Process')=>docEvents[type]({code,key,target:{tagName:'BODY'},preventDefault(){}});
  return {sent,frame,key,get,document,docEvents,winEvents,last:()=>sent.at(-1)};
}
const p=page([{connected:true,axes:[0,0,0]}]); // Neutral gamepad used to erase W.
p.frame();p.key('keydown','KeyW');p.frame(20);assert(p.last().vx>.4);
p.key('keyup','KeyW');p.frame(30);assert.equal(p.last().vx,0);
p.get('btn-stop').events.click();p.key('keydown','KeyW');p.frame(20);assert(p.last().vx>.4);
p.key('keydown','Escape');assert.deepEqual(p.last(),{stop:true});
p.key('keydown','KeyS');p.frame(20);assert(p.last().vx<-.4);
p.winEvents.blur();p.frame(30);assert.equal(p.last().vx,0);
p.key('keydown','KeyA');p.frame(10);assert(p.last().yaw>0);
p.document.hidden=true;p.docEvents.visibilitychange();p.frame(30);assert.equal(p.last().yaw,0);
const slider=page();slider.get('vx-target').events.input({target:{value:'1'}});
slider.frame();assert.equal(slider.get('vx-target').value,'1.00'); // Thumb stays at the requested setpoint.
slider.frame(60);assert.equal(slider.last().vx,1);slider.frame(10);assert.equal(slider.last().vx,1);
slider.winEvents.blur();slider.document.hidden=true;slider.docEvents.visibilitychange();
slider.frame(20);assert.equal(slider.last().vx,1);
slider.get('cap-speed').events.change({target:{value:'0.5',checkValidity:()=>true}});
slider.frame(30);assert.equal(slider.last().vx,.5);
assert.equal(slider.get('vx-target').max,'0.5');
slider.get('cap-speed').events.change({target:{value:'',checkValidity:()=>false}});
slider.frame(2);assert.equal(slider.last().vx,.5);

slider.key('keydown','KeyW');slider.frame(2);slider.key('keyup','KeyW');slider.frame(60);
assert.equal(slider.last().vx,0);
slider.get('btn-reset').events.click();assert.deepEqual(slider.last(),{reset:true,center:true});
console.log(JSON.stringify(p.sent));
"""
    result = subprocess.run([node, '-e', harness, str(HTML_PATH)], capture_output=True,
                            text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    # Real page payloads must be accepted unchanged by the Python command holder
    # and produce a nonzero command through the same ramp as the GUI loop.
    from native_stand.command_v6.domain.keyboard_command import KeyboardCommand
    from native_stand.students.motion_preview import _merge_command
    plan = Plan(2., 1.)
    controls = KeyboardCommand(1., 0., max_speed=2., max_yaw_speed=1.)
    maximum = 0.
    for payload in json.loads(result.stdout):
        plan.apply(payload)
        row = plan.take()
        if row['stop'] or row['center']:
            controls.current = [0., 0.]
        command, _ = _merge_command(controls, row, .01, False, (2., 1.))
        maximum = max(maximum, abs(command[0]))
    assert maximum > .05


@pytest.mark.parametrize("mode",["Motion","Replay"])
def test_page_renders_schema_metrics_charts_and_reuses_canvas_buffers(mode):
    """Metric/chart tables stay wired to real ids and canvas buffers are not reallocated per frame."""
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed to execute the actual panel JavaScript')
    harness = r"""
const vm=require('node:vm'),fs=require('node:fs');
const source=fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0];
const ctx=()=>new Proxy({},{get(t,k){
  if(k==='createImageData')return (w,h)=>({data:new Uint8ClampedArray(w*h*4)});
  if(k in t)return t[k];return ()=>{};},set(t,k,v){t[k]=v;return true;}});
const elements={},widthWrites={};
function node(id,isCanvas){
  if(elements[id])return elements[id];
  const box={id,value:'',textContent:'',hidden:false,style:{},events:{},
    clientWidth:320,clientHeight:120,getAttribute(){return null;},setAttribute(){},
    addEventListener(k,f){this.events[k]=f;},appendChild(){},removeChild(){},firstChild:null,
    querySelector(){return null;},closest(){return null;},classList:{toggle(){},contains(){return false;}}};
  if(isCanvas){
    let w=0,h=0;widthWrites[id]=0;
    Object.defineProperty(box,'width',{get(){return w;},set(v){widthWrites[id]+=1;w=v;}});
    Object.defineProperty(box,'height',{get(){return h;},set(v){h=v;}});
    box.getContext=()=>box.ctx||(box.ctx=ctx());
  }
  elements[id]=box;return box;
}
const document={hidden:false,activeElement:null,
  getElementById:id=>node(id,/^chart-/.test(id)),
  createElement:tag=>node('el-'+tag+'-'+Object.keys(elements).length,tag==='canvas'),
  createTextNode:t=>({textContent:t}),addEventListener(){},querySelectorAll(){return [];}};
const sample={t:1,steps:10,vx:.2,cmd_vx:.2,yaw:0,cmd_yaw:0,cmd_height:.36,height:.33,vz:0,
  swing_diff_rad:.01,swing_mean_rad:.02,leg_length_diff_m:.001,wheel_fore_aft_m:.002,
  wheel_qdot_l:1,wheel_qdot_r:1,wheel_tau_l:.1,wheel_tau_r:.1,
  leg_tau_0:.1,leg_tau_1:.1,leg_tau_2:.1,leg_tau_3:.1,
  act_0:.1,act_1:.1,act_2:.1,act_3:.1,act_4:.1,act_5:.1,pitch:.01,roll:.01,
  skill:'Motion',jump:'NONE',actor_task_tail:[1,0,0,1,0,0],actor_input:new Array(31).fill(.1),
  task_mode:'Motion',initialized:true,actor_valid:true,key_events:0,source:'test',
  simulation_time_s:1,wall_time_s:1,realtime_factor:1,realtime_factor_total:1,resets:0,
  physics:[['腿 armature (kg·m²)','0.350']]};
const payload={version:'x',mode:'Motion',backend:'MuJoCo',editable_limits:true,
  target:{vx:.2,yaw:0},height:.36,skill:'Motion',jump:'NONE',height_bounds:[.2,.45],
  limits:{max_speed:.5,max_yaw_speed:.5},declared_limits:{max_speed:.5,max_yaw_speed:.5},
  requests:1,paused:false,pause_supported:true,telemetry:sample};
const replay=process.argv[2]==='Replay',sent=[];
if(replay){payload.mode='Replay';payload.models=['logged','closed'];payload.model='logged';payload.playback_speed=.25;
  Object.assign(sample,{sample_id:'logged:1',task_mode:'Replay',contact_l:15,contact_r:20,closure_m:.001,
                       q:[0,1,2,3,4,5],adopted:[0,1,2,3,4,5]});}
let poll=null,next=null;
const sandbox={document,window:{addEventListener(){},devicePixelRatio:2},
  navigator:{getGamepads:()=>[]},requestAnimationFrame(f){next=f;},setInterval(f){poll=f;},
  Number,Math,Array,JSON,isFinite,NaN,Infinity,String,Blob:function(){},
  URL:{createObjectURL(){return 'x';},revokeObjectURL(){}},
  fetch:(url,options)=>{if(options)sent.push(JSON.parse(options.body));return Promise.resolve({ok:true,json:async()=>url==='/telemetry'?payload:{}});}};
vm.runInNewContext(source,sandbox,{filename:'panel.js'});
const frame=n=>{for(let i=0;i<n;i++){if(next){const f=next;next=null;f(1000+i*50);}}};
(async()=>{
  for(let i=0;i<4;i++){
    if(replay && i<2){sample.sample_id='logged:'+i;sample.simulation_time_s=1+i*.01;}
    await poll();
  }
  frame(6);
  const automaticPosts=sent.slice();
  if(replay){
    node('replay-speed').events.input({target:{value:'0.5'}});
    node('btn-pause').events.click();
    node('model-select').events.change({target:{value:'closed'}});
    node('btn-reset').events.click();
  }
  const metrics={};
  ['m-vx','m-height','m-tail','m-mode','m-keys','m-contact','m-adopted'].forEach(id=>{metrics[id]=node(id).textContent;});
  console.log(JSON.stringify({metrics,widthWrites,commandWidth:node('chart-command',true).width,
    inputHeight:node('chart-input',true).height,status:node('status').textContent,sent,automaticPosts,
    controlsHidden:node('motion-controls').hidden,replayControlsHidden:node('replay-controls').hidden}));
})();
"""
    result = subprocess.run([node, '-e', harness, str(HTML_PATH),mode], capture_output=True,
                            text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    page = json.loads(result.stdout.strip().splitlines()[-1])
    assert page['metrics']['m-vx'] == '0.200'
    assert page['metrics']['m-height'] == '0.36 / 0.330'
    assert page['metrics']['m-tail'] == '1.00  0.00  0.00  1.00  0.00  0.00'
    assert page['metrics']['m-mode'] == mode
    assert page['metrics']['m-keys'] == '0'
    # 320 CSS px at DPR 2, and the backing store is sized once, not once per frame.
    assert page['commandWidth'] == 640 and page['widthWrites']['chart-command'] == 1
    assert page['widthWrites']['chart-input'] == 1 and page['inputHeight'] == 240
    assert page['status'] == '已连接'

    if mode=='Replay':
        assert page['controlsHidden'] and not page['replayControlsHidden']
        assert page['automaticPosts']==[]  # Never send hidden movement commands during playback.
        assert page['sent']==[{'playback_speed':.5},{'paused':True},{'model':'closed'},{'reset':True}]
        assert page['metrics']['m-contact']=='15.00 / 20.00'
        assert page['metrics']['m-adopted']=='0.000 / 1.000 / 2.000 / 3.000 / 4.000 / 5.000'


def test_page_gamepad_analog_stick_deadzone_and_release():
    """Left stick is analog forward/back, right stick yaw, deadzone rescaled, release returns to the slider."""
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed to execute the actual panel JavaScript')
    harness = r"""
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const source=fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0];
const pads=[{connected:true,axes:[0,-1,0.6]}];
const elements={},sent=[];let next,now=0;
const get=id=>elements[id]||(elements[id]={value:'0',textContent:'',events:{},
  addEventListener(k,f){this.events[k]=f;}});
const document={hidden:false,activeElement:null,getElementById:get,addEventListener(){},
  createElement:()=>({style:{},appendChild(){},setAttribute(){}}),
  createTextNode:t=>({textContent:t}),querySelectorAll:()=>[]};
vm.runInNewContext(source,{document,window:{addEventListener(){},devicePixelRatio:1},
  navigator:{getGamepads:()=>pads},requestAnimationFrame(f){next=f;},setInterval(){},
  fetch:(url,options)=>{if(url==='/telemetry')return new Promise(()=>{});
    sent.push(JSON.parse(options.body));return Promise.resolve({ok:true,json:async()=>({})});}});
const frame=n=>{for(let i=0;i<n;i++){now+=50;next(now);}};
const last=()=>sent.at(-1);
frame(80);                       // full up + half right: analog ramp reaches the caps
assert(last().vx>1.4, 'left stick up must drive forward speed');
assert(last().yaw<-.4, 'right stick must drive negative yaw');
assert(get('m-pad').textContent.indexOf('已连接')===0);
pads[0].axes=[0,-0.15,0];        // inside the .20 deadzone
frame(120);assert.equal(last().vx,0);assert.equal(last().yaw,0);
pads[0].axes=[0,1,0];            // pull back
frame(80);assert(last().vx<-1.4);
pads[0].axes=[0,0,0];            // release returns to the latched slider setpoint (zero)
frame(80);assert.equal(last().vx,0);
assert(get('m-pad').textContent.indexOf('已连接')===0);
pads[0].connected=false;frame(2);
assert(get('m-pad').textContent.indexOf('未连接')===0);
console.log(JSON.stringify(last()));
"""
    result = subprocess.run([node, '-e', harness, str(HTML_PATH)], capture_output=True,
                            text=True, timeout=15)
    assert result.returncode == 0, result.stderr
