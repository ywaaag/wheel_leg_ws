'use strict';
const $ = selector => document.querySelector(selector);
const colors = ['#4edac5', '#86b0ff', '#f6b86a', '#d9a6ff'];
let snapshot, timer;
const finite = value => typeof value === 'number' && Number.isFinite(value);
const fmt = (value, digits = 3) => !finite(value) ? '未知' : value.toLocaleString('zh-CN', {maximumFractionDigits: digits});
const present = value => value === undefined || value === null ? '未知' : typeof value === 'object' ? JSON.stringify(value) : String(value);
const short = value => typeof value === 'string' ? value.split('/').pop() : '暂无';
const taskName = task => ({motion:'Motion', jump:'Jump', recovery:'Recovery', spin:'Spin'}[task] || task || '领域未知');
const statusName = status => ({RUNNING:'运行状态已记录', FAILED:'失败', STOPPED_BOUNDED:'已停止', COMPLETED:'已完成', UNKNOWN:'状态未知'}[status] || status);
function el(tag, text, parent, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = present(text);
  if (className) node.className = className;
  if (parent) parent.append(node);
  return node;
}
function section(id, title, note) {
  const node = el('section', undefined, $('#content')); node.id = id;
  el('h2', title, node); if (note) el('p', note, node, 'muted'); return node;
}
function empty(parent, text) { return el('div', text, parent, 'empty'); }
function details(parent, title, data) {
  const node = el('details', undefined, parent); el('summary', title, node);
  el('pre', JSON.stringify(data, null, 2), node); return node;
}
function evidence(parent, id) {
  if (!id) return;
  const link = el('a', '查看原始证据 ↗', parent); link.href = '/api/evidence/' + id;
  link.target = '_blank'; link.rel = 'noopener';
}
function kv(parent, values) {
  const list = el('dl', undefined, parent, 'kv');
  values.forEach(([label, value]) => { el('dt', label, list); el('dd', present(value), list); });
}
function select(parent, label, options) {
  const wrapper = el('label', label + ' ', parent);
  const input = el('select', undefined, wrapper); input.setAttribute('aria-label', label);
  options.forEach(([value, title]) => { const o = el('option', title, input); o.value = value; });
  return input;
}
function table(parent, rows, columns) {
  const wrapper = el('div', undefined, parent, 'table-wrap'), table = el('table', undefined, wrapper);
  const header = el('tr', undefined, el('thead', undefined, table));
  columns.forEach(([name]) => el('th', name, header));
  const body = el('tbody', undefined, table);
  rows.forEach(row => {
    const tr = el('tr', undefined, body);
    columns.forEach(([, render]) => {
      const td = el('td', undefined, tr); const value = render(row, td);
      if (value !== undefined) td.textContent = present(value);
    });
  });
  return table;
}
function rate(parent, value) {
  if (!finite(value)) { el('span', '未知', parent, 'muted'); return; }
  el('div', fmt(value * 100, 1) + '%', parent, 'rate');
  const bar = el('div', undefined, parent, 'bar');
  el('span', undefined, bar).style.width = Math.max(0, Math.min(100, value * 100)) + '%';
}
function flatten(row) {
  const result = {...row};
  function visit(value, prefix) {
    Object.entries(value || {}).forEach(([key, item]) => {
      const name = prefix ? prefix + '.' + key : key;
      if (finite(item)) result[name] = item;
      else if (item && typeof item === 'object' && !Array.isArray(item)) visit(item, name);
    });
  }
  visit(row, ''); return result;
}
function svgNode(tag, attrs, parent, text) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
  if (text !== undefined) node.textContent = text;
  parent.append(node); return node;
}
function chart(parent, title, rows, series, unit = '') {
  const box = el('article', undefined, parent, 'chart'); el('h3', title, box);
  const available = series.filter(item => rows.some(row => finite(row[item[0]]) && finite(row.update)));
  if (!available.length) { empty(box, '暂无该指标：日志未记录，不以0代替。'); return; }
  const legend = el('div', undefined, box, 'legend');
  available.forEach((item, i) => { const label = el('span', undefined, legend); el('i', undefined, label).style.background = colors[i % colors.length]; label.append(document.createTextNode(item[1])); });
  const points = rows.filter(row => finite(row.update) && available.some(item => finite(row[item[0]])));
  const xs = points.map(row => row.update), ys = points.flatMap(row => available.map(item => row[item[0]] * (item[2] || 1)).filter(finite));
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const pad = (ymax - ymin) * .12 || Math.abs(ymax) * .08 || 1;
  const low = ymin - pad, high = ymax + pad;
  const X = value => 64 + (value - xmin) / (xmax - xmin || 1) * 620;
  const Y = value => 198 - (value - low) / (high - low) * 170;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 710 240'); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', title + '原始曲线'); box.append(svg);
  for (let i = 0; i < 5; i++) {
    const value = low + (high - low) * i / 4, y = Y(value);
    svgNode('line', {x1:64,x2:684,y1:y,y2:y,stroke:'#293a53'}, svg);
    svgNode('text', {x:56,y:y+4,fill:'#9dacc4','font-size':11,'text-anchor':'end'}, svg, fmt(value));
    const xvalue = xmin + (xmax - xmin) * i / 4;
    svgNode('text', {x:X(xvalue),y:218,fill:'#9dacc4','font-size':11,'text-anchor':'middle'}, svg, fmt(xvalue,0));
  }
  svgNode('text', {x:680,y:237,fill:'#9dacc4','font-size':11,'text-anchor':'end'}, svg, '更新 update');
  if (unit) svgNode('text', {x:65,y:15,fill:'#9dacc4','font-size':11}, svg, unit);
  available.forEach((item, index) => {
    let path = '', previous;
    rows.forEach(row => {
      if (!finite(row.update) || !finite(row[item[0]])) { previous = null; return; }
      const x = X(row.update), y = Y(row[item[0]] * (item[2] || 1));
      const continuous = previous && row.stage === previous.stage && row.attempt === previous.attempt && row.update > previous.update;
      path += (continuous ? 'L' : 'M') + x + ',' + y;
      if (points.length < 100) {
        const point = svgNode('circle', {cx:x,cy:y,r:2.7,fill:colors[index % colors.length]}, svg);
        svgNode('title', {}, point, `update ${row.update} · ${item[1]} ${fmt(row[item[0]] * (item[2] || 1),6)} ${unit}`);
      }
      previous = row;
    });
    svgNode('path', {d:path,fill:'none',stroke:colors[index % colors.length],'stroke-width':2,'data-series':item[0]}, svg);
  });
  const note = el('div', `${points.length}条日志 · update ${fmt(xmin,0)}–${fmt(xmax,0)} · 原始值，未平滑`, box, 'chart-note');
  svg.addEventListener('pointermove', event => {
    const rect = svg.getBoundingClientRect(), target = xmin + (((event.clientX - rect.left) / rect.width * 710 - 64) / 620) * (xmax - xmin);
    const nearest = points.reduce((a,b) => Math.abs(a.update-target) < Math.abs(b.update-target) ? a : b);
    note.textContent = `update ${nearest.update} · ` + available.map(item => `${item[1]} ${fmt(nearest[item[0]] * (item[2] || 1),6)} ${unit}`).join(' / ');
  });
}
function modelCard(parent, label, model) {
  const box = el('div', undefined, parent, 'model'); el('span', label, box, 'muted');
  if (!model || typeof model !== 'object') { el('div','暂无记录',box,'name'); return; }
  const path = model.checkpoint || model.path;
  el('div', path ? short(path) : '已记录，路径字段未声明', box, 'name');
  kv(box, [['模型更新',model.update],['SHA256',model.sha256 ? model.sha256.slice(0,16)+'…' : null]]);
  details(box, '完整身份', model);
}
async function compare(run, layer, left, right, output) {
  if (left === '' || right === '') { output.textContent = '证据不足：没有可比较的两条记录。'; return; }
  try {
    const response = await fetch('/api/compare?' + new URLSearchParams({run:run.id,layer,left,right}));
    if (!response.ok) throw Error('比较数据已变化，请刷新');
    const result = await response.json();
    output.textContent = `${result.verdict} · 结束回合 n=${result.n.map(x => x ?? '未知').join(' / ')}。${result.note}`;
  } catch (error) { output.textContent = '读取失败：' + error.message; }
}
function comparisons(parent, run, layer) {
  const rows = run[layer], panel = el('div', undefined, parent, 'panel');
  el('h3', layer === 'windows' ? '训练窗口比较' : '固定模型比较', panel);
  if (!rows.length) { empty(panel, '没有可比较记录。'); return; }
  const controls = el('div',undefined,panel,'row');
  const options = rows.map((row,i) => [i, layer === 'windows' ? `u${row.update} · ${present(row.key)}` : `${(row.checkpoint_sha256 || '未知SHA').slice(0,12)} · ${present(row.condition)}`]);
  const left = select(controls,'起点',options), right = select(controls,'对照',options);
  if (options.length > 1) right.value = String(options.length - 1);
  const result = el('div',undefined,panel,'comparison-result');
  const update = () => compare(run,layer,left.value,right.value,result);
  left.onchange = right.onchange = update; update();
}
function render() {
  const run = snapshot.runs.find(item => item.id === $('#run').value);
  $('#content').replaceChildren();
  if (!run) { empty($('#content'),'当前筛选没有任务。'); return; }
  const rows = run.metrics.map(flatten), latest = rows.at(-1) || {};
  if(run.remote){const view=snapshot.remote;const note=el('div',`远端快照 · ${view?.status || '未知'} · 数据年龄 ${fmt(view?.age_seconds,0)}秒 · 仅最后成功采集，不是实时存活保证。`,$('#content'),'notice');if(view?.stale)note.textContent+=' 当前刷新失败或连接资料变更；显示过期缓存。';}
  let sectionNode = section('overview','任务总览');
  const stats = el('div',undefined,sectionNode,'stats');
  [['已完成 PPO更新',fmt(run.updates_used,0),'state记录，非模型保存进度'],['执行块已耗',Object.entries(run.used || {}).map(([k,v])=>`${k}: ${fmt(v,0)}`).join(' · ') || '未知','原budget账本，不从曲线推算'],['执行块剩余',Object.entries(run.remaining || {}).map(([k,v])=>`${k}: ${fmt(v,0)}`).join(' · ') || '未知','不含未声明的祖先额度'],['最新奖励',fmt(latest.reward_mean,5),rows.length ? `日志 update ${latest.update}` : '该run尚无指标日志']].forEach(([label,value,note]) => { const card=el('div',undefined,stats,'stat');el('div',label,card,'muted');el('div',value,card,'value');el('small',note,card); });
  const info = el('div',undefined,sectionNode,'panel');
  const summaryLine=el('div',undefined,info,'row');el('strong',`${taskName(run.task)} / ${present(run.backend)}`,summaryLine);el('span',statusName(run.status),summaryLine,'badge');el('span','进程未核验',summaryLine,'muted');
  const identity=el('details',undefined,info);el('summary','任务、课程与数据身份',identity);
  kv(identity,[['任务',run.name],['领域 / 后端',`${taskName(run.task)} / ${present(run.backend)}`],['运行状态',statusName(run.status)],['进程身份',run.process],['冻结源码根（声明）',run.source_root_declared],['worker cwd（实际）',run.remote_worker?.cwd],['远端核验时间',run.remote_identity?.verified_unix_s ? new Date(run.remote_identity.verified_unix_s*1000).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai'}):null],['课程版本',run.course],['指标文件更新时间',run.metrics_file_updated_at]]);
  el('small','状态来自落盘文件，不代表进程当前存活；文件时间不作为训练曲线时间轴。',info);
  sectionNode = section('trends','训练曲线','只显示已记录的原始值。阶段/attempt切换或update回退时断开；没有条件身份的奖励变化不代表能力进退。');
  if (!rows.length) empty(sectionNode,'该run没有可读取的metrics.jsonl，因此暂无曲线。可在顶部选择其他有日志的任务；读取问题见页面底部。');
  const grid = el('div',undefined,sectionNode,'grid');
  chart(grid,'平均奖励',rows,[['reward_mean','平均奖励']]);
  chart(grid,'PPO损失',rows,[['value_loss','Critic损失'],['surrogate','Actor surrogate']]);
  chart(grid,'策略KL',rows,[['kl','PPO KL'],['accepted_KL_peak','接受KL峰值']]);
  const phaseKey=(plain,nested)=>rows.some(row=>finite(row[plain]))?plain:nested;
  chart(grid,'每轮阶段耗时',rows,[['whole_update_wall_seconds_before_metrics','整轮（指标写入前）'],['seconds_per_update','更新耗时记录'],[phaseKey('rollout_seconds','performance_seconds.rollout'),'Rollout'],[phaseKey('learning_seconds','performance_seconds.learning'),'Learning']],'秒 / 日志原值');
  chart(grid,'梯度范数',rows,[['actor_gradient_norm','Actor'],['critic_gradient_norm','Critic']]);
  chart(grid,'Actor / Critic优化步数',rows,[['actor_optimizer_steps','Actor'],['critic_optimizer_steps','Critic']],'步 / 日志口径');
  chart(grid,'学习率',rows,[['actor_lr','Actor LR'],['critic_lr','Critic LR']]);
  chart(grid,'Torch峰值显存',rows,[['peak_memory_allocated_bytes','Allocated',1/1048576],['peak_memory_reserved_bytes','Reserved',1/1048576]],'MiB');
  el('p','Torch allocated/reserved不代表进程总显存或整卡占用。CPU墙钟不代表GPU kernel时间；未记录的显存类型不推断。',sectionNode,'muted tiny');
  const custom=el('details',undefined,sectionNode);el('summary','更多性能指标 / 自选曲线',custom);
  const keys=[...new Set(rows.flatMap(Object.keys))].filter(key=>key!=='update'&&rows.some(row=>finite(row[key])));
  const metric=select(custom,'指标',keys.map(key=>[key,key])), customChart=el('div',undefined,custom);
  const updateCustom=()=>{customChart.replaceChildren();if(metric.value)chart(customChart,metric.value,rows,[[metric.value,metric.value]]);};metric.onchange=updateCustom;custom.addEventListener('toggle',()=>{if(custom.open)updateCustom();});
  sectionNode=section('models','保存点与候选','来源、最新完整保存点、选中模型与训练候选分开；不生成跨比较域排名。');
  const models=el('div',undefined,sectionNode,'models');
  modelCard(models,'最新完整保存点',run.latest_model);modelCard(models,'显式选中模型',run.selected_model);modelCard(models,'初始化来源',run.source_model);
  const candidates=run.candidates || {};
  const candidatePanel=el('div',undefined,sectionNode,'panel');el('h3','训练统计候选',candidatePanel);
  kv(candidatePanel,[['标签',candidates.label || '暂无候选记录'],['比较域SHA',candidates.comparison_sha256],['独立验证证书',candidates.independent_certificate === false ? '否：训练统计不是独立验证' : '未知']]);
  const pools=candidates.candidates_by_objective || {};
  if (!Object.keys(pools).length) el('p','当前没有已发布的目标候选。',candidatePanel,'muted');
  else Object.entries(pools).forEach(([objective,pool])=>{ const list=el('div',undefined,candidatePanel,'models');el('h3',`目标：${objective}`,candidatePanel);Object.values(pool || {}).slice(0,12).forEach(model=>modelCard(list,'训练候选',model)); });
  details(candidatePanel,'候选 / inventory 原始记录', {candidates,inventory:run.inventory});
  sectionNode=section('capabilities','能力证据','三层证据独立展示；单项通过不代表整个能力已掌握。');
  const goal=el('details',undefined,sectionNode);el('summary','目标规划矩阵（不是本配方覆盖或通过结果）',goal);
  const targets=snapshot.targets?.[run.task];
  if (targets?.rows?.length) table(goal,targets.rows,[['能力ID',r=>r.capability_id],['目标',r=>r.target],['保留关系',r=>r.relationship],['优先级',r=>r.priority]]);else empty(goal,'该领域尚无目标展示适配。');
  const training=el('div',undefined,sectionNode,'panel');el('h3','A · 正常PPO训练窗口',training);
  el('p','随机训练回合、多个策略版本；不是窗口末端checkpoint的固定成绩。',training,'muted');
  if (!run.windows.length) empty(training,'暂无可用训练窗口：该run未产出支持的领域窗口，或尚未到导出边界。不能把reward当通过率。');
  else {
    const controls=el('div',undefined,training,'row'), updates=[...new Set(run.windows.map(w=>w.update))].sort((a,b)=>b-a);
    const windowSelect=select(controls,'统计窗口',updates.map(update=>[update,`截至 update ${update}`]));
    const search=el('input',undefined,controls);search.placeholder='筛选命令 / cohort';search.setAttribute('aria-label','筛选训练窗口');
    const body=el('div',undefined,training);
    const renderWindows=()=>{body.replaceChildren();const windows=run.windows.filter(w=>String(w.update)===windowSelect.value&&JSON.stringify([w.key,w.command,w.height]).includes(search.value));
      table(body,windows,[['目标 / cohort',w=>present(w.key)],['目标命令',w=>present(w.command)],['目标高度 m',w=>fmt(w.height)],['地形 / DR',w=>`${present(w.terrain)} / ${present(w.DR)}`],['结束 / 完整回合',w=>`${fmt(w.ended_episodes,0)} / ${fmt(w.completed_episodes,0)}`],['任务通过率',(w,td)=>rate(td,w.pass_fraction)],['质量通过率',(w,td)=>rate(td,w.quality_pass_fraction)],['实际vx / yaw',w=>`${fmt(w.mean_physical_vx)} / ${fmt(w.mean_physical_yaw)}`],['高度MAE m',w=>fmt(w.height_MAE_m,5)],['双支撑比例',(w,td)=>rate(td,w.mean_dual_support_fraction)],['窗口样本',w=>w.sample_count_status==='KNOWN'?`${fmt(w.actual_student_policy_samples,0)} Student / ${fmt(w.actual_all_transitions,0)} 全部`:'未知：缺窗口身份'],['详情',(w,td)=>{evidence(td,w.evidence_id);details(td,'条件 / 阈值 / 原判定',{condition:{terrain:w.terrain,DR:w.DR,perturbation:w.perturbation,actual_height_m:w.actual_height_m},thresholds:w.thresholds,reasons:w.reasons,failed_check_counts:w.failed_check_counts,window_selection_evidence:w.window_selection_evidence});}]]);
    };windowSelect.onchange=search.oninput=renderWindows;renderWindows();
  }
  comparisons(sectionNode,run,'windows');
  const fixed=el('div',undefined,sectionNode,'panel');el('h3','B · 固定模型独立验证',fixed);
  if (!run.evaluations.length) empty(fixed,'未独立验证：没有可读取且SHA一致的已有评估记录。');
  else run.evaluations.forEach(e=>{const card=el('div',undefined,fixed,'model');kv(card,[['模型SHA',e.entry.model_sha256],['协议',e.entry.protocol_version],['运行状态',e.entry.status],['身份材料',e.identity_status],['缺失材料',e.missing?.join('、') || '无已声明缺项']]);table(card,e.result.conditions || [],[['条件',r=>present(r.condition)],['结束回合',r=>fmt(r.ended_episodes,0)],['任务通过率',(r,td)=>rate(td,r.pass_fraction)],['质量通过率',(r,td)=>rate(td,r.quality_pass_fraction)]]);evidence(card,e.evidence_id);details(card,'评估原始记录',e.result);});
  comparisons(sectionNode,run,'fixed_rows');
  const hardware=el('div',undefined,sectionNode,'panel');el('h3','C · 实机验证',hardware);empty(hardware,run.hardware_status);
  sectionNode=section('data','数据来源与读取状态');
  const source=el('div',undefined,sectionNode,'panel');kv(source,[['允许的产物根',run.source],['本次读取时间',snapshot.read_at],['状态读取范围',run.state_read_scope],['进程核验',run.process],['事件目录',run.tensorboard_events_present?'检测到已有TensorBoard事件':'未检测到事件目录']]);
  if (run.errors.length) {el('div',`${run.errors.length}项缺失或读取限制；不等于任务失败。`,source,'notice');table(source,run.errors,[['文件',r=>r.file],['读取结果',r=>({FileNotFoundError:'文件未产生 / 不存在',ValueError:'大小、身份或格式限制'}[r.error] || r.error)],['说明',r=>r.message || '未知']]);}
  if(run.remote_worker){const worker=run.remote_worker;el('h3','远端worker身份快照',source);kv(source,[['PID / start_ticks',`${worker.pid} / ${worker.start_ticks}`],['进程状态',worker.state],['实际cwd',worker.cwd],['完整argv',worker.argv],['RSS MiB',fmt(worker.rss_bytes/1048576)],['累计CPU ticks',present(worker.cpu_ticks)],['核验结论',run.remote_identity?.status]]);details(source,'NSpid及进程状态（不能自动推断宿主GPU PID映射）',worker.status);}
  details(source,'高级：原始预算与引用索引', {used:run.used,limits:run.limits,remaining:run.remaining,attempts:run.attempts,model_references:run.model_references});
}
function choices(preferred) {
  const runs=snapshot.runs.filter(run=>!$('#task').value||run.task===$('#task').value);
  runs.sort((a,b)=>Number(Boolean(b.metrics.length))-Number(Boolean(a.metrics.length)) || Number(b.status==='RUNNING')-Number(a.status==='RUNNING') || String(b.metrics_file_updated_at || b.run_metadata_updated_at).localeCompare(String(a.metrics_file_updated_at || a.run_metadata_updated_at)));
  $('#run').replaceChildren();runs.forEach(run=>{const option=el('option',`${run.remote?'远端':'本地'} · ${taskName(run.task)} · ${run.name} · ${run.metrics.length}条指标`,$('#run'));option.value=run.id;});
  if (runs.some(run=>run.id===preferred)) $('#run').value=preferred;
  render();
}
async function refresh() {
  const button=$('#refresh');button.disabled=true;button.textContent='读取中…';
  try {
    const response=await fetch('/api/snapshot');if(!response.ok)throw Error('HTTP '+response.status);
    const selected=$('#run').value, task=$('#task').value;snapshot=await response.json();
    $('#task').replaceChildren();el('option','全部领域',$('#task')).value='';
    [...new Set(snapshot.runs.map(run=>run.task).filter(Boolean))].sort().forEach(name=>{el('option',taskName(name),$('#task')).value=name;});
    $('#task').value=task;$('#load-state').textContent=`读取时间 ${snapshot.read_at} · ${snapshot.runs.length}个已接入run（本地/远端见任务标签） · 默认手动刷新`;
    if(snapshot.tensorboard_url){const link=el('a',' 打开已有TensorBoard',$('#load-state'));link.href=snapshot.tensorboard_url;link.target='_blank';link.rel='noopener';}
    choices(selected);
  } catch(error) {$('#load-state').textContent='读取失败（不是训练失败）：'+error.message;}
  finally {button.disabled=false;button.textContent='刷新数据';}
}
$('#refresh').onclick=refresh;$('#run').onchange=render;$('#task').onchange=()=>choices();
$('#auto').onchange=()=>{clearInterval(timer);if($('#auto').checked)timer=setInterval(()=>{if(!$('#refresh').disabled)refresh();},30000);};
window.addEventListener('pagehide',()=>clearInterval(timer));refresh();

// Remote actions are restricted to collecting snapshots and editing the explicit SSH port.
const remotePanel=el('div',undefined,document.querySelector('header'),'panel');
el('h3','远程算力平台 · 只读',remotePanel);
const remoteInfo=el('p','未启用远端源：后端需使用 --remote-motion。',remotePanel,'muted');
const remoteControls=el('div',undefined,remotePanel,'row');
const remoteRefresh=el('button','刷新远端快照',remoteControls);
const portInput=el('input',undefined,remoteControls);portInput.type='number';portInput.placeholder='SSH端口';portInput.setAttribute('aria-label','远端SSH端口');
const hostInput=el('input',undefined,remoteControls);hostInput.placeholder='SSH地址';
const userInput=el('input',undefined,remoteControls);userInput.placeholder='用户名';
const fingerprintInput=el('input',undefined,remoteControls);fingerprintInput.placeholder='已核验SHA256主机指纹';
const portButton=el('button','保存连接资料',remoteControls);
el('small','密码只在后端终端私密输入；修改密码需重新启动看板后端并输入新凭据。不扫描端口、不重连训练。',remotePanel);
const resourceBox=el('div',undefined,remotePanel);
function remoteDisplay(view){
 if(!view)return;
 if(document.activeElement!==portInput)portInput.value=view.port;
 if(document.activeElement!==hostInput)hostInput.value=view.host;
 if(document.activeElement!==userInput)userInput.value=view.username;
 if(document.activeElement!==fingerprintInput)fingerprintInput.value=view.fingerprint;
 remoteInfo.textContent=`${view.host}:${view.port} · ${view.status} · ${view.last_success_unix_s?new Date(view.last_success_unix_s*1000).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai'}):'尚无成功采集'} · 数据年龄 ${fmt(view.age_seconds,0)}秒 · ${view.stale?'过期缓存/无法核实当前状态':'缓存快照（非持续监控）'}`;
 resourceBox.replaceChildren();const res=view.snapshot?.resources;if(!res)return;
 kv(resourceBox,[['CPU affinity',res.affinity?`${res.affinity.length}个逻辑CPU可见（不是quota）`:'未知'],['容器CPU quota',res['cpu.max']?.trim().split(' ')[0]==='max'?'不限额':fmt(Number(res['cpu.max']?.trim().split(' ')[0])/Number(res['cpu.max']?.trim().split(' ')[1]))+'核'],['容器cpu.max',res['cpu.max']],['容器内存上限',res['memory.max']?.trim()==='max'?'不限额':fmt(Number(res['memory.max'])/1073741824)+' GiB'],['容器内存当前',fmt(Number(res['memory.current'])/1073741824)+' GiB'],['数据盘空闲GiB',fmt(res.disk?.free_bytes/1073741824)],['GPU PID映射',res.gpu_pid_mapping]]);
 const gpu=(res.gpu||'').trim().split('\n').filter(Boolean).map(line=>line.split(',').map(x=>x.trim()));
 table(resourceBox,gpu,[['GPU名称',r=>r[0]],['UUID',r=>r[1]],['总显存MiB',r=>r[2]],['已用MiB',r=>r[3]],['空闲MiB',r=>r[4]],['瞬时利用率%',r=>r[5]],['温度°C',r=>r[6]],['功耗W',r=>r[7]]]);
 const gpuProcesses=(res.gpu_processes||'').trim().split('\n').filter(Boolean).map(line=>line.split(',').map(x=>x.trim()));el('h3','GPU compute进程（宿主PID）',resourceBox);table(resourceBox,gpuProcesses,[['宿主PID',r=>r[0]],['GPU UUID',r=>r[1]],['进程显存MiB',r=>r[2]],['容器PID映射',()=> 'UNKNOWN（不直接等同）']]);
}
async function remoteAction(path,data){remoteRefresh.disabled=true;remoteInfo.textContent='连接中…（不改变训练）';try{let response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!response.ok)throw Error('远端未启用或请求被拒绝');remoteDisplay(await response.json());await refresh();}catch(error){remoteInfo.textContent='读取失败，训练状态无法核实：'+error.message;}finally{remoteRefresh.disabled=false;}}
remoteRefresh.onclick=()=>remoteAction('/api/remote/refresh',{});
portButton.onclick=()=>remoteAction('/api/remote/connection',{host:hostInput.value.trim(),port:Number(portInput.value),username:userInput.value.trim(),fingerprint:fingerprintInput.value.trim()});

// Existing local refresh stays local; it displays cached remote data without connecting SSH.
const displayRemote=()=>{if(snapshot?.remote)remoteDisplay(snapshot.remote);};
const observer=new MutationObserver(displayRemote);observer.observe($('#load-state'),{childList:true});

let remoteTimer;
const remoteAutoLabel=el('label',undefined,remoteControls,'muted');
const remoteAuto=el('input',undefined,remoteAutoLabel);remoteAuto.type='checkbox';remoteAutoLabel.append(document.createTextNode(' 页面打开期间每120秒采集远端'));
remoteAuto.onchange=()=>{clearInterval(remoteTimer);if(remoteAuto.checked)remoteTimer=setInterval(()=>{if(!remoteRefresh.disabled)remoteAction('/api/remote/refresh',{});},120000);};
window.addEventListener('pagehide',()=>clearInterval(remoteTimer));
