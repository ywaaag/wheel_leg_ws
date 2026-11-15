# 旧tjb_recover失败与原生软件链对照

结论：当前集成配置在抬升后未持续接住机体；恢复原生高增益的软件链，在相同V4场景也未获得稳定起身。模型原始训练执行定义、真实弹簧与实机延迟缺失，不能把结果外推为模型在原机器人必然失败。

## 对照来源

原生分支feature/rl-tjb-recover，已提交c94bb9f。该工作区有未提交修改，本次只用git show c94bb9f提取源码，未修改它。feature/rl-motion31-supported015也保持只读参考。

源码快照与逐文件SHA在数据根recovery_motion_handoff/tjb_native_c94bb9f_sources。当前集成比较点40e1b8b，网络仍旧RL_10_19的31-128-64-32-6 ELU/Tanh，不能用目录名recovery2100认定模型身份。此前128状态C/PyTorch对照误差约mean1.79e-7、物理目标1.88e-5，未发现权重转换数值错误。不能由导出文件名确定原训练checkpoint身份。

|项目|原生c94bb9f|当前集成40e1b8b|
|---|---|---|
|Recovery输出|腿1.5×mean，轮100×mean|相同；没有用2100三次公式|
|31D观测|同列序/尺度，腿q先裁剪|同列序/尺度；实机越界拒绝|
|previous|上次网络解码目标，启动零；影子推理也可推进|实际PD采用目标，启动q与轮0代理|
|腿PD|60/2|40/2|
|轮速增益|1|0.2|
|算法限矩|腿40/轮5.5Nm|腿40/轮4.8Nm|
|最终轮电流限幅|8192原始单位|16384原始单位|
|代码比例换算后轮有效上限|约2.446948Nm，低于5.5|4.8Nm PD限制先于约4.893896Nm电流限制|
|PD执行|10ms推理任务基于快照算PD并保持|5ms执行端以反馈重算PD|
|实机代码补偿（非下方仿真设置）|Recovery输出时添加原软件补偿|Recovery当前无补偿，需求未确认；Motion目标采用后有015补偿|
|显式延迟队列|没有Actor20/target5/wheel7/11队列；实际延迟未知|名义20/5/7/11ms，DR变化另记录|
|切换|始终Recovery直到急停，无Motion自动接管|20ms本体门+目标连续性后切Motion|

原生定位：rl_motion31.c:113观测、:163解码、:185 PD；rl_shadow.c:183附近previous、:311 PD；rl_control.c:110电流限幅；chassis.c:710输出与补偿；robot_task.h:137机器人循环5ms，rl_shadow.c的策略循环10ms。行号以数据根原生快照为准，不能引用脏工作区内容。

## 当前配置如何失败

7个Swing development条件都因机体直立余弦<0.5（偏离直立超过60度）触发仿真故障。终止时闭链误差约1e-8m，没有触发5mm闭链故障，也没有低于0.10m高度故障。0.49–1.21秒内先抬到最高0.338–0.352m，再倾倒；并非完全没有抬升动作。

标称组0.30秒高0.318m、0.40秒高0.351m，两个时刻双轮法向支撑均0；之后俯仰继续增长，1.21秒约1.06rad且直立余弦0.488。轮速曾接近100rad/s、轮力矩曾触4.8Nm。这个证据支持抬升后接稳/姿态发散，不能单独证明哪项参数导致。Motion没有接管，故不是Motion接管后的补偿缺陷。

## 原生软件链仿真方法

复用wheel-leg sim2sim与原RecoveryMuJoCoCollection，原生C直接计算31D观测、线性解码、PD与最终电流限幅。策略/PD10ms，previous按原生虚拟缓存、启动零；去掉非原生target5/wheel7/11队列与关节目标投影。原生C PD输出再经过原V4的modal电机响应/运输与NET，保留原MuJoCo机械/地面/DR与故障判定。

原生分支本身不包含MuJoCo物理、独立真实弹簧和完整实测延迟合同。因此本次是原生软件链在既有V4上的有界诊断，不是完整原生固件实机仿真。完整V4上不额外叠加软件补偿，避免重复计算可能已包含的系统作用；也没有反加负补偿伪造物理弹簧。额外Actor年龄0ms只是扫描点，不能解释成实车零延迟。V4内原有q/dq年龄、motor运输/响应仍保留。

## 结果

同7个development条件、每组最长13秒，不接Motion：

|方案|保持10秒|故障时间范围|最长连续合格接稳|最高高度范围|
|---|---:|---:|---:|---:|
|当前完整配置|0/7|0.49–1.21s|0s|0.338–0.352m|
|原生软件链，40/2及轮0.2，额外Actor年龄0|0/7|3.49–8.13s|2.684s|0.365–0.382m|
|原生软件链，60/2及轮1，额外Actor年龄0|0/7|0.94–3.36s|0.120s|0.397–0.425m|
|原生软件链，60/2及轮1，额外Actor年龄20ms|0/7|0.59–0.86s|0.025s|0.379–0.411m|

原生高增益相对同原生软件链低增益，抬升更高、接稳时间更短；加入20ms年龄又恶化。由此不能推论原生高增益比当前更可靠。低增益原生链同时改变previous、PD采样/保持、额外延迟和最终轮限幅，是组合诊断；不能把改善单独归到previous或某个延迟。

根因仍待确认：旧模型训练物理/弹簧定义、观测时序与部署系统的匹配尚未建立。下一步应取得RL_10_19原训练配置/初始姿态与实机记录，再对previous、PD节拍、各延迟和轮限幅逐项消融。当前不建议凭原生分支名称直接替换整套控制参数上车。

配置configs/jobs/tjb_native_firmware_v4_diagnosis.json；数据根recovery_motion_handoff/tjb_native_c94bb9f_v4_r1与r2。r1开始后加入了数值等价的显式C decoder调用（原Tanh×scale→C clip×scale），保留r1，不作最终冻结结果；r2使用最终源码复测。新代码不进入训练或实机路径。

## 验证与修改清单

最终r2完成28回合、125080个0.5ms子步、墙钟90.876秒；四方案与r1的失败时间/最长保持逐项相同，28回合均倾倒终止。完整portable2016通过、31跳过、14 warnings，150.16秒；架构219文件/0例外，Ruff0.8.4 check/format通过。原生C在121帧真实状态上编码/解码/PD/电流上限数学核对通过，物理目标解码误差0。

修改：simulation/recovery_motion_handoff.py接线与边界、simulation/native_recovery_firmware.py原生C factory适配、configs/jobs/tjb_native_firmware_v4_diagnosis.json配方、tests/test_native_recovery_firmware.py及tests/test_simulation_recovery_motion_handoff.py消费者、ARCHITECTURE.md职责说明、STATUS.md结果导航、本报告。未新增scripts；未修改冻结源码/模型、两条电控分支或运行中训练。原有sim2sim模式默认路径保留，只有显式native_firmware才选原生诊断。
