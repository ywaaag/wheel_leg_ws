# Jump 首次离地课程与配方

2026-11-16。用户重新开启配方设计；本轮不启动训练。代码在独立worktree `/home/kellen/wheel_leg_jump_first_takeoff_ws`，分支 `jump/first-takeoff-course`。复用已保存的WIP `cbdaedf`，不修改旧run、远端冻结代码、Motion或Recovery。

## 依据与假设

旧远端u1363已正常 `STOPPED_BOUNDED`，failure=null。累计3790/10000，剩6210；BC轮数、CPU数学测试与真实PPO额度分开。完整终态checkpoint备份在数据根 `jump_first_takeoff_design_20261116/remote_terminal_model_update_1363.pt`，196267265字节，SHA `01d3e9c6ec855abe803603a8b77a28384dcda2fe1689780b8e682d7aa490d83f`。终态以同目录 `pause_terminal_verified.json` 为准，不按历史STATUS中的RUNNING条目继续提交。

旧配方共出现6个旧门槛离地标记、2个apex标记、0个落稳、0个完整跳跃。这6个标记不表示6次成功跳跃。旧门槛为双轮净空>2mm、无接触4ms、离地起点COM向上速度>0.1m/s及主动控制来源；新门槛更严格，两个计数不能直接比较。

u1000窗口68039回合中61640是 `NO_TAKEOFF`，6393是任务超时，6个旧目标故障；“故障率高”不能解释成同等比例摔倒。本机两环境有界诊断看到反复压伸、COM向上速度峰约0.54/0.48m/s，但双轮没有同时离地；电机峰22～29Nm，没有撞40Nm限矩。诊断原始结果见旧worktree `artifacts/jump_stall_diagnosis_20261116/DIAGNOSIS.md`。本机Isaac6.0.1证据不能冒充远端6.2复验。

当前假设：完整完成之前的结果信号稀疏，且旧训练将腿探索std从0.06rad压缩到约0.03rad，不利于形成新的离地动作。旧势函数的终态抵消符合设计，不能说成奖励算错。新配方以可达到的真实技能事件逐步推进，并保持探索；是否能解决学习停滞，仍需后续有界物理和正常PPO证据，不能由CPU测试证明。

## 控制与初始化

- 独立Jump Actor负责准备、蹬伸、腾空、落地、落稳及自身回到BALANCE。运行时没有Motion专家切换或Teacher控制；Motion只提供此前闭环补教的来源。
- 使用BC60 Actor-only，来源为数据根 `2026_11_15__jump_closed_loop_teaching_v4/exports/BC60_actor_v192/actor.pt`，SHA `871bf6316a0ec44cb342c4f42579928265fb163e4d15775909a58e97d30d9d5e`。其60轮／3060优化步来源在CPU核验中与原fit产物对照；导出本身没有该计数时，按显式作业来源记录保留该注记。它不计入PPO额度，也不导入BC Adam。
- 继承Actor均值，新的Critic、双Adam、RNG与课程。旧u1363完整学习状态保留；新配方不是旧PPO完整resume。
- Actor保持ABI4、37D、2048/1024/512 ELU、raw Normal线性direct输出；腿增益0.5rad、轮增益10rad/s，无tanh、轨迹残差或新增滤波。
- 物理保持原Cylinder＋原生Plane、限矩腿40Nm／轮4.8Nm、物理／PD／Actor周期1/5/10ms，真实共享C/FSM与V4；motor、被动力和NET分路。
- reset仍原0.34m支撑站姿、腿长0.28m。300ms后持续保留跳跃请求，直到共享C安全接受或回合结束；允许直接PUSH，不要求先完成下蹲姿势。C的PUSH超时仍1.2s，未起跳总期限3s，整回合最多6s。

BC60此前短闭环站立与前向表现只作为初始化依据，不代表已会跳，也不代表全部基础运动合格。用户允许平移漂移，当前落稳不要求回到出生位置。

## 课程

每个环境复用 `AdaptiveCourse`，100%主动跳，无被动下落复习。每32个正常PPO结束回合形成一窗；连续两窗成功率≥80%且故障率<5%晋级。连续两窗成功率<40%降一级。旧generation回合可以结算，但不能填充新档窗口；保存恢复保留课程、RNG、计数和来源。

| 档位 | 结束成功条件 | 是否记完整Jump成功 |
|---|---|---|
| 主动离地 | 先有支撑及有接触的PUSH，无外部辅助；双轮均无真实接触、轮缘净空均>5mm持续20ms，离地起点COM向上速度>0.15m/s | 否 |
| 起跳后落稳 | 上述离地＋物理apex＋双轮落稳连续200ms | 否；除非实际同时满足全部完整判据 |
| 完整2cm | 离地后COM增高≥2cm、峰值双轮净空≥5mm、apex、落稳及共享C自身完成 | 是 |
| 完整5cm | 完整判据，增高≥5cm、峰值双轮净空≥2cm | 是 |
| 完整10cm | 完整判据，增高≥10cm、峰值双轮净空≥4cm | 是 |

落稳要求双轮接触、倾角<5°、角速度范数<0.30rad/s、COM竖直速度绝对值<0.10m/s、水平速度范数<0.5m/s、root高度相对0.34m误差<3.5cm。最后一项完成是同一个Jump的C/FSM回到BALANCE，不是切换给Motion。

净空使用实际圆柱支撑函数，接触使用实际轮法向力和原物理读回；不以C_FLIGHT、轮心高度、压伸幅度或支撑估计替代。20ms／5mm使单帧抖动更难被计入，但相比旧4ms／2mm确实更难探索；这是需要后续物理数据检验的设计取舍。

当前37D Actor没有新增目标高度输入。这些高度是逐档最低验收条件，策略可用同一种更高的跳跃满足较低档；本版本不提供按命令选择任意跳高。带速度跳需要后续明确扩展共享C请求条件和课程，本版不混入该任务。

## 奖励与防刷分

所有物理事件每回合只付一次；高度只按已达到最大值的增量支付并封顶。真实事件收益在后续失败时保留，失败另扣分。蹲伏、蹬伸、腿伸长、收腿和站高均不给直接正奖励。

| 项目 | 离地／落稳档 | 完整跳跃档 |
|---|---:|---:|
| 首次真实主动离地 | +6 | +1.5 |
| 离地后高度进度 | 最多+4 | 最多+1 |
| 物理apex，按目标高度比例封顶 | 最多+2 | 最多+0.5 |
| 首次落稳 | +10 | +2.5 |
| 达到高度、净空、apex、落稳及C自身完成 | +42 | +42 |
| 故障／任务超时 | -40 | -40 |
| 时间 | -1/控制秒 | -1/控制秒 |

完整跳跃档的部分事件系数为0.25。该档不足目标高度的低跳即使落稳，也不能领取42分、不能晋级或进入完整Jump候选池。故障、机械越界与事件同帧时拒绝该帧正事件；先前有效事件仍可统计。

保留向上速度势函数 `exp(-0.2*dt)*Phi(next)-Phi(previous)`，仅在有接触的PUSH有效，终态Phi=0；在1ms→10ms聚合时对齐PPO折扣。反复压伸或保持局部姿势不能累计这项收益。它用于提供局部梯度，不能代替真实离地奖励。

按实际PPO折扣核对奖励排序：落稳档失败之前的事件收益最多12；完整档未完成的事件收益最多5.5。6秒末终态扣分的折扣值至少 `40*exp(-0.2*6)=12.0478`，再计负时间成本，因此低跳后拖延、落稳失败或晚故障不能靠折扣把未完成回合刷成正回报。配置入口拒绝破坏这项边界的过长horizon。

姿态、角速度、对称、力矩、目标变化、滑移、冲击峰、行程边缘、落地向下速度和落稳误差继续只扣分。PUSH的力矩／目标变化惩罚沿用0.05松弛系数；空中漂移权重0.05，不强迫回原地。持续成本按实际时长，目标变化与冲击峰按事件／峰值增量计费。

## PPO配方与恢复

| 参数 | 主配方 |
|---|---|
| 并行／rollout | 2048环境×64步＝131072样本/更新 |
| 优化 | 4 epochs×64 minibatches；Actor/Critic各256优化步/有效更新 |
| Actor LR | 6e-5；另备3e-5对照，其他变量相同 |
| Critic | 512/512/256，LR9e-5；先10次Critic-only更新，计入真实PPO额度 |
| 探索 | 固定物理std：腿0.06rad、轮1rad/s；raw std为0.12/0.1 |
| 分布 | 每10ms独立Normal；log_std冻结、entropy_coef=0 |
| 折扣 | gamma=0.9980019986673331，lambda=0.95 |
| 保存 | 每100更新、终态保存、最近2份及里程碑/候选/引用并集保护 |
| 时间／输出 | 训练墙钟null；产物4GiB上限、512MiB终态预留 |

2048沿用之前远端共享运行规模，不在未做资源验收时增至4096。当前prepared的本机GPU资源声明是启动前的待复核输入，不表示已满足共享显存门槛；改去远端时还要明确GPU UUID、SDK、依赖映射和持久盘。

新Critic为 `jump_isaac_skill_execution_critic_v2`，5593D／228字段；新增 `objective` 与 `takeoff_clearance_m`，与所有奖励历史一起进入实际Row/Batch Critic。旧Isaac5591D／226字段和MuJoCo6370D／235字段保持独立。完整恢复检查奖励、课程、分布、Critic、raw语义、双Adam绑定和超参；不静默接受旧5591D状态。学习状态完整恢复与PhysX `fresh_reset` 仍分开表达，不宣称恢复隐藏solver/manifold。

## 正式调用链及验证边界

配方：

- `configs/jobs/jump_first_takeoff_isaac_prepared.json`
- `configs/jobs/jump_first_takeoff_isaac_lr3e5_prepared.json`

两份均 `formal_updates=0`，未分配新额度。统一CLI计划检查：

```bash
PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python -B -m native_stand train \
  --config configs/jobs/jump_first_takeoff_isaac_prepared.json --dry-run --json
```

复用 CLI→owner/worker→公共driver→JumpAdapter→IsaacJump→共享PPO。新增 `tasks/jump/skill_course.py` 只负责目标、奖励参数及校验；没有新增trainer、worker创建入口、watcher或独立能力评估。旧奖励/Critic继续服务旧冻结消费者。

`course_success`只推进当前目标，`success`仍表示完整跳跃，三类候选分池。新课程成功可在一个10ms策略步中途发生；该正常有限终态保留为有效PPO样本，终态观测在reset前冻结，失败短步仍保持原排除规则。

CPU验证覆盖奖励排序与实际折扣、无效离地来源、同帧失败、事件幂等、混合目标scalar/device一致性、课程generation、候选分离、实际adapter样本掩码、reset、真实C/V4 Critic、BC60均值继承、固定std优化和完整保存恢复。联合231 passed（47.30s）、补充fullgraph捕获1 passed、公共资产最终39 passed；最终portable2192 passed／31历史skip／10warning（156.79s），架构235文件／0例外，Ruff0.8.4 check和240文件format通过。两份正式CLI dry-run均 `total_updates=0, starts_engine=false, writes=false`。精确命令、结果与源码身份统一保存于数据根 `jump_first_takeoff_design_20261116/`；最终收据为该目录 `recipe_delivery.json`。

这轮只做源码、CPU计算和只读计划检查，没有新增PhysX积分、GPU或正式PPO。长训前仍须用冻结新版本完成目标平台的有界物理门、探索动作下的正常终态/reset、正常PPO优化与保存恢复；每次真实更新都计入原任务剩余6210。再次分配前核对原任务账本，不能同时消费旧run余额并给新run重复分配。首个训练块应检查请求接受、真实离地、课程成功、完整成功、故障原因、实际std和吞吐；继续无离地时停止诊断，不用loss下降代替技能证据。
