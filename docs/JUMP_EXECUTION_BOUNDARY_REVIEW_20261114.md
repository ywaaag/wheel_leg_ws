# Jump 执行边界与训练准备复核

本轮范围：读取已有冻结来源、probe 收据和当前领域源码；编译真实 ABI4 C 并做 CPU 单元调用。新增物理积分、GPU 任务、真实 PPO、真实 BC 均为 0。未操作其他 owner 的进程、活动 worktree、模型或账本。

已经证实 Motion 与 Jump 的执行 Plant 不同。相同物理目标解码、相同 V4 合同不代表相同执行。原 probe 的 `max_good≈0.04s` 不能单独解释为 Actor 退化或摔倒。当前证据支持先处理执行边界和指标口径；尚未唯一定位全部动态差异。

## 已核实的执行差异

模型通道顺序始终是 LH/LA/LW/RH/RA/RW，模型 left 为几何右。下表不交换通道。

| 项目 | Motion4300 的冻结正式来源 | 当前 Jump ABI4 来源 | 判断 |
|---|---|---|---|
| 腿控制 | 完整 V4 的内部 PD；`compensation.zero_()` | 真实 C 的 PD→显式弹簧补偿→final，作为 V4 `external_pd` | 两种执行边界 |
| PD | Kp40、Kd2，轮 Kv0.2 | 相同名义增益 | 增益相同不足以证明 Plant 相同 |
| 腿限矩 | 40Nm，motor 与 NET 分路 | 40Nm，补偿后限矩，motor 与 NET 分路 | 不能把 NET、软件补偿与实体弹簧当成可自由叠加的驱动力 |
| 轮 driver 限矩 | 2.44694786377709Nm | 4.8Nm | 已有明确变化；未因本审查改变 |
| 两轮附加 armature，模型 ch2/ch5 | 0.00353641 / 0.00353991 kg·m² | mechanics 指定 0.003228287699 / 0.002952963759 kg·m² | 已确认来源/setter 差异；本次未启动引擎重新读回 |
| 轮粘性阻尼 B，模型 ch2/ch5 | nominal为0；随 DR 幅度生成 | 0.00066957 / 0.0025485 Nm·s/rad | 不同的名义阻力 |
| 轮 Coulomb 项 Tc，模型 ch2/ch5 | nominal为0；随 DR 幅度生成 | 0.052843 / 0.054699 Nm | 不同的名义阻力 |
| 时间分层 | 1/5/10ms；Actor 物理包龄20ms、target5ms；轮7/11ms FIFO、tau0 | 同名义分层；腿 C 的 final 请求进入 V4 响应 | 不追加另一份20ms运输，不把包龄当固件反馈延迟 |
| 观测 | Motion/NONE flags，实际采用目标为 previous | Jump37 前25复用同一编码；Teacher 查询须替换 flags | Teacher 预测不能写入 Student previous |

来源定位：

- Motion：冻结 `effective_environment.json` 的 `motion_identified_response.legacy_spring_and_C_compensation=false`；其绑定的 `robot_adapter.py:65–95` 在 `leg.step` 后将 compensation 清零，并把 motor 写入 actuator、NET 写入对向 body torque。
- Motion 绑定 `robot_adapter.py` SHA 为 `aaef360b66973f97b064c4dbdf5368336afde258380dfd4dac38d33dd1788c87`，`response_tensor.py` SHA 为 `7cdd53f8354fc1b2491b8e3b92e2b28d9e5eb02913c59ce684afe43ff6f227d3`。数据根副本与冻结 manifest 相符。
- Jump：`src/native_stand/simulation/jump_control_source/rl_jump_core.c:335` 的 motor 边界；`jump_cuda_execution.py:107` 将 C `final_model_nm` 返回给 V4。CPU `_pd` 位于 `jump_shared_v4.py:129`。
- Motion 轮 armature：冻结 `motion_wheel_matching.py:28` 直接写入并读回两轮 armature，数值不是轮 body 本身的轴向惯量。Jump：`recovery_isaac/physics.py:29` 从 mechanics 建表，`:63` 写入。同名义 V4 不覆盖这两项。
- 本次 C 构建的9份 C/H 与读取时 teaching worktree 的9份 C/H 逐 SHA 相同。其他 owner 的当前源码快照不自动等同于活动 worker 的内存源码身份。

详细绝对路径、SHA、V4 observer 参数及读回缺口均在 `artifacts/jump_execution_audit_20261114/read_only_evidence_v1.json`。旧 `mcu_execution.profile` 的弹簧占位描述不能覆盖真实正式消费者。

### 轮惯量与被动阻力补核

本项固定审查 `c5b3ad8` 的默认 PhysX 链，不查询或修改 worker。其绑定的 mechanics 文件 SHA 为 `e4c209d6eb57fc6f80d1bdae5bb4b05eb54019ea9d6cf1441c331471d6664f64`。上表两轮数值按模型 ch2/ch5 排列。

实际表现：Jump 的轮端名义阻力非零，并且左右系数不同。根因在 `recovery_isaac/physics.py:36`、`:41` 从 mechanics 读取固定 damping/Coulomb；`:66` 将主动关节的引擎内置 damping 置零、`:71` 将内置 friction 置零；`jump_isaac_scene.py:492` 计算 `-B*omega-Tc*sign(omega)`，`:504` 与 NET 一起写入对向 body wrench。这里的 Coulomb 是显式速度符号近似，不能据此宣称静摩擦或 stiction 已认证。

Motion4300 的绑定来源 `runtime_support.py:63` 构造 DR 幅度，`:98–99` 将 wheel_B/wheel_Tc 乘以相应幅度，因此 nominal幅度0时均为0。`robot_adapter.py:84` 使用 `-B*omega-Tc*tanh(omega/.1)`，`:88` 加到 actuator effort target。DR 的 Tc 参考上限写的是 `[.0528843,.054699]`，与 Jump 首轮 `.052843` 也不是完全相同的数字。不能把 Motion 的非零 DR 样本统一描述成 nominal零阻力。

纯代数示例：若两轮速度均为 `+1rad/s`，Jump 上述被动项为 `[-.05351257,-.0572475]Nm`，Motion4300 nominal为 `[0,0]`。这不是仿真结果，也不能据其量级直接断言低速失败由该项引起。

两边完整 V4 都缩放具名 wheel body 的质量/惯量。合同中 sideL 的 `right_wheel` 缩放系数为1.8792026833539217，sideR 的 `left_wheel` 为1.4615451529198569。比较应绑定源 USD、具名 body 缩放、added armature 及实际读回；两份 profile 都写 `totalJ=.004204/.0042075` 不能证明完整惯量相同。

处理建议：保持现有参数与原证据，分别记录轮 cap、added armature、V4 wheel body 缩放、B/Tc、函数形式与施加路径。`motion_no_C_spring_to_jump_C_spring_v1` 只处理瞬时腿部前馈边界，不能标为整条同Plant转换。当前追踪差异的唯一根因仍待确认。完整 SHA 与只读代数收据为 `artifacts/jump_execution_audit_20261114/wheel_load_evidence_v1.json`；本补核新增物理/GPU/PPO/BC均0。

## 真实 C 的同输入结果

新增脚本只调用 `RLJumpMotorCommand`，不运行 FSM 转移、V4 step 或物理引擎。输入显式提供 q/dq、腿长和 spread；输入与 FSM 字节在调用前后不变。

当 `q=target、dq=0、leg_length=.28m、spread=2.07742691rad`：

```text
PD = [0, 0, 0, 0, 0, 0] Nm
final_model = [+14.4273424, +14.4273424, 0, -14.4273424, +14.4273424, 0] Nm
```

这对应 Kp40 下约0.360684rad 的目标偏移，不是舍入误差。它证明执行边界差异，不能单凭此单元调用判定真实机器人应该删除补偿。

对当前 C 的腿通道，审查的瞬时关系是：

```text
Motion边界 = clip(40*(target-q) - 2*dq, -40, 40)
Jump边界   = clip(40*(target-q) - 2*dq + compensation_model, -40, 40)
target'    = target - compensation_model/40
```

| CPU 案例 | 实际 C 的腿边界最大差 | 结论 |
|---|---:|---|
| 零 PD | 9.537e-7Nm | 同输入下可匹配 |
| 左右不同几何，非零 D | 1.073e-6Nm | 映射与符号可匹配 |
| 可表示的饱和目标 | 0Nm | 当前 Jump 的 sum-before-cap 可匹配 |
| 修正后越过±1.5rad目标保护 | 代数差0Nm | 仍不可作为合法标签，不能裁剪后声称等价 |
| 几何改为另一组合，却沿用旧修正 | 残差1.833Nm | 合成反例；不是实测的队列动态误差 |

单次独立诊断共13次 C motor 单元调用。原/新目标、P、D、PD、限幅、motor-order compensation、final motor/model 均保存。构建身份含源码、头文件、ABI、编译参数与工具链；身份为 `c6b3c6f8cc880aac908455c90683367edf3bebe6386db4130696db028b3aad89`。

另一 Jump owner 已实现版本化 `motion_no_C_spring_to_jump_C_spring_v1` 标签转换；本轮不复制其运行实现，也不热改补偿。该转换可作为显式监督初始化设计，但瞬时匹配不证明经过 target FIFO、5ms PD 和变化几何后的整条执行等价。轮 cap、armature、两套 previous 历史及状态分布也未由这条代数关系消除。

更精确地说，在 PD 反馈相同、参考目标相同且采用时未被保护改写的前提下，用标签时刻的 `comp(tLabel)` 修正目标后，限矩前仍可能剩下 `comp(tPD)-comp(tLabel)`。该请求再经过限矩与左右不对称 V4 响应；不能把它直接写成实际 motor 的等式。需要真实 label/adopted sequence 对齐，不能用相邻10ms样本插值冒充5ms PD读回。

CPU 复现命令（输出目录必须不存在）：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src \
  /home/kellen/anaconda3/envs/issaclab/bin/python -B \
  scripts/audit_jump_execution_boundary.py --output /tmp/jump_boundary_new
```

## 已有 probe 的正确解释

只读分析 `teacher_motion4300_probe` 与 `teacher_executor_probe_v181`，各自收据记录12000 engine ticks、64环境、1200 policy steps，PPO/BC均0。这些是既有消耗，不是本轮新增运行。

原 probe 五组平均高度约0.2945–0.2951m，双轮接触比例约99.6%，全部组 `episode_ends=0`。其 good 高度门实际为 `abs(h-.34)<.035`。保存的768个稀疏样本中704个不满足高度门；这些样本不能替代全时段失败分项。保存 phase 为697个 BALANCE、71个 SETTLE，不能把 phase 转换当 reset 或死亡。

原脚本的 vx 读模型 X、yaw 读世界 Z；v181 改为 forward=-模型 Z、yaw=模型 Y，且加入目标转换。因此两份命令误差不是只改补偿的 A/B，不能直接作误差改善的因果结论。高度、回合结束计数仍是可分别读取的记录。

v181 平均高度约0.354m，768个保存样本均在高度门内，保存 phase 全为 BALANCE，回合结束数仍为0。其各命令结果如下；这是12秒有限诊断，未授予能力通过：

| 指令 vx/yaw | good 比例 | 最大连续 good | 最后保存样本的平均 vx/yaw |
|---|---:|---:|---:|
| 0 / 0 | 99.75% | 11.97s | +0.1394 / -0.0212 |
| +0.2 / 0 | 97.83% | 11.74s | +0.2681 / -0.0188 |
| -0.2 / 0 | 0.58% | 0.07s | -0.0212 / -0.0117 |
| 0 / +0.3 | 99.42% | 11.87s | +0.0894 / +0.0053 |
| 0 / -0.3 | 10.81% | 1.32s | +0.1749 / -0.3067 |

尤其 `yaw=+0.3` 的 good 高，不代表转向跟踪好：其 yaw RMSE 为0.29285rad/s，而 good 允许误差小于0.30rad/s。零命令也存在约0.139m/s 后段漂移。不要将这些 good 数字直接转授成全命令成功 Teacher；本审查不改奖励、课程门槛或探索分布。

probe 收据没有同一 PD tick 的 held q/dq、P/D、补偿、adopted 序列、实际 V4 motor/NET，也没有 good 的逐条件失败计数和完整 quaternion/gyro/gravity。不能从现有稀疏样本补造这些字段。后续已有授权块如继续执行，应在同一次运行补齐这些读回，不为本审查再开一组探针矩阵。旧 construction 的 CPU C 字符串与实际 device 请求也不一致；旧文件保留，新收据应从实际执行消费者读取身份。

### 消费现有采集与原始诊断

已逐SHA读取 `2026_11_15__jump_closed_loop_teaching_v2/supervision/phase_000_collect/collection_00001200.json` 绑定的19个NPZ，共76800条、64环境×1200 policy步。manifest 的768000物理步是原有消耗，本轮新增物理步为0。样本ID及 env/episode/policy_index 组合均唯一；记录的全部 transition 为10个子步，101条带结束标志。这是采集内容核验，不是成功率或新训练统计。

可用字段为37D pre-action观测、env/episode/policy_index/sample_id、owner、transition/end标志、Teacher-safe及四个标签时刻补偿条件。缺少网络输出、请求/保护/采用目标序列、PD tick、held q/dq、P/D、PD时刻补偿、实际motor/NET/passive、同期root姿态与接触。因此只能按样本身份对齐，无法从该collection闭合标签→采用→出力链；不能零填、借其他run拼接或凭previous数值推造采用序列。

观测有效性有一项需单列：51条37D观测全零，但 `teacher_safe=true`、owner=2，不能作为有效Teacher状态。其根因待确认，采集未保存同期C encode返回码/Actor-valid掩码。现有 teaching分支的 `motion_teacher_eligible` 已在本次CPU调用中拒绝这51条全零输入（实际资格数0），故不重复开发过滤器，也不修改旧数据；该调用不证明过去的label阶段使用了同一源码。`transition_valid` 不能代替 Actor/Teacher资格。

另外逐行消费了 v181 的 step100..1100稀疏保存样本，沿用原有门槛，没有重算完整good或授成功：

| 命令 | 保存样本数 | 实际范围 | 已能解释的现象 |
|---|---:|---|---|
| vx=-.2 | 143 | vx=-.02236～-.00699m/s | 143/143不满足原vx误差<.15门 |
| yaw=+.3 | 143 | yaw=+.00164～+.01196rad/s | 同向但响应很弱；仍全部通过原yaw误差<.30门，不能称跟踪成功 |
| yaw=-.3 | 132 | yaw=-.32898～-.30279rad/s；vx=+.12857～+.17680m/s | 无错向；120/132不满足原零vx误差<.15门，漂移与转向响应要分开 |

这三组保存样本的高度门均通过。缺少tilt/restricted/逐轮接触，不能凭这张表补齐其余质量门或唯一归因于轮机械差异。

4300原 `diagnostics/update_04300.json`（update=4300）直接报告轮延迟 ch2=6–8ms、ch5=10–12ms，tau最大均0；`old_wheel20ms_tau2_added=false`、`separate_V4_motor_NET=true`，motor统计排除NET。该JSON没有 spring/compensation数值字段；Cspring为0由SHA绑定的 `robot_adapter.compensation.zero_()` 证实，不能写成该JSON独立记录了逐tick零补偿。旧20ms/tau2占位不应叠加到这条正式链。

本次消费收据为 `saved_alignment_evidence_v2.json`、`current_teacher_guard_v1.json`、`v181_response_from_saved_samples_v1.json`，均在现有审查artifact目录。未查询worker、未新建标签、未推理Teacher、未优化或采样。

## 小 Actor 只读来源准备

`configs/contracts/jump_motion1600_C0_reference_v1.json` 固定5ec49436…小 Actor、原输出合同、support center与历史子域；它不是正式训练配方，不使 Teacher 自动 qualified。

公共 inference factory 已在1488条原始 golden 上验证 mean 逐位差0，物理腿目标最大差5.833e-8rad、轮目标最大差1.908e-6rad/s；转 Jump raw 再解码的物理差0。raw 最大绝对值5.424，未裁到1；无二次 tanh。Actor、RNG、Student previous 均未改变。ONNX本轮只验SHA，未执行。原 golden/final 没有回灌拟合。

本工作区早期未提交的参考查询函数与另一 owner 的版本化 observation adapter 重叠，已连同原diff保存在 `artifacts/jump_motion_reference_20261114/`，不再向正式 runtime 引入第二套 API。`check_source_v3.py` 通过离线 snapshot 复核；此前 v2 收据和一次误读 legacy trace.mean 的早期收据全部保留。该 trace.mean 是旧执行桥输出，不能当网络 raw mean 判定 Actor 不一致。

小 Actor 的历史62/62存活、50/62任务，和未通过的yaw±1、未独立测量的低速子域均原样保留；`full_task_accepted=false`、当前 PhysX Plant 资格 UNKNOWN。其历史 V4 同样不额外加本次 C 补偿，因此替换 Teacher 不能绕过执行差异。

### 参考来源的可用范围

| 来源 | 本次可用性核验 | 可引用的历史范围 | 不可继承的结论 |
|---|---|---|---|
| 1600+C0小Actor，SHA5ec49436… | student.pt、原output_contract及support_center三项SHA再次匹配 | 原V4站立/扰动恢复；有±1/2/3直线证据 | ±.2直线、±.3yaw未验证；yaw±1承重84–88%失败，后漂-.203m/s；不授Jump Plant或完整双向基础能力 |
| 大source855 | 原 `runs/2026_11_2-09_40__motion852_groundheight691_new10000/formal/model_update_855.pt` 不存在，默认数据映射也不存在；所查根映射未定位可加载替代 | 用户提供的历史证据：±.25/±.5直线MAE .055–.068，四项质量通过；±.5yaw因支撑<95%失败 | 历史表不能变成不存在的可用Teacher路径 |
| 小855 ONNX，SHA9d417567… | 已定位 `deployment_protocol/2026_11_3-16_30__motion855_small31_antiosc_onnx/actor.onnx` 并验SHA；manifest与reference_adapter已读 | manifest记录PT28/33、holdout28/33保真，full_fidelity_passed=false | 解码为reference+1.5×mean腿、100×mean轮，无C0中心；不能套当前C0转换，不能继承大855低速通过表 |

因此目前没有现成单Teacher证据保证当前Jump Plant的全部低速。小855文件可读不等于其闭环能力已认证；本轮未运行ONNX、重建缺失大模型或以旧final/holdout拟合新标签。

## 验证和剩余门槛

- 领域 CPU56项通过（21.59s），其中新增真实 C 边界5项；使用 portable guard 禁止 mj_step、CUDA 初始化及 AppLauncher。
- 离线参考查询6项通过（0.85s）；1488 golden 数值验证单独通过，不计为真实 BC。
- 公共生命周期、phase2/3与架构规则169项通过（53.85s）；无失败或skip，收据为 `CPU_lifecycle_tests_v1.xml`。
- 架构检查219文件、0例外；新增脚本与测试 Ruff0.8.4 check/format通过。
- 原始收据在 `artifacts/jump_execution_audit_20261114/` 和 `artifacts/jump_motion_reference_20261114/`；旧probe未重写，运行分支未同步。

目前不能宣告全部长训前置已通过。读取时数据盘可用6032084992字节（约5.62GiB），低于既有8589934592字节资源门。本轮没有请求新阶段，因此没有调用公共helper，也不存在本轮helper拒绝记录。下一物理/BC阶段仍由唯一 owner 执行真实helper前检；不自动删run、降低资源门或接管Recovery。

后续应复用已有 owner 的串行监督块与正常PPO结束回合：保持显式执行合同，按真实状态/命令域授标签，保留 UNKNOWN，不把本次边界验证当完整起跳、落地或长期稳态认证。目标转换后的学生在同Plant上的覆盖、队列时序及正常保存恢复证据仍由对应执行块提供；不另开独立PPO能力评估。
