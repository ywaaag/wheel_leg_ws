# Jump Isaac GPU 最小边界

当前正式 ABI4 链仅 MuJoCo CPU。本文件是后续接入合同，不注册假后端，不创建第二套 trainer/registry/launcher。CPU 交付独立集成。

## 复用与后端职责

沿用统一 CLI→owner/worker→driver→JumpAdapter/GenericPPOAdapter、DirectActor/raw Gaussian、C/FSM、Reward/Course、三池候选和 ModelIndex。后端只负责场景、物理 I/O、批量化和引擎实际状态读回。MuJoCo 保留播放器和 CPU 数值参考。

后端必须提供 observations、批量 step(raw)、partial reset(ids)、runtime_state/load_runtime_state、set_measurement。step 返回：普通观测、每环境奖励、terminated/truncated/done、reset 前 terminal 观测、每环境实际物理子步数、真实 C owner/phase、transition_valid、故障原因、实际物理条件。1/5/10ms、actual previous、motor/passive/NET 分路不变。phase 变化不能 reset；未走满10ms不产生合法 Actor 样本。结束回合继续由正常 PPO 收集，不能加独立能力评估。

初始化只加载已匹配的 Actor-only 权重；后续同版本恢复保留双 Adam/std/RNG/course/cursor/pending。训练状态恢复与引擎物理现场恢复分别标记。PhysX 不能承诺恢复所有隐藏 solver 状态；fresh-reset 必须清除未结束回合证据，不能承接原离地/落地/stable dwell。不能把 MuJoCo full state 当 Isaac full resume。

## C 与 GPU 的边界

现有 C 接口已有 Batch/Encode/Publish/PD，状态显式逐环境持有。最小正确版本可采用一个 n 环境 host C 批次：5ms FSM/PD、10ms Encode/Publish，批次之间保留状态。物理/INS/V4 仍每1ms。需单独量化反馈 D→H、C 调用、目标 H→D、V4 与奖励同步成本；它们不是 CUDA kernel 时间。

现有 SharedV4Execution 的 ctypes/NumPy/Python 循环不是 GPU kernel，不能仅更改 device 即称为 GPU 后端。若要 GPU 执行 C 控制逻辑，必须把同一 C 核心编译成可验证的批处理设备实现，逐项验证 phase/owner/timer/adoption/previous 与 CPU；不能重写 Python FSM。第一版可接受 host C 桥的性能代价，但不能省略 1ms 回读/反馈或更改控制年龄来规避同步。

还有两个实际前置问题：当前支撑估计依赖 MuJoCo 全状态逆动力学、闭链 Jacobian 和 qacc；不能把 Isaac 接触真值灌入 Actor 支撑信号冒充传感器。C 外部 IMU specific acceleration、driver torque、internal load 和 support_J 都必须有真实、带时间戳的来源。完整动力学桥或明确版本化的支撑模型缺失时拒绝启动。

## 6370D/235 字段逐项可构造性

本次 CPU schema 不改。后续机器可读清单应为每个具体字段列出源张量、顺序/单位/shape/时间戳、reset/restore 行为及实测证据，缺少任一项不能以零填充通过。

| 字段组（数量/列） | Isaac 可构造性与必要证据 |
|---|---|
| C（36/104） | 同一 C states/frames，逐字段可直接构造；需要 Batch 每环境状态与 tick 语义验收。 |
| input（27/68） | 同一 Input 结构可编码；support/internal_tau/加速度来源尚未实现，不能用接触真值替换。 |
| Actor（1/37） | 同一 C Encode，保留 V4 观测年龄、actual previous 和真实 phase。 |
| physics.qpos（45）、qvel（44） | Isaac root pose/quaternion、linear/angular velocity、具名全部 active/passive joint 状态按明确坐标约定重排；不能只填6个电机。实际 USD articulation 拓扑需读回。 |
| physics.qacc（44） | Isaac 部分 joint_acc 为有限差分，root 加速度和 MuJoCo 广义加速度约定需核验；不能静默视为同语义。 |
| physics.qacc_warmstart（44） | MuJoCo solver 状态；没有已证明的 PhysX 同义读回。阻断沿用原 CPU schema。 |
| physics.ctrl（6） | 可保留实际采用的 motor effort 命令；与 applied torque 分开。 |
| physics.qfrc_applied（44）、xfrc_applied（40×6） | 可记录后端实际施加的 NET/passive/external 力及 wrench，但必须具名、坐标明确并与实际引擎读回核对。 |
| physics.qfrc_actuator（44） | 实际电机力矩与被动关节广义力需要可验证映射；不能用未经过响应/限矩的 PD demand。 |
| model.body_mass（40）、body_inertia（40×3） | Isaac native mass/inertia getter 可读；USD fixed-body 合并与转动惯量坐标/主轴映射需核验，不得重复计质量。 |
| model.dof_armature/damping/frictionloss（各44） | native getter/显式被动力配置按全部具名 DOF 读回。V4 已替代的作用不得再施加。PhysX 摩擦语义不自动等于 Coulomb loss。 |
| model.geom_friction（41×3） | MuJoCo sliding/torsional/rolling 与 PhysX static/dynamic/restitution 含义不同；不能把三值凑齐当相同物理量。 |
| model.eq_solref（4×2）、eq_solimp（4×5） | MuJoCo equality solver 参数；PhysX 闭链设置没有已证明一一对应。阻断原 schema。 |
| execution（1/2） | 显式控制 subcycle，可直接构造。 |
| V4（41/3432）、response（29/1648） | 原 tensor kernel/全部 FIFO 与 timestamps，逐字段可构造；保持128环形队列、16观测历史、partial reset 和 age，不压缩。 |
| task（8/8）、reward（12/12）、dense（61/102） | 复用同一任务/奖励状态，可直接构造；逐1ms在线更新与普通/terminal 同定义。 |
| request/elapsed（3/3） | 显式 episode/request 时钟，可直接构造；fresh reset 不承接旧 dwell。 |

合计235字段/6370列。qacc_warmstart、eq_solref、eq_solimp、geom_friction 等不是“还没接线”的普通零值，必须作版本化语义决策。可选路线为新增明确的 Isaac Critic schema、用真实 PhysX 状态并从 fresh Critic 开始；保留 Actor 接口、任务算法和全部控制历史。若维持原6370列硬合同，则须先实现并证明逐项真实等价，不能训练后再补说明。CPU schema 永不被此决策覆盖。

后续真实 GPU 开启前需要：SDK/资产/全部DOF和闭链/NET映射、支撑模型及上述 Critic 决策、脚本 CPU 适配器测试、实际物理小批步进与partial reset/终态、完整学习保存恢复、受控并行显存和性能测量。工程通过不代表稳定跳跃；能力从正常 PPO 结束回合逐步产生。
