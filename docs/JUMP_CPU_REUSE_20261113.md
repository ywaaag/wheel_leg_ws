# Jump 主仓迁移与 CPU 减负

本交付只验证本地 CPU 工程与脚本物理 I/O。未调用真实物理积分、GUI、CUDA 或线上 PPO。工作树 `/home/kellen/wheel_leg_jump_ws`，分支 `jump/cpu-execution-reuse`；主仓和其他工作树未修改。初始 HEAD 为 `16f81c8813d295843ab157df2a3f095d64993fb8`，随后快进到公共 `e0373b1`。领域提交依赖公共 `dd4a228`、`8e00d4a`、`e0373b1`，不复制其改动。

## 正式消费者与合同

`wheel-leg train` → owner/worker `_execute_stage` → 公共 driver → JumpAdapter/GenericPPOAdapter → jump_backend → JumpBatch → UnifiedLearningEnv → ABI4 C/FSM 与完整 V4。

Actor 为 37D、ELU、末层线性 DirectActor。PPO 保存原始 Normal 样本及其 log-prob；C 线性增益为腿 0.5、轮 10，物理目标保护与 raw 采样分开，无 tanh。输入尺度仍为 1.5rad/100rad/s。轮最终限矩 4.8Nm，腿 40Nm。Critic 保留 6370D/235 字段，1/5/10ms 时序、奖励、课程门槛、探索、网络和 DR 均未改变。

真实初始化 Actor SHA 为 `2a4d94517943559eaed51841e27be51e4672162a862ff07cd06979efc2e685fb`，来源 supervised_updates=1500；BC optimizer 不作为 PPO 恢复来源。公共双 Adam、std、RNG、课程及消费游标保留。MuJoCo 仍为 CPU 后端，`device=cuda` 和 Jump/Isaac 组合继续拒绝。

## 实际修改

| 文件 | 修改 |
|---|---|
| `simulation/jump_build.py`、`jump_backend.py` | job 边界一次校验/读取输入、一次编译，显式 binding 注入各 plant。构建身份包括九项 C/H、manifest、ABI/合同、flags、编译器 SHA/version/路径及平台。私有临时编译、排他发布；失败不留下可复用库，身份变化拒绝覆盖。 |
| `jump_probe.py`、`jump_direct_env.py`、`jump_unified.py` | 显式 execution factory，消除先构造再丢弃的 V4。合法初始几何按腿长缓存。几何/COM/姿态按 physics tick、执行 tick、reset epoch 缓存；reset、restore、显式场景变更失效。 |
| `jump_dense_env.py`、`jump_unified_learning.py` | 删除无人消费的逐 ms 日志列表及重复 packet；保留每 ms 正式奖励，在线累加 terms。普通步骤不复制 terminal Critic。Actor/Critic 输出拥有自己的存储。 |
| `jump_direct_critic.py`、`tasks/jump/controller_critic.py`、`dense_reward.py` | 复用字段布局、模型参数 epoch；删除编码前重复数组 copy 和 dataclass 深拷贝。动态名称/shape/finite 检查保留。普通与 terminal 均为同一完整 schema。 |
| `jump_shared_c.py`、`jump_shared_v4.py` | PD 正常路径直接消费 ctypes frame；诊断 dict/数组只在显式 capture 时生成。保留 C frame 与非 Jump fallback 的区别，不改变 checkpoint/Critic 数值。 |
| `jump_batch.py`、`jump_adapter.py` | 批量一次 host 动作转换；复用 step 后观测；按真实 C owner 报告控制归属；引用登记去重；接入公共 Measurement。balance 候选将早失败计入分母，不足时长且无故障仍为 UNKNOWN。 |

C/FSM 可变状态在每个 ctypes states/inputs/frames/config 中；V4、response、计时和 course RNG 各自持有。库中的全局 `g_rl_jump_log` 仅供固件 `RLJumpPlatformCycle` 日志，正式 Python 消费的 Batch/Encode/Publish/PD 不调用该日志入口。共享库不共享控制状态。MuJoCo model 的 wheel cap/物理参数可写，因此继续逐环境创建 model/data。

显式场景参数写入后调用 `batch.invalidate_observations(ids, parameters_changed=True)`。只写 q/dq 的外部诊断也需失效观测。不能绕过该边界修改 model 后继续消费旧观测。模型参数是冻结 job 状态，不被悄悄加入原物理现场 checkpoint。

## 高频消费者审查

| 频率/事件 | 必须保留的消费者 |
|---|---|
| 每 1ms | V4 响应、实际 motor/NET；已求解接触力/滑移；弹簧行程/闭链保护；DenseReward 的冲量、峰值、能量、事件和稳定时长。 |
| 每 5ms | 真实 C/FSM 及 PD、目标采用、支撑估计；实际 previous 与运输队列。 |
| 每 10ms/短步终态 | Actor/Critic、raw/log-prob、实际子步数、终止前观测和 fault。2ms 不伪装成完整 Actor 样本。 |
| 回合结束 | course.record、三类候选摘要及 cohort/generation；early failure 计入，未完成回合 reset 不制造结果。 |
| 保存/诊断 | 完整 C/V4/物理可见状态、RNG、消费游标、pending 发布；诊断 frame 仅按需。 |

原 ScoringPlant 额外 packet 列表无正式消费者，可删除。它还用 ABI3 Packet 接收 ABI4 BALANCE9，导致脚本 I/O 第一 ms 报“未知C阶段”。根因为旧 `jump_dense_env.py::integrate_control/scoring_packet` 默认 Packet 的 phase 范围，正式 UnifiedLearningEnv 已另建 UnifiedPacket。删除该冗余路径同时消除故障。原失败收据仍保留在本地 `artifacts/jump_cpu_reuse/baseline`，不覆盖。

## CPU 测量

同机、同数据根、OMP/MKL/OpenBLAS 各 1 线程。源码对照用 `git archive 16f81c8` 的隔离检出，不移动工作树 HEAD。每组启动 5 次、每组观测打包 150 次×5、reset 8 次×5，表内取中位数。无真实物理积分。

| 指标 | 基线 | 本次 | 变化 |
|---|---:|---:|---:|
| 3 环境冷构造 | 1.4303s | 0.6321s | −55.8% |
| 3 环境 C 编译次数/job | 3 | 1 | −2 次 |
| 单环境完整观测重新打包 | 2.0174ms | 1.4858ms | −26.3% |
| 单环境全量 reset | 41.2141ms | 4.1563ms | −89.9% |
| 20 次打包 Python traced peak | 351283B | 248756B | −29.2% |
| 同口径 retained | 25974B | 70516B | 增加；包含新缓存，不能声称所有内存下降。 |
| 1600 脚本 tick 的 C PD 次数 | 320 | 320 | 不变 |

上述是启动、打包和 reset 成本，不是 MuJoCo rollout 或 GPU 训练加速。Python tracemalloc 不包含所有 Torch/native 分配，也不等于累计分配次数。结构性消除了每个 PD 的 17 个字段 dict/11 个数组快照、两段 Critic 拼接中一段、正常 terminal clone、编码前动态数组 copy、每 ms 无消费者 packet。

脚本对照为了绕过旧日志自身的 phase 错误，仅在隔离基线为该日志显式传 UnifiedPacket。两端连续 1600 tick，37D Actor、6370D Critic、奖励、motor/NET 逐元素完全一致；C states/inputs/frames 字节哈希、reward memory 和 235 字段 schema 完全一致。该事实不证明真实引擎隐藏 solver 状态可逐位恢复。

## 验收与复核

联合回归243项通过（63.63s），架构206文件/0例外，Ruff0.8.4 check/format通过。原始输出、源码 SHA、命令及测量见本地 `artifacts/jump_cpu_reuse`。新增 `tests/test_jump_cpu_reuse.py` 使用真实 ABI4 单元调用和脚本物理 I/O；全局 guard 禁止 mj_step/mj_step1/mj_step2、CUDA 初始化。覆盖独立 C/V4/model/data、全量/partial/fresh reset、环形缓冲、2ms 故障、普通/terminal 布局、双 Adam/std/RNG 连续恢复、三池候选、pending 幂等发布和引用保护。

`tests/fixtures/jump_clone_check.py` 用 RuntimePaths 映射数据根，禁止旧仓及主工作树读取，验证统一 CLI dry-run、真实大 Actor 加载、fresh 双 Adam、6370/235 schema 和静态恢复。干净 clone 验证不复制模型/资产/runs。`jump_cpu_measure.py` 仅为 CPU 组件测量器，不是训练入口。

公共负责人集成时仅 cherry-pick 本领域提交；主仓 STATUS/公共 driver/engine/performance/metrics 没有领域重复修改。GPU 接入边界另见 `JUMP_GPU_BACKEND_BOUNDARY.md`，不作为 CPU 交付前置条件。
