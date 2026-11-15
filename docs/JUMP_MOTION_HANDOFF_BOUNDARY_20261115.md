# 独立 Jump 与未来 Motion 交接边界

本次只读核对 outcome 工作区源码及4300冻结绑定，并在独立 audit 工作区保存审查记录。未查询worker，未修改outcome或其他活动源码，新增C编译、物理、GPU、BC、PPO均0。以下验收为后续设计，尚未实施或分配运行额度。

先完成独立Jump的完整起落，再接Motion→Jump→Motion高层切换。独立专家网与冻结Motion适合这一划分；本设计不采用共享可训练trunk，也不将两网raw或关节输出相加。未来交接不是当前独立Jump训练的前置阻塞。

## 当前成功的实际含义

本次读取 `/home/kellen/wheel_leg_jump_outcome_ws`，基点 `66e5c85449ecdee04ab7b9c75d4b2dc80b4a7b06`，有未提交改动。读取的共享C SHA为 `11ff26b7cec8409759080fe39a9fc979b11a44491c67f8c92d93f12a7163ef30`，与用户审阅一致；`tasks/jump/outcome.py` 此刻SHA为 `c48cd13829f8a97b1b3c897e055706181920440cfbc33e93399a162390e037e1`，不同于用户此前的 `d8837e52…`。此前文件字节未在本次取得，不能推断其具体修改范围，不能把旧摘要授予新版本或活动worker。

| 实际表现 | 源码定位（相对outcome工作区） | 结论及处理 |
|---|---|---|
| 正常启用后owner置为J_JUMP | `simulation/jump_control_source/rl_jump_core.c:196` | Jump自持控制 |
| SETTLE完成后回BALANCE，未切owner | 同文件`:244–249` | 不表示Motion接管 |
| 组装upstream_owner=2、motion_success=1 | `simulation/jump_cuda_execution.py:38–41` | 是自主Jump环境输入，不是真Motion时间或能力证据 |
| motion_confirmed由BALANCE及C success生成 | `simulation/jump_isaac.py:481`、`:563` | 旧字段名不改变控制归属 |
| handover奖励项改名own_completion | `tasks/jump/unified_reward.py:43`；`:94–96`解释继承字段语义 | 本阶段成功只报告Jump自身完成，不能报告交回Motion成功 |

源码读取快照及逐文件SHA在 `artifacts/jump_execution_audit_20261114/outcome_handoff_20261115_v1/receipt.json`。该快照只证明审查字节，不是运行现场或准入证书。

## Upstream helper 的合同风险

实际表现：`RLJumpUpstreamMotorCommand` 名字含Upstream，但不是4300原no-C执行器。根因在 `rl_jump_core.c:392–394`，该函数调用 `motor_command(...,0)`；`:357`只选择raw或limited PD，`:359–362`两条路径均调用 `RLControlCompensateSpring`。CPU桥 `simulation/jump_shared_v4.py:157–167` 在非Jump返回码分支会消费该helper。

4300正式冻结 `robot_adapter.py:81` 明确 `compensation.zero_()`；SHA `aaef360b66973f97b064c4dbdf5368336afde258380dfd4dac38d33dd1788c87` 与原manifest一致。因而不能把helper注释中的“original C”解释成4300完整V4的原执行合同。当前Jump自持训练没有因此实施真实Motion切换；风险在未来误用helper，不据此热改当前C或停训。

同目标切换到另一补偿模式可能改变电机请求。`tJ=tM-comp/40`只在同PD反馈、同几何、同目标保护条件下给出瞬时腿边界匹配；不是完整Plant、经过队列的力矩匹配或双向交接证书。处理建议是分别声明真实executor合同及补偿模式，保持电机、NET与被动力分路，不把NET随owner关闭或拿来抵消软件补偿。

## 后续交接需要的连续状态

控制权在5ms控制边界原子切换，10ms推理快照携带epoch。交接记录至少包含 `{owner, epoch, request_seq, adopted_seq/tick, 实际采用的物理target, action_contract_sha, PD/executor_contract_sha, 补偿模式}`。action合同相同不能代替executor合同相同。

旧owner尚未采用的请求应失效并有记录；已经采用的物理target与previous必须保留。target历史、V4 motor FIFO、实际响应力矩、传感历史、滤波和真实q/dq连续，不能调用 `RLJumpInit/reset`、清队列或改写previous来伪造平顺。真实previous即使偏离Teacher分布，也不能伪写成旧目标。

新owner没有有效目标时，允许hold多久、从哪一个真实采用tick计龄、何时stale/fault须在合同中明确。不能漏掉target5ms等待，也不能重复叠加等待。以上为待实现约束，不宣称当前已具备通用仲裁器或hold语义。

## 回到 Motion 的独立准入

Jump自身落稳成功与Motion可接管应分别判定。4300中心解码范围为高度.30–.40m、腿中心±.15rad、轮±80rad/s；当前Jump是线性raw增益.5rad/10rad/s及独立物理目标保护，不能混用两种包络。落稳h/tilt合格不保证q/dq、轮速与实际采用目标已进入Motion接管分布；禁止先裁剪Jump adopted到Motion包络再假称连续交接。

首版可限定.34m、stationary request。`rl_jump_core.c:52–54` 的stationary只检查命令，不证明实速0；还需按已验证范围读取实际轮速、姿态和两轮支撑，区分容许漂移与命令跟踪。`outcome.py:43` 的 `stable_vxy_m_s=.5` 是Jump自身完成范围，不能自动成为4300的接管域。本轮不为Motion编造新的速度或姿态通过阈值；带速跳跃交接另建包络，不取消当前stationary guard。

## 最小后续验收顺序

1. **CPU工程**：真实decoder/C，在5ms边界及前后1ms检查双向交接、连续请求、旧epoch、过期包、保护/限矩及故障。用完整状态快照证明未reset且始终只有一个owner；P/D/C、motor与NET分列。尚未实施通用切换代码，当前不运行假交接测试代替它。
2. **静止、无起跳**：按以后明确的有界物理额度，在多合法初态下运行真实Motion→Jump-BALANCE→Motion，队列连续，回Motion保持10s。分别统计0–.5/.5–2/2–10s，硬机械门不放宽，漂移与tracking分开；失败保留，不靠隐藏fallback维持表面成功。
3. **完整起落**：独立Jump通过后，检查完整起跳、空中、落稳后的真Motion接管并保持10s。真实owner、Motion有效时间与Jump奖励分别记录。单次失败不得授accepted；带速交接另行验证。

这些工程验收不新增独立PPO能力评估或训练循环。后续任务继续复用公共owner/worker/driver，当前训练与额度不因本设计变更。
