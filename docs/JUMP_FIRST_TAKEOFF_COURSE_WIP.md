# Jump 暂停现场：新课程与配方 WIP

2026-11-16用户明确要求暂停工作、暂停训练、保存现场，之后由用户重新开启Jump项目。不要自动继续开发、物理诊断、PPO、BC或resume。

## 位置和冻结边界

- 新开发worktree：`/home/kellen/wheel_leg_jump_first_takeoff_ws`，分支`jump/first-takeoff-course`。
- 从当前main `31aaeb9`建立，合并原Jump远端实现`08a4683`得到基线`b34d974`。主仓、Motion、Recovery及远端活动源码均未热改。
- 原远端源码冻结实现`5d1b1ec`，run `/root/gpufree-data/wheel_leg_jump/outcome_migration_20261115/runs/jump_outcome_shared2048_remaining7573`。本次已从该冻结wrapper调用统一CLI `stop --request-only`，不发OS信号、不强杀。
- 最终停训状态、checkpoint SHA与预算以数据根收据 `jump_first_takeoff_design_20261116/pause_terminal_verified.json` 为准。远端此前额度7573、此前已耗2427；本次终态实际用了N次，则任务累计2427+N、剩7573-N，不能刷新或退款。
- WIP没有提交新训练，prepared配置`formal_updates=0`；不能把旧run的剩余额度自动复制给新版本。

## 已完成的定位

旧u1000窗口68039回合中61640是NO_TAKEOFF、6393任务超时、6个旧目标故障；不是同等比例摔倒。两环境本机2200世界tick诊断中会反复压伸、质心向上峰.54/.48m/s，双轮没有同时离地；电机22～29Nm，没有触及40Nm限矩。C估计与实际轮力差中位约.4～.5N。首次物理起跳瓶颈与已有事实一致；不能据本机6.0.1短诊断宣称远端6.2复验。

原配方过程势函数终态抵消是设计行为，不是算错；仍有即时shaping信号。完整落稳之前的任务回报稀疏，腿探索std从初始.06rad降至约.03rad；这些是瓶颈解释，尚无受控对照证明唯一因果。原始定位在 `/home/kellen/wheel_leg_jump_remote_ws/artifacts/jump_stall_diagnosis_20261116/DIAGNOSIS.md`。

## 已写入但未验收的新设计

1. 复用现有课程，扩展具名`SkillLevel`与`SkillTaskSpec`，目标依次为主动离地、低跳落稳、完整2/5/10cm。晋级只消费正常PPO结束回合。第一阶段离地成功会结束回合，**仅记课程成功，不记完整跳跃成功，也不发布完整Jump候选**。
2. 新离地目标草案：轮真实接触均无、轮缘净空均超过5mm、连续20ms、离地起点向上速度>.15m/s，必须此前有地面支撑、PUSH控制与无外部辅助。不是轮心升高或C_FLIGHT计数。
3. 每回合首次起跳6分、起跳后高度进度最多4分、顶点最多2分、落稳10分、完整完成42分。中间真实事件收益不因随后失败全部抵消，失败-24、未起跳超时-24、时间-1/s；蹲伏/反复伸腿不给直接正收益。已有向上速度势函数保留，仍终态抵消。数值为草案，需反刷分及排序测试。
4. 固定物理Normal探索：腿.06rad、轮1rad/s，每10ms按原raw分布采样，raw输出增益.5/10。固定log_std不训练，entropy_coef=0；不偷偷过滤、相关采样或改执行器。
5. Actor学习率6e-5主配方及3e-5对照，Critic9e-5、10次新Critic预热；2048×64、4epochs×64minibatches、save100、墙钟null。来源为原BC60 Actor-only，新Critic/双Adam，**明确不是旧PPO完整恢复**。旧完整checkpoint保留。
6. Actor仍37D ABI4，原Cylinder+native Plane、40/4.8Nm、1/5/10ms。Critic新增目标与离地净空两个字段，实际CPU C/V4布局生成5593D/228字段，版本`jump_isaac_skill_execution_critic_v2`；旧5591D/226字段独立保留。
7. 配方用SHA绑定的`assets.value_files`读取现有小型合同JSON，避免再复制2000多行schema；入口扩展位于artifacts/runtime_paths和experiments/config。此公共接口仍待消费者回归与架构说明，不能当作已完成公共能力。

## 实际验证状态

- 合并基线b34d974：portable2140通过、31历史skip、10warning，140.86s；当前main架构检查通过，固定Ruff0.8.4 check/format通过（233文件）。这些结果只认证合并基线，不认证之后WIP。
- 中间态旧奖励相关16项通过，9.12s；该测试之后又有修改，不能拿它冒充当前WIP全部通过。
- 新Critic用真实CPU C/V4及实际奖励字段生成schema，5593D/228字段；未启动PhysX。
- 新主配方统一CLI train dry-run通过，明确`total_updates=0`、`starts_engine=false`、`writes=false`。只是计划解析，不是可训练或能力验收。
- 新语义CPU/GPU一致性、恢复、优化器绑定、晋级/候选语义、防刷分、资产读边界回归和正式物理/PPO均未完成。

## 用户重新开启后从这里继续

先读AGENTS、STATUS、本文件及最终停训收据，核实Git与原run终态；不要直接启动prepared配方。

需要完成：

- 新奖励对蹲伏、反复压伸、被动掉落、单轮离地、短暂接触抖动、下降收腿、辅助起跳及同tick失败的拒绝；事件只付一次、失败后保留但收益上限低于完整跳跃。
- 混合课程目标CPU/device逐步对照、首次课程终态、partial reset及新reward state序列化，确认goal布局和所有额外记忆都进Critic。
- 课程晋级/降级、跨generation、早期课程成功与最终候选严格分离；核对旧MuJoCo/Isaac消费者未受影响。
- 固定std在Actor-only首次构造与完整恢复、双Adam、真实优化、推理/部署合同及保存加载一致；拒绝旧5591D完整PPO/新5593D混用。学习率对照控制其他变量相同。
- 审阅`asset_values`公共入口的相对路径/RuntimePaths/SHA/冲突拒绝及旧引用行为；补测试和ARCHITECTURE。此次暂停不继续扩大实现。
- 相关CPU、portable、固定Ruff、架构和dry-run；必要时受授权有界本机/远端物理、正常PPO保存恢复，再按终态实际账本分配剩额。若旧run已完整保存，不能重新消费它的未用额度并重复分配给新run。

当前代码是保存工作的WIP，不是正式交付。所有新配置保持零额度，只有用户重新开启项目才继续。
