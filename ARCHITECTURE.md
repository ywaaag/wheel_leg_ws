# 工程结构与依赖方向

## 主调用链

```text
wheel-leg / python -m native_stand          cli.py：参数、展示、退出码
  → experiments/config.py                  一次解析，形成不可变JobConfig
  → experiments/plan.py                    --dry-run 共用同一份计划
  → simulation/dispatch.py                 操作×任务×算法×后端；execution_service()是唯一判定点
  → experiments/owner.py                   建run、写run.json/state、一次握手、停止、续训
  → experiments/worker.py                  阶段循环、计费、指标、checkpoint、终态、报告
  → simulation/driver.py                   公共阶段驱动器：调度、计费、保存、边界、资源关闭
  → simulation/engine.py                   通用PPO适配者（GenericPPOAdapter）
  → simulation/isaac/motion_backend.py     生产Motion适配者（_Adapter：V4环境+受控PPO）
  → learning/ppo/networks.py               显式输入/隐层，state_dict与既有Student一致
  → learning/ppo/assembly.py               观测列、两路学习率、完整恢复状态
  → learning/ppo/evaluate.py               完整评估（逐条件逐环境）
  → learning/distillation/loop.py          BC蒸馏与选模
  → simulation/<backend>/                  引擎适配；纯控制与时序在 simulation/timing.py
  → artifacts/                             IO、checkpoint、指标、评估、保存策略、模型引用、报告
  → integrations/hapi.py                   可选通知
```

## 模块职责

| 模块 | 职责 | 不做什么 |
|---|---|---|
| `cli.py` | 参数、帮助、展示、退出码 | 不实现训练、不写状态文件 |
| `experiments/config.py` | preset 加显式覆盖、严格校验、路径解析 | 不读运行状态 |
| `experiments/plan.py` | 只读估算与计划 | 不创建文件 |
| `experiments/owner.py` | run 生命周期、进程身份、停止请求 | 不训练 |
| `experiments/worker.py` | 阶段执行、计费、终态、报告 | 不解析CLI |
| `experiments/budget.py` | 单文件预算账本，attempt 级记账 | 不存配置副本 |
| `experiments/state.py` | `state.json` 单写入者协议 | 不从曲线反推消耗 |
| `tasks/motion` | 31D 输入编码、6D 解码、支撑中心、语义记录 | 不做项目健康检查 |
| `tasks/recovery` | 起身门槛、保持时间、进入条件 | 不复用 Motion 动作版本 |
| `learning/ppo` | 显式隐层构造、受控PPO装配、warmup 语义 | 不读状态文件 |
| `learning/distillation` | Teacher路由、单位转换、确定性BC数学循环 | 不创建worker、不消费holdout补教 |
| `simulation` | 时相纯函数、后端适配、公共阶段驱动器、分派表 | 不依赖 HAPI 或文档 |
| `learning/ppo/evaluate.py` | 完整评估执行 | 不授予能力结论 |
| `learning/distillation/loop.py` | BC蒸馏与选模 | 不用留出集补教 |
| `artifacts` | IO、checkpoint、指标、评估、保存策略、模型引用、报告 | 不知道日期配方 |
| `integrations/hapi.py` | 可选通知，异常不外抛 | 决定能否训练 |

## 依赖方向

- `tasks` 不导入 `cli`、`owner`、`report`。
- `learning` 不导入脚本或 CLI。
- `simulation` 不依赖 `integrations` 与文档。
- `artifacts` 不导入任务模块。
- `simulation/dispatch.py` 是唯一的组合判定点；后端可用性由后端 `available(job)` 决定。
- `artifacts/io.py` 是文件SHA、规范化JSON摘要、原子写入、NPZ 的唯一实现；新代码不要再写本地 helper。
- 函数内延迟导入只用于 Isaac、torch 等重依赖，不用来隐藏模块循环。

## 关键不变量

- `action` 与 `observation` 的尺度分别校验：输入 1.5 rad / 100 rad/s；输出按显式动作版本为0.15 rad / 40或80 rad/s，当前Motion／Spin配方为80，不能随输出改输入。
- `previous` 只由实际采用目标换算（`tasks/motion/encoding.py::previous_from_adopted`）。
- 完整恢复状态必须包含 Actor、Critic、optimizer、std、RNG、计数与课程状态；Actor-only 导出不能替代。
- Critic-only warmup 次数来自配置；完整恢复的 formal 不重复 warmup。
- 保存策略只清理本run中未选中、未引用的中间 checkpoint。

## 历史实现与保护

`command_v6/`、`students/`、`recovery/`、`world_model/` 是冻结的历史实现，按原身份只读复用。新支撑中心Motion通过 `simulation/isaac/motion_backend.py` 使用统一worker、账本和checkpoint；物理环境只读借用r4视图，修正奖励与更新后KL位于 `tasks/motion/scale_corrections.py`、`learning/ppo/supported.py`。不启动旧supervisor。

祖先 `isolated/` 视图中以下四个文件仍软链接到仓库源码，继续作为冻结依赖保护：

- `src/native_stand/students/motion_supported_output.py`
- `src/native_stand/students/motion_centered_output.py`
- `src/native_stand/students/motion_kinematic_output.py`
- `src/native_stand/command_v6/tools/validate_cpu.py`

这些文件可能被新作业复用的环境直接读取，不能在运行时修改。

## 扩展位置与完成边界

在现有组件内接线，不设计插件发现系统，不要求每个功能新建类。配置可表达的差异先用配置；未知或未实现的操作/任务/后端组合由dispatch在引擎前拒绝，不用静默fallback或补零伪造支持。

| 变化 | 落点与复用方式 | 正式消费者/接线 |
|---|---|---|
| 超参、网络宽度、实验配方 | `configs/jobs`、`experiments/config.py`，复用网络构造器 | CLI解析与dry-run消费同一JobConfig |
| 新技能、奖励、课程、终止 | `tasks`；任务声明在`tasks/registry.py`，任务内课程/统计 | 接任务定义、观测/奖励/结束摘要及实际adapter |
| 网络、损失、算法 | `learning`；`learning/policies/factory.py`与PPO装配 | PPO/BC数学循环不同，复用owner/worker/driver/预算与恢复 |
| 后端、场景/地形、时序/执行 | `simulation`；`dispatch.py`能力选择、backend factory、`scenes.py` | 当前任务环境实际消费并读回，不只注册名称 |
| 多Teacher、聚合、补教 | `experiments/supervision_flow.py`、`learning/distillation/routing.py`、`artifacts/dataset.py` | 原collect/label/fit流程、数据/episode/split身份，不另建监督worker |
| 播放、sim2sim、部署 | `learning/inference.py`公共推理factory，`simulation/playback.py`与`simulation/sim2sim.py` | 任务合同/执行适配，复用唯一GUI/headless循环与部署golden |
| 模型/数据、恢复与保留 | `artifacts`；`ModelIndex`、`RuntimePaths`、LoadedSource/checkpoint/handoff | 唯一引用权威与加载边界；预算不随路径搬迁重置 |

实现前说明所属模块、复用能力和新增必要性。完成必须同时具有实现、正式入口/注册、必要配置/合同、实际消费者测试及当前文档。helper/独立演示不是正式接通证据。替换旧实现须审阅剩余消费者：无用途分支退役，冻结历史依赖保持原身份；不因抽象一致而合并不同控制语义。

新增配方在`configs/jobs`使用preset加显式覆盖，入口严格拒绝未知字段；执行`wheel-leg train --config ... --dry-run`检查计划，不复制旧spec或日期trainer。公开CLI/任务/后端支持变化必须一起审阅help、配置示例或当前说明，内部bug不要求机械修改README。

## 工具与脚本边界

一次性定位、打印、复现和微基准先放临时目录；有用证据持久化到run/artifacts，不把/tmp当长期依赖。长期训练、蒸馏、播放、导出、数据处理进入正式模块与统一CLI，不新增另一套launcher或训练循环。已有历史工具不批量整改；判断实际职责，不按文件名一刀切。

`scripts`允许离线分析、编译/打包、契约验证、受控测试和可复用微基准。工具调用正式模块，正式模块不导入scripts。新增工具需在此简述职责及与已有工具的区别，无需另建README/报告。例如现有`benchmark_recovery_reset_cpu.py`只测脚本I/O下reset组件与调用数；`prepare_recovery_continuation.py`只读核对来源/预算并派生准备材料，不提交worker；`test_portable_cpu.py`选择明确PORTABLE测试并启用物理/CUDA保护。

`check_architecture.py`为纯标准库静态检查，不执行目标脚本：无参数维持全量正式源码依赖/入口/全局替换检查；`--base refactor-migration-v1`额外审查tag之后的Python/shell脚本（含已提交、暂存、工作树和未忽略新增文件）。Git使用NUL列表，删除跳过、重命名检查目标。无效/缺失基点退出2；确定违规退出1；动态命令/进程target及未支持脚本语言输出REVIEW供人工核对，不声称理解任意动态行为、跨文件间接调用或所有代码重复。编译器、pytest和打包subprocess不是一律违规。公开入口/声明变化输出PUBLIC-REVIEW，不因任意文档被修改就自动认证内容正确。

## 集成门槛

按`.github/PULL_REQUEST_TEMPLATE.md`回答问题、扩展位置/消费者、复用/新增理由、替代路径、文档影响、验证与限制。合入main前审阅正式调用链、唯一生命周期、支持说明、相关测试/架构/Ruff，以及模型/数据/预算/控制合同未意外改变。CI获取完整历史和tag，执行增量检查；当前没有remote或required checks，CI文件存在不代表合并已被平台阻止。未来配置远端后再设置required checks，本次不新增远端或push。

迁移已按`refactor-migration-v1`冻结范围结项；模型能力、持续提速、Recovery新配方、Jump新后端均为后续普通任务，不重启迁移项目。新工作从当前main建独立分支/worktree，活动领域worktree及训练冻结版本不被rebase或热改。

## 生产路径与公共驱动器

`motion_backend.run()` 用 `_Adapter` 把生产步骤函数接进 `driver.run_stage`：

- `run()` 负责构造 V4 环境与受控PPO、按 `_resolve_source()` 恢复；
- `_Adapter.update_once()` 执行一次 rollout + 受控PPO 更新并返回指标；
- 阶段额度、计费、停止/样本/墙钟/空间边界、周期与终态保存、`close()` 全部由驱动器执行；
- 生产算法（KL 约束、辅助损失、控制 mask、奖励修正、课程）保持原样，未被替换为通用PPO。

`worker._execute_stage()` 按 `dispatch.execution_service()` 选适配器；评估用
`dispatch.backend_for_evaluation()` 走同一个服务选择，因此生产配置不会落到要求
`legacy_spec_path` 的通用Isaac适配器。

## 恢复来源

`motion_backend._resolve_source()` 与 `engine.initialization_payload()` 遵循同一优先级：
显式 resume → 阶段接续 → 配置声明的初始化。三种来源都只解析一次，后续适配器消费该结果。


## 架构约束（2026-11-08 起，由 scripts/check_architecture.py 执行）

| 模块 | 职责 | 不得依赖 |
|---|---|---|
| cli.py | 唯一正式入口（train/distill/resume/inspect/stop/evaluate） | — |
| experiments | owner/worker/状态/停止/预算/结算，唯一的训练进程创建点 | scripts |
| learning | PPO/BC 数学循环、蒸馏、网络与优化 | cli |
| tasks | 观测/动作/奖励/终止的任务语义与单位契约 | cli、owner、report |
| simulation | 后端、驱动器、环境交互；执行判定收敛于 dispatch.execution_service | integrations |
| artifacts | checkpoint/评估/保留的身份与原子写入 | tasks |

新增能力接入：配置可表达→只加配置；新组件→落在所属模块并由配置选择；显式 factory 参数注入，禁止跨模块改写模块全局对象。模型形状变化用配置表达；物理语义变化（列序/归一化/单位/包络/时序）必须有来源与目标契约。检查：`python scripts/check_architecture.py`（违规退出1，参数错误退出2）。

## Motion／Spin领域组合

Motion目标与能力证据的共同入口为[Motion能力矩阵](docs/MOTION_CAPABILITY_MATRIX.md)，版本`motion_capability_matrix_v1`。涉及Motion课程、奖励/评分、播放器和部署的开发先按该矩阵核对场景约束及证据等级；0.34m基础持续保留，只有平地站立/直线要求双腿对称，转弯允许内倾和腿长差。目标文档不表示运行实现已接通或模型已通过。

配方复用`configs/jobs`，课程/统计/候选复用`tasks/motion`，执行复用生产adapter，逐模型成绩和引用复用`artifacts`及ModelIndex；不新增平行训练、评估或模型索引。目标按配方及实际消费者逐项接线，不回写冻结训练；当前范围见下述v3准备配方。

`motion48_turn_focus_branch_v3`复用SplitCourse，增加显式focus权重、初始档位、基础保留晋级门槛；stage将75%行进转弯响应门槛传给同一maneuver_statistics，原地自转和旧课程默认不变。未实现全矩阵自动调度或高度前置门槛。u1500的`full_motion_course_preparation_v1`由既有artifacts.branch_initialization读取真实run/state/budget快照及SHA，在完整来源边界只重建课程证据，保留学习状态和独立模型进度；生产adapter复用同一依赖登记、课程、保存恢复，不增加owner或worker。该记录仅支持零额度准备，require_allocated始终拒绝，不能替代终态结算与正式接续。

两者保持task=motion，由不同配置选择`CourseStrategy`中的`SplitCourse`、统计目标与独立候选比较域；同用生产adapter、环境工厂、受控PPO和公共保存/引用。分支初始化在LoadedSource边界显式消费公共选中模型契约，领域层只重建目标/窗口/确认，不重建Actor、Critic、Adam、std、RNG或已完成warmup，也不放宽同run恢复。

课程确认绑定实际方向/目标/高度/DR和执行合同，不能跨条件累计，不因无关revision丢失其他方向确认。Spin只生成有限且可表示的机体yaw端点；轮±80rad/s包络是独立动作合同。候选资格、任务成绩与独立认证分开，输出交给公共ModelIndex；领域模块不删除产物。

## 公共监督与推理消费者（2026-11-12）

`experiments/supervision.py`只声明本轮使用的类型和顺序约束；`supervision_flow.py`由现有distillation_job/worker执行有序流程。流程位置phase_cursor与领域课程revision分开。一个BC循环负责优化，公共Budget分别记录round、优化步和物理子步；已发生但无法安全恢复的消耗不回滚。

`artifacts/dataset.py`保存collection及物理episode身份、持久split和数据视图。label阶段组合累计collection与受信rehearsal，按sample_id去重。Teacher路由及各自动作合同在`learning/distillation/routing.py`，不能平均不同raw空间。Recovery physical Actor的forward即物理标签空间；BC、播放器与导出都使用它，不能再次cubic解码。

`learning/inference.py`构造无Critic/Adam的显式Actor；Teacher、标签、播放器、sim2sim与部署导出复用。模型加载绑定同一打开文件的SHA/权重。语义身份仅忽略有内容SHA的存储引用路径，不递归删除任意path键。RuntimePaths在读取边界映射，旧记录只读。

`simulation/collection.py`按实际环境行数分配、设备端有界缓存与分块落盘，要求真实owner/路由条件和实际子步。采样工厂与评估工厂分开。partial reset不拼接其他环境，fresh-reset不拼旧未完成episode。正常资源/停止边界仍由公共worker管理。

`simulation/scenes.py`显式提供平地高度、摩擦、合法初态；Recovery环境和MuJoCo播放器消费，未知/不支持场景拒绝。Motion／Spin非默认场景需领域factory支持，不宣称rough已实现。

`artifacts/deployment.py`临时目录构造、校验后原子发布；一致的既有包可恢复提交，身份冲突拒绝覆盖。sim2sim复用collection factory及create_player，以显式live协议顺序采集后调用同一recorded诊断；后端必须提供实际条件和信号，未实现即拒绝。共享C保留各ABI身份，电控仓库不写入。

监督跨目录导入使用现有handoff边界，保留BC Adam、RNG、数据/Teacher身份、轮数和原预算；目标阶段状态也必须是SHA核验依赖。活动run不得导入并行消费。云端依赖闭包只打包显式消费者和当前源码，不打包整个Conda或历史runs。

## 故障保存边界与诊断编排

driver只对优化、计费、任务后处理全部完成的边界补存。优化完成收据只证明应计费，不证明当前课程/执行现场可恢复；半完成rollout不得用旧update号发布新物理现场。保存失败保留上一有效引用，cleanup错误附注在primary异常上供worker结算。普通resume不放宽计数一致校验。

sim2sim不是训练课程或另一套worker：显式协议经CLI直接顺序拥有两个后端会话，不创建子进程。collection adapter复用make_environment和公共推理factory；MuJoCo adapter复用playback.create_player、同一控制循环与模型合同。实际条件/积分接触定义、原始信号及真实子步形成统一trace后调用已有diagnose。故障保留partial数据与原因；不更改评分口径，不做隐式独立能力评估。

## 公共性能观测接口

复用`learning/ppo/performance.py::Measurement`。driver在adapter.build前注入同一`context.measurement`和`adapter.measurement`，GenericPPO再传给ControlledPPO与环境的`set_measurement`。需覆盖构造/reset的后端可声明`Backend.make_measured_env(job, measurement)`，原单参数factory仍有效。计时不参与预算、模型状态、RNG或课程决策。

`section(name, parent=None, synchronize=None)`记录inclusive墙钟；嵌套默认不CUDA同步，hotpath显式False。`count(name, amount=1, reason=None)`累计已知主机计数；`reset()`只在update边界清空窗口，启动摘要另存。域内记录reset行数/full/partial、材质与mass/COM/inertia调用、整池/使用行数、已知传输字节、重建原因、编译和缓存失效；不得为计数新增设备同步。环境重建保持先关闭旧环境，再构造/读回，不能以缓存请求值代替实际后端核验。

Recovery监督扩展：`learning/distillation/objective.py`负责显式物理Huber和回合/阶段采样质量；`tasks/recovery/distillation_metrics.py`只计算真实首回合指标。`supervision_flow`复用collector、现有Recovery环境和owner预算，显式选择教师/学生行为策略、固定更新数fit与有界闭环compare；环境在worker内复用并在finally关闭。compare不向数据聚合回灌。部署模块接收带SHA和行号的真实golden来源，复用C核心核对三次decoder。未新增scripts工具或训练进程创建点。
