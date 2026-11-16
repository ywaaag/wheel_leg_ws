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

- `action` 与 `observation` 的尺度分别校验：输入 1.5 rad / 100 rad/s；输出按显式动作版本为腿0.15或0.50 rad / 轮40或80 rad/s，当前Motion／Spin配方为80，不能随输出改输入。
- `previous` 只由实际采用目标换算（`tasks/motion/encoding.py::previous_from_adopted`）。
- 完整恢复状态必须包含 Actor、Critic、optimizer、std、RNG、计数与课程状态；Actor-only 导出不能替代。
- Critic-only warmup 次数来自配置；完整恢复的 formal 不重复 warmup。
- 保存策略只清理本run中未选中、未引用的中间 checkpoint。

## Motion跨后端执行口径

训练、sim2sim、播放器和部署负责人先读[Motion执行口径](docs/MOTION_EXECUTION_ALIGNMENT.md)。当前Isaac和MuJoCo使用同源完整V4与NET分路，不额外叠旧弹簧／C补偿；已有015实机保留原补偿。采集与拟合的等效边界、尚未绑定的原CSV／烧录身份及完整电控验证要求在该文档集中维护。

复用`simulation/isaac`、`v4_response.py`、`playback_execution.py`与`control_core.py`现有消费者。共享C编码／解码／PD不等于完整`RLControlExecute`补偿链，Isaac张量执行也不等于每个子步直接调用C固件。本次说明不新增执行模式、接口或生命周期。

## 实机反馈与版本追溯

[实机版本反馈](docs/HARDWARE_FEEDBACK.md)集中记录用户提供的Motion／Recovery实机观察，字段包括类型、版本／拍摄日期、新增功能、存在问题、电控Git哈希、效果视频、数据效果.csv和备注。拍摄日期与Git日期分开，用户观察、原因假设、仿真成绩和主机测试分别标识；未提供的固件身份、视频或CSV保持UNKNOWN／未提供，不由最新分支反推。

该文档是领域证据入口，不是模型注册表、执行配置或硬件放行开关。原始附件沿用统一数据根与已有资产引用机制；ModelIndex仍负责模型引用，能力矩阵仍负责目标和验收，执行口径仍负责V4／实机合同。正式源码不读取反馈文档来改变策略、评分或控制，新增反馈不创建训练入口或生命周期。

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

`tasks/motion/execution_statistics.py`累计Student有效样本的六通道执行诊断；生产adapter仅在每100更新的报告更新采集，并写入同一`diagnostics/update_*.json`的`cohort_execution_statistics`。cohort_index与同文件onpolicy.rows/阶段cohort顺序一致，用于区分命令、DR和方向。统计覆盖该更新rollout的策略步末reset前快照，非100更新窗口、非1ms积分；设备端固定形状归约，导出时统一读回。不接课程/候选评分，不改变PPO、执行缓存或保存恢复状态；原physical_control保留为最后策略步的全环境读回。

`motion48_turn_focus_branch_v3`复用SplitCourse，增加显式focus权重、初始档位、基础保留晋级门槛；stage将75%行进转弯响应门槛传给同一maneuver_statistics，原地自转和旧课程默认不变。未实现全矩阵自动调度或高度前置门槛。u1500的`full_motion_course_preparation_v1`由既有artifacts.branch_initialization读取真实run/state/budget快照及SHA，在完整来源边界只重建课程证据，保留学习状态和独立模型进度；生产adapter复用同一依赖登记、课程、保存恢复，不增加owner或worker。该记录仅支持零额度准备，require_allocated始终拒绝，不能替代终态结算与正式接续。

`full_motion_course_allocated_v2`沿同一边界接真实终态和新授权，只允许已终止旧任务、原消耗保留、旧余额退役及SHA绑定的唯一新分配。它允许完整881D/现有高度合同继续，不重复877→881或高度扩展；新目标仅重建课程证据，不重建Actor/Critic/双Adam/RNG。准备版仍拒绝训练。GPU由用户通知后走一次正式CLI提交，不增加后台监测、资源抢占、owner预处理hook或另一套launcher。

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

低高度补教复用`ActionLabels(kind="motion_support_teacher")`：原Teacher仍以自身推理/输出合同加载，`tasks/motion/support_teacher.py`只对其解码后的四腿物理目标施加显式、镜像、按命令高度分段线性的示范偏移。`support_shift`包含版本、0.20～0.40m节点、四腿rad偏移及研究证据SHA；不改变轮目标、输入或执行缓存，不能作为Student逆解合同。监督preflight在提交前构造标签codec并校验声明；Teacher SHA与偏移共同进入既有supervision语义身份，恢复时拒绝变化。

诊断协议的`teacher_support_shift`使用同一纯计算模块的NumPy版本，供明确干预的Teacher采集；Torch监督与NumPy采集有数值对照。Student训练/播放不带该示范偏移，使用自身Motion动作合同直接输出。新数据通过原dataset/supervision累计，保留真实physical_episode_keys与角色，Student补教只增加训练回合，不将开发/留出变为训练。BC轮数、完整状态、停止恢复仍由公共owner/worker管理。

`wheel-leg diagnose --config`识别显式`motion_support_probe_v1`协议，调用`simulation/motion_support_probe.py`，复用`playback.plan/create_player`和MotionPlayer的共享C/V4执行。逐case只注入实例级物理目标变换，保留原Actor、原始解码目标和实际采用目标，不改冻结合同。支持15～40秒case、总墙钟及输出边界；中心候选仅来自完整、稳态、双支撑的静止片段并校验平地对称目标。±0.30/±0.50仅表示旧raw的显式诊断干预，不代表正式新动作合同、蒸馏完成或能力认证；不接owner/PPO、不消费在线更新。

Motion新增显式`motion_multiheight020_support_center_050_80_v1`，腿±0.50rad、轮±80rad/s，沿用0.20～0.40m分段中心与1.5rad/100rad/s输入尺度。与旧015合同各自校验；共享C结构ABI不变，新增center_kind=5只接受0.50/80表中心，旧kind拒绝0.50。源码manifest另记扩展摘要，不修改冻结副本。`ActionLabels.motion_contract`经Teacher物理目标反解Student标签，越包络拒绝；BC候选及完整BC状态保存Student输出与时序，统一推理/播放器/部署直接消费。监督蒸馏、播放/导出沿既有入口。Isaac PPO通过`artifacts/motion_actor_admission.py`在来源边界核验新授权、旧结算、唯一分配、BC Actor/合同和课程；`actor_initialization`新建Critic/双Adam/RNG/课程，生产adapter登记依赖、使用真实050解码与扩展高度质量。来源bank绑定同一050合同，物理误差诊断按声明尺度换算；不改变固定来源KL归约或PPO采样分布。同run完整恢复保留原Actor准入身份、warmup与bank游标，不重复初始化；不得拿BC状态冒充完整PPO恢复。

有界诊断扩展至最多64case，可显式`terrain_map=true`并为case声明`spawn`，复用既有平地＋四H1地图及合法reset。轨迹保留实际前动作31D observation供离线监督，原始与采用目标分开；未知或未启用地图的地形拒绝。完整episode划分、Teacher标签与BC训练仍交给公共supervision/dataset/owner，诊断本身不训练。

低高度探查可通过case的`center_delta_start_height_m`声明按命令高度渐进施加的中心偏移；原始C目标和干预后采用目标分别记录，初始高站姿无偏移。它只扩展诊断实例参数，不进入训练/部署或覆盖正式中心。研究可在产物中另行声明受载筛选门槛，不能覆盖原严格中心候选标记，未达目标高度的数据不能标定为目标高度平衡。

driver只对优化、计费、任务后处理全部完成的边界补存。优化完成收据只证明应计费，不证明当前课程/执行现场可恢复；半完成rollout不得用旧update号发布新物理现场。保存失败保留上一有效引用，cleanup错误附注在primary异常上供worker结算。普通resume不放宽计数一致校验。

sim2sim不是训练课程或另一套worker：显式协议经CLI直接顺序拥有两个后端会话，不创建子进程。collection adapter复用make_environment和公共推理factory；MuJoCo adapter复用playback.create_player、同一控制循环与模型合同。实际条件/积分接触定义、原始信号及真实子步形成统一trace后调用已有diagnose。故障保留partial数据与原因；不更改评分口径，不做隐式独立能力评估。

## 公共性能观测接口

复用`learning/ppo/performance.py::Measurement`。driver在adapter.build前注入同一`context.measurement`和`adapter.measurement`，GenericPPO再传给ControlledPPO与环境的`set_measurement`。需覆盖构造/reset的后端可声明`Backend.make_measured_env(job, measurement)`，原单参数factory仍有效。计时不参与预算、模型状态、RNG或课程决策。

`section(name, parent=None, synchronize=None)`记录inclusive墙钟；嵌套默认不CUDA同步，hotpath显式False。`count(name, amount=1, reason=None)`累计已知主机计数；`reset()`只在update边界清空窗口，启动摘要另存。域内记录reset行数/full/partial、材质与mass/COM/inertia调用、整池/使用行数、已知传输字节、重建原因、编译和缓存失效；不得为计数新增设备同步。环境重建保持先关闭旧环境，再构造/读回，不能以缓存请求值代替实际后端核验。

## 公共播放器网页与H1碰撞地图

`cli play`通过`playback.plan`读取原run及可选只读RuntimePaths视图，核对模型/资产/动作合同；
`create_player`复用`learning.inference.from_source`，`run_loop`仍是唯一GUI/headless物理循环。
`playback_panel`仅适配既有`students.motion_panel`的锁/HTTP handler到Controls，网页拥有命令、GUI键盘不竞争；
`playback_panel.html`只展示此适配器实际提供的信号。网页会话强制不保存，普通批量播放可显式`--no-save`。
`playback_terrain`属于场景/后端适配，加载SHA绑定的实际USD碰撞网格与匹配高度数组，构造单一有限MuJoCo高度场，
碰撞射线是反馈高度唯一来源；不将旧NPZ vertices副本或平面当作rough。
选择出生区域显式reset，平地与四块H1仍在同一model；不另建场景切换循环。
4.8 N·m为Motion/Recovery播放器的明确轮限矩边界；求解器profile复用已有实现并显式展示非等价范围。

网页监听由同一播放器`--panel-host/--panel-port`显式配置，默认loopback；无需新增代理或控制循环。
本机路径映射留数据根，不提交机器运行配置；源码示例只指向已核验的数据材料。

网页展示与键盘适配位于`simulation/playback_panel.html/.js`，由现有`start()`内联加载为同一页面；
JS作为package data打包。借鉴历史Motion面板的焦点保护、物理按键和有界曲线，不搬入其斜坡、手柄或技能仲裁。
键盘/数值/按钮通过同一个串行命令队列接既有HTTP owner，不直接调用物理、不引入第二个控制循环。


Motion少量高度配方`configs/jobs/motion_height_small_dose_prepared.json`复用SplitCourse v3的targets/focus_weights与既有height-first时序，不新增课程版本、执行器或训练入口。10%高度、28%转弯、7%rough、55%基础；职责和语义见能力矩阵末节。`artifacts.branch_initialization.CONTINUED_COURSE`在现有完整来源/分配校验中支持真实终态后的剩余额度接续，复用生产adapter依赖登记和公共预算；不更改同run恢复合同，不热改活动冻结源码。

### 只读看板扩展

`cli.dashboard → integrations.dashboard → artifacts.dashboard.Catalog`。HTTP服务仅绑定loopback，固定路由和opaque证据ID，不暴露文件目录、命令执行或训练控制。标准库读取不导入训练或物理依赖。领域通过纯计算展示适配器消费既有诊断，不重新评分；首个适配为 `tasks.motion.dashboard.training_windows`。新增领域提供目标说明、条件身份和已有判定；未提供的列保持UNKNOWN。不得从key猜测DR/地形或由部分指标推导已掌握。训练、固定模型和实机证据不得混排；后台采集/独立评估不属于看板。

看板远端源由 `integrations.remote_dashboard.RemoteSource` 管理一次显式OpenSSH读取，固定 `remote_probe.collect` 只读探针由远端系统Python执行，不是训练入口或远端agent。传输校验后在私有临时目录形成派生视图，继续经 `artifacts.dashboard.Catalog` 和领域适配解析；原远端文件不改。连接身份由运行参数提供，密码仅后端内存，known_hosts加显式指纹双校验。HTTP只允许显式刷新与连接元数据修改，不提供shell/任意文件接口；无watcher、自动重连或训练控制。

训练发现复用既有run与startup_handshake作为唯一登记事实，不新增registry：CLI提交结果附带看板产物根和身份记录提示。远端probe仅发现授权runs根的直接子目录（或单个run），汇总后仍交给同一Catalog/领域适配；资源单次读取、总传输有界。显示或采集失败不重提训练、不改预算。

## 存储边界

`artifacts/runtime_paths.py::resolve_runs_root`在写入边界解析显式runs-root或WHEEL_LEG_DATA_ROOT，不在模块导入时绑定源码目录。owner创建run并记录同一根目录的models.json，PPO与监督均复用；resume遵循原run记录。export默认使用数据根，不回写源码run。显式导出/诊断/导入目标由调用者选择已确认数据盘，测试可用/tmp。

`integrations/cloud_bundle.py::adopt_local_dependencies`接收明确保留集合，逐文件核验来源/目标、拒绝冲突，复用已有内容；只对显式不可变文件硬链接，可变文件复制。ModelIndex保护先于派生映射发布，故障重试重新核SHA；该接口不扫描历史、不删除原件。回执不是第二套资产保护索引，删除仍以唯一ModelIndex及实际消费者为准。RuntimePaths读取派生映射，原配置/spec和模型不改写。

### 固定Motion复刻诊断

后续Motion蒸馏固定使用`motion_vm1_mujoco_v1`。`simulation.motion_vm1_plant`只构造vM1原MJCF机械模型并恢复已核验的历史运行时覆盖；仍由公共`MotionPlayer.step`/`PlaybackExecution`执行唯一物理循环、共享C编码解码/PD、完整V4和运输时序。轮电机保留原传动比，ctrl/forcerange按gear换算，不能把关节Nm直接当电机ctrl。原几何、惯量、关节身份和网格有SHA，不从网络维度猜兼容。

参考必须同时绑定原MJCF（SHA `663e121ef9aed4bfab09ff0d3d98321b7ce77d80b63584235e4da691bd3aebf3`）、全部网格、原来源实现和实际编译读回。原XML为implicitfast/Newton、2mm margin；历史`build_model`随后运行`apply_current_plant`及`closure_contact_tight_v1`，所以最终闭链solref=.002/1、solimp=.9999/.9999/.001/.5/2，轮地solref=.002/1、solimp=.999/.9999/.0001/.5/2，300/100迭代、容差1e-12/1e-8；滑动摩擦.5，扭转/滚动摩擦0。不能只读取原XML的.008柔顺值就称vM1实际配置。当前腿40/轮4.8Nm是既定部署合同，历史轮2.44695Nm另列，不静默复原或提高限矩。

新正式协议`motion_replication_probe_v2`强制显式`plant.motion_vm1_contract={path,sha256}`，输出实际model、gear、solver与时序身份。Teacher与Student同一配置，31D输入/previous语义、各自输出decoder及050原u2200中心不受物理参考选择影响。验收目标是保留MuJoCo中Teacher合格能力，无需超越Teacher；双轮各>2N的1kHz承重比例严格>0.60，跟踪、扰后恢复、HF和漂移逐项报告。Isaac和实机证据分别记录，不以该参考宣称跨引擎或硬件等价。

v1诊断与USD生成模型保留供既有冻结消费者和明确标注的诊断对照；`motion_vm1_contact_20261109_v1`是曾仅复制原XML局部参数的探索配置，未复现完整vM1调用链，不能用于后续正式蒸馏。新配方不得继续选它。数据、模型和全部原始证据仍进统一数据根及同一ModelIndex，无第二套索引。

提速复用同一BC循环的显式runtime.device：CPU MuJoCo按独立物理回合并行；大Teacher批量标注与BC可在已核验的本机CUDA设备上运行。保存额外包含CUDA RNG和optimization_device，同run恢复禁止偷偷换设备，Adam/RNG/rounds仍由原owner/worker结算。单回合策略推理可留CPU，避免小batch设备往返；基准报告实测吞吐。portable测试保持CPU且不调用CUDA或物理积分。

`cli diagnose`显式`motion_replication_probe_v1`→`simulation.motion_replication_probe`→公共`playback.plan`/`learning.inference`/`MotionPlayer.step`。与旧承重中心探查不同，此协议保留每个Actor自身decoder、不做raw重标或中心干预；共同物理执行条件绑定，允许命令时间序列、显式root速度增量、单位四腿模态外扰。外扰在公共step中与NET分路相加，力矩上限不变。逐1ms真实接触力、q/dq、命令/previous/目标/PD/motor/NET/外扰进入数据根，`tasks.motion.replication_metrics`只算离线窗口/带通/恢复指标。诊断没有训练生命周期；状态及完整episode身份交给现有dataset/supervision，多Teacher与BC继续由原owner/worker管理。不同于sim2sim双后端对照，不构造第二套物理积分器或模型索引。

DAgger来源引用可显式携带`observation_envelope`（`teacher_observation_box_v1`、完整输入的有限lower/upper）。`artifacts.dataset.read_source`在采样前合并原transition_valid和范围准入；`supervision_flow`把准入mask传给既有`routing.label`。不合格行保留样本/回合身份，owner=-1、标签NaN，不查询Teacher、不做物理逆变换、不训练。观测box是已声明的保守适用域，不能当作Teacher在所有box内状态成功的证明；来源轨迹与固定余量由配方/证据绑定。该机制不增加Student输入、部署路由或控制器。

监督新增可选`loss=motion_physical_huber`，仍由同一BC循环优化/开发比较；mean差乘Student自身腿/轮包络后按显式rad/rad/s标尺归一化，不重复解码。默认MSE配方不变。公共dataset引用可声明有时间列与eligible列的互不重叠窗口配额，按每个物理episode取原样本、保留角色/原sample_id；静态condition_values仅为有来源的场景标签，不能覆盖实测condition列。采样和损失进入原监督身份，不新建训练或数据生命周期。
