# 项目协作规则

## 执行原则

- 先读本文件，再从 `STATUS.md` 定位当前任务；用当前源码、配置和原始结果核实事实，文档只作导航。
- 使用中文，结论在前。用户当前指令优先；授权内的实现、CPU测试和可逆修复直接完成。
- 保留无关修改；不使用 `git reset`/`clean`；不停止、不重启、不热改他人正在运行的任务。
- 不编造路径、命令、数据或测试结果。文档与代码不一致时，以实际运行的冻结源码与spec为准。

## 阅读入口

| 要做什么 | 入口 |
|---|---|
| 命令、参数、退出码 | `wheel-leg --help`、`README.md` |
| 当前任务、负责人、证据 | `STATUS.md` |
| 模块职责与依赖方向 | `ARCHITECTURE.md` |
| 本次重构范围、删除项、未完成项 | `docs/MIGRATION_20261105.md` |
| 训练经验、课程与选模 | `TRAINING_EXPERIENCE_REUSE.md` |
| 长期方向 | `PROJECT_MASTER_PLAN.md` |
| 历史原文 | `docs/history/navigation_20261105/` |

日期报告按需查阅，不列为每次必读。

## 架构规则（稳定，改动需评审）

1. 统一入口：正式训练/蒸馏只经 `wheel-leg` CLI（`cli.py`）；训练进程创建只允许 `experiments/owner.py`。新增并行入口或绕过 owner 的 worker 创建会被 `python scripts/check_architecture.py` 拦截。
2. 生命周期复用：PPO 与 BC 共用 owner、worker、状态、停止恢复与预算机制（`experiments/`）；各算法只保留自己的数学训练循环。BC 只写 `rounds_used`，不写 PPO `updates_used`。
3. 扩展位置：网络宽度/层数/超参→配置；新网络/损失/算法→`learning`；观测/动作/奖励/终止→`tasks`；后端与时序→`simulation`；checkpoint/评估读写→`artifacts`；一次性分析→`scripts`。
4. 依赖方向：tasks↛cli/owner/report；learning↛cli；simulation↛integrations；artifacts↛tasks；正式源码↛scripts。配置数据类型（`experiments.config`）允许被各层导入。执行判定收敛在 `dispatch.execution_service()`。
5. 禁止跨模块改写实现对象：不得 `module.Implementation = Other` 替换已导入模块的全局对象；用显式参数注入 factory。
6. 模型形状 ≠ 物理语义：相同输入维度不证明兼容；观测列序、归一化、动作中心/单位/包络、控制时序变化必须有明确来源与目标契约（以冻结 spec 为准）。
7. 初始化边界清楚：完整恢复（LoadedSource）、Actor 初始化、部署导出使用各自接口；完整恢复不得退化为 Actor 初始化或裁剪 checkpoint。
8. 校验在必要边界：配置入口、外部产物加载、恢复入口；内部已建立的约束不逐函数重复核验。
9. 新能力先问"能否只用配置"：能用配置组合表达的实验不新增 launcher/训练循环；替换旧实现时删除失去用途的分支，不保留两套有效入口。
10. 必跑检查：`python scripts/check_architecture.py`；相关回归 `tests/test_new_*.py + test_phase2/3 + test_architecture_rules.py`；Ruff check/format。

## 核心契约

- Motion负责站立、前后运动与转向；Recovery负责从合法Swing自主起身并连续保持10秒。两者来源、预算与能力结果分别记录。
- 六通道顺序：模型 left hip/active/wheel、right hip/active/wheel。模型 left 对应几何右；电控所说左右指站车尾朝车头看的几何左右。不要交换网络数组。
- 输入与输出尺度分开：31D 输入中四腿位置与上一实际腿目标按 1.5 rad 归一化、上一轮速目标按 100 rad/s 归一化；输出按显式动作合同选择腿包络及轮 ±40／±80 rad/s等版本，不得固定为单一默认；输出变化不得带动输入尺度。
- previous 保存实际采用的物理目标；影子推理不能写入执行缓存。原始动作、网络mean、执行目标与力矩分别保存。
- 默认物理/PD/策略周期 1/5/10 ms；具体 PD、限矩、动作尺度、USD 与地面以冻结 spec 为准。
- 不暗加滤波、限速、补偿、静态偏置或替代控制器。新增设计必须同时更新训练、评估、播放与部署定义。
- 区分观测年龄、目标运输、PD后力矩运输与电机响应，不重复叠加同一延迟。

## 实验与预算

- 每个任务只有一个训练负责人；先核对现有提交与真实身份，不接管其他会话的任务。
- 仿真、GUI、PPO、硬件与外部通知按用户明确范围执行；文档或CPU测试授权不扩展成这些操作。
- 新配方用 `configs/jobs/` 下的配置（preset 加显式覆盖），不复制旧spec逐层删改。
- 任务级不限额写 `null`；执行块仍需保存、墙钟、空间和停止安排。累计消耗记录在预算账本，不能重置。
- 旧run、模型、原始评估、spec、账本与运行中的冻结代码保持只读。需要修复时用新版本，保留原失败证据。
- 提交前核验session、物理GPU与UUID、CUDA映射、共享资源、磁盘和负责人；不自动换GPU、提高限矩或停止其他进程。

## 实现与验证

- 新代码进 `src/native_stand/{cli,experiments,tasks,learning,simulation,artifacts,integrations}`；依赖方向见 `ARCHITECTURE.md`，禁止任务模块导入CLI或报告器。
- 复用公共准备器、物理环境与受控PPO；不另建日期型trainer，不整树复制源码。
- 纯计算模块只接收明确数据；文件读写与身份检查放在入口。`AppLauncher` 必须先于 Isaac runtime 导入。
- 先跑相关CPU测试与 `--dry-run` 计划检查，再走正式提交。smoke检查真实控制归属、非零奖励、有效优化、结束/reset与保存加载。
- 预算计数以运行状态为准，不从稀疏曲线反推；`COMPLETED` 不等于能力 accepted，`submitted` 不等于训练已开始。
- 后台提交只做一次有界启动握手；随后由owner与worker负责训练、保存、评估、终态与结算。不新增watcher，不反复轮询。
- 通知需明确授权与目标会话；通知失败不改变训练终态。
- 旧入口 `command_v6.supervisor` 的多轮提交流程已退役；不要为了兼容重新加回 `--write/--check-submission/--launch` 链。

## 文档维护

- `README`：用途、环境与统一CLI；`AGENTS`：稳定规则与契约；`STATUS`：当前任务卡；`ARCHITECTURE`：模块与依赖；命令参数由 `--help` 维护。
- 逐次实验进度写进对应 run 的报告与 `artifacts/`，不追加回 AGENTS。
- 历史原文留在 `docs/history/`，不因清理导航宣称旧证据重新可复核。

## 中文输出规则

先给结论再给证据；一句一个意思；同一事物只用一个名称；删除套话与无信息修饰。

- 已知时间、大小和成本写明数字；未知时写明缺什么证据。
- 缺陷与待确认项写四件事：标题、实际表现、代码根因（含文件与行号）、处理建议。尚未定位时写“根因待确认”，不编造行号。
- 报告正文不重复免责声明；审计范围在开头说明一次。
- 内部编号、路径和行号放在括号或表格定位列，标题优先用功能名称。
