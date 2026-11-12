# Native 闭链轮腿机器人训练

统一入口为 `wheel-leg`，源码在 `src/native_stand`。当前工程支持Motion／Spin、Recovery，以及Jump ABI4 MuJoCo CPU链；CPU工程验证不代表真实物理、GPU、云端或能力认证。当前事实与限制见STATUS和迁移报告首页。

## 主仓与本地依赖

后续源码开发以 `/home/kellen/wheel_leg_refactor_ws` 为主仓，Git独立于旧生产仓。旧仓 `/home/kellen/wheel_leg_native_train` 原路径保留，保存历史模型、账本和冻结依赖；不交换目录、不建立旧路径转发软链接。

本地依赖根为 `/home/kellen/wheel_leg_training_data`。`manifest.json`记录来源SHA、实际位置和消费者，`runtime_paths.json`为现有RuntimePaths逐文件映射，`models.json`沿用ModelIndex。它不是独立备份；同内容只在数据根内部硬链接，未链接可变原run文件。旧142份缺失数据与失效候选继续UNAVAILABLE。

复用已有已验证包，不为Git重新制作大型包：

```bash
PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python -B -m native_stand cloud-plan \
  --manifest artifacts/cloud_handoff/dependencies_current.json \
  --dependency-archive artifacts/cloud_handoff/motion_recovery_prepare.tar \
  --restore-dependencies /home/kellen/wheel_leg_training_data --dry-run
```

确认manifest与包身份后去掉dry-run只恢复依赖，不建run、预留额度或训练。原始spec/checkpoint不改；派生配置显式加入assets.path_mapping的path/SHA，source_root指向本次检出的src。新领域增量所需但旧包未含的依赖仍需其独立SHA清单，不能冒充已全部脱离旧目录。干净clone使用自己的PYTHONPATH=src，不借旧仓editable install；不要重装共享训练环境。

## 环境与入口

Python3.12。安装使用已有环境，或从仓库设置 `PYTHONPATH=src` 后使用 `python -m native_stand`，与wheel-leg同一解析器。离线BC不要求Isaac运行环境。`scripts/test_portable_cpu.py`启用执行保护，真实mj_step、CUDA初始化或AppLauncher调用会失败而非静默跳过；历史物理用例不得借portable名称自动执行。

```bash
wheel-leg train --config CONFIG.json --dry-run
wheel-leg distill --config CONFIG.json --dry-run
wheel-leg resume --run RUN --dry-run
wheel-leg stop --run RUN --dry-run
wheel-leg inspect --run RUN
wheel-leg play --run RUN --checkpoint CHECKPOINT --headless --steps 100 --dry-run
wheel-leg sim2sim --config PROTOCOL.json --output RESULT.json --dry-run
wheel-leg export --run RUN --checkpoint CHECKPOINT --output PACKAGE --dry-run
wheel-leg import-checkpoint --manifest HANDOFF.json --mapping PATHS.json --destination CHECKPOINT --run-id LOGICAL_ID --dry-run
wheel-leg cloud-plan --config CONFIG.json --manifest MANIFEST.json --dry-run
```

执行命令的dry-run不创建run、预算、worker、引擎或通知，不反序列化模型。真正提交需要显式正额度、来源与资源前检。Motion／Spin原prepared配方额度为0；后续smoke/远端授权、实际消耗和结转以各owner绑定的分配及结算为准，不得沿用历史4889余额。用户未授权前不得去掉dry-run启动真实训练。

## 实际流程与边界

| 流程 | 正式消费者与限制 |
|---|---|
| PPO | CLI→dispatch→owner/worker→公共driver→任务adapter；Motion／Spin领域组件由对应负责人维护；Recovery为当前v8/909D/4.8Nm配方。Motion已有其owner绑定的Isaac远端工程运行收据；Recovery真实接续由其owner安排，不能将Motion结果外推其他任务 |
| BC／DAgger／补教 | `SupervisionConfig`声明Teacher集合和collect/label/fit/compare/export顺序及有界重复；同一owner/worker和BC数学循环。累计数据视图保留来源、episode划分、实际Teacher路由；holdout不回灌。`--max-rounds N`可作监督轮边界停止，resume不刷新额度 |
| 采集 | 显式make_collection_env，按真实观测行数、owner和实际子步计费，分块持久化；未实现采样工厂的后端拒绝，不能借evaluation.envs包装器代替 |
| 播放 | 一个GUI/headless循环，公共推理factory与模型执行合同；暂停不推进tick，Recovery physical输出不重复cubic解码 |
| sim2sim | recorded或live显式协议→顺序打开后端→统一轨迹→既有诊断；live核对实际条件/子步/终止/覆盖，复用采集factory及播放器。未提供诊断读回的后端拒绝，不自动加入PPO或课程 |
| export | 原子发布完整部署目录：Actor、可选ONNX、推理架构、合同、执行参数、共享C身份、golden、来源与依赖。可搬迁读回；完整续训checkpoint另行保留 |
| Jump | ABI4、37D线性raw、6370D MuJoCo CPU Critic；没有Isaac GPU backend，不允许改device冒充支持 |

PPO更新、监督round、实际监督优化步和物理子步分别记账。compare只读已有开发数据，不额外采样。训练候选使用正常PPO窗口，不需要独立评估。历史evaluate入口和证据保留，不构成训练或选模依赖。

最新完整恢复点、各目标候选、selected/pin及真实来源依赖取并集保护。项目级ModelIndex为唯一权威，多消费者分别登记；本轮不清理真实文件。BC latest与selected用途不同。数据域变化时重新计算incumbent开发损失基线，旧best作为历史保留。

`configs/jobs/dual_teacher_offline_import.json`是可审阅配方，真实提交仍因旧142条数据索引中的NPZ缺失而拒绝；dry-run成功不等于数据齐全。换目录必须使用SHA绑定RuntimePaths及显式handoff，不改旧文件。

## 性能观测

显式配置 `diagnostics.performance_measurement=true` 才启用分段计时和计数。公共driver、GenericPPO、ControlledPPO与任务环境共用一次运行的 `Measurement`；默认关闭不触发CUDA或引擎。`performance_seconds`及`performance.jsonl`保留每轮数据，保存轮用`saved_this_update`区分；启动构造单列`initialization_measurement`，不藏入下一轮rollout。

计时是包含子项的墙钟：`performance_parents`／`parents`声明层次，父子不能相加。CUDA外层边界可同步，嵌套子项默认仅`cpu_wall`，不是GPU kernel耗时。`after_save_total`包含候选与必要重建，不能当成候选登记耗时；领域分别记录候选、关闭、构造、首次reset/readback和编译。`count(name, amount, reason=...)`只收已知主机数字，不为统计回读设备张量；按update输出小摘要。真实GPU细分与吞吐由任务owner在后续安全窗口验证。

## 目录

| 路径 | 内容 |
|---|---|
| `src/native_stand/` | 新架构：`cli`、`experiments`、`tasks`、`learning`、`simulation`、`artifacts`、`integrations` |
| `src/native_stand/command_v6`、`students`、`recovery`、`world_model` | 冻结的历史实现；活动训练与历史证据按其运行，不作为新入口 |
| `configs/jobs/` | 新配方（preset加显式覆盖） |
| `configs/experiments/` | 历史实验spec与预算账本，只读 |
| `scripts/` | 离线分析与确有用途的一次性工具 |
| `tests/` | CPU契约与工程行为测试 |
| `runs/`、`isolated/`、`artifacts/` | 运行、隔离实验与评估证据，原路径保留 |
| `deployment_protocol/` | 按模型身份组织的协议与导出包 |

## 接下来读什么

- 当前任务、负责人与证据入口：`STATUS.md`
- 模块职责与依赖方向：`ARCHITECTURE.md`
- 协作规则与物理约定：`AGENTS.md`
- 迁移范围、删除清单与未完成项：`docs/MIGRATION_20261105.md`

## 验证与云端

最终命令、源码SHA和结果在 `artifacts/integration/`，参见迁移报告首页。架构检查 `python scripts/check_architecture.py`；CI使用Ruff0.8.4，范围见 `.github/workflows/cpu.yml`。

云端包是准备材料，活动Recovery快照不是最终handoff。SSH、持久盘、镜像和GPU映射尚待确认；不连接、不上传、不安装或启动。保留PhysX与镜像LD_LIBRARY_PATH。ORT、ARM、实机与真实引擎验证未执行；ONNX reference evaluator通过不能替代它们。

## 实时跨引擎诊断协议

`wheel-leg sim2sim --config PROTOCOL.json --output RESULT.json --dry-run`只列计划。实际运行必须另有明确物理步预算；本次集成只使用脚本I/O。`version=sim2sim_live_protocol_v1`声明model(path/SHA/inference_spec)、input_contract、output_contract、scene、command、initial_state、timing(physics_ms/policy_ms)、V4、motor_caps、scoring_window_s、steps与两个backends。每端点声明adapter(collection或mujoco_playback)、engine、integration、contact_sampling；collection用config，播放器用run/checkpoint。当前顺序单环境，拒绝资源/条件不符；可显式设wall_s与max_output_bytes。collection配置若用worker_atexit延迟关闭则拒绝，避免下一引擎与旧App重叠。

采集factory必须实现configure_diagnostic(protocol)实际条件读回及step.extras.diagnostic的q/dq/motor/NET/state。每步terminated/truncated、actual_physics_steps与故障如实保留；短步故障不按10ms计费。输出包含有SHA的统一NPZ、实际读回及诊断JSON，不宣称PhysX/MuJoCo等价。timing采用physics_ms／PD_ms／policy_ms、actor_packet_ms／actor_leg_packet_ms、target_ms、wheel_transport_ms／wheel_tau_ms；初态声明kind=model_bound_reset和seed，实际初态另记录。MuJoCo侧只支持已声明平地，rough拒绝；V4逐端核对同一SHA。两引擎严格先close再创建下一引擎，没有Popen或独立监督生命周期。未知后端或未提供实际读回时在执行前拒绝；支持范围以实际adapter为准。

## 故障与恢复操作

完整优化成功先按真实更新计费；任务后处理完成后才是可补存完整状态边界。rollout/后处理失败不发布当前半完成状态；保存失败不重试覆盖最后有效点。清理异常不得掩盖原始工程故障。若最新有效模型早于已耗额度，普通resume继续拒绝计数不符，需显式的新attempt/分支交接记录模型进度与已耗差异，不退款、不补造缺失更新。旧失败run和账本只读保留。

## 可搬迁包与云端操作

先冻结当前源码SHA并完成对应CPU回归，再执行`wheel-leg cloud-plan --config CONFIG.json --manifest MANIFEST.json --archive BUNDLE.tar`。依赖只沿SHA引用有限展开；缺失/冲突拒绝打包，内容去重，失败保留旧包。Recovery已在旧owner空间边界自然停止，准备包仍不构成正式handoff或接续授权。显式有效依赖可用`--reference PATH SHA256 ROLE`逐条加入；失效旧候选保留为UNAVAILABLE证据，不得改SHA冒充原候选。

在已确认持久盘解包后，按manifest逐条生成`runtime_path_mapping_v1.files`，将每个原path映射到解包根/archive_path；候选源码旧根可用prefixes映射。派生配置只加入assets.path_mapping(path/SHA)、实际source_root和执行资源，不改原始文件。GPU UUID、CUDA映射、输出runs-root与缓存必须按新机器实际填写，不照抄本机授权。保留指定IsaacSim6.2/IsaacLab3镜像、PhysX、uv及LD_LIBRARY_PATH。顺序为环境身份→持久盘/依赖SHA→dry-run→单链有界工程运行→保存恢复→分段计时；本侧不连接、上传或启动。
