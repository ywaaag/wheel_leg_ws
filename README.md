# Native 闭链轮腿机器人训练

统一入口为 `wheel-leg`，源码在 `src/native_stand`。公共重构迁移已按本地tag `refactor-migration-v1`（`1cc65e1`）验收结项；后续开发从main建立独立分支/worktree。main已有Motion／Spin、Recovery v8，以及Jump ABI4 MuJoCo CPU链。领域分支的新Recovery v9与Jump Isaac GPU实现不自动属于main支持范围；它们按普通扩展任务独立验收。CPU工程通过不代表模型能力或所有平台认证。

## 统一资产存储

源码主仓只保存源码、自研C/H、构建、测试、CI、配方和小型契约。模型、数据、部署包、USD/网格、冻结依赖和证据保存在数据根。旧仓是历史来源，不清空、不重写原始spec或checkpoint。

新run输出优先级为 `--runs-root` > `WHEEL_LEG_DATA_ROOT`；两者都没有时真实提交拒绝，dry-run仍只读。环境变量在调用时解析，不受worktree或cwd影响。本机使用现有数据根及其中的`models.json`，不要为每个worktree另建索引；远端显式指定已确认的持久盘，不默认写系统盘。resume继续使用原run记录，不因环境变量改变而搬迁或刷新预算。

```bash
export WHEEL_LEG_DATA_ROOT=/home/kellen/wheel_leg_training_data
PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python -B -m native_stand train --config CONFIG.json --dry-run
# 正式提交仍须由唯一owner在原授权范围内执行；不要因迁移自行去掉dry-run。
```

train和distill共用此规则；监督数据、labels和完整恢复点位于该run。export默认输出`$WHEEL_LEG_DATA_ROOT/exports/<run>/<checkpoint>`，也可显式指定数据盘上的`--output`；sim2sim、部署和导入等显式输出应同样选数据盘，测试使用/tmp。历史run中的原有输出不自动搬迁。

本批逐文件接管、ModelIndex消费者登记、错误SHA/冲突拒绝和派生映射复用`cloud_bundle.adopt_local_dependencies`及RuntimePaths，不新增registry。回执在数据根`history/asset_storage_20261115/`：`receipt.json`、`runtime_paths.json`、`source_history.json`和`cpu_model_load.json`。派生运行配置引用映射文件及其SHA；不修改原配置。硬链接不是独立备份；只有确认不可变文件可使用，可变记录逐文件复制。原路径仍有消费者或身份不明时保留，不执行真实清理。

## 主仓与本地依赖

Motion后续配方、评分、选模和交互测试先读[Motion能力矩阵](docs/MOTION_CAPABILITY_MATRIX.md)：0.34m基础优先，转弯、地形和0.20～0.40m高度逐级扩展。它描述最终目标，当前准备配方的实现范围单独列明；训练窗口、固定checkpoint验证与实机成绩分别记录，当前后端支持仍以实际实现为准。

训练、sim2sim、播放与部署共同遵循[Motion执行口径](docs/MOTION_EXECUTION_ALIGNMENT.md)：Isaac和MuJoCo使用同源完整V4，不额外叠加旧弹簧／C补偿；已有015实机保留原补偿。V4表示等效响应，采集力矩扣除补偿不消除实测q／dq中的补偿影响；当前共享C播放不等于完整电控补偿链验证。采集版本绑定和新050部署边界见该说明。

第二轮第一块的准备配置为`configs/jobs/motion_u1500_turn_focus_v3_prepared.json`：约36%转弯、9%地形和55%基础复习，0.34m固定高度，细化yaw阶梯并检查基础保留。完整u1500来源和准备记录通过数据根SHA引用，缺材料不能直接运行。正式CLI计划检查：`PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python -B -m native_stand train --config configs/jobs/motion_u1500_turn_focus_v3_prepared.json --dry-run --json`。配置为零额度，正式提交仍被拒绝；不代表已有GPU或能力验证。具体设计和接续边界见矩阵末节。

后续已获用户新10000次/不限训练墙钟授权，正式配置为`configs/jobs/motion_u1500_turn_focus_v3_cloud10000.json`；旧8429余额退役，历史消耗保留，允许共享GPU。授权及结算依赖在数据根`motion_authorization_20261113/`，旧终态快照在`motion_cloud10000_20261113/round2_handoff_20261113/`。远端冻结源码及启动说明部署到`/root/gpufree-data/wheel_leg_motion/candidate_revisions/motion_turn_focus_v3_10000`与`round2_handoff_20261113/START.txt`。GPU由用户手动开启并通知后再核实UUID/资源、沿正式CLI唯一提交；本配置不监测、不自动启动。材料缺失或资源不符时拒绝；已存在run不得重复train。

后续源码开发以 `/home/kellen/wheel_leg_refactor_ws` 为主仓，Git独立于旧生产仓。旧仓 `/home/kellen/wheel_leg_native_train` 原路径保留，保存历史模型、账本和冻结依赖；不交换目录、不建立旧路径转发软链接。

本地依赖根为 `/home/kellen/wheel_leg_training_data`。`manifest.json`记录来源SHA、实际位置和消费者，`runtime_paths.json`为现有RuntimePaths逐文件映射，`models.json`沿用ModelIndex。它不是独立备份；同内容只在数据根内部硬链接，未链接可变原run文件。旧142份缺失数据与失效候选继续UNAVAILABLE。

复用已有已验证包，不为Git重新制作大型包：

```bash
PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python -B -m native_stand cloud-plan \
  --manifest /home/kellen/wheel_leg_training_data/bundle_manifest.json \
  --dependency-archive /home/kellen/wheel_leg_training_data/archives/motion_recovery_prepare.tar \
  --restore-dependencies /home/kellen/wheel_leg_training_data --dry-run
```

确认manifest与包身份后去掉dry-run只恢复依赖，不建run、预留额度或训练。原始spec/checkpoint不改；派生配置显式加入assets.path_mapping的path/SHA，source_root指向本次检出的src。新领域增量所需但旧包未含的依赖仍需其独立SHA清单，不能冒充已全部脱离旧目录。干净clone使用自己的PYTHONPATH=src，不借旧仓editable install；不要重装共享训练环境。

## 实机版本反馈

[Motion／Recovery实机版本反馈](docs/HARDWARE_FEEDBACK.md)记录vM0、vM1、vM2和vR0的拍摄日期、功能、问题、Git版本及视频／CSV证据状态。记录包含用户实测反馈与尚未确认的原因；仿真、CPU测试和实机成绩分别保留，缺失的烧录身份不按日期猜测。其职责和已有模型／资产机制的关系见[架构说明](ARCHITECTURE.md#实机反馈与版本追溯)。

## 迁移前历史文档

迁移时保留的192份未跟踪文档及附件已归档到数据根 `history/docs_pre_refactor/`，正文保留原`docs/...`结构。`manifest.json`逐项记录原绝对路径、归档路径、SHA、大小及与旧仓的比较；`difference_review.json`和`review/`保留5份差异，`references.json`保留历史引用位置。归档原文不是当前用法；正式入口仍以本README、ARCHITECTURE和`--help`为准。

5份差异没有候选独有的新功能说明：4份为缺少旧仓后续追加段落，1份仅多空行；不将旧监督/播放入口重新提升为正式能力。两份历史design配置的`documentation`及旧脚本报告字符串保持原身份；需要查阅时按manifest的`original_path`／`relative_path`查到`archived_path`，不能将已移出的路径当现有输入。旧仓同名资料保持只读。后续新文档正常进入Git审阅，不忽略整个docs，也不以`git add .`收录历史材料。

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
- 已结项迁移的冻结范围与历史证据：`docs/MIGRATION_20261105.md`

## 验证与云端

冻结验收命令、源码SHA与结果在 `artifacts/integration_final/performance_delivery.json`，数据根归档为 `integration/performance_observability/b60f0d7`；运行证据不随Git clone提供。新开发按AGENTS、ARCHITECTURE与PR模板交付：`python scripts/check_architecture.py --base refactor-migration-v1`；无参数仍只检查正式源码。CI使用Ruff0.8.4并取得完整历史/tag，范围见 `.github/workflows/cpu.yml`。当前没有remote或required checks，不能声称平台已强制阻止合并。

云端包与运行收据绑定各自源码版本；Motion已有owner管理的远端运行，不能继续以旧“尚未启动”快照代表当前事实，也不能将其结果外推Recovery/Jump。新任务按实际持久盘、镜像、GPU映射和授权准备；保留PhysX与镜像LD_LIBRARY_PATH。ORT、ARM、实机及各后端能力以对应证据为准；ONNX reference evaluator不能替代它们。本轮治理不连接远端或操作训练。

## 实时跨引擎诊断协议

`wheel-leg sim2sim --config PROTOCOL.json --output RESULT.json --dry-run`只列计划。实际运行必须另有明确物理步预算；本次集成只使用脚本I/O。`version=sim2sim_live_protocol_v1`声明model(path/SHA/inference_spec)、input_contract、output_contract、scene、command、initial_state、timing(physics_ms/policy_ms)、V4、motor_caps、scoring_window_s、steps与两个backends。每端点声明adapter(collection或mujoco_playback)、engine、integration、contact_sampling；collection用config，播放器用run/checkpoint。当前顺序单环境，拒绝资源/条件不符；可显式设wall_s与max_output_bytes。collection配置若用worker_atexit延迟关闭则拒绝，避免下一引擎与旧App重叠。

采集factory必须实现configure_diagnostic(protocol)实际条件读回及step.extras.diagnostic的q/dq/motor/NET/state。每步terminated/truncated、actual_physics_steps与故障如实保留；短步故障不按10ms计费。输出包含有SHA的统一NPZ、实际读回及诊断JSON，不宣称PhysX/MuJoCo等价。timing采用physics_ms／PD_ms／policy_ms、actor_packet_ms／actor_leg_packet_ms、target_ms、wheel_transport_ms／wheel_tau_ms；初态声明kind=model_bound_reset和seed，实际初态另记录。MuJoCo侧只支持已声明平地，rough拒绝；V4逐端核对同一SHA。两引擎严格先close再创建下一引擎，没有Popen或独立监督生命周期。未知后端或未提供实际读回时在执行前拒绝；支持范围以实际adapter为准。

## 故障与恢复操作

完整优化成功先按真实更新计费；任务后处理完成后才是可补存完整状态边界。rollout/后处理失败不发布当前半完成状态；保存失败不重试覆盖最后有效点。清理异常不得掩盖原始工程故障。若最新有效模型早于已耗额度，普通resume继续拒绝计数不符，需显式的新attempt/分支交接记录模型进度与已耗差异，不退款、不补造缺失更新。旧失败run和账本只读保留。

## 可搬迁包与云端操作

先冻结当前源码SHA并完成对应CPU回归，再执行`wheel-leg cloud-plan --config CONFIG.json --manifest MANIFEST.json --archive BUNDLE.tar`。依赖只沿SHA引用有限展开；缺失/冲突拒绝打包，内容去重，失败保留旧包。Recovery已在旧owner空间边界自然停止，准备包仍不构成正式handoff或接续授权。显式有效依赖可用`--reference PATH SHA256 ROLE`逐条加入；失效旧候选保留为UNAVAILABLE证据，不得改SHA冒充原候选。

在已确认持久盘解包后，按manifest逐条生成`runtime_path_mapping_v1.files`，将每个原path映射到解包根/archive_path；候选源码旧根可用prefixes映射。派生配置只加入assets.path_mapping(path/SHA)、实际source_root和执行资源，不改原始文件。GPU UUID、CUDA映射、输出runs-root与缓存必须按新机器实际填写，不照抄本机授权。保留指定IsaacSim6.2/IsaacLab3镜像、PhysX、uv及LD_LIBRARY_PATH。顺序为环境身份→持久盘/依赖SHA→dry-run→单链有界工程运行→保存恢复→分段计时；本侧不连接、上传或启动。

## Motion原Student050 PPO接续

新作业复用`wheel-leg train`→owner/worker→生产Motion adapter→受控PPO。原Student SHA `258aae9d904fc3718ddc7938a556c523ff75228a0a7f2707f72b2162d9930d98`是唯一Actor来源；其原支撑中心原样绑定，不采用短补教实验Student或新0.24m中心。正式配方`configs/jobs/motion_original_student050_ppo10000.json`使用preset加显式覆盖。

`wide_actor_admission`与`branch_initialization`引用同一内容绑定记录，明确Actor/输出合同、课程、新授权、旧结算和唯一run分配；不复制生命周期。Actor-only不导入旧Critic、BC Adam、旧RNG或旧课程确认；新建881D Critic、双Adam和课程，先50次Critic-only warmup并计入在线额度。输入不变，探索腿.02rad/轮2rad/s对应raw std .04/.025。来源KL使用同合同冻结Actor和train-only bank，bank不进入PPO rollout/GAE；rough、新DR、事件和新转弯不套旧名义标签。完整PPO保存含双Adam、RNG、warmup、冻结Teacher、bank游标和课程；同run恢复不再执行Actor初始化或重复warmup。

新授权10000次在线更新、训练墙钟不限，旧累计9519次与BC104轮保留、旧7219余额退役。smoke和formal共用该授权分配；真实消耗以run预算为准。GPU由用户开启并通知后核对实际UUID/显存/共享资源，再使用数据根`motion_wide050_ppo_ready_20261115/START.txt`中的唯一正式命令。先`wheel-leg train --config CONFIG --dry-run --json`审阅；必须显式`--run-id`匹配分配，不能因看板未显示重新train。已有run改用公共inspect/resume；不新增watcher或自动GPU探测。

## Motion MuJoCo网页交互播放

固定Actor承重与腿包络对照使用`wheel-leg diagnose --config PROTOCOL.json --dry-run --json`检查，再去掉`--dry-run`执行有界headless诊断。协议版本为`motion_support_probe_v1`，必须显式声明`run`、`checkpoint`、`checkpoint_sha256`、`path_mapping`、`solver_profile`、新建`output`、`wall_s`（最多1800秒）、`max_output_bytes`（最多512MiB）及`cases`。每个case给出唯一`name`、`height_m`（0.20～0.40）、`vx_mps`、`yaw_rad_s`、`leg_scale_rad`（0.15/0.30/0.50）、`duration_s`（15～40）及`mapping=rescale_source_raw`；可选`center_delta_rad`是四腿偏移。复用原播放器物理循环及高度优先命令，输出原始目标、实际目标和承重统计。放大旧Actor raw是诊断干预，不可当作兼容迁移或已通过的新模型；MuJoCo结果不等同PhysX能力。

最多64个case；`terrain_map=true`允许case声明`spawn=flat`或四种H1名称，使用既有单地图reset。轨迹含每个策略动作真正采用的前动作`observation`，可按完整回合进入公共蒸馏数据流程。

低高度中心研究可显式给`center_delta_start_height_m`（高于该case目标且不超过0.40m）：四腿`center_delta_rad`从该命令高度开始线性增至目标高度处的完整偏移，初始高站姿不施加偏移。这是声明的开环诊断干预，不是自动补偿或已训练能力；省略该字段仍是全程固定偏移。

Motion的`motion_multiheight020_support_center_050_80_v1`是独立±0.50rad/±80rad/s输出合同，输入归一化和原分段支撑中心保持。经`wheel-leg distill --config CONFIG --dry-run`审阅、去掉`--dry-run`提交公共监督配方；配方的Teacher/Student分别声明自身`motion_contract`标签变换，先还原物理目标再反解Student标签。越包络拒绝，不能直接放大旧raw。BC候选可通过同一`wheel-leg play`及部署导出消费；完整BC恢复使用原run的`resume`。050合同的Isaac PPO仅接受显式`wide_actor_admission`绑定的BC Actor初始化，或同run完整PPO恢复；不能将Actor候选或BC状态当成完整PPO恢复点。模仿误差与闭环能力必须分别验证。

低高度受载补教可显式声明Teacher标签`kind=motion_support_teacher`、原`contract`和`support_shift`。后者字段为`version=motion_height_support_teacher_v1`、递增`height_m`（覆盖0.20～0.40m）、每节点四腿`offset_rad`（镜像、绝对值≤0.35rad）和`evidence_sha256`。Teacher先按原合同解码，再加偏移；Student仍使用`motion_contract`反解。采集Teacher时，在有界诊断协议声明同一`teacher_support_shift`，轮目标不变；Student播放/采集必须省略此字段，避免重复施加。受载示范与真实模型能力分别记录，不把工程偏移当作Student已学会。

可复用配方为`configs/jobs/motion_low_support_reteach_bc16.json`（preset加显式覆盖）：绑定本机数据根中的Teacher、完整episode划分数据、上一块Actor及输出合同SHA，每块16轮BC、新Adam；先`wheel-leg distill --config configs/jobs/motion_low_support_reteach_bc16.json --dry-run --json`审阅。它是已完成第3块的复现实验配方，不是继续执行指令，也不刷新PPO额度。后续补教须使用新的训练数据/版本；开发及留出不能回流本候选训练。

转弯支撑与0.24m短补教配方为`configs/jobs/motion_turn_height_short_bc12.json`，复用同一监督入口及已有conditions路由。各冻结Teacher按自身输出合同生成物理标签；Teacher选择依据仅来自训练条件，路由不进入Student执行。该配方绑定既有数据与Actor SHA，是单块12轮BC复现配置，不自动开启PPO或刷新额度；新数据补教需新run，既有开发/留出保持独立。

正式链为 `wheel-leg play` → `simulation.playback.plan/create_player/run_loop` →
`MotionPlayer` → 公共推理factory、共享C编码/解码/PD及`PlaybackExecution`完整V4。
网页仅复用已有`students.motion_panel.Plan`锁与HTTP命令/遥测服务，不创建第二个物理循环。
Motion／Recovery播放器只接受轮限矩4.8 N·m，旧轮限矩合同明确拒绝，不静默覆盖。

本机u1500示例（保留数据只读，不导入远端SDK或执行远端环境脚本）：

```bash
cd /home/kellen/wheel_leg_refactor_ws
DISPLAY=:2 XAUTHORITY=/home/kellen/.Xauthority \
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/kellen/anaconda3/envs/issaclab/bin/python -B -m native_stand play \
  --run /home/kellen/wheel_leg_training_data/motion_preservation/20261113_r15_u1500/wheel_leg_motion/runs/2026_11_12__motion_contact_grid_cloud10000_r15_speed \
  --checkpoint checkpoints/model_update_1500.pt \
  --checkpoint-sha256 6f84a7f66f64e7a30ca4cf0b4d5a1f78acaf9f3c9e96fbe12c03756624a543d2 \
  --path-mapping /home/kellen/wheel_leg_training_data/playback/motion_u1500/runtime_paths.json \
  --terrain-map --solver-profile closure_tight_v1 --panel-port 8765 --no-save
```

追加`--dry-run`只做文件身份、合同和计划检查，不开引擎/端口。
`--path-mapping`覆盖的是本次只读执行视图，原run/config/spec及已有RuntimePaths不改。
示例映射是本机材料，其他机器必须重新核验路径/SHA。远端SDK绝对链接不作为MuJoCo依赖。

打开 `http://127.0.0.1:8765`，初始暂停，点击“继续”开始积分。可发送正负vx/yaw和0.20～0.40m高度，查看待采用与策略实际采用命令、
FLU前进/yaw速度、root距实际碰撞面高度、FLU roll/pitch、几何左右双轮接触、关节、raw、执行目标、motor与NET。
接触表示MuJoCo轮—地接触距离≤0，不冒充实机接触传感器或承重认证。缺少的信号显示“不可用”，不补零。
命令直接写入，没有额外滤波/速率限制/偏置；高度越合同拒绝。网页独占运动控制，GUI方向键/切模键无效。
关闭网页不会停止仿真，最后命令保持；先按“停止命令”或“暂停”。“关闭播放器”、关闭MuJoCo窗口或启动终端Ctrl+C均可退出。
网页“停止命令”是vx/yaw归零，并非关闭电机或固定机体位置。

`--terrain-map`是一张16×15m有限碰撞地图，四块原始6×3m H1网格各一块：
平地初始(0,0)，测试带Y=3～9m，四种地形依配置顺序中心X=-6/-2/+2/+6m。
网页“重置到此区域”会将运动命令归零、清空控制历史并reset到该地形入口Y=3.1m；
普通高度/速度修改不reset，手动跨区域不reset。相机跟随机器人，可用MuJoCo鼠标操作旋转和缩放。
地形外周相邻2cm单元接周围平地，原测试网格顶点不改；这是人工测试布局，不是训练时150×9m重复道路。
离开地图或闭链残差>5mm会暂停并显示故障，用户重置后继续，不投影状态、不自动补地面。

四种USD与NPZ分别核验冻结配置SHA。周期化NPZ的`vertices_m`为旧副本；加载实际USD碰撞points/faces与
`height_m`核对后构造MuJoCo高度场。可视几何与碰撞共用此高度场，反馈高度通过该碰撞体的实际垂直射线获得。
MuJoCo高度场三角划分、接触求解和PhysX静态三角网格不宣称等价。
`closure_tight_v1`复用已有显式profile：1ms步长不变，闭链solref=0.002/1、solimp=0.9999/0.9999/0.001/0.5/2，
Newton迭代300；它改变约束柔度，不是硬件标定，接触材质/PD/电机参数不据播放效果拟合。
原默认profile在u1500短步中触发闭链残差门，不能将其失败隐藏为策略认证。

网页播放始终不保存轨迹、视频、数据集、评估结果、模型或文件日志；实时状态仅在进程与网页内存。
`--no-save`也可单独用于headless/键盘播放；临时C编译产物仅在`/dev/shm`并在退出时清理。
不加载训练用Critic/Adam执行优化，不启动训练worker，不写原预算/统计。
本工具为人工交互测试，不因能播放、短步通过或训练窗口统计授予能力认证。

网页默认仅监听127.0.0.1。显式追加`--panel-host 192.168.250.1`可直接供该网段浏览器控制；
`--panel-host 0.0.0.0`同时监听本机与所有IPv4接口，使用前确认访问范围。所有客户端共享一个命令通道，最后命令生效。
本机3006承载HAPI，不占用或替换它；当前已运行的8765转发可继续用，新的直监听实例不能与它同时占用同一地址。
`DISPLAY=:2 XAUTHORITY=/home/kellen/.Xauthority`指独立Openbox X11会话，不是GNOME虚拟工作区。

### 网页键盘与实时曲线

网页采用旧Motion面板的暗色卡片、命令/反馈曲线、物理按键映射与焦点保护思路；
没有移植旧页面的命令斜坡、手柄仲裁、技能切换或第二个控制循环。
点击“启用键盘控制”，将焦点移出输入框后：W/S或↑/↓前后，A/D或←/→转向，支持组合。
键盘速度在页面单独设置；按住发送设定值，松开对应方向归零，正反同时按下互相抵消，无额外斜坡。
Space停止，Esc停止并暂停，R归零并重置；键盘不自动解除暂停。
输入框、下拉框及Ctrl/Alt/Meta组合不被运动按键抢占；点击“继续”等按钮后可直接使用运动按键。
失焦、切换标签页或关闭键盘控制会释放本页正在按住的方向并尝试发送零命令；
彻底断网时浏览器无法保证送达，服务器仍保持最后收到的命令，不伪称断网自动停车。
数值/按钮与键盘复用同一串行`/command`请求队列，避免松键请求超越按下请求；不重复后台发送静态命令。
曲线最多300点、最近30秒，reset清空；缺失信号留空，不补零，不写浏览器存储或文件。


少量高度训练准备配置：`configs/jobs/motion_height_small_dose_prepared.json`。使用原v3课程将10%总样本分给0.20～0.40m六个离散高度的站立与±0.25m/s平地运动，先变高再运动，55%基础复习保留。准备配置零额度，实际切换需最终完整停止点及剩余额度SHA分配；详情见Motion能力矩阵“少量变高度并行训练”。不新增授权，不把准备配置或高度输出范围称为已掌握能力。

### 只读训练与能力看板

```bash
PYTHONPATH=src python -m native_stand dashboard --root /path/to/runs --dry-run
PYTHONPATH=src python -m native_stand dashboard --root /path/to/runs --port 8765
```

打开 `http://127.0.0.1:8765`。`--root` 可重复，指定单个 run 或直接包含 run 的目录；不是任意文件浏览器。远端可在远端运行同一命令，再自行 SSH 端口转发；本命令不连接远端。默认手动刷新，可显式开启页面内30秒刷新。没有训练控制、评估或删除接口，也不启动 TensorBoard。

状态和消耗读取 run/state/budget；曲线直接读取 JSONL，不要求 events。RUNNING 不代表进程已核验。训练窗口、固定模型评估、实机证据分别显示；缺失值为 UNKNOWN，不生成能力排名。Motion展示已有 maneuver 训练窗口；Jump/Recovery通用状态和曲线可读，领域能力适配暂不支持。固定模型展示已有评估索引并核对SHA；实机适配尚未支持。读取限制为最近2000条指标、100份诊断，每root最多50个run，普通文件读取上限4MiB；大state仅解析前256KiB的完整顶层摘要字段，不宣称整体校验通过；共享索引超限单独报错，不阻塞其他任务；截断边界和读取错误不代表任务失败。不同评分域不直接比较，训练窗口不是保存点的固定成绩。

若已有 TensorBoard 服务，可显式传 `--tensorboard-url http://127.0.0.1:6006` 提供跳转；只检测已有事件目录，不转换、不写入原run。TensorBoard自身使用方式参见[官方文档](https://www.tensorflow.org/tensorboard/get_started)。目标矩阵从现有Motion文档读取；不包含该文档的安装显示不支持。旧诊断缺少明确窗口计数身份时，样本计数显示UNKNOWN，原始值保留在证据中。评分阈值或条件未产出时不猜测，比较显示证据不足。

Tailscale IP入口可用TCP转发：看板增加 `--public-host <本机Tailscale IPv4>`，仍只监听127.0.0.1；管理员显式执行 `sudo tailscale serve --bg --tcp=80 tcp://127.0.0.1:18765` 后，通过该Tailscale IP访问。HTTP Serve按域名路由，不能替代此IP入口。只对tailnet提供服务，不使用Funnel、不开放公网；其他Host仍拒绝。

看板首页优先选择有指标的最近任务，提供中文摘要卡片和奖励、loss、KL、耗时、梯度、优化步数、LR、Torch显存八组SVG曲线；曲线可悬停看原值，缺指标明确留空。完整身份和原始JSON置于折叠证据区，不默认堆放。领域/任务可筛选；训练矩阵默认展示最近窗口。网页改版后刷新整个页面（Ctrl+F5），页面内“刷新数据”只更新产物。

### 远程算力平台只读接入

每次开启都显式传连接资料，没有源码默认地址、端口或密码：

```bash
PYTHONPATH=src python -m native_stand dashboard --root /path/to/local/runs \
  --remote-motion --remote-host HOST --remote-port PORT --remote-user USER \
  --remote-root /explicit/authorized/run --remote-fingerprint SHA256:VERIFIED_FINGERPRINT \
  --remote-password-prompt
```

密码由后端 `getpass` 输入，仅当前进程内存，不在argv、配置、日志或网页中；不传prompt选项时使用现有SSH key/agent。下一次更换服务器或密码，重新启动**看板后端**输入新资料，不涉及训练。网页允许显式编辑host/port/user/已核验指纹；run允许目录仅由后端启动参数指定，不开放任意路径输入。修改连接后旧快照保留并标过期。已有known_hosts严格校验且额外核对ED25519指纹；新端口需先核对主机密钥再登记，绝不自动接受。OpenSSH/pexpect读取固定stdlib探针，不使用SFTP/scp，不安装远端agent；远端需系统python3，不依赖训练Python。

点击“刷新远端快照”才连接；页面普通刷新只读缓存，不持续SSH轮询。采集总时限45秒、传输上限18MiB，文件白名单与SHA校验，最近5份诊断、20份有界评估、JSONL尾部最多2MiB。过大的候选/索引显示读取限制，不下载checkpoint、不自动hash模型。连接拒绝、认证失败、超时、主机身份问题与训练状态分别报告；失败不覆盖上次成功快照。

资源是瞬时快照：GPU整卡/compute进程显存与Torch指标分开；容器CPU配额取cgroup，affinity不是quota；GPU宿主PID不能直接当容器PID。缺启动身份start_ticks时进程保持UNKNOWN。没有启动/停止/恢复/清理/任意远端命令接口。Jump/Recovery远端路径未授权时不接入。

连接地址、端口、用户名和指纹均为本次运行资料，不写死在源码；不能把上一次GPU、PID或机器配额当成当前事实。网页连接资料只保存到当前后端内存，密码变更仍在后端私密重新输入。默认手动采集；可显式开启“页面打开期间每120秒采集远端”，关闭页面即停止，无独立watcher。快照超过300秒标过期；普通本地刷新不触发SSH。已有看板systemd服务不自动存储远端密码，后台自动重启不会凭空恢复凭据。

### 新训练自动出现在看板

不需要另一套登记表。正式 `wheel-leg train` 已写入 `run.json` 和 `startup_handshake.json`，提交JSON现在返回 `dashboard.artifact_root/run/identity_record`，Codex应把该run及其身份交付给用户；看板登记失败不改变训练提交或预算，禁止重复提交训练来修复显示。

本地看板 `--root /authorized/runs` 在每次页面数据刷新时发现直接子目录的新run。远端 `--remote-root` 可以指定单个run，或由用户明确授权的**runs目录**；点击“刷新远端快照”后自动发现直接子run，不递归扫描磁盘、不扫描端口。最多20个run（按产物文件更新时间选择最近项，不作为训练时间轴）、总传输14MiB；资源只采一次。新run继续复用原SSH连接资料与私密凭据，不加载checkpoint。下次更换平台重新配置连接；远端根目录只在后端参数中授权，网页不能自行扩大范围。页面自动刷新默认关闭。

固定Motion同条件复刻诊断使用`wheel-leg diagnose --config PROTOCOL.json [--dry-run]`，协议版本`motion_replication_probe_v1`。它复用现有MuJoCo播放器，显式有限case/墙钟/积分步/输出空间，保留各Actor合同与1ms证据；不改中心、模型或电控。源Actor、公共plant、命令/事件、完整episode角色和评分门槛均须明示，能力结论以逐条件指标为准。

后续正式Motion蒸馏使用`motion_replication_probe_v2`与`motion_vm1_mujoco_v1`参考合同，`plant.motion_vm1_contract`必须显式提供path/SHA。它绑定vM1原MJCF/网格和历史运行时覆盖；Teacher与Student共用MuJoCo参考，1kHz双轮各>2N承重比例严格>60%，目标为复刻Teacher。完整物理定义与传动比见ARCHITECTURE；既有v1冻结证据保留。仅复制原XML柔顺值的`motion_vm1_contact_20261109_v1`是探索对照，不是完整vM1参考。

离线监督使用`runtime.device=cpu`或`cuda:0`，同一`wheel-leg distill → owner/worker → supervision_flow → BC loop`。CUDA保存RNG/设备身份，完整resume沿用原设备；MuJoCo仍是CPU逐回合积分。资源授权与物理GPU UUID在每次正式提交前核验。

当前多Teacher配方使用Teacher的`observation_domain`，版本`motion_observable_teacher_domain_v1`。C0安静、vM1核心、u2200新命令三个角色只读取当前31D并互斥；不会按回合未来用途给相同输入冲突标签。该字段进入监督身份与label_provenance，旧conditions接口保留给原冻结任务。

承重统计使用每步`mj_contactForce`法向力，包含正距离margin接触的求解器承重；额外保存`positive_gap_normal_force_N`供审计。旧触碰布尔值仍表示距离≤0，不能代替承重力。结果记录读回版本与post-step/forward时点；旧统计结果不覆盖，跨版本比较应重评分或重新同条件验证。

学生状态补教可在dataset来源引用中声明`observation_envelope={version:teacher_observation_box_v1,lower:[...],upper:[...]}`。向量长度必须等于输入维度；范围外及无效transition保留为UNKNOWN，监督循环在Teacher查询前跳过。范围由有资格来源和固定余量给出，不能继承Teacher原回合的成功身份。

诊断协议可显式指定`actor_packet_age_ms`为0、20或30；只同时改变Actor物理包与腿包年龄，目标5ms、轮7/11ms和V4保持。缺省使用冻结名义值；结果与每回合initial_state记录实际时序。应力条件分别评分，不能当作全DR或实机人为延迟授权。

历史MuJoCo接触对照可显式选择`motion_vm1_contact_20261109_v1`：implicitfast/Newton、100/50迭代、接触solref .008/1、2mm margin、闭链solref .008/1，恢复vM1来源MJCF及播放摩擦覆盖。只对当前绑定机械模型恢复这些接触/引擎参数，不能称整份旧MJCF等价。Teacher与Student必须使用同一配置。门槛`support_fraction_comparison=gt`表示严格大于指定比例；本轮用户修订为双轮法向力各>2N的1kHz比例>0.60，旧0.90协议保持原身份。
