# Native 闭链轮腿机器人训练

统一入口为 `wheel-leg`，源码在 `src/native_stand`。公共重构迁移已按本地tag `refactor-migration-v1`（`1cc65e1`）验收结项；后续开发从main建立独立分支/worktree。main已有Motion／Spin、Recovery v8，以及Jump ABI4 MuJoCo CPU链。领域分支的新Recovery v9与Jump Isaac GPU实现不自动属于main支持范围；它们按普通扩展任务独立验收。CPU工程通过不代表模型能力或所有平台认证。

## 主仓与本地依赖

Motion后续配方、评分、选模和交互测试先读[Motion能力矩阵](docs/MOTION_CAPABILITY_MATRIX.md)：0.34m基础优先，转弯、地形和0.20～0.40m高度逐级扩展。它描述最终目标，当前准备配方的实现范围单独列明；训练窗口、固定checkpoint验证与实机成绩分别记录，当前后端支持仍以实际实现为准。

第二轮第一块的准备配置为`configs/jobs/motion_u1500_turn_focus_v3_prepared.json`：约36%转弯、9%地形和55%基础复习，0.34m固定高度，细化yaw阶梯并检查基础保留。完整u1500来源和准备记录通过数据根SHA引用，缺材料不能直接运行。正式CLI计划检查：`PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python -B -m native_stand train --config configs/jobs/motion_u1500_turn_focus_v3_prepared.json --dry-run --json`。配置为零额度，正式提交仍被拒绝；不代表已有GPU或能力验证。具体设计和接续边界见矩阵末节。

后续已获用户新10000次/不限训练墙钟授权，正式配置为`configs/jobs/motion_u1500_turn_focus_v3_cloud10000.json`；旧8429余额退役，历史消耗保留，允许共享GPU。授权及结算依赖在数据根`motion_authorization_20261113/`，旧终态快照在`motion_cloud10000_20261113/round2_handoff_20261113/`。远端冻结源码及启动说明部署到`/root/gpufree-data/wheel_leg_motion/candidate_revisions/motion_turn_focus_v3_10000`与`round2_handoff_20261113/START.txt`。GPU由用户手动开启并通知后再核实UUID/资源、沿正式CLI唯一提交；本配置不监测、不自动启动。材料缺失或资源不符时拒绝；已存在run不得重复train。

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

## Motion MuJoCo网页交互播放

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
输入框、下拉框、按钮焦点及Ctrl/Alt/Meta组合不被运动按键抢占。
失焦、切换标签页或关闭键盘控制会释放本页正在按住的方向并尝试发送零命令；
彻底断网时浏览器无法保证送达，服务器仍保持最后收到的命令，不伪称断网自动停车。
数值/按钮与键盘复用同一串行`/command`请求队列，避免松键请求超越按下请求；不重复后台发送静态命令。
曲线最多300点、最近30秒，reset清空；缺失信号留空，不补零，不写浏览器存储或文件。

Recovery 2100 小模型蒸馏使用 `configs/jobs/recovery2100_distillation.json`：复用监督生命周期，固定2100教师、三次物理输出、按回合/阶段均衡的归一化Huber、学生真实闭环补教和独立延迟比较。运行前先 `wheel-leg distill --config configs/jobs/recovery2100_distillation.json --dry-run`。参数、来源、预算与部署边界见 [蒸馏说明](docs/RECOVERY2100_DISTILLATION.md)。

Recovery 2100 小模型蒸馏当前使用 `configs/jobs/recovery2100_delay_dagger.json`（长延迟学生补教）；初始 `recovery2100_distillation.json`仅保留首轮复现。当前配方：复用监督生命周期，固定2100教师、三次物理输出、按回合/阶段均衡的归一化Huber、学生真实闭环补教和独立延迟比较。运行前先 `wheel-leg distill --config configs/jobs/recovery2100_delay_dagger.json --dry-run`。参数、来源、预算与部署边界见 [蒸馏说明](docs/RECOVERY2100_DISTILLATION.md)。

Recovery 2100 起身小模型：`wheel-leg distill --config configs/jobs/recovery2100_mujoco_rise.json --dry-run`。正式采集复用原XML MuJoCo模型快照、完整V4与统一owner/worker；起身优先，慢速漂移单独记录。配置依赖本机SHA绑定资源，使用前核对可用性。接口与数据边界见[蒸馏说明](docs/RECOVERY2100_DISTILLATION.md)。

真实状态导出可用 `wheel-leg export --run RUN --checkpoint CHECKPOINT --output PACKAGE --onnx --golden GOLDEN.json`，golden JSON声明states NPZ的path/SHA/indices。Recovery ONNX与C完整网络直接输出物理目标，decoder只执行一次；C源码在包内按原名保存，可独立重编译。

## Recovery → supported015 Motion 仿真探查

统一入口：`wheel-leg sim2sim` → `simulation.sim2sim.from_file` → `simulation.recovery_motion_handoff`。
复用 `RecoveryMuJoCoCollection` 原 XML、公共推理 factory、C control_core 解码器及 SHA 绑定 C gate，不创建训练 worker。

```bash
PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python -m native_stand sim2sim \
  --config configs/jobs/recovery2100_motion015_handoff_multicase.json --dry-run --json
# 实际运行时去掉 --dry-run，并指定不存在的 --output 目录。
```

现有 `recovery2100_motion015_handoff_probe.json` 保留单 case 六 variant 对照。
新增 multicase 配置使用同一 31D supported015 Motion（腿 C0_poly5(height)+0.15×mean、轮80×mean），不是旧31D Motion。
输入腿位置/previous仍按1.5rad，previous轮目标按100rad/s；命令为高度0.34m、零速度/零yaw。
`gate_source/gate_header` 绑定 supported015-recovery2100 路径及 finite 配置校验后的 SHA（source 192289e0…、header 4f861000…），本模块不修改电控。

multicase 覆盖源配置全部7个 development Swing 条件（nominal、yaw+90°、DR0.25×yaw−90°、DR0.25、DR0.5×yaw±90°、DR1.0）× Recovery-only/100ms/200ms dwell，共21回合，每回合13s。
100/200ms是gate条件连续满足的持续时间，不是通信延迟。
DR仅采用既有reset-only子集，具体启用范围见 `tasks/recovery/cubic_randomization.py`；覆盖既有7条件矩阵，不宣称DR×yaw全组合或最终留出验收。
保留模型通道顺序，不交换网络数组。

切换使用同一世界，不重置状态、previous、PD或运输队列；Motion影子输出不写执行缓存。
Motion观察沿用Recovery测量编码，保持命令槽22:25=[0,0,.68]，仅替换技能槽25:28为[1,0,0]，由既有C解码器产生supported015物理目标。
门槛高度/倾角/支撑/腿速度来自1kHz物理轨迹；角速度使用该策略步收到的Actor观测范数，在10个1ms样本内保持。
因此本探查不是电控完整采样时相或传感链等价证明。

预算在dry-run校验完整计划：最多546000个0.5ms子步、900s墙钟、1GiB空间。
执行前按不可分割的10ms调用预留20子步；异常按引擎报告的实际子步记账。
墙钟从输出建立后开始，含编译/模型加载，逐策略步检查；不抢占执行中的原生调用，最坏超时量取决于单调用/关闭/文件写入耗时，不承诺硬实时截止。
空间在计划中采用保守原始数组上界及16MiB编译/元数据余量，并在每回合轨迹写入前检查当前目录和未压缩数组大小；不是文件系统配额。
回合达到truncated立即结束，避免下一步自动reset后拼接证据。

输出包含protocol/plan、逐回合trace.npz与SHA、episodes.json、terminal.json。
原始轨迹区分network_mean、requested、adopted、motor、NET、owner及1kHz物理状态；记录实际延迟与真实子步数。
Motion hold10仅计切换后连续质量窗口：高度0.34±0.02m、pitch/roll≤0.18rad、双支撑>2N、闭链误差≤0.005。
Recovery-only单独写`recovery_only_hold10`，绝不计入`post_switch_hold10`。
未切换或窗口不足不能当作Motion通过；故障、速度漂移和HP10 RMS另报。hold10仅说明存在连续质量窗口，不包含速度上限，不等于整回合无故障或完整能力accepted。
旧Motion失败对照保留原配置与原输出身份，不重新跑或覆盖；此路径保留用于历史对照。

CPU测试用脚本I/O引擎替身贯穿正式分派，验证预算、异常记账、完整/不足窗口、基线与连续切换；真实闭环结果另存数据根，不写入旧run。

015普通ChassisTask仍为5ms；RecoverySession由电控负责人接1ms motor callback，以承载5/7/11ms运输。仿真使用既有1ms执行，不验证硬件callback实际调用。
015 raw current limit为16384；本配置没有使用8192包络。仿真轮限矩4.8N·m低于提供的理论换算4.8939N·m，物理电流/力矩标定仍UNKNOWN，不能据此证明硬件限矩等价。

`firmware_motion015_contract_r1/actor.pt` 是原supported015模型的输入合同元数据更正版，原产物只读。全部权重张量相同，128组随机输入推理最大差0；更名不改变编码或模型输出。原模型与新模型SHA及核验记录位于同目录verification.json。
电控实际支撑证据与实际高度生产者尚缺，仿真直接读取物理真值不能代替生产传感证据；本轮不构成硬件交接放行。

### 本体感知Recovery→Motion交接

`configs/jobs/recovery2100_motion015_handoff_posture.json`使用同一电控C的编码器/IMU姿态与构型准备门，不读取地面高度或法向力作为交接条件。准备门要求直立锥0.10rad、角速度≤1rad/s、腿速度≤3rad/s、四腿相对Motion0.34m中心误差≤0.50rad，连续200ms后检查Motion目标连续性。它不是触地证明，悬空直立也可能通过。默认真实电控配置与操作见独立015集成仓中的Recovery2100交接说明。

7个development Swing条件中6个在0.60–0.94秒交接并连续保持12.06–12.40秒，另一个在交接前失败。数据根`recovery_motion_handoff/posture200ms_r2`保留原始证据；r1的窄门未交接证据保持只读。观测时序差异与硬件标定仍须实测，不作为实机认证。

### 仿真与实机的弹簧补偿口径

完整V4训练、评估与sim2sim不额外添加软件弹簧补偿。实机015 Motion保留原补偿；本集成电控40e1b8b在Motion目标实际采用后恢复同一补偿。集成Recovery当前没有补偿，但这只是代码现状，不能由V4仿真不加补偿推论实机Recovery不需要；该模型部署合同仍待确认。电控软件也是软件，因此避免统称“软件层面都不需要补偿”。

如果V4采集系统当时包含软件补偿＋真实弹簧，其整体等效响应可解释这种路径差异；当前尚缺采集开关确证，不能宣称完整实机等效已认证。下面原A/B属于历史附加力矩敏感性诊断，不是当前仿真执行建议。

### 015软件弹簧补偿历史A/B

`configs/jobs/tjb_recover_motion015_spring_comparison.json`经正式`wheel-leg sim2sim`比较旧tjb_recover在当前PD40/2/.2、40/4.8Nm下有无015补偿；`configs/jobs/motion015_spring_standing_comparison.json`另比较015 Motion从站立启动。两者仅仿真，不默认改电控固件。

variant的`spring_compensation=true`启用SHA绑定的电控`RLControlCompensateSpring`和`getPhi`。补偿按5ms更新并保持，仅补偿项保持，V4原电机响应仍1ms变化；按真实网络→电机顺序/符号换算，相加后腿限40Nm，轮不变。保持相同V4 NET/地面/随机种子，未额外添加被动弹簧模型。

当前V4 NET是拟合负载而非经实机认证的明确被动弹簧。此试验比较“在原仿真上附加015补偿命令”的效果，不能等同于有真实弹簧实车的补偿辨识。C依赖SHA、原始物理轨迹、实际消耗、饱和与补偿量随run保存。

[015补偿A/B结果、调用链和仿真边界](docs/SPRING_COMPENSATION_AB.md)。

[015补偿A/B有效性修正：不能据此否定实机补偿](docs/SPRING_AB_VALIDITY_REVIEW.md)。
