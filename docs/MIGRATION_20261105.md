# 工程架构重构迁移报告

## 一次性结项记录（2026-11-13）

本次公共重构迁移正式结项。本地不可移动tag `refactor-migration-v1`指向已验收`1cc65e1234839870d24e31eac95214fbf88a0514`，本轮重新核对main与交付收据61项文件SHA一致。验收范围为下文冻结快照：核心636通过/10历史skip，portable1920通过/32历史skip；结果不是新领域代码或模型能力证书。治理收尾只修改协作文档、架构检查器及反例、CI/PR模板，不改奖励、课程、控制、模型、预算或活动训练依赖。

主仓main已有Jump ABI4 MuJoCo CPU链；独立`jump/isaac-training`中的Isaac GPU链尚未合入本结项基线，其进展由领域owner独立记录。Recovery v8验收与后续v9配方同样分开。旧缺失数据/无效候选保留UNAVAILABLE，真实物理、GPU性能、模型能力及新后端按正常任务验收，不阻止迁移结项。

从此新功能和普通修复不再追加本迁移报告：用README维护支持与用法、ARCHITECTURE维护职责/扩展、AGENTS维护稳定约定、STATUS维护当前任务、run/artifacts维护实验与故障证据。结项治理CPU命令/结果/最终提交身份留`artifacts/migration_closeout`及数据根对应小型收据；无remote、未push，CI required checks尚未在平台强制。

以下原文及收据保持各自身份，不因结项改写旧证据。

公共性能观测与Recovery/Jump增量已集成，冻结`b60f0d7`在干净clone验收：核心636通过/10历史skip/3既有ONNX warning（122.43s），portable1920通过/32历史skip/10warning（125.69s），架构209/0，CI Ruff0.8.4 check及208文件format通过。61项准确文件与SHA、完整命令、前两次fixture失败和最终通过结果绑定`artifacts/integration_final/performance_delivery.json`及`performance_final_checks.json`，数据根归档`integration/performance_observability/b60f0d7`。后续文档提交不改已测执行文件。旧结果保留各自身份，不拼接。

公共Measurement只保留一份：driver→GenericPPO→ControlledPPO→环境显式共享，关闭时不访问CUDA；嵌套子项默认CPU墙钟、父项inclusive不可重复相加。`after_save_total`与候选/引用、旧环境close、新建、首次reset-readback、编译分开；启动单列，计数在update边界集中落盘。MetricsWriter修复性能摘要被白名单丢弃。保存/计费/引用/重建/轮换顺序与原异常保留不变。Recovery只对reset行处理V4、合并结束读回、缓存合法模板和partial profile，保留真实物理读回/909D/BC/prior/双Adam；Jump复用ABI4 C构建及输入，避免重复V4/观测/terminal/临时分配，保留6370D/235字段和C/FSM。运动2交付窗口/累计与候选多原因、重建输入归因，不改变资格或是否重建。

CPU组件微基准（非整轮GPU结论）：一万轮原每update两临时测量对象median4.97ms/P955.04ms，复用run对象reset为0.479/0.491ms。Recovery 1024环境脚本I/O，3预热/21重复，单行reset median8.396→1.895ms/P958.938→1.997ms，16行9.501→2.346/P959.770→2.395，全量102.589→39.839/P95106.232→43.114；原始分配/调用数在`recovery_reset_benchmark_warmed.json`。Jump组件收据保留所有结果，包括缓存使retained内存上升，不外推整轮吞吐。真实GPU事件、PhysX/reset及新源码真实训练NOT_RUN；不改变网络、采样量、奖励、时序、DR或预算。

干净clone禁止旧仓/原工作树读取的四配置dry-run、Recovery真实完整u1800状态（Actor/Critic/双Adam/std/CPU与CUDA RNG字节/warmup5）及部署golden通过；Jump真实Actor、ABI4固定源码/库、6370/235、静态恢复通过。CPU恢复不声明PhysX隐藏现场恢复。旧142缺失数据与失效候选继续UNAVAILABLE。Recovery owner独立新v9任务不混入本轮v8回归；其活动源码位于独立worktree，未被公共合并热改。本公共轮新真实PPO/监督/物理/GUI/GPU/远端/真实删除均0。

## 主仓基线与依赖归档（2026-11-12，当前收尾）

主源码仓固定为`/home/kellen/wheel_leg_refactor_ws`，独立Git/main；旧仓及其.git保留，不交换目录、不转发软链接、不配置remote或push。源码/测试/CI/两套共享C源码及小型契约纳入基线；run、预算、模型、机器映射和原始证据留文件存储。精确提交与干净clone收据保存在本地`artifacts/repository_baseline`，不提交运行数据。

本地数据根`/home/kellen/wheel_leg_training_data`从既有6074包恢复1930项SHA依赖，新增内容1782908312字节；同内容只在数据根内部硬链接，不链接旧仓原文件，非独立备份。当前四配方依赖另有12项小型增量7120399字节；`manifest.json`/`current_manifest.json`/`runtime_paths.json`记录真实来源及映射，`models.json`仍为既有ModelIndex。禁止旧依赖根读取下四配置dry-run、真实Recovery u1800完整加载/双Adam及部署golden通过。旧142份缺失数据、失效best SHA继续UNAVAILABLE；归档不代表正式handoff或训练授权。

527/旧200文件/6074包仅是各自历史冻结；此后Motion接触/H1/881及提速另有领域收据，不能用旧结果覆盖。Motion活动远端由唯一owner管理，本侧不连接、不停止、不热改；最新实际状态以其run及领域交付为准。H1长度冲突已由领域新增CPU实现处理，真实自然终止、完整保存恢复、GPU吞吐仍须对应版本工程证据，不因Git完成而视为能力通过。Recovery接续仍需其负责人明确交接；Jump无Isaac GPU后端。

本轮未创建训练run、未预留额度、未做GUI/GPU/远端或真实删除。工作树portable审查时误选了已有MuJoCo历史短循环，发生一次受1.5秒墙钟限制的非训练积分，实际子步数未记录，不能报告物理积分0；随后为portable入口加了物理/CUDA/AppLauncher失败保护，干净clone只运行自包含CPU工程。CI portable包含历史条件skip，具体原因按本轮收据逐项列出；新增Git/依赖消费者不skip/xfail。源码主仓建立、本地依赖准备和真实物理/云端验收分别记录。

## 前次总集成冻结摘要（历史快照）

- 公共生命周期、PPO效率、多Teacher、累计聚合、补教、停止恢复、Recovery、Motion／Spin现有adapter、Jump ABI4 CPU、播放、部署、引用和搬迁：同一最终源码快照527 passed、8历史skip、3既有ONNX警告（118.26s），测试后源码SHA无漂移；架构200文件/0例外，CI Ruff0.8.4 check及199文件format通过。精确命令/文件SHA/结果在`artifacts/integration_final`。8项skip只因`test_motion_spin_confirmation.py:12`所需旧local750父配置/checkpoint缺失；新增消费者未skip/xfail。
- 公共故障边界已修：优化完成后计费，后处理成功才允许补存完整状态；半完成rollout不保存当前世界，保存失败不重试覆盖；清理异常保留原始故障。Generic adapter在优化后、telemetry前给完成收据。checkpoint发布不覆盖同名已引用文件。r4实际消耗27与u16模型进度仍分开，旧账本未改、不退款；缺失更新不能用普通resume前检绕过。
- 实时sim2sim已接统一CLI、推理factory、顺序collection／公共播放器适配、统一轨迹和既有诊断。脚本I/O核验实际条件、短步故障、终止和覆盖；真实跨引擎积分NOT_RUN。Motion在线collect公共链已用脚本I/O通过collect→label→aggregate→fit；领域make_collection_env／configure_diagnostic已由运动2冻结交付（motion_cpu_wiring_delivery.json，200文件SHA逐项核验），公共正式factory→两轮collect→累计view→多Teacher标签→真实小模型BC已通过；采样故障实际子步保留、无完整transition不伪造。实际PhysX采集和跨引擎积分仍NOT_RUN。
- 旧双Teacher索引142份NPZ及对应source trace均UNAVAILABLE，不能按原数据域续训。u1600 pin和C0 Teacher文件仍存在；另有完整offline70数据域，不能代替原142来源或回灌holdout。逐项只读结果在`artifacts/integration_final/historical_data.json`。
- Recovery当前v8／own-hold-v2／909D／4.8Nm／全池BC+受限prior，真实u1800 CPU装配/save/搬迁/load/adapter恢复完成，双Adam、std、RNG、warmup和已结束统计保留，禁止原根读取通过。旧owner PID1901402/start_ticks136388404自行因max_output_bytes在1800次STOPPED_BOUNDED、exit0；本侧未停止/重启。祖先1666+本段1800=3466，剩6534；正式handoff/真实接续仍需明确授权。旧best候选SHA失配为UNAVAILABLE，不能冒用实际u1800 SHA补认证；standing u1400身份有效。
- Motion领域冻结清单200项逐SHA核验无漂移，manifest SHA `64e968a07ac31e41e3f38bc3475eeff72a3b0e93bb059947edd48635310f8a39`。领域make_collection_env／configure_diagnostic已交付，不能再将其列为缺口。运动2远端CPU收据`artifacts/cloud_handoff/motion_remote_cpu_readiness.json`（SHA `dea13f2c3a4c512ffacb4288c87c2ff996853eea97abb614f3dd56d0bca09c25`）记录61份执行源码/准备脚本同步核对、真实u3000及双Adam/KL游标CPU装配、2432份SDK文件核验；r9正式10000 dry-run通过、run不存在、active worker0、新额度消耗0。此为领域owner远端CPU证据，不与本侧527项本地回归拼接为GPU通过。
- 当前包仍是已验证的本地迁移准备快照，不含上述后来新增远端receipt，不因文档更新重打包或改写旧包manifest。Motion由运动2唯一提交，本侧未分配额度、连接或启动远端；后续仅GPU恢复后按既有授权唯一正式提交。`motion_gpu_concurrency.json`仅CPU估算：单4090先独占Motion，Recovery并行未实测，不绕过14000MiB前检；Jump保持ABI4 MuJoCo CPU，没有Isaac GPU后端。GPU首更、峰值显存、吞吐、物理与并行性能仍未验证。旧4889/4825/旧包均仅对应历史快照。

以下章节保留分批历史证据；其中旧PID、RUNNING、余额、包SHA及通过数不代表当前状态。当前包验证收据另见`artifacts/integration_final/package_verification.json`（未生成时表示尚未验收）。本侧本轮新真实PPO/监督训练/物理积分/GUI/GPU/远端任务/真实产物删除/电控写入均0；/tmp小模型优化及删除fixture不属于真实任务。

## 公共迁移与Recovery分批记录（历史快照）

### 云依赖有限闭包增量（2026-11-12）

`cloud_bundle.inventory`对内容SHA匹配的JSON引用继续读取嵌套显式引用，保留每文件多消费者、按resolved路径+SHA去重，引用对象附带的其他依赖也继续处理。原recipe由已解析JobConfig消费，不重复展开；只读取SHA绑定JSON，不反序列化PT/NPZ、不扫描runs、不追history/traces/evaluations。深度16、JSON512份、单份8MiB为失败关闭上限；错SHA、读取期间变化、损坏JSON或越限进入unresolved，package拒绝。现有allocation→retired/prior state/budget、中心绑定→source_binding均进入真实闭包。

r5当前配置只读inventory：24份JSON、1796项产物、unresolved0；这是准备时快照，不是r5终态或最终handoff。新增test_cloud_dependencies.py覆盖嵌套闭包、多消费者、错误SHA不信任、历史排除、深度上限、/tmp打包及依赖整体搬迁；联合公共迁移回归38 passed（21.00s，3条既有ONNX警告）。架构194/0；CI Ruff0.8.4 check及193文件format通过。证据仅/tmp/wheel_leg_cloud_closure，未重打云包、未连接/上传/启动远端，未改Motion/Recovery领域、真实run或预算。

最小路径映射：由manifest生成files[path]=已确认持久盘bundle根/archive_path；prefixes仅将候选源码旧根映射到远端candidate根；JobConfig.assets.path_mapping记录映射文件绝对路径+SHA，source_root显式远端candidate/src，runs-root、cache/log另在持久盘。不能将整个/home/kellen盲映射到远端工作目录，也不改原始证据。

long continuation应单独绑定smoke终态完整checkpoint SHA、实际计费与原总额度；当前BOUNDED_SMOKE接口仍明确formal0，不能直接改数绕过。r4已耗27不退款，只有r5真实完成37时累计已耗5175、余额4825成立；full continuation保留训练/课程/统计状态，不能再次走u3000新分支初始化。由Motion唯一owner提交，本次不实现或启动新的长训分配。

### 公共PPO／Recovery／Jump效率（历史CPU验收）

公共rollout使用inference_mode，returns在no_grad中生成普通常量目标；学习保持梯度。ControlledPPO保留原GAE递推，只做一次有效Actor全rollout优势归一化；Critic仍消费全部transition。每minibatch复用一次资格计数，纯日志detach后设备端float64累计、update末集中读回；Jump总样本门与minibatch门不变。warmup／不合格Actor保留随机前向及分布缓存，仅关闭无用途反向图，不改RNG、双Adam、std、BC采样或shuffle。

Recovery合并45个基础Critic字段finite回读为1次，故障仍在观测边界拒绝；physical_profile在reset／DR及恢复边界失效，同tick复用。环境发布已detach的transition快照，backend不再deepcopy一次；prior MAE只在诊断消费边界回读，缓存常量与字段索引，保留冻结Teacher真实前向、max_rows、BC全池分层和每Actor minibatch调用。Actor变化核验仅保留实际消费的前5次／第12次。Jump复用step返回的下一观测，仅结束时复制request；消费游标推进后由pending窗口持有结束摘要，批次不再无限保留已消费记录；schema保持真实6370D/235字段逐列校验，复用布局而不反复构造JSON摘要。ABI4、线性raw、owner/FSM/V4、子步、reward和partial reset保持。

显式diagnostics.performance_measurement开启时，公共engine/ControlledPPO/driver记录rollout、GAE、minibatch取数、Actor/Critic前向、backward、BC/prior辅助项、Adam、diagnostics、capture、checkpoint/SHA、候选引用/视图、轮换、metrics写入、整轮墙钟和阶段均值。performance.jsonl区分保存轮；整轮计时包含主metrics写入，不包含其自身计时行写入。默认不增加CUDA同步；详细测量仅在阶段边界同步，CPU墙钟不冒充GPU kernel时间。

正式新增test_ppo_efficiency.py覆盖真实PhysicalActor+BC/prior、Jump DirectActor、四类mask、warmup、raw/log-prob/分布、GAE/targets、双Adam、RNG、finite/infinite terminal、脚本Jump短子步/partial reset、schema漂移和driver计时。最终完整范围450 passed、8项既有历史750来源skip、3条既有ONNX警告（111.27s）；未新增skip/xfail。首次联合的2项低磁盘前检失败与1项并发Motion错误文案失败已处理；最新前检fixture的资源I/O独立于主机空闲量，并额外验证真实0磁盘拒绝。最后前检增量结果见/tmp/wheel_leg_ppo_efficiency/preflight_final.txt。架构194正式文件/0例外；CI Ruff0.8.4 check通过、format193文件通过，本轮20个实现/新增测试check和format通过。

CPU微基准：seed123、Torch单线程、预热10次、50次测量。小模型rollout+returns median/P95为1.497970/1.576397→1.193510/1.244217ms，saved tensors 64→0；Recovery pack为0.637699/0.653272→0.446673/0.461350ms，finite标量45→1；235字段schema比较为1.074651/1.089935→0.203554/0.214414ms。只代表固定CPU组件，不外推整轮GPU比例。华南虎、复旦原始plane/jump与派生runner/PPO/storage仅只读参考，未照搬估计器、timeout、归一化或物理参数；RSL storage已预分配/flatten，未另造storage或改变取样顺序、尾部处理。

剩余：Recovery完整fine_ring与1ms写入仍保留，因为当前完整执行快照/故障证据格式包含它；909D普通/terminal及物理几何仍有可审计重复，未用未经证明的缓存删除。Jump仍逐环境MuJoCo CPU，不是GPU批量后端；6370D真实取数成本保留。Motion专用post-Adam KL回退不是本轮Recovery/Jump瓶颈。AMP、TF32、compile、CUDA Graph、fused Adam均未默认开启。真实GPU A/B、物理、GUI、远端、ARM与任务切换NOT_RUN；旧云包仍绑定优化前快照，本轮未重打包，不能冒充当前源码包。

首末只读观测Recovery PID1901402/start_ticks136388404均RUNNING；观测窗口formal1641→1679（原任务自然消耗38），剩6655。最新落盘1600 SHA17b108d351f147f812e0e4104483d3c4f7ecb8228efbd62cdac9c38014c638de，不是最终handoff。本轮新真实训练/物理积分/GPU/GUI/远端/真实删除/电控写入均0，未修改生产、环境、预算或Git。本轮所有测试产物与微基准在/tmp/wheel_leg_ppo_efficiency；完整命令、源码SHA、原始结果和精确文件表同处保存。

Motion rollout最新增量已纳入总集成：正式hotpath/速度合同/生产adapter与公共监督、Recovery消费者联合64项通过（22.82s，3条既有ONNX警告）；架构188/0、CI范围Ruff check/format通过。云源码清单和准备包重新绑定本批SHA，旧406项结果仅代表其原快照；组件传感包CPU减时不代表GPU整轮提速。

累计collection视图已接正式label阶段：两轮100新样本加100旧回放得到200/300，来源/样本去重，episode split跨重标注和恢复保留，holdout不回灌，标签记录实际Teacher路由与合同。采集按真实环境行数/设备/owner/子步计费，分块持久化；不再使用评估envs包装器。监督仍共用owner/worker/Budget，支持正式CLI轮边界停止、新attempt恢复、额度耗尽后的compare/export、保存/发布故障判定和部署幂等发布。

Recovery脚本物理I/O通过真实v8奖励、own-hold-v2、909D普通/terminal、PhysicalActor及BC/prior、双Adam优化、公共worker保存恢复和双目标候选。物理标签与确定性forward一致；播放器轮50rad/s不重复cubic解码。完整监督跨目录导入验证了原根目录不可用、依赖一起搬迁、预算不刷新和后续导出。

**Recovery当前配方承接与rollout提速消费者（2026-11-12，CPU准备完成，未切换）：** 新证据为`artifacts/cloud_handoff/recovery_continuation_cpu.json`，交接模板为同目录`recovery_continuation_template.json`，52项实际直接依赖/角色/SHA及三个模型角色核对在`recovery_continuation_dependencies.json`。未重打旧云包；旧包不能冒充本次源码。

正式链为`wheel-leg train/resume → config/dispatch.execution_service(generic, isaac_recovery) → owner → worker._execute_stage（登记初始化/恢复及BC、prior、runtime绑定引用）→ RecoveryAdapter/GenericPPOAdapter → recovery_isaac.backend.IsaacRecovery → environment.build_class → Execution + preentry_lifecycle(v8) + swing_priority_course(v2) + observations.observe(909) → policies.factory → recovery_cubic.build_policy → PhysicalRetentionPPO/ControlledPPO → driver.run_stage → Context.save/SHA → after_save/ModelIndex候选发布 → 轮换`。本轮只替换物理I/O做CPU消费验证；未创建训练owner或真实引擎。

配方保持31→2048→1024→512→6、Critic909/512/512/256、1024×128、4epoch×4minibatch、LR1e-5/9e-5、原gamma/lambda与v8密度。首回合full Swing，Student秒数配额.15/.075/.075/.70，自己连续保持10秒；23秒有限终点不bootstrap。PD40/2/.2、腿40/轮4.8Nm、1/5/10ms、Actor20ms/目标5ms、轮PD后7/11ms仅一次、V4电机/NET分路均沿真实factory/执行CPU消费者核对。Recovery腿仍是中心三次绝对目标±1.5rad，轮100×mean；不是Motion±.15包络；raw物理Gaussian腿.02rad/轮2rad/s、实际adopted previous不变。

本次修复限定在新仓库：`artifacts/handoff.py`修复pool/best/last_review共用记录对象导致重复映射而拒绝合法导入，仍保留路径/SHA错误拒绝；`engine.py`在真实恢复边界把旧单GPU CUDA RNG Tensor明确封装为一条设备流，错误设备数仍拒绝；`recovery_adapter.py`在空项目索引恢复checkpoint携带的映射后候选并先取得保护，pending只做一次原子发布，空候选不伪造review，UNKNOWN/早失败不制造成功；`phase_readback.py`及环境/backend补齐生产收腿、转腿、抬升、首次入口与最大连续保持统计的更新/reset/完整保存恢复，既有统计不加入奖励或课程门。公共driver原有“同update已经保存则终态不重写”已经生效，新增消费者反例验收，不重复实现提速。

联合252项CPU通过（62.22s，3条既有ONNX警告）；最后候选别名导入修复后的针对性70项通过（15.92s），覆盖实际大模型factory/完整来源、公共worker、两Adam和RNG、PPO精确等价及9项接续测试；原始输出为`recovery_continuation_{tests,incremental}.txt`。架构195文件/0例外，Ruff0.8.4对6个正式模块、脚本和测试check/format通过；相关架构/phase2/phase3回归包含在252项中，无skip。portable小模型精确优化对照和HOST实际大模型恢复分别记录，不混称GPU验证。

真实来源演练先核u1700 SHA`a8e7478f…82919`，结束又对已落盘完整u1800 SHA`578f22a8a03a39235c4b7bbed9985efbf226ffe2ebd3d692d9176080261c6ea2`执行factory装配→capture/save→整个临时目录搬迁→load→正式adapter恢复。后段禁止读取旧根目录，实际31,040条BC与prior/合同一同搬迁；Actor/Critic、双Adam的所有moment/绑定/std、CPU RNG与1024行完整可观测执行状态逐项一致，CUDA RNG逐字节保留，warmup=5不重复。实际大网的BC和prior隔离损失均产生非零Actor梯度、RNG恢复后辅助损失精确重复；对应数值及路径在CPU摘要。GPU RNG真实执行/连续性NOT_RUN，CUDA映射用受控CPU替身反例验证。原checkpoint无新游标/窗口，4096行结束尾部只保留UNKNOWN历史，不假造新窗口；新schema窗口、游标、pending和两目标候选由临时fixture验证。新run full、同run恢复和fresh reset边界分别覆盖。

公共提速已被Recovery实际消费：rollout inference_mode；无梯度GAE；仅一次全rollout有效Actor优势归一化；设备端指标累计/资格计数复用；无Actor更新时保留随机前向与分布缓存而关闭图；profile按reset/恢复失效、finite合并和transition少复制。无AMP/compile/CUDA Graph/fused Adam。`fine_ring`1000×N×65的1ms写入仍被完整快照/恢复消费，未裁剪；909的target/q-dq/motor/NET各自环形历史保留。`_get_dones`的pre-reset Critic与后续普通观测仍有重复几何/打包成本，尚无真实GPU分段数据，不以不安全缓存删列，也不阻塞主体交接。真实GPU rollout秒数与整轮收益NOT_RUN，不能把CPU组件减时换算倍数。

**结束发现的生产终态，非本会话操作：** 开始PID1901402/start_ticks136388404、cwd旧仓库、完整argv保存于CPU摘要，RUNNING1798；结束PID已退出，run/attempt为STOPPED_BOUNDED、exit0、reason=`max_output_bytes`、本段1800。祖先1666+本段1800=3466，原10000剩6534；最终点与账本均为1800，不从早期演练点1700退款。751项生产源码/冻结依赖SHA不变，52项直接来源重新核SHA。旧state的`best_candidate`仍记u1800周期SHA`6a96cb47…3f40d`，该同名文件已被旧终态保存重写为`578f22a8…c6ea2`；原记录不改，作为STALE_SHA证据保留，不在新索引伪授合格候选。standing u1400 SHA`6132cba2…762a`仍匹配。新公共driver避免该同update重写，正式候选引用保护在/tmp验收。未对旧训练发信号、停止请求或重启，不因其自行结算擅自切换。

交接模板仍`owner_terminated=false`/`switch_authorized=false`，正式checkpoint/终态/结算字段留待切换边界重核；另含只读observed_terminal供复核，不是可执行授权。实际训练状态可完整导入，PhysX隐藏solver状态不承诺逐位恢复。建议未来显式`assets.execution_resume="fresh_reset"`：保留课程/RNG/已完成证据与预算，丢弃未完成回合的dwell/hold/执行FIFO并记录边界，不接成旧回合或制造完成记录，不退还这些回合已耗样本。

本机CPU命令（本轮实际执行，输出目录须是未存在的/tmp路径）：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/kellen/anaconda3/envs/issaclab/bin/python scripts/recovery_resume_rehearsal.py \
  --config configs/jobs/recovery_preentry48_remaining.json \
  --checkpoint /home/kellen/wheel_leg_native_train/runs/2026_11_11__recovery_preentry48_remaining/checkpoints/model_update_1800.pt \
  --sha256 578f22a8a03a39235c4b7bbed9985efbf226ffe2ebd3d692d9176080261c6ea2 \
  --output /tmp/recovery-continuation-NEW-UNUSED
PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python -m native_stand.cli train \
  --config configs/jobs/recovery_preentry48_remaining.json --dry-run --json
```

后续切换只剩：按新安排确认旧owner终态与最终源/候选身份→填写manifest、最终记录/账本/来源SHA和实际路径映射→在目标持久盘准备同逻辑run的run.json/ModelIndex及原8334本段额度、已用1800的budget（原始总额度仍10000）→公共`import-checkpoint`→同run `resume --max-updates 12`有界工程窗口。12次从6534扣，Actor-only初始化和5次warmup均不重做；后续继续也不刷新预算。准确CLI为`wheel-leg import-checkpoint --manifest FILE --mapping FILE --destination RUN/checkpoints/import.pt --run-id LOGICAL_ID`，再`wheel-leg resume --run RUN --checkpoint RUN/checkpoints/import.pt --max-updates 12`，本轮均未执行。source域新增`model.kind=mlp`只是显式默认元数据，源/目标摘要分别保留，不能混作同文件SHA。GPU UUID/CUDA映射、IsaacLab/PhysX与目标资源、真实控制/限矩/延迟、自然终止/reset、有效优化和完整保存恢复留给该窗口；不要求独立评估或重新学会起身。本轮物理积分、新真实PPO、GPU、GUI、远端、真实删除、电控及Git操作均0。


最终核心406通过、8历史skip、3条ONNX导出警告（114.48s）。8项是既有历史750来源缺失，不是本轮核心用例skip；新增消费者未skip。其后仅BC导入新增“不覆盖来源”保护及对应fixture更新，受影响81项回归通过（30.72s）；随后运动2增量涉及adapter/host测试，受影响76项通过（18.20s）；测试分别绑定源码快照。架构188正式文件/0例外；CI Ruff0.8.4 check通过、format187文件通过。命令/测试清单/源码哈希在artifacts/integration，不能沿用旧314全绿。

仍有代码缺口：sim2sim仅迁入已录制轨迹诊断，实时跨引擎采集未迁完；Motion／Spin在线收集需要领域make_collection_env及owner/条件/子步接口，缺少时正式拒绝而不回退evaluation工厂；旧双Teacher142项索引引用NPZ缺失，审阅配方只能dry-run，真实提交拒绝。新仓库尚不能称全部常用流程迁移完成。

结束只读发现Recovery PID1901402/start_ticks136388404仍RUNNING，已耗1446/剩6888；最新已落盘1400 SHA6132cba2ed634f40c8e845a47f49dfc3542ecdbe64d2b3b233cf5d8a8bcd762a，不是最终handoff。Motion已停止，分支领域结算见下一节。精简云包2824文件逐项SHA核验通过；在另一临时目录，禁止读取旧项目路径，四份配置dry-run及真实Recovery Actor加载通过。云包为准备快照，无SSH/持久盘/镜像/GPU映射；远端API、PhysX、GPU、真实积分、GUI、ORT、ARM及实机均NOT_RUN。Jump只有ABI4 MuJoCo CPU。

本轮真实PPO/监督训练/物理积分/新GUI/GPU/远端任务/真实删除/电控写入均0。小模型CPU优化与/tmp逐文件删除为工程fixture，未改生产run、模型或账本；未提交Git。

### 本次正式消费者与证据

- CLI/配置/owner/worker：SupervisionConfig与有序phase；distillation_job仍为原worker分支。fit计费与保存失败不退款；已保存未发布可修复；完整BC导入不覆盖来源。PPO、监督round/优化步、物理子步分开。
- 数据/算法：artifacts/dataset.py、supervision_flow.py、routing.py、同一loop.py；labels保存Teacher集合、实际teacher_owner、物理episode和split。新数据域先计算incumbent基线，保留Adam/RNG及旧域best引用。真实Teacher两个不同宽度CPU均值核对见teacher_inference.json。
- Recovery：recovery_isaac/observations.py的同一909列pack由普通和pre-reset terminal消费；环境/奖励/BC/prior与公共driver组成脚本I/O闭环，见test_public_migration.py。未声明恢复未保存的PhysX现场。
- 播放/导出：InferenceSpec同一factory；Recovery用physical forward，Jump线性raw。部署含PT/可选ONNX、合同、执行、C、golden和来源；源目录移走后重建匹配。ONNX reference evaluator不等于ORT/ARM通过。
- 场景：Recovery与播放器实际消费平地高度、摩擦和合法初态；不支持的场景拒绝。无rough能力声明。
- 引用：仍为项目ModelIndex；累计view/chunk/labels/selected/full state及依赖登记。多消费者/删除竞争/硬链接/替换身份回归只在/tmp，真实目录不执行删除。
- 机器证据：changed_files.json精确公共文件清单；core_test_files.json为完整核心测试文件；core_regression.txt记录结果；source_before_core.json与source_final.json绑定源码；activity_start/end.json保存完整argv、cwd、源码根、状态与预算；dependencies_current.json及同名准备tar记录精简依赖。

核心执行环境：PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1；Python为/home/kellen/anaconda3/envs/issaclab/bin/python，python -B -m pytest -q -rf -p no:cacheprovider，文件列表为artifacts/integration/core_test_files.json。架构：python -B scripts/check_architecture.py；Ruff0.8.4范围按.github/workflows/cpu.yml，未新增例外。

云端首次运行前：填写SSH、持久盘容量/挂载、镜像身份及GPU映射；本地核验manifest、映射后dry-run与模型SHA；活动Recovery必须由其owner完成边界停止和最终结算后才允许导入。Motion／Spin共享余额未分配，prepared不可提交。此处不授权启动或追加额度。

## Motion／Spin领域交付（2026-11-12，当前）

CI portable最小收尾：`tests/test_motion_rollout_hotpath.py`补实际可选contact时序fixture与pre-reset采样快照独立性断言，并把TensorDict首次导入移出稳态scalar-read计数（不放宽一次同步门）。旧`configs/jobs/motion442_scale_corrected_resume.json`仅显式覆盖课程版本，与原spec的`student_training.course_version`一致；仍由既有LEGACY_VERSION分派到冻结fixed_recovery，不转为现代课程，也不接纳任意未知版本。`tests/test_motion_supported_scale_corrections.py`核对绑定和未知版本拒绝。相关120项通过（7.20s），冻结Git索引627文件、受管fixture加三文件增量的独立/tmp回归32项通过（1.51s）；架构206/0、Ruff0.8.4 check通过。未改学习目标、生产路径源码、公共生命周期或远端训练。其余C编译工具和旧播放器断言由公共owner处理，本侧不宣称全套portable通过。证据沿用`artifacts/repository_baseline/motion_fixture_followup.json`。

retry_evidence的portable增量仅改`tests/test_motion_differential.py`：历史build/verify用例完整检查原终态、评估、trace与来源依赖，缺失属于历史skip；新tmp合成证据包始终运行真实retry_evidence，正例绑定18份SHA及零PPO消耗，四种反例验证缺终态/篡改trace/已消耗PPO/活动attempt拒绝。联合43 passed、2项历史skip；未重建或改写真实历史证据。收据`artifacts/repository_baseline/motion_retry_fixture_followup.json`，原训练与r15均不接触。

本节替代下方11-11摘要中的Motion活动状态和u600测试入口；其他领域由公共集成负责人验收。原领域准备/速度阶段仅CPU；用户随后另授权下节有界Isaac smoke，不含长训练。

### 新授权：压弯／0.20m／四种H1工程smoke

**速度优化接续覆盖（当前）：** r12正常停止于完整u88（SHA`85c879d079008c6e4c4ff7c2656f86eeaba647255000d53cbe23939ad496955b`，88次/23068672样本、failure=null）。公共handoff导入、同逻辑run接续至`2026_11_12__motion_contact_grid_cloud10000_r15_speed`，源码独立冻结`candidate_revisions/motion_speed_r15/src`，worker1461/start381772380有界核验时RUNNING至u90，failure=null；u89 Actor/Critic各16次真实优化、完整u89保存SHA核验通过，warmup50未重做。原10000总额度此前16+r12的88=104，余9896，不创建新额度、不重复warmup50。两次0更新恢复失败原记录保留；根因为PPO未调用既有reopen_for_resume、以及新增前检误用通用updates而Motion实际保存new_run_updates，均由明确来源/预算身份校验保护的公共worker路径修复。

- `simulation/isaac/response_acceleration.py`显式接收已构造冻结Response，按原状态采用，向量化128槽暖填与编译两类PD/hold张量阶段；不替换模块全局对象、不改V4公式或因果队列。首次编译保护策略RNG。
- `critic_cache.py`缓存常量/历史gather索引与reset-only实际DR参数；partial reset使缓存失效，普通及terminal Critic881合同不变。`ground_surface.py`仅显式开启的Motion按真实物理tick/reset epoch复用同一XY查询，有限路面与出界判定仍有效。
- `motion_backend.py`返回分段performance_seconds；metrics白名单写入原公共driver消费者，当前job显式启用阶段计时。没有逐物理步新增同步；原128槽真实历史、640物理子步/更新、4096×64样本、奖励、DR、PPO均保留。
- 本地最终相关131项通过（29.58s），远端最终恢复6项通过（3.44s），原速度联合241项（35.31s）保持各自身份；架构206/0与Ruff通过。4090纯Response8192腿侧1280tick微基准0.69589→0.22234s、3.13倍，电机误差上界3.815e-6Nm，其余NET/heldq/dq为0；该结果不包含PhysX/PPO且在旧训练共享时测得，不能承诺整轮倍数。

新job、源/导入checkpoint完整身份、源码摘要、预算、测试及启动证据位于既有`artifacts/cloud_handoff/motion_speed_r15_*`和`motion_response_*_GPU_microbenchmark.json`。已完成一次有界首更核验：u89保存轮28.117秒，u90不保存轮25.157秒（rollout24.172秒、learning0.975秒），较旧41–43秒耗时约减少40%，仅两次启动短窗不是长期均值。2类编译阶段、881D、40/4.8Nm与NET分路实际读回；核验时总106/10000、余9894。`motion_speed_r15_execution_summary.json`记录精简证据，原始snapshot同目录。继续由原owner后台负责，无新增watcher、例行评估或第二PPO。下方r12“运行中”等是先前启动快照。

**最新覆盖（服务器31193，长训已真实开始）：** 唯一远端run为`2026_11_12__motion_contact_grid_cloud10000_r12`，worker PID1087/start_ticks381264793，cwd`/root/gpufree-data/wheel_leg_motion/candidate`，完整argv与启动状态见`artifacts/cloud_handoff/motion_cloud_r12_startup_verified.json`。有界启动核验为RUNNING、6次/1572864样本、failure=null；4090 UUID`GPU-07694748-3d8b-666b-3933-6443416715dd`独占、4096×64，瞬时显存12022MiB不是峰值。无需再次提交；后续由既有owner/worker后台执行，不新增watcher/例行独立评估。

本次10000授权累计已耗22（本地10、r10失败3、r11失败3、当前r12已6），余9978。r12分配9984包含其已6次，任务墙钟null；历史5167保留，r12 historical5183。来源仍选中完整u3000/model5097，回退及未保存更新不退款；旧配置不得再次提交。实际u1 SHA`fc896dbae43fd96c33a7dfbf2a70cfb9dbd216904942cee4ca7eed40246e2f0f`已CPU完整恢复：881 Critic、双Adam、Teacher/bank游标、warmup50不重做；第一轮Actor/Critic各16次Adam更新、Actor相对来源最大改动0.000216007（网络参数单位）。CPU重载不是物理现场恢复，30s自然timeout/能力验收尚未声明通过。

r10/r11的实际故障已定位为rough穿透检测位置错误：lower-leg的最低点Z减去了body原点XY的地形高度。真实故障q/root经USD FK重建原公式−5.227mm（与原记录差8.34e−8m），最低支撑点自身XY下为+39.337mm，该时刻全体支撑点最小间隙+1.240mm。新`simulation/terrain_geometry.py`由实际SDK quaternion函数计算四类primitive的垂直最低支撑点，FullEnv仅rough查询改用此XY，原平地几何与5mm安全阈值不变；它仍是垂直间隙代理，不冒称完整mesh signed distance。故障dump将无界关节限位显式编码为inf字符串，并保留原始异常；没有放宽机械或工程保护。r12完成rollout3840ms已越过原2210ms故障点。

本机188项针对性回归通过（35.22s）、远端3项实际CPU通过（1.78s）、架构204/0、Ruff0.8.4通过；11份r12增量逐SHA与传输包`19061040b8b6f49008e7c480a300fa5ef3d21967863572aec8fecf5a79af7e10`匹配，2432SDK及u3000/881/双Adam远端核验通过。此前200文件/527回归和旧628传输保留各自身份，不冒用为新物理结果。最新job`artifacts/cloud_handoff/cloud_motion_contact_grid_r12.json`，新CPU/故障复现/首更完整恢复证据由`motion_cloud_gpu_ready_handoff.json`绑定。当前运行源码与依赖冻结，不热改。

以下接触DR与H1段落是先前CPU准备阶段的历史快照；其“未传输/未运行”等结论不覆盖上述最新实际启动。

**最新独立增量：轮地接触DR与H1面积（CPU完成，GPU仍未运行）。** 新配方`configs/jobs/2026_11_12__motion_u3000_bank_height020_H1_cloud10000_contact_grid_v2.json`替代未提交的r9方案；沿用同一未消耗10000授权和历史5167，不创建run或预留额度。原200文件/527回归/2954文件包保持原快照身份，本增量单独绑定与验收。

- 轮地柔顺采用原生PhysX material stiffness/damping API；仅combined profile0中50%回合采用，其余含名义锚保持刚性。reset-only独立无状态流19031，strength0不采样、不耗策略RNG；一回合内保持。最终工程范围K=100000–1000000 N/m、C=500–3000 N·s/m，初始强度.5时围绕几何均值使用半宽对数范围，以原DR课程逐步放宽。不是实机辨识范围，不改变PD40/2/.2、电机40/4.8或V4/NET力路由。
- 原DR每reset先核验静/动摩擦与恢复系数，再由接触适配器写入K/C并读回；rigid回合清除上回合残余damping并恢复ordinary restitution。原生setter只作用于两个具名wheel RigidBodyView，再从原ArticulationView交叉读回；不重写非轮材质，标称地面仍刚性。GPU上的实际接触力、solver行为、轮地组合与吞吐尚未验证。
- Critic877→881只增实际左右K/C四列，普通和pre-reset terminal同路径；新分支首层及对应Adam矩扩展零列，原Actor、877列、双Adam其余状态、std、RNG、已完成warmup50不重建。同run恢复不得再次迁移。Actor仍31D，输入1.5/100与输出中心+.15/80分开。
- 四种H1原6×3m通过实例化铺成150×9m，每个rough环境75块；USD碰撞与NPZ三角查询同源，横向仅边缘各.2m拼缝修整，中间物理float32表面保持。已有路面长度冲突CPU修复：3.6m/s×30s×1.2=129.6m，前向实际可用137.78m；不缩短30s门槛，出界仍真实终止/reset。碰撞实例化的GPU显存/启动时间待测。
- 高度统一定义`root_world_z - actual_triangle_z(root_xy)`；垂速为`vz-gradient·vxy`。rough reset整体root只平移实际地面高度，不改q/dq/目标/FIFO；后续仅自然运动，不逐步强制状态。支撑中心只用相对高度命令，不再叠世界地面Z。
- 旧能力保护延续实际来源配方：KL-v4系数.02、三命令训练bank与合格on-policy各.5；旧链实际BC=0，本次不重复叠BC。rough/柔顺/事件/Teacher控制帧不进入旧平地on-policy来源约束，bank保留原身份。

本次183项相关CPU回归通过（35.39s），架构203/0、Ruff0.8.4 check/format通过。正式CPU验证命令、文件和SHA见`artifacts/cloud_handoff/motion_contact_grid_delivery.json`；真实大u3000/881装配、双Adam/std/游标/暖身保存恢复、native Torch/Warp边界、USD实例/高度数学分别核验，不冒称物理通过。远端曾只读确认IsaacSim6.2的native getter/setter均存在；随后SSH31680连接被拒，新增量尚未传输，不能沿用旧远端CPU收据证明已同步。恢复连接后先传同SHA增量、远端CPU复核，再在GPU恢复后按同一10000额度做实际材质/接触/自然30s结束和保存恢复检查；当前无自动启动。

以下r9及原200文件为本增量之前的冻结快照，不是当前contact_grid可直接提交命令。

用户最新选择**Motion独占远端4090**。新增资源配置`cloud_motion10000_r9_exclusive.json` SHA `ce14df839416e301f9d4026cf92c5672d7df62c43c10a5f7b67b20d3843b0db1`，仅resource_mode由shared改exclusive，job/run身份与原10000额度不变。GPU恢复后负责人先核无其他compute任务，不启动Recovery/Spin/Jump，不停止他人任务；此字段本身不是系统级GPU锁。远端CPU完整装配通过，当前不开训。

**当前增量（无卡阶段）：领域接线/CPU验收已完成并冻结；恢复GPU后可经公共CLI提交，当前未开训。** `artifacts/cloud_handoff/motion_cpu_wiring_delivery.json` SHA `64e968a07ac31e41e3f38bc3475eeff72a3b0e93bb059947edd48635310f8a39`绑定200项领域实现/配置/测试。133项联合CPU通过（21.91s），最后diagnostic命令reset后实际读回补丁14项通过（1.67s）；架构200/0，15个新增/修改文件Ruff0.8.4 check/format通过。不是物理能力证书。

采样正式消费者：`simulation.collection.make_environment`→`motion_backend.make_collection_env`→`prepare_session/_build_environment`→`isaac.collection.MotionCollectionEnv`。明确使用runtime.envs/device，Student=0/Teacher=1/initialization=2；实际Handover owner在partial reset前快照，命令/高度/cohort/DR/模式条件与pre-step输入同帧，实际引擎step计费，异常累计保留，不创建PPO或推进课程。脚本I/O贯穿真实多Teacher标签、累计view和小网络fit；物理I/O被替身，真实在线DAgger仍NOT_RUN。诊断当前仅877单环境，严格核完整V4、实际时序/电机cap、场景/初态声明；实际目标命令先修改，再读回可能被reset重新采样的DR。不静默改摩擦或时序；公共sim2sim仍可拒rough与worker_atexit，未作真实跨引擎等价声明。

实际修复两处可搬迁问题：`numeric_bridge/fixed_recovery`只提供冻结DR所需9个family、4档幅度与863列schema，不递归加载旧实验课程；课程仍唯一消费CourseStrategy。`motion_commands`按实际decoder高度范围接命令，0.20不是伪造的reset姿态，而是从合法闭链初态按原时序下降；`maneuver_goal_override`在partial reset后不被旧表覆盖。普通/terminal Critic877、input1.5/100、output中心+.15/80、previous采用目标、40/4.8Nm与NET分路均保留。

远端数据已同步，`motion_remote_cpu_readiness.json`验证实际u3000完整来源、双Adam、固定std、冻结KL Teacher/bank游标、warmup50不重复、2432 SDK文件SHA；10000正式dry-run通过。r8 FAILED/0完整更新保留；r9被无卡GPU前检拒绝、没有run/worker/新增额度消耗。当前远端活动worker0。恢复GPU后先核实际UUID/CUDA/显存/磁盘，若GPU身份改变明确更新资源绑定，不能跳过校验。只提交一次：

```bash
cd /root/gpufree-data/wheel_leg_motion/candidate
source /root/gpufree-data/wheel_leg_motion/cloud_shell_env.sh
/opt/isaaclab-venv/bin/python -m native_stand train \
  --config configs/jobs/cloud_motion10000_r9_exclusive.json \
  --run-id 2026_11_12__motion_u3000_bank_height020_H1_cloud10000_r9 \
  --runs-root /root/gpufree-data/wheel_leg_motion/runs --wait-seconds 120 --json
```

此命令会实际提交，仅恢复GPU且负责人执行时使用；没有watcher/自动启动。CPU准备脚本为`scripts/check_motion_cloud_readiness.py --config CONFIG --output RECEIPT`，不会创建run、预留预算或积分。首个真实远端更新、自然30秒timeout和远端完整保存恢复仍待实际验证；不恢复本地余8次。新独立10000与历史5167分别记账。

并行资源建议见`motion_gpu_concurrency.json`：单4090先跑Motion4096×64。观测下界0.887GiB、单minibatch前向激活下界1.188GiB不能当峰值；实际PhysX/反向/历史/CUDA未测。Motion/Recovery各要求14000MiB空闲，24GiB不静态承诺两条满配并行，不绕过保护。Jump当前是MuJoCo CPU ABI4，CPU/RAM会与Isaac竞争。真实并行吞吐、GPU峰值均NOT_RUN，不给Recovery/Jump新增授权。

最新实际：r6 `runs/2026_11_12__motion_u3000_bank_height020_roughH1_smoke37_r6` 已正常`STOPPED_BOUNDED`、failure=null、29次/237568样本，完整u29 SHA `bf701764a9acee5553d5c858f19de35719a4038477f52beddad728096c22cc39`。用户要求尽快云训，取消本地余8次恢复；不能声称真实完整恢复或30秒自然timeout已通过。实际发生21个任务终止/partial reset，0 timeout。正常结束21回合均计入，41cohort有真实样本；这是接线证据，不是能力认证。r4已耗27、有效保存只u16，11次未保存更新不退款；本轮合计56，原实际累计5167。r1/r2/r3/r5失败保留；r2因SDK快速退出漏终态经公共owner.reconcile补结算；其他失败均准确记录。r3物理tick10，无完整PPO；物理消耗不冒充0。

用户随后明确授权**独立Motion10000更新/不限时**，本地配置`configs/jobs/2026_11_12__motion_u3000_bank_height020_H1_cloud10000.json`记录原5167不退款、旧未用4833不叠加；Spin未获新额度。远端数据盘`/root/gpufree-data/wheel_leg_motion`，GPU4090 UUID `GPU-d8240549-897e-46d7-04c8-86d26a927783`，4096环境×64步。路径经RuntimePaths映射，源u3000完整状态和Teacher/bank同SHA；低高度从0.30逐级到0.20、H1速度从0.15逐级提升，直线±3.6锚和纯yaw±2复习保留。

远端启动兼容处理：现成uv环境的RSL-RL/tensordict固定为本地验证5.0.1/0.14.0，Torch仍平台2.11；镜像isaaclab当前API缺clone_environments，携带本地验证SDK代码/扩展为明确依赖（2432文件/18.2MB），保留IsaacSim6.2和PhysX。AppLauncher需ISAAC_PATH及正确libpython；不改固件、物理、PD或源资产。远端各0更新失败独立留档，当前r7已收到worker启动握手（PID2883），真实首更结果待记录，不把RUNNING握手称完成更新。没有第二PPO、没有新Recovery或独立评估。

已修实际首步错误：Task嵌套旧中心表与新范围不一致；height内层在fullchain追加前错误读取transfer奖励；AppLauncher快速退出跳过结算。当前模式reward统一高度/垂速，关闭前持久化异常，平台忽略fast_shutdown时延后到worker结算后关闭。有限路面出界是普通任务终止→合法起点reset，不用虚构平面续算；失效高度明确mask，不授timeout或整窗成功。对应CPU回归67项，后续实际full-source/生命周期45项通过；未完成的在线DAgger工厂另由公共集成收尾。

`configs/jobs/2026_11_12__motion_u3000_bank_height020_roughH1_smoke64.json`经现有CLI→owner→worker→MotionProductionAdapter；公共CLI仅透传既有worker的`--max-updates`有界参数。`selected_full_bounded_branch_v2`绑定单一共享分配计划及run_id/源/输出/执行SHA；分配64 smoke、formal0，共同余额未分配4825。原0额度prepared和v1未分配合同继续拒绝真实提交。计划56次安全保存后同run恢复8次，不刷新额度、不重复warmup。

`height_extension`只在明确新分支边界扩展decoder，保留完整训练状态；新中心绑定关联u3000、新输出合同与静态几何证据。新表0.20–0.30由完整USD闭链FK解构造并延续0.30处支撑目标偏移；0.30–0.40同raw/高度目标差≤1e-6。几何闭合不是实际承重辨识，尚无0.20m实机能力。命令v3以0.02m/s先调整高度，再爬升运动命令；评分等待高度与运动过渡结束。

`terrain_scene`向原环境工厂注入四种冻结H1网格，移除对应rough环境的重复平面、建立独立碰撞组；原V4/材质/电机与NET分路保持。`TriangleRoads`按同NPZ三角面、旋转和重复tile查询实际地面；冻结奖励/值函数经显式numeric_bridge消费同一height API，平面算法不变。rough的高度/垂速相对当地网格；几何预警采用body中心地面修正，不能称精确凸体-网格距离。rough失去路面范围即明确故障，不悄悄退回平面。对称仅平地非转弯，压弯保留合理roll与腿长差。H1是静态起伏网格，不是独立可动碎石；新接触柔顺DR未实现。

新增实现：`tasks/motion/{height_extension,maneuver_program,split_course,maneuver_targets,maneuver_reward,maneuver_statistics,branch_state,action_contract,spec}`，`simulation/{terrain_surface,ground_surface}`，`simulation/isaac/{terrain_scene,numeric_bridge/ground_height,legacy_support,fullchain,environment,maneuver_runtime,maneuver_planned,motion_backend}`，`artifacts/branch_initialization`及CLI/owner参数透传。普通/terminal保持877；修正正式启动读回对完整877错误调用旧863读取器的问题，改为实际863因果前缀与877计划尾列分别核验。Source KL Teacher仍使用自己的原合同。

针对性CPU：`PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/kellen/anaconda3/envs/issaclab/bin/python -m pytest -q tests/test_motion_height_rough_smoke.py tests/test_current_recipe_migration.py tests/test_motion_production_integration.py --basetemp=/dev/shm/motion_height_rough_smoke_cpu`，62 passed/11.80s；架构194/0。该结果包含真实u3000完整加载、严格新中心绑定、额度负例、0.20命令、压弯符号、同源地形SHA与CPU三角查询，尚不等价PhysX/GPU通过。实际启动、消耗和读回由run结果记录。

### 来源与完整分支初始化

权威停止与选择为生产仓库`artifacts/2026_11_12__motion_spin_motion_start_selection`。选中`model_update_3000.pt` SHA `abe10db68b0ad62a3e0ceac5285ebb0041ff084134bc313edb235ae46d427d7e`；最终`final_model_update_3014.pt` SHA `8f6cbc98d0c4934eb9474de8dd0282929726716e484cd6ecf0e2ccfdbd3650c9`。旧链正常停止、无故障。**选中模型进度5097 / 原授权实际消耗5111 / 共同余额4889 / 新局部0**；两分支不能分别领取4889，回退不退款。

两个`configs/branches/*_u3000_initialization.json`引用原selection/settlement，`configs/jobs/{motion,spin}48_u3000_prepared.json`均`full`且0更新。来源完整Actor、877D Critic、两Adam、真实log_std、训练RNG、冻结Teacher、bank游标与warmup50保留；课程、确认、统计、物理现场及候选域重建。旧课程/榜单仅为来源历史。公共同run恢复和最终停止点导入规则没有放宽。

当前正式链：CLI→dispatch→`MotionProductionAdapter`→`_resolve_source`的显式branch准备→`CourseStrategy`/`SplitCourse`→同一环境工厂→`supported.assemble_from_full`/真实CommandBankAnchor→capture/save→首次候选发布→权威引用。无第二trainer、PPO循环或Spin播放器。

### 两份配方

| 项目 | Motion | Spin |
|---|---|---|
| 主任务 | 四种符号组合的vx+yaw行进转弯、适当内倾；高度变化完成后再运动 | 双向纯自转，实际有限端点逐级增长 |
| 采样 | 站立/受扰30%，固定复习锚30%，新目标40% | 站立/受扰40%，固定Spin锚20%，新端点40% |
| 起点 | 行进转弯2m/s、0.5rad/s重新确认；高度先0.32/0.36 | ±9.313225746154785rad/s是学习起点，不继承旧通过证明 |
| 范围 | 直线±3.6m/s，普通纯自转不超过±2rad/s；高度0.30–0.40m按现有中心表 | growth1.15，`spin_terminal_cap=null`；显式有限cap可限制后续端点 |
| 对称/姿态 | 平地站立、直行保持对称，转弯允许合理内倾与双腿差异 | 不强迫双腿对称，保留姿态、双轮承重及恢复 |
| 旧能力保护 | KL-v4/0.02、固定Teacher及3命令train-only bank；资格mask排除DR、事件、Teacher帧和不匹配命令 | 同Motion；不给新Spin或转弯回灌旧标签 |

原地交接由用户控制；课程前控比例0，不擅增跨技能接管。此表两份原prepared仍为0.30–0.40平地完整V4及原DR；新0.20m/H1和轮地柔顺DR见上节独立增量，CPU接线不宣称已掌握或PhysX响应已通过。

DR从level1/amplitude0.5重新确认，依次可至0.75/1.0。配置显式`shared_strength_independent_of_task_axes`：dq及各family共用强度，DR晋级独立于任务方向；不是每个family各自升降。各任务方向2次合格确认晋级、3次明确退化仅回退自身一级；窗口至少16个已结束回合、完整比例0.9、任务通过0.7，退化阈值0.35。UNKNOWN中断连续确认，不制造失败或回退。

### 已关闭的领域缺陷

- `split_course`此前将DR1确认带入DR2。现以实际目标/高度/DR分布/执行合同建立确认身份，推进后只清受影响链；不把全局revision当身份。成绩历史与档位保留，重复窗口不再消费，状态保存后下一窗口一致。
- `spin_terminal_cap=10`此前仍产生10.7102目标。现可达精确±10，重复通过不再产生无意义档位或revision；null逐步探索，每个实际目标仍须float32有限。增长使用先除后乘与端点历史，消除幂运算先溢出的问题。机体yaw目标与轮±80rad/s是两个参数。
- `split_candidates`分别记录物理候选资格与训练任务成绩；样本不足为UNKNOWN。低任务通过率候选列出具体未通过目标，不授能力认证；Standing与主任务独立，不跨Motion/Spin比较域或DR池混排。发布顺序为完整保存→SHA→公共引用→视图；含pending的checkpoint恢复可补发布，不重复晋级。
- KL搬迁通过公共语义身份/RuntimePaths：同内容搬迁可恢复，错SHA/损失/合同仍拒绝。历史`std`字段兼有sigma/log_std；Motion装配核对Actor实际分布，新capture显式`std_parameterization=log_std`，未改变探索大小。

### CPU证据和命令

最新领域针对性：`tests/test_motion_split_progression.py tests/test_motion_spin_branches.py tests/test_motion_spin_host.py` **57 passed in 10.22s**。最终联合核心/领域/执行数值回归 **311 passed in 66.54s**（0失败/skip/xfail）；架构检查188个正式文件/0例外；固定Ruff0.8.4对本领域13个实现/测试文件check及format check通过。真实u3000只做CPU完整装配/临时保存恢复，CUDA RNG保留原字节、未在CUDA激活。正式首次候选发布使用受管小网络和已结束回合统计I/O fixture；课程、decoder、KL、capture/save/load和候选/引用消费者均为真实实现，不把临时fixture计数称为新增真实PPO。

```bash
cd /home/kellen/wheel_leg_refactor_ws
PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python -m pytest -q \
  tests/test_motion_split_progression.py tests/test_motion_spin_branches.py tests/test_motion_spin_host.py
PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python -m native_stand train \
  --config configs/jobs/motion48_u3000_prepared.json --dry-run --json
PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python -m native_stand train \
  --config configs/jobs/spin48_u3000_prepared.json --dry-run --json
```

两份计划只解析、不写run、不预留、不启动；正式提交在公共preflight分支明确拒绝未分配余额。最终Motion/Spin配置SHA分别为`223b83d94403f4ad694e2a836964174e63c021add3780905b6fdad22a9ba43f6` / `24f467326e485f93c3148f2b2103312b9e0e8958817200a274a4fa88bf6b3328`。41个原归档文件逐SHA核验无变化；新仓库训练worker为0，原Recovery PID1901402/start136388404未操作。

### Motion／Spin速度配置与实际消费者

两prepared job实际消费的SHA绑定runtime template继承performance v2：`motion_hotpath`、`motion_runtime_reduction`、`combined_step_checks`、`fixed_shape_statistics`、`contact_history_disabled`均true。`legacy_spec_adapter`保留performance→正式`environment`写入cfg→冻结V4环境按physics_step/reset_epoch缓存状态和几何、批量collider计算。缓存失效及partial reset已有CPU数值测试；不删除接触body、安全检查或真实时序队列。物理数值实现仍是固定版本依赖，生命周期/配置/课程/PPO装配使用新仓库范式。

当前headless、2560env×64步（163840样本/update）、4epochs×4minibatches；不做例行独立评估，训练统计每100次，trace关闭、checkpoint每100次、指标每10次。已完成warmup50不重复。独立评估相关buffer/group开关虽然来自template，不等于训练中运行独立评估。

本轮另在`MotionProductionAdapter.update_once`减少未消费诊断：辅助`MotionStatistics`与`value_diagnostics`此前每update计算、仅每100报告使用，现仅报告更新计算；相应age/index临时张量也只在需要时创建。正常完整回合统计、奖励/终止/课程/GAE/控制归属/安全不变。`restoration.runtime.performance`新增实际cfg/contact history读回；不能只以模板字段宣称runtime生效。

`tests/test_motion_speed_contract.py`以真实PPO数学和物理I/O替身，对照report/nonreport：原始actions/log-prob、reward/returns、advantages/values、Actor/Critic、两Adam、Torch RNG逐值相同，辅助诊断只在报告轮调用。`test_motion_spin_host`穿过真实u3000→实际spec构建核对开关。定向37项通过（7.49s），最终相关核心/领域/性能回归320项通过（66.99s，0失败）；架构188文件/0例外，固定Ruff0.8.4的3个增量文件check/format通过。真实引擎/GPU计时尚未运行，结构性减负不等于已经测得加速倍数。

**后续rollout优化（2026-11-12，CPU已验证）：**

| 实际消费者及修改文件 | 减少的工作与保留的不变量 |
|---|---|
| `tasks/motion/{encoding,spec,action_contract}.py` → `simulation/isaac/handover.py::Handover.actions` | 公开decoder默认仍立即严格验证；正式路由将Student/Teacher高度、运动学和动作有限值检查交给同一`StepChecks`，在控制目标、owner、计数和PD路径写入前统一验证。未删任何数值检查，Student/Teacher各自合同、输入1.5/100与输出.15/80不变。实际CPU scalar-read计数Student5→1、含Teacher10→1；错误值在采用前拒绝。 |
| `simulation/isaac/sensor_packets.py::Packets.read` | 复用已有Python reset脏标记；稳定采样不再对四槽历史做整块`where/copy`，仅初始化/partial reset暖填。取样/噪声/dq处理调用、延迟年龄、出生时刻、同tick reset、环形回绕和其他环境历史逐值不变。 |
| `simulation/isaac/{environment,fullchain,maneuver_planned}.py` | 工厂显式指定877外层拥有最终terminal快照；不再构造被外层覆盖的FullV4中间快照。外层仅在实际结束时构造完整pre-reset877，包含自然终止和timeout；无结束时该字段为None，普通后继观测照常构造。冻结基类的原timeout快照仍保留，不宣称所有timeout只调用一次。普通/终态字段、最终奖励记忆、reset前克隆和timeout bootstrap不变。 |
| `simulation/isaac/motion_backend.py::update_once` | 仅显式开启`diagnostics.performance_measurement`时输出`rollout_seconds/returns_seconds/diagnostics_seconds/learning_seconds`；CUDA同步在阶段边界，没有逐物理或策略步增加同步。默认计时和训练数据流不变。 |

参考了本机华南虎`/home/kellen/wheeled-legged_RL/source/agent_rl/agent_rl/rsl_rl/runners/on_policy_runner_him.py`及复旦`/home/kellen/fudan_train/plane/wheel_legged_gym/learning/runners/on_policy_runner.py`的整批Tensor rollout、inference_mode和采集/优化分段计时。两仓库也存在逐步`.cpu().numpy().tolist()`日志，未照搬；其物理步长、网络、观测和控制参数不作为当前配方替换项。本轮未改参考仓库或旧生产/冻结依赖。

增量测试为`tests/test_motion_rollout_hotpath.py`及`tests/test_motion_speed_contract.py`：真实decoder/路由、公开严格校验、错误目标采用前拒绝、冷热环形历史、同tick和partial reset、完整877自然/timeout/mixed终态、bootstrap、RNG与真实小模型PPO/双Adam对照。物理I/O是测试替身；没有Isaac、MuJoCo积分或真实训练。相关核心/领域/性能**368 passed in72.35s**，0失败；架构188文件/0例外；Ruff0.8.4 check/format覆盖本轮11个Python文件。验证命令为前一轮320项集合追加`test_motion_rollout_hotpath.py/test_motion_control_contract.py/test_motion_kinematic_output.py/test_motion_host_consumers.py`。

有界CPU组件对照使用改动前源码快照`/tmp/motion_rollout_cpu_dujnneej`，2560行、Torch单线程、预热10次、5轮×100次；结果`benchmark.json`：传感包中位0.704052→0.552922ms（约21.5%减时），控制路由0.424486→0.420659ms（波动范围重叠，无明显CPU收益）。该测试不推进物理、不优化策略，不能外推GPU速度或整轮PPO秒数。GPU/PhysX仍NOT_RUN；后续有界smoke可用上述显式分段计时定位剩余开销。两prepared job、u3000、最终3014与停止结算身份未改变，共同4889未分配。

### 领域修改与公共集成边界

- 配置：两份prepared job、`configs/branches/{motion,spin}_u3000_initialization.json`。
- 领域：`tasks/motion/{split_course,split_candidates,branch_state,course_strategy,action_contract,motion_candidates}.py`，`simulation/isaac/{motion_backend,environment}.py`领域分派，`learning/ppo/supported.py`的Motion完整状态/损失装配。
- 测试：上述3个独立测试、`fixtures/motion_spin_branch_probe.py`；旧u600用例改为受管小fixture，真实归档验证单列HOST，不为测试永久保留已轮换checkpoint。
- 公共身份、生命周期、ModelIndex保护、通用推理/蒸馏/播放/部署和云端交付由公共负责人维护。领域复用`ActionDecoder.decode`和`physical_to_mean`，后者拒绝不可表达的Teacher物理标签，不复制专用蒸馏器。
- 公共依赖清单需要沿branch→selection/settlement→u3000/最终3014/中心合同递归，去重两分支共同依赖；不要携带整个历史实验树。预算分配、远端路径/GPU绑定、真实Isaac工程smoke尚未执行。

以下11-11及更早内容是历史记录，不作为本次Motion/Spin起点或活动身份。

> **以下为2026-11-11阶段摘要，当前Recovery以本页上方2026-11-12增量为准。** 这是候选工程交付，尚未完成云端切换；下方旧Jump/阶段记录均为历史。本轮真实物理积分、PPO/监督新增、GPU/GUI/ARM执行均为0。

## 当前交付与消费者

| 链 | 当前结果 | 正式消费者与证据 | 未完成边界 |
|---|---|---|---|
| Motion | 当前配方CPU链部分完成 | `motion48_supported_v1`→CLI/dispatch/owner→MotionProductionAdapter；真实2097和u600来源装配双Adam和CommandBankAnchor，v1/stage19迁v2/stage20/75组；u600保持stage23/47组。真实adapter完成after_update→capture/save→after_save首次发布，再经新目录import/load/adapter恢复窗口和bank游标；无额外采样。 | 真实u600的独立CPU fixture也走adapter→保存→新目录恢复（保留stage23/47组）；该fixture不代表活动run交接。真实Isaac环境构造/自然终点/实际优化与6.2API均NOT_RUN。 |
| Motion执行/奖励 | CPU消费者通过 | 显式factory接当前maneuver v2、mode reward和计划命令。875因果列＋真实计划端点构成877；普通和reset前terminal走同一编码。40/4.8进入构造缓冲、DR reset、PD/driver、归一化和Critic读回；原5.5不再覆盖当前配方。1/5/10ms、20/5ms、轮7/11ms/tau0不变。 | GPU张量写入、闭链/接触、真实NET/电机分路仍需有界工程smoke。 |
| Motion KL v4 | CPU loss与保存恢复通过 | 固定Teacher＋NPZ/metadata路径/SHA校验；三组均衡bank，无on-policy命令时仍覆盖三组；0.02进入Actor loss。Teacher冻结，辅助loss不消耗Torch RNG或改Gaussian缓存；游标捕获/恢复。名义/事件/姿态/支撑/高度mask接实际输入。 | CUDA RNG行为仅保留状态，未在CUDA执行。 |
| Recovery | 当前factory/奖励/执行CPU部分完成 | 31D PhysicalActor/909D Critic，当前Actor-only实载、两Adam新建；全池31,040行height×dq BC batch512/0.1在Actor minibatch调用，旧prior0.01/max64独立，质量UNKNOWN。v8实际Lifecycle、Swing own-hold v2、4.8Nm注入环境/执行/奖励。真实完整来源双Adam装配、capture/save/import新目录/restore通过；已完成5次warmup不重复。 | 当前909D每列真实环境/terminal的全消费验收、完整公共worker脚本I/O训练闭环和legacy生产身份转换仍未全部完成。不能宣布已可上云稳定训练。 |
| 课程与候选 | 正常训练统计路径 | Motion UNKNOWN中断确认链而不丢档位；key迁移、独立方向和DR轴保留。RecoveryAdapter只读正常已结束回合，按游标累计，每100更新导出；早失败进分母，Teacher时间不冒充Student，DR/高度条件分池，Swing无成功为空，standing不回填Swing。保存→SHA→项目索引原子候选→视图→轮换。旧未定域统计保留UNKNOWN历史。 | Recovery当前Swing v2保持其来源实现的固定曝光/DR语义，未擅造自动升级规则。真实课程revision/环境重建NOT_RUN。 |
| 恢复与云端 | 公共显式导入/准备包 | 逻辑run ID与物理目录分开，manifest绑定原SHA、合同、依赖、预算和父链；错误SHA/跨run/合同/预算拒绝。新位置完整保存再读回，旧checkpoint不改。legacy格式另验run记录/代码身份和终态结算；活动滞后点禁止导入。`RuntimePaths`在加载边界显式映射，不改冻结JSON。`resume --max-updates N`按新增已完成更新有界退出，不重置额度。 | 两个真实活动owner未终止，故没有合法最终handoff；legacy真实终态导入NOT_RUN。路径映射仍需确认远端持久盘后填写；不存在“本地CPU通过＝Isaac6.2/3.0通过”。 |
| Jump | ABI4 MuJoCo CPU保持 | 只补公共正额度提交preflight，编译/引擎/物理均不由preflight启动。 | Isaac GPU backend未实现；不阻塞Motion/Recovery。 |

## 验证与证据

- 最后核心回归：**314 passed in 143.49s (0:02:23)**，0失败/跳过/xfail。后续增量针对性测试最后记录见`artifacts/cloud_handoff/current_recipe_tests.txt`；包含实际v8十秒Student保持、Teacher不成功与23秒有限timeout，均为纯CPU状态计算。
- `scripts/check_architecture.py`：174个正式文件、0例外。固定Ruff0.8.4：CI范围check通过，format check173文件通过；未安装或修改活动Conda环境。
- 本机来源依赖：2097/u600课程准备、Motion实际adapter首次发布/跨目录恢复、Recovery当前Actor/完整来源factory恢复、877冻结数值消费者均为HOST验收。其余小模型/窗口/删除竞态/身份拒绝等为portable或/tmp fixture；不是审计“错误复现通过”。
- 真实生产文件只读；实际逐文件删除仅沿用/tmp fixture核心测试。新增引用在现有ModelIndex按独立消费者登记，候选淘汰不解除Teacher/BC/prior/pin等引用。旧活动worker没有迁移到新索引，未对它们执行新清理器。
- CPU CLI正额度验收穿过真实配置/dispatch/owner preflight，在进程spawn这一I/O边界主动注入失败，确认无引擎启动；不把测试中的spawn失败称作真实训练成功。owner EOF/退出竞态修复后原失败回归通过。

```bash
cd /home/kellen/wheel_leg_refactor_ws
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/kellen/anaconda3/envs/issaclab/bin/python -B -m pytest -q -rf -p no:cacheprovider \
  tests/test_new_cli.py tests/test_new_config.py tests/test_new_artifacts.py \
  tests/test_new_encoding.py tests/test_new_lifecycle.py tests/test_new_learning.py \
  tests/test_phase2_fixes.py tests/test_phase3_fixes.py tests/test_architecture_rules.py \
  tests/test_standing_survival_training.py tests/test_motion_handover_visual.py \
  tests/test_recovery_integration.py tests/test_motion_control_contract.py \
  tests/test_motion_course_resume.py tests/test_motion_production_integration.py \
  tests/test_motion_transient_statistics.py tests/test_motion_value_diagnostics.py \
  tests/test_shared_control_core.py tests/test_jump_integration.py \
  tests/test_unified_playback.py tests/test_motion_host_consumers.py \
  tests/test_motion_training_windows.py tests/test_jump_unified.py \
  tests/test_jump_unified_host.py tests/test_current_recipe_migration.py
```

两当前配方均执行`python -B -m native_stand.cli train --config configs/jobs/NAME.json --dry-run --json`；结果在`artifacts/cloud_handoff/{motion,recovery}_dry_run.json`。没有创建生产run。

## 活动身份、模型与预算

开始与结束只读发现完整argv/start_ticks/cwd/实际source_root、run/state/budget，快照为`artifacts/cloud_handoff/activity_{start,end}.json`。下表是结束核对时刻，非持续轮询；两worker身份与开始一致。

| 任务 | PID / start_ticks | 状态 | 本run已用 / 剩余 | 含祖先累计 / 原额度 |
|---|---|---|---|---|
| recovery | 1901402 / 136388404 | RUNNING | 314 / 8020 | 1980 / 10000 |
| motion | 1905160 / 136466506 | RUNNING | 780 / 7123 | 2877 / 10000 |

Motion保存点u700落后已用780，Recovery u300落后314；不能从旧点退款、重跑或并行消耗预算。最终切换必须等原owner在完整checkpoint边界结算。Recovery生产`best_candidate`当前实际来自standing证据，保留原历史标记，**不解释为Swing成功候选**。

| 消费角色 | 完整路径 | SHA256 |
|---|---|---|
| recovery/latest_model | `/home/kellen/wheel_leg_native_train/runs/2026_11_11__recovery_preentry48_remaining/checkpoints/model_update_300.pt` | `f21388f4cc2d5ce1b26e2ddbb13791256c51d9e664b6a689baf0043cc0eaf98e` |
| recovery/best_candidate | `/home/kellen/wheel_leg_native_train/runs/2026_11_11__recovery_preentry48_remaining/checkpoints/model_update_300.pt` | `f21388f4cc2d5ce1b26e2ddbb13791256c51d9e664b6a689baf0043cc0eaf98e` |
| recovery/best_standing_candidate | `/home/kellen/wheel_leg_native_train/runs/2026_11_11__recovery_preentry48_remaining/checkpoints/model_update_300.pt` | `f21388f4cc2d5ce1b26e2ddbb13791256c51d9e664b6a689baf0043cc0eaf98e` |
| motion/latest_model | `/home/kellen/wheel_leg_native_train/runs/2026_11_11-09_59__motion2097_turn_response_independentDR_full7903/checkpoints/model_update_700.pt` | `25db0cc03cbda4cab292ec1b7dbc08d18b3d46a089bdb4bf51d55232ca20ad77` |
| motion/best_candidate | `/home/kellen/wheel_leg_native_train/runs/2026_11_11-09_59__motion2097_turn_response_independentDR_full7903/checkpoints/model_update_700.pt` | `25db0cc03cbda4cab292ec1b7dbc08d18b3d46a089bdb4bf51d55232ca20ad77` |
| motion/best_standing_candidate | `/home/kellen/wheel_leg_native_train/runs/2026_11_11-09_59__motion2097_turn_response_independentDR_full7903/checkpoints/model_update_500.pt` | `2c62558818582cc55c13b6df2ff4545937a53a8940ebb905d8ad3e34abb50121` |
| motion/initialization | `/home/kellen/wheel_leg_native_train/runs/2026_11_10-21_57__Authorized_command-normalized_source_KL_plus_semantic877_value_warmstart;_independent10000,_wallnull/checkpoints/model_update_2097.pt` | `f729ab9b231ab92bd4a35d6b89e28978d9dd8b2985fcc644ad457e5103d2ff14` |
| motion/source_anchor_identity | `/home/kellen/wheel_leg_native_train/artifacts/2026_11_10__motion48_height_banking_training10000/source4300_motion48_actor_only.pt` | `642d9a841ee3e796b545dfb166c961c160d063c832500b5ed5da4e410842dbe4` |
| motion/bank | `/home/kellen/wheel_leg_native_train/artifacts/2026_11_10__motion48_anchored_recovery10000/source_comparison/source4300/train_anchors.npz` | `f2d2f68671ebec94477814c3f12c709ccc45da23ba52420c61054e78210cf7d6` |
| motion/bank_metadata | `/home/kellen/wheel_leg_native_train/artifacts/2026_11_10__motion48_anchored_recovery10000/source_comparison/source4300/train_anchors.json` | `63396eb099c6e15baa9f546fb8f0810fa470a4e46e963847ad98648ea6f2eaf4` |
| recovery/initialization | `/home/kellen/wheel_leg_native_train/artifacts/2026_11_11__recovery_preentry48_restart/initial_actor.pt` | `9bcdb05e46fdadaaa1957659afa295c41f3cba941687c7f8a69b9cbc1fd7b633` |
| recovery/qualified_physical_BC_anchor | `/home/kellen/wheel_leg_native_train/runs/2026_11_10__recovery_standalone_prior_finalize/qualified_physical_BC_anchor.npz` | `692c4ee40fef43549a1f96e41543f4abd5287e71f09ca5546adc079760722f13` |
| recovery/old_Recovery_action_prior | `/home/kellen/wheel_leg_native_train/artifacts/2026_11_09__recovery_Swing_handover_restart10000/preserved_models/old_Recovery1500_action_prior.pt` | `dc7fad5a2245639fe864f5d7c4e0cd277fb5c9e175d453b545e3519fbc1a9989` |
| recovery/recovery_training_contract | `/home/kellen/wheel_leg_native_train/artifacts/2026_11_11__recovery_preentry48_restart/output_contract.json` | `71d2b06b6617b9499f056e9951b2ca7c3a3ee8b2d100169a4e57f09ba21575db` |
| recovery/recovery_isaac_runtime | `/home/kellen/wheel_leg_native_train/artifacts/2026_11_11__recovery_preentry48_restart/runtime_contract.json` | `81d6c19ec5e64189fb5609310f7e2ec0d55fc0cbc76e7594ee4a1a51c819a25b` |
| user_fixed/u1600 | `/home/kellen/wheel_leg_native_train/artifacts/2026_11_09__motion_yaw1to4_u1600_preserved/model_update_1600.pt` | `09f27b74507b55da1a7cc0c81e2b77ed7b25a0d43bfe7a74a1806d2252915ef5` |

其他前控/C0、完整V4元组、轮辨识、合法初态、USD/层、外部Python/YAML等完整路径/SHA见`artifacts/cloud_handoff/dependencies.json`，不把历史出处全部变成永久消费者。

## 云端准备及尚缺信息

准备包`artifacts/cloud_handoff/motion_recovery_prepare.tar`包含候选源码/本轮dirty状态、有效job、必要冻结数值源码、checkpoint/Teacher/bank/BC/prior、资产闭包和manifest；不含全部runs、历史实验账本、Conda环境或未消费大轨迹。包是**准备包，ready_for_training=false**；活动run快照不是终态授权。USD层已用UsdUtils只读解析；`OmniPBR.mdl`由目标镜像MDL resolver提供，目标解析NOT_RUN。

生成入口：

```bash
wheel-leg cloud-plan --config configs/jobs/motion48_turn_response_DR_full_resume.json \
  --config configs/jobs/recovery_preentry48_remaining.json \
  --run /home/kellen/wheel_leg_native_train/runs/2026_11_11-09_59__motion2097_turn_response_independentDR_full7903 \
  --run /home/kellen/wheel_leg_native_train/runs/2026_11_11__recovery_preentry48_remaining \
  --manifest artifacts/cloud_handoff/dependencies.json \
  --archive artifacts/cloud_handoff/motion_recovery_prepare.tar
```

远端首先确认SSH端点/密钥、持久盘挂载/空间、IsaacSim6.2.0/IsaacLab3.0.0镜像身份与4090 UUID/映射。解包到持久盘；填写`runtime_path_mapping_v1`绝对路径映射及SHA，源码/输出/checkpoint/日志/cache都放持久盘，保留镜像LD_LIBRARY_PATH和uv已有环境。使用本项目`wheel-leg train --dry-run`，不使用官方demo替代。静态dry-run不会验证App/Simulation API、PhysX、Torch/CUDA、RSL-RL、Tensor写入、闭链/接触；这些均NOT_RUN，禁止据此启动长训或切换本机链。

原owner未停止前不生成可执行legacy handoff。远端dry-run和持久盘通过后才做单链有界交接；`wheel-leg import-checkpoint --manifest FILE --mapping FILE --destination PERSIST_RUN/checkpoints/import.pt --run-id LOGICAL_ID`只导入不启动。未来已结算的Motion用`wheel-leg resume --run PERSIST_RUN --checkpoint FULL --max-updates 52`；Recovery当前配方工程窗口为12次，用自身合同而非套用52。二者均消耗原剩余额度，失败停止继续消耗，不作能力认证。当前全部这些远端命令**未执行**；候选`wheel-leg stop --run RUN --request-only`仅写owner停止请求、不发送OS信号，已由/tmp fixture验收；本轮未对任何活动run调用。

## 本轮准确变更文件

- 配方和公共入口：`configs/jobs/motion48_turn_response_DR_full_resume.json`、`configs/jobs/recovery_preentry48_remaining.json`；`src/native_stand/cli.py`、`experiments/{config,owner,worker}.py`、`simulation/engine.py`。
- Motion课程/行为：`tasks/motion/{course_strategy,maneuver_course,maneuver_course_v2,maneuver_statistics,maneuver_targets,maneuver_reward,maneuver_retention_reward,maneuver_candidates,maneuver_program,planned_value,execution_limits,delayed_execution,motion_candidates,legacy_spec_adapter}.py`。
- Motion实际环境：`simulation/isaac/{motion_backend,environment,execution_buffers,fullchain,motion48_execution,maneuver_runtime,maneuver_planned}.py`。
- Loss/factory：`learning/ppo/{supported,command_bank,nominal_bank,recovery_cubic,recovery_action_prior}.py`。已有`physical_anchor.py`保持其全池采样算法，本轮接入正式factory。
- Recovery：`tasks/recovery/{swing_priority_course,legacy_reward}.py`、`simulation/recovery_isaac/{backend,environment,execution,lifecycle,legacy_lifecycle,preentry_lifecycle}.py`、`simulation/recovery_adapter.py`。
- 身份/保护/打包：`artifacts/{handoff,runtime_paths,model_refs}.py`、`integrations/cloud_bundle.py`；Jump仅`simulation/jump_backend.py`增加owner所需的无积分preflight。
- 测试：`tests/test_current_recipe_migration.py`、`tests/fixtures/{motion48_training_probe,motion48_critic_probe}.py`。交付记录仅本报告首页、`STATUS.md`和`artifacts/cloud_handoff/`机器可读证据；未改历史章节或生产文件。

## 历史：此前ABI4 Jump交付（保留原证据，不作为当前云端结论）

## 历史Jump交付

**MuJoCo CPU工程接入完成；真实执行验证未运行。** 当前正式链为`wheel-leg → JobConfig → registry/dispatch → owner/worker → driver → JumpAdapter(GenericPPOAdapter) → JumpBatch/UnifiedLearningEnv → ABI4 C/V4`。不复制PPO/rollout循环，无FunctionType、源码字符串替换、AST或exec接线。仅ABI4/MuJoCo组合开放，Isaac明确拒绝。旧ABI2正式registry入口退役，历史数值实现/测试和原证据保留原身份。

### 实际身份与计数

- 合同`jump_unified_raw37_wheel48_v1`，ABI4；input/output/config ID为`0x4a370102/0x4a370401/0x4a050401`。37→2048→1024→512→6、ELU、线性输出，raw Normal std来源约0.12。raw不做Tanh或±1裁剪；腿/轮增益0.5/10与目标保护1.5/80分开。BALANCE9由真实C phase/owner核验，全零阶段位与时间合法，不按argmax误算CROUCH。
- 选中Actor绝对路径`/home/kellen/wheel_leg_native_train/artifacts/2026_11_10__jump_motion_transfer48_refined/jump_distilled_actor.pt`，SHA `2a4d94517943559eaed51841e27be51e4672162a862ff07cd06979efc2e685fb`。首次PPO为Actor-only，均值逐张量核对，全部特征层可训练，新建Critic和两个Adam，不载入BC优化器。
- BC状态SHA `b4e214128d0473a02499dfa1616d67483005c5274972ba06b2a46c40276e6763`；只按`BC_resume_only`读取，监督1500（累计2500）不计为PPO。ONNX SHA `a6d1c2094196c7a5349bfea569359ca4f271a824be77ffe1deeda3781373b44e`；保留原reference evaluator成果，不新增ORT/ARM/实机通过声明。
- 固定C包来自只读`combat-dm-balance-jump-unified48-v1`，放在候选独立`jump_control_source`，不覆盖Motion/Recovery/ABI2包。实际主机构建库SHA **`75a190bcc2784bb51a78f54df0ac96640d4a926c66d4dc1f7da53383e56aad29`**，与原C_ready一致。包逐源码SHA、ABI结构大小、配置ID和合同摘要在加载边界核验；硬件默认OFF。轮最终4.8Nm只用于Jump。
- 实际静态MuJoCo对象产出`jump_unified_dense_causal_CPU_critic_v1`：**6370维/235字段**，字段名称/shape/单位/scale规范化JSON摘要与冻结schema一致，无补零/截断。Actor37与privileged完全分开。显式phase_count/wheel_cap/schema_class/packet_type及V4外部PD回调替代动态代码改写。
- 本轮新增真实物理积分 **0**、真实Jump PPO **0**、监督 **0**。脚本物理I/O fixture中的小网络100次CPU优化只验证数学/工程，不计真实Jump训练或能力。统一配方smoke/formal额度均0，未填写假10000、未提交任务。使用原CPU数学gamma0.9980019986673331、value_coef1、两路LR1e-5/9e-5，周期保存100；JobConfig必填evaluation节仅保留非执行元数据，Jump独立evaluate在dispatch明确拒绝。

### 正式行为与验收

| 范围 | 实际行为与证据 |
|---|---|
| 线性Actor/C控制 | `test_jump_unified`真实C验证raw4→腿请求2→保护1.5、raw12→轮请求120→保护80、最终轮4.8；C默认OFF，BALANCE37D合法，SETTLE自行回BALANCE且不需要Motion成功。Gaussian存储原raw及原log-prob。 |
| 初始化与完整训练恢复 | 真实选中Actor均值相等，两个Adam为空、所有特征可训练。小网络连续/分段Actor、Critic、两个Adam、std、训练RNG、计数相等；schema/raw/旧ABI错误拒绝。公共保存加入绝对`run_identity`，LoadedSource贯通恢复入口；新run full保留训练状态但重建局部世界/请求身份，同run才可恢复执行现场。 |
| 共享Motion/Recovery修复 | Generic初始化返回LoadedSource；同run核对路径/SHA来源与checkpoint绝对run身份/计数，配置full初始化不误作resume。Motion旧判定读取不存在的`source_run`，已改为真实checkpoint run_identity；Recovery回归改走save_full/load_source，未mock掉恢复。reward严格[N,1]防广播，infinite truncation仅用reset前具名Critic bootstrap一次；有限Jump timeout不bootstrap。 |
| 奖励和课程 | 原版本BALANCE/下蹲/蹬地/收腿/缓冲/恢复/自身完成奖励迁入，1ms真实packet只乘一次dt，终态立即结束子步；主动与landing-only来源锁、单次事件保持。AdaptiveCourse按真实Jump控制秒数，原window/确认/退档规则，request_id重复拒绝、旧generation只结算不进入新配额；UNKNOWN不推进。无额外采样。 |
| 故障与reset | 2ms脚本故障保留实际步数和终态，transition_valid=false，不当作完整10ms Actor行为。正常失败/task_timeout/工程故障分记。partial reset只触及指定plant；显式episode reset重建本地执行时钟，累计计费步数另存；phase切换不reset。 |
| 6370D执行保存 | 保存MuJoCo mjSTATE_INTEGRATION、C字节/输入/帧/观察令牌、tick、previous/FIFO/V4、奖励、课程RNG/pending/generation/配额、已结束未消费摘要和发布位置。同run精确恢复核对后端/模型/合同/schema；fresh-reset丢旧未完成请求/dwell，保留完成统计；未宣称跨后端恢复。静态对象完整保存恢复Critic/C字节相等；动态物理恢复仍NOT_RUN。 |
| 统计选模和保护 | 仅正常rollout结束摘要，每100更新导出；BALANCE、landing-only、active_jump分池，plant/DR/level/generation分别比较，样本不足为空，主动无成功则active为空。checkpoint成功及SHA→项目索引候选/引用原子提交→视图→轮换。来源Actor/BC/ONNX、Teacher4300及合同、数据/划分/证据、C包/库/build、V4/机械/schema进入既有项目索引；不另建registry、不删真实历史。 |
| 播放器 | 既有`wheel-leg play`的Jump任务适配使用同一C/V4/raw/限矩和Actor；共用GUI/headless run_loop，J请求只允许静止BALANCE，暂停无tick，reset明确重建episode；dry-run只检查来源/配置，不构造引擎/viewer/端口或积分。真实播放NOT_RUN。 |

### 验证与限制

核心回归 **280 passed / 0 failed / 0 skipped / 0 xfail，116.33秒**。最后终态分类调整后两项正式路径回归 **2 passed，30.02秒**；再加强有限timeout及同run完整adapter恢复断言，增强后的两项验收 **2 passed，28.70秒**。最后统一配方/独立evaluate拒绝调整后，Jump全部验收及CLI/config回归 **43 passed，0失败/跳过，32.53秒**。Ruff固定0.8.4：CI范围check通过，format149文件通过；架构150文件、0例外。没有安装或修改活动Conda工具。

```bash
cd /home/kellen/wheel_leg_refactor_ws
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/kellen/anaconda3/envs/issaclab/bin/python -B -m pytest -q -rf -p no:cacheprovider \
 tests/test_new_cli.py tests/test_new_config.py tests/test_new_artifacts.py tests/test_new_encoding.py \
 tests/test_new_lifecycle.py tests/test_new_learning.py tests/test_phase2_fixes.py tests/test_phase3_fixes.py \
 tests/test_architecture_rules.py tests/test_standing_survival_training.py tests/test_motion_handover_visual.py \
 tests/test_recovery_integration.py tests/test_motion_control_contract.py tests/test_motion_course_resume.py \
 tests/test_motion_production_integration.py tests/test_motion_transient_statistics.py tests/test_motion_value_diagnostics.py \
 tests/test_shared_control_core.py tests/test_jump_integration.py tests/test_unified_playback.py \
 tests/test_motion_host_consumers.py tests/test_motion_training_windows.py tests/test_jump_unified.py tests/test_jump_unified_host.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/kellen/anaconda3/envs/issaclab/bin/python -B -m pytest -q -rf -p no:cacheprovider \
 tests/test_jump_unified.py tests/test_jump_unified_host.py tests/test_new_cli.py tests/test_new_config.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python -B scripts/check_architecture.py
/tmp/wheel-leg-refactor-tools/bin/ruff check src/native_stand/{simulation,experiments,artifacts,learning,tasks,integrations} src/native_stand/cli.py scripts/check_architecture.py
/tmp/wheel-leg-refactor-tools/bin/ruff format --check src/native_stand/{simulation,experiments,artifacts,learning,tasks,integrations} src/native_stand/cli.py scripts/check_architecture.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES='' /home/kellen/anaconda3/envs/issaclab/bin/python -B -m native_stand train --config configs/jobs/jump_unified_raw37_wheel48.json --dry-run --json
```

19项Jump新增验收分portable与host：host实际读取选中Actor、BC、生产固定来源并静态构造MuJoCo，测试显式禁止mj_step/mj_step1/mj_step2。正式CPU路径使用脚本物理I/O，未fake decoder、C、V4、奖励、课程、PPO、保存或发布。原审计复现不作能力证据。

现有模型共享C/V4站高约25cm、参考环境约34cm，差异尚未完整归因；主动跳仍未掌握、既有PUSH覆盖0。保留原软件补偿和V4物理NET分路，不把等效载荷称为已辨识350N气弹簧，不提高腿限矩。支持估计仍为标记清楚的仿真全状态逆模型，保留有效性/非有限拒绝原因，接触真值只进入reward/Critic。未来有预算MuJoCo smoke需核对真实力路由、延迟、自然终点/reset和动态恢复；本轮**NOT_RUN：真实积分、GUI、GPU、ORT、ARM、实机、Isaac**。Isaac必须另做真实因果Critic/支持估计/执行，不注册占位。

### 准确变更文件

```text
src/native_stand/experiments/config.py
src/native_stand/experiments/worker.py
src/native_stand/learning/distillation/jump_labels.py
src/native_stand/learning/ppo/jump_nominal.py
src/native_stand/learning/ppo/jump_unified.py
src/native_stand/simulation/dispatch.py
src/native_stand/simulation/engine.py
src/native_stand/simulation/isaac/motion_backend.py
src/native_stand/simulation/jump_adapter.py
src/native_stand/simulation/jump_backend.py
src/native_stand/simulation/jump_batch.py
src/native_stand/simulation/jump_build.py
src/native_stand/simulation/jump_control_source/manifest.json
src/native_stand/simulation/jump_control_source/rl_control.c
src/native_stand/simulation/jump_control_source/rl_control.h
src/native_stand/simulation/jump_control_source/rl_jump_core.c
src/native_stand/simulation/jump_control_source/rl_jump_core.h
src/native_stand/simulation/jump_control_source/rl_jump_platform.c
src/native_stand/simulation/jump_control_source/rl_jump_platform.h
src/native_stand/simulation/jump_control_source/rl_motion31.c
src/native_stand/simulation/jump_control_source/rl_motion31.h
src/native_stand/simulation/jump_control_source/rl_shadow.h
src/native_stand/simulation/jump_dense_env.py
src/native_stand/simulation/jump_direct_critic.py
src/native_stand/simulation/jump_direct_env.py
src/native_stand/simulation/jump_mechanics.py
src/native_stand/simulation/jump_player.py
src/native_stand/simulation/jump_probe.py
src/native_stand/simulation/jump_response.py
src/native_stand/simulation/jump_shared_c.py
src/native_stand/simulation/jump_shared_v4.py
src/native_stand/simulation/jump_support_model.py
src/native_stand/simulation/jump_unified.py
src/native_stand/simulation/jump_unified_learning.py
src/native_stand/simulation/playback.py
src/native_stand/tasks/jump/adaptive_course.py
src/native_stand/tasks/jump/controller_critic.py
src/native_stand/tasks/jump/dense_reward.py
src/native_stand/tasks/jump/direct.py
src/native_stand/tasks/jump/geometry.py
src/native_stand/tasks/jump/phases.py
src/native_stand/tasks/jump/pretraining.py
src/native_stand/tasks/jump/reward.py
src/native_stand/tasks/jump/simple.py
src/native_stand/tasks/jump/spec.py
src/native_stand/tasks/jump/unified.py
src/native_stand/tasks/jump/unified_reward.py
src/native_stand/tasks/registry.py
configs/contracts/jump_unified_dense_causal_CPU_critic_v1.json
configs/contracts/jump_unified_raw37_wheel48_v1.json
configs/jobs/jump_unified_raw37_wheel48.json
tests/fixtures/motion_training_probe.py
tests/test_jump_integration.py
tests/test_jump_unified.py
tests/test_jump_unified_host.py
tests/test_phase3_fixes.py
tests/test_recovery_integration.py
docs/MIGRATION_20261105.md
STATUS.md
```

### 活动任务保护

开始/结束发现身份一致：Recovery PID1901402/start_ticks136388404，run=`2026_11_11__recovery_preentry48_remaining`；Motion PID1905160/start_ticks136466506，run=`2026_11_11-09_59__motion2097_turn_response_independentDR_full7903`。完整argv、cwd、源码/依赖记录只读核对，cwd和source_root均在生产仓库；结束RUNNING，分别249/597次。临时完整身份快照`/tmp/jump_migration_start.json`、`/tmp/jump_migration_end.json`不是长期依赖。

生产源码/冻结依赖/真实run/账本/模型/部署包和电控仓库未写；未停止、接管、恢复其他任务，未删除真实模型/轨迹，未提交/推送/刷板，也没有向其他会话发消息。仅候选与临时CPU fixture有写入。

---

## 历史：2026-11-10 PPO训练窗口交付

## 历史交付

本轮CPU范围**完成**：正常PPO统计 → 每100更新训练窗口 → CourseStrategy → 完整保存 → 项目索引原子发布 → 视图 → 轮换。额外采样步数为0；没有独立评估环境、进程或确定性回合。仅修改候选仓库，生产、冻结依赖、真实run及电控仓库未修改。未提交、推送、启动训练、物理积分或GUI。

| 缺陷 | 修复、正式消费者及正确行为证据 |
|---|---|
| 新run首次发布混入3302旧域 | `MotionProductionAdapter.build/_publish_candidate`新run按当前配置建空榜；来源榜单只存`source_training_candidates`。同run从项目索引恢复并核对域与获选文件SHA，冲突报错。`test_source3302_first_training_candidate_publication`实际加载完整3302，真实课程95→86、真实汇总→capture→context.save→after_save成功；本fixture无合格回合，榜单保持空，不伪造能力通过。 |
| 窗口重复、恢复丢证据 | 窗口包含run绝对身份、比较域、revision、起止更新；CourseStrategy保存消费位置和内容摘要，重复相同输入无动作、冲突/重叠拒绝。完整checkpoint保存当前未导出的已结束统计，恢复到新世界只装载这些cohort聚合，不接续旧未完成回合。`test_window_is_idempotent_and_resume_equivalent`、实际冻结统计器host测试验证。 |
| 早失败/Teacher/UNKNOWN | 继续使用原始随机训练回合，PPO epoch/minibatch不调用统计器。早失败进入Student结束分母，Teacher-only不计Student成功；按实际控制权累计Student/Teacher秒数。UNKNOWN中断确认但不增晋退级计数、不清已掌握档位；缺字段/非法计数/非有限统计报错。保留当前两次晋级、三次质量退级、四方向与DR覆盖规则。 |
| 跨run引用相互不可见、消费者覆盖 | owner把项目`models.json`绝对路径写入run.json，worker/阶段/候选/清理使用同一路径，runs-root隔离。索引保留多个consumer/role、产物类型/归属与多来源依赖。Motion full/Actor-only、resume、来源Teacher和前控模型/合同、通用初始化、蒸馏Teacher/数据/恢复及导出边界登记。输出的真实来源依赖在释放run角色后仍受保护。 |
| 固定u1600 | `configs/model_refs.json`显式记录用户固定实际路径及完整SHA；默认受管runs-root首次创建时核验登记，不扫描历史或硬编码95条白名单。host测试在临时项目索引核验u1600和3302保护。 |
| 发布崩溃与旧checkpoint视图覆盖 | checkpoint携带待发布证据；索引一次锁内提交榜单、引用和generation。保存后未提交可补发布；已提交但视图失败可重建，重放不增generation；权威榜单优先于checkpoint旧视图。`test_publication_crash_after_commit_replays_without_generation_change`和真实3302同run恢复验证。 |
| 清理计划过期、删除竞争 | `retention.prune`逐文件调用同锁`delete_owned`：重新读引用/依赖→核对inventory、归属、SHA→先持久化删除意图→unlink→实际结果。新消费者不能取得intent/deleted文件。保留最近两份、里程碑、停止点、双候选、固定及真实依赖的并集；无20份上限，不裁剪完整checkpoint。临时目录验证计划后新增引用、真实并发登记/删除、unlink中断恢复和硬链接释放0块。真实历史目录未执行清理。 |

### 验证结果与命令

最终核心回归：**261 passed，0 failed，0 skipped，0 xfail，57.23秒**。新增测试断言正确行为，不是审计错误值复现测试。真实3302仅在host子进程按一次加载结果装配与保存；同run恢复复用已捕获payload，不再反序列化第二个大模型。CPU fixture显式不恢复来源CUDA generator映射；GPU RNG恢复和真实物理仍NOT_RUN。小网络CPU完整恢复、DualAdam、解码、延迟、SourceAnchor、事件及旧生命周期均在核心回归内。

```bash
cd /home/kellen/wheel_leg_refactor_ws
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/kellen/anaconda3/envs/issaclab/bin/python -B -m pytest -q -rf -p no:cacheprovider \
 tests/test_new_cli.py tests/test_new_config.py tests/test_new_artifacts.py \
 tests/test_new_encoding.py tests/test_new_lifecycle.py tests/test_new_learning.py \
 tests/test_phase2_fixes.py tests/test_phase3_fixes.py tests/test_architecture_rules.py \
 tests/test_standing_survival_training.py tests/test_motion_handover_visual.py \
 tests/test_recovery_integration.py tests/test_motion_control_contract.py \
 tests/test_motion_course_resume.py tests/test_motion_production_integration.py \
 tests/test_motion_transient_statistics.py tests/test_motion_value_diagnostics.py \
 tests/test_shared_control_core.py tests/test_jump_integration.py \
 tests/test_unified_playback.py tests/test_motion_host_consumers.py tests/test_motion_training_windows.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python -B scripts/check_architecture.py
/tmp/wheel-leg-refactor-tools/bin/ruff check src/native_stand/{simulation,experiments,artifacts,learning,tasks,integrations} src/native_stand/cli.py scripts/check_architecture.py
/tmp/wheel-leg-refactor-tools/bin/ruff format --check src/native_stand/{simulation,experiments,artifacts,learning,tasks,integrations} src/native_stand/cli.py scripts/check_architecture.py
```

架构检查：121个正式文件，0基线例外。CI固定Ruff0.8.4：check通过、format检查120文件通过，未改活动Conda环境。实际删除仅在pytest的`/tmp` fixture。原Portable全量历史结果不是本轮结果，本轮未重跑其超出授权的物理用例。

### 准确变更文件

| 文件（相对候选根） | 变化 |
|---|---|
| `src/native_stand/simulation/isaac/motion_backend.py` | 新run榜单隔离、恢复检查/补发布、训练窗口、已结束统计捕获、实际来源与前控引用 |
| `src/native_stand/tasks/motion/progressive_episodes.py` | 已结束聚合快照/恢复、实际Student/Teacher计分秒数 |
| `src/native_stand/tasks/motion/course_strategy.py` | 窗口身份、内容冲突检测及消费幂等 |
| `src/native_stand/tasks/motion/highspeed_course.py` | 区分合法UNKNOWN与损坏课程统计，保留既有阈值 |
| `src/native_stand/tasks/motion/motion_candidates.py` | 拒绝缺失/损坏证据、记录有效结束回合数 |
| `src/native_stand/artifacts/model_refs.py` | 项目权威索引、多消费者/来源、原子发布generation、固定引用与同锁删除意图 |
| `src/native_stand/artifacts/retention.py` | 执行时引用复核、同锁逐文件删除及逻辑/实际释放空间 |
| `src/native_stand/experiments/owner.py` | 固定索引路径写入run记录、默认项目固定资产初始化 |
| `src/native_stand/experiments/worker.py` | 共享索引、产物/来源登记、公共清理接线 |
| `src/native_stand/experiments/distillation_job.py` | 蒸馏真实Teacher、数据及恢复引用 |
| `src/native_stand/cli.py` | 导出来源及输出依赖登记 |
| `configs/model_refs.json` | 用户固定u1600实际路径/SHA声明 |
| `tests/test_motion_training_windows.py` | 窗口/恢复/UNKNOWN、项目索引、发布崩溃、并发/意图/硬链接实际删除回归 |
| `tests/test_motion_host_consumers.py` | 增加真实3302正式发布和冻结回合统计器子进程验收 |
| `tests/fixtures/motion_training_probe.py` | 真实3302单次加载、实际课程/汇总/保存/发布/同run恢复，物理I/O为CPU fixture |
| `tests/fixtures/motion_episode_probe.py` | 实际冻结统计器的早失败、Teacher、跨世界恢复及单次导出 |
| `tests/test_phase3_fixes.py` | 既有legacy路由fixture补显式课程身份，不放宽断言 |
| `docs/MIGRATION_20261105.md`、`STATUS.md` | 唯一当前摘要及简短任务记录 |

### 活动训练与边界

开始/结束重新发现的worker PID/start_ticks、完整argv及cwd一致：Motion `348863/114992785`，Recovery `1265175/128088780`（实际run为`2026_11_10__recovery_legacy_reward_Swing10000`，不是旧交接中的PID）。两者cwd为生产根，源码根为生产`src`；Motion依赖原`runtime_view_r4/src`及冻结bundle，未改写。结束只读状态为RUNNING，分别5477/71更新；TensorBoard169976存活。临时身份快照`/tmp/refactor_task_start.json`、`/tmp/refactor_task_end.json`不是长期项目依赖。

旧worker未迁移到新索引，未对它们调用新清理器。未改变真实run、账本、模型、部署包或活动环境；没有提交/推送/刷板/长训。物理、GPU、GUI均NOT_RUN；后续有界执行smoke只验证执行链，不建立独立能力评估，也不为晋级额外采样。

---

## 历史：上一轮通用重构交付（不作为本轮验收）


本次只改候选 `/home/kellen/wheel_leg_refactor_ws` 与独立电控集成worktree `/home/kellen/combat-dm-balance-control-integration`。生产、冻结依赖、活动run及另两处电控owner工作区只读。未提交、推送、刷板、启动训练或执行物理积分。

| 交付域 | 状态 | 已证实与具体剩余项 |
|---|---|---|
| Motion生产语义CPU闭环 | 部分完成 | 公共worker→正式adapter→真实decoder/课程/策略装配→完整状态捕获通过；真实3302来源装配通过。新环境工厂的Isaac构造、V4实际力路由与自然reset仍NOT_RUN，不能宣布生产迁移完成。 |
| Recovery当前版本 | 部分完成 | 显式PhysicalActor与909D高度合同、独立课程、物理std、实际来源、PPO参数及保存恢复接通；真实物理与整台执行快照恢复NOT_RUN。旧908交Motion不再充当当前目标。 |
| 唯一播放器 | 部分完成 | `wheel-leg play --run RUN --checkpoint CHECKPOINT`；Motion/909 Recovery同一GUI/headless循环、共享C、V4和实际adopted previous；dry-run无模型反序列化/引擎/viewer/端口。真实GUI/积分NOT_RUN；Jump播放、动态Recovery高度输入和跨技能连续交接尚未接通。 |
| 共享C核心 | 部分完成 | 电控集成worktree权威源码；训练绑定带SHA的固定源码包。实际电控Motion观测/decoder/PD/adopt调用该核心；主机8/8通过，表中心与poly5身份分离，CPU/device张量参考核验。Jump同库ABI2数值通过；完整六轴owner统一与GPU实际一致性NOT_RUN。 |
| Jump接口 | 部分完成 | 当前37D/ABI2、名义轨迹＋PUSH腿.05/其他.10、轮10残差、原轨迹身份、公共registry/policy/保存恢复、实际C状态通过。没有迁入历史FunctionType/exec/AST trainer。完整物理环境/因果Critic生产布局、公共worker物理执行、播放器尚未接通；禁止用CPU fixture64作为生产Critic。 |
| 性能 | 部分完成 | 系数0跳过FD/SVD；删除正式链中被863覆盖的中间Critic和多重队列reset；单轮耗时不再被累计均值覆盖；详细GPU计时只显式开启并在update边界同步。A/B未测，不报告提速比例。几何/接触进一步缓存尚未实施。 |
| 真实物理/GUI/GPU | 未运行 | 本轮没有授权。下方给出有界准备命令。 |

### 八项Motion缺陷与正式消费者

| 原错误 | 实际修复与正式消费者 | 正确行为证据 | 剩余限制 |
|---|---|---|---|
| Student40/旧中心，Teacher按1.5/100解释mean | `environment.control_router`分别从Student嵌入合同和SHA绑定Teacher合同建`ActionDecoder`，`Handover.actions`只在末端转桥单位；机械保护仍下游消费 | `test_motion_control_contract`：Student±80、零腿中心、Teacher.5→40、actual previous轮.8；共享C与正式decoder对照 | 真实物理保护/连续交接NOT_RUN |
| 延迟未进入DR与Critic | 环境构造注入`DelayedDR`；reset后历史填充前写Actor20、target5、轮7/11 tau0；`sensor_packets`实际读队列；最终863读实际状态 | 名义/单项/组合/部分reset、环形缓冲、实际到达时刻；`test_motion_host_consumers`实际冻结863时序列读回 | DR完整物理reset及真实队列执行NOT_RUN |
| 95来源被37替代 | `CourseStrategy`集中版本、来源迁移、keys/mix、stage/advance；未知拒绝。来源95，当前yaw-floor迁移为86，保留匹配key状态 | `test_motion_course_resume`：95/86一致、方向独立两次确认/质量退档、低速失败不降高速 | Isaac真实revision重建NOT_RUN |
| 单Adam/full/warmup/独立预算错误 | `supported.assemble_from_full`按明确schema装DualAdam；Actor-only接通；warmup剩余额度，独立run祖先另记，RNG最后恢复；course_updates独立于本run update | 真实3302网络/两个Adam/Teacher全等；小网络连续/分段一致、Actor前5不变第6变；worker生产装配capture | 真实full首更新及Actor-only物理5/6窗口NOT_RUN |
| 来源KL未入loss | `SourceAnchor`进入实际Actor loss；因果nominal/student/no-event/姿态承重mask；Teacher路径/SHA从显式祖先引用解析并保护 | 扰动Student后loss/梯度非零；DR/event/Teacher/空mask无保留梯度；缓存/RNG/Teacher不污染 | 大网络GPU梯度窗口NOT_RUN |
| C0事件matcher/force-on/零样本0 | `transient_events.event_class`复用原1ms力生成；迁入force-off统计、连续1s quiet、能量RMS、UNKNOWN | `test_motion_transient_statistics`及实际冻结pulse消费者；双侧四模式正负事件 | 真实NET/PD-D/力路由物理对照NOT_RUN |
| 周期保存未发布/清理 | `driver.save_now`保存成功→`after_save`候选与引用→周期prune；分别保护运动/站立/各DR；失败不发布 | 临时run保存/发布/轮换最终保留集合；不授best/accepted；旧保存失败回归保持 | 第一次候选在成功保存后发布，完整index还写sidecar；迁移run须带sidecar或后续checkpoint index |
| 先建后关违反单例 | 课程后处理先准备状态，保存成功后才旧close→释放→新factory；构造失败env=None，保存点已存在；重建后恢复RNG | 工厂顺序/失败与最后保存顺序CPU回归 | 真正Isaac单例构造/资源释放NOT_RUN |

### 验证

- 最终核心回归：**249 passed / 0 failed / 0 skipped / 0 xfail，51.96秒**；命令见下方。所有新增portable回归断言正确结果，未把审计“错误值复现通过”当修复通过。
- 真实3302：`/home/kellen/wheel_leg_native_train/runs/2026_11_8-00_13__motion_C0_boundary_KL10000/checkpoints/model_update_3302.pt`，SHA `739e38287a8a12508422291de0cc82c863ce3c4d3047f8ab2e5cc9867f947107`。一次CPU反序列化，Actor/Critic/Teacher逐张量相等，DualAdam各8个state，warmup5，独立预算0；来源物理现场未恢复。
- Recovery真实Actor来源SHA `bd0d3cb608c5e447de7a82495dcecbbbfb12c735cd87b909e781a05d38f41326`：PhysicalActor、909D、物理std、原job/run-record一致，未做优化/积分。
- 本机证据暂存 `/tmp/motion_source_3302_verified.json`、`/tmp/recovery_source_verified.json`、`/tmp/motion_consumers_verified.json`、`/tmp/wheel_leg_core_final.xml`、`/tmp/wheel_leg_portable.xml`。这些是本次主机结果，不宣称永久可复核。
- Portable CI完整选择（以`/tmp/wheel_leg_portable.log`为准）：**1639 passed / 4 failed / 23 skipped**，92.93秒；封锁mj_step/mj_step1/mj_step2，无积分。两项失败是无授权旧MuJoCo积分被拦截；`test_motion_differential::test_retry_bound_to_exact_settled_zero_ppo_failure`缺候选历史run；旧`test_motion_mcu_playback::test_case_never_hides_the_execution_step_inside_the_5ms_branch`断言旧源码字符串。未扩大排除/xfail/放宽断言。后续源码修改另由最终核心回归验证，未重跑整个Portable范围。
- Ruff固定0.8.4，独立 `/tmp/wheel-leg-refactor-tools`，不改活动环境；CI规定范围check/format通过。架构检查121文件、0基线例外。
- 静态真实MuJoCo模型构造：nq45、4闭链、root Z0.340033872m、控制tick0、积分0。使用小网络验证装配，不冒充真实来源物理能力。

```bash
cd /home/kellen/wheel_leg_refactor_ws
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/kellen/anaconda3/envs/issaclab/bin/python -m pytest -q -rf -p no:cacheprovider \
 tests/test_new_cli.py tests/test_new_config.py tests/test_new_artifacts.py tests/test_new_encoding.py \
 tests/test_new_lifecycle.py tests/test_new_learning.py tests/test_phase2_fixes.py tests/test_phase3_fixes.py \
 tests/test_architecture_rules.py tests/test_standing_survival_training.py tests/test_motion_handover_visual.py \
 tests/test_recovery_integration.py tests/test_motion_control_contract.py tests/test_motion_course_resume.py \
 tests/test_motion_production_integration.py tests/test_motion_transient_statistics.py tests/test_motion_value_diagnostics.py \
 tests/test_shared_control_core.py tests/test_jump_integration.py tests/test_unified_playback.py tests/test_motion_host_consumers.py
```

### 未运行验证的准备

只准备，不执行以下物理命令。候选新任务必须使用新的run及固定源码；不能resume活动run。物理smoke应明确覆盖V4电机/NET、DR延迟、Teacher→Student现场、full首更、Actor-only5/6、更长于30s自然done、一次revision重建。当前并未把这些缺项标成accepted。

```bash
# CPU准备A/B配置；每块25次，预热5、测量20，墙钟1800s，metrics每轮；不提交
PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python scripts/motion_performance.py prepare \
 --config /home/kellen/wheel_leg_native_train/configs/jobs/motion_C0_yaw1to4_full_KL10000.json \
 --output /tmp/wheel_leg_performance_plan \
 --baseline-source /home/kellen/wheel_leg_native_train/src --candidate-source /home/kellen/wheel_leg_refactor_ws/src
# 有真实A/B结果之后，只读比较中位数/P95/吞吐/显存；不同模型/DR/保存/配置拒绝
PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python scripts/motion_performance.py compare BASELINE_RUN CANDIDATE_RUN
# 唯一播放入口；本轮只执行--dry-run。真正播放须另有物理授权。
PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python -m native_stand play --run RUN --checkpoint CHECKPOINT --headless --steps 100 --dry-run
```

A/B旧生产计时/显存未具备同接口记录时比较必须拒绝，不能拿旧累计均值比较。真实物理命令未授权，标NOT_RUN；这不代表CPU实现已全部完成。

### 活动任务保护

结束核对Motion PID348863，start_ticks114992785，完整argv仍为原生产worker，CUDA_VISIBLE_DEVICES=0，原owner session191bacc3-e5c7-4b8f-ae3d-5677eada837b，RUNNING/5222次。新Recovery PID1119711已退出，STOPPED_BOUNDED/761次。旧交Motion链STOPPED_BOUNDED/856次，另有此前12次=868，不恢复。记录来自一次结束快照；没有watcher、停止、重启或热改。

### 实际文件

以下列本次源码/fixture/必要脚本的准确路径。`artifacts/metrics.py`、`experiments/config.py`及架构脚本只将六处tuple-isinstance改成Python3.12联合类型以满足CI Ruff0.8.4；未扩大排除。删除无消费者全局替换入口`simulation/isaac/prefix.py`和`progressive.py`。另改本报告与STATUS顶部摘要。

<details><summary>候选实际文件清单</summary>

```text
scripts/check_architecture.py
scripts/motion_performance.py
src/native_stand/artifacts/metrics.py
src/native_stand/artifacts/model_refs.py
src/native_stand/cli.py
src/native_stand/experiments/config.py
src/native_stand/experiments/worker.py
src/native_stand/learning/policies/factory.py
src/native_stand/learning/ppo/actor_initialization.py
src/native_stand/learning/ppo/assembly.py
src/native_stand/learning/ppo/controlled.py
src/native_stand/learning/ppo/dual_adam.py
src/native_stand/learning/ppo/jump_nominal.py
src/native_stand/learning/ppo/recovery_cubic.py
src/native_stand/learning/ppo/source_anchor.py
src/native_stand/learning/ppo/supported.py
src/native_stand/simulation/control_core.py
src/native_stand/simulation/control_core_source/manifest.json
src/native_stand/simulation/control_core_source/rl_control.c
src/native_stand/simulation/control_core_source/rl_control.h
src/native_stand/simulation/control_core_source/rl_jump_core.c
src/native_stand/simulation/control_core_source/rl_jump_core.h
src/native_stand/simulation/control_core_source/rl_jump_nominal.c
src/native_stand/simulation/control_core_source/rl_jump_nominal.h
src/native_stand/simulation/control_core_source/rl_jump_nominal_table.c
src/native_stand/simulation/control_core_source/rl_jump_nominal_table.h
src/native_stand/simulation/control_core_source/rl_jump_platform.c
src/native_stand/simulation/control_core_source/rl_jump_platform.h
src/native_stand/simulation/control_core_source/rl_motion31.c
src/native_stand/simulation/control_core_source/rl_motion31.h
src/native_stand/simulation/control_core_source/rl_policy_core.c
src/native_stand/simulation/control_core_source/rl_policy_core.h
src/native_stand/simulation/control_core_source/rl_shadow.h
src/native_stand/simulation/driver.py
src/native_stand/simulation/engine.py
src/native_stand/simulation/isaac/delayed_execution.py
src/native_stand/simulation/isaac/environment.py
src/native_stand/simulation/isaac/execution_buffers.py
src/native_stand/simulation/isaac/fullchain.py
src/native_stand/simulation/isaac/handover.py
src/native_stand/simulation/isaac/height.py
src/native_stand/simulation/isaac/motion_backend.py
src/native_stand/simulation/isaac/motion_base.py
src/native_stand/simulation/isaac/motion_quality.py
src/native_stand/simulation/isaac/sensor_packets.py
src/native_stand/simulation/isaac/transient_events.py
src/native_stand/simulation/jump_nominal_c.py
src/native_stand/simulation/jump_shared_c.py
src/native_stand/simulation/mujoco_model.py
src/native_stand/simulation/mujoco_playback.py
src/native_stand/simulation/playback.py
src/native_stand/simulation/playback_assets.json
src/native_stand/simulation/playback_execution.py
src/native_stand/simulation/recovery_isaac/backend.py
src/native_stand/simulation/recovery_isaac/environment.py
src/native_stand/simulation/recovery_isaac/execution.py
src/native_stand/simulation/recovery_isaac/handover_routing.py
src/native_stand/simulation/recovery_isaac/lifecycle.py
src/native_stand/simulation/recovery_isaac/observations.py
src/native_stand/simulation/recovery_isaac/physics.py
src/native_stand/simulation/recovery_isaac/routing.py
src/native_stand/simulation/v4_response.py
src/native_stand/tasks/jump/__init__.py
src/native_stand/tasks/jump/nominal.py
src/native_stand/tasks/jump/simple_critic.py
src/native_stand/tasks/jump/spec.py
src/native_stand/tasks/motion/action_contract.py
src/native_stand/tasks/motion/c0_boundary_course.py
src/native_stand/tasks/motion/c0_course.py
src/native_stand/tasks/motion/c0_events_course.py
src/native_stand/tasks/motion/c0_fast_yaw_course.py
src/native_stand/tasks/motion/course_strategy.py
src/native_stand/tasks/motion/delayed_execution.py
src/native_stand/tasks/motion/encoding.py
src/native_stand/tasks/motion/highspeed_course.py
src/native_stand/tasks/motion/legacy_spec_adapter.py
src/native_stand/tasks/motion/motion_candidates.py
src/native_stand/tasks/motion/scale_corrections.py
src/native_stand/tasks/motion/training_candidate.py
src/native_stand/tasks/motion/transient_statistics.py
src/native_stand/tasks/motion/value_diagnostics.py
src/native_stand/tasks/recovery/cubic_course.py
src/native_stand/tasks/recovery/cubic_randomization.py
src/native_stand/tasks/recovery/handover_course.py
src/native_stand/tasks/recovery/height_context.py
src/native_stand/tasks/recovery/standalone_course.py
src/native_stand/tasks/registry.py
tests/fixtures/cpu_jump_job.json
tests/fixtures/motion_C0_control_contract.json
tests/fixtures/motion_C0_course_state.json
tests/fixtures/motion_host_consumer_probe.py
tests/test_jump_integration.py
tests/test_motion_control_contract.py
tests/test_motion_course_resume.py
tests/test_motion_host_consumers.py
tests/test_motion_production_integration.py
tests/test_motion_transient_statistics.py
tests/test_motion_value_diagnostics.py
tests/test_new_lifecycle.py
tests/test_phase3_fixes.py
tests/test_recovery_integration.py
tests/test_shared_control_core.py
tests/test_unified_playback.py
```

</details>

电控集成worktree另改`Makefile`、`application/chassis/{rl_motion31.c,rl_control.c}`、`tests/run_rl_host_tests.sh`；新增`rl_policy_core.c/.h`与从Jump owner只读快照迁入的8个`rl_jump_*`源/头文件。两个原owner worktree未改，保持两个Git仓库，没有第三套长期控制源码。

## 审计范围与边界

- 对象：`/home/kellen/wheel_leg_native_train` 的源码、测试、工程配置与维护文档。
- 方法：读源码与配置核实调用关系；CPU后端跑真实网络与真实优化器；子进程验证生命周期；SHA核对保护对象。
- 本会话未执行：新的 Isaac／MuJoCo 物理训练、PPO0、蒸馏优化、World Model 优化；未停止或接管任何正在运行的任务；未 Git 提交；未发送消息。
- 本会话观察到另一个 owner 已在 21:21 用新CLI提交 Isaac 续训（见“活动任务保护”）；物理能力结论不属于本报告。

## 第一阶段（已交付，保持）

统一入口 `wheel-leg` / `python -m native_stand`；`JobConfig` 一次解析；`runs/<run_id>/` 单一结果结构；三类检查边界；文档链接与HAPI退出硬门槛；`artifacts/io.py` 作为SHA／摘要／原子写入的唯一实现。

## 第二阶段：已修复的复现缺陷

| 缺陷 | 修复前表现 | 修复后 | 证据 |
|---|---|---|---|
| protected无效 | `10,20,30,40`、keep_recent=2、protected=[10] → 删10、20 | 保留集合=最近两份∪里程碑∪best∪selected∪恢复来源∪引用模型；`keep_reasons` 记录原因 | `tests/test_phase2_fixes.py::test_earlier_best_outside_recent_window_is_kept`、`tests/test_new_artifacts.py` |
| 轮换误伤外部 | 只按文件名匹配 | 只处理 `checkpoints/` 内、非软链接、真实属于本run的文件 | `test_symlink_candidate_is_not_treated_as_owned` |
| 引用索引未接入 | `ModelIndex` 只被测试使用 | run owner 在初始化/teacher/stage续训/best/selected/部署登记；轮换读 `consumer_paths()` | `test_model_registered_by_reference_is_protected` |
| 同SHA覆盖路径 | 同权重第二个路径覆盖第一个 | 身份=(SHA,完整路径)，两路径各自保护 | `test_same_sha_two_paths_both_protected` |
| 恢复配置丢失 | run.json=full，worker=scratch | worker 只从 `run.json` 重建 `JobConfig`；原始文件被删也能跑完 | `test_worker_uses_run_record_not_original_config` |
| resume领新预算 | `cmd_resume` 新建run与新额度 | `resume --run` 在同一run上开新attempt，预算不刷新，耗尽即拒绝 | `test_resume_continuation_and_budget_not_refreshed`、`test_resume_refuses_when_budget_exhausted` |
| 阶段不接续 | formal 每阶段重新初始化 | formal 从 smoke 最终完整checkpoint 载入 Actor/Critic/Adam/std/RNG/课程 | `test_full_state_restore_reproduces_next_update`、端点跑 smoke2+formal3 |
| 非周期结束丢状态 | 最新checkpoint停在4，run报COMPLETED | 所有结束路径落一次最新完整状态 | `test_final_save_covers_non_periodic_end`（5次更新→`model_update_5.pt`） |
| Critic吃错列 | 配置863，实际首层 `[8,31]` | Critic 只吃 `student_critic` 声明列；宽度不符直接报错 | `test_critic_uses_declared_privileged_columns`（首层31/64，改特权列会改变输出） |
| critic_lr未生效 | optimizer 只有 `[1e-5]` | 两个参数组 `[1e-5, 9e-5]`，参数归属各自独立 | `test_optimizer_splits_actor_and_critic_learning_rates` |
| warmup未作用于真实更新 | 只测布尔函数 | 真实 Adam：前1次Actor参数完全不变、Critic改变；之后Actor改变 | `test_warmup_updates_only_critic_then_actor` |
| 算法未参与分派 | 只看 backend+task，world_model 也走PPO计数 | `simulation/dispatch.py` 按 操作×任务×算法×后端 判定，不支持组合在创建run前拒绝 | `test_world_model_training_is_refused`、`test_evaluate_requires_ppo` |
| 两个状态写入者 | launcher 在握手后仍写 state | launcher 只写 SUBMITTED，worker `claim()` 接管；非owner写入被拒 | `test_launcher_cannot_overwrite_after_claim` |
| 终态可被改写 | COMPLETED 后仍能改 failure | 相同内容重复提交无操作，冲突内容拒绝 | `test_terminal_is_idempotent_and_rejects_conflicts`、`test_state_terminal_cannot_be_overwritten` |
| 握手误报成功 | 子进程立即崩溃仍返回成功 | 无消息且子进程已退出 → FAILED，CLI 退出码非0 | `test_worker_crash_reports_failure` |
| 身份未比启动tick | 只比命令行子串 | 命令行+tick 双条件，tick 不符判 MISMATCH | `test_process_identity_rejects_reused_pid` |
| stop语义含混 | 未等退出也说 `stopped=true` | 分别返回 `request_recorded` / `signal_sent` / `exited` / `stopped` | `test_stop_distinguishes_request_signal_and_exit` |
| 保存失败伪称成功 | — | 保留上一有效checkpoint，写 `save_error.json`，报告真实完成计数 | `test_save_failure_keeps_previous_state_and_reports_counts` |
| 无评估却授予best | 假后端硬编码 `mark_best` | 训练不产生 best/selected；`capability_accepted` 恒 False | `test_training_run_does_not_invent_best_or_accepted` |

## 第二阶段：真实调用接线

| 能力 | 状态 | 说明 |
|---|---|---|
| `train`（cpu） | 真实执行 | 真实 Actor/Critic/Adam/rollout/受控PPO；后端是声明的替代动力学，结果标注 `backend=cpu` |
| `train`（isaac） | 真实执行 | `simulation/isaac/engine.py` 适配既有环境；`AppLauncher` 先于Isaac导入；Critic列来自既有 `fullchain` 定义（863）；一次性启动检查核对 spec 的 actor/critic 宽度 |
| `evaluate` | 真实执行 | 载入完整checkpoint，逐条件逐环境JSON，写入 `evaluations/index.json`；best/final分开；无独立评估不授予accepted |
| `distill` | 真实执行 | 固定Teacher取标签、按episode划分、BC有界round、验证集选模、留出集只报告 |
| `diagnose` | 计划 + 已存状态读取 | 不启动新物理运行 |
| `train`（mujoco） | 明确拒绝 | 跨引擎核对未迁移到新配置，报错说明原因 |
| `train`（world_model） | 明确拒绝 | 训练分派表不含该组合；WM 更新与 PPO 计数分别记录 |
| `resume` | 真实执行 | 同一run、新attempt、预算不刷新、默认取最新完整状态 |

未做物理验证：本会话没有启动 Isaac／MuJoCo 运行；Isaac 适配器的数值行为只做了静态与CPU侧检查。执行时发现另一个 owner 已用该适配器提交正式续训（RUNNING，已用80次更新），其能力结果不属于本报告。

## 第二阶段：合并与退役

- `simulation/engine.py` 成为唯一训练循环；旧 `simulation/cpu.py::run(context)` 占位实现删除，改为提供环境。
- `experiments/worker.py` 重写：单写入者、按run记录取配置、阶段接续、最终保存、预算权威计数、错误分开记录。
- `experiments/owner.py`：删除握手后的状态写入；新增 `resume_run()`；`process_identity` 接入启动tick。
- `artifacts/checkpoint.py` 的 `write_inventory` 合并到 `artifacts/retention.py`（单一权威）。
- `cli.cmd_evaluate` / `cmd_distill` 从“直接报未接入”改为调用共享实现。
- `learning/ppo/{networks,assembly,evaluate}.py`、`learning/distillation/loop.py`、`simulation/dispatch.py` 为新增真实实现。
- `configs/jobs/` 保留真实任务配置；**工程fixture移到 `tests/fixtures/cpu_motion_job.json`**，不再作为默认训练配置。

## 测试

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_new_{cli,config,artifacts,encoding,lifecycle,learning}.py tests/test_phase2_fixes.py
# 98 passed
```

覆盖到的真实边界：文件保留与轮换、引用索引、run记录→worker→算法恢复、预算不刷新、阶段接续、非周期最终保存、真实优化器参数归属与warmup、分派拒绝、单写入者、终态幂等、握手失败、停止语义、评估矩阵、蒸馏划分与选模隔离。

## 全量测试

`pytest tests/`（2026-11-05 22:0x）：**2283 passed, 82 failed, 70 errors, 15 skipped**。
错误与失败来自三类，均未由本次改动引入：

1. 缺失历史产物（`FileNotFoundError`）：`runs/standing_scissor_20261006/...`、`isolated/recovery_mcu_v2_bc_20261011/best_validation.pt` 等；
2. 外部仓库的固件C测试：`/home/kellen/combat-dm-balance-main/tests/test_rl_execution.c:37` 断言失败，连带 `test_firmware_sampling`、`test_motion_firmware_control`、`test_oscillation_search` 报 command failed；
3. 已退役行为（本轮已改写或替代，见下）。

其余失败抽样证据指向缺失历史产物（例如 `runs/standing_scissor_20261006/formal/model_update_50.pt`、`isolated/recovery_mcu_v2_bc_20261011/best_validation.pt`）与已退役行为。本次触及的旧测试逐项处理：

- 改写：`test_new_artifacts` 的保留用例（新API与语义）、`test_new_lifecycle` 的终态/保存失败/引擎不可用用例、`test_new_cli` 的 evaluate dry-run。
- 替代契约：旧 supervisor 多轮CLI用例删除，由 `tests/test_new_cli.py` 与 `tests/test_phase2_fixes.py` 覆盖。
- 未修复的历史失败不再笼统归因，报告中保留具体 node 与错误文本。

## 活动任务保护（执行时核对）

| 检查 | 结果 |
|---|---|
| 16:51 run | 已于 20:46 `STOPPED_BOUNDED`（user_signal，442次更新，`model_update_442.pt` 已保存）；本会话未重启 |
| 21:21 新run | 另一 owner 提交，`backend=isaac`，源码根 = 仓库 `src`，RUNNING，`state.json` 由 worker PID 写入 |
| 本会话对 `src/` 的改动 | 均在新架构模块内；未改 `students/`、`command_v6/`、`isolated/` 下任何文件 |
| 关键模型 | 442最终状态、round02来源、863初始化、Recovery1500、Motion4500、Motion1900、2405、两份WM best SHA 与记录一致 |
| 丢失项 | `r3/smoke/model_update_12.pt` 已不存在，STATUS 已删除该“受保护”说法 |

## 未完成

1. World Model 的新入口只做了分派拒绝与计数分离约定，没有接入训练循环（本轮不授权 WM 优化）。
2. `diagnose` 仍以计划与已存状态读取为主，没有新的诊断执行器。
3. Isaac 适配器的 `env.step` 返回值形状按既有实现推断，未在真实运行中核对；首次真实运行时需要确认 `(obs, reward, terminated, truncated, info)` 与 `return_extras` 分支。
4. 历史重复实现（`students/` 与 `command_v6/` 的SHA helper、`supervisor` 残留函数）仍在，未批量迁移；它们当前只服务冻结证据与未迁移的旧脚本。
5. 未安装 Ruff（环境无该模块），新代码按行宽100、py312 手写；`pyproject.toml` 已配置。

## 第三阶段（2026-11-05 夜）：独立工作区与执行分叉修复

**工作区**：`/home/kellen/wheel_leg_refactor_ws`（以主工作树当时的 dirty 源码为起点复制，不含 runs/isolated/artifacts/资产）。
主工作树 `src` 仍服务活动训练，本阶段修改**没有**回写主工作树。

### 复现并修复

| 问题 | 复现 | 修复 | 测试 |
|---|---|---|---|
| 统一分派拒绝活动配置 | 活动配置声明 `assets.runtime_template`，`backend_for()` 要求 `assets.legacy_spec_path` | `dispatch.execution_service(job)` 成为唯一判定点：`runtime_template`→生产 Motion 适配器，`legacy_spec_path`→通用Isaac适配器，两者都没有→拒绝；`engine.available()` 接受两种声明且不查GPU | `test_active_config_is_accepted_by_single_dispatch`、`test_execution_service_is_pure_and_distinguishes_paths` |
| full初始化不加载来源 | `run_stage` 只看 `resume_from/restore`，配置声明 full 时传空 | 新增 `_initialization_payload(job)`：显式恢复→阶段接续→配置声明，三选一在引擎入口解析；SHA不符或文件缺失立即报错 | `test_full_initialization_payload_is_loaded_from_config`、`test_full_initialization_reaches_the_optimizer`（哨兵0.375穿过 `restore_full`）、`test_missing_initialization_source_is_a_clear_error` |
| CLI resume 依赖原配置 | `cmd_resume` 读 `record["config"]["config_path"]` | 改 `job_from_record(record["config"])`；默认恢复点改为 inventory/state 里的**最新完整状态**（不再先取 selected） | `test_cli_resume_works_after_original_config_deleted`（删除原文件后仍 exit 0） |
| 探索尺度用输入单位 | `leg_rad/1.5`、`wheel_rad_s/100` | 改输出包络 `leg_scale_rad`(0.15)、`wheel_scale_rad_s`(40)；31D输入归一化不变 | `test_exploration_std_uses_output_envelope`（0.3rad→raw 2.0；8rad/s→raw 0.2） |
| 轮换按glob认定owned | 未登记的 `model_update_10.pt` 也被删 | 候选只来自 inventory；外部路径、软链接、超出 `checkpoints/`、登记SHA不符（64位hex）的文件跳过 | `test_prune_ignores_files_not_in_inventory`、`test_prune_skips_replaced_and_external_and_symlink`、`test_plan_and_execution_match_after_inventory_scope` |

### 测试

```bash
cd /home/kellen/wheel_leg_refactor_ws
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_new_{cli,config,artifacts,encoding,lifecycle,learning}.py \
  tests/test_phase2_fixes.py tests/test_phase3_fixes.py
# 108 passed
```

### 执行分叉现状（未合并，明确记录）

- 生产路径：`simulation/isaac/motion_backend.py::run(context)`，自有循环、自有保存与计费，`assets.runtime_template` 系列字段；活动训练走这条。
- 通用路径：`simulation/engine.py::run_stage`，本次修复了初始化、边界与保存。
- 分叉判定已收敛到 `dispatch.execution_service()`，但两条循环尚未合并。合并需要把 `motion_backend.run` 的阶段循环改为调用共享 `run_stage` 的钩子，并保持 V4 环境与奖励不动；本轮未做，因为它会改动正在运行的同一份实现语义。

### 仍未完成

1. 两条训练循环未合并（见上）。
2. 评估仍是单环境、条件间不隔离、结果按kind覆盖；`evaluate` 的独立条件矩阵未修。
3. Isaac `step()` 仍丢弃 terminated/truncated/reset；`_supports_extras` 仍用 `inspect.signature`。
4. 蒸馏留出集仍重新划分；Teacher/Student 仍共用同一 `job.model` 构造。
5. 未安装 Ruff，未运行格式化。
6. 主工作树未应用本阶段修复；活动训练继续用旧实现。


## 第四阶段（本轮）：接线、评估隔离、蒸馏边界、轮换一致性

### 修复

| 问题 | 修复 | 验证 |
|---|---|---|
| 分派只接受不执行 | `worker._execute_stage()` 按 `dispatch.execution_service()` 选择：`production_motion`→`motion_backend.run`，`generic`→`run_stage`；不支持组合在创建run前拒绝 | `test_worker_executes_production_service_for_runtime_template`（替身Isaac边界，其余走真实分派）、`test_worker_uses_generic_engine_for_cpu_jobs`、`test_unsupported_service_is_refused_before_execution` |
| Isaac适配器要求legacy_spec_path | `available()`与`execution_service()`都以`runtime_template`为主；`legacy_spec_path`只作历史spec只读入口；两者都没有时明确拒绝 | `test_active_config_is_accepted_by_single_dispatch` |
| 环境结束语义丢失 | 构造时确定一次接口形态；`terminated/truncated/dones/reset_envs/time_outs/terminal_observations`分别给出，不再全零 | `test_isaac_step_preserves_termination_and_reset`、`test_isaac_step_handles_dones_only_environments` |
| 评估条件串扰、用训练环境数 | 每条件新环境并关闭；使用`evaluation.envs/seed`；不支持的条件参数直接拒绝 | `test_evaluation_isolates_conditions_and_uses_declared_envs`（2条件→2次构造2次关闭，failures不继承） |
| 同kind评估覆盖历史 | `evaluations/<evaluation_id>/result.json`，索引改为追加并保留`latest`指针；CLI按`protocol.capability_judged`展示 | `test_two_evaluations_of_same_kind_are_both_kept` |
| Teacher用Student架构猜 | Teacher从checkpoint的`network_architecture.actor_hidden_sizes`构造，缺失则要求显式声明；大Teacher/小Student可执行 | `test_distillation_supports_large_teacher_and_small_student` |
| 留出集重新划分 | 划分写入记录；`holdout_loss(job, 记录)`读取同一切分与同一标签来源 | `test_holdout_reads_saved_split_instead_of_resplitting` |
| 轮换丢掉被跳过记录 | `prune()`只移除真正删除的登记项；缺失/身份不符/外部/软链接的记录保留，`plan.skipped`写明原因 | `test_skipped_records_stay_in_inventory` |
| 每次保存重复写同一更新 | 公共驱动器记录`last_saved_update`，同更新不重复落盘 | `test_final_save_covers_non_periodic_end`（5次更新→`model_update_5.pt`，无重复） |

### 执行路径现状

- 判定：`dispatch.execution_service(job)`；计划、`--dry-run`、worker 都读它。
- 通用路径：`driver.run_stage` + `GenericPPOAdapter`。
- 生产路径：`motion_backend.run(context, stage=..., resume_from=...)`，内部仍自管 rollout/update 顺序，
  但额度、计费、保存、边界使用同一组 `context` 服务，并返回统一的 `StageResult`。
- 两条循环尚未合并成一个函数；原因与影响写在 ARCHITECTURE「已知未合并」。

### 本轮测试

```bash
cd /home/kellen/wheel_leg_refactor_ws
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/test_new_{cli,config,artifacts,encoding,lifecycle,learning}.py \
  tests/test_phase2_fixes.py tests/test_phase3_fixes.py
# 118 passed

/home/kellen/ruffenv/bin/ruff check src/native_stand/{simulation,experiments,artifacts,learning,tasks,integrations} src/native_stand/cli.py
# All checks passed
/home/kellen/ruffenv/bin/ruff format --check <同一组路径>
# 全部已格式化
```

### 仍未完成

1. 生产Motion循环与公共驱动器未合并成一个函数。
2. 真实物理验证未执行：Isaac适配器、评估环境、`motion_backend` 均未在仿真或实机运行。
3. 主工作树未接入本轮修改。
4. `learning/ppo/supported.py` 的 B023（循环内闭包）按既知风险临时忽略，改动会变更数值行为。
5. 剩余历史目录（`students/`、`command_v6/`）的重复helper未批量迁移；它们仍被冻结运行与未迁移脚本引用。


## 第五阶段（本轮）：生产路径接入公共驱动器、评估与蒸馏边界收口

| 问题 | 最终行为 | 验证 |
|---|---|---|
| 生产代码与 RunContext 契约不符（`record(force=True)`、`save` 缺 `stage`、`startup_verified`） | 生产循环改为 `_Adapter` + `driver.run_stage`；计费、保存、边界、启动确认都由驱动器执行，三个旧调用全部消失 | `test_production_adapter_uses_real_context_and_driver`（真实 RunContext + 真实驱动器 + 真实保存）、`test_production_incomplete_rollout_is_not_charged` |
| 生产路径忽略 `resume_from`，总读配置初始化 | `_resolve_source(job, resume_from)`：显式 resume → 配置声明；并在入口校验 SHA 与存在性 | `test_production_resume_source_priority`、`test_production_missing_initialization_is_rejected` |
| 评估仍用通用Isaac工厂 | `backend_for_evaluation()` 按 `execution_service()` 取模块；生产配置取 `motion_backend.make_env`，其环境构造与训练共用 `_build_environment()` | `test_evaluation_routes_to_production_module_not_generic_adapter`（生产配置不再报 `legacy_spec_path`） |
| 非 command 条件被静默忽略 | 条件经后端 `apply_condition` 钩子应用；生产端只支持 `command`，摩擦/延迟明确报错而不是假装完成 | `test_evaluation_isolates_conditions_and_uses_declared_envs`、`motion_backend.apply_condition` 的未知字段拒绝 |
| timeout 未接算法 | 通用适配器 `_stored_reward()`：`truncated & ~terminated` 用 reset 前终态 Critic 价值补一次；随后向 `process_env_step` 传空 extras，避免二次修正 | `test_timeout_bootstrap_uses_terminal_value_once`（可手算 1+0.9×2=2.8）、`test_timeout_bootstrap_is_not_repeated_by_algorithm` |
| 蒸馏 latest 与 best 拼错 | 训练结束先快照 latest（最后一轮 Student+Adam+RNG），best 快照带同轮 Adam/RNG；两者分别保存；seed 在构造网络前设置 | `test_distillation_latest_state_matches_last_round_not_best` |
| 数据/Teacher/候选被替换仍继续 | `holdout_loss()` 校验 dataset、teacher、candidate 三段 SHA | `test_distillation_rejects_changed_dataset_identity` |
| `config.py` 格式未通过 | Ruff format 后通过；`supported.py` 的 B023 改为显式绑定默认参数，删除忽略项 | `ruff format --check`、`ruff check` 全部通过 |

### 当前执行链

```text
wheel-leg train --config FILE
  → JobConfig（一次解析）
  → dispatch.execution_service(job)
      ├─ production_motion → motion_backend.run(context, stage, resume_from)
      │                        → _Adapter → driver.run_stage（计费/保存/边界/关闭）
      └─ generic           → engine.run_stage → GenericPPOAdapter → driver.run_stage
  → experiments/worker._execute_stage（同一解析结果）
  → run 目录状态、指标、checkpoint、报告
```

### 验证命令与结果

```bash
cd /home/kellen/wheel_leg_refactor_ws
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES='' \
  /home/kellen/anaconda3/envs/issaclab/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_new_cli.py tests/test_new_config.py tests/test_new_artifacts.py \
  tests/test_new_encoding.py tests/test_new_lifecycle.py tests/test_new_learning.py \
  tests/test_phase2_fixes.py tests/test_phase3_fixes.py
# 127 passed in 26.77s

/home/kellen/ruffenv/bin/ruff check <现代模块> tests/test_phase3_fixes.py tests/test_new_learning.py
# All checks passed
/home/kellen/ruffenv/bin/ruff format --check <同一组路径>
# 52 files already formatted
```

### 仍未完成

1. 真实物理验证未执行：生产适配器、`prepare_session`/`make_env`、`apply_condition` 的摩擦/延迟分支都未在 Isaac 中运行。
2. 蒸馏尚未接入 run/attempt 与 `resume --run`（仍按独立输出目录保存 latest/best）。
3. 主工作树未接入本轮修改；主树已出现新课程与 80 rad/s 输出版本，合并需按文件复核。
4. 生产环境的 `friction`/`delay` 条件当前明确拒绝，需要任务侧提供可覆盖接口后才能支持。


## 第六阶段（本轮）：生产更新函数、恢复计数、保存失败与评估接口

| 问题 | 最终行为 | 验证 |
|---|---|---|
| 生产 `__step` 在 update 100 抛 `UnboundLocalError: state`，课程块重复两遍 | `state` 声明 `nonlocal`；删除重复的第二份课程/首更新块；课程触发改用**累计完成更新数** `context.updates_used + 1`，诊断编号同源 | 源码只有 1 处 `% 100` 与 1 处 `first_update.json`；`test_production_*` 系列 |
| 正常续训 checkpoint 被拒（来源基数 454、恢复累计 455） | 新增 `_check_source_progress()`：首次初始化要求 `online == historical`；resume 时要求 `online - historical == counters.new_run_updates`；两者都校验原始在线上限，不再一律按首次初始化判断 | `motion_backend._check_source_progress` 单元验证（首次/续训/越限三种分支） |
| 全部保存失败仍 COMPLETED 并跨阶段 | 驱动器 `save_now()` 只有拿到引用才推进；`SaveFailed` 抛出后 worker 置 FAILED、退出码非零、不启动下一阶段；已完成更新与消耗保留 | `test_save_failure_fails_run_and_blocks_next_stage`（FAILED、`updates_used>=1`、attempt 记录一致、无 formal 目录） |
| 生产评估工厂返回原始V4环境，评估器接口不符；split 用 `"eval"`；AppLauncher 不关 | 新增 `ProductionEvalEnv`：`observations()`/`step()`（tuple→统一字典，含 terminated/truncated/terminal_observations）/`set_command()`/`close()`（拥有并关闭 env+AppLauncher）；环境来源改用 `holdout`；命令经 `apply_command()` 写入任务命令缓冲 | `test_evaluation_routes_to_production_module_not_generic_adapter`；CPU 条件隔离用例 |
| 预计算标签无来源仍被接受 | 数据集需带 `label_provenance`（teacher SHA、输入/动作版本、单位、维数）；缺失或不匹配在训练前拒绝；在线标签使用实际 Teacher 身份 | `test_precomputed_labels_without_provenance_are_rejected`、`..._with_provenance_are_accepted` |
| shuffle 序列不可恢复 | `distillation_latest.pt` 保存 `shuffle_rng`（generator state）与 best 摘要 | 同上用例断言 `shuffle_rng` 存在 |
| 轮换：非法SHA登记项被删、计划后被替换仍删 | 候选要求 64 位 hex SHA 且与文件一致，否则跳过；删除前复核路径归属、软链接与登记 SHA，失败记 `identity_changed`/`not_owned` 并保留记录 | `test_prune_*`、`test_skipped_records_stay_in_inventory`，测试改用真实 SHA |

### 验证

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python -m pytest -q -p no:cacheprovider \
  tests/test_new_cli.py tests/test_new_config.py tests/test_new_artifacts.py \
  tests/test_new_encoding.py tests/test_new_lifecycle.py tests/test_new_learning.py \
  tests/test_phase2_fixes.py tests/test_phase3_fixes.py
# 129 passed

/home/kellen/ruffenv/bin/ruff check  <现代模块 + 三个测试文件>   # All checks passed
/home/kellen/ruffenv/bin/ruff format --check <同一组路径>        # 49 files already formatted
```

### 仍未完成

1. **真实物理验证未执行**：生产 `run()`（含课程推进）、`prepare_session`/`ProductionEvalEnv`、`apply_command` 未在 Isaac 中运行；生产 step 无 CPU 执行路径，因此 99/100/101、199/200/201 的运行时行为只能靠源码结构与驱动器用例覆盖，未做真实运行验证。
2. **墙钟跨 attempt 累计**：`wall_budget_left()` 仍按当前 context 的 `_started` 计算，阶段/attempt 切换会重新获得额度；本轮未改。
3. **蒸馏未接入 run/attempt 与 `resume --run`**：仍是独立输出目录；BC 轮次未进入运行账本。
4. **`context.runtime_state` 未交给环境恢复**：通用路径把 `runtime_state` 写进 checkpoint，但恢复只重建算法，未调用环境 `load_runtime_state`。
5. **生产适配器仍用嵌套 `__step` 闭包**：构造仍在 `run()` 内，`_Adapter.build()` 是空实现；未改成实例属性的完整 adapter 类（避免在没有物理验证的情况下移动 V4 数值代码）。
6. 主目录未接入；主树已有新课程与 80 rad/s 版本，需逐文件复核合并。

## 架构约束（2026-11-08，本轮新增）

- `scripts/check_architecture.py`（纯标准库AST，--root 支持fixture，退出码 0/1/2）：
  R1 正式源码禁止新增并行入口与 owner 之外的训练进程创建（`experiments/owner.py` 是唯一合法点）；
  R2 依赖方向（tasks↛cli/owner/report、learning↛cli、simulation↛integrations、artifacts↛tasks、正式源码↛scripts；`experiments.config` 数据类型放行）；
  R3 禁止通过导入别名改写项目模块的全局实现对象（如 `workflow.DelayedPlant = ...`），显式 factory 参数是唯一合法方式。
- 基线：当前候选扫描 **0 条例外**，未创建 `architecture_baseline.json`（无例外不建文件）。同文件新增违规或允许数量超额都会失败。
- CI：复用主项目 `.github/workflows/cpu.yml` 与 `scripts/test_portable_cpu.py`（原样复制，未另建平行入口）；workflow 增加架构检查步骤；新测试 `tests/test_architecture_rules.py`（FROZEN_TEST_SCOPE=PORTABLE）已被选择器实际选中。
- 测试：`tests/test_architecture_rules.py` 8 项（合法factory通过、绕过owner失败、并行入口失败/分析入口通过、绝对/相对/别名导入、跨模块改写失败/对象属性赋值通过、基线允许+新增失败、错误root退出码2、检查器零运行依赖）。
- 回归：**160 passed**（原152 + 架构8）；Ruff check/format 对新文件通过。

### 主目录差异与切换要求（当前事实）

- 主项目相对候选新增（必须保留，不能覆盖）：`progressive*.py`、`output_migration.py`、`calibration_geometry.py`、`movement_evidence.py`、015/80 动作版本、movement 分项统计；CI 文件本轮已原样并入候选。
- 原迁移阻塞已清零：上一轮验收清单（CLI 回执、停止恢复、轮边界保存、故障结算、身份核对、评估回归）全部通过（152 项），本轮未引入回归。
- 切换前剩余：①候选与主树的功能文件逐项合并（保留主树新课程/80rad/movement，采用候选的生命周期与检查器）；②合并后跑同一 160 项回归 + 架构检查 + portable 选择器；③真实物理验证仍未执行；④在训练 owner 安全窗口完成主目录接入并核实来源、依赖、入口、运行状态。

## 集成候选状态（2026-11-08 第二轮更新，当前事实）

**回归：163 passed**（原152 + Recovery/结算生命周期1 + 架构10）；架构检查 59 文件 0 违规；portable 集合 **1448 collected, 0 收集错误**，执行 **1421 passed / 4 failed / 23 skipped**。

### 本轮修复（先红后绿）

| 缺陷 | 修复 | 证据 |
|---|---|---|
| Recovery 得到 None 上下文（`return context` 落在 motion 分支内） | 移出分支；Motion 语义记录保留在分支内 | `test_recovery_context_and_init_failure_settlement`（真实 `_new_context`，不 mock） |
| reserve 后初始化失败残留 RUNNING attempt | `_finish_new_context` 拆分：reserve 后任何初始化异常 → 只结算本次 attempt（FAILED）→ `context_init_failure.json` 保留原始异常与堆栈；BC 侧按 attempt 状态防重复结算 | 同上（Motion/Recovery 两条路径的注入失败均无 RUNNING 残留，其他 attempt 不被改写） |

### 检查器修正（tests/test_architecture_rules.py 10 项全绿）

相对导入按文件包路径+level 解析（`from ..cli import main` → native_stand.cli 违规；`..experiments.config` 放行）；进程 API 按 import 绑定识别（`Popen as launch`、`Process as WorkerProcess`、`import subprocess as sp` 全覆盖，`buffer.spawn()` 不误判）；入口白名单用准确包内路径（learning/worker.py 不再继承权限）；R3 复用同一导入解析+项目模块索引（相对导入覆写同样拦截）；baseline 精确身份 = 路径+规则+限定符号+规范化表达式+允许次数，测试证明：登记通过、同问题加次失败、删除旧问题后新增不同表达式（另一符号）失败。

### CI 闭合

- `requirements-cpu.txt` 已从主项目并入（workflow 引用不再缺失）；Ruff 检查范围覆盖本轮治理脚本与新脚本。
- 8 个迁移回归文件未盲标 PORTABLE：其中 7 个已自包含（tmp fixture+真实代码），`test_phase3_fixes.py` 中 1 项依赖本机活动配置（host 证据核验），按"可移植契约与主机证据分开"保留原分类——未用 skip 掩盖。
- 复制主项目 `scripts/`（235 个，含测试引用的分析/审计脚本）、`deployment_protocol/motion31_reference/reference_adapter.py`、`isolated/2026_11_05__motion_interface_research/build_centered_design.py`；portable 收集错误 9→0。

### 主项目能力整合对照

| 能力 | 状态 |
|---|---|
| progressive_course/episodes、output_migration、calibration_geometry、movement_evidence、simulation/isaac/progressive | **已接入候选**（原样并入，59 文件架构扫描 0 违规） |
| 仓库脚本 235 个 + 部署协议 reference_adapter + isolated 构建脚本 | **已接入候选**（portable 收集 0 错误的前提） |
| CI 文件与 requirements-cpu.txt | **已接入候选**（原样复制+架构检查步骤） |
| `worker.py`/`motion_backend.py`/`dispatch.py`/`tasks/motion` 语义级合并（新课程×80rad×候选生命周期） | **尚未接入**：需逐函数合并，影响：生产 run 行为验证只能在主树进行 |
| `students/` 证据链（smoke_certificate 依赖的 students 包其余模块） | **尚未接入**：`test_smoke_engineering_does_not_require_standing_skill` 在候选失败（training_evidence.py 与主树一致，但证书依赖链有更深的版本差） |

### portable 剩余 4 失败（历史集合，非本轮迁移回归）

1. `test_motion_differential`：依赖历史 run 产物 `runs/.../chain_terminal_report.json`（主机证据，按要求不复制运行数据）。
2. `test_motion_handover_visual`：调用 `play_motion31_mujoco.sh` 的 MuJoCo 回放（物理回放边界）。
3. `test_motion_mcu_playback`：测试与脚本源码断言的版本偏斜（双方均来自主树，需在主树核对）。
4. `test_standing_survival_training`：students 证据链依赖差（见上表）。

前三类属主机证据/版本偏斜，第四项列入 students 整合清单；均不阻塞本轮"集成候选完成"判定，但阻塞最终切换。

### 距最终切换

1. 语义级合并：worker/motion_backend/dispatch/tasks/motion（新课程+80rad+候选生命周期）。
2. students 证据链同步（消除 portable 第4项失败）。
3. portable 剩余 3 项主机证据用例在主树环境核验。
4. 真实物理验证未执行；在训练 owner 安全窗口完成主目录接入。

## 集成候选状态（2026-11-09 执行路径整合轮，当前事实）

**迁移回归：168 passed / 0 failed**；架构检查 72 文件 0 违规（新增 Recovery 后端链+Motion spec 更新）。

### 本轮已接通（真实调用路径验证）

| 项 | 验证 |
|---|---|
| Motion `2026_11_8-22_20__motion_C0_yaw1to4_full_KL10000`（015/80、multiheight、full 初始化） | 候选 `tasks/motion/spec.py` 更新为主树版本（SUPPORTED_OUTPUT_VERSIONS 含 015_40/015_80/multiheight+分段支撑表校验）→ `build_task` 构造 MotionTask ✓ |
| Recovery `2026_11_9__recovery_Swing_Motion_handover10000_attempt03`（isaac_recovery） | 补齐候选缺失依赖链（cubic_task、recovery_cubic、recovery_isaac/lifecycle+execution+handover_lifecycle、handover_context、reward_context）→ backend 可导入、RecoveryTask 构造 ✓ |
| 共享执行链 | registry→config→dispatch→build_task 对两条活动配置全通；架构检查 0 违规 |

### 性能：已实测 / 仅源码定位 / NOT_RUN

**已实现（CPU 语义验证）**：
- `simulation/engine.py`：每 policy step 的 `float(rewards.mean())` GPU回读改为设备端累计（None守卫），指标边界一次性转标量；reward_mean 口径不变（本轮均值）。phase2/3 相关 61 项通过。

**仅源码定位（未实现，GPU A/B NOT_RUN）**：
1. Motion reset 128槽历史 Python 循环填充 → 批量张量化：参考实现位于主目录 `isolated/2026_11_05__motion_supported2048_dqgain_ppo/runtime_view_r4/src/native_stand/students/{motion_timing_runtime,motion_mcu_runtime,fullchain_runtime_impl}.py`。前置条件：先把冻结 runtime 依赖链引入候选并建立逐位一致 fixture（tick 跨界/部分reset/非零q,dq），再优化。CPU 探索称公式可逐位一致，但候选实现未验证。
2. Motion 观测/Critic 多层重复编码：需先区分副作用与纯编码（历史/噪声/时钟推进 vs 状态编码），缓存键需区分 physics tick 与部分 reset 失效。
3. Recovery `physical_profile()` 每观测全环境遍历+列表建GPU张量 → 缓存：需先与原实现做完整输出对比（含 DR/restore 失效路径）。
4. Recovery transition deepcopy 与终态 Critic 重复 clone：需先列消费者清单。
5. fine_ring 254MiB：先审计消费方（故障证据 vs 训练/恢复），本轮不裁剪完整恢复状态。

### NOT_RUN 清单

- GPU A/B 对照（基线 vs 优化版，5轮预热+20轮计时）：未获匹配授权与预算，未运行。
- 真实 Isaac 物理验证：未执行。
- 128槽历史批量化的逐位一致 fixture：未实现（依赖 runtime 链引入）。

### 活动任务保护核对

两条链 worker 存活（Motion 348863、Recovery 566821 及派生进程），源码根/软链接/run/账本/冻结依赖全程只读；本轮复制均为主目录→候选方向；无信号、无 Git 操作。



**迁移回归：168 passed / 0 failed**。两条活动链配置在候选中验证通过：
- Motion `2026_11_8-22_20__motion_C0_yaw1to4_full_KL10000`（015/80、full 初始化）→ 解析 ✓、`execution_service`=production_motion ✓、registry 支持 ✓
- Recovery `2026_11_9__recovery_Swing_Motion_handover10000_attempt03`（isaac_recovery 后端）→ 解析 ✓、分派到 `recovery_isaac.backend` ✓、`RecoveryTask` 构造 ✓
- 虚构技能（jump，测试专用注册）全链通过：registry→config→factory（8维观测、actor_obs/critic_obs/valid 键名映射）→driver 真实 optimizer.step（参数变化断言）→计费/保存/close→逐环境成功/失败区分

### 本轮关键修复

| 项 | 修复 |
|---|---|
| factory 映射缺陷 | action_valid 内部名= `recovery_actor_valid`（原映射遗漏导致 build_storage 报缺列）；不再用 `type(ppo)({**ppo})` 重建算法对象，改实例属性替换保持 nn.Module 语义 |
| isaac_recovery 后端 | BACKENDS 注册 + registry 支持声明 + 分派验证（主项目 Recovery 链配置可直接分派） |
| config 任务名单 | 两处硬编码 ("motion","recovery") 全部改为 registry 派生 |
| 虚构技能测试 | registry 保存/恢复（不污染生产表）、真实 optimizer.step+参数变化断言、逐环境成功/失败 |

### 未完成（如实）

1. 语义级合并 worker/motion_backend/dispatch/tasks/motion 的其余差异（新课程/movement 统计的调用路径在候选已纳入源码，CPU 无执行路径）。
2. students 证据链同步（portable 的 standing_survival 项）。
3. portable 4 项主机证据/版本偏斜失败（differential=历史 run 产物、handover_visual=MuJoCo 回放、mcu_playback=版本偏斜、standing_survival=见上）。
4. 蒸馏多教师路由：registry/factory/配方结构已具备，两教师路由实现与测试未完成。
5. 真实物理验证未执行；主目录未切换。



**迁移回归：167 passed / 1 failed**（唯一失败=fictional 技能测试的 assembly 映射缺陷，精确根因见下）；架构检查 63 文件 0 违规；架构测试 12 passed。

### 本轮修复（已验证）

| 项 | 修复 | 证据 |
|---|---|---|
| 初始化三模式 | `InitializationConfig._validate_mode()`：scratch/full/actor_only 真实分支（actor_only 要求 checkpoint+SHA）；跨节重复条件合并 | `test_new_config` 全绿（含 actor_only 拒绝/接受边界） |
| dispatch 注册表化 | 硬编码 TRAINING 表删除；`_training_combinations()` 从 registry 派生；`backend_for`/`service_description` 消费同一来源；`worker.build_task` 走 `build_task_from_registry` | `test_world_model_training_is_refused`（恢复，引用新 API）；config 拒绝未登记任务 |
| registry 结构 | build_symbol/evaluator_symbol/supported_algorithms/supported_backends/observation_keys；`build_task_from_registry`/`observation_keys` 正式入口 | 架构检查 63 文件 |
| policy factory | 任务声明键名 → 冻结内部键名的 canonical 映射（装配边界+推理边界包装）；`ModelConfig.kind` 声明字段（入 run record） | fictional 测试断言 `observation_keys["actor"]=="actor_obs"` 通过 |
| 评估逐环境语义 | `task_success/task_failure` 逐环境；成功终止→SUCCESS、明确失败→FAILED(reason)、无证据终止→terminated_unknown；不再用批次级布尔 | fictional 测试 env0/env1 同批次区分断言 |

### 未完成（1 项，精确根因）

`test_fictional_skill_uses_shared_lifecycle`：factory 的 canonical 映射已把任务键名改名后传给 `assemble_ppo`，但 `build_storage` 仍收到任务键名样例（"观测缺少student_actor列"）。嫌疑：`assemble_ppo` 内部对 observations 的二次引用/复制路径。这是**最后一处**新技能接口缺陷；修复后 fictional 测试即为完整绿色。

### CI 实际状态

- 选择列表包含全部 9 个迁移/架构文件 ✓
- Ruff 已入 workflow（ruff==0.8.4）；治理脚本与新模块 lint 通过
- portable 完整执行的 4 项主机证据/版本偏斜失败保持上轮分类，未用 skip 掩盖
- 本轮未重新运行完整 portable（以迁移回归+针对性测试为准）



**portable 集合（含迁移回归）：1579 passed / 4 failed / 23 skipped**（4 失败为主机证据/版本偏斜的历史用例，见下）；**架构检查 63 文件 0 违规**；架构测试 12 passed；Ruff check/format 对治理脚本与新模块通过。

### 已修复并验证（本轮）

| 缺陷 | 修复 | 证据 |
|---|---|---|
| Recovery 得到空上下文 | `return context` 移出 motion 分支 | `test_recovery_context_and_init_failure_settlement`、`test_recovery_worker_completes_on_cpu`（Recovery CPU worker 真实执行 COMPLETED、消耗一致） |
| metrics 创建后初始化失败未关闭 | `_finish_new_context` 失败路径 `metrics.close()`；清理异常作附加信息不覆盖原始错误 | `test_init_failure_closes_metrics_and_keeps_other_attempts`（close 实际发生；既有 run 的账本/状态逐字节不变——真实既有 run 对照，不再是同账本自比较） |
| 检查器漏检相对导入/别名进程API/顶层scripts | 相对导入按 Python 包规则解析（level-1，空 module 不多上移）；进程 API 按 import 绑定识别（含 `Popen as launch`、`os.fork`）；多成员逐个判定；`scripts.xxx` 拦截 | `tests/test_architecture_rules.py` 12 项（含审阅全部反例：`from .. import cli`、`from native_stand.experiments import owner, config`、`import scripts.analyze`、`os.fork()`、同函数换命令仍失败） |
| baseline 身份不完整 | 精确身份 = 路径+规则+限定符号（支持嵌套/方法链）+完整 `ast.dump`（不含行号）+允许次数；检查器输出 `identity: <json>` 行 | `test_baseline_identity_includes_full_ast_and_symbol`、`test_baseline_binds_symbol_and_expression` |
| smoke 证书 "critic warmup did not update" | 根因：fixture 缺少当前契约要求的 BC/std 证据字段（`BC_loss/BC_gradient_norm/BC_updates/std_unchanged`）；实现规则本身正确，补齐 fixture 真实表达 | `test_standing_survival_training` 9/9 中该项通过（py 主树同版本） |
| 播放脚本硬编码主目录 | `ROOT` 从脚本位置推导；解释器由 `PY` 环境选择；dry-run 只生成命令 | `test_motion_handover_visual` 在 PLAY_DRY_RUN=1 下独立复跑通过 |

### 主项目能力接入（三种状态）

| 能力 | 已纳入源码 | 已接通调用 | 已通过验证 |
|---|---|---|---|
| actor_only 初始化（C0 协议 Isaac Motion） | ✅（config 门控 + `learning/ppo/actor_initialization.py`） | ✅（`_validate_source_mode`；活动配置可解析） | ✅（`test_new_config` 回归；CPU 解析与计划生成） |
| progressive/几何/movement 模块 | ✅ | 部分（主树生产路径消费，候选 CPU 无执行路径） | 架构/收集验证 |
| 语义级合并 worker/motion_backend/dispatch | ❌（尚未接通，影响：生产 run 行为只能在主树验证） | ❌ | ❌ |

### 新技能扩展点（工程测试证明）

- `tasks/registry.py`：静态任务表（task ID→构造/评估/算法/后端组合）；config 与 worker 均消费注册表（`build_task` 走注册表动态导入），新增任务=登记模块，不复制生命周期。
- `learning/policies/factory.py`：策略构造与技能名称解耦；观测样例由任务环境提供（不猜测键名/物理单位）；非 MLP 明确拒绝。
- 评估成功语义：`success_on_terminate` 任务标记 → terminated 评为成功而非失败；Motion 窗口/Recovery 保持时间语义不变。
- **虚构技能测试**（`test_fictional_skill_uses_shared_lifecycle`）：jump 任务（测试专用注册）以 8 维观测、共享驱动器、真实 optimizer update、计费/保存/close 全链通过；成功终止评为成功。仅接口工程测试，非跳跃物理能力。

### CI 实际覆盖

`.github/workflows/cpu.yml`：架构检查 → Ruff check/format（明确依赖 ruff==0.8.4）→ `scripts/test_portable_cpu.py`（**选择列表包含全部 8 个迁移回归文件**；portable 执行 1579 passed，其中迁移回归全绿）。

### portable 剩余 4 失败（主机证据/版本偏斜，非本轮迁移回归）

`test_motion_differential`（历史 run 产物）、`test_motion_handover_visual`（MuJoCo 回放边界）、`test_motion_mcu_playback`（测试↔脚本版本偏斜，需主树核对）、`test_standing_survival_training`（students 证据链版本差，同上）。

### 距最终切换

1. 语义级合并 worker/motion_backend/dispatch/tasks/motion（新课程+80rad+候选生命周期）。
2. students 证据链同步 + portable 剩余 4 项在主树环境核验。
3. 真实物理验证未执行；在训练 owner 安全窗口完成主目录接入。

## 集成候选状态（2026-11-09 执行路径轮，当前事实）

**迁移回归：184 passed / 0 failed**（9 个迁移文件 + 架构 12 项 + smoke 9 项 + handover 6 项）；架构检查 76 文件 0 违规；CI 声明范围 **Ruff check "All checks passed" / format 全部格式化**（冻结选择器 test_portable_cpu.py 按 pyproject 冻结排除收窄 CI 点名；本轮并入的主树模块 B905/B007 以 strict=False 保持原语义修复）。

### 本轮完成

| 项 | 状态 | 证据 |
|---|---|---|
| 检查器反例补全（`from .. import cli`、多成员导入、`scripts.xxx`、`os.fork`）+ baseline 精确身份（短描述+完整 ast.dump，`identity:` JSON 行输出） | 完成 | `test_review_required_patterns`、`test_baseline_identity_includes_full_ast_and_symbol`（同函数换命令仍失败）；检查器测试 12 项全绿 |
| 播放包装器可移植性 | 完成 | `ROOT` 从脚本位置推导、`PY` 显式环境选择（缺省时 python3→python 回退、明确报错）；dry-run 只生成命令（`--viewer`/`--panel --panel-host` 进 stderr，不启动依赖检查）——`test_motion_handover_visual` 6 项通过 |
| smoke 证书 fixture | 完成 | fixture 按**当前契约**补齐 `BC_loss/BC_gradient_norm/BC_updates/std_unchanged/actor_changed_this_update/critic_changed_this_update` 与 `critic_warmup_updates=5` 声明（实现规则正确，fixture 缺证据）；不删 warmup 检查、不硬编码 PASSED——9 项通过 |
| 报告错误归因修正 | 完成 | handover_visual 不再标注为"必须物理回放"（PLAY_DRY_RUN=1 只生成命令）；已从 portable 失败清单移除 |
| Motion 015/80 + Recovery isaac_recovery 生产路径 | 完成 | 活动链 run.json→JobConfig→build_task→execution_service 全通（MotionTask/production_motion、RecoveryTask/recovery_isaac.backend）；registry/dispatch/config 消费同一任务表 |

### portable 剩余失败（主机证据/版本偏斜，逐项保留分类）

`test_motion_differential`（历史 run 产物）、`test_motion_mcu_playback`（测试↔脚本版本偏斜，需主树核对）、`test_standing_survival_training` 的 students 证据链项已随 fixture 契约补齐消除。

### GPU A/B 与物理验证：NOT_RUN

128 槽历史批量化、Recovery physical_profile 缓存、Motion 观测去重为源码定位项（参考 `isolated/2026_11_05__motion_supported2048_dqgain_ppo/runtime_view_r4/src/`）；未实现逐位一致 fixture 前不动实现。

### 距最终切换

1. students 证据链与其余 portable 主机证据项在主树核验。
2. worker/motion_backend/dispatch/tasks/motion 语义级合并（新课程×80rad×候选生命周期）。
3. 真实物理验证未执行；训练 owner 安全窗口完成主目录接入。

## 当前结论（2026-11-09 Recovery公共执行路径轮，唯一有效摘要）

**核心回归：191 passed / 0 failed**（含新增 `tests/test_recovery_integration.py` 7 项）；架构检查 76 文件 0 违规；CI 范围 Ruff check/format 全部通过。历史章节均为历史记录，与当前状态冲突时以本节为准。

### Recovery公共执行路径CPU工程闭环（本轮主交付）

实际接线（全部经真实对象验证，非分派字符串）：

```
原始Recovery job → load_job_config ✓（isaac_recovery经registry合法集合接受）
→ RecoveryTask ✓ → dispatch generic service ✓
→ GenericPPOAdapter.build → registry声明的policy_factory
→ recovery_cubic.build_policy（真实合同+job参数）
→ PhysicalActor（physical_mean/physical_std/固定物理std）
→ actor_only来源加载（load_actor_source + SHA边界核对）
→ ControlledPPO真实CPU更新（warmup期Actor不变、Critic更新；warmup后Actor 12次优化步）
→ capture（含物理std/RNG/counters/合同）原子保存
→ 新对象restore → 继续训练与连续训练逐位一致
```

| 修复 | 根因 | 文件 |
|---|---|---|
| 原始配置被拒 | config后端白名单未从registry派生 | `experiments/config.py`（白名单改为registry并集；未知后端明确拒绝） |
| 普通Actor被创建 | GenericPPOAdapter直接调assembly.build_actor；registry的policy_factory声明丢失 | `simulation/engine.py`（factory分支：经registry进入，恢复由payload形态决定）、`tasks/registry.py`（重新登记） |
| PhysicalActor(job) TypeError | 需要合同dict而非JobConfig；assemble未传job参数 | `learning/ppo/recovery_cubic.py::build_policy`（合同读自assets.recovery_training_contract；actor_lr/critic_lr/epochs/minibatches全部来自job.ppo并经optimizer.param_groups与PPO属性验证） |
| actor_only来源 | 无专用加载器（load_full会拒actor-only文件） | `artifacts/checkpoint.py::load_actor_source`（区分Recovery格式直载与C0一次转换） |
| checkpoint保存失败 | recovery capture缺std/rng/counters契约字段 | `recovery_cubic.py::_capture/_restore`（std=固定物理std；RNG在恢复终点设置） |
| actor_only门控拒绝真实组合+重复 | 门控限mujoco且整块重复两份 | `experiments/config.py`（合并为一段；接受isaac_recovery/mujoco+训练合同） |
| 环境输出位置 | —— | 保持context.output由run提供（未发现共享/tmp路径） |

### 验证

- `tests/test_recovery_integration.py` 7 项：公共adapter装配、actor_only哨兵来源、SHA拒绝、warmup→Actor更新+固定std、保存恢复=连续（权重/Critic/Adam state/RNG）、配置与记录一致、非法组合拒绝。
- `test_recovery_worker_completes_on_cpu`：真实worker全路径 COMPLETED（fixture升级为物理Gaussian+训练合同）。
- 主机只读准备检查（真实job与run记录）：解析✓→RecoveryTask✓→generic✓→builder✓→来源SHA✓→合同✓→双路径同一装配✓；未启动现有run。

### NOT_RUN

- 真实Isaac验证（isaac_recovery物理执行、PhysX求解状态）：未执行。
- GUI/播放器、GPU性能对照、共享C核心：后续目标（已确定的顺序）。

## 当前结论（2026-11-10 领域修复轮，唯一有效摘要）

**核心回归：204 passed / 0 failed**（含新增 `tests/test_recovery_domain_fixes.py` 11 项）；架构检查 122 文件 0 违规；CI 范围 Ruff check/format 通过。历史章节为历史记录。

### B1–B8 逐项状态

| 编号 | 状态 | 证据 |
|---|---|---|
| B2 reward形状/有限终点 | **完成** | [N]→[N,1]显式转换；环境数不匹配立即报错（无[N,N]广播）；`infinite_horizon`任务合同门控bootstrap——无限任务[1,10]+γ·[2,50]=[2.8,10]，有限任务保持[1,10]；terminated终点不bootstrap（`test_recovery_domain_fixes` 5项） |
| B3 全池BC/独立prior | **部分完成** | 全池分层采样器迁入（`learning/ppo/physical_anchor.py`，31040条×batch512×高度×dq活动分层，权重和=组数验证）；prior独立于retention（retention=None时prior仍消费，系数0短路不查询）——2项测试 |
| B6 Motion窗口命名空间 | **完成（代码+单元）** | `same_run`改为按恢复载荷run身份判定（不再以resume_from存在性判断）；新run清除consumed_window+sha256（已掌握课程能力保留）；同run重复窗口幂等/冲突报错——2项测试 |
| B1 Swing v2/legacy v7消费者 | **未完成** | 版本解析点未建立；legacy reward v7环境/缓冲区消费者需从主树迁移 |
| B4 结束摘要/工程核验/双目标 | **未完成** | 需Recovery任务适配器钩子与readback重构 |
| B5 新run/恢复/阶段接续完整分离 | **部分完成** | 窗口命名空间已修（B6）；warmup schema迁移、CUDA RNG恢复、域检查未做 |
| B7 依赖登记 | **未完成** | 需在实际读取边界形成统一登记 |
| B8 元数据准确性 | **未完成** | 随B4候选目标一起处理 |

额外仿真采样：0（所有测试使用受控替身或既有训练数据结构）。

### 保护

两条主目录训练链存活（Motion 1542505 [R]、Recovery 1656737 [R]，2026_11_10 runs）；未停止/热改；无Git操作；删除验证未执行（本轮无真实产物删除需求）。

### 距离正式迁移

B1/B4/B5剩余/B7/B8完成后，方可安排有界执行smoke；真实物理/GPU验证另行安排。

### Recovery数据根接续准备（独立worktree，2026-11-12）

此增量基于实际HEAD `16f81c8`，分支`recovery/continuation-reset`，工作树`/home/kellen/wheel_leg_recovery_ws`，不在主工作树修改公共消费者。仅CPU，未启动Recovery或连接远端Motion。旧源最终仍为u1800（`578f22a8…c6ea2`），STOPPED_BOUNDED/max_output_bytes；本段1800、祖先1666、总耗3466、余6534，未发现较新本地Recovery接续。原best候选旧SHA继续UNAVAILABLE，standing u1400与最终完整点分别保护。旧run/账本/候选不回写。

可重建准备材料（在本工作树执行，`--output`须不存在）：

```sh
PYTHONPATH=src /home/kellen/anaconda3/envs/issaclab/bin/python scripts/prepare_recovery_continuation.py \
 --source-run /home/kellen/wheel_leg_native_train/runs/2026_11_11__recovery_preentry48_remaining \
 --config configs/jobs/recovery_preentry48_remaining.json \
 --data-root /home/kellen/wheel_leg_training_data --output artifacts/recovery_continuation/prepared
```

该脚本仅核对源终态/账本/数据根SHA，复用数据根`runtime_paths.json`，生成`job_record.json`与`receipt.json`。不是launcher或训练循环；记录接受`job_from_record`，不直接当preset配置传给`train`。派生记录full初始化、8334本段上限和原PPO参数不变；目标budget需承接used1800，不重新授权8334。尚未确定的GPU UUID/resource policy/target run不得继承旧机器，明确NOT_READY；未正式import、未写来源账本、未创建owner。准确的目标机器值确定后，使用既有`import-checkpoint`填完整manifest、目标run.json及budget，再`resume --max-updates 12`，不是`train`新建任务。12次最多15.36s采样，不等于已覆盖23s自然timeout；自然生命周期证据另记，不能私加额度。

CPU搬迁复用`scripts/recovery_resume_rehearsal.py --path-mapping /home/kellen/wheel_leg_training_data/runtime_paths.json`，checkpoint指向数据根恢复的完整u1800。正式factory/adapter的Actor/909Critic、双Adam/std/CPU RNG、CUDA RNG字节、已完成5次warmup、全部1024行执行历史及课程保存恢复通过，旧根读取禁止；fixture只替换物理I/O。数据根的manifest/ModelIndex保持只读，不为本次测试复制模型到worktree；临时完整checkpoint仅在/tmp。未证明PhysX隐藏现场恢复；fresh-reset须丢弃未完成回合并记录边界，保留已完成统计、课程、优化器及实际预算。真实GPU恢复仍NOT_RUN。

### Recovery reset等价性能增量（独立领域提交）

正式消费者为`RecoveryEnv._reset_idx→ended_snapshot/InitialStates→PhysicalParameters.reset→Execution.reset→physical_profile`。结束摘要合并一次传输，保持env顺序与每env choose/draw/RNG；初态合法模板只代替反复构造，不代替实际几何观测。材质仍用PhysX要求的全形状缓冲+索引；mass→COM→final inertia和实际读回保留。partial reset只刷新受影响32列profile，完整加载/device/dtype变化全失效。Execution.reset只算reset行的V4 pair，不清未reset行队列。fine_ring/909终态/BC/prior消费者未裁剪。

脚本`benchmark_recovery_reset_cpu.py`通过公共Measurement测量1024环境scripted-I/O，5轮中位数：单行8.756→1.904ms，16行9.525→2.265ms，全量82.805→30.052ms。单行标量提取21→7，tolist13→1；全量14343→7、4105→1。实际物理getter每reset5→4，setter顺序/次数不变；新空集合无操作，旧基线空集合IndexError单列。是CPU fixture结果，GPU时间、PCIe传输和整轮收益均未测。原始计数及源SHA见`artifacts/recovery_continuation/reset_cpu_benchmark.json`。

相关总回归273通过（61.13s，3条原ONNX warning）；随后真实BC/prior/正式adapter及紧凑Execution reset增量49通过（7.10s）。覆盖全/partial/空/连续reset、未reset状态和RNG、实际909终态、动作/reward/GAE/梯度/双Adam/RNG严格一致，以及32后续PD/队列tick。架构与Ruff通过。剩余base reset默认姿态写入和再合法姿态写入因尚无PhysX实证不删除；执行历史、材质/COM/惯量核验均保留。
### Motion窗口统计与领域性能计数（2026-11-13）

Motion领域消费公共`Measurement`，不创建第二套计时器。样本记账将Student有效样本、全部transition、当前窗口和累计来源分开；窗口身份不完整时输出UNKNOWN，重复窗口幂等，恢复/partial reset/课程重建不重复消费。候选说明保留资格与排序，只新增覆盖、质量、任务三类原因和实际阈值。轮地材质统计按实际getter/setter、整池返回行数、使用行数、提交缓冲字节和显式host transfer计数；原生API内部传输不推算。CPU无物理回归覆盖之外的新增运行；远端训练未热改。

独立领域分支`motion/window-candidate-observability`，依赖公共`455a198`（含`dd4a228`的Measurement接口）；不复制公共driver/计时器。改动文件：`tasks/motion/{sample_accounting,maneuver_statistics,split_candidates}.py`；`simulation/isaac/{observability,motion_backend,environment,fullchain,response_acceleration,critic_cache,contact_materials}.py`；`simulation/ground_surface.py`；`tests/test_motion_{window_accounting,measurement_consumers,execution_acceleration,production_integration}.py`；本报告与STATUS。旧production fixture仅补真实episode统计对象3行，不格式化历史测试全文。

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='' /home/kellen/anaconda3/envs/issaclab/bin/python -m pytest -q -p no:cacheprovider --basetemp=/tmp/motion_observability_release tests/test_motion_window_accounting.py tests/test_motion_measurement_consumers.py tests/test_motion_split_progression.py tests/test_motion_spin_branches.py tests/test_motion_contact_terrain.py tests/test_motion_speed_contract.py tests/test_motion_rollout_hotpath.py tests/test_motion_production_integration.py tests/test_motion_execution_acceleration.py tests/test_performance_observability.py tests/test_ppo_resume_activation.py tests/test_ppo_efficiency.py tests/test_current_recipe_migration.py tests/test_architecture_rules.py
```

真实结果224 passed in21.16s，无skip；架构208/0；固定Ruff0.8.4对CI范围check与207文件format、三个新增/扩展现代测试check通过。受管小模型数学/真实保存恢复与脚本native I/O属于CPU工程验证；真实PhysX、GPU吞吐、训练优化与远端均未运行，不外推提速比例。
