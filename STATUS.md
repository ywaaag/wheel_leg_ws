# 当前任务

## 重构迁移v1结项（2026-11-13）

公共架构重构、主仓建立及已有功能迁移按冻结CPU验收范围**完成并结项**。本地tag `refactor-migration-v1`固定指向`1cc65e1234839870d24e31eac95214fbf88a0514`；61项交付文件SHA重新核对无漂移，原干净clone核心636通过/10历史skip、portable1920通过/32历史skip保留原身份。本轮治理改动另做静态/CPU验收，不冒用旧测试覆盖新检查器。后续统一从新仓main建立独立分支/worktree；领域未合入增量按正常功能或修复任务推进，不再以模型效果、Jump故障或持续提速判定“重构未结束”。

冻结支持范围包含Motion／Spin已有链、Recovery **v8**/909D/BC+prior与完整恢复，以及Jump **ABI4 MuJoCo CPU**/6370D链、监督聚合、播放/sim2sim/部署和依赖保护。Recovery新v9配方在`recovery/continuation-reset`，Jump Isaac GPU实现位于`jump/isaac-training`（核对时HEAD `876af89`），两者不属于本tag或main既有验收，不把CPU链说成已合入GPU链。各owner继续独立维护其执行版本、授权与能力证据；本轮不合并领域实现、不改活动训练。旧142份蒸馏数据与失效候选仍UNAVAILABLE；未验证物理/平台/能力如实保留，均不是结项前置条件。

开发约定已收敛到根AGENTS和ARCHITECTURE；正式能力须实现、接线、配置/合同、实际消费者测试、当前文档同时交付。迁移报告保留为一次性结项证据；后续普通改动维护当前文档及对应run/artifacts。新增脚本从固定tag做轻量静态增量检查，不整改未修改的历史脚本。

以下为冻结范围与各阶段原始记录，历史PID/余额/“当前”“未完成”按各自日期理解；不据此覆盖上方结项状态。

## Motion领域窗口统计与性能观测增量（2026-11-13）

Motion独立分支已接入公共`Measurement`（公共提交`455a198`）：训练计时仍由公共driver创建并共享给adapter、环境和PPO；领域只记录实际消费者的关闭、构造、首次reset/readback、编译、缓存命中/失效、材质getter/setter、整池行数、实际使用行数和明确的host transfer bytes。计数不读取GPU张量，不新增CUDA同步。`after_save_total`下分别记录candidate引用和课程重建子项。

Motion统计现区分当前完整review窗口的Student有效样本、全部transition与累计计数。窗口身份绑定run、comparison、stage revision和起始update；相同窗口重复导出幂等，冲突更新拒绝；保存恢复、partial reset和课程重建保留累计计数。旧checkpoint缺少窗口起点或完整窗口快照时标为`UNKNOWN`，不猜测窗口样本。

训练候选报告分别记录`coverage_insufficient`、`quality_insufficient`和`task_insufficient`及实际值/阈值；资格和排名规则不变，UNKNOWN不制造失败或best。最终领域/公共边界联合224 passed（21.16s）、架构208文件/0例外、CI范围Ruff0.8.4 check和207文件format通过；完整命令及改动表在迁移报告本节。本分支物理/GPU/远端/真实训练均0，未热改r15。

## 主仓基线与依赖归档（2026-11-12，当前收尾）

公共性能集成完成（2026-11-13）：独立worktree `/home/kellen/wheel_leg_public_ws` / `integration/performance-observability`实现，合入main前先冻结`b60f0d7`做干净clone验收。公共driver/GenericPPO/ControlledPPO/环境共享Measurement，默认无CUDA同步；嵌套子项是CPU墙钟，父项inclusive不能重复相加。启动、rollout/GAE/学习、capture/写盘SHA、课程、候选/引用、旧环境close/构造/reset-readback/编译、retention/metrics/整轮分别记录；计数只读主机已知元数据。MetricsWriter不再丢性能摘要。Recovery批量结束读回/合法初态缓存/partial profile与V4 reset行选择、Jump单job C构建/输入复用/有失效条件的缓存已集成；Motion窗口/累计计数、候选多原因与重建输入归因由运动2交付，未改资格、排名或重建决策。

同一源码快照核心636 passed/10历史skip/3既有ONNX warning（122.43s），portable1920 passed/32历史skip/10warning（125.69s），架构209/0，CI Ruff0.8.4 check与208文件format通过。精确命令、61项改动SHA、失败尝试及最终收据在`artifacts/integration_final/performance_*`，归档到数据根`integration/performance_observability/b60f0d7`；最终文档提交只改摘要，执行源码与已测SHA相同。干净clone禁止旧根读取的四配方dry-run、Recovery真实完整u1800/双Adam/std/RNG/warmup、Jump真实Actor/ABI4/6370D/235字段静态恢复通过，无新物理。Recovery本侧不再有公共迁移代码阻塞；目标资源/来源/授权由其owner绑定，新v9配方不属于本冻结的v8验收。结束单次只读核对Recovery owner独立worker PID2917966/start_ticks150380220，源码`/home/kellen/wheel_leg_recovery_ws/src`；未干预。Motion远端不连接、不热更。Jump现有MuJoCo CPU链已完整集成，无Isaac GPU后端。真实GPU A/B与新源码PhysX运行由各owner后续核验；本公共轮真实训练/物理/GUI/GPU/远端/真实删除0，不包括其他owner自然消耗。

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

Recovery接续准备增量完成，仅新仓库修改。当前v8/Swing-v2/909D/4.8Nm/全池BC+受限prior已验收公共提速消费者；修复候选共享对象重复映射、旧单设备CUDA RNG格式、跨目录候选引用恢复/空池pending/UNKNOWN，补齐阶段几何统计保存恢复。联合252项CPU通过；最终改动针对性70项通过；架构195/0，Ruff通过。u1700及真实最终u1800大模型CPU装配/save/搬迁/load/正式adapter恢复通过，原根目录读取被禁止；GPU仅字节保留、真实执行NOT_RUN。报告与命令见`docs/MIGRATION_20261105.md`本节，机器证据/52依赖/交接模板在`artifacts/cloud_handoff/recovery_continuation_{cpu,dependencies,template}.json`。

本次结束身份核对有变化：旧Recovery PID1901402/start136388404在1798→1800后**自行因max_output_bytes STOPPED_BOUNDED**、exit0，PID已退出；未被本会话停止/重启。祖先1666+本段1800=3466，剩6534/10000。最终u1800 SHA`578f22a8…c6ea2`真实完整CPU读回通过；旧best_candidate同名旧SHA已失效，保留原证据不伪发布，standing u1400身份仍匹配。751项生产源码/冻结依赖未变。交接模板仍不可执行，未创建新worker或正式handoff，不自动切换；后续明确安排后承接原余额并做12次有界工程窗口。本轮物理/PPO/GPU/GUI/远端/真实删除/生产写入0。下方RUNNING等身份是对应先前阶段的历史快照。

云依赖增量：inventory有限展开SHA绑定JSON，覆盖allocation→prior/retired结算与中心source_binding；r5只读24JSON/1796产物/unresolved0，公共联合38通过，架构194/0、Ruff通过。只修改公共inventory/测试/现有文档；未重包或连接远端。最小映射与long continuation边界见迁移报告对应段；Motion唯一提交，smoke27已耗不退款，r5终态未在本侧宣称。

### 公共PPO／Recovery／Jump效率（历史CPU验收）

已实现rollout inference_mode、常量GAE目标、一次有效Actor全rollout优势归一化；设备端累计日志，Jump资格计数复用；无Actor更新时保留随机前向/RNG/分布缓存但关闭图。Recovery合并finite回读、reset/恢复失效的profile缓存、减少transition复制/prior日志同步；Jump复用step观测、按游标释放已消费摘要、复用真实Critic schema。公共显式计时覆盖采样/GAE/学习细项、保存SHA、引用、指标写入与整轮墙钟，默认不增加CUDA同步。

最终完整回归450通过、8项既有历史skip、3条ONNX警告（111.27s）；架构194/0；CI Ruff0.8.4 check与193文件format通过。最新并发前检fixture的4项增量另见/tmp/wheel_leg_ppo_efficiency/preflight_final.txt；完整命令/源码SHA/微基准/精确文件表同目录，详细消费者与限制在迁移报告本节。CPU组件减时不能外推整轮GPU收益。

fine_ring/完整执行状态、必要terminal和909/6370列保留；Jump仍串行MuJoCo CPU，无Isaac GPU后端。真实GPU A/B、物理、GUI、远端未运行；旧云包是优化前快照，本轮未重打包。Recovery只读首末PID1901402/start136388404保持RUNNING，观测窗口formal1641→1679，剩6655；未干预。新真实训练/物理/GPU/GUI/远端/真实删除/电控写入0。

Motion rollout最新增量已纳入总集成：正式hotpath/速度合同/生产adapter与公共监督、Recovery消费者联合64项通过（22.82s，3条既有ONNX警告）；架构188/0、CI范围Ruff check/format通过。云源码清单和准备包重新绑定本批SHA，旧406项结果仅代表其原快照；组件传感包CPU减时不代表GPU整轮提速。

累计collection视图已接正式label阶段：两轮100新样本加100旧回放得到200/300，来源/样本去重，episode split跨重标注和恢复保留，holdout不回灌，标签记录实际Teacher路由与合同。采集按真实环境行数/设备/owner/子步计费，分块持久化；不再使用评估envs包装器。监督仍共用owner/worker/Budget，支持正式CLI轮边界停止、新attempt恢复、额度耗尽后的compare/export、保存/发布故障判定和部署幂等发布。

Recovery脚本物理I/O通过真实v8奖励、own-hold-v2、909D普通/terminal、PhysicalActor及BC/prior、双Adam优化、公共worker保存恢复和双目标候选。物理标签与确定性forward一致；播放器轮50rad/s不重复cubic解码。完整监督跨目录导入验证了原根目录不可用、依赖一起搬迁、预算不刷新和后续导出。

最终核心406通过、8历史skip、3条ONNX导出警告（114.48s）。8项是既有历史750来源缺失，不是本轮核心用例skip；新增消费者未skip。其后仅BC导入新增“不覆盖来源”保护及对应fixture更新，受影响81项回归通过（30.72s）；随后运动2增量涉及adapter/host测试，受影响76项通过（18.20s）；测试分别绑定源码快照。架构188正式文件/0例外；CI Ruff0.8.4 check通过、format187文件通过。命令/测试清单/源码哈希在artifacts/integration，不能沿用旧314全绿。

仍有代码缺口：sim2sim仅迁入已录制轨迹诊断，实时跨引擎采集未迁完；Motion／Spin在线收集需要领域make_collection_env及owner/条件/子步接口，缺少时正式拒绝而不回退evaluation工厂；旧双Teacher142项索引引用NPZ缺失，审阅配方只能dry-run，真实提交拒绝。新仓库尚不能称全部常用流程迁移完成。

结束只读发现Recovery PID1901402/start_ticks136388404仍RUNNING，已耗1446/剩6888；最新已落盘1400 SHA6132cba2ed634f40c8e845a47f49dfc3542ecdbe64d2b3b233cf5d8a8bcd762a，不是最终handoff。Motion已停止，分支领域结算见下一节。精简云包2824文件逐项SHA核验通过；在另一临时目录，禁止读取旧项目路径，四份配置dry-run及真实Recovery Actor加载通过。云包为准备快照，无SSH/持久盘/镜像/GPU映射；远端API、PhysX、GPU、真实积分、GUI、ORT、ARM及实机均NOT_RUN。Jump只有ABI4 MuJoCo CPU。

本轮真实PPO/监督训练/物理积分/新GUI/GPU/远端任务/真实删除/电控写入均0。小模型CPU优化与/tmp逐文件删除为工程fixture，未改生产run、模型或账本；未提交Git。

## Motion／Spin领域工程准备（2026-11-12，当前）

CI portable增量：补齐877D terminal fixture的`motion_timing.dr.contact=None`，覆盖natural/timeout/mixed与无结束帧的pre-reset collection snapshot；首次TensorDict导入移出scalar-read热路径计数，仍严格断言一次同步。历史`motion442_scale_corrected_resume.json`显式绑定其原冻结spec实际课程`motion31_supported2048_dqgain_ppo_v1`，未给未知版本增加fallback。相关120项、冻结Git索引源码加三文件增量的隔离32项通过；架构206/0、Ruff0.8.4 check通过，1条既有guarded_step warning。明细`artifacts/repository_baseline/motion_fixture_followup.json`。本增量仅配置/tests/文档，driver/worker/owner/远端冻结源码与训练均未修改；不是全套portable已全绿的结论。

portable历史retry fixture收尾：`tests/test_motion_differential.py`的历史集成用例先检查终态/raw/trace/来源完整包，部分历史spec不再误入验证；缺失明确历史skip。新增始终运行的tmp合成fixture穿过真实retry_evidence，核对18份SHA/零在线消耗，并拒绝缺终态、trace变更、非零在线消耗及RUNNING attempt。与前两项联合43 passed/2历史skip，记录`artifacts/repository_baseline/motion_retry_fixture_followup.json`；本轮未改领域源码、学习目标或远端r15。

### Motion压弯／0.20m／H1有界工程smoke（用户新授权）

**速度优化接续（当前，以此覆盖下方r12运行快照）：** 用户授权“优化本地代码之后，再推送到服务器端，重启服务器的训练”。旧r12已正常停止于88次/23068672样本，failure=null，完整u88 SHA`85c879d079008c6e4c4ff7c2656f86eeaba647255000d53cbe23939ad496955b`。新独立源码版本`candidate_revisions/motion_speed_r15/src`已由公共resume启动，run=`2026_11_12__motion_contact_grid_cloud10000_r15_speed`，worker PID1461/start_ticks381772380；源码摘要`ab4e45715824721282c1bbf792fb8ce2f736992a0f9d121145d67238369dd6ef`。有界核验时RUNNING、90次/23592960样本、failure=null，首更u89的Actor/Critic各16次有效Adam更新、262144非零奖励/有效样本；完整u89 SHA`69e4c4153d4e00ec837a11f56970022d2ec2e280f1d8c700e4f386a296a767b0`已CPU加载核验，warmup50不重做。核验时总消耗106/10000、余9894。不是独立能力评估。

本次保留同一逻辑run和额度：r12实际88+之前16=104/10000，接续前余9896；r13/r14的恢复初始化失败各0新增更新，失败记录和attempt均保留。r15从显式跨目录导入的完整u88接续，Actor/Critic/双Adam/std/RNG/课程/Teacher bank逐项相等，warmup50不重做，物理现场重新构造。job为`artifacts/cloud_handoff/cloud_motion_speed_r15.json`，准备与握手收据同目录，不应重复提交旧run。

优化限定在V4张量执行/128槽历史暖填、Critic因果历史索引及reset参数缓存、同物理时刻地面查询复用、分段计时落盘；物理、延迟、DR、奖励、课程、网络、采样量4096×64均未减少。实际冻结Response CPU270步/partial reset对照通过；纯张量4090微基准3.13倍，仅电机响应子模块、不含PhysX/PPO，不外推整轮。完整PPO恢复接管补上现有reopen_for_resume及目录/逻辑身份/本地计数核验，未经验证终态仍不能claim。最终相关联合131项通过（29.58s），6项远端恢复通过（3.44s），速度联合241项保留其原收据；架构206/0，Ruff0.8.4通过。精确增量和测试边界在`artifacts/cloud_handoff/motion_speed_r15_delta.json`。实际u89/u90耗时28.117/25.157秒，其中不保存的u90 rollout24.172秒、learning0.975秒；相对旧41–43秒约减少40%时间，目前仅启动两次，非长期均值。2类编译阶段、881列、40/4.8Nm、V4/NET分路均实际读回，4090瞬时显存12060MiB。精简结果见`motion_speed_r15_execution_summary.json`，原始有界核验见`motion_speed_r15_startup_verified.json`。

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

**配方核对新增阻断（未开GPU）：** H1配置6m×5=30m，出生点12.1m、边界余量.12m，前向有效17.78m；正常命令v3以1m/s理想跟随在18.79s出界，3.6m/s在5.95s出界。实际maneuver_statistics要求30s完整回合且不能terminated，故rough高速晋级受到配置与评分的结构性冲突。CPU反例`artifacts/cloud_handoff/motion_H1_road_length_counterexample.json`使用实际command program与TriangleRoads.contains，物理/PPO0。当前handoff已标开训前需校正，不修改旧冻结200文件/527测试结论；需要让真实碰撞路段、查询范围与完整命令时长一致，不能靠虚构平面或伪造完整回合消除。此前“全部准备完成”的判断未覆盖此项。

远端最终补齐16项公共增量，618个执行源码文件逐SHA匹配绑定`896bd3fd…ecf4917`；完整u3000/双Adam/KL bank游标/2432SDK再次CPU核验及公共train dry-run通过，证据`motion_remote_final_rebind_receipt.json`。本地527结果不冒充远端回归；远端无run/worker/新增PPO，待GPU恢复后独占预检和唯一正式提交。

用户最新选择**Motion独占远端4090**。新增资源配置`cloud_motion10000_r9_exclusive.json` SHA `ce14df839416e301f9d4026cf92c5672d7df62c43c10a5f7b67b20d3843b0db1`，仅resource_mode由shared改exclusive，job/run身份与原10000额度不变。GPU恢复后负责人先核无其他compute任务，不启动Recovery/Spin/Jump，不停止他人任务；此字段本身不是系统级GPU锁。远端CPU完整装配通过，当前不开训。

**最新覆盖：用户已切远端无卡模式，先完成接线，恢复GPU后再提交，不设置自动启动。** 当前r8为FAILED/0完整PPO，r9在GPU预检阶段拒绝，实际未创建run/worker；远端活动worker0。原独立10000额度已准备但消耗0，历史5167不退款。SSH恢复后使用现有r9配置，不重复新建额度或启动旧r8。本地smoke不再接续。

领域冻结清单`artifacts/cloud_handoff/motion_cpu_wiring_delivery.json` SHA `64e968a07ac31e41e3f38bc3475eeff72a3b0e93bb059947edd48635310f8a39`；已发给公共集成负责人，只读绑定200项领域源码/配方/测试。联合133项CPU通过（21.91s），最后diagnostic reset后读回增量14项通过（1.67s）；架构200/0，15个新增/修改文件Ruff0.8.4 check/format通过。正式`make_collection_env`复用同一环境与decoder，不建PPO；保存真实owner、pre-step条件、实际引擎子步和pre-reset状态；脚本I/O已贯穿collect→多Teacher label→aggregate→小fixture fit。低高度推理修复实际合同范围和partial reset目标覆盖；numeric_bridge将DR/863列常量与历史课程脚本分离。上述均未在无卡阶段积分或训练。

远端`artifacts/cloud_handoff/motion_remote_cpu_readiness.json`记录真实u3000 Actor/877Critic/双Adam/std/KL bank游标CPU装配、2432个固定SDK文件SHA核验；配置r9 SHA `ed7c4777a8c19fab1af6ee6e1272fbd17cc0fd86e1c2bac488c58723a208f727`，完整来源SHA不变、warmup50不重复，10000正式dry-run通过。恢复GPU后入口见迁移报告本节；首个真实远端更新、30秒自然timeout及远端真实完整保存恢复仍未证明，不能把CPU_READY称为已开训。

并行评估`artifacts/cloud_handoff/motion_gpu_concurrency.json`：Motion4096×64仅观测存储下界0.887GiB、单minibatch前向激活下界1.188GiB，不含PhysX/反向/队列/CUDA。总峰值/吞吐未知；Motion与Recovery各自启动要求14000MiB空闲，24GiB不能静态保证两份预留。当前建议Motion单独运行，不绕过前检；Jump仍MuJoCo CPU，不是第三个GPU作业。未启动Recovery/Jump，不占用GPU做并行测量。

最新实际：r6 `runs/2026_11_12__motion_u3000_bank_height020_roughH1_smoke37_r6` 已正常`STOPPED_BOUNDED`、failure=null、29次/237568样本，完整u29 SHA `bf701764a9acee5553d5c858f19de35719a4038477f52beddad728096c22cc39`。用户要求尽快云训，取消本地余8次恢复；不能声称真实完整恢复或30秒自然timeout已通过。实际发生21个任务终止/partial reset，0 timeout。正常结束21回合均计入，41cohort有真实样本；这是接线证据，不是能力认证。r4已耗27、有效保存只u16，11次未保存更新不退款；本轮合计56，原实际累计5167。r1/r2/r3/r5失败保留；r2因SDK快速退出漏终态经公共owner.reconcile补结算；其他失败均准确记录。r3物理tick10，无完整PPO；物理消耗不冒充0。

用户随后明确授权**独立Motion10000更新/不限时**，本地配置`configs/jobs/2026_11_12__motion_u3000_bank_height020_H1_cloud10000.json`记录原5167不退款、旧未用4833不叠加；Spin未获新额度。远端数据盘`/root/gpufree-data/wheel_leg_motion`，GPU4090 UUID `GPU-d8240549-897e-46d7-04c8-86d26a927783`，4096环境×64步。路径经RuntimePaths映射，源u3000完整状态和Teacher/bank同SHA；低高度从0.30逐级到0.20、H1速度从0.15逐级提升，直线±3.6锚和纯yaw±2复习保留。

远端启动兼容处理：现成uv环境的RSL-RL/tensordict固定为本地验证5.0.1/0.14.0，Torch仍平台2.11；镜像isaaclab当前API缺clone_environments，携带本地验证SDK代码/扩展为明确依赖（2432文件/18.2MB），保留IsaacSim6.2和PhysX。AppLauncher需ISAAC_PATH及正确libpython；不改固件、物理、PD或源资产。远端各0更新失败独立留档，当前r7已收到worker启动握手（PID2883），真实首更结果待记录，不把RUNNING握手称完成更新。没有第二PPO、没有新Recovery或独立评估。

已修实际首步错误：Task嵌套旧中心表与新范围不一致；height内层在fullchain追加前错误读取transfer奖励；AppLauncher快速退出跳过结算。当前模式reward统一高度/垂速，关闭前持久化异常，平台忽略fast_shutdown时延后到worker结算后关闭。有限路面出界是普通任务终止→合法起点reset，不用虚构平面续算；失效高度明确mask，不授timeout或整窗成功。对应CPU回归67项，后续实际full-source/生命周期45项通过；未完成的在线DAgger工厂另由公共集成收尾。

新配置`configs/jobs/2026_11_12__motion_u3000_bank_height020_roughH1_smoke64.json`沿正式CLI/owner/worker/MotionProductionAdapter接线，只分配64次smoke、formal=0；从共同4889中分配，未分配4825。源u3000完整状态、双Adam、std、RNG和已完成50次warmup保留，模型进度5097与原实际消耗5111不混用。原两份0额度prepared配置仍拒绝提交。计划先有界56次、保存后同run恢复余8次检查加载；没有自动长训。

128环境×64步，含四方向行进压弯、先降至0.20m再低速、0.40m及四种H1静态三角网格路段，保留站立/DR/模态恢复和±3.6直线锚。平地站立/直线约束对称，压弯允许内倾，rough不要求腿对称。低高度中心是完整USD闭链静态FK的几何种子，不是真实承重辨识；0.30–0.40旧域解码不变。实际地面高度/速度改为同碰撞网格三角查询，rough对应平面被移除。

正式提交前新增针对性62项CPU通过（11.80s），架构194/0；同源完整加载、合同/中心绑定、额度拒绝、命令时序与精确地形数学通过。新raw动作/log-prob、原31/877输入尺度、V4/NET分路、20/5ms与轮7/11ms保持。CPU不能证明PhysX碰撞或低高度能力；真实启动与消耗待下方run结果，不把配置解析称已训练。

旧混合Motion已`STOPPED_BOUNDED`、`failure=null`。两分支共用选中完整u3000（SHA `abe10db68b0ad62a3e0ceac5285ebb0041ff084134bc313edb235ae46d427d7e`）；最终3014另保留（SHA `8f6cbc98d0c4934eb9474de8dd0282929726716e484cd6ecf0e2ccfdbd3650c9`）。模型进度5097、实际已耗5111、共同余额4889、新分支局部0分别记录；回退14次不退款。

准备配置为`configs/jobs/motion48_u3000_prepared.json`和`configs/jobs/spin48_u3000_prepared.json`，同用`task=motion`、生产adapter和物理环境。两份更新额度均为0；共同余额未分配、未预留，正式preflight拒绝启动。这是从选中完整模型建立新目标分支，不是旧run原地resume。

已修复跨DR条件继承确认与Spin有限上限失效：连续确认绑定实际方向/命令/高度/DR/执行合同，不因无关revision清空；Spin增长不使用指数，有限上限生效，null仍保证每阶段float32可表示。保留档位、成绩历史与确认链。DR采用显式“共享强度、独立于任务方向推进”，不是各family独立晋级。

最终联合311项CPU回归通过（66.54s，0失败），其中领域针对性57项通过（10.22s），含真实u3000两分支完整加载/临时保存恢复、受管小模型的正式adapter首次候选发布与pending恢复、KL真实梯度/搬迁游标、跨条件确认、有限Spin端点。早失败与UNKNOWN分别处理；物理质量候选不冒充转弯/Spin任务通过。完整验证命令、配方与边界见迁移报告Motion／Spin段。

速度开关已沿真实SHA绑定template核对：hotpath缓存、runtime reduction、合并安全检查、固定形状统计和不使用的ContactSensor历史均开启；headless、2560环境×64步、不做例行独立评估、trace关闭、每100保存。Motion adapter额外把仅用于每100报告的辅助cohort/价值诊断改为报告更新才计算；正常结束回合/课程/奖励/安全检查不降频。CPU同源同RNG对照确认raw/log-prob、reward/returns、Actor/Critic与双Adam完全一致。性能增量后相关320项回归通过（66.99s，0失败），架构/Ruff通过。真实GPU耗时仍NOT_RUN，不能承诺每轮秒数或倍数收益。

后续rollout减负已完成：控制解码的全部数值检查合并到物理采用前验证，Student标量读取5→1，含Teacher10→1；传感包只在初始化/partial reset时暖填四槽历史；877外层仅在回合结束时构造pre-reset终态，并跳过被其覆盖的FullV4中间快照。真实q/dq、previous、FIFO、奖励、完整回合统计及bootstrap语义保持。显式`diagnostics.performance_measurement=true`才增加rollout/returns/diagnostics/learning分段计时与CUDA阶段边界同步，默认不新增逐步同步。相关核心/领域/性能368项通过（72.35s）；架构188/0与固定Ruff通过。2560行单线程CPU组件五轮中位数：传感包0.7041→0.5529ms（约21.5%减时）；解码0.4245→0.4207ms未见明显CPU收益。不是实际GPU或整段rollout提速。具体文件、测试与参考仓库取舍见迁移报告速度段。

以上原准备/速度阶段真实训练、物理积分、GPU、GUI、远端任务均0；新增有界smoke状态单独记录。Recovery等任务未操作。公共生命周期、引用、云端材料由总集成负责人维护。以下11-11及更早PID/活动快照是历史，不代表当前Motion仍运行。

## Motion/Recovery迁移（2026-11-11，历史）

候选已接当前Motion48 v2/877D/4.8Nm/KL-v4和Recovery v8/Swing own-hold-v2/909D/BC+prior的CPU消费者，正式Motion adapter首次发布及新目录恢复、Recovery真实完整来源双Adam捕获/新目录恢复通过。当前配方dry-run通过；核心314 passed in 143.49s (0:02:23)，针对性34通过；架构/Ruff精确结果见迁移报告首页和artifacts/cloud_handoff。仍为部分完成，不能宣布可直接云端稳定训练。

准备包与路径/SHA依赖清单已生成；未提供SSH、持久盘和镜像/GPU身份，Isaac6.2/3.0/PhysX/GPU/真实积分均NOT_RUN。未停止本机任务：结束核对Recovery1901402/start136388404 RUNNING314、Motion1905160/start136466506 RUNNING780；剩余8020/7123。快照非最终handoff，保存点滞后不得退款或重跑。Jump仍ABI4 MuJoCo CPU，Isaac GPU未实现。

生产源码/冻结依赖/真实run/账本/模型/电控只读；无信号、提交、推送、新增物理/PPO/监督训练或真实删除。详细未完成项和准确文件见现有迁移报告首页。

以下2026-11-05任务导航为历史，不能代替当前run记录。

整理核对：2026-11-05 22:00 前后。本页是导航；阶段、消耗与能力以绑定的spec、预算账本、manifest、终态报告和原始评估为准。

## 活动任务（执行时核对）

**新支撑中心 Motion 续训（另一 owner 提交，不是本重构会话）**

- run：`runs/2026_11_5-21_21__motion442_scale_corrected_resume9546`
- 后端 `isaac`，源码根 `/home/kellen/wheel_leg_native_train/src`（新架构），初始化 `full` ← `runs/2026_11_5-16_51__motion_smoke12_supported015_dqgain_formal9988/formal/model_update_442.pt`（SHA `b0b29edf…5785851`）
- 预算：formal 上限 9546，执行时已用 80；checkpoint 每100次，已落 `model_update_1.pt`
- 只读核对：worker PID 2279379，`state.json` 由该 PID 写入，状态 RUNNING
- 本会话未提交、未停止、未恢复该任务

## 已结束任务

**前一段 Motion formal（16:51 提交）**

- 终态 `STOPPED_BOUNDED`（20:46，`stop_kind=user_signal`），formal 实际 442 次更新
- 最终完整状态已保存：`formal/model_update_442.pt`
- 完成报告路径异常：`completion_error.json` 记录 `KeyError('historical_smoke_updates')`；这是旧实现的问题，未被本会话修改
- 恢复点声明与实际不一致：spec 与 `formal_start_verified.json` 声明从 `runs/2026_11_5-16_30__motion_round02_supported015_dqgain_r3_10000/smoke/model_update_12.pt` 接续，但**该文件现在不存在**（该目录只剩日志与报告，无 checkpoint）。同一 run 的 `chain_terminal_report.json` 记录 r3 因 `RuntimeError(['short smoke declaration differs'])` 失败，对应旧代码把 smoke 次数写死为 5 的缺陷，已在新架构中改为按配置取值。旧的 SHA `fa2e5e26…512477` 现在无法复核，不要引用为受保护模型。

## 关键模型（执行时按SHA核对）

| 用途 | 路径 | 结论 |
|---|---|---|
| 前段 formal 最终状态 | `runs/2026_11_5-16_51__motion_smoke12_supported015_dqgain_formal9988/formal/model_update_442.pt` | 存在，SHA `b0b29edf…5785851`，是当前活动 run 的初始化来源 |
| 蒸馏 round02 来源 | `isolated/2026_11_05__motion_supported2048_distillation/round_02/best_validation.pt` | 存在，SHA `8f73714e…1b49d7` |
| 863 派生初始化 | `artifacts/2026_11_05__motion_supported2048_dqgain_ppo/source_round02_actor_critic863_initial.pt` | 存在，SHA `79ec5663…e0ab6` |
| Recovery1500 完整模型 | `runs/2026_10_17-17_59__recovery31_standing_contact_quality/formal/model_update_1500.pt` | 存在，SHA `2c671251…b72cb41` |
| Motion4500 基线 | `runs/2026_10_25-23_52__motion1500_posture_retention312_10000/formal/model_update_4500.pt` | 存在，SHA `0be977b3…fbc619` |
| Motion1900 来源 | `runs/2026_11_4-01_29__motion15_transient_catch863_remainder/formal/model_update_1900.pt` | 存在，SHA `985ee947…1ef6fb` |
| 旧链最终2405 | `runs/2026_11_4-18_24__motion1900_standing_recovery863_10000/formal/model_update_2405.pt` | 存在，SHA `d4eea2fc…96b18ea` |
| World Model contact25 best | `runs/2026_10_28-09_00__motion4500_wm_contact25/round_000/world/fit/best_world.pt` | 存在，SHA `e6367320…9b99c59` |
| World Model 对照 | `runs/2026_10_28-03_58__motion4500_wm_window80_more1000/round_000/world/fit/best_world.pt` | 存在，SHA `6a7c40d9…4b33c8` |

`runs/2026_11_5-16_30__motion_round02_supported015_dqgain_r3_10000/smoke/model_update_12.pt` 不在上表：文件缺失，不能声称受保护。

## 暂停与停止状态

- Recovery：用户暂停保持，见 [暂停文件](configs/experiments/2026_10_18-15_07__recovery1500_friction_pd_dr5000/user_pause.json)；不得自动解除。
- 旧 Motion 18_24 链：用户停止，STOPPED_BOUNDED，未用余额不自动复用。
- World Model：预测门未过，未放行 Actor 更新。

## 证据入口

- 当前活动 run：[run.json](runs/2026_11_5-21_21__motion442_scale_corrected_resume9546/run.json)、[state.json](runs/2026_11_5-21_21__motion442_scale_corrected_resume9546/state.json)、[budget](runs/2026_11_5-21_21__motion442_scale_corrected_resume9546/budget.json)
- 前段 run：[spec](configs/experiments/2026_11_5-16_51__motion_smoke12_supported015_dqgain_formal9988/spec.json)、`formal/supervisor_terminal.json`、`formal/terminal_report.json`、`formal/completion_error.json`
- 重构范围与验证：[迁移报告](docs/MIGRATION_20261105.md)
