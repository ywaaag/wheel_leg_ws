## Motion050多Teacher蒸馏（2026-11-16，本轮结项）

本轮执行完成，候选NOT_ACCEPTED，不替换原vM1或现用部署模型。最终源码执行版本e21fe60，分支motion/multiteacher050-final；数据根motion_multiteacher050_20261116的REPORT.md/CONTINUE.md、independent_final_comparison.json、terminal_accounting.json是完整交付入口。没有活动本任务worker；PPO0、Isaac0、远端0、电控改写0、刷板0、外部通知0。

冻结Actor SHA8890532c241d63e5caa17696e6329b8de285fbfbe20ec96b23ddd3a540e8f9e7，31→256→128→64→6、050原u2200中心。两个新seed各56条件，任务均50/56、保真均43/56；vM1核心任务均33/33，保真30/33、31/33；新能力保真13/23、12/23。核心评分窗承重均100%，整回合最低95.2%，严格超过60%。主要差距是-.5m/s跟踪、后向速度推扰恢复时间，以及±3.6m/s、.20m与部分转弯/高度质量。0/30ms Actor物理观测年龄各3个应力均通过，不称全DR。

正式BC三块6/12/12轮、27042优化步；静态整回合Teacher隐藏未来冲突已在第三块修复，仅按当前31D互斥选择Teacher；未知适用域68309行拒绝标签，未继承成功。52个最终块全部终态，实际4549565步；连同旧探索共176块、20956603步，失败消耗全部保留。最终证据不回灌本候选、不调门或换选中点，后续新版本须另设新开发/留出。

固定vM1 MuJoCo参考motion_vm1_mujoco_v1已写入AGENTS/ARCHITECTURE/README，合同SHA9ae012adbeb675aa3c44930f830aa12ef05279486a00be013cfe95f1c173c6c0。原MJCF/网格加历史closure_contact_tight_v1，不是裸XML柔顺；当前轮4.8Nm与历史2.44695Nm差异另列。CPU独立MuJoCo回合并行、CUDA批量Teacher/BC，仍走统一CLI/owner/worker/ModelIndex。

官方export_candidate_v1保留PT/ONNX/全部依赖/共享C源码；数据根另有256 C权重副本、050中心配置头、96+128条真实状态单条golden，最大PT/ONNX误差2.24e-7、C解码1.49e-7、PD6.68e-6，头文件编译解码误差0。板端050适配、ARM时延/内存与补偿后实机验收NOT_RUN；原015不能只换权重运行。完整BC恢复点与部署Actor分开，原Teacher/旧失败/账本只读。

最终portable2067 passed/31 skipped/10 warnings，导出消费者49 passed、路由81 passed、真实CUDA保存恢复数值核验通过；架构224/0、Ruff0.8.4规定范围与dry-run通过。仓库无remote，未推送。主仓原dirty STATUS与未跟踪文档未修改。

## 实机反馈文档登记（2026-11-15）

按用户提供记录整理Motion vM0（1106/e8658ce）、vM1（1110/4ba73bc）、vM2（1112/786a2c5）及Recovery vR0（1115/烧录哈希未提供），统一入口docs/HARDWARE_FEEDBACK.md，README与ARCHITECTURE已接导航及职责边界。三份Motion提交身份已核对；视频仅飞书占位、CSV未提供。vM1的11月15日震荡／LQR不震荡复测、vR0断电与新满电电池对照、交接未知及两项用户候选办法均保留，欠压未确诊。不更改模型／电控、训练或原始证据。

## 授权磁盘清理（2026-11-15）

本轮按冻结清单逐文件清理374项：失败监督导出临时目录内6份与发布包逐SHA一致的二进制副本（700478372B），以及368份未打开、未跟踪、未登记依赖的pytest/Ruff缓存（7076602B）。共逻辑707554974B，删除时预计解除实际分配块708861952B；现场df净增502816768B（活动训练继续写盘，不能将预计块数当df净增）。未删除任何正式checkpoint、昨天的大模型、Teacher、World Model、训练数据/轨迹、原始矩阵、账本或源码。

644份正式模型路径全保留，49份重点/昨天大模型SHA全部一致，8份完整状态CPU载荷字段核验通过；ModelIndex/RuntimePaths无本次新增悬空引用，22项资产/引用CPU测试通过。Recovery PID275020/start_ticks170596894及Jump PID288628/start_ticks170740409为此次核验身份，结束复核一致，不代表持续监控。没有向训练进程发信号、重启或改代码；清理器自身中断后按持久化删除索引恢复并完成意图结算。

旧1.795GB移交tar仍有当前恢复说明及路径映射消费者，保留；两个Motion保全包及用途未核清历史产物跳过。所有正式模型保留，不按update或大小猜价值。清单/逐文件SHA、inode、分配块、删除索引及前后验证在数据根`history/storage_cleanup_20261115/`；新输出根仍按`--runs-root`/`WHEEL_LEG_DATA_ROOT`规则，不增加清理系统或自动删除。

## 统一数据根接管（2026-11-15）

本轮资产收尾在`integration/asset-storage`完成：新run输出根改为显式`--runs-root`优先、否则`WHEEL_LEG_DATA_ROOT`；缺少时真实写入拒绝，dry-run不创建目录。PPO/BC共用数据根models.json，resume不改原run。部署默认输出到数据根exports，显式诊断/导入/导出路径由owner选数据盘。

数据根`history/asset_storage_20261115/`保存逐文件plan/receipt、派生RuntimePaths映射、保护进程快照、旧Git状态及1490份未提交小型源码/配置/文档实际内容（33035843B）。接管2715项、逻辑2281711832B：复用1895项、不可变硬链接5项、复制815项，新增实际分配45481984B；包含Motion4500、World Model best/对照等8个重点模型CPU加载。另将1795502080B既有依赖包接管至`archives/motion_recovery_prepare.tar`并只移除源码仓该原文件名；同盘硬链接不释放本体空间。旧仓模型原件未移除。所有新目标及源码历史复用唯一ModelIndex登记消费者，不解除原角色。

四份派生Motion/Spin/Recovery/Jump准备配置解析、dispatch、CLI dry-run通过，独立子进程禁止读取旧仓仍通过；不表示活动Recovery当前交接、最新Jump领域实现或新训练授权。旧STATUS两项来源缺失继续UNAVAILABLE，旧142份数据及失效候选不补认证。旧仓2792个>1MiB大型文件与worktree未证明切换的产物保持原位；不能宣称完全脱离旧目录。快照发现Recovery/Jump活动worker，其源码、run、预算和模型未操作。

本轮针对性43 passed；portable2051 passed/31既有skip/10warning（137.98s）；架构220/0，Ruff0.8.4 CI范围check/219文件format通过。未运行真实训练、物理积分、GUI、GPU、远端或真实模型清理。完整回执位于数据根，普通功能不续写迁移报告。

看板自动发现增量：正式train提交结果返回已有产物根/run/启动握手引用；本地刷新和远端显式刷新可发现授权runs目录直接子run，最多20个/14MiB、资源单次采集。无新增registry或watcher，不提交训练；当前运行看板仍使用此前单run授权，扩大远端root须明确配置。

公共看板远端只读接入：运行时显式host/port/user/run-root/已核验指纹，密码仅后端getpass内存；OpenSSH严格known_hosts并额外校验ED25519指纹。页面默认手动采集，可选页面存活期间120秒刷新；无远端agent、watcher或训练控制。真实只读快照确认Motion RUNNING/2079已耗更新、worker1485/start_ticks401486752身份匹配，容器14核quota/50GiB，GPU4090；均为采集时快照，不代表持续核验。读取失败保留旧缓存。Jump/Recovery远端目录未接入。针对性39通过、portable2015通过/31既有skip/10warning，架构217/0、Ruff0.8.4通过；实际IP浏览器资源与曲线验证通过。源码未写死平台地址或端口，下一次更换需显式配置和私密输入。

<!-- dashboard-current -->
公共只读看板：独立分支 `feature/read-only-dashboard` 接通统一CLI、标准产物读取、HTTP和网页。Motion训练窗口已接入；Jump/Recovery提供通用状态与曲线，能力适配未支持；实机适配未支持。仅CPU/脚本fixture验收，不改变活动训练；初版针对性23通过，portable 1999通过/31既有skip，架构215/0、Ruff 0.8.4通过；浏览器验证曲线、筛选、比较和证据入口。未将训练窗口当作固定模型证书。看板后续修复：中文结构化卡片/表格与八组SVG曲线，默认选有指标的最近任务；移除跨run读取限额导致的后续任务饥饿，大state只读256KiB完整顶层摘要，共享大索引单独报错。原始JSON仅折叠展示。本次增量针对性26通过、portable 2002通过/31既有skip/10warning；真实IP浏览器验证Motion 7图、Recovery 5图、Jump 6图，缺指标留空，默认展开JSON为0。

# 当前任务

## Motion原Student050远端PPO（2026-11-16按用户指令暂停）

当前原run已`STOPPED_BOUNDED/failure=null`，最后有界核验worker812已退出；不自动恢复。formal实际已耗2289，加本机smoke52，本次10000已耗2341、剩7659。完整u2200（SHA `7387d0d2f9a42e450d32b35aa47a8b05bfa1de810488cf558d621530bbc6dd89`，53380558字节）、原run/预算/候选/诊断、609模块冻结源码及交付依赖已回传数据根`motion_preservation/20261116_wide050_pause/backup/`；764文件、381292321字节逐SHA一致，7个完整checkpoint已通过现有ModelIndex保护。本机完整u2200 CPU加载及生产PPO装配通过：Actor/Critic/双Adam/std一致，RNG/课程/bank游标保留，warmup_done=50。没有新增PPO或独立能力评估。

用户随后要求精简，本机本次正式长训模型仅保留完整u2200（恢复来源）与u50（Actor未更新的基线），合计85181189字节。通过现有ModelIndex引用锁和retention入口删除备份中u1/u500/u1000/u2000/u2100，并删除启动目录同SHA的u1重复副本，共6文件、277198650逻辑字节；原始7份清单是清理前历史。原run/预算/课程证据与必要Teacher/bank不变，其余757份原备份逐SHA仍一致。收据为上述数据目录`model_prune_20261116.json`及`model_prune_duplicate_20261116.json`；远端、独立smoke和其他领域模型未操作。

停止保存存在缺口：停止请求在rollout中途被`simulation/isaac/motion_backend.py:828`发现，公共`simulation/driver.py:169`的incomplete分支直接返回最后保存点，未调用最终保存；因此u2201～2289共89次已耗更新没有落盘，不能冒称完整保存u2289或退款。普通同run resume会被`experiments/worker.py:138`的计数一致性校验拒绝；用户下次复训前须完成显式模型/消耗差额接续，并在新版本修复停止边界，不能热改或重写原checkpoint。完整收据、逐文件SHA、CPU核验、结算与恢复限制见上述目录`CONTINUE.md`、`settlement.json`。备份完成后的SSH31193连接被拒，未据此推断服务器状态；原件未删除。

以下保留原启动及配方历史，不能按启动快照重新train。

执行口径已集中到[Motion执行口径](docs/MOTION_EXECUTION_ALIGNMENT.md)：Isaac训练与MuJoCo验证／播放均使用同源完整V4，不额外叠旧弹簧／C补偿；已有015实机保留原软件补偿。辨识分支`c8d9636`确认发送补偿后力矩、反馈记录扣除补偿，V4主要拟合目标到q／dq；原CSV与当时烧录身份／通道布局尚未完整绑定，不宣称全高度／负载／饱和等价。新050部署需自身解码、中心及网络适配。此前文档统一未改执行配置；本次按已准备配方和用户明确指令拉起长训。

用户明确选用原Student `258aae9d`，授权完成接线/smoke并新授10000次在线更新、训练墙钟不限；GPU已由用户开启，并于本日明确授权启动。旧累计在线9519与104轮BC保留，旧7219余额退役。52次本机smoke实际COMPLETED，全部计入新额度，formal分配9948；正式历史基数9571、累计上限19519，不退款故障或回退消耗。两个初期0更新失败与账本保留。不建watcher、不持续轮询；原准备配置不再重复train。

实际run为远端`/root/gpufree-data/wheel_leg_motion/runs/motion_wide050_cloud_formal9948`。2026-11-15 13:44:44（Asia/Shanghai）有界启动快照：RUNNING／formal，已用6次、1572864样本、failure=null；本轮总消耗52+6=58，剩9942。worker PID812／start_ticks404984722，冻结源码609模块、SHA `b876ea3166af8185e7a42bc97f0f3eb46c4220fc045c63a62f29859210428af6`。本次SSH端口31193；14核／50GiB、4090 UUID `GPU-0401c8e7-4db1-eeb9-a747-79564b3c1366`、CUDA映射0，未操作其他owner。以上为启动快照，不代表持续监控；已存在run须inspect，不能重新提交。

首个真实更新262144条有效／非零奖励样本，Critic16次Adam、Actor0次，属于已声明的50次Critic warmup，包含在formal9948中。完整u1已保存且回传本机，SHA `ba81288ebc1991060ee111bdcf6cac3f1354ffd2690ab95f51ac2e32b7df4c61`；Actor均值权重与原Student一致、探索std按声明重设、warmup_done=1、Critic Adam／RNG／课程／计数和050输出合同核验通过。本机启动收据位于数据根`motion_wide050_ppo_ready_20261115/cloud_start_20261115/`；CLI看板交接复用artifact_root=`/root/gpufree-data/wheel_leg_motion/runs`、上述run和`startup_handshake.json`，见该目录`dashboard_handoff.json`。

启动前首次提交因准备包漏`candidate/configs/model_refs.json`在创建run／worker／预算前失败，0新增更新；已补原SHA清单和受保护u1600的精确RuntimePaths映射，失败收据保留，未改冻结Python源码或训练语义。真实解码及checkpoint均为050／80；旧`MotorLimits.as_dict()`仍静态报告`action_scales=[.15,80]`（`tasks/motion/execution_limits.py:39`），这是限矩诊断元数据残留，不能据此认定当前解码回到015。应在后续新版本按动作合同修正诊断，当前运行源码未热改。

050/80 PPO复用统一CLI/owner/worker/生产Motion adapter/受控PPO，新增内容绑定Actor准入记录（Actor/合同、课程、旧结算、新授权、唯一分配、smoke结算），无新训练循环。原Student的中心表不变，不采用实验Student及新.24中心。Actor-only只导入均值权重，显式重设物理探索std .02rad/2rad/s（raw .04/.025），新881D Critic/双Adam/RNG/课程；50次Critic warmup计费。同run完整恢复保留原Actor准入、warmup、bank游标，不重复初始化。

来源保护改为原Student同050合同，3000条train-only bank覆盖0/.34及±3.6/.34命令；不把bank送入PPO rollout/GAE。合格名义on-policy与bank仍按原每命令归约和.5混合；rough/事件/接触柔顺/新DR和转弯不被旧名义标签拉回。奖励、KL系数、PD、限矩、V4、真实dq、输入尺度和延迟不变。课程复用55%基础、28%转弯、7%H1、10%高度，起始容易档与新确认。固定模型能力不足与工程smoke分别记录。

启动前工程证据：本机共享4080（UUID e704e7d4…）128env×64实际52次，前50 Actor保持、Critic更新，后2 Actor真实优化，完整u1/u50/u52及最终保存可装配恢复，首轮8192有效/非零奖励样本，真实结束/reset；其他owner未操作。本机执行冻结于v2 worktree，修复/交付版本独立，未热改。远端准备时为CPU模式0.5核/2GiB，Actor/新Critic/双Adam/78cohort/3000 bank与2432 SDK文件SHA通过；本次4096env远端PhysX首更已核验，长期吞吐和能力尚未验收。原START/delivery/REPORT保留准备时身份，本次GPU绑定、清单补齐和真实启动结果以`cloud_start_20261115/`收据为准，不扫描端口、不重复提交。

正式配方`configs/jobs/motion_original_student050_ppo10000.json`，源码/当前文档/相关消费者齐全。针对性55 passed，portable2042 passed/31既有skip/10warning，架构220/0、Ruff0.8.4 check及219文件format通过。后续真实状态以run账本为准，未将smoke或CPU就绪当成全矩阵通过。

## Motion转弯支撑与0.24m短补教（2026-11-15）

用户授权新数据短补教，不启动PPO。本机CPU复用正式distill/owner/worker与既有物理标签、conditions多Teacher路由，完成2×12=24轮BC；原80轮保留，累计104轮。本阶段PPO2781/10000、剩7219不变。数据从70097增至92497条，新增反馈仅train；旧数据仅复用train，开发/留出整回合隔离，原留出文件保持SHA不变。训练源码未改，源摘要`8cc73b303053d5590ff35c1cb65716a72895389dac96b85889e591b9dc143b34`已归档。

0.24m训练条件示范选择.07rad镜像偏移，用真实adopted PD目标拟合新.24输出中心；Student不带运行时Teacher偏移。第二块仅按新train条件的方向支撑表现选择冻结u2200或原Student标签，各自015/050解码后反解新050。输入、PD、限矩、真实dq、V4与延迟不变，不加转弯对称投影。每块新BC Adam，非完整PPO恢复。

第二块开发17/17完整、相对保留15/17；低020/024及基础5项保留。后退负yaw采样双支撑由原Student.691提高至.935，但前进负/正yaw由.972/.952降至.940/.894，未通过原硬门，原Student仍为正式起点。第二块按开发排序冻结为诊断用实验候选，最终新留出Student17/17完成，相对14项通过、2项未过、1项UNKNOWN（旧Teacher波浪回合提前终止）；.2425m站立/前后高度MAE由9.18～9.50mm降至1.61～1.98mm，但三个目标转弯仍未全保住。留出不改变替换决定、不追加训练。精确训练条件复放：.20站立实际.200124m、.24实际.238322m，raw腿均值最大绝对值.01459/.03493；.34前后实际+3.463/-3.637m/s、自转+2.012/-2.029rad/s；自转双支撑偏低仍待解决。不是完整能力矩阵、DR、PhysX或实机通过。

正式复现配方`configs/jobs/motion_turn_height_short_bc12.json`，证据、原启动握手看板引用、模型及完整BC恢复点位于数据根`motion_turn_height_short_20261115/`（REPORT/delivery/CONTINUE/PLAY）。050 PPO入口仍拒绝。新增仅配方与文档，无新训练入口或框架。消费者34 passed（5.36s），portable2037 passed/31既有skip/10warning（141.47s），架构219/0、Ruff0.8.4通过。

## Motion低高度受载补教（2026-11-14）

用户授权反复蒸馏/补教，保留0.34m高速、自转、转弯与地形。本机CPU沿统一distill/owner/worker/supervision完成3×16=48轮BC，10848次监督Adam step；前轮32轮保留，合计80轮，PPO新增0，本阶段2781/10000、剩7219保持。每块为新数据与Actor初始化、新BC Adam；不冒充跨块完整优化器恢复。

原u2200 Teacher按自身015合同解码后，显式高度分段物理示范偏移补低高度平衡；Torch标签与NumPy采集共用任务模块。Student不带运行时示范偏移，仅自身050/80输出合同；0.20/0.22中心取训练站立实际adopted PD目标拟合，输入、真实dq、PD、限矩、V4与延迟不变。新增Teacher标签preflight在owner提交前拒绝非法声明。050 PPO继续显式拒绝。

仅train条件反馈补教，开发/留出整回合隔离。第三块第9轮Actor按预声明开发闭环规则选定，SHA `258aae9d904fc3718ddc7938a556c523ff75228a0a7f2707f72b2162d9930d98`；开发22/22完整且相对筛选22/22，完整BC恢复点第16轮另存。精确训练条件复放0.20m实际0.199962m、双支撑100%、腿raw均值最大绝对值0.01429；0.22m实际0.223691m。约0.24m仍近1cm高度偏差；自转双支撑、欠转及地形欠速均不能因模仿保真判定能力通过。

正式配方`configs/jobs/motion_low_support_reteach_bc16.json`复用preset；公共看板的原startup_handshake交接见数据根`motion_low_support_reteach_20261114/dashboard_handoff.json`，不另建登记表。冻结后43个留出回合全部完成，相对筛选40/43；三个转弯双支撑相对下降5.8/8.8/10.8个百分点，继续列为缺口。两个低站立留出高度MAE为0.485/4.479mm，均满足本轮门槛。完整模型/BC/导出SHA、源码快照和继续起点见同目录REPORT、delivery及PLAY。固定模型MuJoCo标称证据与训练窗口分开；完整DR/扰动、PhysX、实机不授予通过。集成CPU portable2037 passed/31既有skip/10warning（137.83秒），架构219/0，Ruff0.8.4 check/218文件format通过。旧冻结源码、预算、原模型和其他任务保持。

## Motion物理目标蒸馏与受载中心（2026-11-14）

用户授权停训、蒸馏并保全模型。以已回传完整u2200为冻结Teacher，本机CPU经公共distill/owner/worker/supervision实际完成8+24=32轮BC、2600次监督Adam step，PPO新增0；原阶段2781/10000、剩7219保持。首个缺wheel身份的初始化attempt消耗0，原失败与结算保留。Actor结构不缩小，050/80显式合同与Teacher015各自解码物理标签；输入、PD、限矩、V4、真实dq和延迟不变。原完整模型、BC恢复状态、候选和ONNX均保留在数据根`motion_distill_wide_20261114/`及原729文件保全包。

93个Teacher回合＋31个Student训练条件回合聚合48205条真实前动作输入，开发/留出不补教。第二块第23轮按开发损失选为候选，SHA `c35ad9897917cecd8b2d16eeefd1dfdd55cb7d5e47df20c7a85c74b46ce6457b`；完整BC最新24轮。开发闭环31/31完成、相对保真筛选30/31；冻结后留出31/31完成、筛选28/31。留出正自转姿态/速度及两转弯双支撑仍有退化，不授予全矩阵、完整DR、PhysX或实机通过。

训练站立片段形成0.34/0.36/0.40m镜像PD中心，重标物理标签后蒸馏；留出0.345/0.364/0.396m站立四腿raw均值最大绝对值约0.03。原0.20m误差未被该候选解决。另用显式低高度渐进PD偏移研究，0.18rad组在0.20m命令下实际0.200265m、双支撑100%、vx约-0.06198m/s；镜像PD目标候选见`height020_loaded_center_proposal.json`。满足用户0.15漂移死区研究门槛，未满足原0.05严格标记；尚未纳入冻结Student，不能直接换表或当作已学会。

正式配方`configs/jobs/motion_u2200_wide050_recenter_bc24.json`复用preset与公共监督；当前050只支持BC/播放/导出，PPO入口显式拒绝，须另做来源保护及Actor初始化接续。完整证据、失败/owner字段纠正、源码602模块快照、支持边界和命令见数据根REPORT/delivery/PLAY。最终portable2002 passed/31既有skip/10warning（130.78秒），架构213/0、Ruff0.8.4通过；Actor/完整BC状态/episode划分/导出包和SHA核验见`final_artifact_verification.json`。没有自动重启远端或操作其他任务。

## Motion受载中心与输出包络诊断（2026-11-14）

独立`motion/loaded-center-scale-study`复用统一diagnose与MotionPlayer/C/V4，加入固定Actor、有界case、显式物理目标干预及原目标/实际目标记录，不改冻结训练合同。u2000本机平地MuJoCo对照：原0.20m命令稳态高度约0.22418m；旧raw直接按0.30/0.50rad重解码分别7.35/0.84秒触及姿态/高度边界，尚未到低高度评分，不能当新Student性能。没有合格受载中心候选，未改中心或正式scale。原始结果及蒸馏准备见数据根`motion_loaded_center_scale_20261114/`。

用户授权停止Motion、蒸馏并回传模型。重连30548成功后核实原worker1485不存在、state陈旧RUNNING/2281；公共stop request-only记录，保留原状态后owner.reconcile结算FAILED（worker exited without terminal state）。本阶段已用500+2281=2781、剩7219，保存u2200与账本相差81次不退款；没有正常退出最终保存，不重启。保全回传与SHA核验以数据根`motion_preservation/20261114_height_stop/verification.json`为准。当前尚未启动BC或产生新Student。

## Motion少量变高度接续（2026-11-13）

用户要求当前加入0.20～0.40m站立与平地运动。独立`motion/height-small-dose`分支准备10%变高度、28%转弯、7%地形和55%基础复习；六目标高度0.20/0.24/0.28/0.32/0.36/0.40m，各自站立/±0.25m/s，复用原rate-limited高度优先时序及SplitCourse。不改PPO/奖励/KL/执行合同。新增完整课程接续记录只结转原10000授权的实际剩余，累计上限16738不变，真实停止点与云端结果以数据根`motion_height_small_dose_20261113/`收据为准；旧远端Motion已通过公共stop request-only正常STOPPED_BOUNDED于500，failure=null，完整u500 SHA `1ad4a858baebffa1a0cad2e989c57e054934940372a593be9842f4a7e6801e89`；保存点差额0。本授权剩余9500，历史累计7238、模型进度7097，累计上限16738保持。远端新冻结`candidate_revisions/motion_height_small_dose_v2/src`599模块摘要`c408dc029ae2036dd9233f5d98e3f8d54b88e67a6a6710b6ac21d3521460c81e`，配置`height_small_dose_v2_20261113/cloud_config.json`；78cohort/407高度槽位，最小11槽。完整PPO双Adam CPU装配与dry-run通过；首版预检曾拒绝重复877→881路径，修正并保留拒绝现场，新run `motion_height_small_dose_remaining9500`唯一提交成功，worker1485身份OK，启动快照RUNNING/0更新/failure=null，首个PPO更新尚未确认，未新增watcher。最终相关54 passed（6.62s），portable1988 passed/31既有skip/10warning（131.70s），架构212/0、Ruff0.8.4 CI范围check/format通过。


## Motion网页前端与键盘（2026-11-13）

现有播放器页面吸收历史Motion前端的暗色卡片、命令/反馈曲线、焦点保护及物理键映射；不引入旧斜坡、手柄、技能切换或新物理循环。
网页显式启用WASD/方向键，组合转向、松键归零、失焦释放、编辑控件不抢键、Space停止/Esc暂停/R重置；所有输入共用串行command通道。
曲线仅最近30秒/最多300点，缺失留空、reset清空，不写存储或文件。浏览器断网无法保证释放送达，服务端最后命令语义保持。
相关45项通过，portable1980通过/31既有skip/10warning（151.44s）；架构212/0、Ruff0.8.4 check/format及JS语法检查通过。
实际浏览器fixture及真实u1500 MuJoCo网页按键采用已验证，播放器在独立DISPLAY=:2；当前有线网卡DOWN导致192.168.250.1不可达，未修改网络设置。


## Motion MuJoCo交互播放器（2026-11-13）

已纳入统一`wheel-leg play`，复用公共GUI/headless循环、推理factory、共享C/V4和既有网页命令服务。
支持SHA绑定模型/本机RuntimePaths、平地＋四种H1单地图、显式区域reset、网页独占控制、实际状态反馈与不保存模式。
Motion/Recovery轮限矩仅接受4.8 N·m；求解器profile显式选择，非PhysX等价。默认网页仅loopback，局域网用`--panel-host`显式配置。
当前用法见README“Motion MuJoCo网页交互播放”；本机映射留数据根`playback/motion_u1500/runtime_paths.json`，不把远端SDK链接当本机依赖。
独立Openbox桌面使用`DISPLAY=:2 XAUTHORITY=/home/kellen/.Xauthority`，不是GNOME虚拟工作区。
旧播放器worktree作为原运行版本保留，不热改/重启；后续新播放使用main。短步/网页可操作不构成模型能力认证。
集成验证：相关44通过；portable1979通过/31既有skip/10warning（133.96s），架构212/0，Ruff0.8.4 check及211文件format通过。
集成版真实u1500无保存200步通过；原五区域/网页交互证据保持原版本身份。机器映射与审阅收据位于上述数据根。

## Motion开训前审阅与欠转诊断（2026-11-13）

在独立`motion/pretrain-gate-audit`分支复核保留的15个训练统计窗口。正式SplitCourse.advance的基础保留门槛：u100失败（DR±2rad/s自转），u200～u1500全部通过；近期连续6窗未见结构性晋级阻断。新75%行进转向评分缺逐回合记录，重评UNKNOWN。逐cohort结果及来源SHA位于数据根`motion_pretrain_audit/20261113_turn_focus_v3/`。

生产adapter接入`tasks/motion/execution_statistics.py`，仅在报告更新累计Student六通道执行目标差、raw越界和请求/driver/motor饱和，按cohort导出。非100更新窗口、非1ms积分，不改变评分/奖励/控制/PPO；原最后步physical_control保留。调用链和边界见ARCHITECTURE及Motion能力矩阵末节。远端冻结版本尚未更新，本轮无远端连接、GPU、物理或真实训练；待用户通知GPU后再核实现场，不自动启动或监测。

验证：新增/报告时序联合50 passed（7.00s），fixture消费者回归50 passed（5.51s）；portable最终1968 passed/31既有skip/10warning，完整耗时见数据根portable_final.log；架构210/0，固定Ruff0.8.4 CI范围check及209文件format通过。首次portable因4个旧fixture缺新增scope字段失败，补齐后重跑；原失败日志保留。

## Motion新10000次云端接线（2026-11-13）

用户明确授权从完整u1500按转弯重点配方开始新的10000次在线更新、训练墙钟不限，允许共享GPU。旧累计消耗6738保留，旧阶段8429余额退役，不与新额度叠加，不退款55次。旧r15已通过公共owner.reconcile结算FAILED/worker exited without terminal state，实际1555次不变；原始结算前快照仍保留。新run身份`motion_u1500_turn_focus_v3_10000`，正式配置`configs/jobs/motion_u1500_turn_focus_v3_cloud10000.json`，沿既有分支/分配/owner/worker接线，不新增生命周期。

GPU由用户手动开启并通知，无持续监测、watcher或自动启动。共享资源仍做UUID/显存与活动任务核验，不停止其他任务或自动换卡。授权/结算/分配在数据根`motion_authorization_20261113/`，真实旧终态在`motion_cloud10000_20261113/round2_handoff_20261113/`。此处为接线记录，真实远端CPU与部署结果以收据为准；没有GPU前不能宣称物理、PPO或实际能力验证已通过。

部署及CPU_READY已核验：远端`candidate_revisions/motion_turn_focus_v3_10000/src`共598模块，摘要`3dd6cd7f87f76c94da103198f8310c56938f39b3a57bb0b6c6b627e2933e474d`与本机一致；完整u1500实际PPO装配的Actor/Critic/双Adam一致、warmup50保留，2432个SDK文件及物理资产/V4和15项引用SHA通过。远端`round2_handoff_20261113/cloud_config.json`正式dry-run通过，无GPU预检准确拒绝，未创建新run或worker、真实PPO0。`environment.sh`仅设置环境，`START.txt`给出唯一正式CLI，不创建监测器。远端收据已回传数据根。本机相关86 passed（7.68s），portable1967 passed/31既有skip/10warning（144.97s），架构209/0、固定Ruff0.8.4 check/format通过。精确命令、改动SHA、部署身份和边界见数据根`motion_authorization_20261113/delivery.json`。

## Motion转弯重点配方准备（2026-11-13）

Motion负责人在独立`motion/turn-focused-recipe-v2`分支从main实现第二轮第一块。配置`configs/jobs/motion_u1500_turn_focus_v3_prepared.json`复用SplitCourse和生产adapter，固定0.34m，保留55%基础复习，将约36%总样本集中四方向转弯、9%保留四地形；暂缓变高采样。新增0.65/0.8 yaw中间档、75%行进转向响应门槛及基础退化/UNKNOWN时阻止focus晋级。奖励、PPO、KL、动作/输入尺度、物理/PD/电机、延迟与DR范围不变。具体门槛和边界见[能力矩阵末节](docs/MOTION_CAPABILITY_MATRIX.md#当前准备配方034m转弯重点块)。

u1500完整状态通过CPU准备边界，不重复877→881或高度扩展，双Adam/std/RNG/KL bank/warmup50和模型进度保留；地形碎石/坑洼0.25起点保留但确认重积累。原账本已耗1555、checkpoint1500的55次不退款，本阶段余8429。数据根`motion_recipe_preparation/20261113_turn_focus_v3/`保存SHA绑定准备快照与CPU收据；此记录不是停止结算或训练分配，零额度通过正式dry-run，正额度在正式边界拒绝。当前未做旧attempt结算、云端部署、GPU/仿真或真实PPO；模型能力改善仍未证明，欠转物理根因待确认。

验证：相关联合145 passed（10.61s）；portable1959 passed/31既有skip/10warning（141.80s）；最后补严来源center一致性后，针对性22 passed（2.84s）及真实u1500准备重核通过。正式dry-run、架构209/0、Ruff0.8.4 check/format通过。真实u1500只做CPU加载/装配，生产adapter保存恢复使用小型脚本I/O fixture；不将其称为GPU恢复验证。精确命令、逐文件SHA和原始日志保存于上述数据根的`delivery.json`及日志文件。

## Motion能力目标矩阵（2026-11-13）

领域负责人：接替“运动2”的Motion／Spin会话，远程Motion训练唯一负责人。已形成[Motion能力矩阵v1](docs/MOTION_CAPABILITY_MATRIX.md)，供其他Codex审阅配方、评分、播放和部署时共用。优先保留0.34m站立、±3.6m/s直线和±2rad/s普通自转；四方向转弯、四种H1及0.20～0.40m高度按前置能力逐级扩展。仅平地站立/直线约束对称，转弯允许合理内倾和腿长差。逐模型证据与目标分开，未定义的阈值/组合范围须在后续配方明确。

本次仅文档与架构导航，不改源码、训练配置、奖励、预算或冻结版本，不启动训练/仿真/独立验证。后续工作是审阅评分及课程接线，不能把目标矩阵称为已经生效的新训练实现。u1500及完整云端保留收据位于数据根`motion_preservation/20261113_r15_u1500/README.md`；最后只读现场为无GPU/无worker、旧state未结算，账本记录1555，本阶段总耗1571、余8429。这是2026-11-13历史快照，执行前重核；从u1500回退55次不退款，本文不授予新额度或自动续训。

## 迁移前未跟踪资料归档（2026-11-13）

主仓源码无未提交修改；本次处理仅是迁移时刻意保留的历史资料。192文件/5,944,367字节已独立复制到`/home/kellen/wheel_leg_training_data/history/docs_pre_refactor/`并逐SHA核验后，从新仓精确移出；187份与旧仓相同，5份差异完整保留，旧仓未改。manifest SHA `23e6febb308ea280520aa1604b88abf46733dabb08dd5bcc9475ca49bbda4063`，记录原路径、目标、SHA及比较结果；差异和引用另存同目录。4份差异为旧仓后来追加段落在候选中缺失，1份仅空行，无候选独有当前能力需要迁入。README记录归档查询及正式入口边界。历史design说明/旧工具的报告字符串按原身份保留，正式运行源码没有读取这192文件。本轮不改运行源码/配置、模型、预算或活动领域worktree，不使用git clean或宽泛忽略规则。

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
