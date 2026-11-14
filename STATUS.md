看板自动发现增量：正式train提交结果返回已有产物根/run/启动握手引用；本地刷新和远端显式刷新可发现授权runs目录直接子run，最多20个/14MiB、资源单次采集。无新增registry或watcher，不提交训练；当前运行看板仍使用此前单run授权，扩大远端root须明确配置。

公共看板远端只读接入：运行时显式host/port/user/run-root/已核验指纹，密码仅后端getpass内存；OpenSSH严格known_hosts并额外校验ED25519指纹。页面默认手动采集，可选页面存活期间120秒刷新；无远端agent、watcher或训练控制。真实只读快照确认Motion RUNNING/2079已耗更新、worker1485/start_ticks401486752身份匹配，容器14核quota/50GiB，GPU4090；均为采集时快照，不代表持续核验。读取失败保留旧缓存。Jump/Recovery远端目录未接入。针对性39通过、portable2015通过/31既有skip/10warning，架构217/0、Ruff0.8.4通过；实际IP浏览器资源与曲线验证通过。源码未写死平台地址或端口，下一次更换需显式配置和私密输入。

<!-- dashboard-current -->
公共只读看板：独立分支 `feature/read-only-dashboard` 接通统一CLI、标准产物读取、HTTP和网页。Motion训练窗口已接入；Jump/Recovery提供通用状态与曲线，能力适配未支持；实机适配未支持。仅CPU/脚本fixture验收，不改变活动训练；初版针对性23通过，portable 1999通过/31既有skip，架构215/0、Ruff 0.8.4通过；浏览器验证曲线、筛选、比较和证据入口。未将训练窗口当作固定模型证书。看板后续修复：中文结构化卡片/表格与八组SVG曲线，默认选有指标的最近任务；移除跨run读取限额导致的后续任务饥饿，大state只读256KiB完整顶层摘要，共享大索引单独报错。原始JSON仅折叠展示。本次增量针对性26通过、portable 2002通过/31既有skip/10warning；真实IP浏览器验证Motion 7图、Recovery 5图、Jump 6图，缺指标留空，默认展开JSON为0。

# 当前任务

## Motion原Student050新PPO长训准备（2026-11-15）

用户明确选用原Student `258aae9d`，授权完成接线/smoke并新授10000次在线更新、训练墙钟不限；GPU由用户稍后开启，本轮不自动长训、不建watcher。旧累计在线9519与104轮BC保留，旧7219余额退役。52次本机smoke实际COMPLETED，全部计入新额度，余9948；正式历史基数9571、累计上限19519，不退款故障或回退消耗。两个初期0更新失败与账本保留。新正式run尚未创建。

050/80 PPO复用统一CLI/owner/worker/生产Motion adapter/受控PPO，新增内容绑定Actor准入记录（Actor/合同、课程、旧结算、新授权、唯一分配、smoke结算），无新训练循环。原Student的中心表不变，不采用实验Student及新.24中心。Actor-only只导入均值权重，显式重设物理探索std .02rad/2rad/s（raw .04/.025），新881D Critic/双Adam/RNG/课程；50次Critic warmup计费。同run完整恢复保留原Actor准入、warmup、bank游标，不重复初始化。

来源保护改为原Student同050合同，3000条train-only bank覆盖0/.34及±3.6/.34命令；不把bank送入PPO rollout/GAE。合格名义on-policy与bank仍按原每命令归约和.5混合；rough/事件/接触柔顺/新DR和转弯不被旧名义标签拉回。奖励、KL系数、PD、限矩、V4、真实dq、输入尺度和延迟不变。课程复用55%基础、28%转弯、7%H1、10%高度，起始容易档与新确认。固定模型能力不足与工程smoke分别记录。

本机共享4080（UUID e704e7d4…）128env×64实际52次：前50 Actor保持、Critic更新，后2 Actor真实优化，完整u1/u50/u52及最终保存可装配恢复，首轮8192有效/非零奖励样本，真实结束/reset。正式4096env、远端GPU/PhysX/能力与吞吐尚未验证；其他owner未操作。本机执行冻结于v2 worktree，修复/交付版本独立，未热改。远端CPU模式0.5核/2GiB，Actor/新Critic/双Adam/78cohort/3000 bank与2432 SDK文件SHA通过；最终部署与精确命令见数据根`motion_wide050_ppo_ready_20261115/START.txt`、delivery和REPORT。GPU开启后只做一次实际UUID/映射/共享资源/磁盘核验，再按唯一run-id提交，不扫描端口或提前运行formal。

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

## 腿主导低轮速Recovery新任务（2026-11-14）

用户授权新10000更新、墙钟null。独立worktree `/home/kellen/wheel_leg_recovery_leg_ws` / `recovery/leg-driven-low-wheel` 从main `6df1353`建立；旧Recovery训练已用户停止于3149，新任务不挪旧余额。复用2100骨干/四腿输出，显式零初始化两个轮输出末层，新Critic942/双Adam；关闭旧大轮速动作prior与BC。顺序/全程真实轮速/姿态、1ms统计、分段辅助课程与候选门已实现；首次提交在初始观测处因Lifecycle reset误清执行限矩而FAILED/0更新；修复为仅重置声明的自有状态，失败run与原配方保留。后续attempt2分配仍为10000总额度（前段已耗0），不刷新额度。实际开训以对应run收据为准。当前本机另有Jump owner，未操作；用户已明确允许共享GPU，提交前仍核对显存/磁盘及唯一Recovery worker。主源/电控未改。


