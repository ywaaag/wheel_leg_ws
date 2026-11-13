# Jump Isaac GPU 执行链

当前新增显式 `assets.jump_wheel_collision="wheel_convex30_v1"`，由正式 Jump factory 传入 Scene 与 execution identity。运行时仅将两轮 Cylinder 碰撞改为30侧外切凸包；原USD、轮心/轴/宽、质量/COM/惯量与sensor路径保留。60mm半径增加0～0.331mm，不能宣称几何完全不变。本机CPU/GPU cooked读回均60顶点/32面，20000方向support差7.6nm～330.501um。旧配置未声明时保留原Cylinder，供冻结来源与对照使用；两种物理合同不能无声明full互换，需显式transition与新cohort。

最终候选零PPO4096环境6400tick通过，6634个正常结束行，Critic54次逐位对照、inverse48次容差对照、fresh reset通过。CPU/GPU各两环境400tick，partial reset未reset行全部测量差0；最大闭链分别0.185/0.141mm，GPU最小解析净空-1.628um，冲量竖向余额最大0.0268Ns。仍逐1ms执行1mm穿入/2mm闭链门禁，不改变奖励/控制/限矩。证据`artifacts/jump_diagnosis_20261113/convex30_acceptance_v150.json`。这是物理接线验收，未认证站稳/跳跃；当前正在完成正常PPO与完整恢复验收。以下旧版本结论按各自冻结身份读取。

当前执行修复冻结`ab6dc16`，合同`jump_PhysX_tiled_flat_ground_support_valid_v3`。接替后新增8次PPO含真实同run完整resume，累计415/10000、余9585，最终完整模型已搬迁远端并核对SHA。新版本本机CPU/GPU与远端6.2 CPU工程通过；远端无GPU，GPU复验和有界窗口尚未执行。0.34m站稳与主动跳能力未认证。最新收据及原始失败见`JUMP_DIAGNOSIS_20261113.md`末节和`artifacts/jump_diagnosis_20261113/DELIVERY_20261113.md`。下方旧长训RUNNING等为各自日期历史快照。


用户已取消此前设计限制，并独立授权10000次PPO、任务总墙钟不限时。CPU交付`148bcbc`和云端MuJoCo兼容修复`bc95acb`保持原身份；GPU工作在独立worktree/branch `wheel_leg_jump_gpu_ws` / `jump/isaac-training`。不修改或停止活动Motion。

## 实际实现

正式入口仍是统一CLI→owner/worker→driver→JumpAdapter/GenericPPOAdapter。新增明确的训练身份`jump_isaac_ABI4_raw37_wheel48_v1`，旧CPU合同仍只开放MuJoCo。任务registry只增加该实际消费者与对应policy factory；没有新trainer、训练循环、watcher或后台编译服务。

Actor继承已蒸馏的大模型SHA `2a4d94517943559eaed51841e27be51e4672162a862ff07cd06979efc2e685fb`。37D、2048/1024/512、ELU、raw Normal、线性输出腿增益.5/轮10，输入1.5/100分开；不加tanh、残差或隐藏控制器。腿/轮限矩40/4.8Nm，完整V4与1/5/10ms保留。Critic单独初始化，双Adam/std/RNG仍由公共PPO保存恢复。

新Critic5591D/226字段，布局由真实C/V4状态类型和明确PhysX字段生成，实际引擎构造后逐字段核验。保留全部C/输入、V4/response队列、reward/event memory；删除MuJoCo专用qacc_warmstart、solver/material参数，改为具名实际root/joint/COM/接触/弹簧/闭链/力路由/物理参数读回。不是零填或把相同形状当作相同语义。Actor-only继承物理输出合同不变；CPU full checkpoint不可冒充GPU恢复。

PhysX场景直接复用Isaac的SimulationContext/InteractiveScene/Articulation/ContactSensor和已有PhysicalParameters。原USD和机械参数保持来源身份。motor通过唯一effort目标；NET和主动关节被动力通过成对body wrench，与motor cap分开。V4身体质量/惯量只应用一次。场景采用top Z=0的128m有限运动学地面：用8m无重叠薄盒tile共用一个ground actor与材质，避免单个128m薄盒在本机CPU/GPU对照中的轮接触丢失。SDK6.2的GPU过滤不支持普通plane collider，因此保留可读回的ground-filtered接触矩阵。所有body-ground接触保留。自碰撞开关沿用来源YAML的False；没有把机械上互相交叠的闭链连杆全部启用自碰撞。

## C桥、采样与费用

真实C/FSM按一个n环境批次在CPU执行，V4 tensor response和物理/策略网络在GPU。C每行states/inputs/frames独立；不复制Python FSM。支撑来自实际PhysX状态→CPU闭链逆动力学模型，是明确的仿真传感器假设；contact truth只用于reward/诊断，不输入Actor支撑。硬件支撑observer仍未验收。

物理、原V4 histories和奖励每1ms更新。逆动力学只在5ms的实际C/FSM/PD消费者边界计算，Actor每10ms读取；没有消费者的中间逆解不执行。一次物理tick的读回打包为一个batch传输，复用至下一tick/reset；静态物理读回仅在本版本nominal场景复用，未来增加DR必须显式失效。

环境提前故障时，保存首次故障的真实terminal观测及回合时长；本批其他环境继续完成10ms。结束后的被动物理时间计入真实engine ticks，不冒充旧episode或合法Actor样本。partial reset不再次推进未reset行的C/FSM，也不刷新其策略观测。FAULT行保留C安全状态，零PD请求经过原电机响应，不以其他控制器接管。

学习恢复保留双Adam/std/RNG/course/候选游标/pending。引擎恢复明确为fresh-reset：旧未完成请求、离地/稳定时长不接续；已结束回合证据保留。不存在PhysX隐藏solver逐位恢复承诺。

## 训练与退出

一个formal阶段共10000次，总墙钟null；先通过公共`--max-updates 8`有界启动、保存、核验，再resume剩余额度。新Critic前5次只做Critic预热，随后检查双Adam均有真实更新；预热和有界8次均计入10000，恢复不重做预热。该安排避免同一Python进程重启Kit，也不额外增加smoke额度。32×64是初始资源配方，每次2048策略样本，总计划20480000样本；最终吞吐以实际公共measurement为准。

原reward与AdaptiveCourse复用：先低落体/5cm起跳，再5/10/15/20cm；按正常训练结束回合自动晋降级，balance/landing_only/active_jump分池。早失败进分母，不新增独立能力评估。未训练Actor失败是物理结果，工程验收不要求它已经稳定跳跃。

Kit6.2默认fast_shutdown会直接os._exit，不能在adapter.close中执行，否则公共worker来不及写预算和终态。领域只先关闭场景，将Kit关闭延至进程atexit，保留实际异常退出码。构造失败先保存原始traceback；没有覆盖原失败收据。只允许单formal阶段，公共worker仍是唯一生命周期执行者。

## 验证范围

CPU联合252项通过，新增实际n环境C桥验证跨128环形边界、partial reset、FAULT行隔离和完整C/V4继续；schema、真实大Actor继承及双Adam初始化/恢复通过。实际GPU接线版本v7/v8/v9分别通过2环境×400物理tick、80次C PD、partial reset及fresh恢复；保留全部失败版本和对应stdout/receipt。成功接线不是跳跃能力证明；早期物理失败仍如实记账。构造期间SDK内部warmup未逐步计数，不能报告全部准备阶段物理消耗为零。

服务器独立目录`/root/gpufree-data/wheel_leg_jump`。Motion目录仅作为已冻结SDK的只读来源，Jump数据/源码/模型索引/预算/run均独立；正式运行前逐SHA冻结。长训状态、实际消耗、性能与checkpoint以本任务run和`artifacts/jump_isaac`收据为准，文档不宣称尚未发生的训练结果。

## 实际长训交付

冻结训练源码提交`600261c5f85e85b5f3944e23c5db09066b140381`，601个Python模块摘要`7f92a8791d430b90d4a79784c08f9b4d7a325ceb9511c8c41b3d81c71d52da1c`，本地与远端一致。远端run为`/root/gpufree-data/wheel_leg_jump/runs/2026_11_13__jump_isaac_cloud10000_v1`。worker通过统一CLI启动；没有额外launcher、watcher或独立PPO能力评估。

首8次正常`STOPPED_BOUNDED`，16384策略样本，failure=null。实际完整u8 SHA `6c5571c30aa46124da6ee79e34d4b3e433ace23d3c97ac9ab325f4b6041c31fc`，72637311字节。前5次Critic预热，随后3次Actor更新；累计Actor/ Critic Adam分别48/128步。Actor最大权重变化0.0002681208，载入后的Actor/Critic输出、双Adam、std、CPU/CUDA及Python/NumPy RNG均逐位一致。此检查没有增加物理步或PPO更新。

已从该u8完整状态续训同一run，预算没有刷新。交付快照为`RUNNING`、累计10/10000、20480策略样本，failure=null；续训worker PID3392。已落u9 SHA `a91018452bd48f9723ba6a07004c81bf5cea8bca759e069d891cb9136d85d58b`，Actor/Critic Adam64/144，warmup仍为5，没有重复预热。快照会过时，最新值以run内state/budget为准。

物理GPU为4090 UUID `GPU-07694748-3d8b-666b-3933-6443416715dd`，Jump进程实测占用3418MiB；没有修改Motion文件或进程。首8次平均54.31秒/更新；u8观测打包25.10秒、rollout51.91秒、学习0.092秒。这是当前共享服务器上32×64的实际耗时，不是GPU后端相对MuJoCo的提速结论。按此短窗口外推余下约6.3天，仅作资源规划。

本地收据为`artifacts/jump_isaac/bounded8_checkpoint_receipt.json`、`long_running_receipt.json`、`source_600261c_frozen.json`。训练统计候选仍由正常结束回合按原三池发布，当前capability_accepted=false，不能将接线和有效优化等同于已经稳定跳跃。后续重点看正常训练统计中的站稳时长、有效起跳/落地闭环与课程晋降级；DR、运动跳和障碍跳尚未实现于本版PhysX场景。

交付复核发现`jump_backend.preflight`的返回标签沿用“MuJoCo CPU/Isaac_GPU=False”，实际GPU消费者已经运行，错误仅在主机预检收据（修复位置`src/native_stand/simulation/jump_backend.py`）。本地后续小提交让标签跟随实际后端，并显式`GPU_initialized=False`，避免把CPU预检称作GPU验收；没有向活动远端热同步该修改，训练仍冻结在上述600261c。

CPU减负原提交`148bcbc`已由公共负责人以`bb22253`集成到主仓；本次实际核对主仓HEAD已推进到`1cc65e1`，未恢复到历史16f81c。新集成只需按序承接`bc95acb`、`707b6bf`、`600261c`及本交付小提交，公共3提交和已集成CPU改动不重复提交。GPU在冻结快照上通过；与最新主仓其他领域增量的联合验收应由集成方完成，不能拿本分支的252项代替该结果。

## 本机接替后的物理合同

当前物理合同`jump_PhysX_tiled_flat_ground_v2`、CPU/GPU接触与reset证据及u407取回见`JUMP_DIAGNOSIS_20261113.md`首节。正式host读回增加物理不变量拒绝门槛；没有新积分、控制补偿或奖励改动。新run允许SHA绑定的完整学习状态迁移及显式新物理cohort，旧run恢复仍严格检查原身份。远端6.2修复后GPU复验尚待资源。

共享资源helper仍位于`simulation/recovery_isaac/backend.py`。零PPO两环境诊断须双重具名授权；正式Jump共享训练另须policy与SHA绑定授权同时标记`shared_training_task=jump`。Recovery默认调用仍拒绝Recovery冲突，Jump具名调用拒绝同任务或未知任务owner，不把诊断例外扩大为默认并发放行。

C支撑修正执行版本为`jump_PhysX_tiled_flat_ground_support_valid_v3`。真实C区分新鲜可解的力残差与滞回/去抖未确认的接触分类；前者不可用仍故障，后者仍阻止相关FSM事件。接口ABI4与Actor合同不变，C源码及构建manifest SHA明确变更；部署必须携带对应新C包，不能混用旧库。硬件验收没有执行。正常结束回合额外记录C请求次数、接受次数和终态phase，统计不增加推理、积分或独立评价。

## 当前训练语义修订

`UnifiedReward` v2对SETTLE独立应用站高惩罚，不再以成功起跳来源作为站高约束前提；所有正事件和成功门控保留。Isaac主动任务在1.5s后首次BALANCE向C发送一次请求，request_id消费后不重试。此调度只安排任务请求，不控制电机或放宽C进入条件。执行identity包含reward和调度语义，候选比较域包含reward版本；旧版本只可经显式full初始化进入新证据cohort，同run拒绝混用。

Isaac Critic现按dtype批量读回全体V4/response行，保持原字段顺序和全部5591列；PackedCritic与原归一化编码逐位一致。snapshot只覆盖一次观测或终态打包，不跨observe/reset复用。扩容必须在既有full transition中声明environment_expansion，保留旧行课程RNG/计数、新行独立seed；同run仍不能改变环境数。性能验收以正常PPO的有效样本/s、整轮墙钟和峰值显存为准，不以GPU利用率或单个CPU组件收益替代。

当前用户指定4096环境：Isaac配方4096×64，minibatches64。物理合同v4按3m间距的clone网格扩展ground，保持每侧至少32m出生点余量；4096对应256m方形、8m tiles，实际最小33.5m，地面Z/材质与机器人不变。构造时核对实际origins余量。performance_measurement同时记录既有mask有效/无效数及Torch allocator峰值（不含全部PhysX显存）。远端仍需GPU恢复后的真实6.2复验，不把本机4096通过冒充远端通过。

逆动力学桥的腿长采用`ClosedChain.leg_lengths`，与完整measure相同的轮心FK，仅消除未消费的弹簧/碰撞/COM计算及非祖先关节。宽角度腿长与随机完整C输入字节回归比较，完整物理/支撑模型及周期不变。跨本机/远端引擎版本可显式full transition声明force_new_cohort，即使张量布局identity相同也不能沿用能力窗口；保留学习状态和课程RNG/累计计数。

高频奖励包保留原有全部外部边界校验，finite合并单次扫描，异常时定位字段。Isaac Critic静态字段单位/尺度只在首个实际观测构造，动态name/shape/finite每次保留。allocator峰值使用公共MetricsWriter已有peak_memory_*字段，mask样本数进入performance_counters，不使用会被writer丢弃的自定义标量名。


## GPU Critic组装与逆模型最小计算

正常策略观测经`IsaacJump.observations → BatchCritic.pack`，保留实际PhysX/V4/response张量在设备上。原5591列顺序、环形队列时间顺序、全部字段和float64归一化后float32输出保持；CPU侧真实C/FSM和dense reward记忆合并一次上传。首观测按RowCritic建立schema；RowCritic仍用于终态和诊断同tick逐位对照，因此不是删列或以常量代替实际历史。既有base TaskMemory本来每次由request新建，保持该合同，live dense记忆仍读取实际reward。

InverseFeedback仅调用位置/速度相关MuJoCo forward阶段，用实际PhysX有限差分加速度计算逆模型，不运行被丢弃的约束/前向加速度。SupportWorkspace只缓存拓扑索引和每次覆盖的scratch，作用域为单个inverse实例，未缓存动态反馈。真实C/FSM、支持判定、奖励/课程与终止语义不变。没有新增trainer或worker；统一CLI/owner/公共driver保持。CPU逆模型与dense reward仍存在，不能声称全部GPU化；吞吐以对应正式run记录为准。


## 显式全设备计算合同

配置`assets.jump_compute="cuda_firmware_warp_inverse_v1"`选择同一Isaac后端的设备计算版本。正式入口仍为CLI→owner/worker→公共driver→JumpAdapter；没有新增trainer。未声明时保留CPU-C/逆模型消费者，供冻结来源和历史恢复对照。未知值拒绝，不静默回退。

- C/FSM与PD：jump_cuda_build先调用原SHA/ABI构建校验，再只为私有副本的函数声明添加CUDA device限定。原C函数体、ABI4、阈值、动作解码和40/4.8Nm保持；每行一个CUDA线程，禁止fastmath/FMA。当前仅验收sm89。输出仍hardware OFF，非固件部署。
- 逆动力学：MuJoCo Warp仅执行kinematics/com_pos/crb/com_vel/passive/rne和Jacobian，完全不调用step、碰撞或积分。状态和加速度来自实际PhysX。闭链切空间pinv以原rcond1e-9在double执行；同CAD六祖先链计算腿长。只有Isaac/PhysX推进世界。Warp数值原语使用float32，不能称CPU double逐位等价；compute合同进入execution identity，新run需显式来源和force_new_cohort，不继承旧成功窗口。
- 奖励/终止：tasks.jump.device_reward在设备保存完整Dense/Unified记忆和previous packet；double累计与原标量顺序一致，已结束行冻结；仅正常结束/序列化导出标量参考，原once/来源/成功门控保持。
- Critic：实际C连续ABI数组直接在设备解码，onehot/uint32相对年龄/全部5591列在设备组装；GPU奖励记忆直接接入，不经逐行Python。RowCritic保留作终态/逐列参考。
- 物理检查：原1mm轮穿入/2mm闭链/finite门禁在每个真实tick设备归约，只读回少量标量。超限仍调用原host拒绝并保存原始故障packet，未放宽阈值。课程、已结束回合、checkpoint和owner资源管理仍CPU，不声称整个进程没有CPU。
- 同步：Warp运行使用当前Torch CUDA stream，初始化一次同步，避免qpos/质量/Jacobian跨stream竞态。

checkpoint完整学习状态保持，物理仍fresh_reset，不伪造hidden solver恢复。CUDA版需nvcc、MuJoCo Warp和Torch compile；当前本机IsaacSim6.0.1/sm89通过不代表远端6.2通过。具体运行性能与预算在run/artifacts记录。
