# Jump Isaac GPU 执行链

用户已取消此前设计限制，并独立授权10000次PPO、任务总墙钟不限时。CPU交付`148bcbc`和云端MuJoCo兼容修复`bc95acb`保持原身份；GPU工作在独立worktree/branch `wheel_leg_jump_gpu_ws` / `jump/isaac-training`。不修改或停止活动Motion。

## 实际实现

正式入口仍是统一CLI→owner/worker→driver→JumpAdapter/GenericPPOAdapter。新增明确的训练身份`jump_isaac_ABI4_raw37_wheel48_v1`，旧CPU合同仍只开放MuJoCo。任务registry只增加该实际消费者与对应policy factory；没有新trainer、训练循环、watcher或后台编译服务。

Actor继承已蒸馏的大模型SHA `2a4d94517943559eaed51841e27be51e4672162a862ff07cd06979efc2e685fb`。37D、2048/1024/512、ELU、raw Normal、线性输出腿增益.5/轮10，输入1.5/100分开；不加tanh、残差或隐藏控制器。腿/轮限矩40/4.8Nm，完整V4与1/5/10ms保留。Critic单独初始化，双Adam/std/RNG仍由公共PPO保存恢复。

新Critic5591D/226字段，布局由真实C/V4状态类型和明确PhysX字段生成，实际引擎构造后逐字段核验。保留全部C/输入、V4/response队列、reward/event memory；删除MuJoCo专用qacc_warmstart、solver/material参数，改为具名实际root/joint/COM/接触/弹簧/闭链/力路由/物理参数读回。不是零填或把相同形状当作相同语义。Actor-only继承物理输出合同不变；CPU full checkpoint不可冒充GPU恢复。

PhysX场景直接复用Isaac的SimulationContext/InteractiveScene/Articulation/ContactSensor和已有PhysicalParameters。原USD和机械参数保持来源身份。motor通过唯一effort目标；NET和主动关节被动力通过成对body wrench，与motor cap分开。V4身体质量/惯量只应用一次。场景采用top Z=0的有限运动学薄板：SDK6.2的GPU过滤不支持普通plane collider，薄板提供可读回的真实ground-filtered接触矩阵。所有body-ground接触保留。自碰撞开关沿用来源YAML的False；没有把机械上互相交叠的闭链连杆全部启用自碰撞。

## C桥、采样与费用

真实C/FSM按一个n环境批次在CPU执行，V4 tensor response和物理/策略网络在GPU。C每行states/inputs/frames独立；不复制Python FSM。支撑来自实际PhysX状态→CPU闭链逆动力学模型，是明确的仿真传感器假设；contact truth只用于reward/诊断，不输入Actor支撑。硬件支撑observer仍未验收。

物理、原V4 histories和奖励每1ms更新。逆动力学只在5ms的实际C/FSM/PD消费者边界计算，Actor每10ms读取；没有消费者的中间逆解不执行。一次物理tick的读回打包为一个batch传输，复用至下一tick/reset；静态物理读回仅在本版本nominal场景复用，未来增加DR必须显式失效。

环境提前故障时，保存首次故障的真实terminal观测及回合时长；本批其他环境继续完成10ms。结束后的被动物理时间计入真实engine ticks，不冒充旧episode或合法Actor样本。partial reset不再次推进未reset行的C/FSM，也不刷新其策略观测。FAULT行保留C安全状态，零PD请求经过原电机响应，不以其他控制器接管。

学习恢复保留双Adam/std/RNG/course/候选游标/pending。引擎恢复明确为fresh-reset：旧未完成请求、离地/稳定时长不接续；已结束回合证据保留。不存在PhysX隐藏solver逐位恢复承诺。

## 训练与退出

一个formal阶段共10000次，总墙钟null；前几次通过公共`--max-updates`有界启动、保存、核验，再resume剩余额度。该安排避免同一Python进程重启Kit，也不额外增加smoke额度。32×64是初始资源配方，每次2048策略样本，总计划20480000样本；最终吞吐以实际公共measurement为准。

原reward与AdaptiveCourse复用：先低落体/5cm起跳，再5/10/15/20cm；按正常训练结束回合自动晋降级，balance/landing_only/active_jump分池。早失败进分母，不新增独立能力评估。未训练Actor失败是物理结果，工程验收不要求它已经稳定跳跃。

Kit6.2默认fast_shutdown会直接os._exit，不能在adapter.close中执行，否则公共worker来不及写预算和终态。领域只先关闭场景，将Kit关闭延至进程atexit，保留实际异常退出码。构造失败先保存原始traceback；没有覆盖原失败收据。只允许单formal阶段，公共worker仍是唯一生命周期执行者。

## 验证范围

CPU联合252项通过，新增实际n环境C桥验证跨128环形边界、partial reset、FAULT行隔离和完整C/V4继续；schema、真实大Actor继承及双Adam初始化/恢复通过。实际GPU接线版本v7/v8/v9分别通过2环境×400物理tick、80次C PD、partial reset及fresh恢复；保留全部失败版本和对应stdout/receipt。成功接线不是跳跃能力证明；早期物理失败仍如实记账。构造期间SDK内部warmup未逐步计数，不能报告全部准备阶段物理消耗为零。

服务器独立目录`/root/gpufree-data/wheel_leg_jump`。Motion目录仅作为已冻结SDK的只读来源，Jump数据/源码/模型索引/预算/run均独立；正式运行前逐SHA冻结。长训状态、实际消耗、性能与checkpoint以本任务run和`artifacts/jump_isaac`收据为准，文档不宣称尚未发生的训练结果。
