# Jump 失败排查

## 本机接替：大薄盒地面已定位并修复

本机 IsaacSim6.0.1 CPU/GPU 对照将轮地异常定位到单个128×128×0.1m运动学地面盒。保持完整机器人资产、质量、控制、reset与传感器不变，8×8×0.1m地面消除异常；轮轴等价改写未解决。正式实现将原128m地面包络分成8m无重叠tile，共用一个ground actor及材质，上表面仍为Z=0。尚未确认PhysX内部数值根因，不能将该结论扩展成所有Cylinder或所有articulation失效。

原始对照：`artifacts/jump_diagnosis_20261113/cpu_shapes_v15`、`cpu_swap_v20`、`cpu_axis_v21`、`cpu_small_floor_v22`。正式分块CPU/GPU对照分别为`cpu_tiled_floor_v23`、`gpu_tiled_floor_v24`：200 world ticks，最大轮净空穿入约1.1µm、闭链残差小于0.04mm、无非轮接触；tick128只reset行0，行1全部测量变化为0。独立轮体不同姿态/接触初态的`cpu_pose_v19`正常，保留全部失败证据。

正式host消费者每个已有读回边界检查非有限值、轮穿入>1mm及闭链残差>2mm，失败抛出工程异常，不作为可学习回合继续优化；不增加GPU回读或物理积分。倒下、非轮接触、零成功仍按原任务终止处理。临时scene query开关已删除。地面修复阶段合同为`jump_PhysX_tiled_flat_ground_v2`，后续C修复升级为末节的v3；完整恢复身份包含该合同。

`gpu_actor_tiled_v25`只诊断继承Actor的2秒控制：支撑估计接近实际接触力，但根高度降至约0.25m、长时间SETTLE，不证明已继承站稳/起跳。`gpu_normal_io_v26`沿正式环境完成3200 world ticks、真实1ms故障终态、partial reset和fresh reset，未做PPO或能力认证。

远端以无GPU模式恢复后，u407完整checkpoint及预算/run/终态已取回数据根`jump_handoff_20261113/remote_u407`。SHA `fa1a51169c3048260e720077a71ff41dd0fda359a11c9be26b1bd9025f4f5296`、74803819字节匹配。原远端run与账本只读。新物理版本只能显式full初始化，保留Actor/Critic/双Adam/std/全局RNG及已完成warmup；物理fresh reset。新物理cohort保留课程RNG、累计episode/request计数，旧故障物理窗口不用于新物理晋级，新run重新累计候选证据。同run恢复不允许借迁移声明绕过身份检查。

接替初始已耗407/10000、余9593；本机工程8次现已完成，累计415/10000、余9585，非重发10000。实际新增消耗由新run终态账本与后续收据绑定。远端无GPU，尚不能对修复后6.2的实际PhysX作通过声明。

以下保留修复前历史排查记录；其“尚未修复”“本机未启动”等描述只代表当时状态。

实际轮地接触异常尚未修复，不能继续原长训期待网络消除它。独立Jump已通过统一CLI的request-only协议在更新边界保存退出：407/10000次、833536策略样本、剩9593次，状态STOPPED_BOUNDED、failure=null。最终u407 SHA `fa1a51169c3048260e720077a71ff41dd0fda359a11c9be26b1bd9025f4f5296`，74803819字节。没有退款、刷新额度或新增PPO更新；Motion/Recovery没有被修改或停止。

本轮范围是冻结训练统计、原始资产与源码核对、两环境有界物理I/O和CPU工程修复。不做独立PPO能力评估，不用诊断回合晋级课程或发布候选。远端原candidate与run证据保持原样；实验改动分别放candidate_v2/v3，没有热改600261c的冻结源码。

## 轮地接触异常，根因待确认

最近u400统计窗口3734回合全部在第0档，1829主动跳/1905下落均为0成功；1687次终止码为5（support unknown）。有界诊断继续记录首次失败后的物理状态以区分原因，不能把这段额外时间算作旧回合或Actor样本。

原蒸馏Actor、固定初始腿目标、零motor/NET自由下落都出现“名义半径60mm的轮心下降至约22mm，轮接触读数仍为0”的现象，随后由lower_leg等非轮部位接地。以实际关节与root姿态重新计算全部轮FK，两环境最大位置残差分别`4.889e-7m/3.834e-7m`，轮胎净空残差均小于`3e-7m`，排除简单的关节顺序/四元数顺序/轮半径读回错误。用同一几何、质量、V4质量缩放和机械参数作MuJoCo自由下落参考：无落差轮心最低约59.96mm，28.24mm落差最低约58.10mm，没有远端约38mm的穿入。

将地面显式加入global collision group、关闭replicate_physics的对照均未消除异常，不能将这两项称为根因。原USD实际轮CollisionAPI开启，半径0.06m、轴向Z、厚度0.033m；没有发现过滤轮地配对的显式关系。C估计在部分真正接触时接近力传感读数（例如约265.9N对265.0N），因此也不能把所有支撑未知简单归咎于一个固定力符号错误。

根因仍需远端6.2实际碰撞体/readback一致性验证：核对ContactSensor的rigid-body view与contact view行顺序、原生碰撞形状/局部变换、相同轮圆柱的独立自由落体；对比未修改质量/COM之前、之后与reset之后的结果。必须保持原碰撞尺寸/质量/闭链/限矩，先证明接触正确，再重新考虑控制学习。不得用接触真值替代C支撑观测、放松非轮碰撞终止或直接增加“补偿弹簧”掩盖差异。

## Partial reset 的加速度串行污染，已完成CPU修复

实际表现：一个环境reset会增加reset epoch，导致同一物理tick重新读取全部环境。旧`IsaacScene.measured()`每次都将`(velocity-previous_velocity)/0.001`作为新值，并更新所有行的previous；未reset行此时速度未变，会被改成0加速度。`scene.update(dt)`还会在未积分时推进SDK的joint_acc时间，因此只保留root导数仍不充分。该值参与C逆动力学支撑估计与Critic；但尚未实测其对本次零成功的贡献，不能把它当成轮穿入的已证根因。

修复在`src/native_stand/simulation/jump_isaac_scene.py`的`TickDerivatives`、`IsaacScene.reset`与`measured`：只按实际physics tick求root和全部joint速度导数，同tick复读复用；partial reset只初始化选中行，保持其他行的导数和基线。没有推进物理、清空其他环境、增添滤波或改变1/5/10ms控制节拍。地面`collision_group=-1`同时显式表达全局地面；replicate_physics仍为原True，对照实验参数没有进入正式配置。

新增`tests/test_jump_isaac_derivatives.py`覆盖同tick复读、跨128tick边界的partial reset、连续reset epoch、全量reset、真实tick间隔、时钟倒退和实际Scene.reset方法的行选择。相关CPU回归34 passed in22.22s；架构212/0，Ruff0.8.4 check/format通过。修复未同步到任何活动/历史远端candidate，真实PhysX修复验收NOT_RUN。

## 蒸馏能力和阶段覆盖，不能假定已继承

真实来源为Actor-only SHA `2a4d94517943559eaed51841e27be51e4672162a862ff07cd06979efc2e685fb`。原始refined/fit.json明确`phase_columns_untrained=true`；离线目标RMSE约0.0083～0.0195rad（腿）和1.68～2.27rad/s（轮），不等于当前C/FSM/PhysX闭环认证。历史完整参考中，蒸馏模型可以完成10s受控记录，但站高最低约0.249m、期间经过FLIGHT/LAND/SETTLE，仍标`trained_jump_accepted=false`；不得将“完成记录”称为标称站立通过。

当前主动任务在1.5s才发跳跃请求（`jump_isaac.py:249`），u400窗口主动任务平均Jump控制时长仅0.553s；大量回合在请求前结束。BALANCE约15～21ms后进入FLIGHT会很快访问蒸馏未训练阶段。因此必须先修复物理接触/支撑输入，再从正常PPO结束回合核对站立保持和完整阶段覆盖；单看value loss下降不足以证明学习有效。此前8次有界检查证明真实优化/双Adam/恢复接线，但缺少这项物理一致性门槛，验收范围不充分。

## 资源状态与证据

远端最后一次可连接时已确认STOPPED_BOUNDED/407、worker已退出。随后`183.147.142.40:31193`持续Connection refused，用户已确认实例关机；远端物理验收待实例恢复。本机4080仍有Recovery owner PID2917966，公共资源前检拒绝本机诊断；没有绕过前检或发送信号。本机Isaac6.0.1与远端6.2不同，也不能互相冒充通过。

本轮远端共6个两环境、200world-tick的有界I/O诊断，即1200world-tick/2400env-tick、240次C PD batch调用；SDK构造warmup未逐步计数，不能声称准备物理总量精确为上述数字。本地MuJoCo参考400个物理步。本机Isaac两次尝试分别在配置检查、资源前检退出，AppLauncher和物理步均未执行。PPO新增0。

本地证据目录`artifacts/jump_diagnosis_20261113`含已取回的actor_v1/contacts_v1原始轨迹、FK_vs_native_readback.json、CPU_free_drop_reference.json、CPU_regression.txt、local_free_v4/v5失败日志和诊断脚本。远端六份trace位于preflight的`diagnose_actor_v1`、`diagnose_hold_v1`、`diagnose_contacts_v1`、`diagnose_global_floor_v2`、`diagnose_free_v2`、`diagnose_no_replicate_v3`；断线前尚未全部取回，保留这一缺口。

余9593次PPO保持未使用；本次不创建新训练run、不用历史已完成更新反推或改写账本。远端恢复后先完成原生轮接触与修复后partial reset验证，保留旧失败证据，再以新冻结源码安排正常训练中的有界接续，不再次申请已经授予的训练额度。

## 完整u407恢复的std逐位校验缺陷

本机首个新run `2026_11_13__jump_tiled_u407_local8_v2`在PPO之前FAILED，已耗0；失败与旧run保持只读。真实u407保存的GPU exp(log_std)与CPU重算在3通道各差1 ULP（7.45e-9），`learning/ppo/jump_unified.py::restore_state`原逐位比较将其误判为std冲突。现在允许派生exp值最多2 ULP舍入差异，仍严格检查shape/dtype/有限正值；完整恢复使用原始log_std，不重建或替换std/Adam。实际u407验证通过，新增分布被改、NaN/Inf/零、shape/dtype冲突仍拒绝的CPU回归。后继本机v3仍只占用原8次工程划分，不因失败增加总预算。

## C支撑可用性与接触分类混淆：已修复，能力仍未认证

本机v3完整u407初始化后4次PPO正常STOPPED_BOUNDED；非零reward、每次Actor/Critic各16个Adam步、warmup仍5，完整学习状态/双Adam/std/RNG/预测逐位恢复通过。实际累计411/10000、剩9589，保留0消耗的v2恢复失败run与已耗4次的v3run；没有退款。正常PPO的42个结束回合中，19个主动跳均在25ms左右C_fault=5，23个下落timeout且零成功，不能放行长训。

`gpu_u407_actor_v34/trace.json`显示25ms时两侧支撑残差有效，且与真实轮力接近；右侧先约19N、15ms超过20N、20ms变OFF，25ms仍在8ms去抖。根因是`jump_control_source/rl_jump_core.c`把contact_valid的去抖/滞回未确认当成估计不可用，用phase年龄24ms故障。共享C现在分别表达估计可用性和接触确认：真正缺失/过期/奇异估计仍触发原故障；分类未确认仍禁止站稳确认、起跳、着陆事件，但不冒充传感失效。8/20N、5/8ms、100ms进入、1s稳定、力矩/Actor输入输出/1-5-10ms均未改。

真实C旧/新库回放同一力序列，旧库25ms故障、新库30ms正常完成分类（`support_replay_v35/receipt.json`）。GPU `gpu_support_valid_v36`2500tick两环境未发生该早期故障；仍长时间SETTLE，未证明0.34m站稳或起跳。执行合同更新为`jump_PhysX_tiled_flat_ground_support_valid_v3`，C manifest绑定新源码SHA与controller revision，ABI4布局及Actor输出合同不变，硬件输出仍OFF/UNVERIFIED。三类候选只消费正常PPO回合；新增C请求/接受次数仅记录实际事件，不作为额外评估。

原8次工程划分余下4次从v3/u4完整学习状态启动新cohort；所有实际消耗继续累计。新物理证据不沿用旧失败域的课程窗口，课程RNG和累计episode/request计数保留。后续验收与预算以对应run/artifacts收据更新。

C修复后的第一次联合回归为146通过/7失败，7项均由两份现用配方的execution_manifest仍绑定旧SHA触发身份拒绝；没有放宽校验。已原地将两份配方绑定当前worktree的新manifest，复跑155通过（50.16s）。历史run/spec及旧SHA证据保持原样。

## 最终工程交付与远端剩余条件

修复冻结`ab6dc16`，601个Python模块SHA `284a5706b91f7522ccd8a23506dffe4c60b1515228f05832dd28c97bcc40e854`本机/远端一致。四个修复提交为`ac64b8c`、`5cd213f`、`11a008d`、`ab6dc16`。最终主仓只读核对`fc29da3`，后两次主仓提交仅文档，不冒用main验收或合入声明。

两段本机实际PPO各4次，第二段2+2经同run完整resume，均正常终态且failure=null。Actor/Critic双Adam累计6560/6640步、warmup仍5；完整模型、优化器、std、Python/NumPy/CPU/CUDA RNG和预测逐位恢复通过。最终完整checkpoint SHA `2ea1f5c9f42d2e7519e6daa41fc7850c8b1188679145fc536ec71c3c1c276f42`，72438785字节。**累计415/10000、剩9585**，原失败v2用0次、旧v3用4次，旧未用局部划分在任务结算中退役，不是额外预算。

最终相关CPU155通过（50.16s）；portable1915通过/31历史skip/10warning（141.18s）；Ruff0.8.4 check及211文件format通过；当前main检查器按refactor-migration-v1检查212正式文件/0例外。旧7项manifest身份失败、旧C早故障和std恢复失败均保留，不删除证据。

本机CPU/GPU逐tick自由落体冲量验收各200tick：轮净空最小约-2.02µm，闭链最大0.177mm；显式计入partial reset动量突变后竖直冲量平衡残差小于0.012Ns。落差行首次轮接触tick76，零落差行tick1；弹簧长度观测0.15339～0.18405m。独立气弹簧力辨识仍未知，不修改aggregate V4或增加补偿。

远端实际IsaacSim6.2.0-rc.14无GPU下已完成200tick CPU PhysX，轮穿入约0.61µm、闭链0.038mm、partial reset未reset行0差；22项CPU回归通过（12.10s）。最终模型、33项job依赖、三段预算逐SHA核验，正式train dry-run通过，新long run尚不存在。材料位于`/root/gpufree-data/wheel_leg_jump/preflight/ready_support_v44`，代码位于`candidate_support_ab6dc16`。CPU只读收据中的physical_steps=0仅指该次静态preflight，不覆盖另存的200tick CPU物理probe。

继承Actor仍主要SETTLE、约0.25m，没有站稳/主动跳证书。按原Stage0密集站立/落地奖励与raw Normal探索继续学习，不加入隐藏Teacher、滤波或新训练循环；原合同successful_teacher_required_for_PPO=false，零成功不单独否决未训练Stage0。候选仅消费正常PPO回合，当前三池accepted均0。

**尚未通过的开训门槛**：远端GPU/真实UUID/驱动挂载恢复后，在新6.2版本做零PPO物理I/O与8次正常PPO有界完整学习复验；检查请求覆盖、系统性早故障及保存/恢复后，再在同一run续训，继续扣减9585。没有自动开GPU、创建watcher或启动远端训练。命令和边界见材料目录NEXT_STEPS.md。完整文件清单、证据表与机器摘要见`artifacts/jump_diagnosis_20261113/{DELIVERY_20261113.md,delivery_final_v46.json}`。

## 站高奖励与请求调度复核（后续增量）

上一节“保留原Stage0密集站立/落地奖励”不代表奖励已覆盖所有失败路径。本轮沿正常PPO消费者核实两个训练缺口，并保留旧结果：

- 意外离地没有合法takeoff时，`DenseReward.has_landed`保持false；原SETTLE站高惩罚也被这个门控关闭。`UnifiedReward` v2在真实C的SETTLE阶段独立计算原权重/尺度的站高惩罚。合法起跳、落地稳定、active/drop成功与own completion门控不变，不给意外下落补发正奖励。
- Isaac原1.500～1.520s固定请求可能在SETTLE时被C拒绝busy。新Isaac调度在1.5s后首次BALANCE发一次请求；实际C的request_id锁住本回合后续重复发出。原静止、支撑、姿态、100ms进入确认和500ms等待超时不变。尚未回BALANCE仍可能失败，不能把“请求待发”算作接受或起跳。CPU历史播放器/手动请求的20ms按键语义未改。

Isaac执行identity加入reward与request调度语义，同run不允许旧版本直接恢复。新run采用既有显式full transition保留学习状态、课程RNG与累计身份，开启新证据cohort，物理仍fresh_reset；本轮物理参数未变。候选比较域包含reward版本。两份现用配方显式改为v2，旧run/spec与远端ready_support_v44保留，不得继续拿旧配方作新版启动包。

真实C回归覆盖请求在busy窗口保留、回BALANCE消费一次、合法进入CROUCH及reset重新资格；奖励回归覆盖意外下落低高度有代价、站高改善代价下降、时间切分一致、无伪造落地/跳跃成功。Jump相关CPU83项通过（24.46s）。新版正常PPO32次有界验收待执行，拟先30次保存、同run恢复2次，全部从原9585余额扣减。远端本轮再次核实Torch2.11.0+cu130、CUDA=False/count0，未启动远端训练。

## 并行度与采样性能（后续增量）

用户要求尽量利用机器。32×64是诊断配置，不是最大并行度。新版v2正常PPO先在14次更新边界保存，再从完整u14同run恢复2次，实际16次均正常STOPPED_BOUNDED/failure=null，累计431/10000、剩9569。原32次局部上限的未用16次退役，不能再与余额叠加；原定30+2改为14+2，是为转入用户要求的大并行测量。原16s完整任务时长未跑到；当前active仍多为SETTLE阶段超时、请求未发，没有站稳/起跳证书。

32×64原计时约32s/update，rollout约31.7s、学习约0.11s。观测打包约12.9s；物理推进约7.6s、读回约3.7s，子区间有包含关系，不能相加。GPU利用率同时包含另一owner的Recovery，不能据整卡利用率判断Jump已吃满GPU。

新增优化仅用于Isaac Critic：按dtype合并V4/response的全部行读回，再按原字段顺序切行；PackedCritic保留全部5591列、float64归一化后float32输出及name/shape/finite检查，只把多次finite扫描合并。没有减少物理tick、奖励频率、队列或历史。首次CPU对照发现按dtype分组造成字段重排，已修复并保留失败收据v50；v51逐位对照通过。批量snapshot只在一次pack操作内使用，不跨同tick的observe/reset缓存。真实GPU两环境320步I/O v53通过，16次direct-vs-batched Critic逐位一致，含partial reset及fresh-reset恢复；PPO=0，非能力评估。

显式full transition新增环境扩容声明：原行保留课程RNG/累计身份，新增行以job.seed+row独立播种，没有伪造旧成绩；全部从新cohort开始。只允许声明后的扩容，不开放隐式缩容或同run变更环境数。Actor/Critic/双Adam/std/global RNG仍走原完整加载。

相关CPU92通过（27.62s）、portable1925通过/31历史skip/10warning（136.74s），Ruff0.8.4及架构通过。拟同一完整u16源、32/128/256环境各2次正常PPO，属于原余额内6次互斥运行，不并发启动Jump；实测吞吐和显存后决定长训并行度，不能把CPU微基准3.8倍收益外推整段训练。
