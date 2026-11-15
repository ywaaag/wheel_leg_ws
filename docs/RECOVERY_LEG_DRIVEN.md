# 腿主导低轮速Recovery

本配方是独立新任务：合法Swing→收短→短腿转向承重构型→伸腿抬升→接稳0.5秒。不是旧Recovery自主保持10秒能力的重新认证。Motion实际接管仍需独立连续交接验收。

正式入口：`wheel-leg train --config configs/jobs/recovery_leg_driven2100_10000_attempt3.json --dry-run`；实际执行同一CLI、owner/worker、公共driver和PhysicalPPO。几何/奖励/课程在`tasks/recovery/leg_driven.py`，Isaac环境负责真实1ms几何、轮速、姿态、动力学读回，不新增训练循环。完整恢复仍经`wheel-leg resume`，保存阶段、课程级别、1ms统计、双Adam及执行状态。

Actor为31→2048→1024→512→6；输入列、真实dq、实际采用previous和年龄不变。输出仍为固定中心三次腿绝对目标与100×轮mean；轮Kv=.2、PD40/2、40/4.8Nm及1/5/10ms、Actor20ms、目标5ms、轮PD后7/11ms保持。没有运行时轮速裁剪、虚拟锁轮、额外滤波或补偿。

初始化明确复用2100骨干和四腿输出，两个轮末层行（weight与bias）归零；不是2100 Actor整体精确复制或完整训练恢复。此变换只初始化新策略，保留轮输出包络和必要制动力矩。旧Swing动作prior与BC关闭，新Critic/Adam从新任务开始，来源与变换另有SHA绑定产物。旧3149次消耗不退款。

新Critic为显式942D：原904D因果前缀加38列，依次为阶段onehot4、阶段dwell1、初始几何4、上一几何4、几何速度4、课程级别1、初始阶段1、顺序合法1、两轮速度平方积分2、两轮速度峰值2、累计时间1、水平时间1、倾角峰值1、大倾角持续1、超速持续1、上一垂速1、阶段开始时间1、阶段进度1、两轮RMS2、当前两轮速度2、本策略步物理计数1、物理成本累积1。不是把旧909D含义静默替换。只在合法保存边界恢复；未暴露PhysX隐藏solver逐位状态。

阶段单向推进，收腿/转腿/抬升门保持100ms，接稳门500ms；没有新增Actor阶段输入。顺序由奖励与验收引导，不是执行层硬约束。收缩与摆角存在耦合，收腿期允许沿USD求解参考路径的必要转动；不要求不可达的严格固定摆角。安全短腿候选/转腿构型由绑定USD闭链FK/IK得到，初始root按真实碰撞几何落地；离线几何不冒充PhysX接触证书。

奖励为阶段势差、首次合法完成事件、全程有界成本和终态事件；不沿用旧快速抢高度奖励或旧MAX_DENSITY成功尾奖。真实轮速40%、全程倾角20%、角速度8%、顺序10%、腿运动5%、腿目标变化4%、抬升冲击5%、腿effort3%、时间5%。轮目标幅度/差分不作新约束，避免删掉制动力矩。峰值、RMS和倾角由1ms子步统计；不以10ms末帧漏采制造低轮速。失败剩余时间只记reward结算，不虚增物理消耗。

C0→C3按4/2/1/.5rad/s成本尺度收紧，完整Swing比例30%→70%→85%，辅助初态按独立episode标识；C4/C5才加入名义/0.1及0.25DR，分条件统计。每100更新完整窗口，完整Swing至少256结束回合、辅助每组至少64；连续3窗过成功/顺序/质量门才晋级，DR必须同时覆盖名义与stress；覆盖不足UNKNOWN不晋级。训练窗口长期无改善由公共stop request有界停止，不刷新10000额度。不按reward上涨选模。

最终目标：每轮真实轮速RMS≤.5rad/s、峰值≤2rad/s；全程倾角峰值≤15°、至少95%时间≤10°；合法顺序完成并稳定0.5秒。低速不能抵消趴地，成功也不能抵消过度倾斜。每个候选绑定课程级别/条件，辅助reset不能当完整Swing成绩。训练与硬件成功分开报告。

本机资源/授权/几何与CPU收据位于`/home/kellen/wheel_leg_training_data/recovery_leg_driven_20261114/preparation`，具体run身份、实际消耗和真实物理检查取对应run/state/budget，不从本说明推断正在训练。用户授权新10000更新、墙钟null；磁盘/共享GPU/停止仍经提交前检，不停止他人任务。

工程接线修正：worker显式将新reward分派到RecoveryAdapter，公共rollout通过观察回调提供真实transition，课程/候选仍由adapter保存。first12检查仅报告真实控制、奖励、结束与优化器/权重变化，不声称梯度分解诊断。完整来源保留warmup完成数，不按新run重新执行五次Critic-only；u6接续额度9994，源回合未完成部分fresh reset。

## 趴地惩罚准备配方（2026-11-15）

`configs/jobs/recovery_leg_driven2100_anti_prone_prepared.json`通过现有`reward.fields.anti_prone`启用`recovery_anti_prone_v1`。旧配置未启用时奖励逐项保持原值。正式调用链为CLI/JobConfig→Recovery backend校验与cfg→环境Lifecycle.observe；不新增训练入口或物理循环。

仅收腿/转腿阶段、未满足当前阶段门、root高度低于0.26m时生效；阶段开始后0.8s宽限，再用0.8s线性增加。低于0.22m时高度因子为1，0.22～0.26m线性退出。惩罚密度为`-2 × ramp × height_factor × (1-0.5×phase_progress)`，单位reward/s。正常阶段门内100ms停留不罚，阶段推进重启宽限，抬升和接稳阶段不罚。关节抖动和轮速提高不提供豁免。高度只是惩罚掩码，抬高至0.26m不能替代现有顺序、成功与质量门。

失败额外结算`-2×max(6-t,0)`，对所有失败初态一致，避免提前跌倒抹去新增的最大未来成本；成功不扣此项。奖励读回新增`density.anti_prone`与`anti_prone_failure_tail`。不新增隐状态或Critic列，沿用已保存的阶段、阶段起始时刻、几何进度及真实root高度；不变更Actor、decoder、scale、previous、时序、PD或限矩。

这只是可试验的奖励假设，尚未证明能克服hip饱和/顺序不可达问题。prepared配置为零分配，不会自动重启。原attempt3因停滞停止于1400，连同前段6次共耗1406，余8594不退款。配置所列初始Actor仅用于原始配方装配，不代表已决定重置策略；下次实际训练来源、奖励变更后的Critic处理及新run预算分配仍待准备。训练、PhysX、独立验证和部署均NOT_RUN。恢复旧完整点时不可直接继承不同奖励比较域的候选窗口。

## 阶段推进接续配方（2026-11-15）

正式配置`configs/jobs/recovery_leg_progress_remaining8594.json`，新worktree `/home/kellen/wheel_leg_recovery_progress_ws`，从当前main `17e1ac7`建立并承接已验证的Recovery增量。旧worktree/run及u1400只读。用户明确要求复核遗漏并重启，原10000额度已耗1406，新run仅分配8594，墙钟null。共享RTX4080仍经UUID/唯一Recovery worker/空闲显存8GiB/磁盘8GiB门检，不停止Jump或操作Motion远端。

复核发现：收短参考hip本来就需要Tanh mean约±0.974，不能把`abs(mean)>.95`直接当动作包络不足；完整Swing、收短、转腿、站立的主动关节参考均在±1.5rad内。63个CPU闭链插值样本没有关节限位违规，最大closure约5.62e-6m；收腿路径的摆角误差≤0.0166rad，转腿段长度不超过短腿允许边界。该结果不认证自碰撞、接触和动态可行性。u1400在不同初态均收敛至约0.23m腿长、约-0.92rad摆角，原势差奖励在超时终点归零，只有弱时间成本，少动降低成本的局部解仍存在；此为需由新训练验证的因果假设。

新配方同时启用上节anti_prone与`reward.fields.progression=recovery_stage_progress_v1`：每策略步增加`-3×(1-phase_progress) reward/s`，当前阶段门成立时不扣；合法完成收腿/转腿/抬升的事件奖励分别6/6/10。收腿门补短腿摆角±0.1rad，转腿progress以真实收短参考为起点（不能沿用长腿Swing摆角）；顺序一旦违规不可恢复即结束失败回合，不继续采满6秒。失败剩余结算统一为`-9×max(6-t,0)`，替代原基础与anti_prone两项尾款，9覆盖所有最大运行成本，避免提前失败逃成本。成功/质量、1ms轮速和倾角门不放宽。

首级采样时间份额完整Swing/shortened/rotated/lifted=40/35/15/10%，下一组70/10/10/10%，后续85/5/5/5%，保留原按完整窗口三窗确认晋级、名义→DR课程。每回合增加实际phase内连续progress、末态腿长/摆角/phase年龄；窗口导出各最终phase计数和连续进度和，不能再把“未跨阶段”误读成“完全没有收腿”。停滞评分使用连续进度，reward上涨仍不授予候选。

训练探索：腿std保持0.02rad，轮std从2降至0.5rad/s，写入显式sampling contract且由factory核对job。physical Gaussian raw用于log-prob，deterministic mean、三次decoder/固定中心、±100轮目标、±1.5腿范围、Actor31D/真实dq/previous、PD/40与4.8Nm、V4与1/5/10ms及所有年龄延迟不变。部署仍只用mean。旧合同/旧配置默认std2保持。降低探索用于减少无意义轮运动，并不保证实体轮速自动降低。

策略初始化显式回到2100骨干/腿输出+零轮head的原始Actor均值，均值MLP逐tensor相等，仅physical_std缓冲改为新合同；不完整恢复u1400的趴地策略，重建942D Critic和双Adam/课程/RNG，首5 Critic-only计入8594。source Actor、teacher、合同、初始化说明、历次账本、授权与分配均SHA绑定。奖励改变后不混旧候选/窗口；同run完整恢复继续保持双Adam与已完成warmup。

每100更新窗口检查完整Swing成功、各阶段推进、顺序违规、低速/姿态；至少看到收短→转腿真实突破才称动作链改善。前12真实优化工程门只证明链路可运行；动态参考可行性、独立MuJoCo/延迟DR验证、Motion交接、实机与导出仍NOT_RUN。本配方没有额外teacher控制、轨迹跟踪器、虚拟锁轮或输出裁剪；若再停滞，保留失败证据而不无限重复同配方。

## 允许收腿与转腿重叠（2026-11-15）

用户澄清“可以收腿的同时转腿”。旧`sequential_v1`的人为违规并不是硬件fault：收腿时摆角超出随腿长插值参考0.30rad，或转腿阶段腿长超过约0.190m，会永久标记顺序失败；progress配方随即终止。其固定-0.65rad中间构型与收腿速度/100ms停留限制可能拒绝其他合理的协同路径。旧27134个shortened回合的“顺序违规”只说明触发该规则，不证明机器人无法起身，也不能直接套到新语义。

新正式配置`configs/jobs/recovery_leg_overlap_remaining.json`显式设置`reward.fields.progression.coordination=overlap_short_turn_v1`：phase0/1不再使用摆角走廊、过早伸腿硬失败或对应顺序成本；改为软左右腿长度对称成本，不投影/裁剪动作。收腿几何里程碑不要求经过-0.65rad、不要求先停下或等100ms才能转腿；short+0.01m达到一个policy tick即可记录里程碑。转腿可从phase0开始，收腿进度与向承重方向的转腿进度各占phase0连续进度一半；phase1仍综合腿长与摆角，单独伸长或抬高不能抹去未完成误差。进入抬升前仍需两腿≤short+0.015m、支撑摆角误差≤0.10rad、摆角速度≤0.5rad/s连续100ms。允许暂时伸回后再收短。成功仍要求真实承重入口和接稳0.5s，不把辅助reset当完整重构。

实体轮速、倾角、角速度、冲击、闭链及环境原有终止、PD/40与4.8Nm、输出范围/decoder/previous/时序均不改；不新增控制器。新episode记录`coordination`、`sequence_valid=null`（人为顺序证书不适用）、`reconfiguration_completed`与辅助reset身份；新窗口`sequence_applicable=false`，不再用原98%顺序门筛候选，但成功率/质量/覆盖门继续有效。旧配方字段缺省时保留原实现。CPU能走通不代表动态安全或能力通过。

独立worktree `/home/kellen/wheel_leg_recovery_overlap_ws`从当前main `28d240d`建立并合并冻结progress源码；未热改旧worktree。由公共stop有界停止旧run于u391，完整源SHA `8fb903544dcd85af4a0c71abf9f709fa0473ded6df02ee7a5cd62ed811035182`；总已耗1797，新run仅分配8203。沿正式CLI/owner/worker/RecoveryAdapter完整加载Actor、942D Critic、双Adam、std0.5、RNG和warmup5，不重新初始化Actor或重做warmup。物理现场fresh reset；新旧奖励比较域通过SHA绑定`recovery_recipe_transition`显式迁移。旧候选/窗口/review history保留在原run，不作为新规则成绩；新run从新统计窗口开始。同run恢复仍严格比较原域。完整来源与余额均在提交/加载边界核验，保留失败证据。
