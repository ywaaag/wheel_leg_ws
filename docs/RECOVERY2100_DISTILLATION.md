# Recovery 2100 小模型蒸馏

当前起身复刻入口：`wheel-leg distill --config configs/jobs/recovery2100_mujoco_rise.json`。旧 Isaac 配方 `recovery2100_distillation.json` 和额外延迟配方保留为历史复现入口。复用 `SupervisionConfig → owner/worker → supervision_flow → learning.distillation.loop`；没有新增 launcher、训练生命周期或 PPO。开发基点 main `299b8dc`，分支 `recovery/2100-distillation`，worktree `/home/kellen/wheel_leg_distill_ws`。

## 教师与接口

唯一教师 `/home/kellen/wheel_leg_training_data/2026_11_13__recovery_settle48_full1800_10000_attempt4/checkpoints/model_update_2100.pt`，SHA256 `5aec83df5644ae1b96f4ed012be41d41cb15c4729bbc7b632af347912d3d12a9`。不查询2300或旧Recovery。环境绑定的C0资源是已有初始化依赖，不是监督教师；全程Recovery own-control课程，未增加救援路由。

学生 `31→256→128→64→6`，ELU隐藏层、Tanh mean。49734个网络参数，FP32权重198936字节（194.27KiB）；实际checkpoint/ONNX/完整部署包包含额外元数据，另行实测。

教师和学生共用 `PhysicalActor`，输出已经是物理目标，不再重复解码。四腿通道 `[0,1,3,4]`；`d=tanh(network)`，`R=upper-center`（d≥0）或`center-lower`（d<0），`q=center+0.15*d+(R-0.15)*d³`；轮通道 `[2,5]` 为`100*d`。中心固定为教师原0.34m参考：`[-0.13097405433654785,-0.15117678046226501,0.13097405433654785,-0.15117678046226501]` rad。改变高度命令不改变中心。不是Motion线性残差或增量步长。

保留真实dq、31D列序、输入腿1.5rad/previous轮100rad/s、实际采用目标previous和原观测年龄。影子教师查询只读，不写执行缓存。物理/PD/策略1/5/10ms，PD40/2/.2，电机腿40/轮4.8Nm，完整V4、NET分路及源资产不变。名义观测年龄20ms、目标运输5ms、轮PD后运输7/11ms。监督不更新Critic、std或教师权重。

教师训练采用v9奖励；采集复用main的v8生命周期，物理/时序/起身与连续保持判据相同。这里不优化奖励，不恢复教师训练状态，不将v8采集称为v9 PPO接续。原训练run、账本和冻结源码只读。

## 数据、学习和停止

初始空view明确没有样本。首个collect由2100实际闭环生成状态；后续collect只由当前学生闭环生成，标签继续只查询2100。每次4600个策略步、128环境。保留完整Swing和后续真实课程；阶段0初始Swing、1低位收腿/转腿域、2抬升域、3首次接稳/失触、4保持、5扰动窗、6实际高度变化。阶段是采样分层，不能代替精细动作完成认证。原真实传感与控制执行不因分层而更改。

先给每回合相同采样质量，再按该回合出现的阶段权重`[1,1,2,4,1,2,2]`分配；长保持不按帧数压倒接稳。每个监督round固定50次更新，batch1024，共三批、每批20round，即每批1000次更新。Adam/RNG/完整回合划分跨批保存恢复。物理目标误差除以`[.05,.05,10,.05,.05,10]`后用Huber；验证选模同一损失。

训练/开发/最终留出按物理回合身份划分并持久化；相邻帧不随机拆分，最终留出不回灌。标签保存实际教师owner、有效性和质量：教师可查询不等于成功认证，默认`UNKNOWN`，不继承源回合成功身份；非法物理状态和不完整transition不进入拟合。原始物理输出、请求、实际采用目标和电机力矩分别采集。

教师与学生比较使用相同重置seed和明确case的新回合；开发seed与采集seed相差50000，最终seed相差100000，比较后重置课程与DR流，避免验证状态回流到采集；每批均报告教师与学生。连续两次闭环成功率/起身率没有改善则持久化`review_required`并停止重复BC，检查覆盖、教师适用性、物理误差和时序。成功指标不得由BC损失推断。

用户授权任务总预算不限，配置记录null；本执行块仍限定60round、明确采集/比较步数、7200秒与8GiB输出边界，每round保存。可用统一`wheel-leg stop --run RUN --request-only`请求轮边界停止，使用`wheel-leg resume --run RUN`恢复；有review_required时必须先完成诊断，不无条件续堆BC。没有额外watcher或外部通知。

## 比较与部署

同一128环境比较中，每个延迟cohort含24个完整Swing、4个高度变化站立、4个受扰站立。四组分别为名义20/5/7,11ms、仅观测年龄0ms、仅目标运输15ms、仅轮PD后运输11,15ms。不新增第二个延迟阶段。首回合单独评分；失败后的reset不能替代失败首回合。未自然结束的回合标记UNKNOWN。起身率只用Swing分母，接稳失触、连续10秒保持和HF另报；HF不作为纯蒸馏必然改善承诺。

compare产物包括`closed_loop.json`、真实采集块以及学生真实状态上的腿rad/轮rad/s教师保真误差；compare轨迹不进入训练聚合。三次开发比较各2300步；随后先冻结开发候选，再用独立seed（+100000）做4600步最终检查，最终结果不回用于选模。初期2300步检查；前者若有未结束回合如实保留UNKNOWN，不外推成功。

export复用正式部署模块；包中有Actor、带三次decoder的ONNX、输入/输出合同、控制参数与时序、V4/场景/C核心依赖、真实留出状态golden、ONNX ReferenceEvaluator和C三次decoder数值核对。独立的ARM/ORT/完整PD时序实机一致性仍需相应平台证据；默认hardware_output=OFF，不向电控写入。软件数值通过不等于实机接线验收，也不等于小模型闭环能力通过。若连续无改善触发停止，不跳过诊断自动发布部署包。

## 首轮实测与延迟补教接续

r3在20round/1000次监督更新后，开发比较中教师起身/10秒保持均100%；学生为52.08%/61.72%。名义延迟保持96.875%，观测年龄0ms为100%，目标运输15ms和轮运输11/15ms各25%。学生状态上的教师保真误差变为腿0.0616375rad、轮8.131875rad/s，不能以教师数据离线误差0.0074195rad/0.434851rad/s宣称可部署。

实际首轮采集仅覆盖目标4～9ms、轮6～8/10～12ms。根因在初始配方沿用`cubic_randomization.draw(strength=.25)`，而原`reset_collection(fresh=False)`没有使用长延迟case。已通过统一stop入口请求本会话worker有界停止；原r3的模型、评估、spec、账本和源码保持只读。

接续配方改为`configs/jobs/recovery2100_delay_dagger.json`，在当前main `fc29da3`建立的独立worktree `/home/kellen/wheel_leg_distill_delay_ws` 实现。复用现有case/reset/factory，只增加collection case配置；四组显式时序覆盖失效条件，每组保留名义与DR=.25锚。训练数据来自新学生回合，开发/最终比较帧不回灌。`PhysicalActor.decode_mean()`只抽出原三次公式，让collector在一次网络前向后分别保存network mean和物理raw action；实际采用目标、请求及力矩仍来自原执行器，额外保存实际Actor观测年龄。

用现有`wheel-leg import-checkpoint`和`resume`完成版本化完整BC交接，保留逻辑run身份、20round/1000更新、Adam/RNG、固定episode划分、已有物理消耗及phase cursor。新的累计墙钟边界为14400s，原消耗不退款；任务总预算仍不限。仅checkpoint的handoff元数据改变时，采集恢复须明确记录旧/新SHA并核对Actor所有tensor完全相同，不能将换权重伪装成metadata搬迁。新版本不向电控写入。

## MuJoCo 起身复刻（当前）

用户当前目标是复刻2100自主起身，站立后慢速漂移可接受。选模优先起身率，再比较连续10秒保持；接稳失触、原始1kHz保持、HF、已有延迟范围和物理目标保真独立报告。不得将教师未通过的较硬接触、原地静止或额外长延迟作为本轮必须改善的能力。

当前分支从main `12c8be4`建立，worktree `/home/kellen/wheel_leg_distill_mujoco_ws`。`simulation.dispatch.execution_service`按显式`assets.recovery_mujoco`选择`simulation.recovery_mujoco.make_collection_env`。它加载SHA绑定的原XML名义模型二进制快照及readback；快照由原教师探针只构造不积分，保留原接触、闭链与V4缩放，不使用USD重建模型替代原接触。MuJoCo版本必须匹配。1ms执行/5ms PD/10ms策略不变，积分拆为两个0.5ms子步。冻结执行器与生命周期以显式factory加载；不改模块全局对象。每步保留原始1kHz真实状态、q/dq、采用目标、motor/NET和HP10，不以平均支撑力替代连续支撑。故障保留真实已完成子步，不退款。

每个训练块8个环境、4600个策略时刻，保留2个完整23秒回合及自然失败/reset。首块Teacher控制，次块Student控制；均仅由2100只读查询同一31D观测给物理标签，质量默认UNKNOWN，不继承成功身份。两块各20轮、每轮50次优化。开发与最终检查各8个固定条件，使用不同DR种子，保留名义航向和站立锚；相同名义条件的重复不视作独立泛化。命令高度固定0.34m，不宣称多高度能力。原15组只用于回归，新采集数据按完整物理回合固定划分；开发/最终闭环轨迹不进入训练。

本阶段明确用既有学生Actor初始化新数据域实验，Adam重新建立，不称完整BC状态恢复。旧逻辑任务已耗1000次监督更新、13,479,680个Isaac物理步另记不退款；另有3个学生MuJoCo基线共138,000子步、正式adapter smoke46,000子步。误建旧回归数据的offline r1在标签条件检查失败，监督优化0；其数据隔离，不用于本配方。累计任务不限额，当前块40轮、4小时、8GiB边界，沿统一停止/恢复/预算机制结算。

部署依赖同时绑定教师SHA、三次合同、完整V4张量、名义时序、实际previous、PD与限矩、真实留出状态golden及C源码。ONNX内含decoder，仅执行一次。导出额外提供`recovery_actor.c/.h`，FP32 ELU/Tanh全网络调用共享`RLPolicyDecode`一次；真实golden同时核对C mean与物理输出。测本机C调用中位数/P95与权重/工作数组字节。主机全网络核对不代表MCU时序实测；硬件写出关闭。

## 教师行为保留与导出修复

用户明确不要求完成教师失败的任务。新配方选模配置为`teacher_rise_retention`、`teacher_hold_retention`；逐同条件比较，教师失败上的额外成功不能抵消教师成功条件中的丢失，未完成回合仍UNKNOWN。物理目标Huber仍是唯一学习损失，不增加任务奖励。后续补教优先采用已发布的闭环候选，full latest只作为Adam/RNG完整继续来源；两者分别留痕，不把未选中的末轮权重冒充已验证候选。

840d3d1冻结run已完成40轮/2000优化及独立最终闭环，最后在ONNX导出处FAILED。原始失败产物和账本保留。当前导出器将Recovery高级索引原地回填导成空输入Add节点，ONNX checker确认非法；改用六通道显式stack后检查通过，原/新PyTorch逐元素完全一致，网络权重未变。Mean wrapper注册Actor，并在ReferenceEvaluator前执行ONNX完整checker。新增真实golden导出参数，Recovery时序显式保存1/5/10ms及原延迟，硬件输出OFF。

旧run不能热改或伪造COMPLETED。从同一冻结selected.pt经新版本`wheel-leg export --run RUN --checkpoint SELECTED.pt --output NEW_PACKAGE --onnx --golden GOLDEN.json`只做部署导出；golden为同一真实留出状态引用，不重训、不改最终候选、不据最终结果重选。C源码保留原文件名及manifest于control_core_source，生成网络依赖包内同一C decoder，可脱离原工作区重编译。
