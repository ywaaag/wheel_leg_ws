# Recovery 2100 小模型蒸馏

正式入口：`wheel-leg distill --config configs/jobs/recovery2100_distillation.json`。复用 `SupervisionConfig → owner/worker → supervision_flow → learning.distillation.loop`；没有新增 launcher、训练生命周期或 PPO。开发基点 main `299b8dc`，分支 `recovery/2100-distillation`，worktree `/home/kellen/wheel_leg_distill_ws`。

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

比较使用相同重置seed和明确case的新回合；每批均报告教师与学生。连续两次闭环成功率/起身率没有改善则持久化`review_required`并停止重复BC，检查覆盖、教师适用性、物理误差和时序。成功指标不得由BC损失推断。

用户授权任务总预算不限，配置记录null；本执行块仍限定60round、明确采集/比较步数、7200秒与8GiB输出边界，每round保存。可用统一`wheel-leg stop --run RUN --request-only`请求轮边界停止，使用`wheel-leg resume --run RUN`恢复；有review_required时必须先完成诊断，不无条件续堆BC。没有额外watcher或外部通知。

## 比较与部署

同一128环境比较中，每个延迟cohort含24个完整Swing、4个高度变化站立、4个受扰站立。四组分别为名义20/5/7,11ms、仅观测年龄0ms、仅目标运输15ms、仅轮PD后运输11,15ms。不新增第二个延迟阶段。首回合单独评分；失败后的reset不能替代失败首回合。未自然结束的回合标记UNKNOWN。起身率只用Swing分母，接稳失触、连续10秒保持和HF另报；HF不作为纯蒸馏必然改善承诺。

compare产物包括`closed_loop.json`、真实采集块以及学生真实状态上的腿rad/轮rad/s教师保真误差；compare轨迹不进入训练聚合。三次开发比较各2300步；随后先冻结开发候选，再用独立seed（+100000）做4600步最终检查，最终结果不回用于选模。初期2300步检查；前者若有未结束回合如实保留UNKNOWN，不外推成功。

export复用正式部署模块；包中有Actor、带三次decoder的ONNX、输入/输出合同、控制参数与时序、V4/场景/C核心依赖、真实留出状态golden、ONNX ReferenceEvaluator和C三次decoder数值核对。独立的ARM/ORT/完整PD时序实机一致性仍需相应平台证据；默认hardware_output=OFF，不向电控写入。软件数值通过不等于实机接线验收，也不等于小模型闭环能力通过。若连续无改善触发停止，不跳过诊断自动发布部署包。
