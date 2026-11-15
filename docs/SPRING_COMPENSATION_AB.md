> **有效性修正：** 此A/B没有建立本车被动弹簧及完整执行链等价，不能用于否定实机015补偿；详见[有效性复核](SPRING_AB_VALIDITY_REVIEW.md)。原结果仅作当前V4附加力矩敏感性证据。

# 015软件弹簧补偿A/B（2026-11-15）

结论：在当前MuJoCo原接触＋完整V4仿真中，直接附加015普通Motion的同一软件补偿，没有改善旧tjb_recover；Motion从站立启动则站高明显偏低。此次没有开启实车默认补偿。

| 任务/条件 | 无额外补偿 | 加015补偿 |
|---|---|---|
| 旧tjb_recover，7组Swing，单独Recovery | 0/7；0.49–1.21s失稳 | 0/7；0.42–0.48s失稳 |
| 旧tjb_recover→Motion，7组Swing | 0/7；均未交接 | 0/7；均未交接 |
| Motion站立，名义组 | 10秒保持通过；末5秒高度中位0.3427m | 未通过原站高判据；高度0.3001m |
| Motion站立，DR.5组 | 10秒保持通过；末5秒高度中位0.3448m | 未通过原站高判据；高度0.3021m |

Motion四个回合都运行13秒且未触发故障，末5秒双轮支撑比例都是100%。有补偿的Motion不是摔倒，而是相对0.34m命令低约4cm，超出±0.02m的原保持标准；最长连续合格窗口只有0.003s。不能把“未通过保持”写成“立刻倒地”。补偿引入的腿电机指令项：Recovery最高约16.50Nm，Motion约14.48/14.74Nm；相加后仍腿40Nm限矩，未提高轮4.8Nm。

## 实际实现和调用链

正式入口`wheel-leg sim2sim`→recovery_motion_handoff→RecoveryMuJoCoCollection；可选motor_transform调用simulation.firmware_spring。直接编译SHA绑定的电控RLControlCompensateSpring、RLMotion31MapPolicyTargets、getPhi；几何参数0.21/0.25/0.25/0.21/0，网络/电机左右和符号沿现有C映射，不交换网络数组。编码器残余偏差同时作用于补偿几何。

同一补偿公式：幅度clip[0.8(40.242−83.121L),16.5,27]×sin(front−back)，物理motor顺序lf/lb/rf/rb加[-sL,+sL,+sR,-sR]。每5ms更新并保持补偿项，V4原motor响应继续1ms变化，每步相加后限矩；双轮不加补偿。NET、原机械阻尼/摩擦和地面保持相同。未新增被动弹簧，没有改变网络、decoder、PD40/2/.2、延迟和初态随机种子。

## 解释边界

V4包含拟合NET负载，但合同hardware_identified=false，没有明确认证它等于本车真实弹簧或已做015补偿。此次结果是“原V4仿真＋额外软件补偿”的比较；它不能证明带真实机械弹簧的实车也同样变差，更不能据此把补偿反号当成正确做法。要回答实车应补多少，仍需真实弹簧/力矩/几何标定。

旧tjb_recover本身在当前PD/延迟下无补偿也失败，因此此次没有证明某个单因素是其失败根因。Motion站高下移说明附加项改变了闭环受力和平衡点；具体实车反力仍待辨识。

## 证据和验证

最终Recovery28回合：实际35480个0.5ms子步、32.08s。Motion4回合104000个0.5ms子步。两轮均无训练、无硬件。32条trace SHA逐一回验。精确最终报告、协议、逐case结果见数据根recovery_motion_handoff/spring_ab_final_evidence。

初版r1错误地把补偿后的总腿motor也保持5ms，改变了1msV4响应；其证据保留为中间诊断，不作为最终结果。新增回归先复现再修复，只保持附加项；最终r2重跑。CPU相关25项通过，包含实际消费者、异常部分积分计费、补偿保持/轮不变/限矩；Ruff0.8.4和架构检查通过。完整portable为2010 passed/31 skipped，另2项因旧导航指向外仓文档、fixture未设置新增motor_transform字段而失败；两处已修正，相关25项重跑全部通过，未再次完整重跑。完整与修复后针对性输出另存数据根。

电控仓feature/rl-supported015-recovery2100的活动代码/固件未增加补偿，当前仍为旧tjb_recover替换试验版。没有用仿真结果默默改实机控制设计。
