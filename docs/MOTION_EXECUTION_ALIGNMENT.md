# Motion执行口径：Isaac、MuJoCo与实机

口径版本：`motion_execution_alignment_v1`。本文件解释既有Motion执行链、弹簧补偿与V4的关系，供训练、sim2sim、播放器和部署负责人共同查阅。它不新增运行开关、仿真模式或训练授权；实际运行仍以checkpoint绑定的冻结源码、spec和输出合同为准。目标能力另见[Motion能力矩阵](MOTION_CAPABILITY_MATRIX.md)。本约定不自动扩展到Jump或Recovery。

## 当前统一约定

**Isaac训练和MuJoCo验证／播放使用同一来源的完整V4等效响应，不额外叠加旧物理弹簧或旧C弹簧补偿；已有015实机组合继续保留原软件补偿与真实弹簧。**

| 环节 | 执行表示 | 一致性与边界 |
|---|---|---|
| Isaac训练 | Actor→显式动作解码／采用→PD→完整V4响应；执行力矩与NET外载分路进入物理 | 保留完整元组、真实dq、延迟、限矩及相应DR。已有冻结spec声明`legacy_spring_and_C_compensation: false`，不另加旧弹簧或C补偿 |
| MuJoCo验证／播放 | 同一Actor与输出合同→共享C编码／解码／PD→`PlaybackExecution`完整V4→MuJoCo | 核对V4、时序、动作中心／尺度、采样与初态。当前未额外调用实机弹簧补偿；跨引擎接触和约束求解差异仍需记录 |
| 已有015实机部署 | Actor→015解码／采用→PD→电机方向映射→原软件弹簧补偿→最终电机限矩→真实机构 | 保留已有补偿公式、参数、方向和限矩顺序。使用最新有效反馈，不人为添加仿真的传感器等待；板上实际模型与固件身份须核对 |

`legacy_spring_and_C_compensation`是既有冻结spec中的记录，不是本次新增的JobConfig开关。不能将未知字段直接加入配方，或仅凭文档判断某个run的实际行为。

三条链的一致性目标，是同一策略与目标下的观测、运动响应和执行边界对应。当前采用的是等效建模；不要求仿真逐项重演实机的内部补偿，但也不能据此宣称两者已在所有构型、负载和饱和条件下等价。

## 弹簧、补偿与V4分别表示什么

在同一关节坐标下，记`p`为已限幅的PD请求，`c`为带符号的软件补偿，`s`为物理弹簧力矩。忽略执行动态时，实机的关系为：

```text
电机最终请求 = clip(p + c, -40Nm, +40Nm)
机构驱动力矩 = 实际电机力矩 + s
```

未饱和且补偿准确时，`c + s`接近零，控制器可以获得近似无弹簧的响应。弹簧在协助承重的构型可降低电机负载，在其他运动方向也可能阻碍运动；不能宣称所有动作均减负。

V4拟合的是目标到实测q／dq的等效响应。其质量／惯量、阻尼、耦合刚度、NET净外载和执行时序共同描述采集系统，不是独立测得的纯弹簧模型。NET可能包含物理弹簧与软件补偿后的残余、摩擦、工装约束和其他建模误差；不能仅凭NET中有刚度项就要求实机取消补偿。

若原日志确实来自开启原补偿的采集系统，则“仿真用V4且不再额外加补偿、实机保留原补偿”具有一致的建模解释。**这是有采集代码支持的解释，原CSV与烧录版本的完整绑定尚未完成，不能写成已证明的全工况等价。**

## 已确认的采集与拟合口径

本机电控仓`/home/kellen/combat-dm-balance-motion31-supported015`的辨识引用`origin/feature/hip-sine-identification`在本次核对时指向`c8d9636cd0abb8da90c38f1b59b1e2689d06ea44`。按此commit查阅，避免后续分支移动改变证据：

| 证据 | 已确认的实现 |
|---|---|
| `application/chassis/hip_ident.c`，`HipIdentStep` | 实际`motor_request_nm`来自补偿后的`compensated`，对应读取版本第216～222行 |
| 同文件`pre_spring_torque_nm` | 记录PD限幅后、弹簧补偿前的请求，对应第229行 |
| 同文件`feedback_torque_nm` | 电机驱动反馈减去带符号的软件补偿，再转换到PD模型坐标；不取绝对值，对应第230～236行 |
| 同commit的`application/chassis/HIP_IDENTIFICATION.md` | 声明固定机身、测试腿悬空，保留PD、弹簧补偿和限矩，并解释上述字段 |

**从日志力矩列减去补偿，不会消除补偿对真实q／dq运动轨迹的影响。** 该反馈列也不是独立测量的物理弹簧力矩。驱动报告、补偿计算与真实出力存在时序、误差和饱和边界，不能把相减结果当成无误差轴端力矩。

V4实际复用v3的完整闭链拟合：输入来自`mean_actor`还原的物理目标，内部重算PD；主要残差比较预测与实测q／dq，并非直接以扣补偿后的反馈力矩作为最终电机施力。只在日志中出现`pre_spring`或`feedback_torque`字段，不证明该字段进入了拟合目标。

历史证据位置仅供追溯，不是干净检出运行本项目的必需入口：

- V4合同：数据根`dependencies/home/kellen/wheel_leg_native_train/artifacts/2026_10_30__bilateral_detailed_response_v4/candidate_contract.json`，SHA256 `5510088cc1f57f823718fd36218beb32cabf3c1d8ce708cf75531129178dc1cd`。
- 原拟合源码：旧仓`artifacts/2026_10_30__fullchain_response_fit_v3/fit_fullchain.py`的`data_blocks`、`Kernel.run`、`residual`，以及`artifacts/2026_10_30__bilateral_detailed_response_v4/fit_detailed.py`；原报告保留净外载和可辨识范围的说明。
- 原日志：旧仓`logs/Ozone_DataSampling_261029L.csv`、`Ozone_DataSampling_261029R.csv`。本次未将它们与上述commit的烧录身份、左右通道布局和补偿参数逐项绑定，不凭字段名猜测或交换通道。

## sim2sim与完整电控验证的边界

当前Motion的sim2sim应保持Isaac和MuJoCo两端各自绑定的同源完整V4、模型／输出合同、PD、时序、限矩、命令和评分条件。人工地图和训练路段、接触材质、求解器profile不同须单列，不能用不同条件的播放效果解释为同条件能力变化。

MuJoCo当前真实调用链为：

```text
wheel-leg play／已有诊断消费者
  → MotionPlayer
  → ControlCore编码／解码／PD
  → PlaybackExecution → V4 Response
  → d.ctrl执行力矩 + d.qfrc_applied中的NET → mj_step
```

共享C参与不等于整个实机电控循环参与。当前路径没有执行包含弹簧补偿的完整`RLControlExecute`，也不能称Isaac每个1ms子步直接调用C固件。相关消费者见[播放执行](../src/native_stand/simulation/playback_execution.py)、[MuJoCo物理循环](../src/native_stand/simulation/mujoco_playback.py)、[Isaac执行读回](../src/native_stand/simulation/isaac/delayed_execution.py)和[V4响应](../src/native_stand/simulation/v4_response.py)。

若后续目标是验证含补偿和最终限矩的完整电控，则需显式物理弹簧、对应电机响应、相同补偿及限矩顺序，并重新核对或拟合NET与其余等效参数。不能在无显式弹簧的V4路径上只添加补偿，也不能把显式弹簧、完整补偿和原V4净残差直接叠在一起。该方式是后续验证设计，当前没有因此新增或注册运行模式；不得擅自删除V4刚度项、改冻结元组或更换旧run的执行链。

sim2sim通过可以证明对应跨引擎条件下的表现，不能单独证明真实电机负载、补偿后饱和、低高度承重或实机鲁棒性。V4通道名中的`motor`是仿真执行模型的输出；训练的motor effort成本不能直接当成实机补偿后的电流、力矩或节能证书。

## 部署与交接要求

- 已有015实机组合保留原补偿。用户报告其稳定、鲁棒，作为该实际组合的反馈保留；未记录的测试条件不补造成能力通过率，不能直接移作新050的实机证据。
- 015电控使用腿±0.15rad、轮±80rad/s及原0.30～0.40m中心公式。原Student050使用腿±0.50rad、轮±80rad/s及自身0.20～0.40m承重中心。新模型须显式适配解码、中心和板端网络容量，不能只换权重；承重中心是位置目标基准，不是弹簧补偿力矩。
- 31D输入的四腿位置和previous腿目标仍按1.5rad归一化，previous轮目标按100rad/s归一化。保留真实dq、实际采用previous、PD40/2及轮Kv0.2；不因输出合同或补偿表示改变输入尺度、坐标和执行缓存语义。
- 实机保留当前控制链的限矩顺序：PD限幅、映射电机方向、补偿、最终限矩。当前读到的015工作树含未提交修改，其HEAD不能独自代表全部源码；实际烧录文件／模型和补偿参数仍需核对。新050在补偿后可能遇到不同饱和工况，不自动继承015结果。
- 后续追溯优先绑定采集固件commit／构建身份、CSV SHA、左右通道、补偿配置、PD请求、补偿量、最终发送值与有效反馈时刻。只读核对不足时如实记UNKNOWN，不据此自动开新采集、评估矩阵、训练或刷板。

当前训练的工程准备与上述物理等价证据分别记录。本次文档统一没有改配方、奖励、V4、补偿、预算、冻结源码或部署包，也没有授予原Student050实机通过。
