# Motion接管补偿修复与V4复测

只修复电控feature/rl-supported015-recovery2100集成路径；feature/rl-motion31-supported015为只读参考。当前Recovery仍为旧tjb_recover（腿1.5mean、轮100mean），不是2100。

电控原缺陷：recovery_execute在Motion接管后仍直接发送PD腿力矩，遗漏普通015的RLControlCompensateSpring。修复将Motion模式标记随物理目标运输5ms，实际PD采用后按5ms更新/保持补偿，最终腿限矩40Nm；Recovery不启用。主机真实C消费者覆盖采用前不抢先、采用后公式一致、PD保持/更新、限矩及无效几何锁零。

仿真复用正式wheel-leg sim2sim与RecoveryMuJoCoCollection，保持完整V4+NET，不额外加015补偿，不改冻结依赖。V4采集时软件补偿开关尚无确证，不能认证实机弹簧等效；软件修复由C测试验证，以下仅是原V4策略闭环复测。

|方案|条件数|连续10秒保持|结果|
|---|---:|---:|---|
|旧tjb Recovery-only|7|0|0.49–1.21秒失败|
|旧tjb→Motion，本体20ms门|7|0|全部交接前失败|
|015 Motion直接站立|2|2|保持12.984/12.982秒，无故障|

数据根 /home/kellen/wheel_leg_training_data/recovery_motion_handoff：

- tjb_v4_recheck_motion_spring_fix_r1：22480个0.5ms子步，20.588秒墙钟。
- motion015_v4_recheck_motion_spring_fix_r1：52000个0.5ms子步，34.239秒墙钟。

各目录保留plan、protocol、episodes及原始trace；两组均COMPLETED，无训练更新、无真实电机操作。COMPLETED表示运行结束，不表示旧Recovery能力通过。Motion补偿修复不会改变这些未启用额外补偿的V4数值结果；旧tjb在交接前失败，不能归因于Motion补偿。

正式复测配置：configs/jobs/tjb_recover_motion015_current_control.json与configs/jobs/motion015_v4_standing_recheck.json；先dry-run，后显式新output。原错误插入位置A/B仅保留为附加力矩敏感性试验，不能据其约0.30m站高否定015实机补偿。

## 交付文件

电控提交40e1b8b（已推送feature/rl-supported015-recovery2100）：

- application/chassis/rl_control.c：实际采用后启用补偿，5ms更新保持。
- application/chassis/rl_control.h：运输模式标记与保持缓存。
- application/chassis/rl_recovery_config.h：纠正活跃模型注释。
- tests/test_rl_recovery_handoff.c：真实控制消费者回归。
- docs/RECOVERY2100_HANDOFF.md：更新执行定义与验证边界。
- firmware/recovery2100_motion015/README.md：使用与当前模型能力。
- firmware/recovery2100_motion015/manifest.json：源SHA、二进制SHA与构建占用。
- firmware/recovery2100_motion015/recovery2100_motion015.bin。
- firmware/recovery2100_motion015/recovery2100_motion015.hex。

仿真仓库只新增configs/jobs/motion015_v4_standing_recheck.json、本说明并更新STATUS.md；没有修改策略、V4后端或冻结模型。

构建Flash499772B、DTCM82656B、RAM_D159456B。主机9/9套通过，工具9项中3通过、6项因外部原模型缺失跳过。ARM构建通过，真实MCU耗时与栈高水位未测；未刷机。

CPU完整portable：2012通过、31跳过、14 warnings，149.33秒；显式PYTHONPATH=src、单线程CPU。首轮未设置PYTHONPATH，2011通过/1失败（旧脚本子进程无法导入native_stand），未修改无关脚本，以正确环境完整重跑通过。架构218文件/0例外，Ruff0.8.4 check/format通过。日志在数据根motion_spring_fix_40e1b8b_evidence。
