# Jump Isaac 长训准备

用户已独立授权 Jump 最多10000次 PPO、任务总墙钟不限时。该额度不引用 Motion/Recovery 账本；准备阶段没有创建正式训练 run、没有预约或消耗 PPO。CPU 领域交付为 `148bcbc`，公共三提交由集成负责人统一承接。

## 训练来源与目标

直接继承已有 Motion→Jump 蒸馏 Actor，不再从随机站立开始。初始化 Actor SHA `2a4d94517943559eaed51841e27be51e4672162a862ff07cd06979efc2e685fb`，37→2048→1024→512→6、ELU、线性末层；raw Normal 初始std=.12。腿目标=raw×.5rad，轮速目标=raw×10rad/s，原raw/log-prob进入PPO。40/4.8Nm、完整V4和1/5/10ms不变。统一Jump Actor负责BALANCE、下蹲、蹬伸、腾空、落地和重新稳定，不在运行时切回Motion教师。

已有监督权重不保证 PhysX 上稳定。初始化 Critic 和双 Adam 必须是 fresh；BC optimizer 不冒充 PPO optimizer。CPU MuJoCo full state 不能当作 Isaac full resume。

保留现有 reward/version 和自动课程，不为了开训临时换奖励或调低门槛。第一档按实际Jump控制时长混合低落体与5cm主动小跳；课程依次为5/5/10/15/20cm。20cm是研究目标，不是硬件已证明能力。课程32回合一窗、连续2个active窗成功率≥.8且fault<.05、landing窗≥.8才晋级；连续2个active窗<.4回退。失败全部进入分母，无证据为UNKNOWN。

落体只学习缓冲/稳定，不领取主动起跳奖励。balance、landing_only、active_jump候选分池，继续消费正常PPO结束回合；不建立独立能力评估。先完成原地闭环；移动起跳、25cm落差地形、越障及扩展DR不是本配方默认项，不能偷换现有课程。

## 执行分块与额度

| 阶段 | PPO额度 | 工作和通过条件 |
|---|---:|---|
| CPU接入/依赖/模型/恢复验收 | 0 | 已有工程测试、真实C/V4、source/data映射与静态加载；不以CPU结果冒充物理验收。 |
| Isaac物理接线 | 0 | 在不干扰现有任务的资源窗口，以小批验证actual motor/NET、弹簧行程、闭链、接触、1/5/10ms、partial reset和短步terminal；不要求未训练模型稳定跳跃。 |
| 有界PPO smoke | 20（计入10000） | 公共driver，非零/分布合理奖励、有效raw/Actor样本、Actor/Critic双Adam实际更新；检查结束/reset、保存加载、课程和pending/cursor。早失败真实记录。 |
| 正式连续训练 | 9980 | 同一独立预算继续，不追加10000。若smoke提前失败，已完成更新不退款；恢复沿用累计账本，剩余=10000−实际已耗。 |

总任务 `chain_wall_s=null`。执行块由公共 `--max-updates` 或有界smoke阶段控制，保存后正常停机；不新建watcher。建议每100更新保存、至少保留最近2份+三池引用保护；候选保留可能增加实际空间，开训前按真实大checkpoint大小留足终态保存空间。未知故障不通过自动重启反复占预算。

Adam LR沿用Actor 1e−5、Critic 9e−5，4 epochs/4 minibatches、gamma=.9980019986673331、entropy=0、critic_warmup=0；当前网络、探索不为速度改变。候选仍是训练统计，不宣称独立认证。

Isaac并行环境数尚不能冻结。可从256×64的资源候选开始设计，先做真实显存和同步测量再定版；不是已验证配方。该候选每次16384个策略样本，10000次共163840000个样本；仅6370列float32 Critic rollout下界为398.125MiB，含Actor37则400.4375MiB，远少于完整训练峰值，不能据此承诺与Motion可并行。最终N/T一旦冻结，不能在同一比较中缩采样量来制造提速。

## 已核验服务器与隔离边界

服务器为用户给出的算力自由实例，GPU UUID `GPU-07694748-3d8b-666b-3933-6443416715dd`，RTX4090总24564MiB。本次只读核验时GPU已用12468MiB，现有Motion进程约12338MiB。PID1461为既有Motion worker，活动路径在`wheel_leg_motion`。没有停止、重启、热改或接管它。

IsaacLab `/workspace/isaaclab/VERSION` 实读3.0.0；实际Motion沿用的SDK在`/root/gpufree-data/wheel_leg_motion/sdk`。Python应显式用`/opt/isaaclab-venv/bin/python`，非交互SSH没有`python`命令，且未安装`rg`；这两项不代表Isaac缺失。只读包检查为Torch2.11.0+cu130、NumPy2.5.1、SciPy1.17.0、MuJoCo3.12.0、TensorDict.14.0，rsl_rl/warp和cc存在；未启动AppLauncher来“探测版本”。

本任务独立准备目录为`/root/gpufree-data/wheel_leg_jump`，只写自己的candidate/dependencies/preflight。源码包来自实际领域提交，仅包含source及必要小配置/测试，不复制历史runs。33项依赖按SHA单独映射；共有资产和V4仅从已确认来源读取/复制到本任务目录，不覆盖活动Motion目录。历史teacher/trace是当前CPU配置的来源校验依赖，不能因为它们存在就宣称是GPU运行必须的活跃消费者。

33项实际依赖262802507B已逐SHA通过；源码压缩包1479185B，缺失依赖实际补传166214287B，其余从服务器已核验文件复制。远端统一CLI dry-run、真实大Actor/6370D Critic构造、fresh双Adam、静态保存恢复通过；禁止读取Motion目录的审计也通过。云端12项CPU工程回归通过（17.07s），本地兼容性联合31项通过（18.67s）。这些检查没有物理积分、CUDA初始化或真实训练。

实际发现并修复MuJoCo版本差异：本地3.8使用`mj_fullM(model,out,data.qM)`，远端3.12使用`mj_fullM(model,data,out)`；`simulation/jump_support_model.py::full_mass`按实际数据接口显式兼容，不替换环境依赖。直接SSH曾因Python库路径不匹配出现`_ctypes ... undefined symbol _PyErr_SetLocaleString`；加载平台`setup_python_env.sh`并保留`/isaac-sim/kit/python/lib`后通过。修复仅写Jump目录，原失败不覆盖成通过。

连接口令不写入任何配置、收据、命令脚本或Git。机器专用路径用现有RuntimePaths；环境库路径需保留平台Isaac Python目录，不能用错误LD_LIBRARY_PATH加载系统库。

## 当前阻断与后续实现

1. **Isaac物理消费者未实现。** 当前`jump_backend.validate_job`只接受MuJoCo/CPU，registry对应合同拒绝Isaac；不能换device后提交。须实现后端适配器，继续复用公共owner/worker/driver和任务算法，不移植第二套Isaac训练框架。
2. **CPU Critic不能原样映射。** 6370/235中有MuJoCo qacc_warmstart、eq_solref/eq_solimp及不同语义摩擦字段。每字段分析见`JUMP_GPU_BACKEND_BOUNDARY.md`及本地机器可读清单。应显式冻结Isaac Critic版本、真实源张量和fresh Critic恢复规则；不得零填、复制CPU常量伪装成实际PhysX现场，也不得偷偷缩列以提速。
3. **C支撑/反馈桥尚缺。** 当前`jump_support_model.estimate`使用MuJoCo闭链Jacobian、质量矩阵、qacc/bias/passive/NET逆动力学。Isaac必须用可核验的对应数据实现，或另行版本化明确的传感器模型；接触真值仅用于奖励/Critic，不能直接替换Actor支撑反馈。现有host C桥还需要批量化和明确D→H/H→D同步成本。
4. **资源与物理验收未完成。** 现有Motion占用不是Jump显存证据。只有在实际物理后端、上述字段及控制来源合格后，才能有界测量并行峰值，再启动20+9980。无需再次申请已授10000额度，但不以未验证的入口消费额度。

源码/依赖与CPU就绪、真实物理就绪、长训已开始分别记账。授权已取得；技术阻断仍需实际实现，不能通过写“READY”或注册空后端解决。CPU提交可立即独立集成，不等待GPU任务。
