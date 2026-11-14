# Jump cooked collider 审查落实

本轮落实用户提供的只读审查，不接管其他 owner 的训练。工作区为 `/home/kellen/wheel_leg_jump_audit_ws`，分支 `jump/collision-readback-audit`。原工作区 `/home/kellen/wheel_leg_jump_gpu_ws` 在核查时已有 v148 探针、随后出现统一 worker 的16次有界训练；本轮未停止、恢复、热改或同步该工作区。

## 结论

已修复光滑圆柱净空门禁漏算外切凸体的缺口，并增加同一次初始化的 cooked 几何和原生碰撞参数读回。construction 收据按实际 C、逆动力学消费者记录执行设备。原1mm阈值、Actor/Critic布局、奖励、课程、raw动作、限矩及1/5/10ms保持；光滑圆柱净空仍供既有任务定义使用，新的 `collision_clearance` 仅供物理门禁与诊断，不冒充旧Critic字段。

本机CPU PhysX两环境10个显式world ticks通过最终源码检查。尚未验证此补丁的GPU物理路径或远端6.2，不放行长训，不把历史v137/v146的PPO0 PASS转授给新源码。当前公开 cooked representation API不暴露buildGPUData或每形状的实际窄相设备，因此收据明确 `UNVERIFIED`，不能使用CUDA设备或计数限制替代证据。

## 身份与修复

用户审查摘要 `/tmp/jump_cylinder_motion_readonly_review.json` 的SHA核实为 `4c0539ba94d609222f23dbb0194850223c953de68a638eff0f6ba38fbb24d1c6`。原scene源码核实为 `9af9df4f7fec9ff9081f2302734dc1f23aef666b185c02be60b0f7fc46dec568`，已是convex30。

审查输入来自原工作区HEAD `1d26cdc50be7182a0baff5914b546778ae32a352` 及其三个未提交文件，独立复制源码diff到本工作区后保存为快照提交 `19d771c`；没有复制模型/历史runs，也没有改动原工作区。这是可追踪的输入快照，不应把同一convex30 diff重复集成。实际修复另为其后小提交，集成方已有输入改动时只承接修复提交。

| 缺口 | 实际表现与代码根因 | 本轮处理 |
|---|---|---|
| 门禁使用几何不同 | `validate_physical_measurement`和device gate只看光滑Cylinder，30边外切体更大 | `guarded_clearance`同时检查旧圆柱和实际cooked hull，不改变1mm阈值；原圆柱-.900mm而凸体-1.230497mm的反例现在拒绝 |
| cooked证据不完整 | v145只存顶点字符串和polygon对象repr，无法检查真实面数据；输入边数不能代替cooked形状 | 启动读取完整vertices/indices、polygon的index_base/num_vertices/plane；检查索引、有限性、真实外包界，并绑定源资产和执行源码SHA |
| 真实碰撞变换未进入新门禁 | 仅由源description/FK重建不足以证明最终碰撞位置 | 读取native property query的局部位置/四元数/AABB，与USD局部变换交叉核对；门禁采用native变换后的cooked顶点 |
| construction设备标注错误 | `jump_isaac.py`构造收据固定写CPU C bridge及CPU inverse | 新版`jump_construction_readback_v2`从已创建的execution／inverse对象分别判定，CPU路径与CUDA firmware／Warp路径分开；碰撞窄相设备独立标注 |
| GPU compatibility证据不足 | 顶点/面限制、CUDA device及GPU利用率都不证明GPU contact generation | 收据保存计数限制检查，并明确buildGPUData与narrowphase未验证；不提高hullVertexLimit到256，不改写历史receipt |

修复源码：

```
src/native_stand/simulation/jump_collision_readback.py
src/native_stand/simulation/jump_isaac_scene.py
src/native_stand/simulation/jump_isaac.py
tests/test_jump_cooked_collision.py
```

新的物理身份为 `jump_PhysX_wheel_convex30_cooked_guard_v6`。不能在旧run中无声明接续该身份。现有物理迁移机制、full学习状态和fresh-reset边界保持原要求；本轮没有创建或恢复训练run。

## 最终同startup实际读回

最终原始收据：`artifacts/jump_collision_audit_20261114/cpu_startup_v5/receipt.json`。IsaacSim6.0.1.0、本机SDK源码3.0.0，package metadata isaaclab6.1.14／isaaclab-physx1.1.3；CPU PhysX，CUDA_VISIBLE_DEVICES为空，不创建新的GPU物理任务。

- 两轮cooked均60顶点、32面、单面最多30点，完整polygon数据已保存。
- cooked顶点径向最大外包 `0.00033049807600718517m`；轴向最大外包约 `-8.94e-11m`；原圆柱超出cooked面平面的最大值 `3.0167e-9m`，在2µm数值容差内。
- 原生碰撞局部位置均为0，局部四元数xyzw约 `[0,0.70710653,0,0.70710701]`。读取的native和USD变换后的cooked顶点在2µm容差内一致。
- 两轮实际contact offset约 `0.00065999996m`、rest offset为0；地面实际contact offset约 `0.002000000095m`、rest offset为0。offset不被加到几何净空或用于放松1mm阈值。
- 实际轮材质有效槽为静/动摩擦0.5/0.5、恢复系数0。API返回的padded shape槽保留原值，并注明其布局，不能把空槽当第二个真实碰撞形状。
- V4应用后的实际两轮质量分别约0.60325652kg、0.77564597kg；COM和惯量完整保存，不将其误写为原USD未缩放质量。
- 全局 `/physics/collisionApproximateCylinders=False`，逐轮prim未显式custom geometry override；当前替换后轮形状为Mesh。此设置读回不能替代原Cylinder版本的实际窄相证明，更不能认定nativeCylinder独有引擎缺陷。
- 最终10tick记录同时含原圆柱与cooked hull的gap；源代码三文件SHA与最终工作树逐项一致。

## 验证与开销边界

最终联合CPU：66 passed in18.40s，覆盖真实C/schema/模型装配、Critic批量、device reward脚本数据、物理迁移/partial reset、原始及cooked门禁、任意姿态/偏心变换的最低顶点对照、cooked索引及面校验、执行设备收据。Ruff0.8.4 check/format通过；架构219个正式源码文件、0基线例外；git diff --check通过。统一CLI的零额度诊断配置train --dry-run通过，writes=false、starts_engine=false、total_updates=0；这不是正式新物理full恢复认证或训练放行。

几何读回只在构造时进行，物理tick仅计算两个轮的局部竖直方向及固定cooked顶点支撑最大值。凸体顶点是每轮共享的不可变原型，不为每行复制；所有clone的USD authored几何逐项核验。若未来修改碰撞体、几何DR或引擎现场restore，必须重建此身份和读回，不得沿用旧缓存。质量/COM reset不会修改该顶点缓存。未测GPU吞吐，不声称稳态训练提速。

本轮五次CPU构造检查保留各自receipt：v1因把private统计接口误作public而失败，v2通过；v3补native碰撞变换后通过；v4因另建simulation view不适配stage而失败；v5复用当前scene已有view，最终通过。v2/v3/v5各10个显式world ticks，总30world／60env ticks；构造warmup未逐步计数，不冒充物理消耗0。未新增GPU物理或PPO更新，没有信号、远端操作或历史收据覆盖。

## 剩余验收

1. 由当前唯一Jump owner安排，在当前目标runtime同startup验证新读回和门禁，再决定是否开展已有有界PPO。不得热改正在执行的源码。
2. buildGPUData／实际窄相设备若API不暴露，需用目标版本可核实的引擎证据或明确按CPU narrowphase记账；当前结果不足以宣称全流程GPU。
3. GPU partial/fresh reset、10ms正常样本及短故障terminal、正常结束统计、完整学习保存恢复仍要绑定新冻结源码；不沿用v137128或v14630旧门禁的PASS。
4. 倾斜轮的slip仍用center-R*ez，影响奖励/Critic的接触速度口径；逆动力学接触点近似另有其实现，不能把两者当成同一根因。本轮保留奖励和支撑估计定义，后续应独立版本化验证，不借几何审查顺带改变训练目标。

官方依据：PhysX5.6.1要求convex GPU contact cooking数据，并列出64顶点／64面／每面32点的限制；custom geometry在CPU处理。见[GPU Rigid Bodies](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/GPURigidBodies.html)。计数通过只是必要证据，不是目标runtime执行路径认证。

## 后续只读复核与收据模板

用户补充v155/v6审查后，本轮单次只读核实：v14已正常STOPPED_BOUNDED、16次/failure=null；v157在独立目录6400tick/PPO0 PASS；v156因原owner冲突被拒，保持0更新；现有owner已启动v15的2次有界正常PPO。本分支未提交、恢复或停止这些任务，不能重复运行同一strict4096探针。累计额度继续由现有唯一owner结算，不能沿用本分支历史余额。

contact分支`dfd0d7f`的v6热点缓存是authored30顶点的已校核代理，减去1e-7m后与Cylinder gap取min，0.1µm是保守收紧，不是给1mm门增加容差。用户复算65.8nm/70.4nm面约束残差受序列化精度影响，不能由这一证据承诺跨SDK的误差界。更换SDK、cooking参数、源几何或碰撞变换后，必须同启动重新校核。此实现与本审查分支直接缓存public representation API读回顶点的实现不同，集成时选择一种有完整身份的方案，不得混称actual hull。

后续收据新增`physics_execution_readback`：actual_device取live physics manager，GPU dynamics/broadphase取实际physics scene，tensor backend单独记录；`backend_label`由该读回生成。构造收据和`artifacts/jump_collision_audit_20261114/next_receipt_template.py`均使用这一字段。原v155 trace/旧construction及已保存startup脚本/receipt保持字节原样。helper仅读取元数据，新增CPU回归故意让请求device=cuda而实际manager=cpu，确保标签不会跟随请求或目录名。buildGPUData和每碰撞体窄相设备仍为UNVERIFIED。本次不新增物理/GPU/PPO运行。

## Motion Plane 的接触测量来源

本节仅核对源码/config，不启动引擎、不查询或操作worker，不修改实现。`F`为冻结来源目录：

```
/home/kellen/wheel_leg_training_data/dependencies/home/kellen/wheel_leg_native_train/isolated/2026_11_05__motion_supported2048_dqgain_ppo/runtime_view_r4
```

| 核对项 | 源码证据与准确口径 |
|---|---|
| 原生Plane与sensor | `F/src/native_stand/env.py:30`的ContactSensor无对端filter；117–129创建原生UsdGeom.Plane并绑定材质，138–139实例化sensor。没有filtered matrix失败后改用net的fallback |
| 正式接触力消费者 | QualityMotionStudentEnv→MotionStudentEnv→TrainingStudentEnv→StudentEnv→RecoveryEnv；`F/src/native_stand/recovery/runtime.py:313–317`取`net_forces_w.torch`三维norm后按wheel_sensor_ids选两轮。`F/src/native_stand/command_v6/runtime/env.py:217–221`虽有同口径，不能替代最外Motion链的定位 |
| 平面隔离 | 新仓`simulation/isaac/fullchain.py:153–159`调用slope_tiles.setup；`F/src/slope_tiles.py:16`禁用旧ground，42–63设置invertCollisionGroupFilter与env↔assigned Plane双向白名单，并保留各组自身关系。`filter_collisions=False`不等于没有隔离 |
| rough替换 | 新仓`simulation/isaac/terrain_scene.py:72–105`删除rough环境与Plane的双向关系，再改用该环境road，避免两套地面同时参与 |
| 自碰撞来源 | 数据映射中`wheeled-legged_RL/configs/rmuc_wheel_leg.yaml:135`显式false；`source/agent_world/agent_world/assets/rmuc_wheel_leg.py:96–100`启用contact reporter并消费该值。net readback本身不会排除自碰撞 |
| SDK数据语义 | 本机对应SDK的`isaaclab_physx/sensors/contact_sensor/contact_sensor.py:380–385`始终读取net，只有非空filter才读取matrix；`contact_sensor_data.py:74–78`定义其为世界坐标法向接触合力。它不含另取的摩擦力，也不是重力、motor、NET和接触的全机受力总和；没有对端身份 |

此处norm是法向接触向量相加后的长度，不能改称世界Z分量或逐接触力大小之和。源码/config证据不等于已重新读回引擎中的self-collision及collision group现场。

### 既有Cylinder＋Plane对照的测量约束

1. 保留Jump的既有力阈值、1ms采样、1mm几何门禁和控制/奖励定义。当前Jump的filtered matrix是显式测量来源，缺失会报错（`jump_isaac_scene.py:494–500`）；不增加“matrix失败自动转norm”的隐式路径。
2. 若下一既有探针选择原生Plane与unfiltered net，先同启动读回wheel body/sensor顺序、实际selfCollision、invertCollisionGroupFilter、env/Plane成员与双向关系、旧ground禁用状态和其他可能对端。只有能确认该轮唯一可接触对象为所分配Plane时，才可将该net解释为轮地法向接触合力；开启自碰撞或存在未排除对端时拒绝这一解释，不零填冒充。
3. 测量合同明确写`unfiltered normal-force net`。平地保留`clamp_min(net[...,2],0)`并另报`norm(net)`；有坡度时使用已知单位Plane normal的点积，单独声明合同版本。世界Z/平面法向投影都不是Motion的原三维norm口径。
4. 接触读回仍仅供既有奖励/验证消费者，不将接触真值送入Actor支撑观测，不借对照改变C/FSM唯一控制归属。Motion的5mm门通过不能授予Jump的1mm门通过；Cylinder与凸包、Plane与薄盒的PASS也不能跨身份转授。
5. 本轮不因这份审查扩展实验矩阵或新增探针；后续由当前唯一owner在已授权的既有有界检查中采用。未知对端或投影条件明确拒绝，保留旧结果和失败证据。
