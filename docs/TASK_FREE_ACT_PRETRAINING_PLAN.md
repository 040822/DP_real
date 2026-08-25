# 无任务 ID 的 ACT 真机数据预训练与微调任务计划

状态：待实现  
目标执行者：DeepSeek  
适用仓库：DP_real  
目标任务：`collecting_objects`  
目标任务数据量：固定使用 `data/data_wx/collecting_objects.h5` 的全部 50 episodes（20,000 steps）

## 目标与固定决策

本计划要建立一条可持续吸收松灵机械臂历史真机数据的训练流水线：先从不断增长的多任务数据池训练一个无任务 ID 的通用初始化 checkpoint，再用特定任务少量人工示教进行全模型微调，最终得到只服务该任务的单任务策略。

固定决策：

- ACT 输入保持 `cam_high + qpos`，不增加 task ID、任务名称、语言或目标图像。
- 通用预训练 checkpoint 不是可直接部署的多任务策略，只是目标任务训练的初始化。
- collecting_objects 最终微调使用现有全部 50 episodes；不得为了普通 train/val split 将最终训练量降为 40、45 或 47 episodes。
- 第一版沿用当前 ACT 的 30-step chunk、L1＋KL 主体，不同时引入新的自监督损失或网络结构。
- 数据文件来源可以用于采样平衡、切分、审计和日志，但不得进入 policy observation。
- 不生成一个物理拼接的大 HDF5；实现多 HDF5 懒加载和统一索引。
- 预训练、微调与推理必须复用同一份版本化 normalizer，不允许微调时静默重新拟合并覆盖。
- 数据切分必须以完整 episode 为单位；禁止对高度重叠的 window 使用 `random_split`。
- 当前 action 契约为绝对关节目标，已有数据满足 `action[t]` 更接近 `qpos[t+1]`；实现必须增加自动校验，不能只依赖本次人工审计。

## 当前数据基线

| 数据范围 | 文件数 | Episodes | Steps |
|---|---:|---:|---:|
| collecting_objects | 1 | 50 | 20,000 |
| 其他任务 | 10 | 261 | 81,910 |
| 全部真机数据 | 11 | 311 | 101,910 |

所有当前文件均具有：

```text
cam_high: uint8, (T, 3, 256, 256)
qpos: float32, (T, 14)
action: float32, (T, 14)
episode_ends: int64, (N,)
```

当前 HDF5 没有可靠的 `sample_hz`、颜色空间、夹爪标定或相机同步元数据。实现联合训练前必须补外部 manifest 契约并执行校验；不得因为 tensor shape 一致就假定物理语义一致。

## 目标流水线

```text
版本化多 HDF5 manifest
        ↓
数据契约校验 + episode 级索引
        ↓
多源懒加载 + 平衡采样 + 固定 normalizer
        ↓
无任务 ID 的 ACT 通用预训练
        ↓
base checkpoint（不直接部署）
        ↓
collecting_objects 全部 50 episodes 全模型微调
        ↓
离线/Shadow/低速真机验收
        ↓
collecting_objects 专用 checkpoint
```

## 实验组（不得删减 A、B、C）

| 组别 | 预训练数据 | 微调数据 | 用途 |
|---|---|---|---|
| A Scratch-50 | 无 | collecting 50 | 当前单任务基线 |
| B Other-Pretrain-50 | 排除 collecting 的 261 episodes | collecting 50 | 测量真正的跨任务迁移 |
| C All-Pretrain-50 | 全部 311 episodes | collecting 50 | 测量追求最终性能的生产方案 |

可选计算量对照：

- D Scratch-50-Compute-Matched：随机初始化，只重复 collecting 50，使 optimizer steps 接近预训练＋微调总步数。它用于排除“提升只来自更多梯度更新”。
- E Related-Pretrain-50：只使用抓取、搬运、放置等语义接近的数据预训练，用于检测无任务 ID 全量混合的负迁移。

所有组必须使用相同模型结构、图像预处理、目标微调步数、推理配置和真机评测初始条件。预训练额外计算量属于方案成本，必须单独记录。

## #1：建立版本化数据 manifest 与物理契约校验

Blocked by: 无  
Type: Prototype

### Question

如何让不断增长的 HDF5 数据池在不物理拼接的情况下可复现、可拒绝错误数据，并支持 other-only/all-data 两种预训练集合？

### Answer

新增版本化 manifest，例如：

```text
config/dataset/real_robot_pretrain_v1.yaml
```

每个条目至少记录：

```yaml
path: data/data_wx/grabbing_rod.h5
enabled: true
include_in_other_pretrain: true
include_in_all_pretrain: true
sample_hz: <必须由采集配置确认，禁止猜测>
camera_encoding: <rgb8 或 bgr8>
joint_order: <14 维明确顺序>
joint_unit: rad
action_semantics: next_absolute_joint_target
gripper_convention: <明确开闭方向、范围与单位>
```

新增只读校验命令，建议文件：

```text
scripts/validate_real_pretrain_data.py
```

校验项：文件存在、键集合、shape、dtype、episode 边界单调且末值等于 T、非空 episode、NaN/Inf、像素范围、逐关节范围、夹爪范围、相邻 qpos/action 跳变、`action[t]`/`qpos[t+1]` 对齐误差、manifest 与文件一致性。频率或颜色空间缺失时默认失败，不允许仅告警后继续正式预训练。

验收：

- 命令对当前 11 个文件输出汇总表和确定性退出码。
- 人为修改 action 维度、episode_ends 或 manifest sample_hz 的 fixture 会失败。
- manifest 生成稳定 hash，并写入训练 checkpoint/config。
- 不修改原始 HDF5。

## #2：实现 worker-safe 的多 HDF5 懒加载数据集

Blocked by: #1  
Type: Prototype

### Question

如何让训练读取不断增长的数据池，同时避免 `use_mem:true` 将全部图像加载进 RAM？

### Answer

新增多源数据集，建议文件：

```text
source/dataset/multi_h5_dataset.py
```

职责：

- 从 manifest 建立 `source → episode → window` 的全局索引。
- HDF5 handle 在 DataLoader worker 内按需创建，不跨进程共享；提供显式关闭与序列化安全行为。
- 每次只读取 `n_obs_steps` 所需图像和 horizon 所需 qpos/action。
- window 永远不能跨文件或 episode。
- 返回格式与当前 `Dataset2D` 完全兼容：`obs.cam_high`、`obs.qpos`、`action`、`is_pad`。
- 可以额外返回 `source_index` 用于采样审计和日志，但训练 batch 送入 policy 前必须移除；policy observation 中不得出现 task/source ID。
- 支持 `other-only`、`all-data` 和显式文件子集。

验收测试：

```text
tests/test_multi_h5_dataset.py
```

- 两个小型 fixture HDF5 的样本数、padding 和边界正确。
- 不发生跨 episode/window 污染。
- `num_workers=0` 与 `num_workers=2` 返回相同样本。
- policy 实际收到的 obs keys 仍严格等于 `cam_high`、`qpos`。
- 初始化 RSS 不随全部图像解压后大小线性增长。

## #3：改为 episode 级固定切分

Blocked by: #1, #2  
Type: Prototype

### Question

如何消除当前 window-level `random_split` 的验证泄漏，同时保证 collecting_objects 最终模型确实使用全部 50 episodes？

### Answer

预训练：

- 为每个源文件按 episode ID 和固定 seed 生成 train/val mask。
- mask 写入或可由 manifest hash＋seed 确定性重建。
- train/val 分别生成窗口；禁止先生成全部窗口再随机切分。

collecting_objects 最终微调：

- 最终 A/B/C 模型全部使用 50 episodes 训练。
- 不使用泄漏的 window-level val loss 选择 A/B/C 的最佳 checkpoint。
- 微调步数和 checkpoint step 在实验开始前固定；使用统一的 step checkpoint 做最终比较。
- 如果需要调超参数，先进行 episode-level 交叉验证或使用独立开发配置；选定配置后重新用全部 50 episodes 训练最终模型。

验收：

- 测试断言 train/val episode ID 不相交。
- 最终 collecting 配置日志明确打印 `train_episodes=50`。
- checkpoint 保存真实 optimizer step，而不只保存 epoch。

## #4：实现版本化、跨阶段冻结的 normalizer

Blocked by: #1, #3  
Type: Prototype

### Question

如何避免预训练权重加载后被 collecting 数据统计量静默覆盖，从而改变网络输入输出物理尺度？

### Answer

第一版保持当前 normalization 算法，先隔离“预训练是否有效”这一变量，但将统计量变成独立、版本化 artifact：

```text
artifacts/normalizers/real_robot_v1.pt
```

规则：

- 只从指定预训练 train episodes 拟合；val 不参与。
- artifact 记录 manifest hash、episode mask、字段、mode、min/max/mean/std 和创建脚本版本。
- 预训练 checkpoint 保存其 normalizer ID/hash。
- 微调默认加载并冻结 base checkpoint 对应 normalizer；不得调用目标 dataset 自动重拟合。
- 推理严格加载 checkpoint normalizer，并打印 hash。
- A/B/C 公平比较时必须明确 normalizer 控制方案。推荐生成一份固定的 `real_robot_v1` 并让三组共同使用；报告这份统计量使用了哪些数据。

修改 `two_train_ckpt` 路径，使模型权重加载结果被检查，并禁止后续 `model.set_normalizer(target_stats)` 静默覆盖。

验收：

- 同一个原始 batch 在预训练、微调和推理路径归一化后逐元素一致。
- 篡改 normalizer hash 时训练/推理拒绝启动。
- checkpoint round-trip 后 action unnormalize 结果一致。

长期后续（不属于第一版阻塞项）：用机器人关节软限位和固定夹爪标定构建物理 normalizer，避免数据池增长导致统计尺度漂移。

## #5：实现不输入模型的多源平衡采样

Blocked by: #2, #3  
Type: Prototype

### Question

如何利用所有数据，同时避免 20,000-step 文件淹没 1,500-step 文件，又不向 ACT 输入 task ID？

### Answer

实现 source-aware sampler，但 source 信息只存在于 DataLoader：

```text
p(source_i) ∝ num_train_windows_i ** alpha
alpha 默认 0.5
```

随后在 source 内采 episode/window。配置必须支持：

- `alpha=1.0`：按所有 window 均匀采样。
- `alpha=0.0`：按数据源均匀采样。
- `alpha=0.5`：默认平方根温度采样。

日志每固定 step 输出各源实际 batch 占比，并在 epoch/训练结束汇总期望与实际差异。

验收：

- 固定 seed 下抽样序列可复现。
- 统计测试中实际比例在容差内接近期望值。
- batch 送入 `ACT.compute_loss` 前不含 source/task ID。

## #6：建立统一的预训练配置和启动入口

Blocked by: #2, #3, #4, #5  
Type: Prototype

### Question

如何用同一套 ACT 代码启动 Other-Pretrain 和 All-Pretrain，并生成可审计的 base checkpoint？

### Answer

新增 Hydra 配置或明确的 override 组合，至少支持：

```text
pretrain_pool=other_only
pretrain_pool=all_data
dataset_manifest=config/dataset/real_robot_pretrain_v1.yaml
normalizer_artifact=artifacts/normalizers/real_robot_v1.pt
sampler.alpha=0.5
```

保持第一版算法变量不变：

- `n_obs_steps=1`
- `horizon=30`
- `chunk_size=30`
- `kl_weight=10`
- 当前 ACT 网络宽度和层数
- 不加入 task ID

训练以 optimizer step 为主配置保存/验证频率，记录：manifest hash、normalizer hash、git commit、resolved config、总/分源样本数、采样比例、batch size、学习率、optimizer steps 和 checkpoint hash。

预训练 checkpoint 不得直接进入真机发布流程；命名中包含 `base` 和数据版本。

验收：

- CPU 小 fixture 可以 compose、instantiate 并完成一个 train step。
- Other/All 两种 pool 的 episodes/steps 与 manifest 统计一致。
- state dict 严格 round-trip，无未解释的 missing/unexpected keys。
- 训练开始前强制运行或复用 #1 校验结果。

## #7：实现显式的预训练权重迁移语义

Blocked by: #4, #6  
Type: Prototype

### Question

如何安全地从 base checkpoint 初始化目标任务，而不错误恢复 optimizer、scheduler、EMA 或目标 normalizer？

### Answer

将“恢复同一训练”和“迁移微调”分成两个明确入口：

- `resume_ckpt`：恢复模型、optimizer、scheduler、step、EMA，用于中断续训。
- `pretrained_ckpt`：只加载明确列出的模型权重和冻结的 normalizer，用于新任务微调。

不再使用语义模糊的 `two_train_ckpt`，或保留兼容别名但发出弃用提示。

默认加载全部 ACT 权重，因为机械臂、相机和动作空间相同。加载使用 `strict=True`；若未来做选择性迁移，必须由显式 allowlist 控制并打印每个未加载键，禁止裸 `strict=False`。

微调重新创建 optimizer/scheduler/EMA，global step 从 0 开始，同时在 checkpoint metadata 中记录 parent base checkpoint hash。

验收：

- 测试证明 resume 会恢复 optimizer state，而 pretrained 初始化不会。
- 测试证明 pretrained 初始化前后选定模型权重逐元素一致。
- normalizer hash 与 base 一致。
- 任一非预期 state key 都会失败。

## #8：冻结 Scratch-50 基线协议

Blocked by: #3, #4  
Type: Prototype

### Question

如何得到可与预训练方案公平比较的 collecting_objects 50-episode 基线？

### Answer

创建 A 组固定配置：

- 随机初始化 ACT（视觉 backbone 仍按当前配置使用 ImageNet 初始化）。
- 使用 collecting_objects 全部 50 episodes。
- 使用与 B/C 相同的模型结构、normalizer 控制方案、数据增强、batch size、微调 optimizer steps、checkpoint steps 和推理配置。
- 固定至少 3 个 seed，例如 42、43、44。
- 不用泄漏的 window val loss挑选不同组的不同 step；比较预先指定的相同步数 checkpoint。

验收：配置、命令、resolved YAML、日志和 checkpoint 均可由一次清单复现。

## #9：运行 Other-Pretrain 与 All-Pretrain

Blocked by: #6, #7  
Type: Prototype

### Question

其他任务数据是否提供真正迁移，以及生产上把 collecting 也加入 base pool 是否进一步改善？

### Answer

运行：

- B-base：排除 collecting，使用 261 episodes/81,910 steps。
- C-base：包含 collecting，使用 311 episodes/101,910 steps。

两组保持相同模型和 optimizer-step 预算；每组至少 3 seeds。记录 loss 时同时报告全局 loss 和按 source 的离线 loss，避免总体平均掩盖某些任务严重退化。

base checkpoint 不以多任务成功率为验收标准；其最终价值只由 #10 微调后的目标任务结果判断。

## #10：使用 collecting_objects 全部 50 episodes 微调

Blocked by: #7, #8, #9  
Type: Prototype

### Question

如何在固定 50 episodes 的条件下比较 Scratch、Other-Pretrain 和 All-Pretrain？

### Answer

分别训练 A/B/C：

- A：随机初始化 → collecting 50。
- B：B-base → collecting 50。
- C：C-base → collecting 50。

所有组：

- 全模型微调，不只训练 action head。
- 第一轮统一学习率先设为 `1e-5`；若实现参数组，则 backbone `1e-6`、Transformer/CVAE `5e-6`、action head `1e-5`，但不得在 A/B/C 间使用不同规则。
- 微调 optimizer steps 完全一致。
- 使用全部 50 episodes。
- 每组至少 3 seeds。
- 保存固定 step checkpoints 和 last，不基于泄漏 val loss挑选。

训练结束输出 parent checkpoint、数据 manifest、normalizer、seed、总 steps 和每个 checkpoint hash。

## #11：建立离线与 Shadow 评测

Blocked by: #8, #10  
Type: Prototype

### Question

在承担真机执行风险和成本前，如何筛除明显无效或不连续的 checkpoint？

### Answer

新增固定离线评测集/回放工具，至少报告：

- 每关节 action MAE/RMSE，禁止只报告 14 维平均值。
- action chunk 块内速度、加速度和 jerk。
- 固定 execution steps 下的边界 `new[0] - old[K-1]`。
- 重叠预测一致性 `new[j] - old[K+j]`。
- `new[0] - actual_qpos` 与旧命令 tracking error。
- 近/中/远场景分组指标（需要固定场景清单或人工标注）。

Shadow mode 只读取真机观测并保存预测，不发布动作。A/B/C 使用同一输入 trace，生成可直接比较的 golden report。

验收：固定 checkpoint＋固定 trace＋固定 seed 得到确定性指标；任何 NaN/Inf、逐关节越界或过大跳变使 checkpoint 不能进入真机阶段。

## #12：进行受控真机评测

Blocked by: #11  
Type: Prototype

### Question

预训练是否真正改善 collecting_objects 的成功率、远端抓取和运动连续性？

### Answer

只允许通过现有项目安全整改门槛的 checkpoint 上机。至少具备：逐关节位置/速度限制、观测 watchdog、异常退出保持/停止、操作员急停和低速模式。

固定评测协议：

- A/B/C 使用完全相同的物体、光照、相机、机械臂初始姿态和推理参数。
- 方块位置分为近、中、远三个预定义区域，每个区域至少 10 次 trial/模型。
- 模型顺序随机化，避免设备升温、光照漂移或操作员顺序产生偏差。
- 记录成功率、95% 置信区间、抓取位置误差、平均完成时间、边界最大关节增量、安全拒绝和人工干预次数。
- 报告每个 seed，而不只报告最优 seed。

主判据：

- B 相比 A 提升，才说明其他任务产生真实跨任务迁移。
- C 相比 B 提升，只说明生产上重复利用 collecting 数据有额外价值，不能单独证明跨任务迁移。
- 如果总体成功率提高但远端区域没有提高，则不能声称已解决远端抓取问题。
- 如果成功率提高但 chunk 边界指标无改善，则预训练有效，但动作拼接仍需独立整改。

## #13：形成是否进入长期方案的决策

Blocked by: #12  
Type: Discuss

### Question

第一版无任务 ID 的 ACT 行为预训练是否值得成为未来真机模型的标准初始化？

### Answer

按以下规则决策：

- 若 B 在多个 seed 和近/中/远评测中稳定优于 A：将 Other-Pretrain 纳入标准流水线。
- 若 C 优于 A、但 B 不优于 A：主要收益可能来自重复训练目标数据；不得宣传为跨任务迁移，继续测试 Related-Pretrain 或计算量对照 D。
- 若 B/C 都不优于 A：检查采样、normalizer、频率和任务冲突；不要继续盲目扩大无条件 BC 数据池。
- 若语义相近子集 E 优于全量 C：加入基于数据相似度的离线选择/加权，但仍不把任务 ID输入模型。
- 若行为预训练持续负迁移：进入第二版任务自由 sensorimotor pretraining，使用前向动力学、逆动力学、视觉时序一致性或 masked trajectory reconstruction，让全部数据学习机器人状态转移，而不是要求无条件策略平均不同任务目标。

最终输出一份实验报告，包含所有命令、配置、数据/normalizer/checkpoint hash、训练曲线、离线指标、真机 trial 明细和明确结论。

## 建议提交顺序

每个提交保持可运行、可回滚：

1. 数据 manifest 与校验脚本。
2. 多 HDF5 懒加载数据集及测试。
3. episode 级切分及泄漏测试。
4. 版本化 normalizer 及 parity 测试。
5. 多源平衡 sampler 及统计测试。
6. 预训练 Hydra 配置与 CPU smoke test。
7. `pretrained_ckpt`/`resume_ckpt` 明确加载语义及测试。
8. Scratch/Other/All 训练与微调配置。
9. 离线/Shadow 指标工具。
10. 实验结果文档；真机安全整改未完成前不得加入自动发布动作的评测脚本。

## DeepSeek 执行约束

- 开始每个 ticket 前读取本文件、`AGENTS.md` 和 `docs/PROJECT_AUDIT_REPORT.md`。
- 一次只实现一个 ticket；先补测试或可失败的校验，再修改生产代码。
- 不修改、删除或覆盖原始 HDF5。
- 不提交数据、checkpoint、W&B 文件或 `_outputs/` 内容。
- 不覆盖当前工作区中与本 ticket 无关的用户修改。
- 每个 ticket 完成后在本文件对应 `Answer` 末尾追加：实现 commit、验证命令、结果和剩余风险。
- 未获得明确授权时不得驱动真机、发布 ROS action 或执行数据迁移。
