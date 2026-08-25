# DP_real 训练与真机部署审计报告

原始审计日期：2026-08-23，审计基线为提交 `f9e415c`。复核日期：2026-08-25，复核对象为提交 `d8b4bc4`；复核时工作区无未提交修改。本次复核对照了基线后的代码变更和 [ACT Tuning Tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?tab=t.0)。状态含义：**已修复**＝当前代码已消除原问题；**部分修复**＝主要路径已改，但原条目仍有残留风险；**未应用**＝经指南或本项目数据契约复核后，决定不采用原建议；**未修复**＝当前代码仍可复现。未进行完整训练，也未驱动真机。

## 结论摘要

**ACT 的主要训练配置与两项部署语义已有实质修复，但当前版本仍不建议直接用于正式训练对比或无安全监督的真机运动。** 推理图像双缩放、checkpoint 时序传递、ACT padding mask、ACT decoder/crop 配置和默认训练入口已经修复；ACT chunk 也已按约 1 秒执行。当前主要阻塞项转为：窗口级随机切分导致训练/验证泄漏、normalizer 仍使用全数据、EMA resume 生命周期风险、其他根配置仍可能不可启动，以及真机端没有逐关节约束、watchdog 和异常退出安全状态。

严重级别：P0＝会阻断训练或可能造成错误/不安全运动；P1＝显著影响指标、效果或速度；P2＝可维护性和鲁棒性问题。

## P0：必须先修复的问题

| 编号 | 状态 | 本次复核说明 | 后续要求 |
|---|---|---|---|
| P0-01 推理图像重复归一化 | **部分修复** | [`inference.py`](../inference.py) 的 `get_model_input` 已不再除以 255，部署输入保持 `uint8 [0,255]`，由 checkpoint normalizer 完成缩放。提交 `d6af39a` 修复了主推理路径。[`replay_dataset.py`](../replay_dataset.py) 的同名辅助函数仍会 `/255`；当前回放流程不调用策略，因此暂不影响现有回放，但若恢复模型推理会重现问题。 | 增加训练/部署预处理逐元素一致测试，并删除或统一回放脚本中的旧预处理。 |
| P0-02 checkpoint 时序配置被忽略 | **部分修复** | [`inference.py`](../inference.py) 已从 checkpoint 的 resolved config 读取 `n_obs_steps`、`n_action_steps`，传给 `EnvRunner`，并与 policy 属性断言一致（提交 `6ca5e47`）。这消除了 ACT 使用错误历史帧和固定 `3/8` 的问题。 | 仍需校验观测 key/shape、动作维度、horizon、normalizer schema；首帧 `None` 仍应显式拒绝。 |
| P0-03 默认训练和部分配置不可启动 | **部分修复** | [`train.py`](../train.py) 默认配置已由不存在的 `ddp3` 改为 `dp2`（提交 `1598c30`），默认入口不再因配置名直接失败。但 `dp3/cdp2/cdp3` 仍只配置 `ema.update_after_steps`，而训练入口无条件读取 `ema.decay/update_every_n_steps`，其 shape/task 契约问题也未统一验证。 | 为所有公开根配置增加 compose＋instantiate smoke test；不可用配置应修复或从支持列表移除。 |
| P0-04 数据路径和实验配置不可复现 | **未修复** | task YAML 仍使用 `./data/<task>.h5`，当前 `data` 是机器相关的外部符号链接；`.gitignore` 仍忽略 `config/task/*`（只例外 `default_task.yaml`），训练前也没有完整数据 schema 校验。 | 使用 Hydra 运行目录无关的路径解析，提交可复现模板，并在训练开始前验证文件、字段、dtype、episode 边界和非空样本。 |
| P0-05 真机执行改变时域且安全层不足 | **部分修复** | 主推理路径已注释掉无条件 `action_topp`，ACT 现为 `chunk_size=n_action_steps=30`、默认 30 Hz，即每个 chunk 约 1 秒后重新查询。这符合指南“约 1 秒 chunk”以及关闭 temporal aggregation、完整执行 chunk 的建议。 | 安全层仍未修复：只有 14 维平均 MSE，阈值还从 0.01 放宽到 0.1；没有 finite check、逐关节位置/速度/加速度/jerk、工作空间/碰撞限制、观测超时、watchdog、急停和 `finally` 安全停止。 |
| P0-06 轨迹回放不能安全使用 | **部分修复，仍应禁用真机发布** | [`replay_dataset.py`](../replay_dataset.py) 已设置 `action_topp(..., num=0)`、`add_noise=False`，夹爪强制修改也已注释，因而默认不再插值或注入噪声。但外层循环仍会重复整条轨迹，每 100 步仍阻塞 1 秒，且直接发布到机器人。 | 改为单次、确定性、可中止的流式回放，并与推理共用安全执行器；完成前只做离线回放。 |

另外，`EnvRunner.step` 的 `assert left_action is not None, right_action is not None` 只检查左臂；首次取帧失败会直接解包 `None`；相机同步没有最大时间偏差；程序异常退出也不会主动保持或停止。这些均应随 P0-05 一并处理。

## 训练正确性与性能

### 会影响训练效果或指标

1. **训练/验证泄漏（P1，未修复）**：[`train.py`](../train.py) 仍在高度重叠的时间窗口上 `random_split`。同一 episode 的相邻窗口会进入 train/val，验证损失偏乐观。应先按 episode ID 固定划分，再分别生成窗口。
2. **ACT padding mask（P1，已修复）**：[`dataset2d_real.py`](../source/dataset/dataset2d_real.py) 现按真实区间返回 `is_pad`；[`act.py`](../source/policy/act.py) 将其用于 CVAE encoder padding mask 和 L1 mask（提交 `b90380d`）。补齐值虽然仍重复末动作，但已不参与上述学习目标；当前加权后再 `.mean()` 的行为也与 ACT 参考实现一致。`is_pad_head` 与官方参考一样未加入损失，不再把“训练 pad head”列为必须修复项。
3. **EMA 恢复训练（P1，未修复）**：[`callbacks.py`](../source/common/callbacks.py) 仍跳过 `EMAWeightAveraging.setup`，到 `on_fit_start` 才创建平均模型；resume 的 averaging state 仍有生命周期风险。ACT Tuning Tips 没有要求 EMA，因此 EMA 属于本项目扩展，不能用指南证明其正确性。应补连续训练与 save/resume 的逐权重一致测试。
4. **动作/qpos 的 `-1` 偏移（未应用）**：原审计对五份 HDF5 的 84,010 个可比较位置确认 `action[t] == qpos[t+1]`。因此没有迁移 ACT 官方特定数据采集代码中的 `start_ts-1`，这是基于本项目数据契约的有意决定，不是遗漏。应把契约写入数据元数据和回归测试。
5. **normalizer 切换（部分未应用，仍有 P1）**：没有把 min/max 强制改成 ACT 参考代码的 mean/std，因为 ACT Tuning Tips 并未规定归一化方法，且这应作为本地数据的实验变量。但 `dataset.get_normalizer()` 仍在切分后对全数据拟合，造成验证信息泄漏；此部分必须随 episode split 修复。还应记录分位数并对异常极值报警。

### 主要速度瓶颈与优化顺序

- **未修复**：DP2 继承 [`BasePolicy.validation_step`](../source/policy/base_policy.py#L91)，每个验证 batch 先算随机 diffusion loss，再完整执行 100 次反向去噪以计算 action MSE；这通常是验证最重的开销。只在固定少量 batch、固定噪声种子上周期性采样，其余 batch 只算 loss。
- **未修复**：ACT 的 `compute_loss` 每步仍调用 `loss.item()`（[`act.py`](../source/policy/act.py)），即使调用者丢弃字典也会触发 GPU 同步；直接返回 detached tensor 或删除该字典。
- **部分优化**：ACT、DP2、CDP2 已配置 `obs_only_n_steps`，减少单样本无用图像的拷贝和传输；但 `use_mem:true` 仍在 dataset 初始化时读取整份 HDF5，不能降低常驻 RAM。四份 20,000 帧、`3×256×256 uint8` 数据仅图像解压后各约 3.66 GiB，仍建议懒加载 HDF5/Zarr、按 episode cache，并测量 worker/DDP 的实际 RSS。
- ACT 当前保持 `256×256` 输入，不再缩放到 128；这不是 ACT Tuning Tips 规定的错误项。是否使用更小 resize/crop 应作为吞吐与成功率消融，而不是复现前置条件。DP2 仍对 256 图像直接跑 ResNet，可在保留原始数据的前提下评估缓存 resize/crop。
- **部分修复**：训练入口已注释 `SampleCallback`，避免额外采样开销；但 `profiler:simple` 和多卡 `find_unused_parameters=True` 仍常驻。checkpoint 已保存 `last` 和 top 5，但仍每 100 epoch 触发，建议改为固定 optimizer step。

## ACT 与论文/官方代码的一致性

对照 [ACT 论文](https://arxiv.org/abs/2304.13705)、[官方仓库](https://github.com/tonyzhaozh/act)、官方 [`policy.py`](https://github.com/tonyzhaozh/act/blob/main/policy.py)、[`imitate_episodes.py`](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)、[`utils.py`](https://github.com/tonyzhaozh/act/blob/main/utils.py) 和 [ACT Tuning Tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?tab=t.0)。其中“参考实现一致性”和“官方调优建议”是两个不同维度：前者记录实现差异，后者用于判断该差异是否需要应用到当前实验。

| 项目 | 状态 | 本次复核结论 |
|---|---|---|
| CVAE/DETR 主体 | **保持一致** | 训练时以 `[CLS, qpos, action chunk]` 编码 `mu/logvar`，推理使用零 latent；损失为 L1＋`kl_weight×KL`。当前 `kl_weight=10` 属于指南建议的高 KL 权重。 |
| Transformer 深度与 dropout | **已修复** | [`config/act.yaml`](../config/act.yaml) 已从 4/1 层、dropout 0.05 改为 4/7 层、dropout 0.1（提交 `4b79715`），与官方参考结构一致。 |
| padding/mask | **已修复** | 数据集已返回 `is_pad`，CVAE encoder 与 L1 reconstruction loss 均忽略补齐位，见前述 P1。 |
| chunk 与查询频率 | **已按指南调整** | `chunk_size=n_action_steps=30`，推理默认 30 Hz，每个 chunk 对应约 1 秒真实运动并完整执行；`temporal_agg=False`。这直接对应指南最优先的 chunk 调参和“关闭聚合、完整执行 chunk”建议。最终仍应以数据真实 `sample_hz` 校准，而不是只依赖 CLI 默认值。 |
| temporal aggregation 实现 | **未应用，保留为禁用路径** | 原审计指出的 buffer 维度和硬编码 `.cuda()` 问题仍在，但当前配置关闭该功能；指南也明确建议在速度/查询频率场景考虑关闭 temporal aggregation。因此不把重写此路径列为当前 ACT 基线阻塞项；若未来启用，必须先修复并测试。 |
| 图像 crop/尺寸 | **bug 已修复；小 crop 建议未应用** | [`multi_image_obs_tokens_encoder.py`](../source/model/ACT/multi_image_obs_tokens_encoder.py) 已把 CenterCrop 正确赋给 randomizer（提交 `dc72a39`）；配置改为原生 `256×256`、等尺寸 crop。指南没有要求约 76/120 像素 crop，因此不把较小 crop 当成官方调优要求，只保留为性能消融。 |
| 多相机 backbone | **当前配置符合指南** | `share_rgb_model:false` 表示每相机独立 backbone，正是 Tuning Tips 的建议。原报告把“与参考代码共享 backbone 不同”作为负面偏差不恰当，现改为有意调优选择。当前只有一相机时无实质差异；未使用的 `true` 分支问题不阻塞当前配置。 |
| batch size 与学习率 | **已按指南调整** | batch size 保持 128，学习率从 `1e-5` 提高到 `5e-5`，符合“大 batch 相应提高 lr”的示例方向。是否最优仍需按有效 batch size 做消融。 |
| 训练时长与 checkpoint | **已调整，需用 step 复核** | `max_epochs` 从 1 提高到 500，`save_top_k=5` 且保存 `last`，符合长时间训练和尝试多个 checkpoint 的方向。但指南对真机给出 5k–8k steps 的量级，epoch 无法直接证明已满足；报告训练时必须给出 optimizer steps 和 loss plateau。 |
| loss 与控制量 | **符合指南** | 当前使用 L1 而非 L2；数据契约为绝对关节目标而非 delta/velocity control。无需修改。 |
| 归一化 | **未应用 mean/std 切换** | 参考代码使用 mean/std，本项目保留 min/max。Tuning Tips 未规定该项，因此将其记录为实验偏差而非待修 bug；但 train-only 拟合问题仍必须修复。 |
| scheduler/EMA | **本项目扩展** | cosine warmup 和固定 0.9 EMA 不属于 Tuning Tips 要求。可以保留并单独报告，但 EMA resume 正确性仍是 P1，不能借指南豁免。 |

## DP2 与论文/官方代码的一致性

对照 [Diffusion Policy 论文](https://arxiv.org/abs/2303.04137)、[官方仓库](https://github.com/real-stanford/diffusion_policy)、官方 [`diffusion_unet_image_policy.py`](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_image_policy.py) 和[训练配置](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/config/train_diffusion_unet_image_workspace.yaml)。

| 项目 | 结论 |
|---|---|
| 核心算法 | **高度一致**：观测/动作归一化、视觉特征作为 global condition、随机 timestep 加噪、epsilon MSE、Conditional U-Net 1D、DDPM 迭代去噪和从 `To-1` 开始取执行动作均与官方实现一致；视觉特征也只在去噪循环外提取一次。 |
| horizon | **有效但偏离参考**：本项目为 `Tp=16, To=3, Ta=14`，官方图像基线通常为 `16/2/8`。当前 `To-1+Ta=16` 没越界，但一次开放环执行 14 步会降低对扰动的响应；论文也指出过长 action horizon 通常不利。先以 `Ta=8` 建立基线，再按真实控制频率调参。 |
| 网络与视觉增强 | 本项目 U-Net 为 `[256,512,1024]`，官方参考为 `[512,1024,2048]`；本项目 256 图像且无 crop/augmentation（`crop_shape:null` 使 `random_crop:true` 无效），官方配置使用较小随机 crop。前者更轻的 U-Net 可能损失容量，但视觉 backbone 处理更大图又拖慢训练。 |
| 噪声与采样 | 100 个训练/推理 DDPM step、epsilon prediction、squared cosine schedule 与参考方向一致。部署延迟敏感时可以评估 DDIM/更少步数，但必须重新验证成功率，不能只比较单步 latency。 |
| EMA | 官方使用随 step 变化、上限接近 0.9999 的 EMA；本项目固定 0.9，有效记忆仅约十步，且还有 resume 问题。应复刻官方 EMA 或通过消融确定。 |
| 图像 encoder | 本项目主要源自官方实现；其中 `random_crop:false` 的 CenterCrop 覆盖问题在 DP encoder 也存在（[`multi_image_obs_encoder.py`](../source/model/DP2/multi_image_obs_encoder.py#L86)）。当前 crop 为 null，属于潜伏 bug。 |
| 部署语义 | **插值已修复，开放环长度仍需评估**：主 runner 不再额外插值，但仍完整执行 policy 返回的整段动作后才重规划。对 DP2，这一行为是否合适取决于 `n_action_steps` 与真实控制频率；应测扰动响应并与较短 `Ta` 对照。 |

建议为人工迁移的目录补充 `UPSTREAM.md`：记录上游仓库、精确 commit、许可证、复制文件清单和本地 patch。否则未来无法判断“官方代码更新”还是“本地迁移回归”。

## 数据采集与真机部署注意事项

### 数据契约

- 在 HDF5 根属性保存 `sample_hz`、单调时间戳、机器人/固件版本、14 维关节顺序、每维单位、夹爪标定、相机名称/编码、图像尺寸、内外参和同步容差。当前转换脚本通过 `min_len` 静默截断多传感器长度，并丢弃这些元数据，无法证明数据频率或同步质量。
- 明确颜色空间。ROS `passthrough` 加上被注释的 BGR→RGB 转换，不能保证训练与部署颜色一致。用红/绿/蓝标定板制作单元测试。
- 当前 480×640 图像被强行拉伸为 256×256，会改变几何比例。应固定相机 ROI，保持纵横比 resize＋crop，并在训练和推理复用同一可序列化变换。
- 不允许以 `min_len` 掩盖掉帧；按时间戳配对并记录每帧 skew。统计 episode 长度、空帧、重复帧、饱和像素、关节越界、速度突变和 action/qpos 对齐。

### 上机前控制契约

- 核对 AgileX 驱动 topic、消息类型、左右臂顺序、弧度/角度、夹爪单位与零位；仓库内 reset pose 和 `0.025` 夹爪阈值均为硬编码。现有数据不同任务的夹爪范围差异明显，不能共用一个阈值。
- 模型启动时做一次无发布 warmup；显式选择 `device/map_location`，严格加载 state dict，拒绝 missing/unexpected key，并打印 checkpoint hash、resolved config、normalizer 统计和输入 schema。
- 用训练频率确定 `publish_rate` 和 `Ta`。推理默认值已从 40 Hz 改为 30 Hz，使 ACT 的 30-step chunk 名义上对应 1 秒；但 HDF5 没有 `sample_hz` 证据，不能据此证明训练/部署频率一致。
- 同步队列应同时约束相机和关节时间戳，超过最大 age/skew 就保持当前位置并告警。日志限频，避免每个控制周期 `print/cprint` 造成 jitter。
- 控制器外必须有独立 safety supervisor：逐关节软/硬限位、速度/加速度/jerk 限制、工作空间/自碰撞检查、通信 watchdog、硬件急停和异常退出安全状态。模型输出永远不能直接作为最终安全命令。

### 建议验收流程

1. **离线**：所有配置 compose/instantiate；单 batch 过拟合；episode-disjoint split；训练/推理预处理一致；checkpoint resume 与连续训练对齐。
2. **仿真/日志回放**：只生成命令不发布，检查每维范围、速度、加速度、推理延迟、观测 age 和重规划周期；固定随机种子比较 golden trace。
3. **Shadow mode**：真机只读取传感器，预测与人工/历史命令对比，不下发。
4. **低速空载**：降低速度/力矩限制，操作员手持急停，先单臂、短 horizon、无插值，再逐项放开。
5. **正式评测**：固定任务初始条件和成功判据，报告成功率、控制频率、端到端 P50/P95 latency、观测 skew、干预/安全拒绝次数，而不只报告训练 loss。

## 建议整改顺序

1. **完成危险路径整改**：主推理的双缩放、checkpoint 时序和无条件插值已修复；下一步是独立 action safety layer、观测/通信 watchdog、异常退出安全状态，以及把 `replay_dataset.py` 改为单次安全回放。在此之前禁用回放真机发布。
2. **恢复可信训练**：按 episode 切分、只用 train episodes 拟合 normalizer、验证 EMA resume、修正数据路径和全部 Hydra 配置契约。ACT padding 已完成，不再列入待办。
3. **完成 ACT 基线验证**：4/7 层、padding、约 1 秒 chunk、KL=10、L1、独立 backbone、batch 128/lr 5e-5 已落地；下一步按 optimizer step 验证 5k–8k+ steps、loss plateau 和多个 checkpoint。mean/std、小 crop、temporal aggregation 不作为当前官方指南硬性要求。
4. **优化吞吐**：减少 DP 验证采样、删除每步 GPU 同步、优化图像尺寸/缓存、关闭常驻 profiler，再评估混合精度和 DDP 参数。
5. **防回归**：增加 `tests/`，至少覆盖预处理 parity、episode 隔离、action/qpos 对齐、ACT mask、所有配置实例化、checkpoint resume 和无 ROS 的 action safety checker；CI 运行编译、YAML 和 CPU smoke tests。

## 其他可复现性问题

- `requirements.txt` 同时列出 `lightning` 与 `pytorch_lightning`，大量包未锁版本，ROS/OpenCV/IPython 等运行期依赖也未形成完整环境清单。当前 EMA 行为依赖 Lightning 2.6；应以 lockfile/容器固定训练与部署环境。
- README 宣称支持的 DDP3/FP3/ARP 与可用根配置不一致；批量训练示例中的 override 名称也与当前 `optimizer.lr`、`dataloader.train.batch_size` 不一致。
- 仓库仍没有自动化测试或 CI。白像素双缩放、ACT 错误历史帧、ACT CenterCrop 和 padding mask 已由代码修复，但尚未形成回归测试；episode 泄漏、EMA 配置/恢复和非默认根配置问题仍可复现。应把已修复项固化为防回归测试，并为未修复项先添加失败测试。

本报告对静态和轻量数值路径给出高置信结论；模型最终成功率、ROS 驱动单位以及机械臂动力学安全边界仍必须在明确的硬件/任务验收协议下测量，不能由代码审计替代。
