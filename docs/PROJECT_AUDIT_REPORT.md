# DP_real 训练与真机部署审计报告

审计日期：2026-08-23。审计对象为提交 `f9e415c` 及当时工作区中的未提交修改；仓库并非干净状态，因此后续复现应同时保存 `git diff`。本次检查覆盖训练入口、HDF5 数据管线、ACT/DP2 核心实现、Hydra 配置、checkpoint/EMA，以及 ROS 推理与轨迹回放。已执行 Python 编译检查、26 份 YAML 解析、配置实例化、针对关键路径的最小数值测试和现有数据统计；未进行完整训练，也未驱动真机。

## 结论摘要

**当前版本不建议直接用于正式训练对比或真机运动。** ACT 和二维 Diffusion Policy（DP2）的核心网络与官方实现总体同构，但外围数据、验证和部署代码存在会掩盖模型质量或直接改变控制行为的问题。最严重的是：推理图像被重复缩放、推理观测步数不读取 checkpoint、ACT padding 没有 mask、窗口级随机切分导致训练/验证泄漏，以及真机端无完备的关节约束与失联保护。应先修复这些问题，再讨论网络规模或超参数优化。

严重级别：P0＝会阻断训练或可能造成错误/不安全运动；P1＝显著影响指标、效果或速度；P2＝可维护性和鲁棒性问题。

## P0：必须先修复的问题

| 编号 | 证据与影响 | 建议修复 |
|---|---|---|
| P0-01 推理图像重复归一化 | 数据集输出 `uint8`，normalizer 已在 [`dataset2d_real.py`](../source/dataset/dataset2d_real.py#L97) 配置 `1/255`；[`inference.py`](../inference.py#L610) 和 [`replay_dataset.py`](../replay_dataset.py#L606) 又先除以 255。白色像素进入网络后约为 `1/255`，而不是 1，训练/推理分布严重不一致。 | `get_model_input` 保持 `uint8 [0,255]`，只让 checkpoint 内 normalizer 缩放；增加同一帧经训练与部署预处理后逐元素一致的测试。 |
| P0-02 checkpoint 时序配置被忽略 | [`EnvRunner`](../inference.py#L506) 固定为 `n_obs_steps=3,n_action_steps=8`，创建时未传入 checkpoint 配置（[`inference.py`](../inference.py#L663)）。ACT 只取 `[:1]`（[`act.py`](../source/policy/act.py#L218)），因此实际使用三帧历史中的最旧帧；DP3/CDP 的形状也可能错误。 | 从已保存的 resolved config 构造 runner，并在启动时断言观测 key、shape、历史长度、动作维度和 action horizon 完全一致。历史切片统一取“最近 N 帧”。 |
| P0-03 默认训练和部分配置不可启动 | [`train.py`](../train.py#L35) 默认 `ddp3`，但不存在 `config/ddp3.yaml`。`dp3/cdp2/cdp3` 缺少 `train.py` 无条件读取的 `ema.decay/update_every_n_steps`；DP3/CDP3 默认 task 又没有 `pointcloud/agent_pos`。CDP2 默认将 4D RGB 送入要求 3D 点云的 PointNet。 | 把每个根配置纳入 compose＋instantiate smoke test；给可用模型提供完整配置，不可用模型从 README/入口移除。EMA callback 应按配置可选，而不是无条件创建。 |
| P0-04 数据路径和实验配置不可复现 | 当前 task YAML 指向 `./data/<task>.h5`，实际文件位于 `data/data_wx/`；检查时所有配置路径均不存在。除 `default_task.yaml` 外，具体 task 配置还被 `.gitignore` 排除。 | 使用 Hydra 的绝对路径解析（如 `${hydra:runtime.cwd}`），提交不含隐私路径的 task schema/config，并在训练开始前验证文件、字段、dtype、episode 边界和非空样本。 |
| P0-05 真机执行改变了模型时域且安全层不足 | [`inference.py`](../inference.py#L697) 无条件执行 `action_topp(..., num=4)`，命令数从 `T` 变为 `(T-1)×5+1`，CLI 的插值开关无效。DP2 的 14 个动作会变为 66 个，在 40 Hz 下约 1.65 s 后才重规划；ACT 默认可达约 12.4 s。代码仅以 14 维平均 MSE 判断跳变，没有 NaN、单关节位置/速度/加速度/jerk、工作空间或碰撞限制，也没有 watchdog/急停/finally 安全停止。 | 默认禁止插值并按 checkpoint 的 action horizon 做 receding-horizon replanning；在独立安全层逐关节限幅、限速、限加速度并拒绝非有限值。接入硬件急停、通信超时保持/回安全位和最大观测延迟。 |
| P0-06 轨迹回放不能安全使用 | [`replay_dataset.py`](../replay_dataset.py#L734) 的外层循环会把整条轨迹重复执行至 `max_publish_step`；每 16 步阻塞 1 s，并给右臂注入随机噪声、修改夹爪（[`replay_dataset.py`](../replay_dataset.py#L746)）。这既不是忠实回放，也不适合作为真机验证。 | 在修复前禁用真机回放。改为单次、确定性、可中止的流式回放，并与推理共用同一安全执行器；回放前离线检查整条轨迹约束。 |

另外，`EnvRunner.step` 的 `assert left_action is not None, right_action is not None` 只检查左臂；首次取帧失败会直接解包 `None`；相机同步没有最大时间偏差；程序异常退出也不会主动保持或停止。这些均应随 P0-05 一并处理。

## 训练正确性与性能

### 会影响训练效果或指标

1. **训练/验证泄漏（P1）**：[`train.py`](../train.py#L73) 在高度重叠的时间窗口上 `random_split`。同一 episode 的相邻窗口会同时出现在 train/val，验证损失明显偏乐观。应先按 episode ID 固定划分，再分别生成窗口；normalizer 也只能在训练 episode 上拟合。
2. **ACT padding 语义错误（P1）**：数据集在 episode 尾部重复最后一个 action（[`dataset2d_real.py`](../source/dataset/dataset2d_real.py#L149)），却不返回 `valid_mask/is_pad`；ACT 又把所有位置标为有效（[`act.py`](../source/policy/act.py#L163)）。模型因此被迫学习长段“保持最后动作”。应返回真实 mask，ACT 的 VAE encoder、L1 loss 和 pad head 均使用它；或完全复刻官方零填充＋`is_pad` 约定。
3. **EMA 恢复训练高风险（P1）**：[`callbacks.py`](../source/common/callbacks.py#L10) 跳过 Lightning `EMAWeightAveraging.setup`，到 `on_fit_start` 才创建平均模型。按当前 Lightning 2.6 生命周期，checkpoint 恢复时 `_average_model` 尚不存在，原始权重/averaging state 无法按父类语义恢复，resume 可能从 EMA 权重重新建立平均器。应使用上游 callback 生命周期或添加“连续训练 N 步”和“保存后恢复再训练 N 步”权重逐项一致测试。
4. **数据动作对齐是正确的，不要盲目迁移 ACT 的 `-1` 偏移**：对现有五份 HDF5 检查，episode 内所有 84,010 个可比较位置均满足 `action[t] == qpos[t+1]`。这说明当前数据把 action 定义为下一时刻关节目标。官方 ACT 对其特定真机数据的 `start_ts-1` 补偿不应直接复制；应把这一契约写入数据元数据和测试。
5. **min/max normalizer 易受异常值影响**：目前使用全数据极值映射到 `[-1,1]`。现有文件未发现 NaN/Inf，但采集毛刺会压缩绝大多数样本的动态范围。应只用 train split 统计，记录极值/分位数，越界只报警或按已验证策略裁剪。

### 主要速度瓶颈与优化顺序

- DP2 继承 [`BasePolicy.validation_step`](../source/policy/base_policy.py#L91)，每个验证 batch 先算随机 diffusion loss，再完整执行 100 次反向去噪以计算 action MSE；这通常是验证最重的开销。只在固定少量 batch、固定噪声种子上周期性采样，其余 batch 只算 loss。
- ACT 的 `compute_loss` 每步调用 `loss.item()`（[`act.py`](../source/policy/act.py#L187)），即使调用者丢弃字典也会触发 GPU 同步；直接返回 detached tensor 或删除该字典。
- 四份 20,000 帧、`3×256×256 uint8` 数据仅图像解压后的内存各约 3.66 GiB。`use_mem: true` 在多任务/多进程下会放大 RAM 压力。建议懒加载 HDF5/Zarr、按 episode cache，并测量 worker 与 DDP 的实际 RSS。
- ACT 先把 256 图像搬到 GPU，再缩放到 128；DP2 则对 256 图像直接跑 ResNet，而官方示例通常在约 76 像素 crop 上训练。可在保留原始数据的前提下，缓存确定性 resize，随机 crop 仍在训练端完成。256 相对 76 的像素量约 11 倍。
- 正常训练关闭 `profiler: simple`，确认模型不存在未使用参数后关闭 `find_unused_parameters=True`；在目标 GPU 上验证 BF16/FP16 数值稳定后再启用混合精度。checkpoint 不应只每 100 epoch 保存一次，至少按固定 optimizer step 保存 `last`。

## ACT 与论文/官方代码的一致性

对照 [ACT 论文](https://arxiv.org/abs/2304.13705)、[官方仓库](https://github.com/tonyzhaozh/act)、官方 [`policy.py`](https://github.com/tonyzhaozh/act/blob/main/policy.py)、[`imitate_episodes.py`](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py) 和 [`utils.py`](https://github.com/tonyzhaozh/act/blob/main/utils.py)。

| 项目 | 结论 |
|---|---|
| CVAE/DETR 主体 | **基本一致**：训练时以 `[CLS, qpos, action chunk]` 编码 `mu/logvar`，推理使用零 latent；图像特征、proprioception 和 latent 输入 Transformer decoder；损失为 L1＋`kl_weight×KL`。ResNet18、FrozenBN/GroupNorm、正弦位置编码等结构保留。 |
| Transformer 深度 | **重要偏差**：官方参考为 4 层 encoder、7 层 decoder、dropout 0.1；本项目 [`config/act.yaml`](../config/act.yaml#L24) 为 4/1、dropout 0.05。1 层 decoder 显著降低容量，不能把结果直接称为官方 ACT 复现。建议先做 7 层基线，再把轻量版作为明确命名的消融。 |
| padding/mask | **不一致且构成 bug**：官方数据返回 `is_pad`，编码器和重建损失忽略 padding；本项目全部视为有效，见前述 P1。 |
| 归一化 | **实验偏差**：官方 ACT 参考使用 mean/std（并对 std 设下限），本项目使用 min/max。可用，但必须作为实验变量记录，不能与官方结果无条件比较。 |
| temporal aggregation | **实现不等价**：官方每个控制步查询新 chunk，并聚合所有覆盖当前时刻的预测；本项目 buffer 以 `chunk_size // n_action_steps` 建立、首次复制同一 chunk，更新索引又混用 action/history 维度并硬编码 `.cuda()`（[`act.py`](../source/policy/act.py#L237)）。默认 `n_action_steps==chunk_size` 时实际上没有重叠聚合；启用前应重写并用手工序列测试。 |
| 图像 crop | **已确认 bug**：当 `random_crop:false` 时，CenterCrop 被写到 `this_normalizer`，随后又被 `Identity/Normalize` 覆盖（[`multi_image_obs_tokens_encoder.py`](../source/model/ACT/multi_image_obs_tokens_encoder.py#L108)）。配置声称 120 crop，实测输出仍为 128。 |
| 多相机共享 backbone | `share_rgb_model:false` 时每相机独立 backbone，与官方共享单 backbone 不同；当前仅一相机影响不大，多相机会增加参数/显存。`true` 分支还把空列表传给 backbone（[`multi_image_obs_tokens_encoder.py`](../source/model/ACT/multi_image_obs_tokens_encoder.py#L160)），目前不可用。 |
| 训练策略 | cosine warmup、EMA、batch 128 均是本项目新增；官方 ACT 参考没有同样的 EMA/调度。应按 optimizer step 而不是 epoch 对齐训练量，并分别报告这些改动。 |

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
| 部署语义 | 核心 policy 是 receding horizon，但当前 runner 在整段动作并额外插值后才重规划，已不再等价于论文/官方执行流程。该差异比 U-Net 尺寸更可能影响真机结果。 |

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
- 用训练频率确定 `publish_rate` 和 `Ta`。当前采集控制代码暗示 50 Hz，而推理默认 40 Hz，但 HDF5 没有频率证据。频率不一致会改变速度和动作持续时间。
- 同步队列应同时约束相机和关节时间戳，超过最大 age/skew 就保持当前位置并告警。日志限频，避免每个控制周期 `print/cprint` 造成 jitter。
- 控制器外必须有独立 safety supervisor：逐关节软/硬限位、速度/加速度/jerk 限制、工作空间/自碰撞检查、通信 watchdog、硬件急停和异常退出安全状态。模型输出永远不能直接作为最终安全命令。

### 建议验收流程

1. **离线**：所有配置 compose/instantiate；单 batch 过拟合；episode-disjoint split；训练/推理预处理一致；checkpoint resume 与连续训练对齐。
2. **仿真/日志回放**：只生成命令不发布，检查每维范围、速度、加速度、推理延迟、观测 age 和重规划周期；固定随机种子比较 golden trace。
3. **Shadow mode**：真机只读取传感器，预测与人工/历史命令对比，不下发。
4. **低速空载**：降低速度/力矩限制，操作员手持急停，先单臂、短 horizon、无插值，再逐项放开。
5. **正式评测**：固定任务初始条件和成功判据，报告成功率、控制频率、端到端 P50/P95 latency、观测 skew、干预/安全拒绝次数，而不只报告训练 loss。

## 建议整改顺序

1. **停止危险路径**：暂时禁用 `replay_dataset.py` 真机发布；修复推理双缩放、checkpoint schema/时序、动作插值和独立安全层。
2. **恢复可信训练**：按 episode 切分，train-only normalizer，ACT padding mask，EMA resume，修正数据路径和全部 Hydra 配置契约。
3. **建立可比较基线**：ACT 先复现 4/7 层与官方 padding；DP2 先复现 `16/2/8`、随机 crop 和官方 EMA。之后再分别消融轻量网络、14-step horizon、min/max normalization。
4. **优化吞吐**：减少 DP 验证采样、删除每步 GPU 同步、优化图像尺寸/缓存、关闭常驻 profiler，再评估混合精度和 DDP 参数。
5. **防回归**：增加 `tests/`，至少覆盖预处理 parity、episode 隔离、action/qpos 对齐、ACT mask、所有配置实例化、checkpoint resume 和无 ROS 的 action safety checker；CI 运行编译、YAML 和 CPU smoke tests。

## 其他可复现性问题

- `requirements.txt` 同时列出 `lightning` 与 `pytorch_lightning`，大量包未锁版本，ROS/OpenCV/IPython 等运行期依赖也未形成完整环境清单。当前 EMA 行为依赖 Lightning 2.6；应以 lockfile/容器固定训练与部署环境。
- README 宣称支持的 DDP3/FP3/ARP 与可用根配置不一致；批量训练示例中的 override 名称也与当前 `optimizer.lr`、`dataloader.train.batch_size` 不一致。
- 仓库没有自动化测试或 CI。此次最小测试已经稳定复现：白像素双缩放、ACT 使用最旧观测、CenterCrop 无效、episode 泄漏、ACT padding 无 mask、EMA 配置缺失和默认配置缺失。应把这些脚本转成回归测试。

本报告对静态和轻量数值路径给出高置信结论；模型最终成功率、ROS 驱动单位以及机械臂动力学安全边界仍必须在明确的硬件/任务验收协议下测量，不能由代码审计替代。
