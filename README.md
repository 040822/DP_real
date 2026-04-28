# DP_real — 面向真实场景双臂机器人操作的扩散策略

基于 PyTorch 的框架，用于在真实双臂移动机械臂（7 自由度、RGB 相机、ROS 推理）上训练和部署多种**扩散策略（Diffusion Policy）** 变体。

## 支持的算法

| 策略 | 输入 | 骨干网络 | 调度器 | 说明 |
|--------|-------|--------|-----------|-------------|
| **DP2** | 多视角 RGB（ResNet） | 1D UNet | DDPM | 标准扩散策略，使用 2D 图像观测 |
| **DP3** | 3D 点云（PointNet++） | 1D UNet | DDIM | 扩散策略，使用 3D 点云输入 |
| **DDP2** | 多视角 RGB（ResNet） | Transformer | DDIM | **双**扩散策略：粗粒度关键帧 + 细粒度插值（2D） |
| **DDP3** | 3D 点云（PointNet++） | Transformer | DDIM | **双**扩散策略：粗粒度关键帧 + 细粒度插值（3D） |
| **CDP3** | 3D 点云（PointNet++） | Causal Transformer | DDIM | **因果**扩散策略 — 带 KV 缓存的流式自回归推理 |
| **FP3** | 3D 点云（PointNet++） | 1D UNet | ConsistencyFM | 流匹配策略（Flow Policy），使用流匹配替代扩散 |
| **ACT** | 多视角 RGB（ResNet） | DETR Transformer（CVAE） | — | 动作分块 Transformer，支持时序集成 |
| **ARP** | 3D 点云（PointNet++） | Causal Transformer | — | 自回归策略，分块 token 顺序生成 |

## 项目结构

```
DP_real/
├── train.py                  # 训练入口（Hydra 配置）
├── inference.py              # 真实机器人 ROS 推理
├── replay_dataset.py         # 通过 ROS 回放预先录制的轨迹
├── config/                   # Hydra YAML 配置文件
│   ├── dp2.yaml              #   DP2 配置
│   ├── dp3.yaml              #   DP3 配置
│   ├── ddp2.yaml             #   DDP2 配置
│   ├── cdp3.yaml             #   CDP3 配置
│   └── task/                 #   任务配置（数据集路径、观测形状等）
├── source/
│   ├── policy/               # 策略 LightningModules（dp2, dp3, ddp3, cdp3, fp3, act, arp）
│   ├── model/                # 模型架构（UNet, Transformer, PointNet++, ResNet）
│   ├── dataset/              # HDF5 数据集类
│   └── common/               # 共享工具（回调、归一化器、采样器）
└── script/                   # 工具与控制脚本
    ├── train.sh              #   训练启动脚本（封装 train.py，指定策略/任务/GPU）
    ├── control.py            #   PIPER 机械臂 ROS 关节控制接口
    ├── piper.py              #   PIPER 机械臂正逆运动学 + Meshcat 可视化
    ├── piper_pino.py         #   PIPER 机械臂 Pinocchio 运动学接口
    ├── ckpt_change.py        #   检查点参数修改工具（修改 _target_ 路径后另存）
    ├── list_items.py         #   列出 HDF5 数据集内部结构（键名、形状、类型）
    ├── read_hdf5.py          #   读取并导出 HDF5 数据集内容（JSON 摘要 / 完整导出）
    ├── zip.sh                #   项目打包脚本（排除 _outputs、data、outputs 目录）
    ├── cobot_magic/          #   Cobot Magic 机器人数据提取与插值脚本
    ├── cobot_magic_edp/      #   Cobot Magic EDP 变体数据提取与插值脚本
    └── cobot_magic_gau/      #   Cobot Magic Gau 变体配置与数据提取脚本
```

## 环境安装

```bash
# 创建并激活 conda 环境
conda create -n dp_real python=3.9
conda activate dp_real

# 安装 PyTorch（根据 CUDA 版本调整）
pip install torch torchvision torchaudio

# 安装核心依赖
pip install numpy einops diffusers omegaconf hydra-core \
    pytorch-lightning wandb h5py opencv-python numba dill \
    termcolor tqdm zarr

# 真实机器人推理还需要安装 ROS 包：
# rospy, cv_bridge, sensor_msgs, geometry_msgs, nav_msgs, std_msgs
```

克隆仓库：

```bash
git clone <仓库地址>
cd DP_real
```

## 数据集准备

将演示数据以 HDF5（`.h5`）文件格式放置在 `data/` 目录下。每个 episode 应包含：

- **观测**：相机图像（`cam_high`、`cam_left`、`cam_right`）和关节位置（`qpos`）
- **动作**：双臂目标关节位置

在 `config/task/` 下创建任务配置文件，指定数据集路径、观测形状和动作维度。可参考 `config/task/default_task.yaml` 模板。

## 训练

### 使用 Shell 脚本

```bash
bash script/train.sh <策略配置> <任务名> <GPU编号>

# 示例：
bash script/train.sh dp3 2a_grasp_card 0
bash script/train.sh dp2 2a_stack_cubes 1
bash script/train.sh cdp3 2a_collect_block 2
```

可用 `policy_config` 值：`dp2`、`dp3`、`ddp2`、`cdp3`  
可用 `task_name` 值：参见 `config/task/` 目录下的文件

### 直接调用 Python

```bash
python -u train.py --config-name=dp3 task=2a_grasp_card
```

训练日志通过 **Weights & Biases** 追踪（项目名：`DP_real`）。模型检查点和 Hydra 输出保存在 `_outputs/<策略>/<任务>/<时间戳>/` 目录下。

## 真实机器人推理

```bash
python inference.py \
    --ckpt_dir /path/to/checkpoints \
    --ckpt_name policy_best.ckpt \
    --task_name aloha_mobile_dummy \
    --max_publish_step 250 \
    --publish_rate 40
```

主要参数：

| 参数 | 默认值 | 说明 |
|----------|---------|-------------|
| `--ckpt_dir` | *必填* | 检查点目录路径 |
| `--ckpt_name` | `policy_best.ckpt` | 检查点文件名 |
| `--task_name` | `aloha_mobile_dummy` | 任务名称，用于配置解析 |
| `--policy_class` | `DP2` | 策略类名 |
| `--max_publish_step` | `250` | 最大动作发布步数 |
| `--publish_rate` | `40` | 控制频率（Hz） |
| `--use_robot_base` | — | 启用移动底盘控制 |
| `--use_depth_image` | — | 启用深度图像输入 |

ROS 话题可通过命令行参数覆盖（如 `--img_front_topic`、`--puppet_arm_left_topic` 等）。

## 回放预录制数据

```bash
python replay_dataset.py \
    --dataset_dir /data/datasets/task.h5 \
    --task_name playing_card_delivery \
    --episode_idx 1 \
    --max_publish_step 250
```

## 关键配置参数

| 参数 | 说明 |
|-----------|-------------|
| `horizon` | 动作预测总长度（帧数） |
| `n_obs_steps` | 用作输入的观测帧数 |
| `n_action_steps` | 每次推理执行的动作帧数 |
| `policy.noise_scheduler` | DDPM / DDIM / ConsistencyFM 调度器 |
| `policy.num_inference_steps` | 推理时的去噪步数 |
| `optimizer.lr` | 学习率 |
| `trainer.max_epochs` | 最大训练轮数 |
| `dataloader.train.batch_size` | 训练批次大小 |
| `logging.project` | WandB 项目名称 |

## 致谢

本项目基于以下工作构建：

- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)（Chi et al.）
- [3D Diffusion Policy](https://3d-diffusion-policy.github.io/)（Ze et al.）
- [ACT](https://tonyzhaozh.github.io/aloha/)（Zhao et al.）
- HuggingFace [Diffusers](https://github.com/huggingface/diffusers)

## 许可证

本项目基于 MIT 许可证发布。
