# Example
# python -u train.py --config-name=dp3 task=1a_pick_meat_3d 2>&1 | tee 1a_pick_meat_dp3.out


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["HYDRA_FULL_ERROR"] = "1"


import hydra
import torch
import pathlib
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.swa_utils import get_ema_avg_fn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, TQDMProgressBar
from lightning import seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch import LightningModule
from lightning.pytorch.strategies import DDPStrategy

from source.common.callbacks import ModelAveragingCallback, SaveConfigCallback
from source.common.callback_sample import SampleCallback

if __name__ == "__main__":
    import sys
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

OmegaConf.register_new_resolver("eval", eval, replace=True) # 注册eval解析器


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="dp2"
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    output_dir = pathlib.Path(HydraConfig.get().run.dir)
    
    resume_ckpt = cfg.get("resume_ckpt", None)
    two_train_ckpt = cfg.get("two_train_ckpt", None) 
    
    # set seed
    seed_everything(cfg.training.seed)
    # configure model
    model: LightningModule = hydra.utils.instantiate(cfg.policy)
    
    # 二阶段训练，加载第一阶段的模型权重继续训练
    if two_train_ckpt is not None:
        print(f"[Train] Loading model weights from {two_train_ckpt} for two-stage training...")
        state_dict = torch.load(two_train_ckpt, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state_dict, strict=False) # 加载权重，允许不完全匹配
    
    # [DDP3] 处理数据集大小和分割idx
    if cfg.policy_name == "DDP2":
        horizon = cfg.policy.coarse_dp.horizon
        internal = cfg.policy.coarse_dp.internal
        sample_horizon = (horizon-1)*internal + horizon
        print("[Train] sample_horizon:", sample_horizon)
        idx = torch.linspace(0, sample_horizon - 1, steps=horizon).long()
        model.set_idx(idx) # 设置idx
        cfg.task.dataset.horizon = sample_horizon # 设置数据集的horizon
        # cfg.task.dataset.pad_after = int(1 * sample_horizon) # 设置数据集的pad_after, 因为DDP3需要在数据末尾进行padding以满足horizon长度
        cfg.task.dataset.pad_after = int(0.25 * sample_horizon) # 设置数据集的pad_after, 因为DDP3需要在数据末尾进行padding以满足horizon长度
        print("[Train] dataset_pad_after:", cfg.task.dataset.pad_after)
        
    
    dataset = hydra.utils.instantiate(cfg.task.dataset)

    # 按 episode 切分 train / eval，而非按样本随机切分，避免同一 episode 泄漏到两个集合
    episode_ends = dataset.data["episode_ends"]
    n_episodes = len(episode_ends)
    val_ratio = 0.05
    rng = np.random.default_rng(cfg.training.seed)
    n_val = min(max(1, int(n_episodes * val_ratio)), n_episodes - 1)
    val_episodes = rng.choice(n_episodes, size=n_val, replace=False)
    val_ep_mask = np.zeros(n_episodes, dtype=bool)
    val_ep_mask[val_episodes] = True

    # 根据每个样本 buffer_start_idx 推断其所属 episode
    buffer_start = dataset.indices[:, 0]
    sample_episode = np.searchsorted(episode_ends, buffer_start, side="right")
    train_idx = np.where(~val_ep_mask[sample_episode])[0].tolist()
    val_idx = np.where(val_ep_mask[sample_episode])[0].tolist()

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    print(f"[Train] episodes={n_episodes}, train_episodes={n_episodes - n_val}, "
          f"val_episodes={n_val}, train_samples={len(train_idx)}, val_samples={len(val_idx)}")

    train_dataloader = DataLoader(train_dataset, **cfg.dataloader.train)
    val_dataloader = DataLoader(val_dataset, **cfg.dataloader.val)

    model.set_normalizer(dataset.get_normalizer())

    # 设置 callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        hydra.utils.instantiate(cfg.checkpoint, dirpath=output_dir / 'checkpoints'),
        ModelAveragingCallback(decay=cfg.ema.decay, update_every_n_steps=cfg.ema.update_every_n_steps),
        SaveConfigCallback(OmegaConf.to_container(cfg, resolve=True)),
        TQDMProgressBar(refresh_rate=cfg.training.progress_bar_refresh_rate if 'progress_bar_refresh_rate' in cfg.training else 10),
        # SampleCallback(),
    ]

    logger = WandbLogger(
        save_dir=output_dir,
        **cfg.logging,
    )

    trainer = Trainer(
        **cfg.trainer,
        strategy=DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else 'auto',
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
        ckpt_path=resume_ckpt,
        weights_only = False
    )

if __name__ == "__main__":
    main()