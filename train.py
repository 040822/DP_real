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
from torch.utils.data import DataLoader, random_split
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
    train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*0.95), len(dataset) - int(len(dataset)*0.95)])
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