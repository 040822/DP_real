# screen python workspace.py --config-name=dp2 task=2a_collect_block
# screen python workspace.py --config-name=dp2 task=2a_grabbing_rod
# screen python workspace.py --config-name=dp2 task=2a_playing_card_delivery

# screen python workspace.py --config-name=ddp2 task=2a_collect_block
# screen python workspace.py --config-name=ddp2 task=2a_grabbing_rod
# screen python workspace.py --config-name=ddp2 task=2a_playing_card_delivery

# screen python workspace.py --config-name=edp2 task=2a_collect_block
# screen python workspace.py --config-name=edp2 task=2a_grabbing_rod
# screen python workspace.py --config-name=edp2 task=2a_playing_card_delivery



import sys
sys.path.append('./')
sys.path.insert(0, '/root/autodl-tmp/wenxin/EDP_real/')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
    

import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader, random_split
from torch.optim.swa_utils import get_ema_avg_fn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning import seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch import LightningModule


# import sys
# ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
# sys.path.append(ROOT_DIR)
from model.common.callbacks import ModelAveragingCallback, SaveConfigCallback

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str("./config"),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    output_dir = pathlib.Path(HydraConfig.get().run.dir)
    sys.stdout = open(output_dir / "debug.log", "a")
    sys.stderr = sys.stdout
    
    # set seed
    seed = cfg.seed
    seed_everything(seed)
    # configure model
    model: LightningModule = hydra.utils.instantiate(cfg.policy)
    
    if cfg.dp_name == "EDP2" or cfg.dp_name == "DDP2":
        # 处理数据集大小和分割idx
        horizon = cfg.policy.coarse_dp.horizon-1
        internal = cfg.policy.coarse_dp.internal
        sample_horizon = (horizon-1)*internal + horizon + 1
        print("[Train] sample_horizon:", sample_horizon)
        
        idx = torch.linspace(1, sample_horizon - 1, steps=horizon).long()
        idx = torch.cat([torch.zeros(1, dtype=idx.dtype), idx]) #加入0
        model.set_idx(idx) # 设置idx
        cfg.task.dataset.horizon = sample_horizon # 设置数据集的horizon
    
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*0.95), len(dataset) - int(len(dataset)*0.95)])
    train_dataloader = DataLoader(train_dataset, **cfg.dataloader.train)
    val_dataloader = DataLoader(val_dataset, **cfg.dataloader.val)

    model.set_normalizer(dataset.get_normalizer())

    # 设置 callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        hydra.utils.instantiate(cfg.checkpoint, dirpath=output_dir / 'checkpoints'),
        ModelAveragingCallback(None, get_ema_avg_fn(0.9), cfg.ema.update_after_steps),
        SaveConfigCallback(OmegaConf.to_container(cfg, resolve=True)),
        TQDMProgressBar(refresh_rate=cfg.training.progress_bar_refresh_rate if ('training' in cfg and 'progress_bar_refresh_rate' in cfg.training) else 10)
    ]

    logger = WandbLogger(
        save_dir=output_dir,
        **cfg.logging,
    )

    trainer = Trainer(
        **cfg.trainer,
        strategy="ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto",
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
    )

if __name__ == "__main__":
    main()
