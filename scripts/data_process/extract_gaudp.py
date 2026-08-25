from omegaconf import OmegaConf
import argparse
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np
import torch
import torch.nn.functional as F

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if __name__ == "__main__":
    import sys

    ROOT_DIR = str(Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
from model.noposplat.encoder import get_encoder


def main(dataset_path: str, config_path: str, output_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    device = 'cuda:0'
    origin_f = h5py.File(dataset_path, "r")
    data_len = origin_f['action'].shape[0]

    gaussian_encoder = get_encoder(cfg.encoder)
    if cfg.encoder.pretrained_weights:
        weight_path = cfg.encoder.pretrained_weights
        ckpt_weights = torch.load(weight_path, map_location="cpu", weights_only=True)
        ckpt_weights = ckpt_weights["state_dict"]
        ckpt_weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith("encoder.")}
        missing_keys, unexpected_keys = gaussian_encoder.load_state_dict(ckpt_weights)
        print("successfully loaded encoder weights")
    else:
        raise ValueError(f"Invalid checkpoint format: {weight_path}")
    
    gaussian_encoder.to(device)
    gaussian_encoder.eval()

    comp_kwaegs = {'compression': 'gzip', 'compression_opts': 4}
    with h5py.File(output_path, "w") as f:
        for i in tqdm(range(0, data_len, cfg.batch_size)):
            cam_right = origin_f["cam_right"][i:i+cfg.batch_size]
            cam_left = origin_f["cam_left"][i:i+cfg.batch_size]
            
            views = np.stack([cam_right, cam_left], axis=1)
            views = views / 255.0
            views = (views * 2) - 1
            with torch.no_grad():
                gaussians = gaussian_encoder({'image': torch.tensor(views, device=device, dtype=torch.float32)})
                gaussians = gaussians.to('cpu').numpy()
            gaussian_right = gaussians[:, 0]
            gaussian_left = gaussians[:, 1]

            if i == 0:
                f.create_dataset(
                    f"cam_right",
                    data=cam_right,
                    shape=cam_right.shape,
                    maxshape=(None, *cam_right.shape[1:]),
                    # **comp_kwaegs
                )
                f.create_dataset(
                    f"cam_left",
                    data=cam_left,
                    shape=cam_left.shape,
                    maxshape=(None, *cam_left.shape[1:]),
                    # **comp_kwaegs
                )

                f.create_dataset(
                    f"gaussian_right",
                    data=gaussian_right,
                    shape=gaussian_right.shape,
                    maxshape=(None, *gaussian_right.shape[1:]),
                    # **comp_kwaegs
                )
                f.create_dataset(
                    f"gaussian_left",
                    data=gaussian_left,
                    shape=gaussian_left.shape,
                    maxshape=(None, *gaussian_left.shape[1:]),
                    # **comp_kwaegs
                )
            else:
                f["cam_right"].resize((f["cam_right"].shape[0] + cam_right.shape[0]), axis=0)
                f["cam_right"][-cam_right.shape[0]:] = cam_right
                f["cam_left"].resize((f["cam_left"].shape[0] + cam_left.shape[0]), axis=0)
                f["cam_left"][-cam_left.shape[0]:] = cam_left
                f["gaussian_right"].resize((f["gaussian_right"].shape[0] + gaussian_right.shape[0]), axis=0)
                f["gaussian_right"][-gaussian_right.shape[0]:] = gaussian_right
                f["gaussian_left"].resize((f["gaussian_left"].shape[0] + gaussian_left.shape[0]), axis=0)
                f["gaussian_left"][-gaussian_left.shape[0]:] = gaussian_left
                
        f.create_dataset(
            "qpos",
            data=origin_f['qpos'][:],
            # **comp_kwaegs
        )
        action = origin_f['action'][:]
        # 将action的夹爪替换为master_action的夹爪，从而使夹爪能够抓的更牢固。
        # 若 hdf5 不含 master_action，则回退为仅使用 action。
        if 'master_action' in origin_f:
            master_action = origin_f['master_action'][:]
            action[:, 7] = master_action[:, 7]
            action[:, 13] = master_action[:, 13]
        else:
            print('\033[33m[WARN] 源文件不含 master_action，回退为仅使用 action（夹爪未替换）\033[0m')
        f.create_dataset(
            "action",
            data=action,
            # **comp_kwaegs
        )
        f.create_dataset(
            "episode_ends",
            data=origin_f['episode_ends'][:],
            # **comp_kwaegs

            
        )
    origin_f.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="/home/wenxin/office/EDP_real/data/playing_card_delivery.h5", type=str, help="Path to the dataset")
    parser.add_argument("--config_path", default="/home/wenxin/office/EDP_real/script/cobot_magic_gau/config/main.yaml", type=str, help="Path to the config")
    parser.add_argument("--output_path", default="/home/wenxin/office/EDP_real/data/playing_card_delivery_gau.h5", type=str, help="Path to save the output")
    args = parser.parse_args()

    main(args.dataset_path, args.config_path, args.output_path)
