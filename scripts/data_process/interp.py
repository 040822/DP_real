import os
import argparse
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np
import torch
import torch.nn.functional as F


def main(dataset_path: str, output_path: str) -> None:
    batch_size = 256
    device = 'cuda:0'
    origin_f = h5py.File(dataset_path, "r")
    data_len = origin_f['action'].shape[0]

    comp_kwaegs = {'compression': 'gzip', 'compression_opts': 4}
    with h5py.File(output_path, "w") as f:
        for i in tqdm(range(0, data_len, batch_size)):
            cam_right = origin_f["cam_right"][i:i+batch_size, :, :, 80:-80]
            cam_left = origin_f["cam_left"][i:i+batch_size, :, :, 80:-80]

            cam_right = torch.tensor(cam_right).float().to(device)
            cam_left = torch.tensor(cam_left).float().to(device)
            cam_right = F.interpolate(cam_right, size=(256, 256), mode='bilinear').cpu().numpy()
            cam_left = F.interpolate(cam_left, size=(256, 256), mode='bilinear').cpu().numpy()

            if i == 0:
                f.create_dataset(
                    f"cam_right",
                    data=cam_right,
                    shape=cam_right.shape,
                    maxshape=(None, *cam_right.shape[1:]),
                    **comp_kwaegs
                )
                f.create_dataset(
                    f"cam_left",
                    data=cam_left,
                    shape=cam_left.shape,
                    maxshape=(None, *cam_left.shape[1:]),
                    **comp_kwaegs
                )
            else:
                f["cam_right"].resize((f["cam_right"].shape[0] + cam_right.shape[0]), axis=0)
                f["cam_right"][-cam_right.shape[0]:] = cam_right
                f["cam_left"].resize((f["cam_left"].shape[0] + cam_left.shape[0]), axis=0)
                f["cam_left"][-cam_left.shape[0]:] = cam_left
                
        f.create_dataset(
            "qpos",
            data=origin_f['qpos'][:],
            **comp_kwaegs
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
            **comp_kwaegs
        )
        f.create_dataset(
            "episode_ends",
            data=origin_f['episode_ends'][:],
            **comp_kwaegs
        )
    origin_f.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output")
    args = parser.parse_args()

    main(args.dataset_path, args.output_path)
