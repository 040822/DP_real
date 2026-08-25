# 将原始遥操作采集的 episode hdf5 转换为训练用的合并 h5：
# 读取 action / qpos / 各相机图像，把 action 的左右夹爪替换为 master_action 的夹爪（抓取更牢固），
# 并把所有 episode 沿时间维拼接，输出 cam_high（以及存在的左右腕部相机）、qpos、action 与 episode_ends。
import argparse
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class RealTrajectoryDataset(Dataset):
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.data_files = list(self.dataset_dir.glob('*.hdf5'))
        self.data_files.sort()

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        return extract_hdf5_data(self.data_files[idx])


def extract_hdf5_data(hdf5_file):
    """
    /action:                                (200, 14)               float32
    /base_action:                           (200, 2)                float32
    /observations
    /observations/effort:                   (200, 14)               float32
    /observations/images
    /observations/images/cam_high:          (200, 480, 640, 3)      uint8
    /observations/images/cam_left_wrist:    (200, 480, 640, 3)      uint8
    /observations/images/cam_right_wrist:   (200, 480, 640, 3)      uint8
    /observations/qpos:                     (200, 14)               float32
    /observations/qvel:                     (200, 14)               float32
    """
    with h5py.File(hdf5_file, 'r') as f:
        action = f['action'][:]
        # 将action的夹爪替换为master_action的夹爪，从而使夹爪能够抓的更牢固。
        # 若 hdf5 不含 master_action，则回退为仅使用 action。
        if 'master_action' in f:
            master_action = f['master_action'][:]
            action[:, 7] = master_action[:, 7]
            action[:, 13] = master_action[:, 13]
        else:
            print(f'\033[33m[WARN] {hdf5_file} 不含 master_action，回退为仅使用 action（夹爪未替换）\033[0m')
        qpos = f['observations/qpos'][:]
        cam_high = f['observations/images/cam_high'][:]

        # 默认记录左右腕部相机；若某一侧不存在则跳过并给出 WARN。
        obs = {'cam_high': cam_high, 'qpos': qpos}
        if 'cam_left_wrist' in f['observations/images']:
            obs['cam_left'] = f['observations/images/cam_left_wrist'][:]
        else:
            print(f'\033[33m[WARN] {hdf5_file} 左侧腕部相机(cam_left_wrist)不存在，跳过\033[0m')
        if 'cam_right_wrist' in f['observations/images']:
            obs['cam_right'] = f['observations/images/cam_right_wrist'][:]
        else:
            print(f'\033[33m[WARN] {hdf5_file} 右侧腕部相机(cam_right_wrist)不存在，跳过\033[0m')

    return {
        'obs': obs,
        'action': action
    }

def main(dataset_dir: str, output_path: str) -> None:
    dataset = RealTrajectoryDataset(dataset_dir)
    device = 'cuda:0'

    comp_kwaegs = {'compression': 'gzip', 'compression_opts': 4}
    episode_ends = []
    end = 0
    with h5py.File(output_path, "w") as f:
        # 逐 episode 处理：相机（cam_high 及存在的左右腕部相机）、qpos、action 全部对齐到最小长度后拼接；
        # 第一个 episode 时创建带可扩展维度的数据集，后续 episode 直接 append。
        for i, data in tqdm(enumerate(dataset), desc="Loading data", total=len(dataset)):
            obs = data["obs"]
            action = data["action"]
            if data is not None:
                qpos = obs["qpos"]

                # 计算所有相机 / qpos / action 的最小长度并对齐
                cam_names = [k for k in obs.keys() if k != "qpos"]
                lengths = [obs[c].shape[0] for c in cam_names] + [qpos.shape[0], action.shape[0]]
                min_len = min(lengths)

                # 处理各相机：裁剪 + HWC->CHW + 插值到 256x256
                proc_cams = {}
                for c in cam_names:
                    cam = np.array(obs[c][:min_len]).astype(np.uint8)
                    cam = np.moveaxis(cam, -1, -3)
                    cam = torch.tensor(cam).float().to(device)
                    cam = F.interpolate(cam, size=(256, 256), mode='bilinear').cpu().numpy()
                    proc_cams[c] = cam
                qpos = np.array(qpos[:min_len]).astype(np.float32)
                action = np.array(action[:min_len]).astype(np.float32)

                end += min_len
                episode_ends.append(end)

                if i == 0:
                    for c, cam in proc_cams.items():
                        f.create_dataset(
                            c,
                            data=cam,
                            shape=cam.shape,
                            maxshape=(None, *cam.shape[1:]),
                            dtype="uint8",
                            **comp_kwaegs
                        )
                    f.create_dataset(
                        "qpos",
                        data=qpos,
                        shape=qpos.shape,
                        maxshape=(None, *qpos.shape[1:]),
                        dtype="float32",
                        **comp_kwaegs
                    )
                    f.create_dataset(
                        "action",
                        data=action,
                        shape=action.shape,
                        maxshape=(None, *action.shape[1:]),
                        dtype="float32",
                        **comp_kwaegs
                    )
                else:
                    for c, cam in proc_cams.items():
                        f[c].resize((f[c].shape[0] + cam.shape[0]), axis=0)
                        f[c][-cam.shape[0]:] = cam
                    f["qpos"].resize((f["qpos"].shape[0] + qpos.shape[0]), axis=0)
                    f["qpos"][-qpos.shape[0]:] = qpos
                    f["action"].resize((f["action"].shape[0] + action.shape[0]), axis=0)
                    f["action"][-action.shape[0]:] = action
        f.create_dataset(
            "episode_ends",
            data=np.array(episode_ends),
            **comp_kwaegs
        )
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_dir", default="/home/agilex/wenxin/DP_real/data_pre/collecting_objects", type=str, help="Path to the dataset")
    # parser.add_argument("--output_path", default="/home/wenxin/office/agilex_data/collect_block/collect_block_50.h5", type=str, help="Path to save the output")
    # args = parser.parse_args()
    
    task_name = "grabbing_rod"
    dataset_dir = f"/home/agilex/wenxin/DP_real/data_pre/{task_name}/aloha_mobile_dummy/"
    output_path = f"/home/agilex/wenxin/DP_real/data/{task_name}.h5"

    main(dataset_dir, output_path)
