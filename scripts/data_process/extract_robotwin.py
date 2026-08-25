# 将原始遥操作采集的 episode hdf5 转换为训练用的合并 h5：
# 读取 action / qpos / cam_high，把 action 的左右夹爪替换为 master_action 的夹爪（抓取更牢固），
# 并把所有 episode 沿时间维拼接，输出 cam_high、qpos、action 与 episode_ends。
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

    return {
        'obs': {
            'cam_high': cam_high,
            # 'cam_left': cam_left,
            'qpos': qpos
        },
        'action': action
    }

def main(dataset_dir: str, output_path: str) -> None:
    dataset = RealTrajectoryDataset(dataset_dir)
    device = 'cuda:0'

    comp_kwaegs = {'compression': 'gzip', 'compression_opts': 4}
    episode_ends = []
    end = 0
    with h5py.File(output_path, "w") as f:
        for i, data in tqdm(enumerate(dataset), desc="Loading data", total=len(dataset)):
            obs = data["obs"]
            action = data["action"]
            if data is not None:
                # cam_right = obs["cam_right"]
                # cam_left = obs["cam_left"]
                cam_high = obs["cam_high"]
                qpos = obs["qpos"]
                action = action
                # import pdb;pdb.set_trace()
                # min_len = min(cam_right.shape[0], cam_left.shape[0], qpos.shape[0], action.shape[0])
                min_len = min(cam_high.shape[0], qpos.shape[0], action.shape[0])
                # cam_right = np.array(cam_right[:min_len]).astype(np.uint8)
                # cam_right = np.moveaxis(cam_right, -1, -3)
                # cam_left = np.array(cam_left[:min_len]).astype(np.uint8)
                # cam_left = np.moveaxis(cam_left, -1, -3)
                cam_high = np.array(cam_high[:min_len]).astype(np.uint8)
                cam_high = np.moveaxis(cam_high, -1, -3)
                qpos = np.array(qpos[:min_len]).astype(np.float32)
                action = np.array(action[:min_len]).astype(np.float32)

                # cam_right = torch.tensor(cam_right).float().to(device)
                # cam_left = torch.tensor(cam_left).float().to(device)
                cam_high = torch.tensor(cam_high).float().to(device)
                # cam_right = F.interpolate(cam_right, size=(256, 256), mode='bilinear').cpu().numpy()
                # cam_left = F.interpolate(cam_left, size=(256, 256), mode='bilinear').cpu().numpy()
                cam_high = F.interpolate(cam_high, size=(256, 256), mode='bilinear').cpu().numpy()

                end += min_len
                episode_ends.append(end)

                if i == 0:
                    # f.create_dataset(
                    #     f"cam_right",
                    #     data=cam_right,
                    #     shape=cam_right.shape,
                    #     maxshape=(None, *cam_right.shape[1:]),
                    #     dtype="uint8",
                    #     **comp_kwaegs
                    # )
                    # f.create_dataset(
                    #     f"cam_left",
                    #     data=cam_left,
                    #     shape=cam_left.shape,
                    #     maxshape=(None, *cam_left.shape[1:]),
                    #     dtype="uint8",
                    #     **comp_kwaegs
                    # )
                    f.create_dataset(
                        f"cam_high",
                        data=cam_high,
                        shape=cam_high.shape,
                        maxshape=(None, *cam_high.shape[1:]),
                        dtype="uint8",
                        **comp_kwaegs
                    )
                    f.create_dataset(
                        f"qpos",
                        data=qpos,
                        shape=qpos.shape,
                        maxshape=(None, *qpos.shape[1:]),
                        dtype="float32",
                        **comp_kwaegs
                    )
                    f.create_dataset(
                        f"action",
                        data=action,
                        shape=action.shape,
                        maxshape=(None, *action.shape[1:]),
                        dtype="float32",
                        **comp_kwaegs
                    )
                else:
                    # f["cam_right"].resize((f["cam_right"].shape[0] + cam_right.shape[0]), axis=0)
                    # f["cam_right"][-cam_right.shape[0]:] = cam_right
                    f["cam_high"].resize((f["cam_high"].shape[0] + cam_high.shape[0]), axis=0)
                    f["cam_high"][-cam_high.shape[0]:] = cam_high
                    # f["cam_left"].resize((f["cam_left"].shape[0] + cam_left.shape[0]), axis=0)
                    # f["cam_left"][-cam_left.shape[0]:] = cam_left
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="/home/wenxin/office/agilex_data/collect_block/aloha_mobile_dummy/", type=str, help="Path to the dataset")
    parser.add_argument("--output_path", default="/home/wenxin/office/agilex_data/collect_block/collect_block_50.h5", type=str, help="Path to save the output")
    args = parser.parse_args()

    main(args.dataset_dir, args.output_path)
