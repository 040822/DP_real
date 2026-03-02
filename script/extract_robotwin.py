import argparse
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import re

class Dataset:
    def __init__(self, dataset_dir, real_num, sim_num, pseudo_num, pseudo_except_list):
        self.dataset_dir = Path(dataset_dir)
        self.real_dir = self.dataset_dir / "real"
        self.sim_dir = self.dataset_dir / "sim"
        self.pseudo_dir = self.dataset_dir / "pseudo"
        
        self.real_list = sorted(self.real_dir.glob("*.hdf5"), key=lambda x: int(re.search(r'\d+', x.name).group()))
        self.sim_list = sorted(list(self.sim_dir.glob("*.hdf5")), key=lambda x: int(re.search(r'\d+', x.name).group()))
        self.pseudo_list = sorted(list(self.pseudo_dir.glob("*.mp4")), key=lambda x: int(re.search(r'\d+', x.name).group()))
        
        self.real_num = real_num
        self.sim_num = sim_num
        self.pseudo_num = pseudo_num
        self.pseudo_except_list = pseudo_except_list
        self.pseudo_except_preprocess()
        
        assert (pseudo_num == 0) or (sim_num == 0), "目前伪数据和仿真数据是一一对应的，所以 pseudo_num 和 sim_num 只能有一个大于0"
        
    def pseudo_except_preprocess(self):
        # 从伪数据列表中移除被排除的伪数据
        self.pseudo_list = [p for i, p in enumerate(self.pseudo_list) if i not in self.pseudo_except_list]
        
        # 注意：因为伪数据和仿真数据是一一对应的，所以我们也需要从仿真数据列表中移除对应的仿真数据
        self.sim_list = [s for i, s in enumerate(self.sim_list) if i not in self.pseudo_except_list]
        
        self.pseudo_num -= len(self.pseudo_except_list)
        
        expect_len = self.real_num + self.sim_num + self.pseudo_num
        actual_len = 0
        actual_len += len(self.real_list) if self.real_num > 0 else 0
        actual_len += len(self.sim_list) if self.sim_num > 0 else 0
        actual_len += len(self.pseudo_list) if self.pseudo_num > 0 else 0
        
        assert expect_len <= actual_len, f"Expected length {expect_len}, but got actual length {actual_len}"

    
    def get_episode_ends(self):
        # 获得episode_ends，以及total_steps
        total_steps = 0
        episode_ends = []
        
        for i in range(self.real_num):
            path = self.real_list[i]
            with h5py.File(path, 'r') as f:
                # 假设你的动作数据或状态数据在第一维度是时间步 T
                # 我们随便取一个 key 来计算这个 episode 的长度
                ep_length = f['action'].shape[0] 
                total_steps += ep_length
                episode_ends.append(total_steps)
        
        for i in range(self.sim_num):
            path = self.sim_list[i]
            with h5py.File(path, 'r') as f:
                ep_length = f['joint_action/left_arm'].shape[0] 
                total_steps += ep_length
                episode_ends.append(total_steps)
                
        for i in range(self.pseudo_num):
            # 做了pseudo_except_preprocess后不需要排除
            # if i in self.pseudo_except_list:
            #     continue
            path = self.sim_list[i]
            with h5py.File(path, 'r') as f:
                ep_length = f['joint_action/left_arm'].shape[0] 
                total_steps += ep_length
                episode_ends.append(total_steps)
                
        return total_steps, episode_ends

    def __len__(self):
        expect_len = self.real_num + self.sim_num + self.pseudo_num
        return expect_len
    
    def __getitem__(self, idx):
        if idx < self.real_num:
            hdf5_file = self.real_list[idx]
            data = extract_hdf5_data_real(hdf5_file)
        elif idx < self.real_num + self.sim_num:
            hdf5_file = self.sim_list[idx - self.real_num]
            data = extract_hdf5_data_sim(hdf5_file)
        else:
            pseudo_idx = idx - self.real_num - self.sim_num
            # 做了pseudo_except_preprocess后不需要排除
            # if pseudo_idx in self.pseudo_except_list:
            #     return None
            hdf5_file = self.sim_list[pseudo_idx]
            pseudo_video = self.pseudo_list[pseudo_idx]
            data = extract_hdf5_data_pseudo(hdf5_file, pseudo_video)
        return data

def decode_jpeg_frame(frame_bytes):
    # 去掉 fixed-length 字符串末尾的 \0 padding
    if isinstance(frame_bytes, (bytes, np.bytes_)):
        frame_bytes = bytes(frame_bytes).rstrip(b"\0")
    else:
        raise TypeError(f"Unsupported frame type: {type(frame_bytes)}")

    img = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode jpeg frame.")
    # OpenCV是BGR，训练一般用RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img  # HWC, uint8

def cam_preprocess(cam, min_len):
    cam = np.array(cam[:min_len]).astype(np.uint8)
    cam = np.moveaxis(cam, -1, -3)
    cam = torch.tensor(cam).float()
    cam = F.interpolate(cam, size=(256, 256), mode='bilinear').cpu().numpy()
    return cam

def extract_hdf5_data_real(hdf5_file):
    # 加载真机数据的HDF5文件，并提取所需的数据。
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
        # action = f['action'][:]
        # qpos = f['observations/qpos'][:]
        # cam_right = f['observations/images/cam_right_wrist'][:]
        # cam_left = f['observations/images/cam_left_wrist'][:]
        # <KeysViewHDF5 ['action', 'base_action', 'observations']>
        # action = f['action'][:]
        # qpos = f['qpos'][:]
        # cam_right = f['cam_right'][:, :, 80:-80]
        # cam_left = f['cam_left'][:, :, 80:-80]
        action = f['action'][:]
        qpos = f['observations/qpos'][:]
        cam_high = f['observations/images/cam_high'][:]
        
        min_len = min(cam_high.shape[0], qpos.shape[0], action.shape[0])
        cam_high = cam_preprocess(cam_high, min_len)

    return {
        'cam_high': cam_high,
        'qpos': qpos,
        'action': action
    }
    
def extract_hdf5_data_sim(hdf5_file):
    # 加载仿真数据的HDF5文件，并提取所需的数据。
    with h5py.File(hdf5_file, 'r') as src:
        # action
        left_arm = src["joint_action/left_arm"][()]
        left_gripper = src["joint_action/left_gripper"][()]
        right_arm = src["joint_action/right_arm"][()]
        right_gripper = src["joint_action/right_gripper"][()]
        left_gripper = left_gripper[:, None] if left_gripper.ndim == 1 else left_gripper
        right_gripper = right_gripper[:, None] if right_gripper.ndim == 1 else right_gripper
        
        # obs
        head_camera_rgb = src["observation/head_camera/rgb"][()]
        head_camera_rgb = np.stack(
            [decode_jpeg_frame(frame_bytes) for frame_bytes in head_camera_rgb],
            axis=0,
        )

        action = np.concatenate(
            (
                left_arm,
                left_gripper,
                right_arm,
                right_gripper,
            ),
            axis=1,
        )
        
    return {
        'cam_high': head_camera_rgb,
        'qpos': action,
        'action': action
    }
    
def extract_hdf5_data_pseudo(sim_file, pseudo_video):
    # 首先加载仿真数据的HDF5文件，然后将仿真的cam_high换成pseudo_video
    sim_data = extract_hdf5_data_sim(sim_file)
    # TODO
    return sim_data

def process(dataset, output_path: str):
    comp_kwaegs = {'compression': 'gzip', 'compression_opts': 4}
    
    total_steps, episode_ends = dataset.get_episode_ends()

    
    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "episode_ends",
            data=np.array(episode_ends),
            **comp_kwaegs
        )
        datasets = {}
        
        # 预处理数据集
        pre_data = dataset[0]
        for key in pre_data.keys():
            shape = pre_data[key].shape[1:]
            dtype = pre_data[key].dtype
            datasets[key] = f.create_dataset(
                key,
                shape=(total_steps, *shape),
                dtype=dtype,
                chunks = True,
                **comp_kwaegs
            )
        
        # 数据写入
        current_idx = 0
        for i, data in tqdm(enumerate(dataset), desc="Loading data", total=len(dataset)):
            episode_len = data["action"].shape[0]
            end_idx = current_idx + episode_len
            for key in data.keys():
                datasets[key][current_idx:end_idx,:] = data[key][:]
            current_idx = end_idx

def main():
    task_list = ["move_playingcard_color", "move_playingcard_others", "place_mouse_pad_color", "place_mouse_pad_others", "handover_others"]
    task_name = task_list[0]
    
    real_num = 10
    sim_num = 200
    pseudo_num = 0
    pseudo_except_list = []
    output_name = ""
    
    if real_num !=0:
        output_name += f"_real{real_num}"
    if sim_num !=0:
        output_name += f"_sim{sim_num}"
    if pseudo_num !=0:
        output_name += f"_pseudo{pseudo_num}"
    
    dataset_dir = "/home/majiahua/data/DP_real/data_pre/" + task_name
    output_path = "/home/majiahua/data/DP_real/data/" + task_name + f"{output_name}.h5"
    
    dataset = Dataset(
        dataset_dir=dataset_dir,
        real_num=real_num,
        sim_num=sim_num,
        pseudo_num=pseudo_num,
        pseudo_except_list=pseudo_except_list
    )
    
    process(dataset, output_path)
    
if __name__ == "__main__":
    main()