from typing import Union, Dict, Any
import h5py
import numpy as np
from torch.utils.data import Dataset

from source.common.sampler import create_indices
from source.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from source.common.normalize_util import get_identity_normalizer_from_stat


class Dataset2D(Dataset):
    def __init__(self, 
                 dataset_path: str, 
                 horizon: int=1,
                 pad_before: int=0,
                 pad_after: int=0,
                 input_meta: Union[Dict[str, Any], None]=None,
                 seperate_action: bool=False,
                 episode_mask: Union[np.ndarray, None]=None,
                 use_mem: bool=False,
                 obs_only_n_steps: bool=False,
                 obs_n_steps: Union[int, None]=None
                 ) -> None:
        
        if use_mem:
            self.data = {}
            with h5py.File(dataset_path, 'r') as f:
                for key in f.keys():
                    self.data[key] = f[key][:]
        else:
            self.data = h5py.File(dataset_path, "r")
        # import pdb; pdb.set_trace()
        episode_ends = self.data["episode_ends"][:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)
        self.indices = create_indices(episode_ends, 
                sequence_length=horizon, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        self.horizon = horizon
        self.input_meta = input_meta
        self.separate_action = seperate_action
        self.obs_keys = list(self.input_meta["obs"].keys()) if self.input_meta is not None else []
        self.action_keys = [key for key in self.input_meta.keys() if key.startswith("action")] if self.input_meta is not None else []
        self.obs_only_n_steps = obs_only_n_steps
        self.obs_n_steps = obs_n_steps if obs_n_steps is not None else horizon
        self._normalizer = None

    def __len__(self):
        return len(self.indices)
    
    def padding(self, data: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
        """
        对数据进行填充，确保数据长度为指定的范围。
        :param data: 输入数据
        :param start_idx: 起始索引
        :param end_idx: 结束索引
        :return: 填充后的数据
        """
        total_len = data.shape[0]
        if start_idx > 0:
            data[:start_idx] = data[start_idx]
        if end_idx < total_len:
            data[end_idx:] = data[end_idx - 1]

        return data
    
    def get_normalizer(self, mode='limits', **kwargs):
        if self._normalizer is not None:
            return self._normalizer

        data = {
            'action': self.data['action'][:],
            'qpos': self.data['qpos'][:],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        for key in self.input_meta["obs"].keys():
            if key.startswith("cam"):
                normalizer[key] = SingleFieldLinearNormalizer.create_identity()

        self._normalizer = normalizer
        return self._normalizer
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        res = {'obs': {},'sample_start_idx': np.array(sample_start_idx), 'buffer_start_idx': np.array(buffer_start_idx)}
        # import pdb; pdb.set_trace()
        for key in self.obs_keys:
            if key.startswith("cam"):
                if self.obs_only_n_steps:
                    obs_target_len = self.obs_n_steps
                    obs_len = min(obs_target_len, sample_end_idx) - sample_start_idx
                    obs_len = max(1, obs_len)
                    obs_end_idx = buffer_start_idx + obs_len
                    obs = self.data[key][buffer_start_idx:obs_end_idx]
                else:
                    obs = self.data[key][buffer_start_idx:buffer_end_idx]
                obs = np.asarray(obs, dtype=np.uint8)
            elif key.startswith("qpos"):
                obs = self.data[key][buffer_start_idx:buffer_end_idx]
                obs = np.asarray(obs, dtype=np.float32)
            else:
                obs = self.data[key][buffer_start_idx:buffer_end_idx]
                obs = np.asarray(obs, dtype=np.float32)
            data_dtype = np.uint8 if key.startswith("cam") else np.float32
            if key.startswith("cam") and self.obs_only_n_steps:
                data = np.zeros((self.obs_n_steps, *obs.shape[1:]), dtype=data_dtype)
            else:
                data = np.zeros((self.horizon, *obs.shape[1:]), dtype=data_dtype)
            if key.startswith("cam") and self.obs_only_n_steps:
                obs_len = obs.shape[0]
                data[sample_start_idx:sample_start_idx + obs_len] = obs
                data = self.padding(data, sample_start_idx, sample_start_idx + obs_len)
            else:
                data[sample_start_idx:sample_end_idx] = obs
                data = self.padding(data, sample_start_idx, sample_end_idx)
            res['obs'][key] = data
        for key in self.action_keys:
            action = self.data[key][buffer_start_idx:buffer_end_idx]
            action = np.asarray(action, dtype=np.float32)
            data = np.zeros((self.horizon, *action.shape[1:]), dtype=np.float32)
            data[sample_start_idx:sample_end_idx] = action
            data = self.padding(data, sample_start_idx, sample_end_idx)
            res[key] = data

        # if not self.separate_action:
        #     agent_num = len(action_keys)
        #     agent_pos_list = []
        #     action_list = []
        #     for i in range(agent_num):
        #         key = f"agent_pos_{i}"
        #         agent_pos_list.append(res['obs'][key])
        #         del res['obs'][key]
        #         key = f"action_{i}"
        #         action_list.append(res[key])
        #         del res[key]
        #     res['obs']['agent_pos'] = np.concatenate(agent_pos_list, axis=-1)
        #     res['action'] = np.concatenate(action_list, axis=-1)

        return res


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    args = parser.parse_args()
    dataset = Dataset2D(args.dataset_path, horizon=8, 
                        input_meta={
                            "obs": {
                                "head_cam_0": [3, 256, 256],
                                "head_cam_1": [3, 256, 256],
                                "agent_pos_0": [8],
                                "agent_pos_1": [8],
                            },
                            "action_0": [8],
                            "action_1": [8],
                        },
                        seperate_action=False,
                        use_mem=False)
    norms = dataset.get_normalizer()
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        # print(data)
        break