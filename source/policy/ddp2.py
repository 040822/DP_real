import dill
import torch
import random
import pathlib
import numpy as np
from typing import Dict
from torch.optim import AdamW
from omegaconf import DictConfig
from lightning.pytorch import LightningModule
from scipy.interpolate import interp1d, CubicSpline
from scipy.linalg import block_diag
from source.policy.ddp2_fine import Fine_DP2
from source.policy.ddp2_coarse import Coarse_DP2
from source.common.pytorch_util import dict_apply
from source.model.common.lr_scheduler import get_scheduler
from source.model.common.normalizer import LinearNormalizer

class DDP2(LightningModule):
    """
    Double_DP3 是一个分层策略类，结合了粗粒度扩散策略（coarse_dp）和细粒度扩散策略（fine_dp），用于三维环境中的序列决策。
    参数:
        coarse_dp (Coarse_DP3): 粗粒度扩散策略实例。
        fine_dp (Fine_DP3): 细粒度扩散策略实例。
        debug (bool, 可选): 如果为 True，则启用调试打印。默认值为 False。
        **kwargs: 传递给基类策略的其他关键字参数。
    属性:
        coarse_dp (Coarse_DP3): 粗粒度策略。
        fine_dp (Fine_DP3): 细粒度策略。
        coarse_cache (dict 或 None): 粗粒度 DP 推理结果的缓存。
        idx (list 或 None): 用于批次分割的索引列表。
        coarse_cache_idx (int): 粗粒度缓存的当前索引。
        coarse_ratio (float): 粗粒度和细粒度损失的加权比例。
        debug (bool): 调试标志。
    方法:
        set_normalizer(coarse_normalizer, fine_normalizer=None):
            设置粗粒度和细粒度 DP 的归一化器。如果未提供 fine_normalizer，则两者均使用 coarse_normalizer。
        reset():
            清除粗粒度 DP 的缓存并重置缓存索引。
        predict_action(obs_dict, sample_idx=None):
            使用粗粒度和细粒度 DP 进行分层动作预测。
        predict_action_coarse(obs_dict):
            仅返回粗粒度 DP 的动作预测（用于测试）。
        predict_action_fine(obs_dict):
            使用粗粒度 DP 的输出作为输入，仅返回细粒度 DP 的动作预测（用于测试）。
        compute_loss(batch, only_coarse=True):
            计算粗粒度和细粒度 DP 的训练损失。如果 only_coarse 为 True，则仅计算粗粒度损失。
        coarse_compute_loss(batch):
            仅计算粗粒度 DP 的损失。
        fine_compute_loss(batch):
            仅计算细粒度 DP 的损失。
        set_idx(idx):
            设置用于批次分割的索引列表。
        split_batch(batch):
            根据 self.idx 将输入批次分割为粗粒度和细粒度批次，并生成样本/历史索引。
        load_coarse_dp_model():
            从检查点加载粗粒度 DP 模型。
        load_fine_dp_model():
            从检查点加载细粒度 DP 模型。
        load_payload(payload, exclude_keys=None, include_keys=None, model_name=None, **kwargs):
            从 payload 字典加载模型 state_dicts 和 pickles。
            Loads model state_dicts and pickles from a payload dictionary.
        load_checkpoint(path=None, tag='latest', exclude_keys=None, include_keys=None, model_name=None, **kwargs):
            Loads a checkpoint file and applies its payload to the policy.
    """
    def __init__(self, 
            optimizer_cfg: DictConfig,
            scheduler_cfg: DictConfig,
            num_epochs_coarse: int,

            coarse_dp: Coarse_DP2,
            fine_dp: Fine_DP2,
            debug: bool = False,
            **kwargs):
        super().__init__()

        self.num_epochs_coarse = num_epochs_coarse
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.coarse_dp: Coarse_DP2 = coarse_dp
        self.fine_dp: Fine_DP2 = fine_dp
            
        self.coarse_cache = None
        self.idx = None
        
        self.coarse_cache_idx = 0
        self.coarse_ratio = 0.5 # coarse和fine的loss的比例
        self.debug = debug

        self.reset()  # 初始化缓存
        
        # ["fine_dp", "linear", "cubic_spline", "minimum_snap","only_coarse"]
        self.predict_type = "cubic_spline"
        print(f"[Double_DP3 predict_action] predict_type: {self.predict_type}")
    
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    
    # ========= set  ============
    def set_normalizer(self, coarse_normalizer: LinearNormalizer, fine_normalizer: LinearNormalizer=None):
        # normalized_data = (data - mean) / std
        if fine_normalizer is None:
            # 如果没有提供fine_normalizer，则fine DP使用coarse_normalizer
            fine_normalizer = coarse_normalizer
        self.coarse_dp.set_normalizer(coarse_normalizer)
        self.fine_dp.set_normalizer(fine_normalizer)

        
    def set_idx(self, idx):
        # 用于传入self.idx
        self.idx = idx

    # ========= inference  ============
    def reset(self):
        """
        清除缓存,每次跑完env后都需要清除缓存。
        """
        if self.coarse_cache is not None:
            del self.coarse_cache
        self.coarse_cache = None
        self.coarse_cache_idx = 0
        self.coarse_finish_idx = 0 # 记录完成的coarse+fine的总轮数
        self.init_action = None # 记录初始动作
        
        # 分别调用coarse_dp和fine_dp的reset方法，确保它们内部的状态也被重置。
        self.coarse_dp.reset()
        self.fine_dp.reset()
        
    def reset_action(self):
        """
        在完成一定轮数的coarse+fine推理后，执行一次reset_action，返回初始动作并重置缓存。
        这个设计是为了在连续推理过程中引入一些“停顿”，让机械臂有机会回到一个已知的状态，从而提高稳定性和安全性。
        """
        init_action = self.init_action.unsqueeze(1) # [B, 1, Da]
        init_action = init_action.repeat(1, self.fine_dp.horizon, 1) # [B, T, Da]
        return {'action': init_action, 'action_pred': init_action}
        
    
    def linear_interpolation(self, coarse_actions):
        coarse_actions_np = coarse_actions.detach().cpu().numpy() # B, T1, Da
        B, T1, Da = coarse_actions.shape
        T2 = self.fine_dp.horizon - 3 + 1
        # scipy 的 interp1d 要求 x 为 1D；这里为每个 batch 复用同一组时间节点
        time_knots = np.broadcast_to(np.linspace(0, T1 - 1, T1), (B, T1))
        total_steps = (T1 - 1) * T2 + 1
        eval_times = np.linspace(0, time_knots[0, -1], total_steps)
        
        f = interp1d(time_knots[0], coarse_actions_np, axis=1, kind='linear')
        traj = f(eval_times)
        
        action_pred = torch.as_tensor(traj, device=self.device, dtype=self.dtype) # B, T2*(T1-1)+1, Da
        
        # 切分区间
        start = self.coarse_cache_idx * T2
        end = (self.coarse_cache_idx + 1) * T2
        action_pred = action_pred[:,start:end]
        
        action = action_pred

        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result
    
    def cubic_spline_interpolation(self, coarse_actions):
        coarse_actions_np = coarse_actions.detach().cpu().numpy() # B, T1, Da
        B, T1, Da = coarse_actions.shape
        T2 = self.fine_dp.horizon - 3 + 1
        # scipy 的 CubicSpline 要求 x 为 1D；这里为每个 batch 复用同一组时间节点
        time_knots = np.broadcast_to(np.linspace(0, T1 - 1, T1), (B, T1))
        total_steps = (T1 - 1) * T2 + 1
        eval_times = np.linspace(0, time_knots[0, -1], total_steps)
        
        f = CubicSpline(time_knots[0], coarse_actions_np, axis=1, bc_type='natural')
        traj = f(eval_times)
        
        action_pred = torch.as_tensor(traj, device=self.device, dtype=self.dtype) # B, T2*(T1-1)+1, Da
        
        # 切分区间
        start = self.coarse_cache_idx * T2
        end = (self.coarse_cache_idx + 1) * T2
        action_pred = action_pred[:,start:end]
        
        action = action_pred

        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result

    def minimal_snap(self, coarse_actions):
        """
        真正意义上的全局 Minimal Snap (Global Minimal Snap - Pass Through)
        
        区别于之前的 Stop-and-Go，这个版本会求解一个全局线性方程组，
        确保机械臂在中间关键点处速度、加速度、加加速度(Jerk)连续，
        从而实现流畅的“滑过”效果。
        """
        # 1. 数据准备
        waypoints = coarse_actions.detach().cpu().numpy()  # [B, T1, Da]
        B, T1, Da = waypoints.shape
        n_segments = T1 - 1
        
        # 获取插值步长 (沿用你的逻辑)
        T2 = self.fine_dp.horizon - 3 + 1
        
        # 归一化时间设定：假设每一段的时间长度 dt = 1.0
        # 如果需要物理时间，可以根据实际距离调整 dt
        dt = 1.0 
        
        # 2. 构建线性方程组 Ax = b
        # 我们使用 7 阶多项式 (Octic polynomial): p(t) = c0 + c1*t + ... + c7*t^7
        # 每个段有 8 个系数，总共有 n_segments * 8 个未知数 (对于每个维度)
        poly_order = 7
        n_coeffs = poly_order + 1 # 8
        total_vars = n_segments * n_coeffs
        
        # A 矩阵 (约束矩阵) 对所有 batch 和维度是通用的
        # b 向量 (目标值) 对每个 batch 和维度不同
        A = np.zeros((total_vars, total_vars))
        b = np.zeros((B, total_vars, Da))
        
        row_idx = 0
        
        # --- (A) 头部和尾部约束 (Endpoint Constraints) ---
        # 起点 (Segment 0 at t=0): 位置=p0, 速度=0, 加速度=0, Jerk=0
        # 终点 (Segment N-1 at t=dt): 位置=pN, 速度=0, 加速度=0, Jerk=0
        
        # 导数阶数: 0(Pos), 1(Vel), 2(Acc), 3(Jerk)
        for k in range(4): 
            # Start of first segment (t=0)
            # t=0 时，只有 c_k * k! 项非零
            # d^k/dt^k (c_k * t^k) = k! * c_k
            A[row_idx, k] = np.math.factorial(k)
            b[:, row_idx, :] = waypoints[:, 0, :] if k == 0 else 0
            row_idx += 1
            
        for k in range(4):
            # End of last segment (t=dt)
            # 需要计算所有系数对 k 阶导数的贡献
            # coeff idx j from k to 7
            seg_idx_last = n_segments - 1
            col_base = seg_idx_last * n_coeffs
            for j in range(k, n_coeffs):
                # derivative coeff: j*(j-1)*...*(j-k+1)
                d_coef = 1
                for m in range(k): d_coef *= (j - m)
                A[row_idx, col_base + j] = d_coef * (dt ** (j - k))
            b[:, row_idx, :] = waypoints[:, -1, :] if k == 0 else 0
            row_idx += 1

        # --- (B) 中间点位置约束 (Waypoint Constraints) ---
        # 每一段的末尾必须到达下一个 waypoint
        # Segment i at t=dt == Waypoint[i+1]
        for i in range(n_segments - 1): # 不包含最后一段（已经在上面处理了）
            col_base = i * n_coeffs
            # Position (k=0)
            for j in range(n_coeffs):
                A[row_idx, col_base + j] = dt ** j
            b[:, row_idx, :] = waypoints[:, i+1, :]
            row_idx += 1
            
            # 每一段的开头必须从当前 waypoint 开始
            # Segment i+1 at t=0 == Waypoint[i+1]
            col_base_next = (i + 1) * n_coeffs
            A[row_idx, col_base_next] = 1.0 # t=0, only c0 remains
            b[:, row_idx, :] = waypoints[:, i+1, :]
            row_idx += 1

        # --- (C) 中间点连续性约束 (Continuity Constraints) ---
        # Segment i at t=dt 的导数 == Segment i+1 at t=0 的导数
        # 保证 Vel, Acc, Jerk, Snap, Crackle, Pop (最高到 C6 连续)
        # Minimal Snap 通常保证到 C4 或 C6
        for i in range(n_segments - 1):
            col_base_curr = i * n_coeffs
            col_base_next = (i + 1) * n_coeffs
            
            for k in range(1, 7): # 1st to 6th derivative continuity
                # Left side: Segment i at t=dt
                for j in range(k, n_coeffs):
                    d_coef = 1
                    for m in range(k): d_coef *= (j - m)
                    A[row_idx, col_base_curr + j] = d_coef * (dt ** (j - k))
                
                # Right side: Segment i+1 at t=0
                # only k-th coeff contributes: - k! * c_k (moved to left side of eq)
                fact_k = np.math.factorial(k)
                A[row_idx, col_base_next + k] = -fact_k
                
                # b remains 0 (difference is 0)
                row_idx += 1

        # 3. 求解方程组
        # A [total_vars, total_vars] * X [total_vars, Da] = B [total_vars, Da]
        # X 包含了所有段的所有多项式系数
        coeffs_all = []
        for bi in range(B):
            try:
                coeffs_b = np.linalg.solve(A, b[bi]) # [total_vars, Da]
            except np.linalg.LinAlgError:
                # 增加一点点正则化防止奇异 (Fallback)
                coeffs_b = np.linalg.solve(A + np.eye(A.shape[0])*1e-6, b[bi])
            coeffs_all.append(coeffs_b)
        coeffs_all = np.stack(coeffs_all, axis=0) # [B, total_vars, Da]

        # 4. 生成轨迹点
        # 预先计算时间向量 powers: t^0, t^1, ..., t^7
        t_eval = np.linspace(0, dt, T2, endpoint=False) # [T2]
        t_powers = np.zeros((T2, n_coeffs)) # [T2, 8]
        for p in range(n_coeffs):
            t_powers[:, p] = t_eval ** p

        traj_list = []
        for i in range(n_segments):
            # 取出当前段的系数
            # coeffs_seg: [B, 8, Da]
            coeffs_seg = coeffs_all[:, i*n_coeffs : (i+1)*n_coeffs, :]
            
            # 计算位置: P = T * C
            # [T2, 8] x [B, 8, Da] -> [B, T2, Da]
            seg_pos = np.einsum('tp,bpd->btd', t_powers, coeffs_seg)
            traj_list.append(seg_pos)
            
        # 拼接并加上最后一个点
        traj = np.concatenate(traj_list, axis=1) # [B, (T1-1)*T2, Da]
        traj = np.concatenate([traj, waypoints[:, -1:, :]], axis=1) # +终点

        # 5. 封装返回
        action_pred = torch.as_tensor(traj, device=self.device, dtype=self.dtype)
        
        # 切分区间 (沿用你的逻辑)
        start = self.coarse_cache_idx * T2
        end = (self.coarse_cache_idx + 1) * T2
        
        # 边界保护
        if start >= action_pred.shape[1]:
            action_out = action_pred[:, -1:] # Fallback
        else:
            action_out = action_pred[:, start:end]

        result = {
            'action': action_out,
            'action_pred': action_out, # 注意：这里如果显存够，建议返回 full action_pred 用于调试
        }
        return result

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], sample_idx=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        sample_idx: optional, if provided, will use this index to select the coarse DP action.
        result: must include "action" key
        """

        # 获取维度信息
        value = next(iter(obs_dict.values()))
        B, To = value.shape[:2] # B：批次大小
        T = self.fine_dp.horizon
        idx_len = self.coarse_dp.horizon
        Da = self.fine_dp.action_dim # Dim of action

        # Coarse DP推理
        if self.coarse_cache is None:
            result_coarse = self.coarse_dp.predict_action(obs_dict, pre_action=None, history_idxs=self.coarse_cache_idx)
        else:
            result_coarse = self.coarse_dp.predict_action(obs_dict, pre_action=self.coarse_cache['action_pred'], history_idxs=self.coarse_cache_idx)
        self.coarse_cache = result_coarse
            
        # Fine DP推理
        # build input
        device = self.device
        dtype = self.dtype
        pre_action = torch.zeros(size=(B, T, Da), device=device, dtype=dtype) # 产生fine DP的输入动作。

        pre_action[:, 0, :]  = self.coarse_cache['action'][:, self.coarse_cache_idx , :]
        pre_action[:, -1, :] = self.coarse_cache['action'][:, self.coarse_cache_idx + 1, :]
        act_position = torch.tensor(self.coarse_cache_idx, device=device)

        if self.predict_type == "fine_dp":
            action = self.fine_dp.predict_action(
                obs_dict=obs_dict, 
                pre_action=pre_action, 
                act_position=act_position,)
            
            # action = {'action': tensor([1,15,8]), 'action_pred': tensor([1,17,8])} }
            
        elif self.predict_type == "linear":
            actions_coarse = self.coarse_cache['action']  # [B, Hc, Da]
            action = self.linear_interpolation(actions_coarse)
            
        elif self.predict_type == "cubic_spline":
            actions_coarse = self.coarse_cache['action']  # [B, Hc, Da]
            action = self.cubic_spline_interpolation(actions_coarse)
                    
        elif self.predict_type == "minimum_snap":
            actions_coarse = self.coarse_cache['action']  # [B, Hc, Da]
            action = self.minimal_snap(actions_coarse)
        elif self.predict_type == "only_coarse":
            action = self.coarse_cache

        self.coarse_cache_idx += 1
        if self.coarse_cache_idx >= idx_len - 1:
            self.coarse_cache = None # 不清空也会造成性能下降
            self.coarse_cache_idx = 0
            self.coarse_finish_idx += 1
            
        # if self.coarse_cache_idx == 1 and self.coarse_finish_idx == 0:
        #     self.init_action = action['action'][:, 0, :].detach() # 记录初始动作
            
        # if self.coarse_finish_idx % 4 == 0 and self.coarse_cache_idx == 0:
        #     return self.reset_action()
            
        return action

    # ========= training  ============
    def compute_loss(self, batch: Dict[str, torch.Tensor], sample_num = 1, use_all_samples=False) -> Dict[str, torch.Tensor]:
        '''
        计算下一段的损失，对应单步推理和teacher forcing的训练方式。
        '''

        B = batch['action'].shape[0] # B：批次大小
        T = self.fine_dp.horizon
        Da = self.fine_dp.action_dim
        device = self.device
        dtype = self.dtype
        idx_len = len(self.idx) #也等于coarse DP的horizon长度。
        
        if use_all_samples:
            max_start_idx = 0
            sample_num = idx_len - 1 # 结尾无法作为起点，所以是 idx_len-1 段。
        else:
            # 生成 history_idx, 对应已执行的动作段。
            max_start_idx = (idx_len - 1) - sample_num # 最后 sample_num 段无法作为起点，因为每段都需要向后看 sample_num 段。同时注意idx从0开始
            if max_start_idx < 0 or max_start_idx >= idx_len - 1:
                raise ValueError(f"Expected sample_num to be between 1 and {idx_len - 2}, but got {sample_num}. Please adjust sample_num or check the length of idx.")
        
        history_idxs = [random.randint(0, max_start_idx) for _ in range(B)]

        loss_sum = 0.0
        coarse_loss_dict = None
        fine_loss_dict = None
        coarse_cache = None
        
        for step in range(sample_num):
            if step > 0:
                # 每一步向后推进一个history位置。
                history_idxs = [idx + 1 for idx in history_idxs]
                
            obs_dict, traj_coarse, traj_fine = self.split_batch(batch, history_idxs)
            
            if coarse_cache is not None:
                # 如果不是第一步，使用上一步的coarse DP输出作为输入。
                coarse_loss, coarse_loss_dict = self.coarse_dp.compute_loss(obs_dict=obs_dict, trajectory=traj_coarse, history_idxs=history_idxs, pre_action=coarse_cache)  
            else:
                coarse_loss, coarse_loss_dict = self.coarse_dp.compute_loss(obs_dict=obs_dict, trajectory=traj_coarse, history_idxs=history_idxs)
            # 截断跨 step 的计算图，避免 sample_num 内形成长链式反向传播。
            coarse_cache = coarse_loss_dict['action'].detach() # 注意：传入的是unnormalize的action，与推理时一致。
            # 根据coarse DP的输出，产生fine DP的输入动作。
            pre_action = torch.zeros(size=(B, T, Da), device=device, dtype=dtype) 
            # 按样本取当前段和下一段的 coarse 锚点，避免高级索引导致维度错位。
            batch_indices = torch.arange(B, device=device)
            history_idxs_tensor = torch.tensor(history_idxs, device=device, dtype=torch.long)
            
            pre_action[:, 0, :] = coarse_loss_dict['action'][batch_indices, history_idxs_tensor, :]
            pre_action[:, -1, :] = coarse_loss_dict['action'][batch_indices, history_idxs_tensor+1, :]
                
            fine_loss, fine_loss_dict = self.fine_dp.compute_loss(obs_dict=obs_dict, trajectory=traj_fine, pre_action=pre_action, act_position=history_idxs)
            
            step_loss = self.coarse_ratio * coarse_loss + (1 - self.coarse_ratio) * fine_loss
            loss_sum = loss_sum + step_loss

        loss = loss_sum / sample_num
        # TODO：可以对coarse_loss_dict和fine_loss_dict内指标也进行sum。不过非必须，只影响log。

        return loss, coarse_loss_dict, fine_loss_dict
    
    # ========= data process  ============
    
    def split_batch(self, batch, history_idxs):
        B = batch['action'].shape[0]
        device = self.device
        dtype = self.dtype
        
        # 分割batch
        nobs = batch['obs'] # B,T,Do
        nactions = batch['action'] # B,T,Da
        batch_indices = torch.arange(B, device=device)
        
        begin_idx = [self.idx[i] for i in history_idxs]
        end_idx = [self.idx[i+1] for i in history_idxs]
        
        begin_idx_tensor = torch.tensor(begin_idx, device=device, dtype=torch.long)
        
        # 生成obs_dict。注意：推理的时候coarse和fine共用一个obs_dict，所以说训练的时候也要对齐。
        obs_dict = {}
        for k, v in nobs.items():
            obs_dict[k] = v[batch_indices, begin_idx_tensor, :].unsqueeze(1)  # [B, 1, Dk]

        traj_coarse = nactions[:, self.idx]  # [B, idx_len, Da]
        traj_fine = torch.stack([nactions[i, begin_idx[i]:end_idx[i]+1] for i in range(B)])  # [B, T+2, Da]

        return obs_dict, traj_coarse, traj_fine


    # ========= trainer  ============
    def training_step(self, batch, batch_idx):
        raw_loss, coarse_loss_dict, fine_loss_dict = self.compute_loss(batch, sample_num=2, use_all_samples=False)
        self.log('train/loss', raw_loss, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log('train/mse_loss', coarse_loss_dict['mse_loss'], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        if fine_loss_dict is not None:
            self.log('train/fine_loss', fine_loss_dict['loss'], prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return raw_loss
    
    def validation_step(self, batch, batch_idx):
        raw_loss, coarse_loss_dict, fine_loss_dict = self.compute_loss(batch, sample_num=2, use_all_samples=False)
        self.log('val/loss', raw_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/coarse_mse', coarse_loss_dict['mse_loss'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        if fine_loss_dict is not None:
            self.log('val/fine_mse', fine_loss_dict['mse_loss'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return raw_loss

    def configure_optimizers(self):
        optim_groups = self.coarse_dp.model.get_optim_groups(
            weight_decay=self.optimizer_cfg.transformer_weight_decay)
        optim_groups.append({
            "params": self.coarse_dp.obs_encoder.parameters(),
            "weight_decay": self.optimizer_cfg.obs_encoder_weight_decay
        })
        
        optim_groups = optim_groups + self.fine_dp.model.get_optim_groups(
            weight_decay=self.optimizer_cfg.transformer_weight_decay)
        optim_groups.append({
            "params": self.fine_dp.obs_encoder.parameters(),
            "weight_decay": self.optimizer_cfg.obs_encoder_weight_decay
        })

        optimizer = AdamW(
            optim_groups,
            lr=self.optimizer_cfg.lr, 
            betas=self.optimizer_cfg.betas
        )
        lr_scheduler = get_scheduler(
            self.scheduler_cfg.scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.scheduler_cfg.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }