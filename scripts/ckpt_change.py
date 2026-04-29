# 脚本功能：修改模型检查点文件中的超参数，并保存为新文件
import torch
ckpt_path = '/home/agilex/wenxin/EDP_real/_outputs/EDP2/2a_playing_card_delivery/2025.11.12.04.02.23/checkpoints/last1.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
print('keys:', ckpt.keys())  # 查看里面有哪些字段，例如 'hyper_parameters' / 'hparams' / 'state_dict'
# 修改超参数（根据实际字段名调整）

ckpt['cfg']['dp_name'] = "EDP2"
ckpt['cfg']['policy']['_target_'] = 'policy.edp2.DDP2'
ckpt['cfg']['policy']['coarse_dp']['_target_'] = 'policy.edp2_coarse.Coarse_DP2'
ckpt['cfg']['policy']['fine_dp']['_target_'] = 'policy.edp2_fine.Fine_DP2'





# 保存为新文件
torch.save(ckpt, '/home/agilex/wenxin/EDP_real/_outputs/EDP2/2a_playing_card_delivery/2025.11.12.04.02.23/checkpoints/last.ckpt')