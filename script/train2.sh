# Examples:
# tmux new -s train

############ move_playingcard_color ############
# bash script/train2.sh dp2 move_playingcard_color_real10 3 ✅

############ move_playingcard_others ############
# bash script/train2.sh dp2 move_playingcard_others_real10 3 ✅

############ place_mouse_pad_color ############
# bash script/train2.sh dp2 place_mouse_pad_color_real10 3 ✅

############ place_mouse_pad_others ############
# bash script/train2.sh dp2 place_mouse_pad_others_real10 3 ✅

ckpt=/home/majiahua/data/DP_real/_outputs/DP2/place_mouse_pad_others_sim200/2026.03.04.03.57.14/checkpoints/last.ckpt

policy_name=${1}
task_name=${2}
gpu_id=${3}

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python -u train.py --config-name=${policy_name}  task=${task_name} +two_train_ckpt=${ckpt} 2>&1 | tee Temp/${task_name}_${policy_name}.out


                                