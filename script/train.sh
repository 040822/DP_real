# Examples:
# tmux new -s train

############ move_playingcard_color ############
# bash script/train.sh dp2 move_playingcard_color_real10 3
# bash script/train.sh dp2 move_playingcard_color_real20 2
# bash script/train.sh dp2 move_playingcard_color_sim200 0 ✅
# bash script/train.sh dp2 move_playingcard_color_real10_sim200 0 ✅
# bash script/train.sh dp2 move_playingcard_color_pseudo200 0 ✅
# bash script/train.sh dp2 move_playingcard_color_real10_pseudo200 3 ✅

############ move_playingcard_others ############
# bash script/train.sh dp2 move_playingcard_others_real10 3
# bash script/train.sh dp2 move_playingcard_others_real20 2
# bash script/train.sh dp2 move_playingcard_others_sim200 1 ✅
# bash script/train.sh dp2 move_playingcard_others_real10_sim200 1 ✅
# bash script/train.sh dp2 move_playingcard_others_pseudo200 1 ✅
# bash script/train.sh dp2 move_playingcard_others_real10_pseudo200 3 ✅

############ place_mouse_pad_color ############
# bash script/train.sh dp2 place_mouse_pad_color_real10 3
# bash script/train.sh dp2 place_mouse_pad_color_real20 2
# bash script/train.sh dp2 place_mouse_pad_color_sim200 3 ✅
# bash script/train.sh dp2 place_mouse_pad_color_real10_sim200 3 ✅
# bash script/train.sh dp2 place_mouse_pad_color_pseudo200 1 
# bash script/train.sh dp2 place_mouse_pad_color_real10_pseudo200 3 

############ place_mouse_pad_others ############
# bash script/train.sh dp2 place_mouse_pad_others_real10 3
# bash script/train.sh dp2 place_mouse_pad_others_real20 2
# bash script/train.sh dp2 place_mouse_pad_others_sim200 0 ✅
# bash script/train.sh dp2 place_mouse_pad_others_real10_sim200 1 ✅
# bash script/train.sh dp2 place_mouse_pad_others_pseudo200 1 
# bash script/train.sh dp2 place_mouse_pad_others_real10_pseudo200 3 


policy_name=${1}
task_name=${2}
gpu_id=${3}

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python -u train.py --config-name=${policy_name}  task=${task_name} 2>&1 | tee Temp/${task_name}_${policy_name}.out


                                