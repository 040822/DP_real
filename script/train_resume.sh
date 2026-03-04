# Examples:
# tmux new -s train

############ move_playingcard_color ############
# bash script/train_resume.sh dp2 move_playingcard_color_real10 3
# bash script/train_resume.sh dp2 move_playingcard_color_real20 3
# bash script/train_resume.sh dp2 move_playingcard_color_sim200 1 ✅⭕
# bash script/train_resume.sh dp2 move_playingcard_color_real10_sim200 0 ✅⭕
# bash script/train_resume.sh dp2 move_playingcard_color_pseudo200 0 ✅
# bash script/train_resume.sh dp2 move_playingcard_color_real10_pseudo200 1 ✅

############ move_playingcard_others ############
# bash script/train_resume.sh dp2 move_playingcard_others_real10 3
# bash script/train_resume.sh dp2 move_playingcard_others_real20 3
# bash script/train_resume.sh dp2 move_playingcard_others_sim200 1 ✅⭕
# bash script/train_resume.sh dp2 move_playingcard_others_real10_sim200 3 ✅
# bash script/train_resume.sh dp2 move_playingcard_others_pseudo200 1 ✅
# bash script/train_resume.sh dp2 move_playingcard_others_real10_pseudo200 3 ✅

############ place_mouse_pad_color ############
# bash script/train_resume.sh dp2 place_mouse_pad_color_real10 3
# bash script/train_resume.sh dp2 place_mouse_pad_color_real20 3
# bash script/train_resume.sh dp2 place_mouse_pad_color_sim200 0 
# bash script/train_resume.sh dp2 place_mouse_pad_color_real10_sim200 3 
# bash script/train_resume.sh dp2 place_mouse_pad_color_pseudo200 1 
# bash script/train_resume.sh dp2 place_mouse_pad_color_real10_pseudo200 3 

############ place_mouse_pad_others ############
# bash script/train_resume.sh dp2 place_mouse_pad_others_real10 3
# bash script/train_resume.sh dp2 place_mouse_pad_others_real20 3
# bash script/train_resume.sh dp2 place_mouse_pad_others_sim200 1 
# bash script/train_resume.sh dp2 place_mouse_pad_others_real10_sim200 3 
# bash script/train_resume.sh dp2 place_mouse_pad_others_pseudo200 1 
# bash script/train_resume.sh dp2 place_mouse_pad_others_real10_pseudo200 3 


policy_name=${1}
task_name=${2}
gpu_id=${3}

policy_name_upper=${policy_name^^} # dp2 => DP2
base_dir="_outputs/${policy_name_upper}/${task_name}"
latest_date_dir=$(find "${base_dir}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)
resume_ckpt="${latest_date_dir}/checkpoints/last.ckpt"


# 错误检查
if [ ! -d "${base_dir}" ]; then
	echo "[Error] 目录不存在: ${base_dir}"
	exit 1
fi
if [ -z "${latest_date_dir}" ]; then
	echo "[Error] 未找到日期目录: ${base_dir}"
	exit 1
fi
if [ ! -f "${resume_ckpt}" ]; then
	echo "[Error] 未找到checkpoint: ${resume_ckpt}"
	exit 1
fi
echo "[Resume] 使用checkpoint: ${resume_ckpt}"

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python -u train.py --config-name=${policy_name}  task=${task_name} +resume_ckpt=${resume_ckpt} 2>&1 | tee Temp/${task_name}_${policy_name}.out


                                