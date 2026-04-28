# Examples:
# tmux new -s train

############ DP2 Tasks ############
# bash scripts/train.sh dp2 1a_pick_meat_2d 0


policy_name=${1}
task_name=${2}
gpu_id=${3}

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python -u train.py --config-name=${policy_name}  task=${task_name} 2>&1 | tee Temp/${task_name}_${policy_name}.out


                                