# Examples:
# tmux new -s train
# bash scripts/train.sh dp2 collecting_objects 4

if [[ $# -lt 3 ]]; then
    echo "Usage: bash scripts/train.sh <policy_name> <task_name> <gpu_id> [info] [hydra_overrides...]"
    exit 1
fi

policy_name=${1}
task_name=${2}
gpu_id=${3}

if [[ $# -ge 4 ]]; then
    info=${4}
    shift 4
else
    info=""
    shift 3
fi

extra_args=("$@")

policy_name_upper=${policy_name^^} # 转换为大写
task_name_simple=${task_name%_[23]d}  # 去除 _3d 或 _2d 后缀

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

if [[ -z "${info}" ]]; then
    python -u train.py --config-name="${policy_name}" task="${task_name}" "${extra_args[@]}" 2>&1 | tee "Temp/${task_name}_${policy_name}.out"
else
    run_dir="_outputs/${policy_name_upper}/${task_name_simple}/${info}"
    mkdir -p "${run_dir}"
    touch "${run_dir}/debug.out"
    python -u train.py --config-name="${policy_name}" task="${task_name}" hydra.run.dir="${run_dir}" info="${info}" "${extra_args[@]}" 2>&1 | tee "${run_dir}/debug.out"
fi