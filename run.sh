#!/usr/bin/env bash

# folders=("causalnet" "copa" "strategyqa" "hellaswag" "math" "gpqa" "csqa")
# script="stable_run.py"
# work_dir="causal_discovery"

# for folder in "${folders[@]}"; do
#     data_dir="/home/wuchenxi/projects/causal-reasoning-in-pieces/${folder}"
#     for csv_file in "$data_dir"/*.csv; do
#         [ -e "$csv_file" ] || continue
#         base_name=$(basename "$csv_file" .csv)
#         rel_csv_file="../${folder}/$(basename "$csv_file")"
#         for mode in 1 2 3 4; do
#             session_name="${base_name}_mode${mode}"
#             cmds="conda activate causal & python $script --input-file $rel_csv_file --mode $mode"
#             screen -S "$session_name" -dm bash -c "$cmds"
#             echo "启动 $session_name 的screen会话，运行命令：$cmds"
#         done
#     done
# done

# folders=("causalnet" "copa" "strategyqa" "hellaswag" "math" "gpqa" "csqa")
# script="stable_run.py"
# work_dir="causal_discovery"

# for folder in "${folders[@]}"; do
#     data_dir="/home/wuchenxi/projects/causal-reasoning-in-pieces/${folder}"
#     for csv_file in "$data_dir"/*.csv; do
#         [ -e "$csv_file" ] || continue
#         base_name=$(basename "$csv_file" .csv)
#         rel_csv_file="../${folder}/$(basename "$csv_file")"
#         session_name="${base_name}_mode1"
#         cmds="conda activate causal & python $script --input-file $rel_csv_file --mode 1"
#         screen -S "$session_name" -dm bash -c "$cmds"
#         echo "启动 $session_name 的screen会话，运行命令：$cmds"
#     done
# done

#!/usr/bin/env bash

declare -A dataset_models
dataset_models["math"]="llama-8b qwen-7b ds-r1"
dataset_models["csqa"]="llama-8b llama-70b qwen-7b qwen-72b ds-r1"
dataset_models["gpqa"]="llama-8b llama-70b qwen-7b qwen-72b ds-r1 gpt-3.5"
dataset_models["strategyqa"]="llama-8b"
dataset_models["causalnet"]="ds-r1"
dataset_models["copa"]="llama-70b qwen-7b qwen-72b ds-r1"
dataset_models["hellaswag"]="llama-8b llama-70b qwen-7b qwen-72b ds-r1"

script="stable_run.py"

for dataset in "${!dataset_models[@]}"; do
    data_dir="/home/wuchenxi/projects/causal-reasoning-in-pieces/${dataset}"
    for model in ${dataset_models[$dataset]}; do
        for csv_file in "$data_dir"/*"${model}".csv; do
            [ -e "$csv_file" ] || continue
            base_name=$(basename "$csv_file" .csv)
            rel_csv_file="../${dataset}/$(basename "$csv_file")"
            session_name="${base_name}_mode1"
            cmds="conda activate causal & python $script --input-file $rel_csv_file --mode 1"
            screen -S "$session_name" -dm bash -c "$cmds"
            echo "启动 $session_name 的screen会话，运行命令：$cmds"
        done
    done
done