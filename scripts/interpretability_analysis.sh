#!/bin/bash

execute_script() {
    local script_name=$1
    local table_name=$2
    local command="python experiments/${script_name} --config \"./configs/interpretability_analyses.toml\""

    # Append --table argument if table_name is provided
    if [[ ! -z "$table_name" ]]; then
        command+=" --table \"$table_name\""
    fi

    echo "Executing $script_name"
    eval $command
}


vis=false
sparse_codes=false
grad_cam=false
gpt3=false
causal=false
style=false


while getopts "vcbgsay" opt; do
  case $opt in
    v) vis=true ;;
    c) sparse_codes=true ;;
    g) gpt3=true ;;
    s) grad_cam=true ;;
    a) causal=true ;;
    y) style=true ;;
    \?) echo "Invalid option -$OPTARG" >&2 ;;
  esac
done

# Execute scripts based on flags
$vis && execute_script "visualize_embedding.py" "visualization"
$sparse_codes && execute_script "sparse_codes.py" "sparse_codes"
$gpt3 && execute_script "extract_feature_norms.py" "gpt3"
$grad_cam && execute_script "grad_cam.py"
$causal && execute_script "causal_comparison.py"
$style && execute_script "optimize_and_sample_stylegan.py" "act_max"