#!/bin/bash

# Initialize environment and paths
source ~/.bashrc
eval "$(conda shell.bash hook)"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
initial_model="google/gemma-2-2b-it"
base_path="${SCRIPT_DIR}/self_correct_iter_Gemma-2-2b-it"
mkdir -p "$base_path"

export USE_FLASH_ATTENTION=0
export TRANSFORMERS_NO_FLASH_ATTENTION=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# Function to run training iteration
run_iteration() {
    local model_path=$1
    local jsonl_input=$2
    local json_output=$3
    local model_output=$4
    local iteration=$5

    echo "Starting iteration: ${iteration}"
    echo "Using model: ${model_path}"
    
    # Check if final annotated data exists
    if [ ! -f "$model_output" ]; then
        echo "Annotated data not found at ${model_output}, starting data generation..."
        
        # Create output directory for this iteration
        mkdir -p "${json_output}"

        # Check if merged data exists
        if [ ! -f "${json_output}_data.json" ]; then
            echo "Generated data not found, starting generation..."
            
            # Activate environment for generation
            conda deactivate
            conda activate gen-eval

            echo "Generating data using multiple GPUs..."
            # Generate data using multiple GPUs
            my_world_size= 6
            gpu_ids=(0 1 2 3 4 5)
            for i in "${!gpu_ids[@]}"; do
                CUDA_VISIBLE_DEVICES=${gpu_ids[$i]} python "${SCRIPT_DIR}/generation/gen_hf2.py" \
                    --model_name_or_path ${model_path} \
                    --dataset_name_or_path ${jsonl_input} \
                    --output_dir ${json_output} \
                    --K 30 --temperature 1.0 \
                    --local_index $i \
                    --my_world_size ${my_world_size} &
            done
            wait

            echo "Merging generated data..."
            # Merge generated data
            python "${SCRIPT_DIR}/generation/merge_data.py" \
                --base_path ${json_output} \
                --output_dir "${json_output}_data.json" \
                --num_datasets 6
        else
            echo "Found existing generated data at ${json_output}_data.json"
        fi

        echo "Annotating data..."
        # Switch environment and annotate data
        conda deactivate
        conda activate anno-train
        python "${SCRIPT_DIR}/annotate_data/true_label.py" \
            --ds_dir "${json_output}_data.json" \
            --output_dir $model_output
    else
        echo "Found existing annotated data at ${model_output}"
    fi

    echo "Starting training..."
    # Switch environment and run training
    conda deactivate
    conda activate anno-train
    
    # Create output directory for the model
    model_save_dir="${base_path}/${iteration}"
    mkdir -p "${model_save_dir}"
  
    

    # Run training with proper parameters
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "Number of available GPUs: $(nvidia-smi --list-gpus | wc -l)"
    accelerate launch --config_file "${SCRIPT_DIR}/configs/zero3.yaml" \
        "${SCRIPT_DIR}/self_correct.py" \
        --config_path "${SCRIPT_DIR}/configs/train_config.yaml" \
        --model_name_or_path "$model_path" \
        --output_dir "$model_save_dir" \
        --train_dir "$model_output" \
        --run_name "$iteration"   

    echo "Iteration ${iteration} completed"
    echo "Model saved to: ${model_save_dir}"
}

# Main loop
echo "Starting training pipeline..."
for i in {1..1}; do
    iteration_name="Gemma-2-2b-it_iter${i}"
    jsonl_input="RLHF4MATH/prompt_iter${i}"
    json_output="${base_path}/Train${i}_${iteration_name}"
    model_output="${json_output}_reward.json"
    model_path=$([[ $i -eq 1 ]] && echo $initial_model || echo "${base_path}/Gemma-2-2b-it_iter$((i-1))")

    echo "Starting iteration ${i}..."
    run_iteration "$model_path" "$jsonl_input" "$json_output" "$model_output" "$iteration_name"
done

echo "Training pipeline completed" 