# First approach: initialize 4 VLLM processes and split the prompt set to the 4 agents
# The generated samples will be stored at output_dir + local_index + ".jsonl

my_world_size=4 # how many gpu you use
infer_model="ZhangShenao/baseline-gemma-2-2b-it-sft"
prompt_dir=Yuanxin-Liu/Test-Dataset
mkdir data
output_dir=./data/basesft_iter

conda init
conda activate data-eval

CUDA_VISIBLE_DEVICES=0 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 16 --temperature 1.0 --local_index 0 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=1 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 16 --temperature 1.0 --local_index 1 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=2 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 16 --temperature 1.0 --local_index 2 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=3 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 16 --temperature 1.0 --local_index 3 --my_world_size ${my_world_size}  &
# CUDA_VISIBLE_DEVICES=4 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 4 --my_world_size ${my_world_size}  &
# CUDA_VISIBLE_DEVICES=5 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 5 --my_world_size ${my_world_size}  &
# CUDA_VISIBLE_DEVICES=6 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 6 --my_world_size ${my_world_size}  &
# CUDA_VISIBLE_DEVICES=7 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 7 --my_world_size ${my_world_size}  &

# then, we merge the 8 datasets into one dataset.
wait
python ./generation/merge_data.py --base_path ${output_dir} --output_dir './data/gen_data.json' --num_datasets ${my_world_size}
# python ./generation/merge_data.py --base_path './data/basesft_iter3' --output_dir './data/basesft_iter3/merge_data.json' --num_datasets 4