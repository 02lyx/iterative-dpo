# export WORLD_SIZE=4
# accelerate launch ./annotate_data/get_rewards.py --dataset_name_or_path ./data/gen_data.json --output_dir ./data/data_with_rewards.json --K 4
accelerate launch --num_processes 4 ./annotate_data/get_rewards.py --dataset_name_or_path ./data/gen_data.json --output_dir ./data/data_with_rewards.json --K 16 --reward_name_or_path 'RLHFlow/ArmoRM-Llama3-8B-v0.1'
