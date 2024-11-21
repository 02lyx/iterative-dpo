CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --config_file ./configs/zero3.yaml dpo_iteration/run_dpo.py ./configs/training.yaml
# python dpo_iteration/run_dpo.py ./configs/training.yaml