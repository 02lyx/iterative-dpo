CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file ./configs/zero3.yaml dpo_iteration/run_dpo.py ./configs/training.yaml
# python dpo_iteration/run_dpo.py ./configs/training.yaml