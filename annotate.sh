conda init
conda activate anno-train

python annotate_data/true_label.py --ds_dir data/gen_data.json --output_dir data/ 
python annotate_data/true_label.py --ds_dir iter_dpo/Test3_LLaMA3_iter3_data.json --output_dir data/ 