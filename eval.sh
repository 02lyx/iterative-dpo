# lm_eval --model hf \
#     --model_args pretrained=Q_models/tt_3/debug2\
#     --tasks gsm8k \
#     --device cuda:0\
#     --batch_size 8
lm_eval --model hf \
--model_args pretrained=/home/swb9572/iterative-dpo/pipeline_test/baseline_sft \
--tasks gsm8k \
--device cuda:0 \
--batch_size 8 \
--output_path ./Logs \
--log_samples \