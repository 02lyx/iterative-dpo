# lm_eval --model hf \
#     --model_args pretrained=Q_models/tt_3/debug2\
#     --tasks gsm8k \
#     --device cuda:0\
#     --batch_size 8
lm_eval --model hf \
--model_args pretrained=Online-RLHF/pipeline_test/no_fast_attn \
--tasks gsm8k \
--device cuda:0 \
--batch_size 8 \
--output_path ./Logs \
--log_samples \