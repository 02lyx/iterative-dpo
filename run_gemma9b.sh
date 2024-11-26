git clone https://github.com/02lyx/iterative-dpo.git
cd iterative-dpo

bash env_setup.sh

export WANDB_API_KEY=6f9e1eaf73cd08b4f0cd4674c7856201f2453428
huggingface-cli login  --token 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

bash Gemma-2-9b-it.sh

huggingface-cli upload Yuanxin-Liu/Gemma-2-9b-it-idpo-2 Gemma-2-9b-it_iter3 --token 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