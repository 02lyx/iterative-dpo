git clone https://github.com/02lyx/iterative-dpo.git
cd iterative-dpo

bash env_setup.sh

export WANDB_API_KEY=6f9e1eaf73cd08b4f0cd4674c7856201f2453428
huggingface-cli login  --token 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

bash Mistral-7B-it.sh

huggingface-cli upload Yuanxin-Liu/Mistral-7b-it-idpo-2 Mistral-7b-it_iter3 --token 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