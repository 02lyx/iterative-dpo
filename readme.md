## Step 1: Set up environment 
We set up two environments, ```gen-eval``` is used for generating data and evaluating model perforcement, and the other one ```anno-train``` is used for annotating data and DPO training.
To install ```gen-eval```, please follow these steps:

```sh
conda env create -f gen-eval.yaml
```
And to install training environment, please follow these steps:
```sh
conda create -n anno-train python=3.10.9
conda activate anno-train

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout 27f7dbf00663dab66ad7334afb7a1311fa251f41
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install flash-attn==2.6.3
pip install accelerate==0.34.0
pip install huggingface-hub==0.24.7
pip install transformers==4.45.0
pip install trl==0.11.1
pip install wandb==0.17.7
pip install numpy==1.26.4
pip install antlr4-python3-runtime
wandb login
huggingface-cli login
cd ..
```
## Step 2: Run the main loop
The pipeline is: generating data(sample K answers for each prompt) -> annotating data(using true label to annotate responses) -> training(construct pairs using correct responses and incorrect responses to do DPO)

And the whole process is packed, please run 
```sh
bash Gemma-2-2b-it.sh
``` 
to run iterative dpo for the model Gemma-2-2b-it, and then run
```sh
conda activate gen-eval
bash eval.sh
```
to evaluate model perforcement on GSM8K. Thanks for your effort and please let me know the result. If it works well, run the file ```upload_model.py``` under the environment ```anno-train```to upload the model to my huggingface account.