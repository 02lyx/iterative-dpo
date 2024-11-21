## Step 1: Set up environment 
We set up two environments, ```data-eval``` is used for generating data, annotating data, evaluating model perforcement, and the other ```train-only``` is used for DPO training.
To install ```data-eval```, please do the following

```sh
conda env create -f data_preparation.yaml
```
And to install training environment, do the following:
```sh
conda create -n train-only python=3.10.9
conda activate train-only

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
wandb login
huggingface-cli login
cd ..
```
## Step 2: Run the main loop
Run the bash to train it iteratively 