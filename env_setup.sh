conda init
conda env create -f gen-eval.yaml
conda env create -f anno-train.yaml
conda init
eval "$(conda shell.bash hook)"
conda activate anno-train

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout 27f7dbf00663dab66ad7334afb7a1311fa251f41
python -m pip install .
pip install distro
cd ..

