from huggingface_hub import upload_folder
import os

os.environ["HF_TOKEN"] = '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'
upload_folder(
    folder_path="Gemma-2-2b-it_iter3",
    repo_type="model",
    repo_id="Yuanxin-Liu/Gemma-2-2b-it_iter3"
)
