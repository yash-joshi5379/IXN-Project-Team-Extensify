from huggingface_hub import HfApi

# Enter your Hugging Face username here
hf_user = "yjoshi5379"

api = HfApi()
api.upload_large_folder(repo_id=f"{hf_user}/cylinder-pick-place",
                        repo_type="dataset",
                        folder_path="dataset/processed")