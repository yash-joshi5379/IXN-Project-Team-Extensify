from huggingface_hub import snapshot_download

snapshot_download('lerobot/smolvla_base', local_dir='./smolvla_base_weights')