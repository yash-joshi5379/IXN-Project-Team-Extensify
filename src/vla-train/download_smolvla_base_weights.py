from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='lerobot/smolvla_base',
    local_dir='smolvla_base_weights',
    repo_type='model'
)
print('Downloaded to smolvla_base_weights/')