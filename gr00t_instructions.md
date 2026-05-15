# GR00T N1.5 Model Training Guide

---

## Prerequisites

Before starting, you need:

- A dataset in **LeRobot v2.1 format** with `data/`, `videos/`, and `meta/` directories
- A **HuggingFace account** with a write token
- Access to a **Remote GPU**. Using RunPod in this instance. 
- An **SSH key pair** on your own machine

---

## Step 1 - Prepare Your Local Machine

### 1.1 Generate an SSH Key

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# When prompted for a file location, press Enter to use the default (~/.ssh/id_ed25519)
# Or specify a custom path e.g. ~/gr00t_key
```

Add your public key to RunPod:
1. Go to **RunPod → Settings → SSH Public Keys**
2. Paste the contents of `~/.ssh/id_ed25519.pub` (or your custom `.pub` file)

### 1.2 Upload Your Dataset to HuggingFace

Install the HuggingFace CLI and log in:

```bash
pip install huggingface_hub
huggingface-cli login
# or: hf auth login
```

Create a dataset repo and upload:

```bash
huggingface-cli repo create xarm7-cylinder-place --type dataset
huggingface-cli upload ItzSmudge/xarm7-cylinder-place /path/to/your/dataset --repo-type dataset
```

---

## Step 2 - Remote GPU Setup

### 2.1 Create the Pod

1. Go to **RunPod → Pods → + GPU Pod**
2. Select template: **Runpod Pytorch 2.1** (`runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`)
   - Provides Python 3.10 and CUDA 11.8 which are compatible with GR00T N1.5
   - Do **not** use PyTorch 2.4+ templates (Python 3.11, causes dependency conflicts)
3. Select GPU: Any GPU with at least **48 GB** of VRAM works, such as the **A40**. GPUs with less VRAM can still be used but with smaller batch size
4. Set **Container Disk** to **150 GB** minimum
5. Enable **SSH**
6. Deploy

### 2.2 Connect via SSH

```bash
ssh your-pod-id@ssh.runpod.io -i ~/.ssh/id_ed25519
```

The SSH command is shown in the RunPod console under **Connect**.

> **Note:** SSH keys only apply to pods created *after* the key was added to RunPod settings. If SSH fails, terminate the pod and create a new one.

---

## Step 3 - Environment Setup

Run these commands in order. Do not skip steps.

### 3.1 Verify the Base Environment

```bash
python --version    # Must be 3.10.x
nvcc --version      # Should show CUDA 11.8
python -c "import torch; print(torch.__version__)"  # Should be 2.1.0+cu118
```

### 3.2 Clone the GR00T N1.5 Repo 

> **Important:** The main branch is now N1.7. Always pin to the N1.5 release commit.

```bash
cd /workspace
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
git checkout 5e3ef5e
```

### 3.3 Install Dependencies in the Correct Order

Order matters. Follow exactly:

```bash
# Step 1: Pin scipy and scikit-image before anything else (newer versions require Python 3.11)
pip install scipy==1.13.1 scikit-image==0.21.0 numpy==1.26.4 pandas==2.2.3 matplotlib==3.9.0

# Step 2: Install flash-attn (must be done before gr00t to avoid build conflicts)
pip install psutil
pip install --no-build-isolation flash-attn==2.5.8

# Step 3: Install GR00T
pip install -e ".[base]" --ignore-requires-python

# Step 4: Pin transformers to the exact version required
pip install transformers==4.51.3

# Step 5: Clear the HuggingFace module cache (avoids Eagle processor version conflicts)
rm -rf /root/.cache/huggingface/modules
```

### 3.4 Verify the Installation

```bash
python -c "import gr00t; print('gr00t ok')"
python -c "import flash_attn; print('flash_attn ok')"
python scripts/gr00t_finetune.py --help
```

All three should succeed without errors before proceeding.

---

## Step 4 - Prepare Your Data Config

GR00T N1.5 requires a data config registered in `gr00t/experiment/data_config.py`. Your config defines which keys in your dataset correspond to video, state, action, and language modalities.

### 4.1 Check Your Dataset Keys

First inspect your dataset to get the exact key names:

```bash
python3 -c "
import json
info = json.load(open('/workspace/dataset/meta/modality.json'))
print(json.dumps(info, indent=2))
"
```

This tells you the exact video, state, and action key names to use in your config.

### 4.2 Add Your Config to data_config.py

The config must be registered directly in `gr00t/experiment/data_config.py`. Append it at the end of the file:

```python
class YourRobotDataConfig(BaseDataConfig):
    video_keys = ["video.your_camera_1", "video.your_camera_2"]       # from modality.json video keys, prefixed with "video."
    state_keys = ["state.your_arm", "state.your_gripper"]              # from modality.json state keys, prefixed with "state."
    action_keys = ["action.your_arm", "action.your_gripper"]           # from modality.json action keys, prefixed with "action."
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys, normalization_modes={key: "min_max" for key in self.state_keys}),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys, normalization_modes={key: "min_max" for key in self.action_keys}),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)

DATA_CONFIG_MAP["your_robot"] = YourRobotDataConfig()
```

> **Key naming convention:** Video keys in the config must be prefixed with `video.`, state keys with `state.`, and action keys with `action.` - followed by the key names shown in `modality.json`.

For the xArm7 setup specifically, use the pre-built config provided in `examples/xarm7/` (see below).

---

## Step 5 - Download Dataset and Run Training

### 5.1 Download Your Dataset

```bash
huggingface-cli login
huggingface-cli download ItzSmudge/xarm7-cylinder-place \
  --repo-type dataset --local-dir /workspace/dataset
```

### 5.2 Run Training

> **Note:** Use `/tmp/checkpoints` as output directory, not `/workspace/checkpoints`. The `/workspace` network volume has a quota (typically 20GB) that fills up quickly with checkpoints. `/tmp` uses the local container disk (150GB).

```bash
cd /workspace
python Isaac-GR00T/scripts/gr00t_finetune.py \
  --dataset-path /workspace/dataset \
  --num-gpus 1 \
  --output-dir /tmp/checkpoints \
  --max-steps 10000 \
  --data-config xarm7 \
  --batch-size 16 \
  --embodiment-tag new_embodiment \
  --video-backend decord \
  --save-steps 9999 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --report-to tensorboard
```

> **It is reccommended to use `--save-steps 9999`** Each checkpoint is ~16GB. Saving every 1000 steps fills the disk. Save only at the end unless you specifically need intermediate checkpoints.

**Expected training time on A40:** ~2–2.5 hours for 10,000 steps with 2 camera streams.

### 5.3 Monitor Disk During Training

In a second terminal, occasionally run:

```bash
df -h /
```

Make sure you have at least 20GB free before the checkpoint save step.

---

## Step 6 - Save Your Checkpoint

When training completes, immediately upload to HuggingFace before terminating the pod:

```bash
export HF_HOME=/workspace/hf_cache
export TMPDIR=/workspace/tmp
mkdir -p /workspace/hf_cache /workspace/tmp

huggingface-cli login

huggingface-cli upload ItzSmudge/xarm7-gr00t-checkpoints \
  /tmp/checkpoints/checkpoint-10000 \
  --repo-type model \
  --commit-message "Final checkpoint step 10000"
```

> HuggingFace CLI writes temporary files during upload. Pointing these to `/workspace` avoids filling the local disk.

---

## Step 7 - ONNX Export

The ONNX export converts your trained model into a format suitable for deployment.

### 7.1 Patch the Export Script

The export script is hardcoded for the GR1 robot. Patch it for your robot:

```bash
python3 << 'EOF'
content = open("/workspace/Isaac-GR00T/deployment_scripts/export_onnx.py").read()
content = content.replace('data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]', 'data_config = DATA_CONFIG_MAP["xarm7"]')
content = content.replace('EMBODIMENT_TAG = "gr1"', 'EMBODIMENT_TAG = "new_embodiment"')
content = content.replace("opset_version=19", "opset_version=17")
open("/workspace/Isaac-GR00T/deployment_scripts/export_onnx.py", "w").write(content)
print("Done")
EOF
```

### 7.2 Copy experiment_cfg into the Checkpoint

```bash
cp -r /tmp/checkpoints/experiment_cfg /tmp/checkpoints/checkpoint-10000/
```

### 7.3 Run the Export

```bash
export HF_HOME=/workspace/hf_cache
export TMPDIR=/workspace/tmp

python /workspace/Isaac-GR00T/deployment_scripts/export_onnx.py \
  --dataset_path /workspace/dataset \
  --model_path /tmp/checkpoints/checkpoint-10000 \
  --onnx_model_path /tmp/onnx_export
```

This takes 10–20 minutes. The script may print warnings - these are harmless.

### 7.4 Collect Deployment Files

```bash
cp /tmp/checkpoints/checkpoint-10000/config.json /tmp/onnx_export/
cp -r /tmp/checkpoints/checkpoint-10000/experiment_cfg /tmp/onnx_export/
ls /tmp/onnx_export/
```

You should see: `action_head/`, `eagle2/`, `experiment_cfg/`, `config.json`

### 7.5 Upload Deployment Files

```bash
huggingface-cli upload ItzSmudge/xarm7-gr00t-checkpoints \
  /tmp/onnx_export \
  --repo-type model \
  --commit-message "ONNX export - deployment files"
```

---

## Known Issues and Fixes

### flash-attn build fails
**Cause:** torch not installed before flash-attn, or wrong version.  
**Fix:** Always install `scipy`, `scikit-image`, and `psutil` before flash-attn. Use `flash-attn==2.5.8` with torch 2.1+cu118.

### `ModuleNotFoundError: No module named 'flash_attn_2_cuda'`
**Cause:** A stale pre-installed `.so` file conflicts with the newly installed version.  
**Fix:**
```bash
rm -rf /usr/local/lib/python3.10/dist-packages/flash_attn*
pip install --no-build-isolation flash-attn==2.5.8
```

### Eagle processor ImportError (VideoInput, get_patch_output_size)
**Cause:** Wrong transformers version. The Eagle2 processor requires exactly transformers 4.51.3.  
**Fix:**
```bash
rm -rf /root/.cache/huggingface/modules
pip install transformers==4.51.3
```

### `Disk quota exceeded` on /workspace
**Cause:** RunPod network volumes have a small quota (typically 20GB). Checkpoints are 16GB each.  
**Fix:** Always use `/tmp/checkpoints` as `--output-dir`. Use `--save-steps 9999` to save only once.

### `JSONDecodeError` on metadata.json
**Cause:** A previous run failed mid-write, leaving an empty metadata.json.  
**Fix:**
```bash
find /tmp/checkpoints -name "metadata.json" -exec rm {} \;
```

### WandB API key rejected
**Cause:** New WandB keys (`wandb_v1_...`) are 86 characters, but this version of GR00T expects 40-character legacy keys.  
**Fix:** Use `--report-to tensorboard` instead.

### `opset_version=19` error during ONNX export
**Cause:** torch 2.1 only supports ONNX opset up to 17.  
**Fix:** The patch in Step 7.1 handles this automatically.

### Only ~50% of episodes used during training
**Cause:** Episodes shorter than the action chunk size (16 steps × frame skip) are automatically filtered out by GR00T's data loader. This is expected behaviour and not an error.

---

## Files Included in This Repo

| File | Purpose |
|---|---|
| `README.md` | This guide |
| `examples/xarm7/xarm7_config.py` | Pre-built data config for xArm7 + 2-finger gripper + 2 cameras |
| `patch_data_config.py` | Script to register the xarm7 config into GR00T's data_config.py |
| `patch_export_onnx.py` | Script to patch export_onnx.py for new_embodiment and opset 17 |

---

## Quick Reference - Full Command Sequence

```bash
# 1. Clone and pin to N1.5
cd /workspace
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
git checkout 5e3ef5e

# 2. Install dependencies
pip install scipy==1.13.1 scikit-image==0.21.0 numpy==1.26.4 pandas==2.2.3 matplotlib==3.9.0
pip install psutil
pip install --no-build-isolation flash-attn==2.5.8
pip install -e ".[base]" --ignore-requires-python
pip install transformers==4.51.3
rm -rf /root/.cache/huggingface/modules

# 3. Download dataset
huggingface-cli login
huggingface-cli download ItzSmudge/xarm7-cylinder-place --repo-type dataset --local-dir /workspace/dataset

# 4. Register your data config (edit gr00t/experiment/data_config.py)

# 5. Train
cd /workspace
python Isaac-GR00T/scripts/gr00t_finetune.py \
  --dataset-path /workspace/dataset \
  --num-gpus 1 \
  --output-dir /tmp/checkpoints \
  --max-steps 10000 \
  --data-config xarm7 \
  --batch-size 16 \
  --embodiment-tag new_embodiment \
  --video-backend decord \
  --save-steps 9999 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --report-to tensorboard

# 6. Upload checkpoint
export HF_HOME=/workspace/hf_cache && export TMPDIR=/workspace/tmp
mkdir -p /workspace/hf_cache /workspace/tmp
huggingface-cli upload ItzSmudge/xarm7-gr00t-checkpoints --repo-type model /tmp/checkpoints/checkpoint-10000

# 7. ONNX export (after patching)
cp -r /tmp/checkpoints/experiment_cfg /tmp/checkpoints/checkpoint-10000/
python /workspace/Isaac-GR00T/deployment_scripts/export_onnx.py \
  --dataset_path /workspace/dataset \
  --model_path /tmp/checkpoints/checkpoint-10000 \
  --onnx_model_path /tmp/onnx_export
cp /tmp/checkpoints/checkpoint-10000/config.json /tmp/onnx_export/
cp -r /tmp/checkpoints/checkpoint-10000/experiment_cfg /tmp/onnx_export/
huggingface-cli upload ItzSmudge/xarm7-gr00t-checkpoints --repo-type model /tmp/onnx_export
```