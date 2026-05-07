"""
Convert processed_aloha_dataset/ HDF5 files to ACT-compatible format.

What this script fixes:
  1. Adds root.attrs['sim'] = False  (ACT requires this attribute)
  2. Decompresses JPEG images from (T, bytes) → raw pixels (T, H, W, 3)
  3. Renames camera keys to ACT-standard names:
       0_femtobolt  → cam_static
       1_gemini330  → cam_wrist
  4. Renumbers episodes consecutively 0..N-1 regardless of original naming
     (ACT loads episode_{0}.hdf5, episode_{1}.hdf5 ... episode_{N-1}.hdf5)

INPUT:  processed_aloha_dataset/episode_XX.hdf5
OUTPUT: act_dataset/episode_{0..N-1}.hdf5

USAGE:
  pip install h5py opencv-python numpy tqdm
  python convert_to_act_format.py
"""

import glob
import io
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
INPUT_DIR  = Path("processed_aloha_dataset")   # your cleaned HDF5 files
OUTPUT_DIR = Path("act_train_dataset")               # ACT-ready files go here

# Map from your camera names → ACT camera names
# ACT will be told to use these names via --camera_names argument
CAMERA_RENAME = {
    "0_femtobolt": "cam_static",
    "1_gemini330": "cam_wrist",
}

# Whether to resize images. ACT's ResNet backbone works with any size,
# but 480x640 is standard. Your JPEG images are likely already this size.
# Set to None to keep original size, or (H, W) to resize.
RESIZE_TO = None   # e.g. (480, 640) to force standard size
# ─────────────────────────────────────────────────────────────────────────────


def decompress_jpeg_frames(compressed_frames: np.ndarray,
                            compress_len: np.ndarray) -> np.ndarray:
    """
    Decompress a sequence of JPEG-encoded frames.

    Args:
        compressed_frames: shape (T, max_bytes) uint8 — padded JPEG byte arrays
        compress_len:      shape (T,) float32 — actual byte length of each frame

    Returns:
        raw_frames: shape (T, H, W, 3) uint8 RGB
    """
    frames = []
    T = compressed_frames.shape[0]

    for t in range(T):
        n_bytes = int(compress_len[t])
        jpeg_bytes = compressed_frames[t, :n_bytes].tobytes()
        img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # BGR
        if frame is None:
            raise ValueError(f"Failed to decode JPEG at frame {t}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # → RGB

        if RESIZE_TO is not None:
            frame = cv2.resize(frame, (RESIZE_TO[1], RESIZE_TO[0]))  # cv2 uses (W, H)

        frames.append(frame)

    return np.stack(frames, axis=0)   # (T, H, W, 3)


def convert_episode(input_path: Path, output_path: Path, new_episode_id: int):
    """Convert one HDF5 file from your format to ACT format."""

    with h5py.File(input_path, "r") as src:
        # Read action and proprioception
        action      = src["/action"][()]                          # (T, 8)
        qpos        = src["/observations/qpos"][()]               # (T, 8)
        qvel        = src["/observations/qvel"][()]               # (T, 8)
        effort      = src["/observations/effort"][()]             # (T, 8)
        force_torque= src["/observations/force_torque"][()]       # (T, 6)

        T = action.shape[0]

        # Read compress_len — shape (2, T): one row per camera
        compress_len = src["/compress_len"][()]                   # (2, T)

        # Read and decompress both cameras
        image_dict = {}
        for cam_idx, (src_name, dst_name) in enumerate(CAMERA_RENAME.items()):
            compressed = src[f"/observations/images/{src_name}"][()]  # (T, max_bytes)
            lengths    = compress_len[cam_idx]                         # (T,)
            print(f"    Decompressing {src_name} → {dst_name} ({T} frames) ...")
            raw_frames = decompress_jpeg_frames(compressed, lengths)   # (T, H, W, 3)
            image_dict[dst_name] = raw_frames

    # Get image shape from first camera for reference
    sample_shape = list(image_dict.values())[0].shape  # (T, H, W, 3)
    H, W = sample_shape[1], sample_shape[2]
    print(f"    Image shape: {H}x{W}  |  Episode length: {T} steps")

    # Write ACT-compatible HDF5
    with h5py.File(output_path, "w") as dst:
        # REQUIRED: ACT reads root.attrs['sim'] to set is_sim flag
        dst.attrs["sim"] = False

        # Action — ACT reads /action directly
        dst.create_dataset("/action", data=action, dtype="float32")

        # Observations group
        obs = dst.require_group("observations")

        # Joint positions — ACT reads /observations/qpos
        obs.create_dataset("qpos", data=qpos, dtype="float32")

        # Joint velocities (not used by ACT policy but good to preserve)
        obs.create_dataset("qvel", data=qvel, dtype="float32")

        # Extra proprioception (preserved for completeness)
        obs.create_dataset("effort",       data=effort,       dtype="float32")
        obs.create_dataset("force_torque", data=force_torque, dtype="float32")

        # Camera images — ACT reads /observations/images/{cam_name}
        # as raw pixel arrays (T, H, W, 3) uint8
        images_grp = obs.require_group("images")
        for cam_name, frames in image_dict.items():
            images_grp.create_dataset(
                cam_name, data=frames, dtype="uint8",
                chunks=(1, H, W, 3),          # one frame per chunk for fast random access
                compression="gzip",
                compression_opts=4,           # moderate compression, fast decode
            )

    print(f"    ✓ Written: {output_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all input files and sort them
    input_files = sorted(INPUT_DIR.glob("episode_*.hdf5"))
    if not input_files:
        print(f"ERROR: No HDF5 files found in {INPUT_DIR}")
        return

    n = len(input_files)
    print(f"Found {n} episodes in {INPUT_DIR}")
    print(f"Camera mapping: {CAMERA_RENAME}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()

    for new_idx, input_path in enumerate(tqdm(input_files, desc="Converting episodes")):
        output_path = OUTPUT_DIR / f"episode_{new_idx}.hdf5"
        print(f"\n[{new_idx+1}/{n}] {input_path.name} → {output_path.name}")
        convert_episode(input_path, output_path, new_idx)

    print(f"\n✅  Conversion complete.")
    print(f"    {n} episodes written to {OUTPUT_DIR}/")
    print(f"    Camera names for training: {list(CAMERA_RENAME.values())}")
    print(f"\nTraining command:")
    cam_names = " ".join(CAMERA_RENAME.values())
    print(f"  python act/imitate_episodes.py \\")
    print(f"    --task_name cylinder_insert \\")
    print(f"    --ckpt_dir /root/act_output \\")
    print(f"    --policy_class ACT \\")
    print(f"    --kl_weight 10 \\")
    print(f"    --chunk_size 100 \\")
    print(f"    --hidden_dim 512 \\")
    print(f"    --batch_size 8 \\")
    print(f"    --dim_feedforward 3200 \\")
    print(f"    --num_epochs 2000 \\")
    print(f"    --lr 1e-5 \\")
    print(f"    --seed 0 \\")
    print(f"    --camera_names {cam_names} \\")
    print(f"    --dataset_dir {OUTPUT_DIR.resolve()} \\")
    print(f"    --num_episodes {n}")


if __name__ == "__main__":
    main()