#!/usr/bin/env python3
"""
patch_data_config.py

Registers the xArm7 data config into GR00T's data_config.py.
Run this from the Isaac-GR00T root directory after cloning and checking out N1.5.

Usage:
    python patch_data_config.py

This script is idempotent — running it multiple times is safe.
"""

import os
import sys

DATA_CONFIG_PATH = "gr00t/experiment/data_config.py"

XARM7_CONFIG = '''

class Xarm7DataConfig(BaseDataConfig):
    video_keys = ["video.0_femtobolt", "video.1_gemini330"]
    state_keys = [
        "state.0_xarm7+xarmgripper-arm",
        "state.1_xarm7+xarmgripper-hand",
    ]
    action_keys = [
        "action.0_xarm7+xarmgripper-arm",
        "action.1_xarm7+xarmgripper-hand",
    ]
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

DATA_CONFIG_MAP["xarm7"] = Xarm7DataConfig()
'''

def main():
    if not os.path.exists(DATA_CONFIG_PATH):
        print(f"Error: {DATA_CONFIG_PATH} not found.")
        print("Make sure you are running this from the Isaac-GR00T root directory.")
        sys.exit(1)

    with open(DATA_CONFIG_PATH, "r") as f:
        content = f.read()

    if 'DATA_CONFIG_MAP["xarm7"]' in content:
        print("xarm7 config already registered in data_config.py. Nothing to do.")
        return

    # Remove any previous partial attempt
    cutoff = content.find("\nclass Xarm7DataConfig")
    if cutoff != -1:
        content = content[:cutoff]
        print("Removed previous partial xarm7 config.")

    content = content + XARM7_CONFIG

    with open(DATA_CONFIG_PATH, "w") as f:
        f.write(content)

    print("Successfully registered xarm7 config in data_config.py.")
    print("You can now use --data-config xarm7 in the training command.")


if __name__ == "__main__":
    main()
