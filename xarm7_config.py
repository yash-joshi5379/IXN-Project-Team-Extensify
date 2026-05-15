# xArm7 + 2-finger gripper data config for GR00T N1.5
# Robot: xArm7 (7 DoF arm + 1 DoF gripper = 8 DoF total)
# Cameras: 2 (static/femtobolt + wrist/gemini330)
#
# This config is designed to be registered into gr00t/experiment/data_config.py
# via patch_data_config.py. Do not import this file directly.

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.model.transforms import GR00TTransform
from gr00t.experiment.data_config import BaseDataConfig


class Xarm7DataConfig(BaseDataConfig):
    # Video keys: must match keys in meta/modality.json under "video", prefixed with "video."
    video_keys = ["video.0_femtobolt", "video.1_gemini330"]

    # State keys: must match keys in meta/modality.json under "state", prefixed with "state."
    state_keys = [
        "state.0_xarm7+xarmgripper-arm",   # 7 joint positions
        "state.1_xarm7+xarmgripper-hand",  # 1 gripper position
    ]

    # Action keys: must match keys in meta/modality.json under "action", prefixed with "action."
    action_keys = [
        "action.0_xarm7+xarmgripper-arm",
        "action.1_xarm7+xarmgripper-hand",
    ]

    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))  # 16-step action chunk at 30fps = ~0.5s lookahead

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        transforms = [
            # Video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # State transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # Action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # Concat and model transforms
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
