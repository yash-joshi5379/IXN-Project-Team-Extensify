#!/usr/bin/env python3
"""
patch_export_onnx.py

Patches deployment_scripts/export_onnx.py to:
1. Use the xarm7 data config instead of the default GR1 config
2. Use new_embodiment tag instead of gr1
3. Downgrade ONNX opset from 19 to 17 (required for torch 2.1 compatibility)

Run this from the Isaac-GR00T root directory before running export_onnx.py.

Usage:
    python patch_export_onnx.py

This script is idempotent — running it multiple times is safe.
"""

import os
import sys

EXPORT_SCRIPT_PATH = "deployment_scripts/export_onnx.py"


def main():
    if not os.path.exists(EXPORT_SCRIPT_PATH):
        print(f"Error: {EXPORT_SCRIPT_PATH} not found.")
        print("Make sure you are running this from the Isaac-GR00T root directory.")
        sys.exit(1)

    with open(EXPORT_SCRIPT_PATH, "r") as f:
        content = f.read()

    changes = 0

    # Patch data config
    old_config = 'data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]'
    new_config = 'data_config = DATA_CONFIG_MAP["xarm7"]'
    if old_config in content:
        content = content.replace(old_config, new_config)
        changes += 1
        print("Patched: data config -> xarm7")
    elif new_config in content:
        print("Already patched: data config")
    else:
        print("Warning: could not find data config line to patch.")

    # Patch embodiment tag
    old_tag = 'EMBODIMENT_TAG = "gr1"'
    new_tag = 'EMBODIMENT_TAG = "new_embodiment"'
    if old_tag in content:
        content = content.replace(old_tag, new_tag)
        changes += 1
        print("Patched: embodiment tag -> new_embodiment")
    elif new_tag in content:
        print("Already patched: embodiment tag")
    else:
        print("Warning: could not find embodiment tag line to patch.")

    # Patch opset version
    if "opset_version=19" in content:
        content = content.replace("opset_version=19", "opset_version=17")
        changes += 1
        print("Patched: ONNX opset version 19 -> 17")
    elif "opset_version=17" in content:
        print("Already patched: ONNX opset version")
    else:
        print("Warning: could not find opset_version to patch.")

    with open(EXPORT_SCRIPT_PATH, "w") as f:
        f.write(content)

    if changes > 0:
        print(f"\nSuccessfully applied {changes} patch(es) to {EXPORT_SCRIPT_PATH}.")
    else:
        print("\nNo changes were needed.")


if __name__ == "__main__":
    main()
