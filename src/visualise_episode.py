import sys
import shutil
import subprocess
from pathlib import Path

episode = sys.argv[1] if len(sys.argv) > 1 else "0"
print(f"Visualising episode {episode}")

# Clear cache before launching visualiser
cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print(f"Cache cleared.")

# Launch visualiser
subprocess.run([
    "lerobot-dataset-viz",
    "--repo-id", "local/cylinder-pick-place",
    "--root", "dataset",
    "--mode", "local",
    "--episode-index", episode
])