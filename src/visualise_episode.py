import sys
import shutil
import subprocess
from pathlib import Path

episode = sys.argv[1] if len(sys.argv) > 1 else "0"
print(f"Visualising episode {episode}")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
VISUALISE_SCRIPT = PROJECT_ROOT / "lerobot" / "src" / "lerobot" / "scripts" / "visualize_dataset.py"


# Clear cache before launching visualiser
cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print(f"Cache cleared.")

# Launch visualiser
subprocess.run(
    [sys.executable, str(VISUALISE_SCRIPT),
     "--repo-id", "local/xarm7_insert",
     "--root", str(DATASET_ROOT),
     "--episode-index", episode,
     "--mode", "local"],
    cwd=str(PROJECT_ROOT / "lerobot"),
)