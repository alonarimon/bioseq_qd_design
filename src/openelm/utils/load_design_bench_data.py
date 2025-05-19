#todo: this is a debugging file that need to be deleted.

# download the data from huggingface hub

import os
import pathlib
from huggingface_hub import snapshot_download

project_root = pathlib.Path(__file__).resolve().parents[3]
DB_DATA_DIR = os.path.join(project_root, "design-bench-detached", "design_bench_data")
print(f"Downloading design bench data to {DB_DATA_DIR}")

DB_HF_DATASET = os.environ.get("DB_HF_REPO", "beckhamc/design_bench_data")
snapshot_download(DB_HF_DATASET, repo_type="dataset", local_dir=DB_DATA_DIR)