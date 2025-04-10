#todo: this is a debugging file that need to be deleted.

# download the data from huggingface hub

import os

from huggingface_hub import snapshot_download

DB_HF_DATASET = os.environ.get("DB_HF_REPO", "beckhamc/design_bench_data")
DB_DATA_DIR = r"C:\Users\Alona\anaconda3\envs\OpenELM_GenomicQD\Lib\site-packages\design_bench_data"
snapshot_download(DB_HF_DATASET, repo_type="dataset", local_dir=DB_DATA_DIR)