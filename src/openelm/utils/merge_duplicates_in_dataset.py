# This script merges duplicate x's in the dataset by averaging their corresponding y's.

import numpy as np
import os
import pathlib
import tqdm

ROOT_DIR = pathlib.Path(__file__).resolve().parents[3]
SOURCE_DIR = os.path.join(ROOT_DIR, "design-bench-detached", "design_bench_data", "tf_bind_10-pho4")  # original data folder
TARGET_DIR = os.path.join(ROOT_DIR, "design-bench-detached", "design_bench_data", "tf_bind_10-pho4-no-duplicates")  # output folder
os.makedirs(TARGET_DIR, exist_ok=True)

# --- Step 1: Collect all x and y ---
x_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.startswith('tf_bind_10-x-') and f.endswith('.npy')])
y_files = sorted([f for f in os.listdir(SOURCE_DIR) if f.startswith('tf_bind_10-y-') and f.endswith('.npy')])
assert len(x_files) == len(y_files), "Mismatch between number of x and y files"

xs = [np.load(os.path.join(SOURCE_DIR, f)) for f in x_files]
ys = [np.load(os.path.join(SOURCE_DIR, f)) for f in y_files]
x_all = np.concatenate(xs, axis=0)
y_all = np.concatenate(ys, axis=0)
print(f"Loaded {x_all.shape[0]} samples.")

# --- Step 2: Merge duplicate x's by averaging y ---
# Reshape if y is 1D (make sure shape is [N, 1])
if y_all.ndim == 1:
    y_all = y_all[:, None]

# Trick: Make x hashable for np.unique
x_view = x_all.view([('', x_all.dtype)] * x_all.shape[1])
_, unique_indices, inverse_indices = np.unique(x_view, return_index=True, return_inverse=True, axis=0)

print(f"Found {len(unique_indices)} unique x's out of {x_all.shape[0]} samples.")

# Merge y's corresponding to the same x
merged_x = x_all[unique_indices]
merged_y = np.zeros((len(unique_indices), y_all.shape[1]), dtype=y_all.dtype)
for i in tqdm.tqdm(range(len(unique_indices)), desc="Merging y's"):
    merged_y[i] = y_all[inverse_indices == i].mean(axis=0)

print(f"After deduplication: {merged_x.shape[0]} unique x's")

# --- Step 3: Save as new shards ---
# You can change SHARD_SIZE to control how many samples per file.
SHARD_SIZE = 1048576  # 4**10

n = merged_x.shape[0]
num_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE

for i in range(num_shards):
    start = i * SHARD_SIZE
    end = min((i + 1) * SHARD_SIZE, n)
    np.save(os.path.join(TARGET_DIR, f"tf_bind_10-x-{i}.npy"), merged_x[start:end])
    np.save(os.path.join(TARGET_DIR, f"tf_bind_10-y-{i}.npy"), merged_y[start:end])
    print(f"Saved shard {i}: {end-start} samples")

print(f"Deduplicated data saved to {TARGET_DIR}")
