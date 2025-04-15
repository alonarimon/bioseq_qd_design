import os

import numpy as np

utr_datapath = r"C:\Users\Alona\Desktop\Imperial_college_london\MSc_project_code\OpenELM_GenomicQD\design-bench_forked\design_bench_data\utr"
# load utr data (from files utr-x-0.npy, utr-x-1.npy, ..., utr-x-9.npy and utr-y-0.npy, utr-y-1.npy, ..., utr-y-9.npy)
x_files = sorted([f for f in os.listdir(utr_datapath) if f.startswith("utr-x")])
y_files = sorted([f for f in os.listdir(utr_datapath) if f.startswith("utr-y")])

# Load and concatenate
X_all = np.concatenate([np.load(os.path.join(utr_datapath, f)) for f in x_files], axis=0)
Y_all = np.concatenate([np.load(os.path.join(utr_datapath, f)) for f in y_files], axis=0)
print("X_all shape:", X_all.shape)
print("Y_all shape:", Y_all.shape)
print("X_all:", X_all[:3])
print("Y_all:", Y_all[:3])
# y stats
print("y stats (before normalization):")
print("min:", Y_all.min(), "max:", Y_all.max(), "mean:", Y_all.mean(), "std:", Y_all.std())

# Normalize Y using min-max normalization
Y_min = Y_all.min()
Y_max = Y_all.max()
Y_range = Y_max - Y_min
Y_all_normalized_minmax = (Y_all - Y_min) / Y_range
print("Y_all_normalized_minmax shape:", Y_all_normalized_minmax.shape)
print("Y_all_normalized_minmax:", Y_all_normalized_minmax[:3])
# y stats
print("y stats (after min-max normalization):")
print("min:", Y_all_normalized_minmax.min(), "max:", Y_all_normalized_minmax.max(), "mean:", Y_all_normalized_minmax.mean(), "std:", Y_all_normalized_minmax.std())

# Save normalized data
normalized_minmax_path = os.path.join(utr_datapath, "normalized_minmax_Y")
os.makedirs(normalized_minmax_path, exist_ok=True)
np.save(os.path.join(normalized_minmax_path, "utr-y.npy"), Y_all_normalized_minmax)
np.save(os.path.join(normalized_minmax_path, "utr-x.npy"), X_all)