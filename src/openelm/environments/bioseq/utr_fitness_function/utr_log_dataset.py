import os

import torch
from torch.utils.data import Dataset

from preprocess import sequence_nuc_to_one_hot, log_interpolated_one_hot


class UTRLogDataset(Dataset):
    def __init__(self, int_sequences, float_scores, disk_target="one_hot_log_data", file_name='one_hot_log_data.pt'):
        # check if data exists already in disk_target
        disk_file_target = os.path.join(disk_target, file_name)

        if os.path.exists(disk_file_target):
            print(f"Loading data from {disk_file_target}")
            data = torch.load(disk_file_target)
            self.data_x = data["x"]
            self.data_y = data["y"]
            print("Data loaded from disk.")
            # sanity check - make sure the data loaded is the same as the one passed
            assert self.data_x.shape == (len(int_sequences), int_sequences[0].shape[0], 4 - 1)
            assert self.data_y.shape == (len(float_scores),)
            print("loaded y[0:1]:", self.data_y[0:1])
            print("loaded x[0:1]:", self.data_x[0:1])
            print("given y[0:1]:", float_scores[0:1])
            print("given x[0:1]:", int_sequences[0:1])


        else:
            print(f"Data not found in {disk_file_target}. Creating new dataset.")
            os.makedirs(disk_target, exist_ok=True)
            one_hot = sequence_nuc_to_one_hot(torch.tensor(int_sequences))  # (N, L, K)
            self.data_x = log_interpolated_one_hot(one_hot)                 # (N, L, K-1)
            self.data_y = torch.tensor(float_scores).float()
            # save to disk
            file_name = os.path.join(disk_target, file_name)
            torch.save({"x": self.data_x, "y": self.data_y}, file_name)
            print("Data saved to disk.")

        print("Data shapes:",
              "x:", self.data_x.shape,  # (N, L, K-1)
              "y:", self.data_y.shape)


    def __len__(self):
        return len(self.data_y)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]
