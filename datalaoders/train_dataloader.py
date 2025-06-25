import glob

import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
import pyarrow.parquet as pq
from collections import defaultdict
import re
import random

class PHMDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, args, data_type):
        super(PHMDataset, self).__init__()

        if args.data_percentage == "100" or data_type != "train":
            data_file = os.path.join(args.data_path, args.data_id, f"{data_type}.parquet")
            print(f"Loading full {data_type} set of {os.path.basename(os.path.dirname(data_file))} data ...")

        elif data_type == "train" and "shot" in args.data_percentage:
            data_file = os.path.join(args.data_path, args.data_id, f"{data_type}.parquet")
            print(f"Loading full {data_type} set ... now preparing the few-shot samples ...")

        else:
            data_file = os.path.join(args.data_path, args.data_id, f"{data_type}_{args.data_percentage}p.parquet")
            print(f"Loading only {args.data_percentage}% from {data_type} set of {os.path.basename(os.path.dirname(data_file))} data ...")

        # Read .parquet data
        data_file = pq.read_table(data_file)

        # Extract the samples and labels
        x_np_list = data_file['samples'].to_pylist()
        y_np = data_file['labels'].to_numpy() if 'labels' in data_file.column_names else None

        x_np = np.array(x_np_list)  # Expected dimension: [num_samples, num_channels, seq_length]

        if data_type == "train" and "shot" in args.data_percentage:
            x_np, y_np = self.extract_few_shot_samples(x_np, y_np, args.data_percentage)
            print("Extracted few-shots ...")

        x_data = torch.tensor(x_np)
        y_data = torch.tensor(y_np) if y_np is not None else None

        print(f"data shapes: {x_data.shape}, {y_data.shape}")

        # Update class attributes
        x_data = x_data.to(torch.bfloat16)
        self.x_data = x_data.float()
        self.y_data = y_data if y_data is not None else None
        print("================")
        self.len = x_data.shape[0]

    def extract_few_shot_samples(self, x, y, data_percentage):
        """
        Apply few-shot sampling to the dataset.
        `args.data_percentage` is treated as the x-shot value.
        """
        match = re.search(r"(\d+)shot", data_percentage)
        shots = int(match.group(1)) if match else None
        if shots is None:
            raise ValueError(f"Invalid data_percentage format for few-shot: {self.args.data_percentage}")

        grouped = defaultdict(list)

        # Group samples by their labels
        for sample, label in zip(x, y):
            grouped[label.item()].append(sample)

        # Collect few-shot samples
        few_shot_samples = []
        few_shot_labels = []
        for label, samples in grouped.items():
            # Randomly select k-shots samples
            index = random.sample(range(0,len(samples)),shots)
            selected_samples = [samples[i] for i in index]
            # selected_samples = samples[:shots]  # Select up to `shots` samples per class
            few_shot_samples.extend(selected_samples)
            few_shot_labels.extend([label] * len(selected_samples))

        # Update dataset with few-shot samples
        x_few = np.array(few_shot_samples)
        y_few = np.array(few_shot_labels)
        return x_few, y_few

    def __getitem__(self, index):
        return self.x_data[index].squeeze(-1), self.y_data[index]

    def __len__(self):
        return self.len

def get_datasets(args):

    train_dataset = PHMDataset(args,  data_type='train')
    val_dataset = PHMDataset(args, data_type='val')
    test_dataset = PHMDataset(args, data_type='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_single_dataset(args, data_type='train'):
    dataset = PHMDataset(args, data_type=data_type)
    shuffle = True if data_type=='train' else False
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
    return data_loader

