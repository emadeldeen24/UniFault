import os
import torch
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import pyarrow as pa

class PrepareDataset:
    def __init__(self, data_dir, save_dir, sequence_len, stride, train_size=0.01, test_size=0.9, good_label="good", bad_label="bad"):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.sequence_len = sequence_len
        self.stride = stride
        self.train_size = train_size
        self.test_size = test_size
        self.good_label = good_label
        self.bad_label = bad_label

    def prepare(self):
        # Search for all category folders (e.g., machines, devices, etc.)
        for category_path in self.data_dir.iterdir():
            if not category_path.is_dir():
                continue
            self.prepare_category(category_path)

    def prepare_category(self, category_path):
        category_name = category_path.name
        print(f"Preparing category: {category_name}")

        healthy_data, faulty_data = self.load_signals(category_path)

        # Train/test/val split
        healthy_train, healthy_val, healthy_test = self.split_dataset(healthy_data)
        faulty_train, faulty_val, faulty_test = self.split_dataset(faulty_data)

        # Normalize using combined train data
        train_x = torch.cat((healthy_train["x"], faulty_train["x"]), dim=0)
        min_val = train_x.amin(dim=(0, 2), keepdim=True)
        max_val = train_x.amax(dim=(0, 2), keepdim=True)

        # Normalize all sets
        for dset in [healthy_train, healthy_val, healthy_test, faulty_train, faulty_val, faulty_test]:
            dset["x"] = self.normalize(dset["x"], min_val, max_val)

        # Merge and save
        self.save_split("train", category_name, healthy_train, faulty_train)
        self.save_split("val", category_name, healthy_val, faulty_val)
        self.save_split("test", category_name, healthy_test, faulty_test)

    def load_signals(self, category_path):
        healthy = []
        faulty = []

        for label_name in [self.good_label, self.bad_label]:
            label_dir = category_path / label_name
            if not label_dir.exists():
                continue

            label = 0 if label_name == self.good_label else 1
            for file in label_dir.rglob("*.parquet"):
                try:
                    data = self.load_parquet(file)
                    subsampled = self.subsample(data)
                    if label == 0:
                        healthy.append(subsampled)
                    else:
                        faulty.append(subsampled)
                except Exception as e:
                    print(f"Error loading {file}: {e}")

        return (
            {"x": torch.cat(healthy, dim=0), "y": torch.zeros(len(healthy), dtype=torch.long)},
            {"x": torch.cat(faulty, dim=0), "y": torch.ones(len(faulty), dtype=torch.long)}
        )

    def load_parquet(self, path):
        table = pq.read_table(path)
        data = table["samples"].to_pylist()
        return torch.tensor(np.array(data))  # [seq_len, channels]

    def subsample(self, signal):
        signal = signal.unsqueeze(0).permute(0, 2, 1)  # [1, C, L]
        x = signal.unfold(-1, self.sequence_len, self.stride)
        x = x.permute(0, 2, 1, 3).contiguous().view(-1, signal.shape[1], self.sequence_len)
        return x

    def split_dataset(self, data):
        x, y = data["x"], data["y"]
        indices = torch.randperm(x.size(0))
        x, y = x[indices], y[indices]

        total = x.size(0)
        train_end = int(self.train_size * total)
        test_end = train_end + int(self.test_size * total)

        return (
            {"x": x[:train_end], "y": y[:train_end]},
            {"x": x[train_end:test_end], "y": y[train_end:test_end]},
            {"x": x[test_end:], "y": y[test_end:]}
        )

    def normalize(self, x, min_val, max_val):
        return (x - min_val) / (max_val - min_val + 1e-6)

    def save_split(self, split_name, category, healthy, faulty):
        out_dir = self.save_dir / category
        out_dir.mkdir(parents=True, exist_ok=True)

        samples = torch.cat((healthy["x"].squeeze(1), faulty["x"].squeeze(1)), dim=0)
        labels = torch.cat((healthy["y"], faulty["y"]), dim=0)

        table = pa.table({
            "samples": pa.array(samples.tolist()),
            "labels": pa.array(labels.tolist())
        })

        pq.write_table(table, out_dir / f"{split_name}.parquet")


if __name__ == "__main__":
    data_dir = r""
    save_dir = r""
    sequence_len = 1024
    stride = 1024

    processor = PrepareDataset(data_dir, save_dir, sequence_len, stride)
    processor.prepare()
