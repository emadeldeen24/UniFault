## 📦 General Dataset Preprocessing Pipeline

This repository contains a **general-purpose preprocessing pipeline** for any time-series or signal-based dataset with **normal/faulty** labeling. It prepares the data for machine learning or deep learning tasks by performing:

* Recursive dataset loading
* Subsequence extraction via sliding window
* Train/validation/test split
* Min-max normalization (based on training set)
* Export to efficient `.parquet` format

---

## 📁 Expected Input Directory Structure

The data directory should follow a hierarchical structure like:

```
root_data_dir/
  CategoryA/                # e.g., machine, device, location, etc.
    good/                   # Label for normal samples
      file1.parquet
      file2.parquet
    bad/                    # Label for faulty samples
      file3.parquet
  CategoryB/
    good/
    bad/
```

Each `.parquet` file must contain at least the following column:

* `samples`: a list/array of shape `[sequence_length, num_channels]`

### To get the data into .parquet format, run `convert_data_to_parquet.py` file.

---

## 🔧 Parameters & Customization

* `data_dir`: Path to the raw dataset directory
* `save_dir`: Output path for the preprocessed `.parquet` files
* `sequence_len`: Length of each sliding window segment
* `stride`: Step size for the sliding window
* `train_size`: Proportion of data used for training (default: `0.01`)
* `test_size`: Proportion of data used for testing (default: `0.90`)
* `good_label`: Name of the folder for normal data (default: `"good"`)
* `bad_label`: Name of the folder for faulty data (default: `"bad"`)

---

## 🚀 How to Use

### 1. Install Requirements

You must have Python 3 and the following libraries:

```bash
pip install torch pyarrow numpy
```

### 2. Run the Script

Edit the `__main__` section of `prepare_dataset.py`:

```python
if __name__ == "__main__":
    data_dir = r"/path/to/your/raw/data"
    save_dir = r"/path/to/save/prepared/data"
    sequence_len = 1024
    stride = 1024

    processor = PrepareDataset(data_dir, save_dir, sequence_len, stride)
    processor.prepare()
```

Then run:

```bash
python prepare_dataset.py
```

---

## 📤 Output Structure

```
save_dir/
  CategoryA/
    train.parquet
    val.parquet
    test.parquet
  CategoryB/
    ...
```

Each `.parquet` file contains:

* `samples`: Tensor segments of shape `[num_samples, num_channels, sequence_len]`
* `labels`: 0 for normal (good), 1 for faulty (bad)

---

## ✅ Notes

* The `"samples"` field in each `.parquet` file must be convertible to a numeric tensor.
* The data is normalized globally using only the training set.
