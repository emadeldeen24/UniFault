import os
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def convert_h5_to_parquet(h5_path, parquet_path, dataset_key="vibration_data"):
    """
    Convert a single HDF5 (.h5) file to Parquet format.

    Parameters:
        h5_path (Path): Path to the input HDF5 file.
        parquet_path (Path): Path where the output Parquet file will be saved.
        dataset_key (str): Name of the dataset inside the .h5 file to extract.
    """
    with h5py.File(h5_path, 'r') as h5_file:
        if dataset_key not in h5_file:
            raise KeyError(f"'{dataset_key}' not found in {h5_path}")

        data = h5_file[dataset_key][:]
        if not isinstance(data, np.ndarray) or data.ndim != 3:
            raise ValueError(f"Expected a 3D array [samples, channels, seq_len], got shape {data.shape} in {h5_path}")

        # Convert to pyarrow and write to disk
        array = pa.array(data.tolist())
        table = pa.table({'samples': array})
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)

def batch_convert_h5_to_parquet(source_root, target_root, dataset_key="vibration_data"):
    """
    Traverse a directory recursively and convert all .h5 files to .parquet.

    Parameters:
        source_root (Path or str): Root directory containing .h5 files.
        target_root (Path or str): Output root directory for .parquet files.
        dataset_key (str): Dataset key to extract from each .h5 file.
    """
    source_root = Path(source_root)
    target_root = Path(target_root)

    for root, _, files in os.walk(source_root):
        for fname in files:
            if fname.endswith('.h5'):
                h5_path = Path(root) / fname
                rel_path = h5_path.relative_to(source_root).with_suffix('.parquet')
                parquet_path = target_root / rel_path

                try:
                    convert_h5_to_parquet(h5_path, parquet_path, dataset_key)
                    print(f"[✓] Converted: {h5_path} -> {parquet_path}")
                except Exception as e:
                    print(f"[✗] Failed: {h5_path} | Reason: {e}")

if __name__ == "__main__":
    # Example usage: change these paths as needed
    SOURCE_DIR = r"path/to/data"
    TARGET_DIR = r"path/to/result/parquet_data"

    batch_convert_h5_to_parquet(SOURCE_DIR, TARGET_DIR)
