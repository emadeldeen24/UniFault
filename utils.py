from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import datetime
import torch
import os
import shutil
import inspect
import argparse
import numpy as np


def save_copy_of_files(checkpoint_callback, project_root=None):
    caller_frame = inspect.currentframe().f_back
    caller_script = os.path.abspath(caller_frame.f_globals["__file__"])

    if project_root is None:
        project_root = os.path.dirname(caller_script)
    project_root = os.path.abspath(project_root)

    destination_directory = checkpoint_callback.dirpath
    os.makedirs(destination_directory, exist_ok=True)

    visited = set()

    def resolve_module(module_name):
        try:
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin and spec.origin.endswith(".py"):
                path = os.path.abspath(spec.origin)
                if path.startswith(project_root):
                    return path
        except Exception:
            pass
        return None

    def find_imports(file_path):
        imported_files = set()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=file_path)
        except Exception:
            return imported_files

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    path = resolve_module(alias.name)
                    if path:
                        imported_files.add(path)
            elif isinstance(node, ast.ImportFrom) and node.module:
                path = resolve_module(node.module)
                if path:
                    imported_files.add(path)
        return imported_files

    def recursive_collect(file_path):
        if file_path in visited:
            return
        visited.add(file_path)
        for imported in find_imports(file_path):
            recursive_collect(imported)

    recursive_collect(caller_script)

    for file_path in visited:
        rel_path = os.path.relpath(file_path, project_root)
        dest_path = os.path.join(destination_directory, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(file_path, dest_path)

    print(f"Copied {len(visited)} files to {destination_directory}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)
        similarity_matrix = self.similarity_function(representations, representations)

        # Ensure the mask is on the same device as similarity_matrix
        mask = self.mask_samples_from_same_repr.to(similarity_matrix.device)

        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[mask].view(2 * self.batch_size, -1)
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(similarity_matrix.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
