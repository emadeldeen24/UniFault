import os
import argparse
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import torch
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from torchmetrics.classification import Accuracy, MulticlassF1Score, MulticlassConfusionMatrix

from datalaoders.train_dataloader import get_datasets
from model.model import Transformer_bkbone
from utils import save_copy_of_files, str2bool

# ==================== Model Wrapper ====================
class Model(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Transformer_bkbone(args)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train_metrics = MetricCollection({
            "acc": Accuracy(task="multiclass", num_classes=args.num_classes),
            "f1": MulticlassF1Score(num_classes=args.num_classes, average="macro")
        })

        self.val_metrics = MetricCollection({
            "acc": Accuracy(task="multiclass", num_classes=args.num_classes),
            "f1": MulticlassF1Score(num_classes=args.num_classes, average="macro")
        })

        self.test_f1 = MulticlassF1Score(num_classes=args.num_classes, average="macro")
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=args.num_classes)

        self.num_warmup_steps = 2048
        self.total_steps = args.num_epochs * args.tl_length

        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=args.lr, weight_decay=args.wt_decay)

        scheduler = {
            'scheduler': self.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.num_warmup_steps,
                                                              num_training_steps=self.total_steps),
            'name': 'learning_rate', 'interval': 'step', 'frequency': 1,
        }
        return [optimizer], [scheduler]


    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _shared_step(self, batch, stage):
        x, y = batch
        if y.ndim > 1 and y.size(1) > 1:
            y = torch.argmax(y, dim=1)

        logits = self(x)
        preds = self.model.predict(logits)
        loss = self.loss_fn(preds, y)

        if stage == "train":
            metrics = self.train_metrics.clone()
            metrics.update(preds, y)
            self.log_dict({f"train_{k}": v for k, v in metrics.compute().items() if k != "loss"}, on_epoch=True, prog_bar=True)
            self.log("train_loss", loss, on_epoch=True)
        elif stage == "val":
            metrics = self.val_metrics.clone()
            metrics.update(preds, y)
            self.log_dict({f"val_{k}": v for k, v in metrics.compute().items() if k != "loss"}, on_epoch=True, prog_bar=True)
            self.log("val_loss", loss, on_epoch=True)
        elif stage == "test":
            self.test_f1.update(preds, y)
            self.confusion_matrix.update(preds, y)
            self.test_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
            self.test_targets.extend(y.cpu().numpy())
            self.log("test_loss", loss)

            # Move accuracy metric to same device as predictions
            test_accuracy_metric = Accuracy(task="multiclass", num_classes=self.args.num_classes).to(preds.device)
            acc = test_accuracy_metric(preds, y)
            self.log("test_accuracy", acc)

        return loss


    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        f1_score = self.test_f1.compute()
        self.log("test_f1", f1_score)
        self.test_f1.reset()

        fig, ax = self.confusion_matrix.plot()
        fig.tight_layout()
        fig.savefig(f"{self.args.ckpt_dir}/confusion_matrix.png", bbox_inches="tight")
        print("Test Confusion Matrix saved.")
        self.confusion_matrix.reset()

        report = classification_report(
            self.test_targets,
            self.test_preds,
            target_names=self.args.class_names,
            digits=4
        )
        print("=== Classification Report ===")
        print(report)

        with open(f"{self.args.ckpt_dir}/classification_report.txt", "w") as f:
            f.write(report)

        self.test_preds = []
        self.test_targets = []

# ==================== Callbacks ====================
def construct_experiment_dir(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_description = "FT" if args.load_from_pretrained else "Supervised"
    run_description += f"_{args.model_type}"
    run_description += f"_{args.data_id}_from{args.pretraining_epoch_id}_{args.model_id}"
    run_description += f"_bs{args.batch_size}_seed{args.random_seed}_{timestamp}"
    return run_description


def plot_metrics(metrics, ckpt_dir):
    plt.figure()
    plt.plot(metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(f"{ckpt_dir}/loss.png", bbox_inches="tight")

    plt.figure()
    plt.plot(metrics["train_acc"], label="Train Acc")
    plt.plot(metrics["val_acc"], label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.tight_layout()
    plt.savefig(f"{ckpt_dir}/accuracy.png", bbox_inches="tight")


class MetricTrackerCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.losses = {"train_loss": [], "val_loss": []}
        self.accuracies = {"train_acc": [], "val_acc": []}

    def on_validation_epoch_end(self, trainer, pl_module):
        self.losses["val_loss"].append(trainer.callback_metrics["val_loss"].item())
        self.accuracies["val_acc"].append(trainer.callback_metrics["val_acc"].item())

    def on_train_epoch_end(self, trainer, pl_module):
        self.losses["train_loss"].append(trainer.callback_metrics["train_loss"].item())
        self.accuracies["train_acc"].append(trainer.callback_metrics["train_acc"].item())

# ==================== Main ====================
def main(args):
    pl.seed_everything(args.random_seed)
    train_loader, val_loader, test_loader = get_datasets(args)

    # args extracted from the running dataset
    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.class_names = [str(i) for i in range(args.num_classes)]
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]
    args.tl_length = len(train_loader)

    # Callbacks
    run_description = construct_experiment_dir(args)
    print(f"========== {run_description} ===========")
    ckpt_dir = f"checkpoints/{run_description}"

    args.ckpt_dir = ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(monitor="train_f1_epoch", mode="max", save_top_k=1, dirpath=ckpt_dir, filename="best")
    early_stop = EarlyStopping(monitor="train_f1_epoch", patience=args.patience, mode="max")
    tracker = MetricTrackerCallback()

    save_copy_of_files(checkpoint)

    model = Model(args)

    # Optional load pretrained weights
    if args.load_from_pretrained:
        path = os.path.join(args.pretrained_model_dir, f"pretrain-epoch={args.pretraining_epoch_id}.ckpt")
        checkpoint_data = torch.load(path, map_location='cuda')

        # Filter and count matching keys with the same shape
        matched_weights = {
            k: v for k, v in checkpoint_data['state_dict'].items()
            if k in model.state_dict() and model.state_dict()[k].size() == v.size()
        }

        total_pretrained = len(checkpoint_data['state_dict'])
        model.load_state_dict(matched_weights, strict=False)

        print(f"Loaded pretrained weights from {path}")
        print(f"Matched weights: {len(matched_weights)}/{len(model.state_dict())} model parameters matched "
              f"(from {total_pretrained} pretrained parameters)")

    trainer = pl.Trainer(
        default_root_dir=ckpt_dir,
        max_epochs=args.num_epochs,
        callbacks=[checkpoint, early_stop, tracker, TQDMProgressBar(refresh_rate=500)],
        accelerator="auto",
        precision='bf16-mixed',
        devices=[args.gpu_id],
        num_sanity_val_steps=0,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")

    plot_metrics(
        {"train_loss": tracker.losses["train_loss"], "val_loss": tracker.losses["val_loss"],
         "train_acc": tracker.accuracies["train_acc"], "val_acc": tracker.accuracies["val_acc"]},
        args.ckpt_dir
    )


def apply_model_config(args):
    config_map = {
        'tiny':  {'embed_dim': 128, 'heads': 4,  'depth': 4,  'pretrained_model_dir': 'pretrained_models/Tiny'},
        'small': {'embed_dim': 256, 'heads': 8,  'depth': 8,  'pretrained_model_dir': 'pretrained_models/Small'},
        'base':  {'embed_dim': 512, 'heads': 12, 'depth': 16, 'pretrained_model_dir': 'pretrained_models/Base'},
    }
    config = config_map[args.model_type]
    for k, v in config.items():
        setattr(args, k, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'path/to/data', help="Path to all datasets")
    parser.add_argument('--data_id', type=str, default=r'IMS', help="specific dataset inside the parent data dir")
    parser.add_argument('--data_percentage', type=str, default="1")
    parser.add_argument('--model_id', type=str, default="Description_of_experiment")

    parser.add_argument('--model_type', type=str, choices=['tiny', 'small', 'base'], default='tiny')
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--load_from_pretrained', type=str2bool, default=True)
    parser.add_argument('--pretraining_epoch_id', type=int, default=1)

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=50, help="For early stopping")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--wt_decay', type=float, default=1e-4)
    parser.add_argument('--random_seed', type=int, default=42)


    args = parser.parse_args()
    apply_model_config(args)
    main(args)
