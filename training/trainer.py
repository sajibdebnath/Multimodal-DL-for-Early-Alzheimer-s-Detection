"""
Training engine for the MDLF framework.

Implements Algorithm 1 from the paper:
  - End-to-end training with differential learning rates
  - Adam optimizer with cosine annealing + warm restarts
  - Early stopping on validation accuracy
  - Checkpoint saving (best model)
  - Per-epoch logging of loss, accuracy, AUC
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models.mdlf import MDLF
from training.loss import WeightedCrossEntropyLoss, compute_class_weights


class Trainer:
    """
    Full training loop for MDLF.

    Parameters
    ----------
    model : MDLF instance
    train_loader / val_loader : DataLoaders
    cfg : dict loaded from YAML config
    device : torch.device
    checkpoint_dir : path to save model checkpoints
    log_dir : path to save training logs
    """

    def __init__(
        self,
        model: MDLF,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        t = cfg.get("training", cfg)

        # Compute class weights from training set labels
        all_labels = np.array(
            [batch["label"].numpy() for batch in train_loader],
            dtype=object,
        )
        all_labels = np.concatenate(all_labels)
        class_weights = compute_class_weights(all_labels, n_classes=3, device=str(device))
        self.criterion = WeightedCrossEntropyLoss(class_weights=class_weights)

        # Optimizer with differential learning rates
        param_groups = model.get_optimizer_param_groups(
            lr_main=t.get("lr_main", 1e-4),
            lr_backbone=t.get("lr_backbone", 1e-5),
            weight_decay=t.get("weight_decay", 1e-5),
        )
        self.optimizer = Adam(
            param_groups,
            betas=(t.get("adam_beta1", 0.9), t.get("adam_beta2", 0.999)),
            eps=t.get("adam_eps", 1e-8),
        )

        # Cosine annealing with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=t.get("t0", 10),
            T_mult=t.get("t_mult", 2),
            eta_min=t.get("eta_min", 1e-7),
        )

        self.max_epochs = t.get("max_epochs", 100)
        self.patience = t.get("early_stopping_patience", 15)

        # Tracking
        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "val_auc": [],
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0

    # ------------------------------------------------------------------ #
    # Training step
    # ------------------------------------------------------------------ #

    def _step(self, batch: dict, train: bool) -> tuple:
        mri      = batch["mri"].to(self.device)
        cog      = batch["cognitive"].to(self.device)
        life     = batch["lifestyle"].to(self.device)
        labels   = batch["label"].to(self.device)

        with torch.set_grad_enabled(train):
            out    = self.model(mri, cog, life)
            loss   = self.criterion(out["logits"], labels)
            preds  = out["probs"].argmax(dim=-1)
            probs  = out["probs"].detach().cpu().numpy()

        correct = (preds == labels).sum().item()
        return loss, correct, len(labels), probs, labels.cpu().numpy()

    # ------------------------------------------------------------------ #
    # One epoch
    # ------------------------------------------------------------------ #

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        total_loss, total_correct, total_samples = 0.0, 0, 0
        all_probs, all_labels = [], []

        desc = "Train" if train else "Val  "
        for batch in tqdm(loader, desc=desc, leave=False):
            loss, correct, n, probs, lbls = self._step(batch, train=train)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss    += loss.item() * n
            total_correct += correct
            total_samples += n
            all_probs.append(probs)
            all_labels.append(lbls)

        avg_loss = total_loss / total_samples
        acc      = total_correct / total_samples

        all_probs  = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        except ValueError:
            auc = 0.0

        return {"loss": avg_loss, "acc": acc, "auc": auc}

    # ------------------------------------------------------------------ #
    # Main training loop
    # ------------------------------------------------------------------ #

    def fit(self) -> Dict[str, list]:
        """
        Run training for up to max_epochs with early stopping.

        Returns
        -------
        history : dict of training curves
        """
        print(f"\n{'='*60}")
        print(f"  Starting MDLF training ({self.max_epochs} epochs max)")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()

            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics   = self._run_epoch(self.val_loader,   train=False)

            self.scheduler.step(epoch)

            # Record history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["acc"])
            self.history["val_auc"].append(val_metrics["auc"])

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"Train Loss {train_metrics['loss']:.4f} Acc {train_metrics['acc']*100:.2f}% | "
                f"Val Loss {val_metrics['loss']:.4f} Acc {val_metrics['acc']*100:.2f}% "
                f"AUC {val_metrics['auc']:.4f} | {elapsed:.1f}s"
            )

            # Checkpoint best model
            if val_metrics["acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["acc"]
                self.best_epoch   = epoch
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                self.epochs_no_improve += 1

            # Early stopping
            if self.epochs_no_improve >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch} "
                      f"(best val acc {self.best_val_acc*100:.2f}% at epoch {self.best_epoch})")
                break

        print(f"\nTraining complete. Best val accuracy: {self.best_val_acc*100:.2f}% at epoch {self.best_epoch}")
        self._restore_best_checkpoint()
        return self.history

    # ------------------------------------------------------------------ #
    # Checkpoint utilities
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, epoch: int, metrics: dict):
        path = self.checkpoint_dir / "best_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_acc": metrics["acc"],
                "val_auc": metrics["auc"],
            },
            path,
        )

    def _restore_best_checkpoint(self):
        path = self.checkpoint_dir / "best_model.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"Restored best checkpoint from epoch {ckpt['epoch']} "
                  f"(val acc {ckpt['val_acc']*100:.2f}%)")
