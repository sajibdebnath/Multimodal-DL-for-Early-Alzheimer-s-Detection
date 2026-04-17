"""General utility helpers for the MDLF framework."""

from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility (paper: seed=42)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def get_device(no_cuda: bool = False) -> torch.device:
    if torch.cuda.is_available() and not no_cuda:
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_metrics_table(results: Dict[str, Dict]) -> str:
    """Format ablation results as a markdown-style table."""
    header = f"| {'Configuration':<42} | {'Acc (%)':>8} | {'AUC':>7} | {'F1 (%)':>8} |"
    sep    = f"|{'-'*44}|{'-'*10}|{'-'*9}|{'-'*10}|"
    rows   = [header, sep]
    for name, m in results.items():
        rows.append(
            f"| {name:<42} | {m['accuracy']*100:>7.2f}% | "
            f"{m['macro_auc']:>7.4f} | {m['macro_f1']*100:>7.2f}% |"
        )
    return "\n".join(rows)
