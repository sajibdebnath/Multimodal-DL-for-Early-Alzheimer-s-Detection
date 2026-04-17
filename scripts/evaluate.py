#!/usr/bin/env python3
"""
Standalone evaluation script — loads a saved checkpoint and reports all metrics.

Usage:
  python scripts/evaluate.py --checkpoint checkpoints/best_model.pt \\
                              --config configs/default_config.yaml \\
                              --synthetic
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import yaml

from data.dataset import build_dataloaders, stratified_split
from data.preprocessing import (
    preprocess_cognitive_data, preprocess_lifestyle_data,
    generate_synthetic_dataset,
)
from models.mdlf import build_mdlf_from_config
from evaluation.metrics import evaluate_model, compute_metrics, print_metrics
from evaluation.interpretability import run_shap_analysis, plot_attention_weights
from utils.visualization import (
    plot_confusion_matrix, plot_roc_curves, plot_pr_curves,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/default_config.yaml")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--data_root",  default=None)
    p.add_argument("--run_shap",   action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    if args.synthetic:
        meta_df = generate_synthetic_dataset(n_subjects=1900, random_seed=args.seed,
                                             output_dir="data/synthetic")
    else:
        csv = os.path.join(args.data_root or "data/processed", "metadata.csv")
        meta_df = pd.read_csv(csv)

    from data.preprocessing import COGNITIVE_FEATURES
    for col in COGNITIVE_FEATURES:
        if col not in meta_df.columns:
            meta_df[col] = 0.0

    _, _, test_df = stratified_split(meta_df, random_seed=args.seed)
    train_df, val_df, _ = stratified_split(meta_df, random_seed=args.seed)

    cog_train, scaler, imputer = preprocess_cognitive_data(train_df, fit_scaler=True)
    cog_test, _, _ = preprocess_cognitive_data(test_df, False, scaler, imputer)
    cog_val,  _, _ = preprocess_cognitive_data(val_df,  False, scaler, imputer)

    def load_life(df):
        seqs = []
        for _, row in df.iterrows():
            p = row.get("LifestylePath", "")
            seqs.append(np.load(p) if os.path.exists(str(p)) else np.zeros((12,8),np.float32))
        return np.stack(seqs)

    life_tr, lm, ls = preprocess_lifestyle_data(load_life(train_df), fit_stats=True)
    life_va, _, _   = preprocess_lifestyle_data(load_life(val_df),  False, lm, ls)
    life_te, _, _   = preprocess_lifestyle_data(load_life(test_df), False, lm, ls)

    loaders = build_dataloaders(train_df, val_df, test_df,
                                cog_train, cog_val, cog_test,
                                life_tr, life_va, life_te, batch_size=32)

    # Load model
    model = build_mdlf_from_config(cfg)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val acc {ckpt.get('val_acc',0)*100:.2f}%)")

    # Evaluate
    results = evaluate_model(model, loaders["test"], device, return_attention=True)
    metrics = compute_metrics(results["all_labels"], results["all_preds"], results["all_probs"])
    print_metrics(metrics, title="Test Set Evaluation")

    fig_dir = cfg["paths"]["figures_dir"]
    plot_confusion_matrix(results["all_labels"], results["all_preds"],
                          output_path=os.path.join(fig_dir, "confusion_matrix.png"))
    plot_roc_curves(results["all_labels"], results["all_probs"], output_dir=fig_dir)
    plot_pr_curves( results["all_labels"], results["all_probs"], output_dir=fig_dir)

    if "all_alpha" in results:
        plot_attention_weights(results["all_alpha"], results["all_labels"],
                               output_dir=fig_dir)

    # Optional SHAP
    if args.run_shap:
        print("\nRunning SHAP analysis (may take a few minutes)...")
        run_shap_analysis(
            model=model,
            test_cog=cog_test,
            test_life=life_te,
            test_labels=results["all_labels"],
            device=device,
            output_dir=fig_dir,
        )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
