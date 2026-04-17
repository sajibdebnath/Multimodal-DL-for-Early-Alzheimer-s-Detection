"""
Evaluation metrics for the MDLF framework.

Metrics reported in the paper (§Evaluation Protocol):
  - Overall classification accuracy
  - Per-class Precision, Recall, F1-Score, Specificity
  - Macro-average AUC (ROC)
  - Bootstrap 95% CI for accuracy (1000 resamples)
  - McNemar's test with Bonferroni correction for pairwise comparisons
  - Confusion matrix
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


CLASS_NAMES = ["CN", "MCI", "AD"]


# ---------------------------------------------------------------------------
# Core evaluation routine
# ---------------------------------------------------------------------------

def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_attention: bool = False,
) -> Dict:
    """
    Run inference on a DataLoader and collect all outputs.

    Returns
    -------
    results : dict with keys:
        all_labels, all_preds, all_probs, all_alpha (if requested)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    all_alpha = [] if return_attention else None

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            mri  = batch["mri"].to(device)
            cog  = batch["cognitive"].to(device)
            life = batch["lifestyle"].to(device)
            lbl  = batch["label"]

            out = model(mri, cog, life)
            preds = out["probs"].argmax(dim=-1).cpu().numpy()

            all_labels.append(lbl.numpy())
            all_preds.append(preds)
            all_probs.append(out["probs"].cpu().numpy())

            if return_attention:
                all_alpha.append(out["alpha"].cpu().numpy())

    results = {
        "all_labels": np.concatenate(all_labels),
        "all_preds":  np.concatenate(all_preds),
        "all_probs":  np.concatenate(all_probs),
    }
    if return_attention:
        results["all_alpha"] = np.concatenate(all_alpha)
    return results


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: List[str] = None,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> Dict:
    """
    Compute the full set of metrics as reported in the paper (Table 4).

    Parameters
    ----------
    labels : (N,) ground-truth integer class indices
    preds  : (N,) predicted integer class indices
    probs  : (N, K) predicted class probabilities
    n_bootstrap : number of bootstrap resamples for 95% CI

    Returns
    -------
    metrics : dict
    """
    if class_names is None:
        class_names = CLASS_NAMES

    accuracy = accuracy_score(labels, preds)

    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    # Specificity per class = TN / (TN + FP)
    cm = confusion_matrix(labels, preds)
    specificity_per_class = []
    for k in range(len(class_names)):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_per_class.append(spec)
    macro_specificity = np.mean(specificity_per_class)

    # AUC
    try:
        macro_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        per_class_auc = roc_auc_score(
            labels, probs, multi_class="ovr", average=None
        )
    except ValueError:
        macro_auc = float("nan")
        per_class_auc = [float("nan")] * len(class_names)

    # Average precision (macro)
    try:
        macro_ap = average_precision_score(
            np.eye(len(class_names))[labels], probs, average="macro"
        )
    except Exception:
        macro_ap = float("nan")

    # Bootstrap 95% CI for accuracy
    rng = np.random.default_rng(random_seed)
    boot_accs = []
    n = len(labels)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_accs.append(accuracy_score(labels[idx], preds[idx]))
    ci_low = np.percentile(boot_accs, 2.5)
    ci_high = np.percentile(boot_accs, 97.5)

    return {
        "accuracy":           accuracy,
        "macro_precision":    prec,
        "macro_recall":       rec,
        "macro_f1":           f1,
        "macro_specificity":  macro_specificity,
        "macro_auc":          macro_auc,
        "macro_ap":           macro_ap,
        "ci_95":              (ci_low, ci_high),
        "confusion_matrix":   cm,
        "per_class": {
            class_names[k]: {
                "precision":   float(per_class_prec[k]),
                "recall":      float(per_class_rec[k]),
                "f1":          float(per_class_f1[k]),
                "specificity": float(specificity_per_class[k]),
                "auc":         float(per_class_auc[k]) if not np.isnan(macro_auc) else float("nan"),
            }
            for k in range(len(class_names))
        },
        "classification_report": classification_report(
            labels, preds, target_names=class_names
        ),
    }


def print_metrics(metrics: Dict, title: str = "Test Set Results") -> None:
    """Pretty-print the metrics table."""
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(f"  Accuracy:      {metrics['accuracy']*100:.2f}%  "
          f"(95% CI [{metrics['ci_95'][0]*100:.1f}%, {metrics['ci_95'][1]*100:.1f}%])")
    print(f"  Macro Prec:    {metrics['macro_precision']*100:.2f}%")
    print(f"  Macro Recall:  {metrics['macro_recall']*100:.2f}%")
    print(f"  Macro F1:      {metrics['macro_f1']*100:.2f}%")
    print(f"  Macro Spec:    {metrics['macro_specificity']*100:.2f}%")
    print(f"  Macro AUC:     {metrics['macro_auc']:.4f}")
    print(f"  Macro AP:      {metrics['macro_ap']:.4f}")
    print()
    print(f"  {'Class':<8} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print(f"  {'-'*40}")
    for cls, m in metrics["per_class"].items():
        print(f"  {cls:<8} {m['precision']*100:>6.2f}% {m['recall']*100:>6.2f}% "
              f"{m['f1']*100:>6.2f}% {m['auc']:>7.4f}")
    print(f"{'='*65}\n")
    print(metrics["classification_report"])


# ---------------------------------------------------------------------------
# McNemar's test (for pairwise model comparisons)
# ---------------------------------------------------------------------------

def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    alpha: float = 0.05,
    n_comparisons: int = 10,
) -> Dict:
    """
    McNemar's test with Bonferroni correction (paper §Evaluation Protocol).
    Tests whether models A and B have significantly different error rates.

    Parameters
    ----------
    alpha : family-wise error rate
    n_comparisons : number of pairwise tests for Bonferroni correction

    Returns
    -------
    dict with 'statistic', 'p_value', 'corrected_alpha', 'significant'
    """
    from scipy.stats import chi2

    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    b = np.sum(correct_a & ~correct_b)   # A correct, B wrong
    c = np.sum(~correct_a & correct_b)   # A wrong, B correct

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    # McNemar's chi-squared (with continuity correction)
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1.0 - chi2.cdf(stat, df=1)

    corrected_alpha = alpha / n_comparisons  # Bonferroni

    return {
        "statistic": stat,
        "p_value":   p_value,
        "corrected_alpha": corrected_alpha,
        "significant": p_value < corrected_alpha,
    }
