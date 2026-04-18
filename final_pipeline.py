"""
final_pipeline.py  —  5G Network Congestion Prediction & Load Balancing
Generates ALL figures required for the project report:
  Section 2: fig_architecture.png, fig_workflow.png
  Section 4: fig_class_distribution.png, fig_learning_curves.png,
             fig_confusion_matrix.png, fig_per_class_metrics.png,
             fig_prediction_distribution.png, fig_feature_importance.png,
             fig_stress_distribution.png, fig_decision_actions.png
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_PATH  = Path("5g_network_data_processed.csv")
OUTPUT_DIR = Path("figures")

FIG_ARCHITECTURE  = OUTPUT_DIR / "fig_architecture.png"
FIG_WORKFLOW      = OUTPUT_DIR / "fig_workflow.png"
FIG_CLASS_DIST    = OUTPUT_DIR / "fig_class_distribution.png"
FIG_LEARN_CURVES  = OUTPUT_DIR / "fig_learning_curves.png"
FIG_CONFUSION     = OUTPUT_DIR / "fig_confusion_matrix.png"
FIG_CLASS_METRICS = OUTPUT_DIR / "fig_per_class_metrics.png"
FIG_PRED_DIST     = OUTPUT_DIR / "fig_prediction_distribution.png"
FIG_FEAT_IMP      = OUTPUT_DIR / "fig_feature_importance.png"
FIG_STRESS        = OUTPUT_DIR / "fig_stress_distribution.png"
FIG_ACTIONS       = OUTPUT_DIR / "fig_decision_actions.png"

MODEL_PATH       = OUTPUT_DIR / "model.pth"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
METRICS_PATH     = OUTPUT_DIR / "metrics.txt"

LABEL_TO_ID = {"Low": 0, "Medium": 1, "High": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
CLASS_NAMES = ["Low", "Medium", "High"]

EPOCHS        = 300
LEARNING_RATE = 1e-3
SEED          = 42

FEATURES = [
    "Signal_Strength_dBm", "Download_Speed_Mbps", "Upload_Speed_Mbps",
    "Latency_ms", "Jitter_ms", "Ping_to_Google_ms", "Data_Usage_MB",
    "Handover_Count", "Connected_Duration_min",
    "Speed_Ratio", "Latency_Jitter", "Signal_Latency", "Load_Index",
]


# ─────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────
def load_and_clean_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)
    df = df.drop(columns=["Timestamp"], errors="ignore")

    # Feature engineering (same as prepare_data.py)
    df["Speed_Ratio"]    = df["Download_Speed_Mbps"] / (df["Upload_Speed_Mbps"] + 1e-6)
    df["Latency_Jitter"] = df["Latency_ms"] * df["Jitter_ms"]
    df["Signal_Latency"] = df["Signal_Strength_dBm"] / (df["Latency_ms"] + 1)
    df["Load_Index"]     = df["Data_Usage_MB"] / (df["Connected_Duration_min"] + 1)

    y = df["Network_Congestion_Level"].map(LABEL_TO_ID)
    if y.isna().any():
        raise ValueError("Unexpected labels found in Network_Congestion_Level.")
    X = df[FEATURES].copy()
    return df, X, y


def split_and_scale(X: pd.DataFrame, y: pd.Series):
    split = int(len(X) * 0.8)
    X_train_df, X_test_df = X.iloc[:split].copy(), X.iloc[split:].copy()
    y_train,    y_test    = y.iloc[:split].copy(),  y.iloc[split:].copy()

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test  = scaler.transform(X_test_df)
    return X_train, X_test, y_train, y_test, X_train_df.reset_index(drop=True), X_test_df.reset_index(drop=True)


# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
class TELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.exp(torch.clamp(x, max=10.0)))


class CongestionNet(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128), nn.BatchNorm1d(128), TELU(), nn.Dropout(0.4),
            nn.Linear(128, 64),         nn.BatchNorm1d(64),  TELU(), nn.Dropout(0.3),
            nn.Linear(64, 32),          TELU(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
#  TRAINING  (tracks train + test metrics)
# ─────────────────────────────────────────────
def train_model(model, X_train, y_train, X_test, y_test,
                epochs, lr, class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    train_accs,   test_accs   = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        out  = model(X_train)
        loss = criterion(out, y_train)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            t_out  = model(X_train)
            v_out  = model(X_test)
            t_loss = criterion(t_out, y_train).item()
            v_loss = criterion(v_out, y_test).item()
            t_acc  = (torch.argmax(t_out, 1) == y_train).float().mean().item()
            v_acc  = (torch.argmax(v_out, 1) == y_test ).float().mean().item()

        train_losses.append(t_loss); test_losses.append(v_loss)
        train_accs.append(t_acc);   test_accs.append(v_acc)

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch {epoch:>3}/{epochs} | "
                  f"Train Loss {t_loss:.4f}  Test Loss {v_loss:.4f} | "
                  f"Train Acc {t_acc:.4f}  Test Acc {v_acc:.4f}")

    return train_losses, test_losses, train_accs, test_accs


# ─────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predicted = torch.argmax(model(X_test), 1)
    y_np   = y_test.cpu().numpy()
    p_np   = predicted.cpu().numpy()
    acc    = accuracy_score(y_np, p_np)
    report = classification_report(y_np, p_np, labels=[0,1,2],
                                   target_names=CLASS_NAMES, digits=4, zero_division=0)
    report_dict = classification_report(y_np, p_np, labels=[0,1,2],
                                        target_names=CLASS_NAMES, zero_division=0,
                                        output_dict=True)
    cm = confusion_matrix(y_np, p_np, labels=[0,1,2])
    return acc, report, report_dict, cm, y_np, p_np


# ─────────────────────────────────────────────
#  DECISION ENGINE
# ─────────────────────────────────────────────
def _norm(v, lo, hi, invert=False):
    if hi == lo: return 0.0
    n = float(np.clip((v - lo) / (hi - lo), 0, 1))
    return 1.0 - n if invert else n

def compute_stress(row, ranges):
    w = {"Latency_ms":0.20,"Jitter_ms":0.12,"Ping_to_Google_ms":0.13,
         "Download_Speed_Mbps":0.15,"Upload_Speed_Mbps":0.10,
         "Signal_Strength_dBm":0.15,"Data_Usage_MB":0.10,"Handover_Count":0.05}
    inv = {"Download_Speed_Mbps","Upload_Speed_Mbps","Signal_Strength_dBm"}
    return sum(w[f] * _norm(row[f], *ranges[f], f in inv) for f in w)

def decide_action(row, pred, stress, baseline):
    lat, jit, sig, dl, hc = (row["Latency_ms"], row["Jitter_ms"],
                              row["Signal_Strength_dBm"], row["Download_Speed_Mbps"],
                              row["Handover_Count"])
    if pred == 2:
        if lat > 15 and stress >= 0.65:  return "Route to low latency path"
        if sig < -95 and stress >= 0.60: return "Switch to stronger cell"
        if jit > 8 or hc >= 6:           return "Stabilize connection"
        return "Immediate load balancing"
    if pred == 1:
        if stress >= 0.60:               return "Immediate load balancing"
        if lat > 15:                     return "Route to low latency path"
        if sig < -95:                    return "Switch to stronger cell"
        if jit > 8 or hc >= 6:          return "Stabilize connection"
        if stress >= 0.45 or dl < baseline: return "Monitor network"
        return "No action"
    if stress >= 0.55: return "Monitor network"
    if jit > 8 and sig < -95: return "Stabilize connection"
    return "No action"

def run_decision_engine(X_train_orig, X_test_orig, predicted_np):
    feat = ["Latency_ms","Jitter_ms","Ping_to_Google_ms","Download_Speed_Mbps",
            "Upload_Speed_Mbps","Signal_Strength_dBm","Data_Usage_MB","Handover_Count"]
    ranges   = {f: (X_train_orig[f].min(), X_train_orig[f].max()) for f in feat}
    baseline = X_train_orig["Download_Speed_Mbps"].median()
    stresses, actions = [], []
    for i, pred in enumerate(predicted_np):
        row = X_test_orig.iloc[i]
        s   = compute_stress(row, ranges)
        a   = decide_action(row, int(pred), s, baseline)
        stresses.append(s); actions.append(a)
    return stresses, actions, pd.Series(actions).value_counts()


# ═══════════════════════════════════════════════════════════════
#  FIGURES — all 10 required for the project report
# ═══════════════════════════════════════════════════════════════

# ── Section 2 figures ─────────────────────────────────────────

def save_architecture_diagram():
    """System architecture block diagram (Section 2)."""
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.axis("off")

    blocks = [
        ("Raw 5G\nCSV Data",                      0.06, "#AED6F1"),
        ("Data Cleaning\n& Feature Eng.",          0.22, "#A9DFBF"),
        ("Train / Test\nSplit  80 / 20",           0.38, "#A9DFBF"),
        ("TELU Neural\nNetwork",                   0.54, "#F9E79F"),
        ("Evaluation\nAccuracy · F1 · CM",         0.70, "#FAD7A0"),
        ("Decision Engine\nLoad Balancing",        0.86, "#F1948A"),
    ]

    for (label, xc, color) in blocks:
        ax.add_patch(mpatches.FancyBboxPatch(
            (xc - 0.075, 0.20), 0.135, 0.60,
            boxstyle="round,pad=0.02", lw=1.8,
            edgecolor="#444", facecolor=color,
            transform=ax.transAxes, clip_on=False
        ))
        ax.text(xc, 0.50, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", transform=ax.transAxes)

    for i in range(len(blocks) - 1):
        x1 = blocks[i][1]   + 0.065
        x2 = blocks[i+1][1] - 0.076
        ax.annotate("", xy=(x2, 0.50), xytext=(x1, 0.50),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#333", lw=2.0))

    ax.text(0.50, 0.93,
            "System Architecture — 5G Network Congestion Prediction & Load Balancing",
            ha="center", va="center", fontsize=12, fontweight="bold",
            transform=ax.transAxes)

    # Sub-labels
    sub = ["Input", "Preprocessing", "Splitting", "Model", "Metrics", "Output"]
    for (_, xc, _), s in zip(blocks, sub):
        ax.text(xc, 0.13, s, ha="center", va="center",
                fontsize=8, color="#555", transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(FIG_ARCHITECTURE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG_ARCHITECTURE}")


def save_workflow_diagram():
    """Algorithm workflow flowchart (Section 2)."""
    fig, ax = plt.subplots(figsize=(7, 14))
    ax.set_xlim(0, 10); ax.set_ylim(0, 19); ax.axis("off")

    steps = [
        (5, 18.0, "START",                                         "#2ECC71", "oval"),
        (5, 16.2, "Load 5G Telemetry CSV",                         "#AED6F1", "rect"),
        (5, 14.3, "Clean Columns & Encode Booleans",               "#AED6F1", "rect"),
        (5, 12.4, "Feature Engineering\n(Speed Ratio, Load Index…)","#AED6F1", "rect"),
        (5, 10.5, "80/20 Train-Test Split\n(ordered, no shuffle)", "#A9DFBF", "rect"),
        (5,  8.6, "Min-Max Scaling\n(fit on train only)",          "#A9DFBF", "rect"),
        (5,  6.7, "Train TELU Neural Network\n(300 epochs, AdamW, class weights)",
                                                                    "#F9E79F", "rect"),
        (5,  4.8, "Evaluate Model\n(Accuracy, F1, Confusion Matrix)","#FAD7A0","rect"),
        (5,  2.9, "Decision Engine\n(Stress Score → Action)",      "#F1948A", "rect"),
        (5,  1.2, "Save All Figures & Metrics",                    "#D2B4DE", "rect"),
        (5,  0.0, "END",                                           "#E74C3C", "oval"),
    ]

    for (x, y, text, color, shape) in steps:
        if shape == "oval":
            ax.add_patch(mpatches.Ellipse((x, y), 3.6, 0.85,
                         facecolor=color, edgecolor="#444", lw=1.5, zorder=3))
        else:
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - 2.1, y - 0.55), 4.2, 1.1,
                boxstyle="round,pad=0.1", facecolor=color,
                edgecolor="#444", lw=1.2, zorder=3))
        ax.text(x, y, text, ha="center", va="center",
                fontsize=8.5, fontweight="bold", zorder=4)

    for i in range(len(steps) - 1):
        y_top = steps[i][1]   - 0.43
        y_bot = steps[i+1][1] + 0.43
        ax.annotate("", xy=(5, y_bot), xytext=(5, y_top),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    ax.set_title("Algorithm Workflow", fontsize=13, fontweight="bold", pad=8)
    plt.tight_layout()
    fig.savefig(FIG_WORKFLOW, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG_WORKFLOW}")


# ── Section 4 figures ─────────────────────────────────────────

def save_class_distribution(y_full: pd.Series):
    """Bar + pie of dataset class balance."""
    counts = y_full.map(ID_TO_LABEL).value_counts().reindex(CLASS_NAMES)
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    bars = axes[0].bar(CLASS_NAMES, counts.values, color=colors, edgecolor="black", lw=0.8)
    for bar, cnt in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                     str(cnt), ha="center", fontweight="bold", fontsize=10)
    axes[0].set_title("Class Count"); axes[0].set_xlabel("Congestion Level")
    axes[0].set_ylabel("Samples"); axes[0].grid(axis="y", alpha=0.3)

    axes[1].pie(counts.values, labels=CLASS_NAMES, colors=colors,
                autopct="%1.1f%%", startangle=90, pctdistance=0.80)
    axes[1].set_title("Class Proportion")

    fig.suptitle("Dataset Class Distribution — Network Congestion Level",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_CLASS_DIST, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG_CLASS_DIST}")


def save_learning_curves(tr_loss, te_loss, tr_acc, te_acc):
    """2×2 grid: train loss / test loss / train accuracy / test accuracy."""
    ep = range(1, len(tr_loss) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # ── row 0: loss ──────────────────────────────────
    axes[0, 0].plot(ep, tr_loss, color="#2980b9", label="Train Loss")
    axes[0, 0].plot(ep, te_loss, color="#e74c3c", linestyle="--", label="Test Loss")
    axes[0, 0].set_title("Loss Curve (Train vs Test)"); axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross-Entropy Loss"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(ep, tr_loss, color="#2980b9"); axes[0, 1].set_title("Training Loss")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Loss"); axes[0, 1].grid(alpha=0.3)

    # ── row 1: accuracy ──────────────────────────────
    axes[1, 0].plot(ep, tr_acc, color="#27ae60", label="Train Acc")
    axes[1, 0].plot(ep, te_acc, color="#8e44ad", linestyle="--", label="Test Acc")
    axes[1, 0].set_title("Accuracy Curve (Train vs Test)"); axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy"); axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(ep, te_acc, color="#8e44ad"); axes[1, 1].set_title("Test Accuracy")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_ylim(0, 1.05); axes[1, 1].grid(alpha=0.3)
    # Annotate final test accuracy
    final_acc = te_acc[-1]
    axes[1, 1].axhline(final_acc, color="red", linestyle=":", lw=1.2)
    axes[1, 1].text(len(ep)*0.6, final_acc + 0.02, f"Final: {final_acc:.4f}",
                    color="red", fontsize=9)

    fig.suptitle("Neural Network Learning Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_LEARN_CURVES, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG_LEARN_CURVES}")


def save_confusion_matrix_plot(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Actual Label",    fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    thresh = cm.max() / 2.0
    for r in range(3):
        for c in range(3):
            ax.text(c, r, str(int(cm[r, c])), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[r, c] > thresh else "black")
    plt.tight_layout()
    fig.savefig(FIG_CONFUSION, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG_CONFUSION}")


def save_per_class_metrics(report_dict: dict):
    """Grouped bar chart: Precision / Recall / F1 per congestion class."""
    prec = [report_dict[c]["precision"] for c in CLASS_NAMES]
    rec  = [report_dict[c]["recall"]    for c in CLASS_NAMES]
    f1   = [report_dict[c]["f1-score"]  for c in CLASS_NAMES]
    x    = np.arange(3); w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.bar(x - w, prec, w, label="Precision", color="#3498db", edgecolor="black", lw=0.6)
    br = ax.bar(x,     rec,  w, label="Recall",    color="#2ecc71", edgecolor="black", lw=0.6)
    bf = ax.bar(x + w, f1,   w, label="F1-Score",  color="#e74c3c", edgecolor="black", lw=0.6)

    for bars in [bp, br, bf]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylim(0, 1.20); ax.set_xlabel("Congestion Level", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Per-Class Classification Metrics — Precision / Recall / F1",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_CLASS_METRICS, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG_CLASS_METRICS}")


def save_prediction_distribution(predicted_np: np.ndarray):
    counts = [np.sum(predicted_np == i) for i in range(3)]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(CLASS_NAMES, counts, color=colors, edgecolor="black", lw=0.8)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(cnt), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Prediction Distribution — Test Set", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Congestion Level"); ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_PRED_DIST, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG_PRED_DIST}")


def save_feature_importance(model, feature_names: list):
    weights    = model.net[0].weight.detach().cpu().numpy()
    importance = np.mean(np.abs(weights), axis=0)
    idx        = np.argsort(importance)
    feats      = [feature_names[i] for i in idx]
    scores     = importance[idx]
    cmap_vals  = plt.cm.RdYlGn(scores / scores.max())

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(feats, scores, color=cmap_vals, edgecolor="black", lw=0.6)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_width() + 0.0003, bar.get_y() + bar.get_height() / 2,
                f"{s:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |Weight| — First Layer Neurons", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.set_title("Feature Importance Proxy (First-Layer Absolute Weights)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_FEAT_IMP, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG_FEAT_IMP}")


def save_stress_distribution(stress_scores: list, predicted_np: np.ndarray):
    """Histogram of stress scores overall + per predicted class."""
    arr    = np.array(stress_scores)
    colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Overall
    axes[0].hist(arr, bins=30, color="#5dade2", edgecolor="black", lw=0.6)
    axes[0].axvline(arr.mean(), color="red", linestyle="--", lw=1.5,
                    label=f"Mean = {arr.mean():.3f}")
    axes[0].axvline(arr.median() if hasattr(arr, "median") else np.median(arr),
                    color="orange", linestyle=":", lw=1.5,
                    label=f"Median = {np.median(arr):.3f}")
    axes[0].set_title("Overall Stress Score Distribution")
    axes[0].set_xlabel("Stress Score"); axes[0].set_ylabel("Count")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Per class overlay
    for cls_id, cls_name in ID_TO_LABEL.items():
        mask = predicted_np == cls_id
        if mask.sum() > 0:
            axes[1].hist(arr[mask], bins=20, alpha=0.65,
                         label=cls_name, color=colors[cls_id],
                         edgecolor="black", lw=0.4)
    axes[1].set_title("Stress Score by Predicted Congestion Class")
    axes[1].set_xlabel("Stress Score"); axes[1].set_ylabel("Count")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle("Network Stress Score Analysis", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_STRESS, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG_STRESS}")


def save_decision_actions(action_counts: pd.Series):
    """Horizontal bar chart + pie chart of recommended actions."""
    palette = {
        "No action":                 "#2ecc71",
        "Monitor network":           "#f39c12",
        "Immediate load balancing":  "#e67e22",
        "Route to low latency path": "#3498db",
        "Switch to stronger cell":   "#9b59b6",
        "Stabilize connection":      "#e74c3c",
    }
    colors = [palette.get(a, "#95a5a6") for a in action_counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = axes[0].barh(action_counts.index, action_counts.values,
                        color=colors, edgecolor="black", lw=0.6)
    for bar, cnt in zip(bars, action_counts.values):
        axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     str(cnt), va="center", fontweight="bold")
    axes[0].set_title("Action Count (Test Set)")
    axes[0].set_xlabel("Number of Samples"); axes[0].grid(axis="x", alpha=0.3)

    axes[1].pie(action_counts.values, labels=action_counts.index,
                colors=colors, autopct="%1.1f%%", startangle=90, pctdistance=0.80)
    axes[1].set_title("Action Distribution (%)")

    fig.suptitle("Decision Engine — Recommended Network Actions",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_ACTIONS, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIG_ACTIONS}")


# ─────────────────────────────────────────────
#  OUTPUT FILES
# ─────────────────────────────────────────────
def save_metrics(accuracy, report_text, cm, action_counts):
    lines = [
        f"Test Accuracy: {accuracy:.4f}", "",
        "Classification Report:", report_text, "",
        "Confusion Matrix:", np.array2string(cm), "",
        "Decision Engine Action Distribution:", action_counts.to_string(),
    ]
    METRICS_PATH.write_text("\n".join(lines), encoding="utf-8")

def save_predictions(y_np, p_np):
    pd.DataFrame({
        "actual_label":    [ID_TO_LABEL[l] for l in y_np],
        "predicted_label": [ID_TO_LABEL[l] for l in p_np],
    }).to_csv(PREDICTIONS_PATH, index=False)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    set_seed(SEED)

    # ── Data ──────────────────────────────────
    df, X, y = load_and_clean_data(DATA_PATH)
    X_train_np, X_test_np, y_train_s, y_test_s, X_train_orig, X_test_orig = split_and_scale(X, y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Dataset: {len(df)} rows | Train: {len(X_train_orig)} | Test: {len(X_test_orig)}\n")

    X_tr = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    X_te = torch.tensor(X_test_np,  dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train_s.values, dtype=torch.long, device=device)
    y_te = torch.tensor(y_test_s.values,  dtype=torch.long, device=device)

    classes_arr = np.array(sorted(y_train_s.unique()))
    cw = compute_class_weight("balanced", classes=classes_arr, y=y_train_s)
    class_weights = torch.tensor(cw, dtype=torch.float32, device=device)

    # ── Train ─────────────────────────────────
    model = CongestionNet(X_tr.shape[1]).to(device)
    tr_loss, te_loss, tr_acc, te_acc = train_model(
        model, X_tr, y_tr, X_te, y_te, EPOCHS, LEARNING_RATE, class_weights)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")

    # ── Evaluate ──────────────────────────────
    accuracy, report_text, report_dict, cm, y_np, p_np = evaluate_model(model, X_te, y_te)
    stresses, actions, action_counts = run_decision_engine(X_train_orig, X_test_orig, p_np)

    save_predictions(y_np, p_np)
    save_metrics(accuracy, report_text, cm, action_counts)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(report_text)

    # ── Generate ALL Figures ──────────────────
    print("\n--- Saving report figures ---")
    save_architecture_diagram()
    save_workflow_diagram()
    save_class_distribution(y)
    save_learning_curves(tr_loss, te_loss, tr_acc, te_acc)
    save_confusion_matrix_plot(cm)
    save_per_class_metrics(report_dict)
    save_prediction_distribution(p_np)
    save_feature_importance(model, FEATURES)
    save_stress_distribution(stresses, p_np)
    save_decision_actions(action_counts)

    print(f"\n✓ All 10 figures saved to ./{OUTPUT_DIR}/")
    print(f"  fig_architecture.png       — Section 2: System Architecture")
    print(f"  fig_workflow.png           — Section 2: Algorithm Workflow")
    print(f"  fig_class_distribution.png — Section 4: Dataset Balance")
    print(f"  fig_learning_curves.png    — Section 4: Train/Test Loss & Accuracy")
    print(f"  fig_confusion_matrix.png   — Section 4: Confusion Matrix")
    print(f"  fig_per_class_metrics.png  — Section 4: Precision / Recall / F1")
    print(f"  fig_prediction_distribution.png — Section 4: Prediction Distribution")
    print(f"  fig_feature_importance.png — Section 4: Feature Importance")
    print(f"  fig_stress_distribution.png— Section 4: Stress Score Analysis")
    print(f"  fig_decision_actions.png   — Section 4: Decision Engine Actions")


if __name__ == "__main__":
    main()
