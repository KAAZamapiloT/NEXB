from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler


DATA_PATH = Path("5g_network_data.csv")
OUTPUT_DIR = Path("output")
MODEL_PATH = OUTPUT_DIR / "model.pth"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "metrics.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
PREDICTION_DISTRIBUTION_PATH = OUTPUT_DIR / "prediction_distribution.png"
TRAINING_LOSS_PATH = OUTPUT_DIR / "training_loss.png"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.png"

LABEL_TO_ID = {"Low": 0, "Medium": 1, "High": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
CLASS_NAMES = ["Low", "Medium", "High"]

EPOCHS = 40
LEARNING_RATE = 1e-3
SEED = 42


def set_seed(seed: int) -> None:
    """Make the training run reproducible across CPU and GPU execution."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_clean_data(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the dataset, clean the schema, and prepare the target labels."""
    df = pd.read_csv(csv_path)

    # Cleaning column names early keeps the rest of the pipeline readable and avoids fragile string handling later.
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )

    # Boolean service indicators must be converted to numeric flags before neural network training.
    bool_columns = df.select_dtypes(include=["bool"]).columns
    for column in bool_columns:
        df[column] = df[column].astype(int)

    # Timestamp is removed in this version because the model uses snapshot features rather than a temporal sequence model.
    df = df.drop(columns=["Timestamp"], errors="ignore")

    y = df["Network_Congestion_Level"].map(LABEL_TO_ID)
    if y.isna().any():
        raise ValueError("Unexpected labels found in Network_Congestion_Level.")

    X = df.drop(columns=["Network_Congestion_Level"]).copy()
    return df, X, y


def encode_and_split_features(
    X: pd.DataFrame, y: pd.Series
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, list[str]]:
    """Encode categorical columns, preserve order, and split 90/10 without shuffling."""
    X_encoded = pd.get_dummies(X)
    feature_names = X_encoded.columns.tolist()

    # A fixed ordered split preserves time-like ordering and avoids leaking future rows into training.
    split_index = int(len(X_encoded) * 0.9)
    X_train_df = X_encoded.iloc[:split_index].copy()
    X_test_df = X_encoded.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()
    X_train_original = X.iloc[:split_index].reset_index(drop=True)
    X_test_original = X.iloc[split_index:].reset_index(drop=True)

    # Normalization keeps heterogeneous metrics such as latency, throughput, and signal strength on comparable ranges.
    # Fitting the scaler on training data only prevents the test window from influencing preprocessing statistics.
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    return X_train, X_test, y_train, y_test, X_train_original, X_test_original, feature_names


class TELU(nn.Module):
    """TELU activation with clamping to prevent overflow inside the exponential term."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TELU provides a smooth nonlinear response, and clamping keeps exp() numerically stable for large inputs.
        clamped_x = torch.clamp(x, max=10.0)
        return x * torch.tanh(torch.exp(clamped_x))


class CongestionNet(nn.Module):
    """TELU-based multilayer classifier for 5G congestion prediction."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            # BatchNorm stabilizes internal feature distributions, while Dropout reduces overfitting pressure.
            nn.BatchNorm1d(128),
            TELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            TELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            TELU(),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    learning_rate: float,
) -> list[float]:
    """Train the neural network and return the loss history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss = loss.item()
        loss_history.append(epoch_loss)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")

    return loss_history


def evaluate_model(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> tuple[float, str, np.ndarray, np.ndarray, np.ndarray]:
    """Run evaluation and return the main classification artifacts."""
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        predicted = torch.argmax(logits, dim=1)

    # Predictions are moved back to CPU so they can be saved, analyzed, and plotted with standard Python tools.
    y_test_np = y_test.detach().cpu().numpy()
    predicted_np = predicted.detach().cpu().numpy()

    accuracy = accuracy_score(y_test_np, predicted_np)
    report_text = classification_report(
        y_test_np,
        predicted_np,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test_np, predicted_np, labels=[0, 1, 2])

    return accuracy, report_text, cm, y_test_np, predicted_np


def build_stress_feature_ranges(reference_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Create normalization ranges for multi-parameter network stress scoring."""
    monitored_features = [
        "Latency_ms",
        "Jitter_ms",
        "Ping_to_Google_ms",
        "Download_Speed_Mbps",
        "Upload_Speed_Mbps",
        "Signal_Strength_dBm",
        "Data_Usage_MB",
        "Handover_Count",
    ]
    return {
        feature: (reference_df[feature].min(), reference_df[feature].max())
        for feature in monitored_features
    }


def normalize_metric(
    value: float,
    min_val: float,
    max_val: float,
    invert: bool = False,
) -> float:
    """Normalize a metric to 0-1 so different units can be fused into a single stress score."""
    if max_val == min_val:
        return 0.0

    normalized = (value - min_val) / (max_val - min_val)
    normalized = float(np.clip(normalized, 0.0, 1.0))
    return 1.0 - normalized if invert else normalized


def compute_network_stress(
    row: pd.Series, stress_feature_ranges: dict[str, tuple[float, float]]
) -> float:
    """Compute a weighted stress score from multiple network quality indicators."""
    stress_weights = {
        "Latency_ms": 0.20,
        "Jitter_ms": 0.12,
        "Ping_to_Google_ms": 0.13,
        "Download_Speed_Mbps": 0.15,
        "Upload_Speed_Mbps": 0.10,
        "Signal_Strength_dBm": 0.15,
        "Data_Usage_MB": 0.10,
        "Handover_Count": 0.05,
    }

    # Multi-signal scoring is closer to real telecom optimization because congestion rarely manifests through only one metric.
    normalized_metrics = {
        "Latency_ms": normalize_metric(row["Latency_ms"], *stress_feature_ranges["Latency_ms"]),
        "Jitter_ms": normalize_metric(row["Jitter_ms"], *stress_feature_ranges["Jitter_ms"]),
        "Ping_to_Google_ms": normalize_metric(
            row["Ping_to_Google_ms"], *stress_feature_ranges["Ping_to_Google_ms"]
        ),
        "Download_Speed_Mbps": normalize_metric(
            row["Download_Speed_Mbps"],
            *stress_feature_ranges["Download_Speed_Mbps"],
            invert=True,
        ),
        "Upload_Speed_Mbps": normalize_metric(
            row["Upload_Speed_Mbps"],
            *stress_feature_ranges["Upload_Speed_Mbps"],
            invert=True,
        ),
        "Signal_Strength_dBm": normalize_metric(
            row["Signal_Strength_dBm"],
            *stress_feature_ranges["Signal_Strength_dBm"],
            invert=True,
        ),
        "Data_Usage_MB": normalize_metric(
            row["Data_Usage_MB"], *stress_feature_ranges["Data_Usage_MB"]
        ),
        "Handover_Count": normalize_metric(
            row["Handover_Count"], *stress_feature_ranges["Handover_Count"]
        ),
    }

    return sum(
        stress_weights[feature] * normalized_metrics[feature]
        for feature in stress_weights
    )


def optimize_network(
    row: pd.Series,
    predicted_class: int,
    stress_score: float,
    throughput_baseline: float,
) -> str:
    """Translate congestion predictions and stress levels into interpretable network actions."""
    latency = row["Latency_ms"]
    jitter = row["Jitter_ms"]
    signal = row["Signal_Strength_dBm"]
    download = row["Download_Speed_Mbps"]
    handovers = row["Handover_Count"]

    if predicted_class == 2:
        if latency > 15 and stress_score >= 0.65:
            return "Route to low latency path"
        if signal < -95 and stress_score >= 0.60:
            return "Switch to stronger cell"
        if jitter > 8 or handovers >= 6:
            return "Stabilize connection"
        return "Immediate load balancing"

    if predicted_class == 1:
        if stress_score >= 0.60:
            return "Immediate load balancing"
        if latency > 15:
            return "Route to low latency path"
        if signal < -95:
            return "Switch to stronger cell"
        if jitter > 8 or handovers >= 6:
            return "Stabilize connection"
        if stress_score >= 0.45 or download < throughput_baseline:
            return "Monitor network"
        return "No action"

    if stress_score >= 0.55:
        return "Monitor network"
    if jitter > 8 and signal < -95:
        return "Stabilize connection"
    return "No action"


def run_decision_engine(
    X_train_original: pd.DataFrame,
    X_test_original: pd.DataFrame,
    predicted_np: np.ndarray,
) -> tuple[list[float], list[str], pd.Series]:
    """Apply the stress-based decision engine to the test window."""
    stress_feature_ranges = build_stress_feature_ranges(X_train_original)
    throughput_baseline = X_train_original["Download_Speed_Mbps"].median()

    stress_scores: list[float] = []
    actions: list[str] = []
    for index, predicted_class in enumerate(predicted_np):
        row = X_test_original.iloc[index]
        stress_score = compute_network_stress(row, stress_feature_ranges)
        action = optimize_network(row, int(predicted_class), stress_score, throughput_baseline)
        stress_scores.append(stress_score)
        actions.append(action)

    action_counts = pd.Series(actions).value_counts()
    return stress_scores, actions, action_counts


def save_predictions_csv(y_test_np: np.ndarray, predicted_np: np.ndarray) -> None:
    """Save test labels and predictions for offline analysis."""
    predictions_df = pd.DataFrame(
        {
            "actual_label": [ID_TO_LABEL[label] for label in y_test_np],
            "predicted_label": [ID_TO_LABEL[label] for label in predicted_np],
        }
    )
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)


def save_metrics_file(
    accuracy: float,
    report_text: str,
    cm: np.ndarray,
    action_counts: pd.Series,
) -> None:
    """Save the main evaluation metrics and decision-engine summary."""
    metrics_lines = [
        f"Accuracy: {accuracy:.4f}",
        "",
        "Classification Report:",
        report_text,
        "",
        "Confusion Matrix:",
        np.array2string(cm),
        "",
        "Decision Engine Action Distribution:",
        action_counts.to_string(),
    ]
    METRICS_PATH.write_text("\n".join(metrics_lines), encoding="utf-8")


def save_confusion_matrix_plot(cm: np.ndarray) -> None:
    """Save a confusion matrix heatmap using matplotlib only."""
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")

    for row_index in range(cm.shape[0]):
        for col_index in range(cm.shape[1]):
            ax.text(col_index, row_index, int(cm[row_index, col_index]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close(fig)


def save_prediction_distribution_plot(predicted_np: np.ndarray) -> None:
    """Save the predicted class distribution as a histogram."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(predicted_np, bins=np.arange(-0.5, len(CLASS_NAMES) + 0.5, 1))
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted Congestion Level")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Distribution")
    fig.tight_layout()
    fig.savefig(PREDICTION_DISTRIBUTION_PATH, dpi=300)
    plt.close(fig)


def save_training_loss_plot(loss_history: list[float]) -> None:
    """Save training loss versus epoch."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    fig.tight_layout()
    fig.savefig(TRAINING_LOSS_PATH, dpi=300)
    plt.close(fig)


def save_feature_importance_plot(model: CongestionNet, feature_names: list[str]) -> None:
    """Save a simple feature-importance proxy based on first-layer input weights."""
    # This is a lightweight importance proxy that highlights which encoded inputs receive the strongest first-layer weights.
    first_layer_weights = model.net[0].weight.detach().cpu().numpy()
    importance_scores = np.mean(np.abs(first_layer_weights), axis=0)

    top_k = min(20, len(feature_names))
    top_indices = np.argsort(importance_scores)[-top_k:]
    top_features = [feature_names[index] for index in top_indices]
    top_scores = importance_scores[top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features, top_scores)
    ax.set_xlabel("Mean Absolute Weight")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance Proxy")
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    """Run the full congestion-prediction and load-balancing pipeline."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    set_seed(SEED)

    df, X_original, y = load_and_clean_data(DATA_PATH)
    (
        X_train_np,
        X_test_np,
        y_train_series,
        y_test_series,
        X_train_original,
        X_test_original,
        feature_names,
    ) = encode_and_split_features(X_original, y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset rows: {len(df)} | Train rows: {len(X_train_original)} | Test rows: {len(X_test_original)}")

    # Model inputs and labels are moved to the selected device so the same code runs on CPU or GPU.
    X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_series.values, dtype=torch.long, device=device)
    y_test = torch.tensor(y_test_series.values, dtype=torch.long, device=device)

    model = CongestionNet(X_train.shape[1]).to(device)
    loss_history = train_model(model, X_train, y_train, EPOCHS, LEARNING_RATE)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    accuracy, report_text, cm, y_test_np, predicted_np = evaluate_model(model, X_test, y_test)
    stress_scores, actions, action_counts = run_decision_engine(
        X_train_original, X_test_original, predicted_np
    )

    save_predictions_csv(y_test_np, predicted_np)
    save_metrics_file(accuracy, report_text, cm, action_counts)
    save_confusion_matrix_plot(cm)
    save_prediction_distribution_plot(predicted_np)
    save_training_loss_plot(loss_history)
    save_feature_importance_plot(model, feature_names)

    print(f"Accuracy: {accuracy:.4f}")
    print(report_text)
    print("Sample decision engine outputs:")
    for index in range(min(5, len(actions))):
        print(
            f"  Row {index}: stress={stress_scores[index]:.3f}, "
            f"predicted={ID_TO_LABEL[int(predicted_np[index])]}, action={actions[index]}"
        )
    print(f"Saved predictions to {PREDICTIONS_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved confusion matrix plot to {CONFUSION_MATRIX_PATH}")
    print(f"Saved prediction distribution plot to {PREDICTION_DISTRIBUTION_PATH}")
    print(f"Saved training loss plot to {TRAINING_LOSS_PATH}")
    print(f"Saved feature importance plot to {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
