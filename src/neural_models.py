from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import RANDOM_STATE
from .evaluation import add_interval_metrics, classification_metrics, regression_metrics
from .plotting import save_loss_curve


SEQUENCE_LENGTH = 12
MAX_EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
DROPOUT = 0.2
HIDDEN_SIZE = 32


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def make_sequences(df: pd.DataFrame, feature_cols: list[str], target_col: str, classification: bool):
    Xs, ys, meta = [], [], []
    for country, part in df.sort_values(["COUNTRY_CODE", "ISO_WEEKSTARTDATE"]).groupby("COUNTRY_CODE"):
        part = part.reset_index(drop=True)
        X = part[feature_cols].to_numpy(dtype=float)
        y = part[target_col].to_numpy()
        for i in range(SEQUENCE_LENGTH, len(part)):
            target = y[i]
            if pd.isna(target):
                continue
            Xs.append(X[i - SEQUENCE_LENGTH : i])
            ys.append(target)
            meta.append(part.iloc[i])
    if not Xs:
        return np.empty((0, SEQUENCE_LENGTH, len(feature_cols))), np.array([]), pd.DataFrame()
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys), pd.DataFrame(meta)


class SequenceNet:
    def __init__(self, architecture: str, input_size: int, output_size: int, task: str):
        import torch
        from torch import nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.architecture = architecture
                self.task = task
                if architecture == "lstm":
                    self.seq = nn.LSTM(input_size, HIDDEN_SIZE, batch_first=True, dropout=0)
                    head_in = HIDDEN_SIZE
                elif architecture == "gru":
                    self.seq = nn.GRU(input_size, HIDDEN_SIZE, batch_first=True, dropout=0)
                    head_in = HIDDEN_SIZE
                elif architecture == "temporal_cnn":
                    self.conv = nn.Sequential(
                        nn.Conv1d(input_size, HIDDEN_SIZE, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Dropout(DROPOUT),
                        nn.Conv1d(HIDDEN_SIZE, HIDDEN_SIZE, kernel_size=3, padding=1),
                        nn.ReLU(),
                    )
                    head_in = HIDDEN_SIZE
                else:
                    raise ValueError(architecture)
                self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(head_in, output_size))

            def forward(self, x):
                if self.architecture in {"lstm", "gru"}:
                    _, hidden = self.seq(x)
                    if isinstance(hidden, tuple):
                        hidden = hidden[0]
                    z = hidden[-1]
                else:
                    z = self.conv(x.transpose(1, 2)).mean(dim=2)
                return self.head(z)

        self.model = _Net()
        self.torch = torch


def train_sequence_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    architecture: str,
    task: str,
    out_prefix: Path,
):
    if not torch_available():
        raise RuntimeError("PyTorch is not installed")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(RANDOM_STATE)
    train_df = train_df.copy()
    test_df = test_df.copy()

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    flat_train = train_df[feature_cols].to_numpy(dtype=float)
    imputer.fit(flat_train)
    scaler.fit(imputer.transform(flat_train))

    def transform_frame(frame: pd.DataFrame) -> pd.DataFrame:
        transformed = scaler.transform(imputer.transform(frame[feature_cols].to_numpy(dtype=float)))
        transformed_df = pd.DataFrame(transformed, columns=feature_cols, index=frame.index)
        return pd.concat([frame.drop(columns=feature_cols), transformed_df], axis=1)

    train_df = transform_frame(train_df)
    test_df = transform_frame(test_df)

    classification = task == "classification"
    encoder = None
    if classification:
        encoder = LabelEncoder()
        train_df[target_col] = encoder.fit_transform(train_df[target_col].astype(str))
        known = set(encoder.classes_)
        test_df = test_df[test_df[target_col].astype(str).isin(known)].copy()
        test_df[target_col] = encoder.transform(test_df[target_col].astype(str))

    X_train, y_train, _ = make_sequences(train_df, feature_cols, target_col, classification)
    X_test, y_test, meta_test = make_sequences(test_df, feature_cols, target_col, classification)
    if len(X_train) < 30 or len(X_test) < 5:
        raise RuntimeError("not enough sequence rows")

    output_size = len(np.unique(y_train)) if classification else 1
    net = SequenceNet(architecture, X_train.shape[2], output_size, task).model
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    if classification:
        criterion = nn.CrossEntropyLoss()
        y_train_t = torch.tensor(y_train.astype(int), dtype=torch.long)
        y_test_t = torch.tensor(y_test.astype(int), dtype=torch.long)
    else:
        criterion = nn.MSELoss()
        y_train_t = torch.tensor(y_train.astype(np.float32).reshape(-1, 1))
        y_test_t = torch.tensor(y_test.astype(np.float32).reshape(-1, 1))

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=False)

    best_state = None
    best_valid = np.inf
    patience_left = PATIENCE
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        batch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():
            valid_loss = float(criterion(net(X_test_t), y_test_t).detach())
        train_loss = float(np.mean(batch_losses))
        history.append({"iteration": epoch, "train": train_loss, "valid": valid_loss, "metric": "cross_entropy" if classification else "mse"})
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        raw_pred = net(X_test_t)
        if classification:
            proba = torch.softmax(raw_pred, dim=1).numpy()
            pred_idx = proba.argmax(axis=1)
            y_true = encoder.inverse_transform(y_test.astype(int))
            y_pred = encoder.inverse_transform(pred_idx)
            metrics = classification_metrics(y_true, y_pred, proba, list(encoder.classes_))
            pred_values = y_pred
            residual_std = None
        else:
            pred = raw_pred.numpy().reshape(-1)
            y_true = y_test.astype(float)
            residual_std = float(np.nanstd(y_train.astype(float) - net(X_train_t).detach().numpy().reshape(-1)))
            metrics = add_interval_metrics(regression_metrics(y_true, pred), y_true, pred, residual_std)
            pred_values = pred

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_prefix.with_suffix(".loss.csv"), index=False)
    save_loss_curve(history_df, out_prefix.with_suffix(".loss.png"), out_prefix.name)
    joblib.dump(
        {
            "architecture": architecture,
            "state_dict": net.state_dict(),
            "feature_cols": feature_cols,
            "imputer": imputer,
            "scaler": scaler,
            "encoder": encoder,
            "target": target_col,
            "task": task,
            "config": {
                "sequence_length": SEQUENCE_LENGTH,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "dropout": DROPOUT,
                "hidden_size": HIDDEN_SIZE,
            },
        },
        out_prefix.with_suffix(".joblib"),
    )
    return metrics, meta_test, y_true if classification else y_test.astype(float), pred_values, residual_std
