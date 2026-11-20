from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from src.features import add_temporal_features, create_sequences, get_sensor_columns
from src.train_dl import PredictiveMaintenanceLSTM


class RULPredictorService:
    """Servicio desacoplado de inferencia para planta (PyTorch LSTM + LightGBM)."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        dataset_id = self.cfg["data"]["dataset_id"]
        self.rul_cap = self.cfg["physics_and_sensors"]["rul_cap"]
        self.seq_len = self.cfg["features"]["sequence_length"]
        self.rolling_windows = self.cfg["features"]["rolling_windows"]
        self.ewma_span = self.cfg["features"]["ewma_span"]

        self.scaler = joblib.load(f"models/scaler_{dataset_id}.joblib")
        self.feature_cols = joblib.load(f"models/feature_cols_{dataset_id}.joblib")
        self.lgbm_model = joblib.load(f"models/lgbm_{dataset_id}.joblib")

        self.sensor_cols = [c for c in self.feature_cols if not any(k in c for k in ["_ewma", "_roll_", "op_setting"])]
        self.device = torch.device("cpu")
        self.lstm_model = PredictiveMaintenanceLSTM(
            input_size=len(self.sensor_cols), hidden_size=64, num_layers=2, dropout=0.2
        ).to(self.device)
        self.lstm_model.load_state_dict(
            torch.load(f"models/lstm_{dataset_id}.pt", map_location=self.device, weights_only=True)
        )
        self.lstm_model.eval()

    def predict_unit_rul(self, unit_raw_df: pd.DataFrame, safety_buffer: int = 8) -> dict:
        """Recibe lecturas crudas de un motor y devuelve el diagnóstico actual."""
        feat_df = add_temporal_features(unit_raw_df, self.sensor_cols, self.rolling_windows, self.ewma_span)
        feat_df[self.feature_cols] = self.scaler.transform(feat_df[self.feature_cols])

        X_seq, _, _ = create_sequences(feat_df, self.sensor_cols, self.seq_len)
        with torch.no_grad():
            lstm_raw = float(np.clip(self.lstm_model(torch.tensor(X_seq[-1:])).item(), 0, self.rul_cap))

        safe_rul = max(0.0, lstm_raw - safety_buffer)
        lgbm_rul = float(np.clip(self.lgbm_model.predict(feat_df[self.feature_cols].iloc[[-1]])[0], 0, self.rul_cap))

        return {
            "unit_id": int(unit_raw_df["unit_id"].iloc[-1]),
            "current_cycle": int(unit_raw_df["cycle"].iloc[-1]),
            "rul_lstm_nominal": round(lstm_raw, 2),
            "rul_lstm_safe": round(safe_rul, 2),
            "rul_lgbm_baseline": round(lgbm_rul, 2),
        }