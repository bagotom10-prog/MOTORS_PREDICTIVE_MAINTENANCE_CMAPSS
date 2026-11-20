from pathlib import Path
import time
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def asymmetric_plant_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    late_penalty: float = 2.0,
    early_penalty: float = 1.0,
) -> float:
    """
    Métrica industrial de coste por ciclo:
    - Si y_pred > y_true: predicción TARDÍA (el motor falla antes de lo avisado -> muy caro).
    - Si y_pred <= y_true: predicción TEMPRANA (mantenimiento preventivo -> coste bajo).
    """
    diff = y_pred - y_true
    weights = np.where(diff > 0, late_penalty, early_penalty)
    return float(np.mean(weights * np.abs(diff)))


def evaluate_last_cycle_per_unit(
    df: pd.DataFrame, preds: np.ndarray, late_penalty: float, early_penalty: float
) -> dict:
    """
    Evalúa el rendimiento en el último ciclo registrado de cada motor en test
    (el punto real de decisión en el benchmark CMAPSS).
    """
    eval_df = df[["unit_id", "cycle", "RUL"]].copy()
    eval_df["pred"] = preds
    last_cycles = eval_df.sort_values("cycle").groupby("unit_id").last()

    rmse = float(np.sqrt(mean_squared_error(last_cycles["RUL"], last_cycles["pred"])))
    asym_cost = asymmetric_plant_cost(
        last_cycles["RUL"].values,
        last_cycles["pred"].values,
        late_penalty=late_penalty,
        early_penalty=early_penalty,
    )
    late_rate = float(np.mean(last_cycles["pred"] > last_cycles["RUL"]) * 100.0)
    return {"rmse": rmse, "asym_cost": asym_cost, "late_rate_pct": late_rate}


def run_baseline_training(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)
    dataset_id = cfg["data"]["dataset_id"]
    seed = cfg["data"]["random_seed"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    late_pen = cfg["cost_asymmetry"]["late_prediction_penalty"]
    early_pen = cfg["cost_asymmetry"]["early_prediction_penalty"]

    tr_df = pd.read_parquet(processed_dir / f"train_tab_{dataset_id}.parquet")
    va_df = pd.read_parquet(processed_dir / f"val_tab_{dataset_id}.parquet")
    test_df = pd.read_parquet(processed_dir / f"test_tab_{dataset_id}.parquet")

    feature_cols = joblib.load(f"models/feature_cols_{dataset_id}.joblib")

    X_train, y_train = tr_df[feature_cols], tr_df["RUL"]
    X_val, y_val = va_df[feature_cols], va_df["RUL"]
    X_test = test_df[feature_cols]

    print("Entrenando modelo base LightGBM...")
    t0 = time.perf_counter()

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample_for_bin=50000,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    train_time = time.perf_counter() - t0
    joblib.dump(model, f"models/lgbm_{dataset_id}.joblib")

    # Evaluación en Validación y Test (último ciclo de cada motor)
    val_preds = np.clip(model.predict(X_val), 0, cfg["physics_and_sensors"]["rul_cap"])
    test_preds = np.clip(model.predict(X_test), 0, cfg["physics_and_sensors"]["rul_cap"])

    val_metrics = evaluate_last_cycle_per_unit(va_df, val_preds, late_pen, early_pen)
    test_metrics = evaluate_last_cycle_per_unit(test_df, test_preds, late_pen, early_pen)

    print(f"\n[RESULTADOS BASELINE LIGHTGBM] (Tiempo entreno: {train_time:.2f}s)")
    print(f" -> Validación (20 motores) | RMSE: {val_metrics['rmse']:.2f} ciclos | Coste Asimétrico: {val_metrics['asym_cost']:.2f} | % Avisos Tardíos: {val_metrics['late_rate_pct']:.1f}%")
    print(f" -> Test NASA (100 motores) | RMSE: {test_metrics['rmse']:.2f} ciclos | Coste Asimétrico: {test_metrics['asym_cost']:.2f} | % Avisos Tardíos: {test_metrics['late_rate_pct']:.1f}%")


if __name__ == "__main__":
    run_baseline_training()