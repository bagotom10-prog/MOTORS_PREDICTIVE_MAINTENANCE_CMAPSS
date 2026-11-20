from pathlib import Path
import sys

# Asegura que Python encuentre el paquete 'src' al lanzar desde Streamlit
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import torch
import yaml
from sklearn.metrics import mean_squared_error

from src.features import create_sequences, get_sensor_columns
from src.train_baseline import asymmetric_plant_cost
from src.train_dl import PredictiveMaintenanceLSTM

st.set_page_config(
    page_title="CMAPSS Turbofan | Mantenimiento Predictivo Industrial",
    page_icon="⚙️",
    layout="wide",
)


@st.cache_resource
def load_assets():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_id = cfg["data"]["dataset_id"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    rul_cap = cfg["physics_and_sensors"]["rul_cap"]
    seq_len = cfg["features"]["sequence_length"]

    # Datos limpios (unidades físicas originales) y tabulares escalados
    test_clean = pd.read_parquet(processed_dir / f"test_{dataset_id}_clean.parquet")
    test_tab = pd.read_parquet(processed_dir / f"test_tab_{dataset_id}.parquet")
    feature_cols = joblib.load(f"models/feature_cols_{dataset_id}.joblib")

    # Modelo Baseline LightGBM
    lgbm_model = joblib.load(f"models/lgbm_{dataset_id}.joblib")

    # Modelo Deep Learning PyTorch LSTM
    sensor_cols = get_sensor_columns(test_clean)
    device = torch.device("cpu")
    lstm_model = PredictiveMaintenanceLSTM(
        input_size=len(sensor_cols), hidden_size=64, num_layers=2, dropout=0.2
    ).to(device)
    lstm_model.load_state_dict(
        torch.load(f"models/lstm_{dataset_id}.pt", map_location=device, weights_only=True)
    )
    lstm_model.eval()

    # Explicador SHAP para diagnóstico físico de sensores
    explainer = shap.TreeExplainer(lgbm_model)

    return cfg, test_clean, test_tab, feature_cols, sensor_cols, lgbm_model, lstm_model, explainer, rul_cap, seq_len


cfg, test_clean, test_tab, feature_cols, sensor_cols, lgbm_model, lstm_model, explainer, rul_cap, seq_len = load_assets()

# --- BARRA LATERAL DE CONTROL OPERATIVO ---
st.sidebar.title("⚙️ Control de Planta")
st.sidebar.markdown("**Activo:** Turbofán NASA CMAPSS (`FD001`)")

available_units = sorted(test_clean["unit_id"].unique())
selected_unit = st.sidebar.selectbox("Seleccionar Turbina (Unit ID):", available_units, index=33)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Política de Riesgo en Planta")
safety_buffer = st.sidebar.slider(
    "Colchón de Seguridad (ciclos):",
    min_value=0,
    max_value=25,
    value=8,
    help="Resta ciclos a la predicción nominal para evitar paradas no planificadas por fallos tempranos.",
)
warning_threshold = st.sidebar.slider("Umbral Alerta Preventiva (ciclos):", 20, 60, 35)
critical_threshold = st.sidebar.slider("Umbral Parada Inmediata (ciclos):", 5, 25, 15)

# --- INFERENCIA SOBRE LA TURBINA SELECCIONADA ---
unit_clean = test_clean[test_clean["unit_id"] == selected_unit].sort_values("cycle").reset_index(drop=True)
unit_tab = test_tab[test_tab["unit_id"] == selected_unit].sort_values("cycle").reset_index(drop=True)

# Predicción LightGBM
lgbm_preds = np.clip(lgbm_model.predict(unit_tab[feature_cols]), 0, rul_cap)

# Predicción PyTorch LSTM (ventana deslizante de 30 ciclos)
X_unit_seq, _, _ = create_sequences(unit_tab, sensor_cols, seq_len)
with torch.no_grad():
    lstm_seq_preds = np.clip(lstm_model(torch.tensor(X_unit_seq)).numpy(), 0, rul_cap)

# Alineamos longitud de secuencias LSTM con los ciclos del motor
if len(unit_tab) < seq_len:
    lstm_preds = np.repeat(lstm_seq_preds[0], len(unit_tab))
else:
    pad_front = np.repeat(lstm_seq_preds[0], seq_len - 1)
    lstm_preds = np.concatenate([pad_front, lstm_seq_preds])

# Aplicamos el colchón de seguridad operativo
lstm_safe_preds = np.clip(lstm_preds - safety_buffer, 0, rul_cap)

current_cycle = int(unit_clean["cycle"].iloc[-1])
true_rul_now = float(unit_clean["RUL"].iloc[-1])
lstm_rul_now = float(lstm_preds[-1])
safe_rul_now = float(lstm_safe_preds[-1])
lgbm_rul_now = float(lgbm_preds[-1])

# --- CABECERA Y KPIs PRINCIPALES ---
st.title("🏭 Sistema de Mantenimiento Predictivo Industrial (RUL)")
st.caption(
    "Arquitectura híbrida: PyTorch LSTM con función de pérdida asimétrica + LightGBM/SHAP para diagnóstico físico de sensores."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Ciclo Operativo Actual", f"Ciclo {current_cycle}")
col2.metric(
    "RUL PyTorch LSTM (Ajustado)",
    f"{safe_rul_now:.1f} ciclos",
    delta=f"{safe_rul_now - true_rul_now:+.1f} vs Real ({true_rul_now:.0f})",
    delta_color="normal",
)
col3.metric(
    "RUL Nominal LSTM / LightGBM",
    f"{lstm_rul_now:.1f} / {lgbm_rul_now:.1f}",
    help="Predicción pura sin colchón de seguridad.",
)

# Semáforo operativo
if safe_rul_now <= critical_threshold:
    col4.error(f"🚨 PARADA INMEDIATA (≤ {critical_threshold} ciclos)")
elif safe_rul_now <= warning_threshold:
    col4.warning(f"⚠️ PLANIFICAR MANTENIMIENTO (≤ {warning_threshold} ciclos)")
else:
    col4.success("✅ OPERACIÓN NOMINAL")

# --- PESTAÑAS DE ANÁLISIS ---
tab1, tab2, tab3 = st.tabs([
    "📈 Trayectoria de Degradación (RUL)",
    "🌡️ Telemetría de Sensores Físicos",
    "🔍 Diagnóstico SHAP & Benchmark Global",
])

with tab1:
    st.subheader(f"Evolución de Vida Útil Remanente — Turbina #{selected_unit}")
    fig_rul = go.Figure()

    fig_rul.add_trace(go.Scatter(
        x=unit_clean["cycle"], y=unit_clean["RUL"],
        mode="lines", name="RUL Real (Ground Truth)",
        line=dict(color="#1f77b4", width=3, dash="dash"),
    ))
    fig_rul.add_trace(go.Scatter(
        x=unit_clean["cycle"], y=lstm_safe_preds,
        mode="lines", name=f"PyTorch LSTM (Colchón -{safety_buffer}c)",
        line=dict(color="#2ca02c", width=3),
    ))
    fig_rul.add_trace(go.Scatter(
        x=unit_clean["cycle"], y=lgbm_preds,
        mode="lines", name="Baseline LightGBM",
        line=dict(color="#ff7f0e", width=1.8),
        opacity=0.75,
    ))

    fig_rul.add_hline(y=critical_threshold, line_dash="dot", line_color="red", annotation_text="Umbral Crítico")
    fig_rul.add_hline(y=warning_threshold, line_dash="dot", line_color="orange", annotation_text="Umbral Preventivo")

    fig_rul.update_layout(
        xaxis_title="Ciclos de Vuelo / Operación",
        yaxis_title="Vida Útil Remanente (RUL en ciclos)",
        hovermode="x unified",
        height=430,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_rul, use_container_width=True)

with tab2:
    st.subheader(f"Lecturas Físicas en Planta — Turbina #{selected_unit}")
    default_sensors = [
        s for s in ["T50_lpt_outlet_temp", "Ps30_hpc_static_pressure", "Nc_physical_core_speed", "BPR_bypass_ratio"]
        if s in sensor_cols
    ]
    chosen_sensors = st.multiselect("Sensores activos a inspeccionar:", sensor_cols, default=default_sensors)

    if chosen_sensors:
        cols_grid = st.columns(2)
        for idx, sensor in enumerate(chosen_sensors):
            with cols_grid[idx % 2]:
                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(
                    x=unit_clean["cycle"], y=unit_clean[sensor],
                    mode="lines", name=sensor, line=dict(width=2),
                ))
                fig_s.update_layout(
                    title=sensor,
                    xaxis_title="Ciclo",
                    yaxis_title="Valor Físico",
                    height=260,
                    margin=dict(l=15, r=15, t=35, b=20),
                )
                st.plotly_chart(fig_s, use_container_width=True)

with tab3:
    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.subheader(f"Diagnóstico SHAP (Último Ciclo — Turbina #{selected_unit})")
        st.caption("Variables físicas que más reducen (rojo) o extienden (azul) el RUL estimado en este instante.")

        last_row = unit_tab[feature_cols].iloc[[-1]]
        shap_vals = explainer.shap_values(last_row)[0]
        shap_df = pd.DataFrame({
            "Variable Física": feature_cols,
            "Impacto SHAP (ciclos)": shap_vals,
            "Abs": np.abs(shap_vals),
        }).sort_values("Abs", ascending=False).head(10).sort_values("Impacto SHAP (ciclos)")

        colors = ["#d62728" if v < 0 else "#1f77b4" for v in shap_df["Impacto SHAP (ciclos)"]]
        fig_shap = go.Figure(go.Bar(
            x=shap_df["Impacto SHAP (ciclos)"],
            y=shap_df["Variable Física"],
            orientation="h",
            marker_color=colors,
        ))
        fig_shap.update_layout(
            xaxis_title="Contribución al RUL (ciclos)",
            height=400,
            margin=dict(l=10, r=10, t=20, b=20),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_right:
        st.subheader("Comparativa Global (100 Motores Test NASA)")
        st.caption(f"Evaluación en el último ciclo de cada motor aplicando colchón de seguridad = {safety_buffer} ciclos.")

        late_pen = cfg["cost_asymmetry"]["late_prediction_penalty"]
        early_pen = cfg["cost_asymmetry"]["early_prediction_penalty"]

        # Evaluación global LightGBM
        all_lgbm = np.clip(lgbm_model.predict(test_tab[feature_cols]), 0, rul_cap)
        df_eval_lgbm = test_tab[["unit_id", "cycle", "RUL"]].copy()
        df_eval_lgbm["pred"] = all_lgbm
        last_lgbm = df_eval_lgbm.sort_values("cycle").groupby("unit_id").last()

        # Evaluación global PyTorch LSTM
        X_all_seq, y_all_seq, u_all_seq = create_sequences(test_tab, sensor_cols, seq_len)
        with torch.no_grad():
            all_lstm = np.clip(lstm_model(torch.tensor(X_all_seq)).numpy(), 0, rul_cap)

        lstm_last_preds, lstm_last_trues = [], []
        for uid in np.unique(u_all_seq):
            idx = np.where(u_all_seq == uid)[0][-1]
            lstm_last_preds.append(np.clip(all_lstm[idx] - safety_buffer, 0, rul_cap))
            lstm_last_trues.append(y_all_seq[idx])

        lstm_last_preds = np.array(lstm_last_preds)
        lstm_last_trues = np.array(lstm_last_trues)

        summary_df = pd.DataFrame({
            "Métrica Industrial": [
                "RMSE (Error Medio en Ciclos)",
                "Coste Asimétrico de Planta",
                "% Avisos Tardíos (Riesgo Rotura)",
            ],
            "Baseline (LightGBM)": [
                f"{np.sqrt(mean_squared_error(last_lgbm['RUL'], last_lgbm['pred'])):.2f}",
                f"{asymmetric_plant_cost(last_lgbm['RUL'].values, last_lgbm['pred'].values, late_pen, early_pen):.2f}",
                f"{np.mean(last_lgbm['pred'] > last_lgbm['RUL']) * 100:.1f}%",
            ],
            f"PyTorch LSTM (-{safety_buffer}c)": [
                f"{np.sqrt(mean_squared_error(lstm_last_trues, lstm_last_preds)):.2f}",
                f"{asymmetric_plant_cost(lstm_last_trues, lstm_last_preds, late_pen, early_pen):.2f}",
                f"{np.mean(lstm_last_preds > lstm_last_trues) * 100:.1f}%",
            ],
        })
        st.table(summary_df)
        st.info(
            "💡 **Nota de Ingeniería:** Ajusta el *Colchón de Seguridad* en la barra lateral para observar cómo disminuye el **% de Avisos Tardíos** y se protege la integridad mecánica de la flota."
        )