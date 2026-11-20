import numpy as np
import pandas as pd
import torch

from src.features import add_temporal_features, split_train_val_by_unit
from src.train_dl import AsymmetricMSELoss


def test_no_unit_overlap_between_train_and_val():
    """Verifica que ningún motor (unit_id) esté mezclado entre train y validación."""
    df = pd.DataFrame({
        "unit_id": np.repeat(range(1, 21), 10),
        "cycle": np.tile(range(1, 11), 20),
        "sensor_1": np.random.randn(200),
    })
    tr_df, va_df = split_train_val_by_unit(df, train_ratio=0.8, seed=42)

    train_units = set(tr_df["unit_id"].unique())
    val_units = set(va_df["unit_id"].unique())

    assert len(train_units.intersection(val_units)) == 0, "Fuga de datos: hay motores compartidos entre train y val."
    assert len(train_units) == 16
    assert len(val_units) == 4


def test_rolling_features_do_not_leak_future():
    """
    Verifica que alterar un sensor en el ciclo 10 NO cambia en absoluto
    las variables calculadas en el ciclo 9 (cero fuga del futuro).
    """
    df_original = pd.DataFrame({
        "unit_id": [1] * 10,
        "cycle": list(range(1, 11)),
        "T50": [1400.0] * 10,
    })
    df_altered = df_original.copy()
    # Simulamos un pico extremo en el ciclo 10 (última fila)
    df_altered.loc[9, "T50"] = 9999.0

    feat_orig = add_temporal_features(df_original, ["T50"], rolling_windows=[5], ewma_span=5)
    feat_alt = add_temporal_features(df_altered, ["T50"], rolling_windows=[5], ewma_span=5)

    # Hasta el ciclo 9 (índices 0 a 8), las features deben ser idénticas bit a bit
    pd.testing.assert_frame_equal(feat_orig.iloc[:9], feat_alt.iloc[:9])


def test_asymmetric_loss_penalizes_late_predictions_more():
    """Verifica que predecir tarde (y_pred > y_true) cuesta más que predecir pronto."""
    loss_fn = AsymmetricMSELoss(late_penalty=2.0, early_penalty=1.0)
    y_true = torch.tensor([50.0])

    # Error de +10 ciclos (tardío: dice 60 cuando quedan 50 -> peligro de rotura)
    late_loss = loss_fn(torch.tensor([60.0]), y_true).item()
    # Error de -10 ciclos (temprano: dice 40 cuando quedan 50 -> mantenimiento preventivo)
    early_loss = loss_fn(torch.tensor([40.0]), y_true).item()

    assert late_loss == 2.0 * early_loss