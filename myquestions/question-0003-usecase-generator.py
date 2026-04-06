import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def generar_caso_de_uso_estimar_covarianza_regularizada():
    """
    Genera un caso de uso aleatorio, válido y no trivial para evaluar la función
    estimar_covarianza_regularizada(df, feature_cols).

    Retorna:
    - input_data: dict con los argumentos de entrada
    - output_data: pd.DataFrame esperado
    """
    rng = np.random.default_rng()

    possible_feature_sets = [
        ["pm25", "pm10", "no2", "o3"],
        ["so2", "co", "no2", "pm25"],
        ["ozono", "dioxido_nitrogeno", "pm10", "monoxido_carbono"],
    ]

    for _ in range(200):
        feature_cols = list(rng.choice(possible_feature_sets).tolist())
        rng.shuffle(feature_cols)

        n_days = int(rng.integers(12, 21))
        n_features = len(feature_cols)

        shared_factor_1 = rng.normal(
            loc=0.0, scale=rng.uniform(0.8, 1.8), size=n_days
        )
        shared_factor_2 = rng.normal(
            loc=0.0, scale=rng.uniform(0.4, 1.2), size=n_days
        )

        pollutant_data = {}
        for feature_name in feature_cols:
            baseline = rng.uniform(10.0, 90.0)
            loading_1 = rng.uniform(4.0, 11.0)
            loading_2 = rng.uniform(-6.0, 6.0)
            noise_scale = rng.uniform(0.8, 3.0)

            values = (
                baseline
                + loading_1 * shared_factor_1
                + loading_2 * shared_factor_2
                + rng.normal(loc=0.0, scale=noise_scale, size=n_days)
            )

            values = np.clip(values, 0.1, None)
            pollutant_data[feature_name] = np.round(values, 3)

        df = pd.DataFrame(pollutant_data)

        df["estacion"] = rng.choice(
            ["Norte", "Centro", "Sur", "Industrial"], size=n_days
        )
        df["temperatura"] = np.round(
            rng.normal(loc=22.0, scale=4.0, size=n_days), 2
        )
        df["codigo_dia"] = rng.integers(1000, 9999, size=n_days)

        n_null_rows = int(rng.integers(2, min(5, n_days - 2)))
        null_row_indices = rng.choice(np.arange(n_days), size=n_null_rows, replace=False)
        null_feature_choices = rng.choice(feature_cols, size=n_null_rows, replace=True)
        for row_idx, feature_name in zip(null_row_indices, null_feature_choices):
            df.loc[int(row_idx), feature_name] = np.nan

        df = df.sample(
            frac=1.0, random_state=int(rng.integers(0, 1_000_000))
        ).reset_index(drop=True)

        selected_df = df[feature_cols].copy()
        cleaned_df = selected_df.dropna(axis=0, how="any").copy()

        if len(cleaned_df) <= n_features:
            continue
        if len(cleaned_df) >= len(selected_df):
            continue

        feature_matrix = cleaned_df.to_numpy(dtype=float)

        covariance_model = LedoitWolf()
        covariance_model.fit(feature_matrix)
        covariance_matrix = covariance_model.covariance_

        expected_output = pd.DataFrame(
            covariance_matrix,
            index=feature_cols,
            columns=feature_cols,
        )

        if expected_output.shape != (n_features, n_features):
            continue
        if list(expected_output.index) != feature_cols:
            continue
        if list(expected_output.columns) != feature_cols:
            continue
        if not np.allclose(
            expected_output.to_numpy(), expected_output.to_numpy().T
        ):
            continue
        if not np.all(np.diag(expected_output.to_numpy()) > 0):
            continue

        off_diagonal = (
            expected_output.to_numpy()
            - np.diag(np.diag(expected_output.to_numpy()))
        )
        if np.all(np.isclose(off_diagonal, 0.0)):
            continue
        if np.max(np.abs(off_diagonal)) < 0.5:
            continue

        input_data = {
            "df": df.copy(deep=True),
            "feature_cols": list(feature_cols),
        }
        output_data = expected_output.copy(deep=True)

        return input_data, output_data

    raise RuntimeError(
        "No se pudo generar un caso de uso suficientemente robusto tras varios intentos. "
        "Vuelve a ejecutar la función."
    )
