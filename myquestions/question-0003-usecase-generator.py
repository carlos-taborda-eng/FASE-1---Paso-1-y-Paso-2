import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def generar_caso_de_uso_estimar_covarianza_regularizada() -> dict:
    """
    Genera un caso de uso aleatorio, válido y no trivial para evaluar la función
    estimar_covarianza_regularizada(df, feature_cols).

    La función devuelve un diccionario con dos claves:
    - "input": contiene exactamente los argumentos que recibiría la función objetivo:
        * df: pd.DataFrame con columnas numéricas de contaminantes, algunas filas con nulos
          dentro de feature_cols y columnas adicionales irrelevantes.
        * feature_cols: list[str] con el nombre y el orden exacto de las variables que deben
          usarse para limpiar los datos, convertirlos a numpy y estimar la covarianza.
    - "output": pd.DataFrame cuadrado con la matriz de covarianza regularizada esperada,
      calculada con LedoitWolf y con filas y columnas en el mismo orden de feature_cols.

    Cómo se construye el caso:
    1. Se generan mediciones sintéticas de contaminantes a partir de uno o dos factores latentes
       compartidos, de modo que existan correlaciones reales entre variables y la matriz de
       covarianza no sea trivial.
    2. Se añaden columnas extra irrelevantes para comprobar que la solución del estudiante use
       únicamente feature_cols.
    3. Se insertan valores nulos en distintas filas de feature_cols para obligar a eliminar filas
       incompletas, tal como exige el enunciado.
    4. El output esperado se calcula explícitamente sin llamar a la función objetivo del estudiante:
       selección de columnas, eliminación de filas con nulos, conversión a numpy flotante,
       ajuste de LedoitWolf y construcción del DataFrame cuadrado final.

    La aleatoriedad real proviene de:
    - los nombres y el orden de las columnas elegidas como feature_cols,
    - el número de días simulados,
    - las cargas de los factores latentes,
    - el ruido añadido a cada contaminante,
    - las filas y columnas donde se insertan nulos,
    - y el orden final de las filas del DataFrame.
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

        # Factores latentes compartidos para crear correlaciones reales entre contaminantes.
        shared_factor_1 = rng.normal(loc=0.0, scale=rng.uniform(0.8, 1.8), size=n_days)
        shared_factor_2 = rng.normal(loc=0.0, scale=rng.uniform(0.4, 1.2), size=n_days)

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

            # Se limita a valores positivos y se redondea para que el caso parezca natural.
            values = np.clip(values, 0.1, None)
            pollutant_data[feature_name] = np.round(values, 3)

        df = pd.DataFrame(pollutant_data)

        # Columnas extra irrelevantes para comprobar selección correcta de feature_cols.
        df["estacion"] = rng.choice(["Norte", "Centro", "Sur", "Industrial"], size=n_days)
        df["temperatura"] = np.round(rng.normal(loc=22.0, scale=4.0, size=n_days), 2)
        df["codigo_dia"] = rng.integers(1000, 9999, size=n_days)

        # Insertar nulos reales dentro de feature_cols para forzar dropna.
        n_null_rows = int(rng.integers(2, min(5, n_days - 2)))
        null_row_indices = rng.choice(np.arange(n_days), size=n_null_rows, replace=False)
        null_feature_choices = rng.choice(feature_cols, size=n_null_rows, replace=True)
        for row_idx, feature_name in zip(null_row_indices, null_feature_choices):
            df.loc[int(row_idx), feature_name] = np.nan

        # Se mezcla el orden de las filas para evitar estructuras rígidas.
        df = df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)

        # ------------------------------------------------------------------
        # Cálculo explícito del output esperado, sin usar la función objetivo.
        # ------------------------------------------------------------------
        selected_df = df[feature_cols].copy()
        cleaned_df = selected_df.dropna(axis=0, how="any").copy()

        if len(cleaned_df) <= n_features:
            continue
        if len(cleaned_df) >= len(selected_df):
            continue

        feature_matrix = cleaned_df.to_numpy(dtype=float)

        # Ajuste explícito del estimador requerido por el enunciado.
        covariance_model = LedoitWolf()
        covariance_model.fit(feature_matrix)
        covariance_matrix = covariance_model.covariance_

        expected_output = pd.DataFrame(
            covariance_matrix,
            index=feature_cols,
            columns=feature_cols,
        )

        # Validaciones de no trivialidad y estructura esperada.
        if expected_output.shape != (n_features, n_features):
            continue
        if list(expected_output.index) != feature_cols:
            continue
        if list(expected_output.columns) != feature_cols:
            continue

        # La matriz de covarianza debe ser simétrica y con diagonal positiva.
        if not np.allclose(expected_output.to_numpy(), expected_output.to_numpy().T):
            continue
        if not np.all(np.diag(expected_output.to_numpy()) > 0):
            continue

        # El caso no debe ser casi diagonal: debe haber covarianzas cruzadas relevantes.
        off_diagonal = expected_output.to_numpy() - np.diag(np.diag(expected_output.to_numpy()))
        if np.all(np.isclose(off_diagonal, 0.0)):
            continue
        if np.max(np.abs(off_diagonal)) < 0.5:
            continue

        return {
            "input": {
                "df": df.copy(deep=True),
                "feature_cols": list(feature_cols),
            },
            #"output": expected_output.copy(deep=True),
        }

    raise RuntimeError(
        "No se pudo generar un caso de uso suficientemente robusto tras varios intentos. "
        "Vuelve a ejecutar la función."
    )
