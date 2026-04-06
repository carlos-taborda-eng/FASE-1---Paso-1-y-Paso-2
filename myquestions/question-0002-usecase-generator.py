import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def generar_caso_de_uso_estimar_densidad_recorridos():
    """
    Genera un caso de uso aleatorio, válido y no trivial para evaluar la función
    estimar_densidad_recorridos(df, duration_col, bandwidth, num_points).

    Retorna:
    - input_data: dict con los argumentos de entrada
    - output_data: pd.DataFrame esperado
    """
    rng = np.random.default_rng()

    for _ in range(200):
        duration_col = rng.choice(
            ["duracion_minutos", "minutos_recorrido", "duracion_viaje"]
        ).item()

        bandwidth = float(np.round(rng.uniform(0.8, 3.5), 2))
        num_points = int(rng.integers(25, 61))

        short_count = int(rng.integers(6, 12))
        medium_count = int(rng.integers(6, 12))
        long_count = int(rng.integers(5, 10))

        short_rides = rng.normal(
            loc=rng.uniform(6, 10), scale=rng.uniform(1.2, 2.2), size=short_count
        )
        medium_rides = rng.normal(
            loc=rng.uniform(16, 24), scale=rng.uniform(2.0, 3.5), size=medium_count
        )
        long_rides = rng.normal(
            loc=rng.uniform(35, 50), scale=rng.uniform(4.0, 6.5), size=long_count
        )

        durations = np.concatenate([short_rides, medium_rides, long_rides])
        durations = np.clip(durations, 1.0, None)
        durations = np.round(durations, 2)

        station_names = ["Centro", "Norte", "Sur", "Occidente", "Oriente"]
        user_types = ["casual", "suscriptor", "turista"]

        df = pd.DataFrame(
            {
                duration_col: durations,
                "estacion_salida": rng.choice(station_names, size=len(durations)),
                "tipo_usuario": rng.choice(user_types, size=len(durations)),
                "codigo_bici": rng.integers(100, 999, size=len(durations)),
            }
        )

        null_count = int(rng.integers(2, 5))
        null_rows = pd.DataFrame(
            {
                duration_col: [np.nan] * null_count,
                "estacion_salida": rng.choice(station_names, size=null_count),
                "tipo_usuario": rng.choice(user_types, size=null_count),
                "codigo_bici": rng.integers(100, 999, size=null_count),
            }
        )
        df = pd.concat([df, null_rows], ignore_index=True)

        df = df.sample(
            frac=1.0, random_state=int(rng.integers(0, 1_000_000))
        ).reset_index(drop=True)

        cleaned_series = df[[duration_col]][duration_col].dropna()
        cleaned_values = cleaned_series.to_numpy(dtype=float)

        if len(cleaned_values) < 10:
            continue
        if df[duration_col].isna().sum() == 0:
            continue

        training_array = cleaned_values.reshape(-1, 1)
        grid = np.linspace(
            cleaned_values.min(), cleaned_values.max(), num_points
        ).reshape(-1, 1)

        if num_points <= 1:
            continue
        if not np.all(np.diff(grid[:, 0]) >= 0):
            continue

        kde_model = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde_model.fit(training_array)
        log_density = kde_model.score_samples(grid)
        density = np.exp(log_density)

        expected_output = pd.DataFrame(
            {
                "duracion": grid[:, 0],
                "densidad": density,
            }
        ).sort_values(by="duracion", ascending=True, kind="mergesort").reset_index(drop=True)

        if list(expected_output.columns) != ["duracion", "densidad"]:
            continue
        if len(expected_output) != num_points:
            continue
        if not np.all(np.diff(expected_output["duracion"].to_numpy()) >= 0):
            continue
        if not np.all(expected_output["densidad"].to_numpy() > 0):
            continue
        if np.isclose(expected_output["densidad"].std(), 0.0):
            continue

        input_data = {
            "df": df.copy(deep=True),
            "duration_col": duration_col,
            "bandwidth": bandwidth,
            "num_points": num_points,
        }
        output_data = expected_output.copy(deep=True)

        return input_data, output_data

    raise RuntimeError(
        "No se pudo generar un caso de uso suficientemente robusto tras varios intentos. "
        "Vuelve a ejecutar la función."
    )
