import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


def generar_caso_de_uso_estimar_densidad_recorridos() -> dict:
    """
    Genera un caso de uso aleatorio, válido y no trivial para evaluar la función
    estimar_densidad_recorridos(df, duration_col, bandwidth, num_points).

    La función devuelve un diccionario con dos claves:
    - "input": contiene exactamente los argumentos que recibiría la función objetivo:
        * df: pd.DataFrame con una columna numérica de duraciones, valores nulos en esa
          columna y columnas adicionales irrelevantes.
        * duration_col: str con el nombre de la columna de duración que debe usarse.
        * bandwidth: float para el modelo KernelDensity con kernel gaussiano.
        * num_points: int con la cantidad de puntos de la malla donde se evaluará la densidad.
    - "output": pd.DataFrame correcto esperado con las columnas "duracion" y "densidad",
      ordenado de menor a mayor por "duracion".

    Cómo se construye el caso:
    1. Se generan duraciones aleatorias a partir de una mezcla de grupos de recorridos
       cortos, medios y largos para que la distribución no sea trivial.
    2. Se añaden valores nulos en la columna de duración para obligar a ejecutar la limpieza
       pedida por el enunciado.
    3. Se agregan columnas extra irrelevantes para verificar que la solución del estudiante
       use solo la columna indicada por duration_col.
    4. El output esperado se calcula paso a paso sin llamar a la función objetivo del
       estudiante: selección de la columna, eliminación de nulos, creación explícita de la
       malla con numpy, ajuste de KernelDensity y transformación de log-densidades con exp.

    La aleatoriedad real proviene del nombre de la columna de duración, del tamaño de la
    muestra, de la mezcla de duraciones, del ancho de banda, del número de puntos de la
    malla, de la posición de los nulos y del orden final de las filas.
    """
    rng = np.random.default_rng()

    # Se reintenta varias veces para garantizar un caso suficientemente rico.
    for _ in range(200):
        duration_col = rng.choice(
            ["duracion_minutos", "minutos_recorrido", "duracion_viaje"]
        ).item()

        # Parámetros aleatorios del caso de uso.
        bandwidth = float(np.round(rng.uniform(0.8, 3.5), 2))
        num_points = int(rng.integers(25, 61))

        # Tamaños aleatorios para una mezcla multimodal de recorridos.
        short_count = int(rng.integers(6, 12))
        medium_count = int(rng.integers(6, 12))
        long_count = int(rng.integers(5, 10))

        # Mezcla de distribuciones para crear una densidad interesante.
        short_rides = rng.normal(loc=rng.uniform(6, 10), scale=rng.uniform(1.2, 2.2), size=short_count)
        medium_rides = rng.normal(loc=rng.uniform(16, 24), scale=rng.uniform(2.0, 3.5), size=medium_count)
        long_rides = rng.normal(loc=rng.uniform(35, 50), scale=rng.uniform(4.0, 6.5), size=long_count)

        durations = np.concatenate([short_rides, medium_rides, long_rides])

        # Se fuerzan valores positivos y razonables.
        durations = np.clip(durations, 1.0, None)

        # Redondear hace el caso más natural sin volverlo trivial.
        durations = np.round(durations, 2)

        # Se crean columnas adicionales irrelevantes.
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

        # Se agregan filas con nulos en la columna objetivo para obligar a limpiar.
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

        # Se mezcla el orden de las filas para evitar posiciones rígidas.
        df = df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)

        # ------------------------------------------------------------------
        # Cálculo explícito del output esperado, sin llamar a la función objetivo.
        # ------------------------------------------------------------------

        # Paso 1: seleccionar la columna pedida y eliminar nulos.
        cleaned_series = df[[duration_col]][duration_col].dropna()
        cleaned_values = cleaned_series.to_numpy(dtype=float)

        # Se exige que el caso fuerce la limpieza y deje suficientes datos válidos.
        if len(cleaned_values) < 10:
            continue
        if df[duration_col].isna().sum() == 0:
            continue

        # Paso 2: construir la matriz de entrenamiento y la malla de evaluación.
        training_array = cleaned_values.reshape(-1, 1)
        grid = np.linspace(cleaned_values.min(), cleaned_values.max(), num_points).reshape(-1, 1)

        # La malla debe ser realmente creciente y con más de un punto.
        if num_points <= 1:
            continue
        if not np.all(np.diff(grid[:, 0]) >= 0):
            continue

        # Paso 3: ajustar KernelDensity y transformar log-densidades a densidades reales.
        kde_model = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde_model.fit(training_array)
        log_density = kde_model.score_samples(grid)
        density = np.exp(log_density)

        # Paso 4: construir exactamente el DataFrame esperado.
        expected_output = pd.DataFrame(
            {
                "duracion": grid[:, 0],
                "densidad": density,
            }
        ).sort_values(by="duracion", ascending=True, kind="mergesort").reset_index(drop=True)

        # Validaciones de no trivialidad y estructura esperada.
        if list(expected_output.columns) != ["duracion", "densidad"]:
            continue
        if len(expected_output) != num_points:
            continue
        if not np.all(np.diff(expected_output["duracion"].to_numpy()) >= 0):
            continue
        if not np.all(expected_output["densidad"].to_numpy() > 0):
            continue

        # La distribución no debe ser plana ni degenerada.
        if np.isclose(expected_output["densidad"].std(), 0.0):
            continue

        return {
            "input": {
                "df": df.copy(deep=True),
                "duration_col": duration_col,
                "bandwidth": bandwidth,
                "num_points": num_points,
            },
            #"output": expected_output.copy(deep=True),
        }

    raise RuntimeError(
        "No se pudo generar un caso de uso suficientemente robusto tras varios intentos. "
        "Vuelve a ejecutar la función."
    )
