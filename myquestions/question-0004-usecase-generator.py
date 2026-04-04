import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA


def generar_caso_de_uso_calcular_correlacion_canonica() -> dict:
    """
    Genera un caso de uso aleatorio, válido y no trivial para evaluar la función
    calcular_correlacion_canonica(df, habit_cols, result_cols).

    La función devuelve un diccionario con dos claves:
    - "input": contiene exactamente los argumentos que recibiría la función objetivo:
        * df: pd.DataFrame con columnas numéricas de hábitos de estudio y desempeño,
          algunas filas con valores nulos en esas columnas y columnas extra irrelevantes.
        * habit_cols: list[str] con el nombre y el orden exacto de las variables de hábitos.
        * result_cols: list[str] con el nombre y el orden exacto de las variables de resultados.
    - "output": pd.DataFrame de una sola fila con las columnas
      "correlacion_canonica" y "n_muestras_utilizadas", calculado paso a paso
      sin llamar a la función objetivo del estudiante.

    Cómo se construye el caso:
    1. Se generan dos bloques numéricos, uno de hábitos y otro de resultados,
       a partir de factores latentes compartidos y ruido específico. Esto crea una
       relación lineal real entre ambos bloques y evita casos triviales.
    2. Se añaden columnas extra irrelevantes para comprobar que la solución utilice
       únicamente las columnas indicadas en habit_cols y result_cols.
    3. Se insertan valores nulos en distintas filas y columnas de ambos bloques para
       obligar a ejecutar la limpieza pedida por el enunciado.
    4. El output esperado se calcula explícitamente: selección de columnas,
       eliminación de filas con nulos, conversión a arreglos numpy flotantes,
       ajuste de CCA(n_components=1), transformación de ambos bloques y cálculo
       de la correlación de Pearson con numpy.corrcoef.

    La aleatoriedad real proviene de:
    - los nombres y el orden de las columnas elegidas para cada bloque,
    - el número de estudiantes simulados,
    - los factores latentes y sus cargas,
    - el ruido añadido a cada variable,
    - la ubicación de los valores nulos,
    - y el orden final de las filas del DataFrame.
    """
    rng = np.random.default_rng()

    possible_habit_sets = [
        ["horas_estudio", "tareas_entregadas", "sesiones_plataforma"],
        ["minutos_revision", "ejercicios_resueltos", "dias_activos"],
        ["lecturas_completadas", "practicas_realizadas", "horas_tutoria"],
    ]
    possible_result_sets = [
        ["nota_quices", "nota_parciales", "nota_final"],
        ["puntaje_proyecto", "puntaje_examen", "promedio_academico"],
        ["resultado_modulo_1", "resultado_modulo_2", "resultado_final"],
    ]

    for _ in range(200):
        habit_cols = list(rng.choice(possible_habit_sets).tolist())
        result_cols = list(rng.choice(possible_result_sets).tolist())
        rng.shuffle(habit_cols)
        rng.shuffle(result_cols)

        n_students = int(rng.integers(18, 31))
        n_habits = len(habit_cols)
        n_results = len(result_cols)

        # Factores latentes compartidos y específicos para construir dependencia real
        # entre los dos bloques sin volverlos idénticos.
        shared_factor = rng.normal(loc=0.0, scale=rng.uniform(0.9, 1.8), size=n_students)
        habit_specific_factor = rng.normal(loc=0.0, scale=rng.uniform(0.4, 1.0), size=n_students)
        result_specific_factor = rng.normal(loc=0.0, scale=rng.uniform(0.4, 1.0), size=n_students)

        habit_data = {}
        for feature_name in habit_cols:
            baseline = rng.uniform(4.0, 12.0)
            shared_loading = rng.uniform(1.4, 3.2)
            specific_loading = rng.uniform(-1.4, 1.4)
            noise_scale = rng.uniform(0.4, 1.1)

            values = (
                baseline
                + shared_loading * shared_factor
                + specific_loading * habit_specific_factor
                + rng.normal(loc=0.0, scale=noise_scale, size=n_students)
            )
            habit_data[feature_name] = np.round(values, 3)

        result_data = {}
        for feature_name in result_cols:
            baseline = rng.uniform(50.0, 85.0)
            shared_loading = rng.uniform(6.0, 14.0)
            specific_loading = rng.uniform(-5.0, 5.0)
            noise_scale = rng.uniform(1.5, 4.5)

            values = (
                baseline
                + shared_loading * shared_factor
                + specific_loading * result_specific_factor
                + rng.normal(loc=0.0, scale=noise_scale, size=n_students)
            )
            result_data[feature_name] = np.round(values, 3)

        df = pd.DataFrame({**habit_data, **result_data})

        # Columnas adicionales irrelevantes para comprobar selección correcta.
        df["programa"] = rng.choice(["A", "B", "C"], size=n_students)
        df["edad"] = rng.integers(17, 35, size=n_students)
        df["codigo_estudiante"] = rng.integers(10000, 99999, size=n_students)

        # Insertar nulos reales en ambos bloques para forzar dropna sobre todas las
        # columnas requeridas por el problema.
        total_nulls = int(rng.integers(3, 7))
        candidate_rows = np.arange(n_students)
        null_rows = rng.choice(candidate_rows, size=total_nulls, replace=False)
        all_target_cols = habit_cols + result_cols
        null_cols = rng.choice(all_target_cols, size=total_nulls, replace=True)
        for row_idx, col_name in zip(null_rows, null_cols):
            df.loc[int(row_idx), col_name] = np.nan

        # Mezclar el orden de las filas para evitar dependencias de posición.
        df = df.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)

        # ------------------------------------------------------------------
        # Cálculo explícito del output esperado, sin usar la función objetivo.
        # ------------------------------------------------------------------
        selected_df = df[habit_cols + result_cols].copy()
        cleaned_df = selected_df.dropna(axis=0, how="any").copy()

        dropped_rows = len(selected_df) - len(cleaned_df)
        if dropped_rows < 2:
            continue
        if len(cleaned_df) <= max(n_habits, n_results) + 2:
            continue

        X = cleaned_df[habit_cols].to_numpy(dtype=float)
        Y = cleaned_df[result_cols].to_numpy(dtype=float)

        # Validar que no haya columnas constantes tras la limpieza.
        if np.any(np.std(X, axis=0, ddof=0) == 0):
            continue
        if np.any(np.std(Y, axis=0, ddof=0) == 0):
            continue

        cca_model = CCA(n_components=1)
        cca_model.fit(X, Y)
        X_c, Y_c = cca_model.transform(X, Y)

        canonical_correlation = float(np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1])
        if not np.isfinite(canonical_correlation):
            continue

        # El caso no debe ser trivial: la correlación canónica debe reflejar una
        # asociación clara pero no una identidad perfecta por construcción.
        if canonical_correlation < 0.35:
            continue
        if np.isclose(abs(canonical_correlation), 1.0):
            continue

        expected_output = pd.DataFrame(
            {
                "correlacion_canonica": [canonical_correlation],
                "n_muestras_utilizadas": [int(len(cleaned_df))],
            }
        )

        # Validaciones finales de estructura esperada.
        if list(expected_output.columns) != ["correlacion_canonica", "n_muestras_utilizadas"]:
            continue
        if expected_output.shape != (1, 2):
            continue
        if expected_output.loc[0, "n_muestras_utilizadas"] != len(cleaned_df):
            continue

        return {
            "input": {
                "df": df.copy(deep=True),
                "habit_cols": list(habit_cols),
                "result_cols": list(result_cols),
            },
            #"output": expected_output.copy(deep=True),
        }

    raise RuntimeError(
        "No se pudo generar un caso de uso suficientemente robusto tras varios intentos. "
        "Vuelve a ejecutar la función."
    )
