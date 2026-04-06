import numpy as np
import pandas as pd


def generar_caso_de_uso_emparejar_pacientes_similares():
    """
    Genera un caso de uso aleatorio, válido y no trivial para evaluar la función
    emparejar_pacientes_similares(df, id_col, treatment_col, feature_cols).

    Retorna:
    - input_data: dict con los argumentos de entrada
    - output_data: pd.DataFrame esperado
    """
    rng = np.random.default_rng()

    for _ in range(200):
        id_col = rng.choice(["paciente_id", "id_paciente", "codigo_paciente"]).item()
        treatment_col = rng.choice(
            ["tratamiento", "programa_preventivo", "recibio_programa"]
        ).item()

        all_feature_sets = [
            ["edad", "imc", "pasos_diarios", "colesterol"],
            ["edad", "glucosa", "presion_sistolica", "horas_sueno"],
            ["imc", "pasos_diarios", "trigliceridos", "frecuencia_cardiaca"],
        ]
        feature_cols = list(rng.choice(all_feature_sets).tolist())

        n_controls = int(rng.integers(4, 7))
        n_treated = int(rng.integers(4, 7))

        control_profiles = []
        for _ in range(n_controls):
            profile = {}
            for feature_name in feature_cols:
                if feature_name == "edad":
                    value = rng.uniform(25, 80)
                elif feature_name == "imc":
                    value = rng.uniform(18, 38)
                elif feature_name == "pasos_diarios":
                    value = rng.uniform(1500, 16000)
                elif feature_name == "colesterol":
                    value = rng.uniform(130, 290)
                elif feature_name == "glucosa":
                    value = rng.uniform(70, 210)
                elif feature_name == "presion_sistolica":
                    value = rng.uniform(95, 180)
                elif feature_name == "horas_sueno":
                    value = rng.uniform(4.5, 9.5)
                elif feature_name == "trigliceridos":
                    value = rng.uniform(70, 320)
                elif feature_name == "frecuencia_cardiaca":
                    value = rng.uniform(50, 105)
                else:
                    value = rng.uniform(0, 100)
                profile[feature_name] = float(value)
            control_profiles.append(profile)

        repeated_anchor = int(rng.integers(0, n_controls))
        treated_anchors = [repeated_anchor, repeated_anchor]
        while len(treated_anchors) < n_treated:
            treated_anchors.append(int(rng.integers(0, n_controls)))
        rng.shuffle(treated_anchors)

        rows = []

        for idx, base_profile in enumerate(control_profiles):
            row = {
                id_col: f"C{idx + 1:03d}",
                treatment_col: 0,
            }
            for feature_name in feature_cols:
                row[feature_name] = float(base_profile[feature_name])

            row["hospital"] = rng.choice(["Norte", "Centro", "Sur"]).item()
            row["ruido_admin"] = int(rng.integers(1000, 9999))
            rows.append(row)

        for idx, anchor in enumerate(treated_anchors):
            base_profile = control_profiles[anchor]
            row = {
                id_col: f"T{idx + 1:03d}",
                treatment_col: 1,
            }
            for feature_name in feature_cols:
                base_value = base_profile[feature_name]
                if feature_name == "edad":
                    noise = rng.normal(0, 4.0)
                elif feature_name == "imc":
                    noise = rng.normal(0, 1.1)
                elif feature_name == "pasos_diarios":
                    noise = rng.normal(0, 1800.0)
                elif feature_name == "colesterol":
                    noise = rng.normal(0, 14.0)
                elif feature_name == "glucosa":
                    noise = rng.normal(0, 12.0)
                elif feature_name == "presion_sistolica":
                    noise = rng.normal(0, 8.0)
                elif feature_name == "horas_sueno":
                    noise = rng.normal(0, 0.7)
                elif feature_name == "trigliceridos":
                    noise = rng.normal(0, 18.0)
                elif feature_name == "frecuencia_cardiaca":
                    noise = rng.normal(0, 5.0)
                else:
                    noise = rng.normal(0, 5.0)
                row[feature_name] = float(base_value + noise)

            row["hospital"] = rng.choice(["Norte", "Centro", "Sur"]).item()
            row["ruido_admin"] = int(rng.integers(1000, 9999))
            rows.append(row)

        raw_df = pd.DataFrame(rows)

        null_id_row = {
            id_col: np.nan,
            treatment_col: int(rng.choice([0, 1])),
            "hospital": "Centro",
            "ruido_admin": int(rng.integers(1000, 9999)),
        }
        null_treatment_row = {
            id_col: "DESCARTAR_TRAT",
            treatment_col: np.nan,
            "hospital": "Norte",
            "ruido_admin": int(rng.integers(1000, 9999)),
        }

        for feature_name in feature_cols:
            null_id_row[feature_name] = float(rng.normal(0, 1))
            null_treatment_row[feature_name] = float(rng.normal(0, 1))

        raw_df = pd.concat(
            [raw_df, pd.DataFrame([null_id_row, null_treatment_row])],
            ignore_index=True,
        )

        candidate_indices = raw_df[
            raw_df[id_col].notna() & raw_df[treatment_col].notna()
        ].index.to_list()
        if len(candidate_indices) < 4:
            continue

        n_missing_cells = min(len(feature_cols) + 1, len(candidate_indices))
        missing_row_indices = rng.choice(
            candidate_indices, size=n_missing_cells, replace=False
        )
        missing_feature_names = [
            feature_cols[i % len(feature_cols)] for i in range(n_missing_cells)
        ]
        for row_idx, feature_name in zip(missing_row_indices, missing_feature_names):
            raw_df.loc[int(row_idx), feature_name] = np.nan

        raw_df = raw_df.sample(
            frac=1.0, random_state=int(rng.integers(0, 1_000_000))
        ).reset_index(drop=True)
        shuffled_columns = list(raw_df.columns)
        rng.shuffle(shuffled_columns)
        raw_df = raw_df[shuffled_columns]

        working_df = raw_df[[id_col, treatment_col] + feature_cols].copy()
        rows_before_drop = len(working_df)
        working_df = working_df.dropna(subset=[id_col, treatment_col]).copy()
        dropped_rows = rows_before_drop - len(working_df)

        if dropped_rows < 2:
            continue

        treatment_values = working_df[treatment_col].astype(int).to_numpy()
        if not np.any(treatment_values == 0) or not np.any(treatment_values == 1):
            continue

        for feature_name in feature_cols:
            median_value = float(working_df[feature_name].median())
            working_df[feature_name] = working_df[feature_name].fillna(median_value)

        if not raw_df.loc[
            raw_df[id_col].notna() & raw_df[treatment_col].notna(), feature_cols
        ].isna().any().any():
            continue

        feature_matrix = working_df[feature_cols].to_numpy(dtype=float)
        means = feature_matrix.mean(axis=0)
        stds = feature_matrix.std(axis=0, ddof=0)
        stds = np.where(stds == 0, 1.0, stds)
        standardized = (feature_matrix - means) / stds

        clean_ids = working_df[id_col].astype(str).to_numpy()
        clean_treatment = working_df[treatment_col].astype(int).to_numpy()

        treated_mask = clean_treatment == 1
        control_mask = clean_treatment == 0

        treated_ids = clean_ids[treated_mask]
        control_ids = clean_ids[control_mask]
        treated_matrix = standardized[treated_mask]
        control_matrix = standardized[control_mask]

        if len(treated_ids) == 0 or len(control_ids) == 0:
            continue

        pairwise_diff = (
            treated_matrix[:, np.newaxis, :] - control_matrix[np.newaxis, :, :]
        )
        pairwise_distances = np.sqrt(np.sum(pairwise_diff ** 2, axis=2))

        nearest_control_positions = np.argmin(pairwise_distances, axis=1)
        nearest_distances = pairwise_distances[
            np.arange(len(treated_ids)), nearest_control_positions
        ]
        nearest_control_ids = control_ids[nearest_control_positions]

        expected_output = pd.DataFrame(
            {
                "paciente_tratado": treated_ids,
                "paciente_control": nearest_control_ids,
                "distancia": nearest_distances.astype(float),
            }
        ).sort_values(by="distancia", ascending=True, kind="mergesort").reset_index(drop=True)

        surviving_rows_mask = raw_df[id_col].notna() & raw_df[treatment_col].notna()
        if not raw_df.loc[surviving_rows_mask, feature_cols].isna().any().any():
            continue

        if not expected_output["paciente_control"].duplicated().any():
            continue

        raw_feature_matrix = working_df[feature_cols].to_numpy(dtype=float)
        raw_treated_matrix = raw_feature_matrix[treated_mask]
        raw_control_matrix = raw_feature_matrix[control_mask]
        raw_diff = (
            raw_treated_matrix[:, np.newaxis, :] - raw_control_matrix[np.newaxis, :, :]
        )
        raw_distances = np.sqrt(np.sum(raw_diff ** 2, axis=2))
        raw_nearest_controls = control_ids[np.argmin(raw_distances, axis=1)]

        if np.array_equal(raw_nearest_controls, nearest_control_ids):
            continue

        if len(expected_output) != int(np.sum(treated_mask)):
            continue

        input_data = {
            "df": raw_df.copy(deep=True),
            "id_col": id_col,
            "treatment_col": treatment_col,
            "feature_cols": list(feature_cols),
        }
        output_data = expected_output.copy(deep=True)

        return input_data, output_data

    raise RuntimeError(
        "No se pudo generar un caso suficientemente robusto tras varios intentos. "
        "Vuelva a ejecutar la función."
    )

