"""Pruebas unitarias para canonicalización de partidas, transformers y pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from src.extract_data import assign_match_ids, canonicalize_match_ids
from src.features.transformers import (
    DERIVED_FEATURES,
    LEXICOGRAPHIC_FEATURES,
    SYMMETRIC_EDGEGUARD_FEATURES,
    MeleeFeatureEngineer,
    build_model_pipeline,
)


def _make_frame(frame: int, stage: str, p1_char: str, p2_char: str, p1_cost: int = 0, p2_cost: int = 3) -> dict:
    return {
        "frame": frame,
        "stage": stage,
        "players": {
            1: {"character": p1_char, "costume": p1_cost},
            2: {"character": p2_char, "costume": p2_cost},
        },
    }


def _make_sample_df(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "row_id": [f"0000_{i}" for i in range(n)],
            "match_id": ["0000_m01"] * (n // 2) + ["0000_m02"] * (n - n // 2),
            "frame": np.arange(n, dtype=np.int64),
            "stage": ["BATTLEFIELD"] * n,
            "p1_character": ["FOX"] * n,
            "p2_character": ["MARTH"] * n,
            "p1_stock": rng.integers(1, 5, size=n),
            "p2_stock": rng.integers(1, 5, size=n),
            "p1_percent": rng.uniform(0, 140, size=n),
            "p2_percent": rng.uniform(0, 140, size=n),
            "p1_x": rng.uniform(-50, 50, size=n),
            "p2_x": rng.uniform(-50, 50, size=n),
            "p1_y": rng.uniform(-10, 50, size=n),
            "p2_y": rng.uniform(-10, 50, size=n),
            "p1_off_stage": rng.choice([True, False], size=n),
            "p2_off_stage": rng.choice([True, False], size=n),
            "p1_is_powershield": [False] * n,
            "p2_is_powershield": [False] * n,
            "p1_facing": [True] * n,
            "p2_facing": [False] * n,
            "p1_invulnerable": [False] * n,
            "p2_invulnerable": [False] * n,
            "p1_on_ground": [True] * n,
            "p2_on_ground": [True] * n,
        }
    )


def test_assign_match_ids_unifies_sheik_zelda_contiguous():
    frames = [
        _make_frame(14680, "BATTLEFIELD", "SHEIK", "MARTH"),
        _make_frame(14681, "BATTLEFIELD", "SHEIK", "MARTH"),
        _make_frame(14682, "BATTLEFIELD", "ZELDA", "MARTH"),
        _make_frame(14683, "BATTLEFIELD", "ZELDA", "MARTH"),
        # Partida separada no contigua (reinicia en -123)
        _make_frame(-123, "BATTLEFIELD", "ZELDA", "MARTH"),
        _make_frame(-122, "BATTLEFIELD", "ZELDA", "MARTH"),
    ]
    match_map = assign_match_ids(frames, "0000")
    assert match_map[0] == match_map[1] == match_map[2] == match_map[3]
    assert match_map[4] == match_map[5]
    assert match_map[0] != match_map[4]


def test_canonicalize_match_ids_unifies_m16_m25():
    df = pd.DataFrame(
        {
            "match_id": ["0000_m16", "0000_m16", "0000_m25", "0000_m25"],
            "frame": [14680, 14681, 14682, 14683],
            "stage": ["BATTLEFIELD"] * 4,
            "p1_costume": [0, 0, 0, 0],
            "p2_costume": [3, 3, 3, 3],
            "p1_character": ["SHEIK", "SHEIK", "ZELDA", "ZELDA"],
            "p2_character": ["MARTH", "MARTH", "MARTH", "MARTH"],
        }
    )
    out = canonicalize_match_ids(df)
    assert out["match_id"].nunique() == 1
    assert set(out["match_id"]) == {"0000_m16"}


def test_transformer_immutability_and_binary_imputation():
    df = _make_sample_df(10)
    df_orig = df.copy(deep=True)

    # Introducir nulos en columnas binarias (dtype object, como ocurre al extraer nulos)
    df["p1_off_stage"] = df["p1_off_stage"].astype(object)
    df["p2_off_stage"] = df["p2_off_stage"].astype(object)
    df.loc[0, "p1_off_stage"] = None
    df.loc[1, "p2_off_stage"] = None
    df["p1_is_powershield"] = df["p1_is_powershield"].astype(object)
    df.loc[2, "p1_is_powershield"] = None

    fe = MeleeFeatureEngineer(
        include_symmetric_edgeguard=True,
        include_lexicographic_features=True,
    )
    transformed = fe.fit_transform(df)

    # El DataFrame original de entrada no debe mutar
    assert list(df.columns) == list(df_orig.columns)
    assert all(c in transformed.columns for c in DERIVED_FEATURES)
    assert all(c in transformed.columns for c in SYMMETRIC_EDGEGUARD_FEATURES)
    assert all(c in transformed.columns for c in LEXICOGRAPHIC_FEATURES)
    assert set(transformed["edgeguard_advantage"].unique()).issubset({-1, 0, 1})

    pipe = build_model_pipeline(
        feature_set="base5",
        include_symmetric_edgeguard=True,
        include_lexicographic_features=True,
    )
    mat = pipe.fit_transform(df)
    assert mat.dtype == np.float64
    assert not np.isnan(mat).any()
    assert not np.isinf(mat).any()


def test_build_model_pipeline_no_leakage_and_unfitted():
    df_train = _make_sample_df(12)
    df_test = _make_sample_df(6)
    # Desplazar fuertemente las escalas en test para detectar fuga si se usara
    df_test["p1_percent"] = df_test["p1_percent"] + 1000.0
    y_train = np.array([1, 2] * 6)

    pipe = build_model_pipeline(
        estimator=LogisticRegression(max_iter=200),
        feature_set="base5",
        include_symmetric_edgeguard=True,
    )

    with pytest.raises(NotFittedError):
        check_is_fitted(pipe)

    pipe.fit(df_train, y_train)
    scaler = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["escalado"]
    # La media aprendida debe corresponder solo a df_train (percent_diff cercano a [-140, 140]), no +1000
    assert abs(float(scaler.mean_[1])) < 200.0

    preds = pipe.predict(df_test)
    assert len(preds) == len(df_test)


def test_get_feature_names_out_consistency():
    df = _make_sample_df(8)
    fe = MeleeFeatureEngineer(
        include_symmetric_edgeguard=True,
        include_lexicographic_features=True,
    )
    out_df = fe.fit_transform(df)
    fe_names = list(fe.get_feature_names_out())
    assert fe_names == list(out_df.columns)

    pipe = build_model_pipeline(
        feature_set="base5_frame",
        include_symmetric_edgeguard=True,
        include_lexicographic_features=True,
    )
    mat = pipe.fit_transform(df)
    pipe_names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    assert len(pipe_names) == mat.shape[1]
