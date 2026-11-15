"""Transformadores y constructores de pipeline para el predictor de victorias de SSBM.

Este módulo implementa :class:`MeleeFeatureEngineer` y :func:`build_model_pipeline`,
unificando la ingeniería de características de dominio y el preprocesamiento sin
fugas de información para los notebooks de preparación y modelado.

Todas las salidas numéricas del preprocesador se unifican en ``float64`` para
garantizar consistencia entre ``StandardScaler``, ``OneHotEncoder`` y la rama
binaria con ``SimpleImputer(strategy="most_frequent")``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils import Tags, TransformerTags

__all__ = [
    "EPSILON",
    "DERIVED_FEATURES",
    "SYMMETRIC_EDGEGUARD_FEATURES",
    "LEXICOGRAPHIC_FEATURES",
    "REQUIRED_COLUMNS",
    "BASE_NUMERIC_COLS",
    "CATEGORICAL_COLS",
    "ACTION_COLS",
    "BASE_BINARY_COLS",
    "as_float64_matrix",
    "MeleeFeatureEngineer",
    "build_preprocessor",
    "build_model_pipeline",
]

#: Constante de estabilización para divisiones por cero en ratios de stock.
EPSILON: float = 1e-5

#: Las 5 variables derivadas base, en orden estable.
DERIVED_FEATURES: tuple[str, ...] = (
    "stock_diff",
    "percent_diff",
    "stock_ratio",
    "spatial_distance",
    "p1_edgeguard_advantage",
)

#: Contraparte simétrica de edgeguard y codificación trivaluada {-1, 0, +1}.
SYMMETRIC_EDGEGUARD_FEATURES: tuple[str, ...] = (
    "p2_edgeguard_advantage",
    "edgeguard_advantage",
)

#: Variables alineadas con la regla lexicográfica del dominio.
LEXICOGRAPHIC_FEATURES: tuple[str, ...] = (
    "stocks_tied",
    "stocks_tied_x_percent_diff",
    "heuristic_pred",
    "advantage_sign",
    "advantage_magnitude",
)

#: Columnas que el transformer necesita encontrar en el DataFrame de entrada.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "p1_stock",
    "p2_stock",
    "p1_percent",
    "p2_percent",
    "p1_x",
    "p2_x",
    "p1_y",
    "p2_y",
    "p1_off_stage",
    "p2_off_stage",
)

#: Columnas numéricas crudas del estado de juego (27 columnas).
BASE_NUMERIC_COLS: tuple[str, ...] = (
    "frame",
    "p1_stock", "p2_stock", "p1_percent", "p2_percent",
    "p1_shield_strength", "p2_shield_strength",
    "p1_action_frame", "p2_action_frame",
    "p1_invulnerability_left", "p2_invulnerability_left",
    "p1_hitlag_left", "p2_hitlag_left",
    "p1_hitstun_frames_left", "p2_hitstun_frames_left",
    "p1_jumps_left", "p2_jumps_left",
    "p1_x", "p1_y", "p2_x", "p2_y",
    "p1_speed_air_x_self", "p1_speed_y_self", "p1_speed_ground_x_self",
    "p2_speed_air_x_self", "p2_speed_y_self", "p2_speed_ground_x_self",
)

CATEGORICAL_COLS: tuple[str, ...] = (
    "stage",
    "p1_character",
    "p2_character",
)

ACTION_COLS: tuple[str, ...] = (
    "p1_action",
    "p2_action",
)

BASE_BINARY_COLS: tuple[str, ...] = (
    "p1_is_powershield", "p2_is_powershield",
    "p1_facing", "p2_facing",
    "p1_invulnerable", "p2_invulnerable",
    "p1_on_ground", "p2_on_ground",
    "p1_off_stage", "p2_off_stage",
    "p1_edgeguard_advantage",
)


def as_float64_matrix(X: Any) -> np.ndarray:
    """Convierte la salida binaria imputada a matriz densa ``np.float64``."""
    return np.asarray(X, dtype=np.float64)


class MeleeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Agrega variables de dominio de SSBM a un DataFrame de frames.

    Parameters
    ----------
    epsilon : float, default=EPSILON
        Constante de estabilización añadida al denominador de ``stock_ratio``.
    include_symmetric_edgeguard : bool, default=True
        Si es ``True``, anexa ``p2_edgeguard_advantage`` y ``edgeguard_advantage``
        en ``{-1, 0, +1}`` además de ``p1_edgeguard_advantage``.
    include_lexicographic_features : bool, default=False
        Si es ``True``, anexa las variables alineadas con la regla lexicográfica
        (``stocks_tied``, ``stocks_tied_x_percent_diff``, ``heuristic_pred``,
        ``advantage_sign``, ``advantage_magnitude``).
    """

    def __init__(
        self,
        epsilon: float = EPSILON,
        include_symmetric_edgeguard: bool = True,
        include_lexicographic_features: bool = False,
    ) -> None:
        self.epsilon = epsilon
        self.include_symmetric_edgeguard = include_symmetric_edgeguard
        self.include_lexicographic_features = include_lexicographic_features

    def _derived_feature_names(self) -> tuple[str, ...]:
        cols = list(DERIVED_FEATURES)
        if self.include_symmetric_edgeguard:
            cols.extend(SYMMETRIC_EDGEGUARD_FEATURES)
        if self.include_lexicographic_features:
            cols.extend(LEXICOGRAPHIC_FEATURES)
        return tuple(cols)

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"
        tags.transformer_tags = TransformerTags(preserves_dtype=[])
        tags.target_tags.required = False
        return tags

    def fit(self, X: pd.DataFrame, y: object = None) -> "MeleeFeatureEngineer":
        frame = self._copy_and_validate(X)
        self._fitted_feature_names = list(frame.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._copy_and_validate(X)

        stock_diff = frame["p1_stock"] - frame["p2_stock"]
        percent_diff = frame["p2_percent"] - frame["p1_percent"]
        frame["stock_diff"] = stock_diff
        frame["percent_diff"] = percent_diff
        frame["stock_ratio"] = frame["p1_stock"] / (frame["p2_stock"] + self.epsilon)

        delta_x = frame["p1_x"] - frame["p2_x"]
        delta_y = frame["p1_y"] - frame["p2_y"]
        frame["spatial_distance"] = np.sqrt(delta_x**2 + delta_y**2)

        p1_adv = ((frame["p2_off_stage"] == 1) & (frame["p1_off_stage"] == 0)).astype(np.int8)
        frame["p1_edgeguard_advantage"] = p1_adv

        if self.include_symmetric_edgeguard:
            p2_adv = ((frame["p1_off_stage"] == 1) & (frame["p2_off_stage"] == 0)).astype(np.int8)
            frame["p2_edgeguard_advantage"] = p2_adv
            frame["edgeguard_advantage"] = (p1_adv - p2_adv).astype(np.int8)

        if self.include_lexicographic_features:
            stocks_tied = (stock_diff == 0).astype(np.int8)
            frame["stocks_tied"] = stocks_tied
            frame["stocks_tied_x_percent_diff"] = (stocks_tied * percent_diff).astype(np.float64)
            frame["heuristic_pred"] = np.where(
                (stock_diff > 0) | ((stock_diff == 0) & (percent_diff >= 0)),
                1,
                0,
            ).astype(np.int8)
            frame["advantage_sign"] = np.where(
                stock_diff != 0,
                np.sign(stock_diff),
                np.sign(percent_diff),
            ).astype(np.int8)
            frame["advantage_magnitude"] = (
                np.abs(stock_diff) * 100.0
                + np.where(stock_diff == 0, np.abs(percent_diff), 0.0)
            ).astype(np.float64)

        return frame

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        if input_features is None:
            if self._fitted_feature_names is None:
                raise ValueError(
                    "No se pueden obtener los nombres de salida: hay que "
                    "llamar a fit antes, o pasar input_features explícitamente."
                )
            base = list(self._fitted_feature_names)
        else:
            base = list(input_features)

        missing = [c for c in REQUIRED_COLUMNS if c not in base]
        if missing:
            raise ValueError(
                f"Faltan columnas requeridas en input_features: {missing}"
            )

        return np.asarray(base + list(self._derived_feature_names()), dtype=object)

    def _copy_and_validate(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"MeleeFeatureEngineer espera un pandas.DataFrame, "
                f"recibió {type(X).__name__}."
            )

        missing = [column for column in REQUIRED_COLUMNS if column not in X.columns]
        if missing:
            raise ValueError(
                f"Faltan columnas requeridas en la entrada: {missing}. "
                f"Columnas presentes: {list(X.columns)}"
            )

        return X.copy()

    _fitted_feature_names: Optional[list[str]] = None


def build_preprocessor(
    feature_set: str = "base5",
    include_symmetric_edgeguard: bool = True,
    include_lexicographic_features: bool = False,
    include_actions: bool = False,
    categorical_min_frequency: Optional[int | float] = None,
) -> ColumnTransformer:
    """Construye un ``ColumnTransformer`` sin ajustar con dtype uniforme ``float64``.

    Presets de ``feature_set``:
    - ``"base5"``: 4 numéricas derivadas + edgeguard (5 base o 7 si simétrico).
    - ``"base5_frame"``: ``base5`` + ``frame``.
    - ``"base5_abs"``: ``base5`` + ``p1_stock``, ``p2_stock``, ``p1_percent``, ``p2_percent``.
    - ``"numeric31"``: todas las numéricas crudas (27) + derivadas numéricas (4) + binarias base.
    - ``"full"``: ``numeric31`` + categóricas (``stage``, ``p1_character``, ``p2_character``).
    """
    valid_sets = {"base5", "base5_frame", "base5_abs", "numeric31", "full"}
    if feature_set not in valid_sets:
        raise ValueError(f"feature_set inválido: {feature_set}. Opciones: {sorted(valid_sets)}")

    num_cols: list[str] = []
    cat_cols: list[str] = []
    bin_cols: list[str] = []

    if feature_set == "base5":
        num_cols = ["stock_diff", "percent_diff", "stock_ratio", "spatial_distance"]
        bin_cols = ["p1_edgeguard_advantage"]
    elif feature_set == "base5_frame":
        num_cols = ["frame", "stock_diff", "percent_diff", "stock_ratio", "spatial_distance"]
        bin_cols = ["p1_edgeguard_advantage"]
    elif feature_set == "base5_abs":
        num_cols = [
            "p1_stock", "p2_stock", "p1_percent", "p2_percent",
            "stock_diff", "percent_diff", "stock_ratio", "spatial_distance",
        ]
        bin_cols = ["p1_edgeguard_advantage"]
    elif feature_set == "numeric31":
        num_cols = list(BASE_NUMERIC_COLS) + [
            "stock_diff", "percent_diff", "stock_ratio", "spatial_distance"
        ]
        bin_cols = list(BASE_BINARY_COLS)
    elif feature_set == "full":
        num_cols = list(BASE_NUMERIC_COLS) + [
            "stock_diff", "percent_diff", "stock_ratio", "spatial_distance"
        ]
        cat_cols = list(CATEGORICAL_COLS)
        bin_cols = list(BASE_BINARY_COLS)

    if include_symmetric_edgeguard:
        if "p2_edgeguard_advantage" not in bin_cols:
            bin_cols.append("p2_edgeguard_advantage")
        if "edgeguard_advantage" not in num_cols:
            num_cols.append("edgeguard_advantage")

    if include_lexicographic_features:
        num_cols.extend(["stocks_tied_x_percent_diff", "advantage_sign", "advantage_magnitude"])
        bin_cols.extend(["stocks_tied", "heuristic_pred"])

    if include_actions:
        cat_cols.extend(ACTION_COLS)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="median")),
            ("escalado", StandardScaler()),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputacion", SimpleImputer(strategy="most_frequent")),
            ("to_float64", FunctionTransformer(as_float64_matrix, feature_names_out="one-to-one")),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", numeric_pipeline, num_cols),
        ("bin", binary_pipeline, bin_cols),
    ]

    if cat_cols:
        ohe_kwargs: dict[str, Any] = {
            "handle_unknown": "ignore",
            "sparse_output": False,
            "dtype": np.float64,
        }
        if categorical_min_frequency is not None:
            ohe_kwargs["min_frequency"] = categorical_min_frequency

        categorical_pipeline = Pipeline(
            steps=[
                ("imputacion", SimpleImputer(strategy="constant", fill_value="missing")),
                ("codificacion", OneHotEncoder(**ohe_kwargs)),
            ]
        )
        transformers.insert(1, ("cat", categorical_pipeline, cat_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_pipeline(
    estimator: Optional[BaseEstimator] = None,
    feature_set: str = "base5",
    include_symmetric_edgeguard: bool = True,
    include_lexicographic_features: bool = False,
    include_actions: bool = False,
    categorical_min_frequency: Optional[int | float] = None,
    epsilon: float = EPSILON,
) -> Pipeline:
    """Construye un ``Pipeline`` de Scikit-Learn sin ajustar.

    Devuelve ``features -> preprocess`` si ``estimator is None``, o
    ``features -> preprocess -> clf`` si se pasa un estimador.
    """
    fe = MeleeFeatureEngineer(
        epsilon=epsilon,
        include_symmetric_edgeguard=include_symmetric_edgeguard,
        include_lexicographic_features=include_lexicographic_features,
    )
    preprocessor = build_preprocessor(
        feature_set=feature_set,
        include_symmetric_edgeguard=include_symmetric_edgeguard,
        include_lexicographic_features=include_lexicographic_features,
        include_actions=include_actions,
        categorical_min_frequency=categorical_min_frequency,
    )
    steps: list[tuple[str, Any]] = [
        ("features", fe),
        ("preprocess", preprocessor),
    ]
    if estimator is not None:
        steps.append(("clf", estimator))
    return Pipeline(steps=steps)
