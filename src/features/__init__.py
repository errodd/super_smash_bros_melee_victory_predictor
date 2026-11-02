"""Paquete de feature engineering para el predictor de victorias de SSBM.

Expone la versión canónica de :class:`MeleeFeatureEngineer`, con las 5 variables
de dominio usadas en los notebooks 02 y 03. La variante histórica con
``matchup`` se eliminó porque actuaba como identificador de partida y provocaba
sobreajuste con pocos grupos.
"""

from src.features.transformers import (
    DERIVED_FEATURES,
    EPSILON,
    REQUIRED_COLUMNS,
    MeleeFeatureEngineer,
)

__all__ = [
    "MeleeFeatureEngineer",
    "DERIVED_FEATURES",
    "EPSILON",
    "REQUIRED_COLUMNS",
]