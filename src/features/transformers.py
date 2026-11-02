"""Transformadores de feature engineering para el predictor de victorias de SSBM.

Este módulo implementa :class:`MeleeFeatureEngineer`, un transformer de Scikit-Learn
que agrega variables de dominio derivadas del estado de una partida 1v1 de
Super Smash Bros. Melee.

La versión canónica es la de **5 variables derivadas**, la misma que se usa en
``notebooks/02_data_preparation.ipynb`` y
``notebooks/03_modeling_and_model_selection.ipynb``:

* **Diferencias y ratios de dominio**: ventaja material y de daño entre P1 y P2.
* **Reglas tácticas y distancia espacial**: riesgo de edgeguard y separación física.

La variable ``matchup`` (cruce de personajes) se excluyó deliberadamente: con
solo 38 grupos de partida actuaba como un identificador de la partida y
provocaba sobreajuste. El diagnóstico está documentado en el notebook 03.

Todas las constantes de dominio están declaradas a nivel de módulo para evitar
números mágicos dispersos en las fórmulas.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags, TransformerTags

__all__ = [
    "EPSILON",
    "DERIVED_FEATURES",
    "REQUIRED_COLUMNS",
    "MeleeFeatureEngineer",
]

#: Constante de estabilización para divisiones por cero en ratios de stock.
EPSILON: float = 1e-5

#: Variables derivadas, en el orden estable en que se anexan al DataFrame.
DERIVED_FEATURES: tuple[str, ...] = (
    "stock_diff",
    "percent_diff",
    "stock_ratio",
    "spatial_distance",
    "p1_edgeguard_advantage",
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


class MeleeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Agrega variables de dominio de SSBM a un DataFrame de frames.

    El transformador se sitúa **al inicio** de un ``Pipeline`` de Scikit-Learn:
    primero calcula las variables derivadas y después un ``ColumnTransformer``
    aplica imputación y escalado únicamente a las columnas que interesen.

    El estado de la partida se modela como fotograma a fotograma (1/60 s). Cada
    fila de entrada contiene el estado completo de los dos jugadores en un
    instante concreto, por lo que todas las variables derivadas son
    deterministas y sin dependencias temporales: no requieren información de
    frames vecinos ni del resultado final de la partida.

    Notes
    -----
    La entrada **nunca se modifica**: ``fit`` y ``transform`` trabajan sobre una
    copia y anexan las columnas nuevas a esa copia.

    ``stock_ratio`` usa :data:`EPSILON` en el denominador para evitar divisiones
    por cero. Sobre los datos ya depurados, donde ``p2_stock >= 1`` en todas las
    filas, el denominador nunca se aproxima a cero y la variable queda acotada en
    [0,2500 ; 4,0000], con máximo 3,99996. El caso extremo aparece solo si este
    transformador se aplica a datos sin depurar: ``p2_stock`` llega a 0 en 26 de
    las 540 940 filas crudas y ahí la variable alcanza 400 000, lo que tras
    ``StandardScaler`` comprimiría el rango del resto de las filas hacia cero.

    ``matchup`` no se calcula en esta versión. Con solo 38 grupos de partida,
    la combinación de personajes actuaba como un identificador de la partida y
    provocaba sobreajuste (diagnóstico documentado en el notebook 03). Esta
    clase mantiene únicamente las 5 variables de dominio.
    """

    def __init__(self, epsilon: float = EPSILON) -> None:
        """Inicializa el transformador.

        Parameters
        ----------
        epsilon : float, default=EPSILON
            Constante de estabilización añadida al denominador de los ratios
            para evitar divisiones por cero. Debe ser estrictamente positiva.

        Notes
        -----
        Scikit-Learn exige que ``__init__`` declare todos los parámetros como
        argumentos con nombre y los guarde en atributos homónimos, para que
        ``get_params`` y ``clone`` funcionen dentro de un ``Pipeline``.
        """
        self.epsilon = epsilon

    # ------------------------------------------------------------------
    # API de Scikit-Learn
    # ------------------------------------------------------------------

    def __sklearn_tags__(self) -> Tags:
        """Declara el rol de transformer para la introspección de Scikit-Learn.

        Returns
        -------
        Tags
            Etiquetas con ``estimator_type='transformer'`` y sin requirement de
            ``y``. ``preserves_dtype`` se deja vacío porque el transformador
            cambia la forma del DataFrame: no preserva ni el dtype ni el número
            de columnas.
        """
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"
        tags.transformer_tags = TransformerTags(preserves_dtype=[])
        tags.target_tags.required = False
        return tags

    def fit(self, X: pd.DataFrame, y: object = None) -> "MeleeFeatureEngineer":
        """Valida la entrada y memoriza sus columnas.

        Parameters
        ----------
        X : pandas.DataFrame
            Estado de la partida con al menos las columnas de
            :data:`REQUIRED_COLUMNS`. No se modifica.
        y : object, default=None
            Vector objetivo. Se acepta por compatibilidad con la API de
            Scikit-Learn y se ignora: este transformador es no supervisado.

        Returns
        -------
        MeleeFeatureEngineer
            El propio instancia, ya ajustada.

        Raises
        ------
        TypeError
            Si ``X`` no es un ``pandas.DataFrame``.
        ValueError
            Si falta alguna columna de :data:`REQUIRED_COLUMNS`.

        Notes
        -----
        El cálculo de todas las variables derivadas es determinista, por lo que
        ``fit`` no aprende ningún parámetro y ``transform`` puede aplicarse sin
        estado previo más allá del nombre de las columnas de entrada.
        """
        frame = self._copy_and_validate(X)
        self._fitted_feature_names = list(frame.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calcula y anexa las variables de dominio de SSBM.

        Parameters
        ----------
        X : pandas.DataFrame
            Estado de la partida con al menos las columnas de
            :data:`REQUIRED_COLUMNS`. No se modifica.

        Returns
        -------
        pandas.DataFrame
            Copia de ``X`` con las columnas de :data:`DERIVED_FEATURES` anexadas
            al final, en ese orden.

        Raises
        ------
        TypeError
            Si ``X`` no es un ``pandas.DataFrame``.
        ValueError
            Si falta alguna columna de :data:`REQUIRED_COLUMNS`.

        Notes
        -----
        ``p1_off_stage`` y ``p2_off_stage`` llegan como ``bool`` en este dataset,
        de modo que la comparación contra ``1`` / ``0`` es intencional y equivale
        a un operador ``~`` sobre el booleano.
        """
        frame = self._copy_and_validate(X)

        # Familia: diferencias y ratios de dominio.
        frame["stock_diff"] = frame["p1_stock"] - frame["p2_stock"]
        frame["percent_diff"] = frame["p2_percent"] - frame["p1_percent"]
        frame["stock_ratio"] = frame["p1_stock"] / (
            frame["p2_stock"] + self.epsilon
        )

        # Familia: reglas tácticas y distancia espacial.
        delta_x = frame["p1_x"] - frame["p2_x"]
        delta_y = frame["p1_y"] - frame["p2_y"]
        frame["spatial_distance"] = np.sqrt(delta_x**2 + delta_y**2)
        frame["p1_edgeguard_advantage"] = (
            (frame["p2_off_stage"] == 1) & (frame["p1_off_stage"] == 0)
        ).astype(np.int8)

        return frame

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        """Devuelve los nombres de las columnas producidas por ``transform``.

        Parameters
        ----------
        input_features : sequence of str, optional
            Nombres de las columnas de entrada. Si se omite, se usan las
            memorizadas durante ``fit``.

        Returns
        -------
        numpy.ndarray
            Nombres de salida: las columnas de entrada seguidas de
            :data:`DERIVED_FEATURES`.

        Raises
        ------
        ValueError
            Si no se recibe ``input_features`` y el transformador no ha sido
            ajustado, o si las columnas de entrada no cubren
            :data:`REQUIRED_COLUMNS`.

        Notes
        -----
        Permite que ``Pipeline.get_feature_names_out()`` enumere el espacio de
        características completo, que es lo que se inspecciona en la sección de
        verificaciones estructurales del notebook de preparación de datos.
        """
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

        return np.asarray(base + list(DERIVED_FEATURES), dtype=object)

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    def _copy_and_validate(self, X: pd.DataFrame) -> pd.DataFrame:
        """Valida el tipo y las columnas requeridas, y devuelve una copia.

        Parameters
        ----------
        X : pandas.DataFrame
            Entrada a validar.

        Returns
        -------
        pandas.DataFrame
            Copia de ``X``, segura de modificar.

        Raises
        ------
        TypeError
            Si ``X`` no es un ``pandas.DataFrame``.
        ValueError
            Si falta alguna columna de :data:`REQUIRED_COLUMNS`.
        """
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

    # ``_fitted_feature_names`` se declara a nivel de clase para que
    # ``get_feature_names_out`` tenga un valor por defecto con el que fallar de
    # forma explícita antes del primer ``fit``.
    _fitted_feature_names: Optional[list[str]] = None
