"""Model-specific feature contributions used by the demo and validation audit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import Pool

from .train_baselines import CATEGORICAL_FEATURES, TEXT_MODEL_NAMES, TIME_FEATURES

_FEATURE_NAMES_CACHE: dict[int, np.ndarray] = {}


def decompose_prediction(
    model_name: str,
    model: Any,
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    if model_name.startswith("context_catboost"):
        pool = Pool(
            features[CATEGORICAL_FEATURES + TIME_FEATURES],
            cat_features=CATEGORICAL_FEATURES,
        )
        shap_values = model.get_feature_importance(pool, type="ShapValues")[0]
        return (
            np.asarray(CATEGORICAL_FEATURES + TIME_FEATURES, dtype=object),
            np.asarray(shap_values[:-1], dtype=np.float64),
            float(shap_values[-1]),
        )
    if model_name in TEXT_MODEL_NAMES:
        vectorizer = model.named_steps["tfidf"]
        matrix = vectorizer.transform(features["text"])
        regressor = model.named_steps["regressor"]
        names = _FEATURE_NAMES_CACHE.get(id(model))
        if names is None:
            names = np.asarray(vectorizer.get_feature_names_out(), dtype=object)
            _FEATURE_NAMES_CACHE[id(model)] = names
    else:
        transformer = model.named_steps["features"]
        matrix = transformer.transform(features)
        regressor = model.named_steps["regressor"]
        names = _FEATURE_NAMES_CACHE.get(id(model))
        if names is None:
            names = np.asarray(transformer.get_feature_names_out(), dtype=object)
            _FEATURE_NAMES_CACHE[id(model)] = names
    values = matrix.multiply(regressor.coef_).toarray()[0]
    return (
        np.asarray(names, dtype=object),
        np.asarray(values, dtype=np.float64),
        float(np.asarray(regressor.intercept_).reshape(-1)[0]),
    )


def top_contributions(
    model_name: str,
    model: Any,
    features: pd.DataFrame,
    limit: int = 6,
) -> list[dict[str, Any]]:
    names, values, _ = decompose_prediction(model_name, model, features)
    indexes = np.argsort(np.abs(values))[-limit:][::-1]
    return [
        {"feature": str(names[index]), "contribution": float(values[index])}
        for index in indexes
        if abs(float(values[index])) > 1e-10
    ]
