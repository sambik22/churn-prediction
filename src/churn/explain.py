"""Пояснення моделі за допомогою SHAP.

SHAP (SHapley Additive exPlanations) розкладає кожен прогноз на внески ознак:
``logit(p) = base_value + Σ shap_i``. Додатний внесок штовхає клієнта до відтоку,
від'ємний — утримує. Сума по всіх ознаках точно дорівнює прогнозу моделі,
тому пояснення чесне, а не приблизне.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


class ChurnExplainer:
    """Обгортка, що пояснює прогнози пайплайну в термінах ознак після препроцесингу."""

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame):
        self.pipeline = pipeline
        self.prep = pipeline[:-1]
        self.model = pipeline[-1]
        self.feature_names = list(self.prep.get_feature_names_out())
        bg = self._transform(background)
        # shap.Explainer сам обирає алгоритм: TreeExplainer для дерев,
        # LinearExplainer для логістичної регресії. Результат — у просторі логітів.
        self.explainer = shap.Explainer(self.model, bg, feature_names=self.feature_names)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            np.asarray(self.prep.transform(X), dtype=float), columns=self.feature_names
        )

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        sv = self.explainer(self._transform(X))
        if sv.values.ndim == 3:  # деякі моделі повертають внески для обох класів
            sv = sv[:, :, 1]
        return sv

    def top_factors(self, X: pd.DataFrame, k: int = 5) -> list[list[dict]]:
        """Для кожного рядка — k ознак з найбільшим за модулем внеском."""
        sv = self.explain(X)
        result = []
        for row_vals, row_data in zip(sv.values, sv.data, strict=True):
            order = np.argsort(-np.abs(row_vals))[:k]
            result.append(
                [
                    {
                        "feature": self.feature_names[i],
                        "value": float(row_data[i]),
                        "impact": float(row_vals[i]),
                        "direction": "збільшує ризик" if row_vals[i] > 0 else "зменшує ризик",
                    }
                    for i in order
                ]
            )
        return result


def save_shap_plots(explainer: ChurnExplainer, X: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sv = explainer.explain(X)

    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP: як кожна ознака впливає на ризик відтоку")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_beeswarm.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    plt.figure()
    shap.plots.bar(sv, max_display=15, show=False)
    plt.title("SHAP: середня важливість ознак |внесок|")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    # Пояснення одного клієнта з найвищим ризиком
    idx = int(np.argmax(sv.values.sum(axis=1)))
    plt.figure()
    shap.plots.waterfall(sv[idx], max_display=12, show=False)
    plt.title("SHAP: чому модель вважає цього клієнта ризиковим")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_waterfall_example.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    for feature in ("tenure", "MonthlyCharges"):
        if feature in explainer.feature_names:
            plt.figure()
            shap.plots.scatter(sv[:, feature], color=sv, show=False)
            plt.tight_layout()
            plt.savefig(out_dir / f"shap_dependence_{feature}.png", dpi=120, bbox_inches="tight")
            plt.close("all")
