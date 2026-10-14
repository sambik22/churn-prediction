"""Повний цикл навчання: python -m churn.train

1. Завантаження й очищення даних.
2. Стратифікований train/test split (тест не чіпаємо до самого кінця).
3. Крос-валідація всіх кандидатів на train, вибір найкращого за PR-AUC.
4. Вибір порогу за очікуваним прибутком на out-of-fold прогнозах train.
5. Фінальне навчання на всьому train, оцінка на test.
6. Збереження моделі, метрик і графіків (включно з SHAP).
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from churn import config
from churn.data import download_data, load_dataset
from churn.evaluate import (
    best_profit_threshold,
    classification_metrics,
    expected_profit,
    plot_model_comparison,
    plot_model_report,
)
from churn.explain import save_shap_plots
from churn.models import get_candidates
from churn.predict import ChurnModel


def cross_validate_candidates(X, y, cv) -> dict[str, dict[str, float]]:
    results = {}
    for name, pipe in get_candidates().items():
        start = time.perf_counter()
        scores = cross_validate(
            pipe, X, y, cv=cv, scoring=["roc_auc", "average_precision"], n_jobs=1
        )
        results[name] = {
            "roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "roc_auc_std": float(scores["test_roc_auc"].std()),
            "pr_auc_mean": float(scores["test_average_precision"].mean()),
            "pr_auc_std": float(scores["test_average_precision"].std()),
            "fit_seconds": round(time.perf_counter() - start, 2),
        }
        r = results[name]
        print(
            f"  {name:<14} ROC-AUC {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f}   "
            f"PR-AUC {r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}"
        )
    return results


def main(model_name: str | None = None) -> dict:
    download_data()
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    print(f"Train: {len(X_train)} рядків, Test: {len(X_test)} рядків, відтік {y.mean():.1%}")

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    print("Крос-валідація кандидатів:")
    cv_results = cross_validate_candidates(X_train, y_train, cv)

    if model_name is None:
        model_name = max(
            (n for n in cv_results if n != "baseline"),
            key=lambda n: cv_results[n]["pr_auc_mean"],
        )
    print(f"Обрана модель: {model_name}")
    pipeline = get_candidates()[model_name]

    oof_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold, _ = best_profit_threshold(y_train, oof_proba)
    print(f"Поріг за максимальним прибутком (OOF на train): {threshold:.2f}")

    pipeline.fit(X_train, y_train)
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    test_metrics = classification_metrics(y_test, test_proba, threshold)
    default_metrics = classification_metrics(y_test, test_proba, 0.5)
    profit = {
        "model_at_threshold": expected_profit(y_test, test_proba, threshold),
        "model_at_0.5": expected_profit(y_test, test_proba, 0.5),
        "contact_everyone": expected_profit(y_test, test_proba, 0.0),
        "contact_nobody": 0.0,
        "assumptions": {
            "customer_value": config.CUSTOMER_VALUE,
            "offer_cost": config.OFFER_COST,
            "offer_success_rate": config.OFFER_SUCCESS_RATE,
        },
    }
    print("Тест:", {k: round(v, 4) for k, v in test_metrics.items()})
    print("Прибуток на тесті:", {k: v for k, v in profit.items() if k != "assumptions"})

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_model_comparison(cv_results, config.FIGURES_DIR)
    plot_model_report(y_test.to_numpy(), test_proba, threshold, model_name, config.FIGURES_DIR)

    background = X_train.sample(100, random_state=config.RANDOM_STATE)
    model = ChurnModel(
        pipeline=pipeline,
        threshold=threshold,
        background=background,
        metadata={
            "model_name": model_name,
            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "n_train": len(X_train),
            "test_metrics": test_metrics,
        },
    )
    model.save()
    save_shap_plots(model.explainer, X_test, config.FIGURES_DIR)

    report = {
        "selected_model": model_name,
        "threshold": threshold,
        "cv": cv_results,
        "test": test_metrics,
        "test_at_0.5": default_metrics,
        "profit_test": profit,
    }
    (config.REPORTS_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(f"Модель збережено: {config.MODEL_PATH}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Навчання моделі відтоку")
    parser.add_argument(
        "--model",
        choices=list(get_candidates()),
        default=None,
        help="Примусово обрати модель (за замовчуванням — найкраща за CV PR-AUC)",
    )
    main(parser.parse_args().model)
