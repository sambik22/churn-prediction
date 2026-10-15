# Customer Churn Prediction

**English** | [Українська](README.uk.md)

An end-to-end ML project, from a raw CSV to a REST API in Docker. The model predicts whether
a telecom customer will leave, **explains why** (SHAP), and picks the decision threshold by the
**expected profit** of a retention campaign instead of the default 0.5.

| | |
|---|---|
| **Task** | Binary classification: will the customer churn (`Churn = 1`) or not |
| **Data** | [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d), 7,043 customers, 19 features, 26.5% churn |
| **Models** | Dummy → Logistic Regression → Random Forest → XGBoost → LightGBM |
| **Result (test)** | ROC-AUC **0.843**, PR-AUC **0.665**, recall 0.70, precision 0.57 |
| **Business impact (test)** | **+$16,300** vs **−$14,350** when sending the discount to everyone |
| **Stack** | pandas, scikit-learn, XGBoost, LightGBM, SHAP, FastAPI, Docker, pytest, GitHub Actions |

![SHAP model explanation](reports/figures/shap_beeswarm.png)

---

## Contents

1. [Business problem](#1-business-problem)
2. [Quick start](#2-quick-start)
3. [Project structure](#3-project-structure)
4. [EDA: what the data tells us](#4-eda-what-the-data-tells-us)
5. [Feature engineering](#5-feature-engineering)
6. [Modeling and validation](#6-modeling-and-validation)
7. [Threshold: from probability to decision](#7-threshold-from-probability-to-decision)
8. [Explaining the model (SHAP)](#8-explaining-the-model-shap)
9. [REST API](#9-rest-api)
10. [Docker and CI](#10-docker-and-ci)
11. [Limitations and next steps](#11-limitations-and-next-steps)

---

## 1. Business problem

In telecom, acquiring a new customer costs several times more than keeping an existing one.
So the operator wants to find customers who are about to leave **in advance** and offer them
something valuable (a discount, a better plan, free tech support).

The model has to answer three questions:
1. **Who** should we try to retain? (churn probability)
2. **Is it worth** spending budget on them? (a threshold that maximizes profit)
3. **Why** are they at risk, and what should we offer? (SHAP factors for every customer)

## 2. Quick start

```bash
git clone https://github.com/sambik22/churn-prediction.git
cd churn-prediction
python -m venv .venv && source .venv/bin/activate

make install     # dependencies + the churn package in editable mode
make train       # downloads the data, compares models, saves the best one (~30 s)
make test        # unit and API tests
make api         # http://localhost:8000/docs
```

Or everything in Docker (the model is trained during the build):

```bash
make docker-build && make docker-run
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d @scripts/example_customer.json
```

## 3. Project structure

```
churn-prediction/
├── notebooks/
│   ├── 01_eda.ipynb                         # exploratory analysis, explained step by step
│   ├── 02_modeling_and_explainability.ipynb # models, threshold, SHAP
│   └── uk/                                  # the same notebooks in Ukrainian
├── src/churn/                # all logic lives in a regular Python package
│   ├── config.py             # paths, constants, business assumptions
│   ├── data.py               # download (with SHA-256 check) and cleaning
│   ├── features.py           # feature engineering as an sklearn transformer
│   ├── models.py             # candidate models as full pipelines
│   ├── evaluate.py           # metrics, profit, plots
│   ├── explain.py            # SHAP explanations
│   ├── predict.py            # saving / loading the model
│   └── train.py              # full training run: python -m churn.train
├── api/                      # FastAPI service
├── tests/                    # pytest: data, features, profit logic, API
├── reports/
│   ├── figures/              # all plots
│   └── metrics.json          # metrics of the latest run
├── Dockerfile                # multi-stage: train → slim runtime
├── Makefile
└── .github/workflows/ci.yml  # lint + tests + training + Docker smoke test
```

**Why is the code in `src/` and not in notebooks?** Notebooks are great for exploration but
hard to test, reuse and review. So the logic lives in a package, and the notebooks only call
it and explain the results. Training, tests and the API all use the same
`ChurnFeatureEngineer`, so "research" and "production" cannot drift apart.

## 4. EDA: what the data tells us

Full analysis: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

**Data quality.** `TotalCharges` is stored as a string, and 11 values are blanks. All of these
customers have `tenure = 0`: they have not been billed yet, so the right value is 0, not the
median, and the rows should not be dropped.

**Key observations** (churn rate within the group; overall average 26.5%):

| Factor | Churn |
|---|---|
| Contract month-to-month / one year / two years | **43%** / 11% / 3% |
| Tenure 0–6 months / over 4 years | **53%** / 9.5% |
| Payment by electronic check / automatic payment | **45%** / 15–17% |
| Internet: fiber optic / DSL | **42%** / 19% |
| Neither OnlineSecurity nor TechSupport / at least one of them | **49%** / 17% |
| 0 add-on services / 6 services (customers with internet) | **52%** / 5% |

The profile of a high-risk customer: **new, on a month-to-month contract, on expensive fiber
optic, with no add-on services, paying manually.**

## 5. Feature engineering

Every new feature is a hypothesis from the EDA, encoded explicitly
([`src/churn/features.py`](src/churn/features.py)):

| Feature | Formula | Hypothesis |
|---|---|---|
| `num_addon_services` | number of active add-on services | more services means higher switching costs |
| `avg_monthly_spend` | `TotalCharges / tenure` | average actual payment |
| `charge_delta` | `MonthlyCharges − avg_monthly_spend` | a change in the bill triggers churn |
| `tenure_years` | `tenure / 12` | convenient scale |
| `is_month_to_month` | month-to-month contract | nothing keeps the customer |
| `is_electronic_check` | manual payment by check | no autopay inertia |
| `has_fiber` | fiber optic | expensive plan / quality issues |
| `no_protection` | has internet, but neither Security nor TechSupport | weak tie to the service |
| `is_new_customer` | `tenure ≤ 6` | the riskiest period |

Then a `ColumnTransformer` standardizes numbers (only for logistic regression) and one-hot
encodes categories. **All of this is part of the `Pipeline`**, so the transformers are fitted
only on the training data of each fold, and there is no data leakage.

SHAP later confirmed that 6 of the engineered features made it into the top 10 most important.

## 6. Modeling and validation

```
7043 ──stratified split──► train 80% ──► 5-fold CV: model comparison
                        │              └─► out-of-fold predictions: threshold choice
                        └► test 20%  ────► one final evaluation
```

**Metrics.** Accuracy is misleading here: a model that says "nobody leaves" has 73.5% accuracy
and zero value. So we use:
* **ROC-AUC**: how well the model *ranks* customers by risk;
* **PR-AUC**: quality on the rarer class specifically (the main model selection metric).

**Class imbalance** is handled with weights (`class_weight="balanced"`, `scale_pos_weight`)
rather than oversampling: it is simpler and does not create synthetic customers.

**5-fold cross-validation results on train:**

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| baseline (dummy) | 0.500 | 0.265 |
| Logistic Regression | 0.847 ± 0.012 | 0.660 ± 0.016 |
| Random Forest | 0.848 ± 0.010 | 0.662 ± 0.024 |
| **XGBoost** | **0.847 ± 0.010** | **0.664 ± 0.022** |
| LightGBM | 0.842 ± 0.009 | 0.654 ± 0.022 |

![Model comparison](reports/figures/model_comparison.png)

**The honest conclusion:** the gap between logistic regression, Random Forest and XGBoost
(≈0.003) is several times smaller than the spread across folds (≈0.02). No algorithm "won".
That is normal for small tabular datasets: **the data sets the ceiling, not the model.**
XGBoost was chosen for the highest mean PR-AUC; logistic regression is an equally good and
simpler alternative (`python -m churn.train --model logreg`).

**Final evaluation on test** (looked at exactly once):

| | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| threshold 0.5 | 0.843 | 0.665 | 0.51 | 0.79 | 0.62 |
| **threshold 0.61** | 0.843 | 0.665 | **0.57** | **0.70** | **0.63** |

Test ROC-AUC (0.843) matches cross-validation (0.847), so the model did not overfit.

![Model report](reports/figures/model_report.png)

## 7. Threshold: from probability to decision

The model outputs a probability, but the business needs a decision: "offer a discount or not".
A threshold of 0.5 is arbitrary. The right threshold depends on the **cost of errors**:

| Parameter (`config.py`) | Value |
|---|---|
| Value of a retained customer | $500 |
| Cost of the offer | $50 |
| Share of customers the offer actually retains | 30% |

* **True Positive** (leaving, we made an offer): `0.3 × 500 − 50 = +$100`
* **False Positive** (would have stayed, got a discount): `−$50`
* **FN / TN:** $0

The profit-maximizing threshold is chosen on **out-of-fold train predictions** (not on test!):
**0.61**.

| Strategy (test, 1,409 customers) | Profit |
|---|---|
| Offer to everyone | **−$14,350** |
| Offer to nobody | $0 |
| Model, threshold 0.5 | +$15,650 |
| **Model, threshold 0.61** | **+$16,300** |

![Profit curve](reports/figures/profit_curve.png)

Sending the discount to everyone loses money, while the model turns the same campaign into a
profitable one. Change the assumptions in `config.py` and the threshold is recomputed
automatically.

## 8. Explaining the model (SHAP)

**SHAP** (SHapley Additive exPlanations) splits every prediction into feature contributions:
`logit(p) = base value + Σ contributions`. A positive contribution increases risk, a negative
one decreases it. The contributions add up exactly to the prediction, so this is not an
approximation.

**Globally** (the plot at the top of this README): the most important factors are a
month-to-month contract, short tenure, no protection services, fiber optic, and paying by
check. The model learned the same things we saw in the EDA, which is an important sanity check.

**Locally**, for a single customer:

![SHAP waterfall](reports/figures/shap_waterfall_example.png)

This is exactly what the API returns in `top_factors`. A retention manager sees not just
"91% risk" but **why**, and can make a relevant offer: a one-year contract instead of
month-to-month, free TechSupport, switching to autopay.

## 9. REST API

```bash
make api   # or: uvicorn api.main:app --reload
```

Interactive Swagger docs: http://localhost:8000/docs

| Method | Path | Description |
|---|---|---|
| GET | `/health` | service status and whether the model is loaded |
| GET | `/model` | model name, threshold, test metrics |
| POST | `/predict?explain=true` | prediction for one customer + top-5 SHAP factors |
| POST | `/predict/batch` | up to 1,000 customers per request |

Example:

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d @scripts/example_customer.json
```

```json
{
  "churn_probability": 0.9123,
  "will_churn": true,
  "risk_level": "high",
  "threshold": 0.61,
  "top_factors": [
    {"feature": "tenure", "value": 2.0, "impact": 0.736, "direction": "increases risk"},
    {"feature": "is_month_to_month", "value": 1.0, "impact": 0.482, "direction": "increases risk"},
    {"feature": "no_protection", "value": 1.0, "impact": 0.264, "direction": "increases risk"}
  ]
}
```

Input is validated by a Pydantic schema: an unknown contract type or a negative charge returns
`422` with an explanation instead of a silently broken prediction.

## 10. Docker and CI

The **Dockerfile** has two stages:
1. `trainer` installs dependencies, downloads the data (with a SHA-256 check) and trains the model;
2. the runtime stage is a clean image with only the API code and the trained model, running as a non-root user.

The image is self-contained and reproducible: the same code and the same data give the same model.

**GitHub Actions** on every PR: `ruff` (lint and formatting) → `pytest` → full training →
Docker image build → smoke-test request to the container.

The tests do not need the internet: they generate a synthetic dataset in the IBM Telco
format and train a small model right inside a fixture.

## 11. Limitations and next steps

* **Calibration.** Class weights "boost" churn, so the model overestimates probabilities
  (see the calibration plot). That does not hurt ranking or threshold selection, but for honest
  percentages add `CalibratedClassifierCV` (isotonic) on top of the best model.
* **The business assumptions** ($500, $50, 30%) are made up. In a real project they come from
  financial data and an A/B test of the retention campaign.
* **A snapshot, not a time series.** The dataset is a single snapshot without dates. In
  practice you validate "through time" (train on past months, test on the next one) and
  monitor data drift.
* **Uplift modeling.** The model finds who *will leave*, but the business really needs those
  who *can be retained*. The next step is an uplift model trained on A/B test data.
* **Data beats algorithms.** All models hit the same ceiling (ROC-AUC ≈ 0.85). The biggest
  gains would come from new sources: support tickets, traffic usage, plan history.
* **Hyperparameter tuning** (Optuna) was deliberately skipped: the gap between models is already
  within the noise.

---

**Data license:** IBM Telco Customer Churn, published by IBM in the
[telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d) repository
under Apache 2.0. The dataset is not stored in this repository; a script downloads it.
