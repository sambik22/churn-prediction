from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """Один клієнт у тому ж форматі, що й у датасеті IBM Telco."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.7,
                "TotalCharges": 189.2,
            }
        }
    )

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Скільки місяців клієнт з компанією")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class Factor(BaseModel):
    feature: str
    value: float
    impact: float = Field(description="SHAP-внесок у логіт: >0 підвищує ризик відтоку")
    direction: str


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: Literal["low", "medium", "high"]
    threshold: float
    top_factors: list[Factor] | None = None


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model_name: str
    threshold: float
    trained_at: str
    test_metrics: dict[str, float]
