from fastapi import APIRouter

from api.schemas import CustomerData, PredictionResponse
from api.predictor import predict

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)


@router.post(
    "",
    response_model=PredictionResponse
)
def predict_customer(data: CustomerData):
    """
    Predict customer churn.
    """

    result = predict(data.model_dump())

    return result