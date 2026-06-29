from fastapi import FastAPI

from api.routers.predict import router as predict_router
from api.routers.health import router as health_router

app = FastAPI(

    title="Bank Customer Churn Prediction API",

    description="""
Production-ready ML API

Features

• FastAPI

• XGBoost

• MLflow

• DVC

• Docker

• CI/CD Ready

""",

    version="1.0.0",

    contact={

        "name":"Muhammed Niyas",

        "email":"mdniyas.vp@gmail.com"

    }

)

app.include_router(health_router)
app.include_router(predict_router)


@app.get("/")
def home():
    return {
        "message": "Bank Customer Churn Prediction API"
    }