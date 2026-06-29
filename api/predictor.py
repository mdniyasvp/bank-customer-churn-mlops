import time
import joblib
import pandas as pd

from src.config.config import MODEL_DIR, MODEL_NAME

from api.utils.logger import logger

model_path = MODEL_DIR / f"{MODEL_NAME}.joblib"

logger.info(f"Loading model from {model_path}")

model = joblib.load(model_path)

logger.info("Model loaded successfully")


def predict(data: dict):

    logger.info("Prediction request received")

    start = time.perf_counter()

    try:

        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]

        probability = float(
            model.predict_proba(df)[0][1]
        )

        label = "Exited" if prediction else "Stayed"

        elapsed = time.perf_counter() - start

        logger.info(
            f"Prediction={label} "
            f"Probability={probability:.4f} "
            f"Time={elapsed:.4f}s"
        )

        return {

            "prediction": int(prediction),

            "label": label,

            "probability": round(probability, 4)

        }

    except Exception as e:

        logger.exception(e)

        raise