from .conftest import client


def test_predict():

    payload = {
        "CreditScore": 668,
        "Geography": "France",
        "Gender": "Male",
        "Age": 33,
        "Tenure": 3,
        "Balance": 0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 181449.97
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "label" in data
    assert "probability" in data

    assert data["prediction"] in [0, 1]