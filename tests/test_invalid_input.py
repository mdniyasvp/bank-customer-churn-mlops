from .conftest import client


def test_invalid_age():

    payload = {
        "CreditScore": 668,
        "Geography": "France",
        "Gender": "Male",
        "Age": 15,
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

    assert response.status_code == 422