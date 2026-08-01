def get_recommendation(probability):

    if probability < 0.30:

        return {

            "risk": "🟢 LOW",

            "action": [

                "Customer appears loyal.",

                "No immediate action required.",

                "Continue normal engagement."

            ]

        }

    elif probability < 0.70:

        return {

            "risk": "🟡 MEDIUM",

            "action": [

                "Monitor customer activity.",

                "Offer personalized promotions.",

                "Encourage product usage."

            ]

        }

    else:

        return {

            "risk": "🔴 HIGH",

            "action": [

                "Contact customer immediately.",

                "Offer a retention incentive.",

                "Assign a relationship manager."

            ]

        }