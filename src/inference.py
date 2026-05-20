import pandas as pd
import joblib

model = joblib.load("models/best_titanic_survival_model.joblib")

sample_passenger = pd.DataFrame([{
    "PassengerId": 10001,
    "Pclass": 3,
    "Name": "Doe, Mr. John",
    "Sex": "male",
    "Age": 28,
    "SibSp": 1,
    "Parch": 0,
    "Ticket": "A/5 21171",
    "Fare": 7.25,
    "Cabin": None,
    "Embarked": "S"
}])

prediction = model.predict(sample_passenger)[0]
probability = model.predict_proba(sample_passenger)[0][1]

print("Prediction:", "Survived" if prediction == 1 else "Not Survived")
print("Survival probability:", round(float(probability), 4))
