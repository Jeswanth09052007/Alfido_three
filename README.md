# Titanic Survival Prediction

## Task Goal
Predict Titanic passenger survival using passenger attributes. The project emphasizes feature creation, missing-value handling, categorical encoding, model comparison, explainability, and example inference.

## Dataset
The dataset is stored at:

```text
data/titanic.csv
```

Target column:

```text
Survived
```

## Feature Engineering
The preprocessing pipeline creates the following features:

- `Title`: extracted from passenger names such as Mr, Mrs, Miss, Master, Rare
- `FamilySize`: `SibSp + Parch + 1`
- `IsAlone`: binary indicator for passengers travelling alone
- `CabinPresent`: binary indicator showing whether cabin information exists
- `Deck`: first cabin letter, or `Unknown`
- `FareLog`: `log1p(Fare)` transformation to reduce skewness

## Missing Value Handling
- `Age`: median imputation
- `Fare`: median imputation
- `Cabin`: converted into `CabinPresent` and `Deck`
- Categorical variables: most-frequent imputation and one-hot encoding
- Numeric variables: median imputation and standard scaling

## Models Compared
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

## Final Result
The best model is selected using F1 score and test accuracy.

```text
Best model: Logistic Regression
Accuracy: 1.0000
Precision: 1.0000
Recall: 1.0000
F1 Score: 1.0000
```

Note: The uploaded dataset shows very strong class separability, so all tested models reached perfect validation/test metrics on this provided file. This may not always generalize to a larger unseen Titanic dataset.

## Project Structure

```text
titanic_survival_project_submission_ready/
├── data/
│   └── titanic.csv
├── docs/
│   ├── Titanic_Survival_Submission_Document.docx
│   └── Titanic_Survival_Submission_Document.pdf
├── models/
│   ├── best_titanic_survival_model.joblib
│   └── best_titanic_survival_model.pkl
├── notebooks/
│   └── Titanic_Survival_Prediction.ipynb
├── outputs/
│   ├── best_model_summary.json
│   ├── classification_report.json
│   ├── feature_importance.csv
│   └── model_comparison_metrics.csv
├── plots/
│   ├── survival_distribution.png
│   ├── survival_by_sex.png
│   ├── survival_by_pclass.png
│   ├── survival_by_title.png
│   ├── confusion_matrix_best_model.png
│   └── feature_importance_best_model.png
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── inference.py
├── README.md
└── requirements.txt
```

## How to Run

### 1. Create and activate environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Train models and generate outputs

```bash
python src/train_model.py
```

### 4. Run inference

```bash
python src/inference.py
```

## Example Inference Code

```python
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
```

## Submission Instructions

As per the task rules:

1. Push this full project folder to GitHub.
2. Upload the saved model file or full ZIP to Google Drive if required.
3. Open `docs/Titanic_Survival_Submission_Document.docx` or `.pdf`.
4. Replace placeholder links with:
   - GitHub repository link
   - Notebook link
   - Model file link
   - Screenshots/plots link
   - Google Drive ZIP/model link, if required
5. Submit the links on the Task Submission page and clearly mention the task name/number.

## Package Versions

Important packages are listed in `requirements.txt`.
