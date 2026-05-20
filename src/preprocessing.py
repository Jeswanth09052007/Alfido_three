import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create Titanic-specific features before sklearn preprocessing.

    Features created:
    - Title: extracted from passenger name, rare titles grouped together
    - FamilySize: SibSp + Parch + 1
    - IsAlone: 1 if passenger travelled alone, else 0
    - CabinPresent: 1 if Cabin is available, else 0
    - Deck: first letter of Cabin, or Unknown
    - FareLog: log1p(Fare) to reduce skewness
    """

    rare_titles = {"Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"}
    title_map = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in ["Name", "Cabin", "SibSp", "Parch", "Fare"]:
            if col not in X.columns:
                X[col] = np.nan

        X["Title"] = X["Name"].fillna("").apply(self._extract_title)
        X["Title"] = X["Title"].replace(self.title_map)
        X["Title"] = X["Title"].apply(lambda t: "Rare" if t in self.rare_titles or t == "" else t)

        X["FamilySize"] = X["SibSp"].fillna(0) + X["Parch"].fillna(0) + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        X["CabinPresent"] = X["Cabin"].notna().astype(int)
        X["Deck"] = X["Cabin"].fillna("Unknown").astype(str).str[0]
        X.loc[X["Deck"].isin(["n", "N"]), "Deck"] = "Unknown"
        X["FareLog"] = np.log1p(X["Fare"].fillna(X["Fare"].median()))
        return X

    @staticmethod
    def _extract_title(name):
        match = re.search(r",\s*([^\.]+)\.", str(name))
        return match.group(1).strip() if match else "Unknown"
