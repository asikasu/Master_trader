import joblib
from pathlib import Path
from xgboost import XGBClassifier


class AIModel:

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.model = XGBClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            gamma=0.5,
            scale_pos_weight=1,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_probability(self, X):
        return self.model.predict_proba(X)[0][1]

    def save(self):

        model_dir = self.project_root / "models"
        model_dir.mkdir(exist_ok=True)

        joblib.dump(
            self.model,
            model_dir / "best_model.pkl"
        )

        print("💾 Model saved")

    def load(self):

        model_file = self.project_root / "models" / "best_model.pkl"

        if model_file.exists():

            self.model = joblib.load(
                model_file
            )

            print("📂 Model loaded")

        else:

            print(
                "⚠ No saved model found."
            )