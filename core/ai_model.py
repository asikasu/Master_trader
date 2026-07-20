import joblib
from pathlib import Path
from xgboost import XGBClassifier


class AIModel:

    def __init__(self, xgb_config=None):
        self.project_root = Path(__file__).resolve().parents[1]
        if xgb_config:
            self.model = XGBClassifier(
                n_estimators=xgb_config.n_estimators,
                max_depth=xgb_config.max_depth,
                learning_rate=xgb_config.learning_rate,
                subsample=xgb_config.subsample,
                colsample_bytree=xgb_config.colsample_bytree,
                min_child_weight=xgb_config.min_child_weight,
                gamma=xgb_config.gamma,
                scale_pos_weight=xgb_config.scale_pos_weight,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1
            )
        else:
            self.model = XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                gamma=0.5,
                scale_pos_weight=1,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1
            )

    def train(self, X, y, eval_set=None, verbose=True):
        self.model.fit(X, y, eval_set=eval_set, verbose=verbose)

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

        print("[SAVE] Model saved")

    def load(self):

        model_file = self.project_root / "models" / "best_model.pkl"

        if model_file.exists():

            self.model = joblib.load(
                model_file
            )

            print("[LOAD] Model loaded")

        else:

            print("[WARNING] No saved model found.")