"""Train models: (1) predict final score, (2) classify at-risk students (final < 50)."""
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             mean_absolute_error, r2_score)
from sklearn.model_selection import train_test_split

from data_gen import generate

FEATURES = [
    "attendance_pct", "study_hours_week", "prev_gpa", "internal1", "internal2",
    "assignment_score", "sleep_hours", "part_time_job", "backlogs",
]
PASS_MARK = 50


def main():
    if not os.path.exists("data.csv"):
        generate().to_csv("data.csv", index=False)
    df = pd.read_csv("data.csv")
    df["at_risk"] = (df["final_score"] < PASS_MARK).astype(int)

    X = df[FEATURES]
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, df["final_score"], df["at_risk"], test_size=0.2, random_state=42,
        stratify=df["at_risk"],
    )

    reg = GradientBoostingRegressor(random_state=42).fit(X_tr, yr_tr)
    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42
    ).fit(X_tr, yc_tr)

    pred_r = reg.predict(X_te)
    pred_c = clf.predict(X_te)
    metrics = {
        "regression": {"MAE": round(mean_absolute_error(yr_te, pred_r), 2),
                       "R2": round(r2_score(yr_te, pred_r), 3)},
        "classification": {"accuracy": round(accuracy_score(yc_te, pred_c), 3),
                           "f1_at_risk": round(f1_score(yc_te, pred_c), 3)},
        "at_risk_rate": round(float(df["at_risk"].mean()), 3),
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(reg, "models/regressor.joblib")
    joblib.dump(clf, "models/classifier.joblib")
    pd.Series(reg.feature_importances_, index=FEATURES).sort_values(ascending=False) \
        .to_csv("models/feature_importance.csv", header=["importance"])
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(classification_report(yc_te, pred_c, target_names=["OK", "At risk"]))


if __name__ == "__main__":
    main()
