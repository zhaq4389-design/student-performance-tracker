"""Student Performance Tracker - Streamlit dashboard.  Run: streamlit run app.py"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

from train import FEATURES, PASS_MARK, main as train_models

st.set_page_config(page_title="Student Performance Tracker", page_icon="🎓", layout="wide")

if not os.path.exists("models/regressor.joblib"):
    with st.spinner("First run: training models..."):
        train_models()


@st.cache_resource
def load_models():
    return joblib.load("models/regressor.joblib"), joblib.load("models/classifier.joblib")


@st.cache_data
def load_data():
    return pd.read_csv("data.csv")


reg, clf = load_models()
df = load_data()
df["predicted_score"] = reg.predict(df[FEATURES]).round(1)
df["risk_probability"] = (clf.predict_proba(df[FEATURES])[:, 1] * 100).round(1)


def advice(row) -> list[str]:
    tips = []
    if row["attendance_pct"] < 75:
        tips.append("Attendance is below 75% - arrange a mentor meeting.")
    if row["study_hours_week"] < 8:
        tips.append("Low weekly study hours - suggest a study plan.")
    if row["internal2"] < row["internal1"] - 5:
        tips.append("Internal marks are dropping - offer remedial classes.")
    if row["backlogs"] >= 2:
        tips.append("Multiple backlogs - academic counselling recommended.")
    if row["sleep_hours"] < 6:
        tips.append("Poor sleep may be hurting performance.")
    return tips or ["No major warning signs."]


st.title("🎓 Student Performance Tracker")
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🚨 At-risk students", "🔮 Predict a student", "📁 Batch upload"]
)

with tab1:
    dept = st.multiselect("Department", sorted(df["department"].unique()),
                          default=sorted(df["department"].unique()))
    view = df[df["department"].isin(dept)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", len(view))
    c2.metric("Avg final score", f"{view['final_score'].mean():.1f}")
    c3.metric("Avg attendance", f"{view['attendance_pct'].mean():.1f}%")
    c4.metric("At risk (>50% prob.)", int((view["risk_probability"] > 50).sum()))

    left, right = st.columns(2)
    left.subheader("Average score by department")
    left.bar_chart(view.groupby("department")["final_score"].mean())
    right.subheader("Score distribution")
    right.bar_chart(pd.cut(view["final_score"], bins=range(0, 101, 10)).value_counts().sort_index()
                    .rename(index=str))

    st.subheader("What drives performance? (feature importance)")
    st.bar_chart(pd.read_csv("models/feature_importance.csv", index_col=0))

    if os.path.exists("models/metrics.json"):
        with st.expander("Model performance"):
            st.json(json.load(open("models/metrics.json")))

with tab2:
    threshold = st.slider("Risk threshold (%)", 10, 90, 50)
    risky = df[df["risk_probability"] >= threshold].sort_values("risk_probability", ascending=False)
    st.write(f"**{len(risky)} students** at or above {threshold}% risk")
    st.dataframe(
        risky[["student_id", "department", "semester", "attendance_pct", "internal1",
               "internal2", "predicted_score", "risk_probability"]],
        use_container_width=True, hide_index=True,
    )
    st.download_button("Download list (CSV)", risky.to_csv(index=False), "at_risk_students.csv")

with tab3:
    st.write("Enter a student's details to predict their final score and risk.")
    a, b, c = st.columns(3)
    row = {
        "attendance_pct": a.slider("Attendance %", 0, 100, 75),
        "study_hours_week": a.slider("Study hours / week", 0, 40, 10),
        "prev_gpa": a.slider("Previous GPA (10-pt)", 3.0, 10.0, 7.0, 0.1),
        "internal1": b.slider("Internal 1 (/100)", 0, 100, 60),
        "internal2": b.slider("Internal 2 (/100)", 0, 100, 60),
        "assignment_score": b.slider("Assignment score", 0, 100, 65),
        "sleep_hours": c.slider("Sleep hours", 3.0, 10.0, 7.0, 0.5),
        "part_time_job": int(c.checkbox("Has part-time job")),
        "backlogs": c.number_input("Backlogs", 0, 10, 0),
    }
    X = pd.DataFrame([row])[FEATURES]
    score = float(reg.predict(X)[0])
    risk = float(clf.predict_proba(X)[0, 1]) * 100
    m1, m2 = st.columns(2)
    m1.metric("Predicted final score", f"{score:.1f} / 100")
    m2.metric("Risk of scoring below %d" % PASS_MARK, f"{risk:.0f}%")
    st.progress(min(int(risk), 100))
    st.subheader("Suggestions")
    for t in advice(row):
        st.write("•", t)

with tab4:
    st.write("Upload a CSV with columns: " + ", ".join(FEATURES))
    up = st.file_uploader("CSV file", type="csv")
    if up:
        new = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in new.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            new["predicted_score"] = reg.predict(new[FEATURES]).round(1)
            new["risk_probability"] = (clf.predict_proba(new[FEATURES])[:, 1] * 100).round(1)
            st.dataframe(new, use_container_width=True)
            st.download_button("Download predictions", new.to_csv(index=False), "predictions.csv")
