# 🎓 Student Performance Tracker (AI/ML)

![CI](https://github.com/<your-username>/student-performance-tracker/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A machine-learning project for colleges that predicts students' final scores and flags those at risk of failing, so faculty can step in early. Includes an interactive Streamlit dashboard.

## Features
- **Score prediction**: Gradient Boosting regressor predicts the final score (out of 100).
- **At-risk detection**: Random Forest classifier estimates the probability of scoring below 50.
- **Dashboard**: department overview, downloadable at-risk list, single-student prediction with suggestions, and batch CSV upload.
- **Explainability**: feature-importance chart shows what drives performance.
- **Tests + CI**: pytest suite runs automatically on every push via GitHub Actions.

## Project structure
```
├── app.py               # Streamlit dashboard
├── train.py             # Trains models, saves to models/, prints metrics
├── data_gen.py          # Synthetic dataset generator (data.csv)
├── data.csv             # Sample data (1000 students)
├── tests/               # pytest tests
├── .github/workflows/   # CI pipeline
├── requirements.txt
└── LICENSE
```

## Quick start
```bash
git clone https://github.com/<your-username>/student-performance-tracker.git
cd student-performance-tracker
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py          # trains models and prints metrics
streamlit run app.py     # opens the dashboard
```
Run tests with `pip install pytest && pytest`.

## Sample results (synthetic data, 80/20 split)
| Task | Metric | Value |
|---|---|---|
| Score prediction | MAE | ~3.4 marks |
| Score prediction | R² | ~0.78 |
| At-risk detection | Accuracy | ~0.87 |
| At-risk detection | F1 (at-risk class) | ~0.67 |

## Input features
attendance %, weekly study hours, previous GPA, internal 1 & 2 marks, assignment score, sleep hours, part-time job, backlogs.

## Using real data
Replace `data.csv` with your college's records using the same columns (plus `final_score`, `student_id`, `department`, `semester`), then re-run `python train.py`. Get permission and anonymise student IDs first. Never commit real student data to a public repository.

## Deploy
Push to GitHub, then deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud): choose the repo, set the main file to `app.py`. Models train automatically on first launch.

## Ideas to extend
- SHAP explanations for each prediction
- Track students across semesters (trend analysis)
- Email alerts to mentors for high-risk students
- Compare more models (XGBoost, Logistic Regression)

## Limitations
The bundled data is synthetic, so metrics reflect the data generator, not real students. Predictions should support, never replace, a mentor's judgement.

## License
MIT. See [LICENSE](LICENSE).
