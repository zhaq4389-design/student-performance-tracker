"""Generate a realistic synthetic college student dataset.

Replace data.csv with your college's real data (same columns) to use real records.
"""
import numpy as np
import pandas as pd

DEPARTMENTS = ["CSE", "ECE", "Mechanical", "Civil", "IT", "AI&DS"]


def generate(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    attendance = np.clip(rng.normal(78, 14, n), 30, 100)
    study_hours = np.clip(rng.normal(12, 5, n), 0, 40)
    prev_gpa = np.clip(rng.normal(7.0, 1.2, n), 3, 10)
    sleep_hours = np.clip(rng.normal(6.8, 1.1, n), 3, 10)
    part_time_job = rng.binomial(1, 0.2, n)
    backlogs = rng.poisson(0.6, n)

    internal1 = np.clip(prev_gpa * 8 + study_hours * 0.4 + rng.normal(0, 8, n), 0, 100)
    internal2 = np.clip(prev_gpa * 8 + study_hours * 0.5 + rng.normal(0, 8, n), 0, 100)
    assignment = np.clip(55 + attendance * 0.25 + study_hours * 0.6 + rng.normal(0, 8, n), 0, 100)

    final = (
        0.25 * prev_gpa * 10
        + 0.15 * internal1
        + 0.20 * internal2
        + 0.10 * assignment
        + 0.10 * attendance
        + 0.4 * study_hours
        + 1.0 * (sleep_hours - 6)
        - 2.5 * backlogs
        - 3.0 * part_time_job
        + rng.normal(0, 4, n)
    )
    final = np.clip(final, 0, 100)

    return pd.DataFrame(
        {
            "student_id": [f"S{1000 + i}" for i in range(n)],
            "department": rng.choice(DEPARTMENTS, n),
            "semester": rng.integers(1, 9, n),
            "attendance_pct": attendance.round(1),
            "study_hours_week": study_hours.round(1),
            "prev_gpa": prev_gpa.round(2),
            "internal1": internal1.round(1),
            "internal2": internal2.round(1),
            "assignment_score": assignment.round(1),
            "sleep_hours": sleep_hours.round(1),
            "part_time_job": part_time_job,
            "backlogs": backlogs,
            "final_score": final.round(1),
        }
    )


if __name__ == "__main__":
    df = generate()
    df.to_csv("data.csv", index=False)
    print(f"Saved data.csv with {len(df)} students")
