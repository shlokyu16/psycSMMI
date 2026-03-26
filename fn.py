"""
fn.py — Preprocessing and inference utilities for PsycAI.

This module provides the data transformation pipeline used both during model
training (final.ipynb) and at inference time (main.py). All functions operate
in-place or return modified DataFrames for chain-ability.
"""

import pandas as pd
import joblib


# ---------------------------------------------------------------------------
# Core drop helpers
# ---------------------------------------------------------------------------

def drop(df: pd.DataFrame, col) -> None:
    """Drop a single column or a column slice from df in-place."""
    df.drop(col, axis=1, inplace=True)


# ---------------------------------------------------------------------------
# Custom Grading System (Updated Scoring)
# ---------------------------------------------------------------------------

def ugsys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the custom weighted grading system to all Likert-scale question
    columns (everything after the Time Spent column at index 0).

    Standard questions (all except Self Esteem Q2):
        Original Likert  →  Weighted Score
        1 (Very Positive)   →  -1
        2 (Slightly Positive) →  0
        3 (Neutral)         →   0
        4 (Slightly Negative) →  3
        5 (Very Negative)   →   5  (unchanged)

    Self Esteem Q2 is inverted because the original response scale runs
    from Very Negative (1) → Very Positive (5), opposite to all other
    questions. Inversion ensures higher scores always indicate greater risk:
        1 (Very Negative)   →   5
        2 (Slightly Negative) →  3
        3 (Neutral)         →   0
        4 (Slightly Positive) →  -2
        5 (Very Positive)   →  -4
    """
    titles = list(df.columns)
    questions = titles[1:]  # Skip Time Spent at index 0

    for q in questions:
        df.loc[df[q] == 3, q] = 0
        if q == 'Self Esteem Q2':
            df.loc[df[q] == 1, q] = 5
            df.loc[df[q] == 2, q] = 3
            df.loc[df[q] == 4, q] = -2
            df.loc[df[q] == 5, q] = -4
        else:
            df.loc[df[q] == 1, q] = -1
            df.loc[df[q] == 2, q] = 0
            df.loc[df[q] == 4, q] = 3
            # 5 stays as 5 (already correct)

    return df


# ---------------------------------------------------------------------------
# Score Aggregation
# ---------------------------------------------------------------------------

def merge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate individual question responses into four domain scores and a
    total. Then drop the raw question columns, leaving only:
        Time Spent | ADHD Score | Anxiety Score | Self Esteem Score |
        Depression Score | Total Score
    """
    df['ADHD Score']       = df[['ADHD Q1', 'ADHD Q2', 'ADHD Q3', 'ADHD Q4']].sum(axis=1)
    df['Anxiety Score']    = df[['Anxiety Q1', 'Anxiety Q2']].sum(axis=1)
    df['Self Esteem Score']= df[['Self Esteem Q1', 'Self Esteem Q2', 'Self Esteem Q3']].sum(axis=1)
    df['Depression Score'] = df[['Depression Q1', 'Depression Q2', 'Depression Q3']].sum(axis=1)
    df['Total Score']      = df[['ADHD Score', 'Anxiety Score',
                                  'Self Esteem Score', 'Depression Score']].sum(axis=1)

    # Drop the 12 individual question columns (indices 1–12)
    drop(df, df.iloc[:, 1:13])
    return df


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

def refine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full inference-time preprocessing pipeline.
    Expects a single-row DataFrame with columns:
        Time Spent (int 0–5), ADHD Q1–Q4, Anxiety Q1–Q2,
        Self Esteem Q1–Q3, Depression Q1–Q3
    Returns a DataFrame ready for model.predict().
    """
    df = ugsys(df)
    df = merge(df)
    drop(df, 'Total Score')
    return df


def predict(df: pd.DataFrame) -> int:
    """Load the saved model and return a binary prediction (0 or 1)."""
    model = joblib.load("model.pkl")
    prediction = model.predict(df)
    return int(prediction[0])
