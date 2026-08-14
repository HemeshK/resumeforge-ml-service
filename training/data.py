"""Load and prepare the resume<->JD match dataset.

Source: med2425/resume-job-fit-merged-v1 (Hugging Face). The dataset ships
with its own train/test split, but we pool both and re-split ourselves so we
have full control over stratification for the ablation experiment in
train.py.
"""
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET_NAME = "med2425/resume-job-fit-merged-v1"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw() -> pd.DataFrame:
    ds = load_dataset(DATASET_NAME)
    return pd.concat([ds[split].to_pandas() for split in ds.keys()], ignore_index=True)


def collapse_label(df: pd.DataFrame) -> pd.DataFrame:
    """Good Fit -> 1, {Potential Fit, No Fit} -> 0."""
    df = df.copy()
    df["label_bin"] = (df["label"] == "Good Fit").astype(int)
    return df


def load_and_prepare() -> pd.DataFrame:
    return collapse_label(load_raw())


def train_test_split_stratified(df: pd.DataFrame):
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label_bin"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def same_domain_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["resume_domain"] == df["jd_domain"]].reset_index(drop=True)
