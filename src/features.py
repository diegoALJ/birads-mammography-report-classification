import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold


def add_basic_text_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df.copy()
    df["n_chars"] = df[text_col].apply(len)
    df["n_words"] = df[text_col].apply(lambda x: len(x.split()))
    return df


def validate_targets(df: pd.DataFrame, target_col: str, n_classes: int) -> None:
    assert sorted(df[target_col].unique()) == list(range(n_classes)), "Las clases no son 0..6"


def create_folds(df: pd.DataFrame, target_col: str, n_folds: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["fold"] = -1

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed
    )

    for fold, (_, valid_idx) in enumerate(skf.split(df, df[target_col])):
        df.loc[valid_idx, "fold"] = fold

    return df


def get_class_weights(df: pd.DataFrame, target_col: str, n_classes: int):
    counts = df[target_col].value_counts().sort_index()
    weights = len(df) / (n_classes * counts.values)
    return torch.tensor(weights, dtype=torch.float)
