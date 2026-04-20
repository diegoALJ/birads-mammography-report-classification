import re
import pandas as pd


def normalize_for_leakage(text: str) -> str:
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_leakage(df: pd.DataFrame, text_col: str, id_col: str):
    df = df.copy()

    df["_report_leak_check"] = df[text_col].fillna("").apply(normalize_for_leakage)
    leakage_regex = r"\bcategoria\s*[:\-]?\s*[0-6]\b"

    leakage_mask = df["_report_leak_check"].str.contains(leakage_regex, regex=True, na=False)
    leakage_df = df.loc[leakage_mask].copy().reset_index(drop=True)
    leakage_ids = leakage_df[id_col].astype(str).tolist()

    return leakage_df, leakage_ids


def remove_leakage_records(df: pd.DataFrame, id_col: str, leakage_ids):
    df = df.copy()
    if leakage_ids:
        df = df[~df[id_col].astype(str).isin(set(leakage_ids))].reset_index(drop=True)
    df.drop(columns=["_report_leak_check"], errors="ignore", inplace=True)
    return df


def preprocess_text(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text_light(text: str) -> str:
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_text_preprocessing(train_df, test_df, text_col: str, use_text_preprocessing: bool, text_version: str):
    train_df = train_df.copy()
    test_df = test_df.copy()

    if use_text_preprocessing:
        if text_version == "clean":
            train_df[text_col] = train_df[text_col].fillna("").apply(preprocess_text_light)
            test_df[text_col] = test_df[text_col].fillna("").apply(preprocess_text_light)
        else:
            train_df[text_col] = train_df[text_col].fillna("").apply(preprocess_text)
            test_df[text_col] = test_df[text_col].fillna("").apply(preprocess_text)
    else:
        train_df[text_col] = train_df[text_col].fillna("").astype(str)
        test_df[text_col] = test_df[text_col].fillna("").astype(str)

    return train_df, test_df
