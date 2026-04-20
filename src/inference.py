import os
import numpy as np
import pandas as pd


def build_oof_dataframe(all_fold_results, id_col: str, target_col: str, n_classes: int):
    oof_df_list = [fr["val_result"] for fr in all_fold_results]
    oof_df = pd.concat(oof_df_list, axis=0).sort_values(id_col).reset_index(drop=True)
    oof_df["pred"] = oof_df[[f"prob_{c}" for c in range(n_classes)]].values.argmax(axis=1)
    return oof_df


def ensemble_test_predictions(all_fold_results, ensemble_method: str = "mean"):
    test_prob_list = [fr["test_probs"] for fr in all_fold_results]
    test_prob_stack = np.stack(test_prob_list, axis=0)

    if ensemble_method == "mean":
        test_probs_ens = test_prob_stack.mean(axis=0)
    else:
        test_probs_ens = test_prob_stack.mean(axis=0)

    test_preds = np.argmax(test_probs_ens, axis=1)
    return test_probs_ens, test_preds


def build_submission(test_df, id_col: str, target_col: str, test_preds, output_path: str):
    submission = test_df[[id_col]].copy()
    submission[target_col] = test_preds.astype(int)
    submission.to_csv(output_path, index=False)
    return submission
