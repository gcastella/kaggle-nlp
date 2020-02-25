import pandas as pd
import numpy as np
from datetime import datetime
import os


def is_number(s: str):
    """
    Checks if a string is a float.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def print_row(df: pd.Series):
    """
    Print every value in a pd.Series.
    """
    for value, index in zip(df, df.index):
        print(index, ": \n", value)


def evaluate_model(predictions, ground_truth, config, positive=1):
    """
    Return F1 metric used in the competition from prediction and ground truth.
    """
    predictions = pd.Series(predictions["prediction"])
    values = list(set([*predictions, *ground_truth]))[:2]
    cj = pd.merge(
        pd.DataFrame({"key": np.zeros(2), "pred": values}),
        pd.DataFrame({"key": np.zeros(2), "gt": values})
    )
    df = pd.DataFrame({"pred": predictions, "gt": ground_truth})
    df["count"] = 1
    print(df.head())
    dfg = df.groupby(["pred", "gt"], as_index=False).count()

    full_df = pd.merge(cj, dfg, on=["pred", "gt"], how="left").fillna(0)

    precision = int(full_df.loc[(full_df["pred"] == positive) & (full_df["gt"] == positive), "count"]) / (
            int(full_df.loc[(full_df["pred"] == positive) & (full_df["gt"] == positive), "count"]) +
            int(full_df.loc[(full_df["pred"] == positive) & (full_df["gt"] != positive), "count"]) +
            float(config.general.eps)
    )

    recall = int(full_df.loc[(full_df["pred"] == positive) & (full_df["gt"] == positive), "count"]) / (
            int(full_df.loc[(full_df["pred"] == positive) & (full_df["gt"] == positive), "count"]) +
            int(full_df.loc[(full_df["pred"] != positive) & (full_df["gt"] == positive), "count"]) +
            float(config.general.eps)
    )

    return 2 * precision * recall / (precision + recall + float(config.general.eps))


def create_run():
    now = datetime.now()
    run = f"{now.year:04}{now.month:02}{now.day:02}{now.hour:02}{now.minute:02}"
    print(f"Creating directories for run {run}.")
    folders = [folder for folder in os.listdir("data")
               if all(x not in folder for x in [".DS_Store", "0_raw", "1_processed"])]
    for folder in folders:
        try:
            os.mkdir(f"data/{folder}/{run}")
            print(f"Directory {folder}/{run} created.")
        except FileExistsError:
            print(f"Directory {folder}/{run} already exists.")
    print("Done!")
    return run
