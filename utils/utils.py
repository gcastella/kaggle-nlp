import pandas as pd
import yaml
import munch


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


def read_config(file_path="../settings/config.yaml"):
    """
    Read yaml config file. Use x.etl.impute_missings to refer to elements.
    """
    with open(file_path) as file:
        context = yaml.load(file, Loader=yaml.FullLoader)
    return munch.DefaultMunch().fromDict(context)
