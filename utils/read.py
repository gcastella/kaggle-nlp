import pandas as pd
import yaml
import munch


def read_config(file_path="./settings/config.yaml"):
    """
    Read yaml config file. Use x.etl.impute_missings to refer to elements.
    """
    with open(file_path) as file:
        context = yaml.load(file, Loader=yaml.FullLoader)
    return munch.DefaultMunch().fromDict(context)


def read_raw_data(config) -> {pd.DataFrame}:
    """
    Read the 3 kaggle csv's (train, test, 4_submission).
    """
    train = pd.read_csv(f"./{config.etl.raw.folder}/{config.etl.raw.train_file}")
    test = pd.read_csv(f"./{config.etl.raw.folder}/{config.etl.raw.test_file}")
    return {"train": train, "test": test}


def read_processed(config) -> {pd.DataFrame}:
    """
    Read 3 data files from 1_processed folder.
    """
    processed_folder = config.etl.processed.folder
    train = pd.read_parquet(f"./{processed_folder}/train.parquet")
    test = pd.read_parquet(f"./{processed_folder}/test.parquet")
    return {"train": train, "test": test}


def read_split(config, run):
    proc_dict = read_processed(config)
    proc_dict["name"] = "1_processed"
    return [proc_dict]


def read_prediction(config, run: str):
    pass


def load_model(config, run: str):
    pass


def read_submission(config, run: str):
    pass
