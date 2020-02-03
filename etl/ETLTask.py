import pandas as pd
from typing import List
from utils.utils import read_config, is_number
from utils.utils_etl import (
    text_process,
    is_punctuation,
    is_mention,
    is_hashtag,
    is_link,
)


class ETLTask:
    def __init__(self):
        self.config = read_config()
        (self.train, self.test, self.submission) = self.read_raw_datasets()

    def run(self):
        """
        Main function
        """
        train_extended = self.add_features(self.train)
        test_extended = self.add_features(self.test)
        submission = self.submission

        self.write_processed_data(train_extended, "train.parquet")
        self.write_processed_data(test_extended, "test.parquet")
        self.write_processed_data(submission, "submission.parquet")

    def read_raw_datasets(self) -> List[pd.DataFrame]:
        """
        Read the 3 kaggle csv's (train, test, submission).
        """
        train = pd.read_csv("./data/raw/train.csv")
        test = pd.read_csv("./data/raw/test.csv")
        sub = pd.read_csv("./data/raw/sample_submission.csv")
        return [train, test, sub]

    def add_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the nlp kaggle dataset with new features.

        Args:
            dataset: Either train or test data.
        """

        df = dataset.copy()

        # Process text
        df["text_clean"] = df["text"].apply(text_process)

        # Variables over original text
        df["n_punctuation"] = df["text"].apply(
            lambda x: len([c for c in x if is_punctuation(c)])
        )
        df["n_capitals"] = df["text"].apply(
            lambda x: len([c for c in x if c.isupper()])
        )
        df["n_arroba"] = df["text"].apply(
            lambda x: len([i for i in x.split() if is_mention(i)])
        )
        df["n_hashtag"] = df["text"].apply(
            lambda x: len([i for i in x.split() if is_hashtag(i)])
        )
        df["n_links"] = df["text"].apply(
            lambda x: len([i for i in x.split() if is_link(i)])
        )

        # Variables over cleaned text
        df["n_char"] = df["text_clean"].apply(lambda x: len(x))
        df["n_words"] = df["text_clean"].apply(lambda x: len(x.split()))
        df["n_numbers"] = df["text_clean"].apply(
            lambda x: len([i for i in x.split() if is_number(i)])
        )

        # Ratio
        df["ratio_char_word"] = df["n_char"] / df["n_words"]
        df["ratio_capital_char"] = df["n_capitals"] / df["n_char"]
        df["ratio_hashtag_word"] = df["n_hashtag"] / df["n_words"]
        df["ratio_arroba_word"] = df["n_arroba"] / df["n_words"]
        df["ratio_number_word"] = df["n_numbers"] / df["n_words"]
        df["ratio_punctuation_word"] = df["n_punctuation"] / df["n_words"]

        # Location is missing
        df["location_isna"] = df["location"].isna().apply(lambda x: int(x))

        return df

    def write_processed_data(self, df, file):
        df.to_parquet(f"./data/{self.config.etl.processed_folder}/{file}")
