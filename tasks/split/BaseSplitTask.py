from utils.read import read_processed
import random
import pandas as pd
import numpy as np
import os

class BaseSplitTask:
    """
    This class provides the base interface to split the 1_processed dataset into the training and verification sets.
    The class is supposed to be an iterator that computes "on the fly" the next set of training / validation set
    """
    def __init__(self, config, run, training_set_proportion=0.7):
        self.config = config
        self.run = run
        self.processed_data = read_processed(config)
        self.whole_training_data = self.processed_data["train"]
        self.iteration_count = 0
        self.training_set_proportion = training_set_proportion

    def run(self):
        """
        main function.
        """
        pass

    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns the next splitted training / validation set.
        This function can be inherited and changes to implement different split behaviours
        """
        temp_split = self.regular_split()
        if self.iteration_count >= 1:
            raise StopIteration
        self.write_split(temp_split)
        temp_split["name"] = f"regular_split_{self.iteration_count}"
        self.iteration_count += 1
        return temp_split

    def regular_split(self) -> {pd.DataFrame}:
        """
        Splits the processed dataset into training / test set
        """
        processed_data_size = len(self.whole_training_data.index)
        training_set_mask = np.random.rand(processed_data_size) < self.training_set_proportion
        training_set = self.whole_training_data[training_set_mask]
        validation_set = self.whole_training_data[~training_set_mask]
        return {"train": training_set, "test": validation_set}

    def write_split(self, split_df: {pd.DataFrame}):
        """
        Writes into a csv file the training and test sets of the current split
        :param split_df: splitted dictionary of dataframes (expected to contain the "test" and "train" keys
        """
        split_output_folder = f"{self.config.split.folder}/{self.run}"
        try:
            os.makedirs(split_output_folder)
        except FileExistsError:
            if self.config.verbose:
                print(f"Split folder for the current run {self.run} already exists")

        filename_training = f"split_training_set_{self.iteration_count}.csv"
        filename_test = f"split_test_set_{self.iteration_count}.csv"

        split_df["train"].to_csv(os.path.join(split_output_folder, filename_training))
        split_df["test"].to_csv(os.path.join(split_output_folder, filename_test))






