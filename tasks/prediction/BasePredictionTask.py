import pandas as pd
from utils.read import read_split
from utils.utils import evaluate_model
import pickle as pk
import os
import platform

class BasePredictionTask:
    """
    This task implements prediction using the null model.
    It is intended to be used for inheriting when using more complex models.
    """

    def __init__(self, run, config, split=False):
        """
        Init 3_prediction task.
        """
        self.run_str = run
        self.config = config
        if not split:
            self.split = read_split(config=config, run=run)
        else:
            self.split = split
        self.train_metric = []
        self.test_metric = []
        self.model_split = []

    def run(self):
        """
        Main function.
        This function can be changed when inheriting from this class.
        """
        target_str = self.config.prediction.target_var
        for split in self.split:
            print(f"Predictions for split {split['name']}.")

            # Train
            model = self.fit_model(df=split["train"])
            self.model_split.append(model)

            # Predictions
            train_pred = self.model_predictions(df=split["train"], model=model)
            test_pred = self.model_predictions(df=split["test"], model=model)
            self.write_predictions(train_pred, f"train_pred_{split['name']}.csv")
            self.write_predictions(test_pred, f"test_pred_{split['name']}.csv")

            # Calculate metrics
            self.train_metric.append(
                evaluate_model(train_pred, split["train"][target_str], config=self.config)
            )
            self.test_metric.append(
                evaluate_model(test_pred, split["test"][target_str], config=self.config)
            )

        print("Global model.")
        # full_data = pd.concat(self.split[0]["train"], self.split[0]["test"])
        full_data = self.split.whole_training_data
        print(full_data)
        self.model = self.fit_model(df=full_data)
        self.save_model()

    def write_predictions(self, df: pd.DataFrame, name: str):
        """
        Write prediction file.
        Args:
            df: Data frame to be writen.
            results_folder: Folder in config for 3_prediction.
            name: string for the name of the folder with extension.
        """
        if platform.system() == "Windows":
            folder_name = f"{self.config.prediction.folder}/{self.run_str}"
        else:
            folder_name = f"./{self.config.prediction.folder}/{self.run_str}"

        try:
            os.makedirs(folder_name)
        except FileExistsError:
            if self.config.verbose:
                print(f"Prediction folder for the current run {self.run_str} already exists")
        file_name = os.path.join(folder_name, name)
        df.to_csv(file_name, index=False)

    def fit_model(self, df: pd.DataFrame):
        """
        Fitting for the null model.
        This function can be changed when inheriting from this class.
        """
        mode = df.groupby(self.config.prediction.target_var).count()["id"].idxmax()
        return mode

    def model_predictions(self, df: pd.DataFrame, model) -> pd.DataFrame:
        """
        Predictions for the null model.
        This function can be changed when inheriting from this class.
        """
        df["prediction"] = model
        return df[["id", "prediction"]]

    def save_model(self):
        folder = self.config.prediction.model.folder
        label = self.config.prediction.model.label
        try:
            os.makedirs(folder)
        except FileExistsError:
            if self.config.verbose:
                print(f"Model folder for the current run {self.run} already exists")

        pk.dump(self, open(f"{folder}/{self.run_str}_{label}.p", "wb"))

        return 1
