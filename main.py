from tasks.etl.ETLTask import ETLTask
from tasks.split.BaseSplitTask import BaseSplitTask
from tasks.split.CrossValidationSplitTask import CrossValidationSplitTask
from tasks.prediction.LogisticRegressionTask import LogisticRegressionTask, BasePredictionTask
from utils.utils import create_run
from utils.read import read_config


def main():
    config = read_config()
    run = create_run()
    ETLTask(config=config).run()
    split_data_generator = CrossValidationSplitTask(config=config,
                                                      run=run,
                                                      training_set_proportion=0.7,
                                                      n_cross_validation=10
                                                    )
    BasePredictionTask(config=config,
                       run=run,
                       split=split_data_generator
                       ).run()

if __name__ == "__main__":
    main()
