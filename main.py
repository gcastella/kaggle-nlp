from tasks.etl.ETLTask import ETLTask
from tasks.split.BaseSplitTask import BaseSplitTask
from tasks.split.CrossValidationSplitTask import CrossValidationSplitTask
from tasks.prediction.LogisticRegressionTask import LogisticRegressionTask, BasePredictionTask
from utils.read import read_config


def main():
    config = read_config()
    run = "test"
    ETLTask(config=config).run()
    BasePredictionTask(config=config,
                       run=run,
                       split=CrossValidationSplitTask(config=config,
                                                      run=run,
                                                      training_set_proportion=0.7,
                                                      n_cross_validation=10
                                                      )
                       ).run()


if __name__ == "__main__":
    main()
