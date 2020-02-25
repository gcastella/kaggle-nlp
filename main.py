from tasks.etl.ETLTask import ETLTask
from tasks.split.BaseSplitTask import BaseSplitTask
from tasks.prediction.LogisticRegressionTask import LogisticRegressionTask
from utils.read import read_config


def main():
    config = read_config()
    ETLTask(config=config).run()
    BaseSplitTask(config=config).run()
    # LogisticRegressionTask(run="try", config=config).run()


if __name__ == "__main__":
    main()
