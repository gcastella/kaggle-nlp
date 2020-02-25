class BaseSplitTask:
    """
    This class provides the base interface to split the 1_processed dataset into the training and verification sets
    """
    def __init__(self, config):
        self.config = config
        pass

    def run(self):
        pass

