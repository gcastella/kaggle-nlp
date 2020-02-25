from .BaseSplitTask import BaseSplitTask

class CrossValidationSplitTask(BaseSplitTask):
    """
    This class provides the base interface to split the 1_processed dataset into n splits following the
    cross-validation scheme
    """
    def __init__(self, config, run, training_set_proportion, n_cross_validation):
        super().__init__(config=config, run=run)
        self.n_cross_validation = n_cross_validation

    def __next__(self):
        """
        Returns the next splitted training / validation set.
        This function can be inherited and changes to implement different split behaviours
        """
        temp_split = self.regular_split()
        if self.iteration_count >= self.n_cross_validation:
            raise StopIteration
        self.write_split(temp_split)
        temp_split["name"] = f"cross_validation_split_{self.iteration_count}"
        self.iteration_count += 1
        return temp_split


