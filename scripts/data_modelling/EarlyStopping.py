import numpy as np

class EarlyStopping:
    def __init__(self, tolerance_early_stop = 5, tolerance_training_rate = 3, min_delta=0):

        self.tolerance_early_stop = tolerance_early_stop
        self.tolerance_training_rate = tolerance_training_rate
        self.min_delta = min_delta
        self.counter_early_stop = 0
        self.counter_training_rate = 0
        self.early_stop = False
        self.min_validation_loss = np.inf
        self.min_train_loss = np.inf

    def early_stop_check(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter_early_stop = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter_early_stop += 1
            if self.counter_early_stop >= self.tolerance_early_stop:
                return True
        return False
    
    # def early_stop_check(self, train_loss, validation_loss):
    #     if (validation_loss - train_loss) > self.min_delta:
    #         self.counter_early_stop +=1
    #         if self.counter_early_stop >= self.tolerance_early_stop:  
    #             self.early_stop = True
        
    def decrease_training_rate(self, train_loss):
        if ((train_loss+self.min_delta) < self.min_train_loss):
            self.min_train_loss = train_loss
            self.counter_training_rate = 0  # reset the counter if validation loss decreased at least by min_delta
        elif ((train_loss+self.min_delta) > self.min_train_loss):
            self.counter_training_rate += 1 # increase the counter if validation loss is not decreased by the min_delta
            if self.counter_training_rate >= self.tolerance_training_rate:
                return True
        return False