class EarlyStopping():
    def __init__(self, patience: int=10):
        self.patience = patience
        self.best_value = None
        self.best_epoch = None
        self.last_value = None
        self.counter = 0
        self.stop = False

    def step(self, value, epoch):
        if self.last_value is not None:
            if round(value, 4) > round(self.last_value, 4):
                self.counter += 1
            else:
                self.counter = 0
        
        self.last_value = value

        if self.best_value is None:
            self.best_value = value
        
        if value < self.best_value:
            self.best_value = value
            self.best_epoch = epoch

        if self.counter == self.patience:
            self.stop = True

        return self.stop