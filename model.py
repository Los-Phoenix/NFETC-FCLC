class Model(object):
    def add_placeholders(self):
        raise NotImplementedError('Each Model must re-implement this method.')

    def add_prediction_op(self):
        raise NotImplementedError('Each Model must re-implement this method.')

    def add_loss_op(self):
        raise NotImplementedError('Each Model must re-implement this method.')

    def add_training_op(self):
        raise NotImplementedError('Each Model must re-implement this method.')

    def build(self):
        self.add_placeholders()
        self.add_prediction_op()
        self.add_loss_op()
        self.add_training_op()
