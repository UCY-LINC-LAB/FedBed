from clients.base import ModelHandler, DatasetHandler


class TensorflowModelHandler(ModelHandler):

    @classmethod
    def set_model_parameters(cls, model, parameters, *args, **kwargs):
        model.set_weights(parameters)

    @classmethod
    def get_model_parameters(cls, model, *args, **kwargs):
        return model.get_weights()

    @classmethod
    def train(cls, model, dataset, epochs, batch_size, *args, **kwargs):
        x_train, y_train = dataset
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        return len(y_train)

    @classmethod
    def test(cls, model, dataset, *args, **kwargs):
        (x_test, y_test) = dataset
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, accuracy, len(y_test)

class TensorflowDatasetHandler(DatasetHandler):

    dataset = None

    @classmethod
    def load_data(cls,
                  partition: int,
                  nodes: int,
                  training_set_size: int = 50_000,
                  test_size: int = 10_000,
                  random: bool = False,
                  distribution: str = 'flat',
                  distribution_parameters: dict = {}):
        pass

