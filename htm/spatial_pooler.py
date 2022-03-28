from numpy import ndarray, zeros, sum, exp
from numpy.random import randn 

from model.utils import ModelSettings

class SpatialPooler:
    def __init__(self) -> None:
        self.__initialise_weights()
        self.__initialise_duty_cycle()

    def predict(self,sparse_vector:ndarray) -> ndarray:
        active_columns_ranked = (self._boost() * self._project_inputs(sparse_vector)).argsort()
        return active_columns_ranked[-ModelSettings.NUMBER_ACTIVE_COLUMNS.value:]

    def partial_fit(self, sparse_vector:ndarray) -> ndarray:
        outputs = self.predict(sparse_vector)
        self.__update_duty_cycle(outputs)
        self.__update_weights(sparse_vector, outputs)

    def __initialise_weights(self) -> None:
        self.weights_permanence = randn(ModelSettings.NUMBER_COLUMNS.value, ModelSettings.INPUT_SIZE.value)

    def __initialise_duty_cycle(self) -> None:
        self.duty_cycle = zeros(ModelSettings.NUMBER_COLUMNS.value)

    def __update_weights(self, inputs:ndarray, active_indexes:ndarray) -> None:
        self.weights_permanence[active_indexes] += inputs * (
            ModelSettings.PERMANENCE_INCREMENT.value + ModelSettings.PERMANENCE_DECREMENT.value
        ) - ModelSettings.PERMANENCE_DECREMENT.value

    def __update_duty_cycle(self, active_indexes:ndarray) -> None:
        self.duty_cycle *= ModelSettings.DUTY_CYCLE_INERTIA.value
        self.duty_cycle[active_indexes] += 1. - ModelSettings.DUTY_CYCLE_INERTIA.value

    def _project_inputs(self, inputs:ndarray) -> ndarray:
        return sum(inputs & self.binarise_weights(self.weights_permanence), axis=1)
        
    def _boost(self) -> ndarray:
        _sparsity = ModelSettings.NUMBER_ACTIVE_COLUMNS.value / ModelSettings.NUMBER_COLUMNS.value
        return exp(ModelSettings.BOOSTING_INTENSITY.value * -self.duty_cycle / _sparsity)

    @staticmethod
    def binarise_weights(weights:ndarray) -> ndarray:
        return weights > ModelSettings.PERMANENCE_THRESHOLD.value