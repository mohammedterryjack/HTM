from typing import List, Union, Generator

from numpy import ndarray, sum, any, all

from model.spatial_pooler import SpatialPooler
from model.temporal_memory import TemporalMemory

class HierarchicalTemporalMemory:
    def __init__(self):
        self.encoder = SpatialPooler()
        self.decoder = TemporalMemory()

    def predict(self, input:ndarray) -> ndarray:
        output = self.encoder.predict(input)
        return self.decoder.predict(output)

    def fit(self, examples:Union[List[ndarray],Generator[ndarray,None,None]], epochs:int) -> None:
        for _ in range(epochs):
            self.partial_fit(examples)

    def partial_fit(self, inputs:ndarray) -> None:
        for example in inputs:
            self.encoder.partial_fit(example) 
            output = self.encoder.predict(example)       
            self.decoder.partial_fit(output)
            contextual_output = self.decoder.predict(output)
