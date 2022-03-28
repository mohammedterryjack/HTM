#from ffast import load
from numpy.random import randn 

from htm.hierarchical_temporal_memory import HierarchicalTemporalMemory
from htm.utils import ModelSettings

model = HierarchicalTemporalMemory()

#tokeniser = load()
#examples = [
#    "how",
#    "hello",
#    "are",
#    "you"
#]
#example_vectors = list(map(lambda text:tokeniser.encode(text).vector[:ModelSettings.INPUT_SIZE.value], examples))
example_vectors = randn(100, 1000) > 1.0


model.fit(example_vectors,ModelSettings.EPOCHS.value)