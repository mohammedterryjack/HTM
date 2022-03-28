from enum import Enum 

class ModelSettings(Enum):
    INPUT_SIZE = 1000 #235724
    NUMBER_CELLS = 32
    NUMBER_COLUMNS = 2048
    NUMBER_ACTIVE_COLUMNS = 40 #NUMBER_COLUMNS * 0.02
    BOOSTING_INTENSITY = 0.3
    PERMANENCE_THRESHOLD = 0.0
    DUTY_CYCLE_INERTIA = 0.99
    PERMANENCE_INCREMENT = 0.1
    PERMANENCE_DECREMENT = 0.2
    PERMANENCE_INVALID = -1.0
    SEGMENT_ACTIVE_THRESHOLD = 10
    SEGMENT_MATCHING_THRESHOLD = 10
    TEMPORAL_PERMANENCE_THRESHOLD = 0.5
    TEMPORAL_PERMANENCE_INCREMENT = .3
    TEMPORAL_PERMANENCE_DECREMENT = .05
    TEMPORAL_PERMANENCE_PUNISHMENT = .01
    SYNAPSE_SAMPLE_SIZE = 20
    EPOCHS = 100
    PERMANENCE_INITIAL = 0.01