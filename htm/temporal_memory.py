from ast import Mod
from typing import Tuple 

import numpy
from numpy import (
    ndarray, zeros, arange, zeros_like, full, apply_along_axis,
    any, argmax, argmin, nonzero, tile, sort, sum, all,
    concatenate, maximum, minimum, where
)
from numpy.random import shuffle, randn

from model.utils import ModelSettings
   
class TemporalMemory:
    def __init__(self) -> None:
        self.cell_active = zeros((ModelSettings.NUMBER_COLUMNS.value, ModelSettings.NUMBER_CELLS.value), dtype=bool)
        self.cell_predictive = zeros_like(self.cell_active)
        self.cell_segments = zeros((ModelSettings.NUMBER_COLUMNS.value, ModelSettings.NUMBER_CELLS.value), dtype=int)

        self.segment_capacity = 1
        self.segment_index = arange(ModelSettings.NUMBER_CELLS.value * self.segment_capacity, dtype=int).reshape(1, ModelSettings.NUMBER_CELLS.value, self.segment_capacity)
        self.segment_activation = zeros((ModelSettings.NUMBER_COLUMNS.value, ModelSettings.NUMBER_CELLS.value, self.segment_capacity), dtype=int)
        self.segment_potential = zeros_like(self.segment_activation)
        self.segment_active = zeros((ModelSettings.NUMBER_COLUMNS.value, ModelSettings.NUMBER_CELLS.value, self.segment_capacity), dtype=bool)
        self.segment_matching = zeros_like(self.segment_active)
        self.segment_synapses = zeros((ModelSettings.NUMBER_COLUMNS.value, ModelSettings.NUMBER_CELLS.value, self.segment_capacity), dtype=int)

        self.cell_synapse_capacity = 0
        self.cell_synapse_cell = full((ModelSettings.NUMBER_COLUMNS.value, ModelSettings.NUMBER_CELLS.value, self.cell_synapse_capacity), -1, dtype=int)

        self.segment_synapse_capacity = 1
        self.segment_synapse_cell = full((ModelSettings.NUMBER_COLUMNS.value, ModelSettings.NUMBER_CELLS.value, self.segment_capacity, self.segment_synapse_capacity), -1, dtype=int)
        self.segment_synapse_permanence = zeros((ModelSettings.NUMBER_COLUMNS.value, ModelSettings.NUMBER_CELLS.value, self.segment_capacity, self.segment_synapse_capacity), dtype=float)

        self.prev_winner_cell = zeros(0, dtype=int)
        self.prev_target_segment = zeros(0, dtype=int)

    def partial_fit(self, inputs:ndarray) -> None:
        cell_predictive = self.cell_predictive[inputs]
        column_bursting = ~any(cell_predictive, axis=1)

        segment_potential = self.segment_potential[inputs].reshape(len(inputs), -1)
        column_best_matching_segment = argmax(segment_potential, axis=1)
        column_least_used_cell = argmin(self.cell_segments[inputs], axis=1)
        column_grow_segment = segment_potential[(arange(len(inputs), dtype=int), column_best_matching_segment)] == 0
        segment_learning = self.segment_active[inputs] | ((self.segment_index == column_best_matching_segment[:, None, None]) & (column_bursting & ~column_grow_segment)[:, None, None])

        learning_segment = nonzero(segment_learning)
        learning_segment = inputs[learning_segment[0]] * (ModelSettings.NUMBER_CELLS.value * self.segment_capacity) + learning_segment[1] * self.segment_capacity + learning_segment[2]
        learning_segment_synapse_cell = self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[learning_segment]
        learning_segment_synapse_cell_valid = nonzero(learning_segment_synapse_cell >= 0)
        learning_segment_synapse_cell = learning_segment_synapse_cell[learning_segment_synapse_cell_valid]
        self.segment_synapse_permanence.reshape(-1, self.segment_synapse_capacity)[(learning_segment[learning_segment_synapse_cell_valid[0]], learning_segment_synapse_cell_valid[1])] += self.cell_active.reshape(-1)[learning_segment_synapse_cell] * (ModelSettings.TEMPORAL_PERMANENCE_INCREMENT.value + ModelSettings.TEMPORAL_PERMANENCE_DECREMENT.value) - ModelSettings.TEMPORAL_PERMANENCE_DECREMENT.value
        
        punished_segment = nonzero(self.segment_active.reshape(-1)[self.prev_target_segment])[0]
        punished_segment_synapse_cell = self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[punished_segment]
        punished_segment_synapse_cell_valid = nonzero(punished_segment_synapse_cell >= 0)
        punished_segment_synapse_cell = punished_segment_synapse_cell[punished_segment_synapse_cell_valid]
        self.segment_synapse_permanence.reshape(-1, self.segment_synapse_capacity)[(punished_segment[punished_segment_synapse_cell_valid[0]], punished_segment_synapse_cell_valid[1])] -= self.cell_active.reshape(-1)[punished_segment_synapse_cell] * ModelSettings.TEMPORAL_PERMANENCE_PUNISHMENT.value

        growing_segment_column = nonzero(column_grow_segment)[0]
        growing_segment_cell = column_least_used_cell[growing_segment_column]
        winner_cell = cell_predictive.copy()
        winner_cell[(growing_segment_column, growing_segment_cell)] = True
        winner_cell = nonzero(winner_cell)
        winner_cell = inputs[winner_cell[0]] * ModelSettings.NUMBER_CELLS.value + winner_cell[1]

        if len(self.prev_winner_cell) > 0:
            growing_segment_column = inputs[growing_segment_column]
            growing_segment = self.cell_segments[(growing_segment_column, growing_segment_cell)]

            max_cell_segments = max(growing_segment) + 1 if len(growing_segment) > 0 else 0
            if max_cell_segments > self.segment_capacity:
                segment_capacity = max_cell_segments
                self.segment_index = arange(ModelSettings.NUMBER_CELLS.value * segment_capacity, dtype=int).reshape(1, ModelSettings.NUMBER_CELLS.value, segment_capacity)
                self.segment_activation = zeros((ModelSettings.NUMBER_COLUMNS.value, ModelSettings.NUMBER_CELLS.value, segment_capacity), dtype=int)
                self.segment_potential = zeros_like(self.segment_activation)

                segment_synapses = zeros((ModelSettings.NUMBER_COLUMNS.value, ModelSettings.NUMBER_CELLS.value, segment_capacity), dtype=int)
                segment_synapse_cell = full((ModelSettings.NUMBER_COLUMNS.value, ModelSettings.NUMBER_CELLS.value, segment_capacity, self.segment_synapse_capacity), -1, dtype=int)
                segment_synapse_permanence = zeros((ModelSettings.NUMBER_COLUMNS.value, ModelSettings.NUMBER_CELLS.value, segment_capacity, self.segment_synapse_capacity), dtype=float)
                segment_synapses[:, :, :self.segment_capacity] = self.segment_synapses
                segment_synapse_cell[:, :, :self.segment_capacity, :] = self.segment_synapse_cell
                segment_synapse_permanence[:, :, :self.segment_capacity, :] = self.segment_synapse_permanence

                self.segment_capacity = segment_capacity
                self.segment_synapses = segment_synapses
                self.segment_synapse_cell = segment_synapse_cell
                self.segment_synapse_permanence = segment_synapse_permanence

            learning_segment = concatenate([learning_segment, growing_segment_column * (ModelSettings.NUMBER_CELLS.value * self.segment_capacity) + growing_segment_cell * self.segment_capacity + growing_segment])
            segment_candidate = sort(concatenate([tile(self.prev_winner_cell, (len(learning_segment), 1)), tile(self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[learning_segment], 2)], axis=1), axis=1)
            segment_winner_targeted = segment_candidate[:, :-1] == segment_candidate[:, 1:]
            segment_candidate[:, :-1][segment_winner_targeted] = -1
            segment_candidate[:, 1:][segment_winner_targeted] = -1
            segment_index = arange(segment_candidate.shape[0])[:, None]
            candidate_index = arange(segment_candidate.shape[1])
            shuffled_candidate_index = tile(candidate_index, (segment_candidate.shape[0], 1))
            apply_along_axis(shuffle, 1, shuffled_candidate_index)
            segment_candidate[:, candidate_index] = segment_candidate[(segment_index, shuffled_candidate_index)]
            
            segment_new_synapses = maximum(minimum(ModelSettings.SYNAPSE_SAMPLE_SIZE.value - self.segment_potential.reshape(-1)[learning_segment], sum(segment_candidate >= 0, axis=1)), 0)
            new_synapse_segment = nonzero(segment_new_synapses)[0]
            if len(new_synapse_segment) > 0:
                learning_segment = learning_segment[new_synapse_segment]
                segment_candidate = segment_candidate[new_synapse_segment]
                segment_new_synapses = segment_new_synapses[new_synapse_segment]
                shuffled_candidate_index = shuffled_candidate_index[new_synapse_segment]
                
                segment_synapses = self.segment_synapses.reshape(-1)[learning_segment]
                max_segment_synapses = max(segment_synapses + segment_new_synapses) if len(learning_segment) > 0 else 0
                if max_segment_synapses > self.segment_synapse_capacity:
                    segment_synapses = zeros(len(learning_segment), dtype=int)
                    valid_segment_synapse = nonzero(self.segment_synapse_permanence.reshape(-1, self.segment_synapse_capacity)[learning_segment] > 0)
                    segment_synapse_offset = zeros(len(learning_segment), dtype=int)
                    if len(valid_segment_synapse[0]) > 0:
                        valid_segment_synapse_offset = concatenate([zeros(1, dtype=int), 1 + nonzero(valid_segment_synapse[0][1:] != valid_segment_synapse[0][:-1])[0]])
                        valid_segment = valid_segment_synapse[0][valid_segment_synapse_offset]
                        segment_synapses[valid_segment] = concatenate([valid_segment_synapse_offset[1:] - valid_segment_synapse_offset[:-1], len(valid_segment_synapse[0]) - valid_segment_synapse_offset[-1].reshape(1)])
                        segment_synapse_offset[valid_segment] = valid_segment_synapse_offset
                    valid_segment_synapse_target = (valid_segment_synapse[0], arange(len(valid_segment_synapse[0]), dtype=int) - segment_synapse_offset[valid_segment_synapse[0]])
                    valid_segment_synapse = (learning_segment[valid_segment_synapse[0]], valid_segment_synapse[1])
                    self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[valid_segment_synapse_target] = self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[valid_segment_synapse]
                    self.segment_synapse_permanence.reshape(-1, self.segment_synapse_capacity)[valid_segment_synapse_target] = self.segment_synapse_permanence.reshape(-1, self.segment_synapse_capacity)[valid_segment_synapse]

                    max_segment_synapses = max(segment_synapses + segment_new_synapses) if len(learning_segment) > 0 else 0
                    if max_segment_synapses > self.segment_synapse_capacity:
                        segment_synapse_capacity = max_segment_synapses
                        segment_synapse_cell = full((ModelSettings.NUMBER_COLUMNS.value, ModelSettings.NUMBER_CELLS.value, self.segment_capacity, segment_synapse_capacity), -1, dtype=int)
                        segment_synapse_permanence = zeros((ModelSettings.NUMBER_COLUMNS.value, ModelSettings.NUMBER_CELLS.value, self.segment_capacity, segment_synapse_capacity), dtype=float)
                        segment_synapse_cell[:, :, :, :self.segment_synapse_capacity] = self.segment_synapse_cell
                        segment_synapse_permanence[:, :, :, :self.segment_synapse_capacity] = self.segment_synapse_permanence
                        self.segment_synapse_capacity = segment_synapse_capacity
                        self.segment_synapse_cell = segment_synapse_cell
                        self.segment_synapse_permanence = segment_synapse_permanence

                segment_target = nonzero(segment_candidate >= 0)
                segment_target_offset = concatenate([zeros(1, dtype=int), 1 + nonzero(segment_target[0][1:] != segment_target[0][:-1])[0]])
                segment_target_end = where(segment_new_synapses > 0, segment_target[1][segment_target_offset + segment_new_synapses - 1], 0)
                segment_new_synapse = arange(len(segment_target[0]), dtype=int) - segment_target_offset[segment_target[0]]
                segment_target_valid = nonzero(segment_target[1] <= segment_target_end[segment_target[0]])
                segment_target = (segment_target[0][segment_target_valid], segment_target[1][segment_target_valid])
                segment_new_synapse = segment_synapses[segment_target[0]] + segment_new_synapse[segment_target_valid]

                segment_target_segment = learning_segment[segment_target[0]]
                segment_target_candidate = segment_candidate[segment_target]
                self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[(segment_target_segment, segment_new_synapse)] = segment_target_candidate

                self.cell_segments[(growing_segment_column, growing_segment_cell)] += 1
                self.segment_synapses.reshape(-1)[learning_segment] = segment_new_synapses + segment_new_synapses
                
                candidate_target = (shuffled_candidate_index[segment_target], segment_target[0])
                candidate_synapse_cell = full((segment_candidate.shape[1], segment_candidate.shape[0]), -1, dtype=int)
                candidate_synapse_cell[candidate_target] = segment_target_candidate
                candidate_valid = nonzero(any(candidate_synapse_cell >= 0, axis=1))[0]

                candidate_synapse_cell_candidate = candidate_synapse_cell[candidate_valid]
                candidate_synapse_cell_candidate_valid = nonzero(candidate_synapse_cell_candidate >= 0)
                candidate_synapse_cell_candidate[(candidate_synapse_cell_candidate_valid[0], 0)] = candidate_synapse_cell_candidate[candidate_synapse_cell_candidate_valid]
                candidate_synapse_cell_candidate = candidate_synapse_cell_candidate[:, 0]

                candidate_synapse_cell[candidate_target] = segment_target_segment // self.segment_capacity
                candidate_synapse_cell = candidate_synapse_cell[candidate_valid]
                candidate_synapse_cell = concatenate([candidate_synapse_cell, self.cell_synapse_cell.reshape(ModelSettings.NUMBER_COLUMNS.value * ModelSettings.NUMBER_CELLS.value, -1)[candidate_synapse_cell_candidate]], axis=1)
                candidate_synapse_cell = sort(candidate_synapse_cell, axis=1)
                candidate_synapse_cell[:, 1:][candidate_synapse_cell[:, 1:] == candidate_synapse_cell[:, :-1]] = -1
                candidate_synapse_cell_valid = candidate_synapse_cell >= 0

                candidate_synapses = sum(candidate_synapse_cell_valid, axis=1)
                max_cell_synapses = max(candidate_synapses)
                if max_cell_synapses > self.cell_synapse_capacity:
                    cell_synapse_capacity = max_cell_synapses
                    cell_synapse_cell = full((ModelSettings.NUMBER_COLUMNS.value, ModelSettings.NUMBER_CELLS.value, cell_synapse_capacity), -1, dtype=int)
                    cell_synapse_cell[:, :, :self.cell_synapse_capacity] = self.cell_synapse_cell
                    self.cell_synapse_capacity = cell_synapse_capacity
                    self.cell_synapse_cell = cell_synapse_cell

                candidate_synapse_cell_valid = nonzero(candidate_synapse_cell_valid)
                candidate_synapse_cell_offset = concatenate([zeros(1, dtype=int), 1 + nonzero(candidate_synapse_cell_valid[0][1:] != candidate_synapse_cell_valid[0][:-1])[0]])
                candidate_synapse_cell_index = arange(len(candidate_synapse_cell_valid[0]), dtype=int) - candidate_synapse_cell_offset[candidate_synapse_cell_valid[0]]
                candidate_synapse_cell_candidate = candidate_synapse_cell_candidate[candidate_synapse_cell_valid[0]]
                self.cell_synapse_cell.reshape(-1, self.cell_synapse_capacity)[(candidate_synapse_cell_candidate, candidate_synapse_cell_index)] = candidate_synapse_cell[candidate_synapse_cell_valid]

        cell_active = cell_predictive | column_bursting[:, None]
        self.cell_active[:, :] = False
        self.cell_active[inputs] = cell_active

        active_cell = nonzero(cell_active)
        active_cell = (inputs[active_cell[0]], active_cell[1])

        cell_targeted = zeros(ModelSettings.NUMBER_COLUMNS.value * ModelSettings.NUMBER_CELLS.value, dtype=bool)
        active_cell_synapse_cell = self.cell_synapse_cell[active_cell]
        active_cell_synapse_cell = active_cell_synapse_cell[active_cell_synapse_cell >= 0]
        cell_targeted[active_cell_synapse_cell] = True
        target_cell = nonzero(cell_targeted)[0]
        target_segment = nonzero(arange(self.segment_capacity)[None, :] < self.cell_segments.reshape(-1)[target_cell][:, None])
        target_segment = target_cell[target_segment[0]] * self.segment_capacity + target_segment[1]
        
        segment_synapse_cell_active = self.cell_active.reshape(-1)[self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[target_segment]]
        segment_synapse_permanence = self.segment_synapse_permanence.reshape(-1, self.segment_synapse_capacity)[target_segment]
        segment_synapse_weight = segment_synapse_permanence > ModelSettings.TEMPORAL_PERMANENCE_THRESHOLD.value

        self.segment_activation[:, :, :] = 0
        self.segment_potential[:, :, :] = 0
        self.segment_activation.reshape(-1)[target_segment] = sum(segment_synapse_cell_active & segment_synapse_weight, axis=1)
        self.segment_potential.reshape(-1)[target_segment] = sum(segment_synapse_cell_active, axis=1)
        self.segment_active = self.segment_activation >= ModelSettings.SEGMENT_ACTIVE_THRESHOLD.value
        self.segment_matching = self.segment_potential >= ModelSettings.SEGMENT_MATCHING_THRESHOLD.value
        self.cell_predictive = any(self.segment_active, axis=2)

        self.prev_winner_cell = winner_cell
        self.prev_target_segment = target_segment

        print(f'correctly predicted columns: {self.score()}')

    def predict(self,inputs:ndarray) -> ndarray:
        output = self.cell_predictive[inputs]
        return any(output,axis=1)

    def score(self) -> int:
        return sum(any(
            self.cell_active & ~all(self.cell_active, axis=1)[:, None],
            axis=1
        ))