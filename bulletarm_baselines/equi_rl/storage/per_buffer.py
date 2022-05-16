import sys
import numpy as np
import random

from .buffer import QLearningBuffer, QLearningBufferExpert
from .segment_tree import SumSegmentTree, MinSegmentTree

NORMAL = 0
EXPERT = 1

class PrioritizedQLearningBuffer:
    def __init__(self, size, alpha, base_buffer=NORMAL):
        if base_buffer == EXPERT:
            self.buffer = QLearningBufferExpert(size)
        elif base_buffer == NORMAL:
            self.buffer = QLearningBuffer(size)
        else:
            raise NotImplementedError

        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, key):
        return self.buffer[key]

    def __setitem__(self, key, value):
        self.buffer[key] = value

    def add(self, *args, **kwargs):
        '''
        See ReplayBuffer.store_effect
        '''
        idx = self.buffer._next_idx
        self.buffer.add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self.buffer) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        '''
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        Args:
          - batch_size: How many transitions to sample.
          - beta: To what degree to use importance weights
                  (0 - no corrections, 1 - full correction)

        Returns (obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, weights, idxes)
          - obs_batch: batch of observations
          - act_batch: batch of actions executed given obs_batch
          - rew_batch: rewards received as results of executing act_batch
          - next_obs_batch: next set of observations seen after executing act_batch
          - done_mask: done_mask[i] = 1 if executing act_batch[i] resulted in
                       the end of an episode and 0 otherwise.
          - weights: Array of shape (batch_size,) and dtype np.float32
                     denoting importance weight of each sampled transition
          - idxes: Array of shape (batch_size,) and dtype np.int32
                   idexes in buffer of sampled experiences
        '''
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.buffer)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.buffer)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        batch = [self.buffer._storage[idx] for idx in idxes]
        return batch, weights, idxes

    def update_priorities(self, idxes, priorities):
        '''
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Args:
          - idxes: List of idxes of sampled transitions
          - priorities: List of updated priorities corresponding to
                        transitions at the sampled idxes denoted by
                        variable `idxes`.
        '''
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):

            if priority <= 0:
                print("Invalid priority:", priority)
                print("All priorities:", priorities)

            assert priority > 0
            assert 0 <= idx < len(self.buffer)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def getSaveState(self):
        state = self.buffer.getSaveState()
        state.update(
            {
                'alpha': self._alpha,
                'it_sum': self._it_sum,
                'it_min': self._it_min,
                'max_priority': self._max_priority
            }
        )
        return state

    def loadFromState(self, save_state):
        self.buffer.loadFromState(save_state)
        self._alpha = save_state['alpha']
        self._it_sum = save_state['it_sum']
        self._it_min = save_state['it_min']
        self._max_priority = save_state['max_priority']


# class PrioritizedQLearningBuffer(QLearningBufferExpert):
#     def __init__(self, size, alpha):
#         '''
#         Create Prioritized Replay buffer.
#
#         Args:
#           - size: Max number of transitions to store in the buffer.
#           - alpha: How much prioritization is used
#                    (0 - no prioritization, 1 - full prioritization)
#
#         See Also
#         --------
#         ReplayBuffer.__init__
#         '''
#         super(PrioritizedQLearningBuffer, self).__init__(size)
#         assert alpha > 0
#         self._alpha = alpha
#
#         it_capacity = 1
#         while it_capacity < size:
#             it_capacity *= 2
#
#         self._it_sum = SumSegmentTree(it_capacity)
#         self._it_min = MinSegmentTree(it_capacity)
#         self._max_priority = 1.0
#
#     def add(self, *args, **kwargs):
#         '''
#         See ReplayBuffer.store_effect
#         '''
#         idx = self._next_idx
#         super(PrioritizedQLearningBuffer, self).add(*args, **kwargs)
#         self._it_sum[idx] = self._max_priority ** self._alpha
#         self._it_min[idx] = self._max_priority ** self._alpha
#
#     def _sample_proportional(self, batch_size):
#         res = []
#         for _ in range(batch_size):
#             mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
#             idx = self._it_sum.find_prefixsum_idx(mass)
#             res.append(idx)
#         return res
#
#     def sample(self, batch_size, beta):
#         '''
#         Sample a batch of experiences.
#
#         compared to ReplayBuffer.sample
#         it also returns importance weights and idxes
#         of sampled experiences.
#
#         Args:
#           - batch_size: How many transitions to sample.
#           - beta: To what degree to use importance weights
#                   (0 - no corrections, 1 - full correction)
#
#         Returns (obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, weights, idxes)
#           - obs_batch: batch of observations
#           - act_batch: batch of actions executed given obs_batch
#           - rew_batch: rewards received as results of executing act_batch
#           - next_obs_batch: next set of observations seen after executing act_batch
#           - done_mask: done_mask[i] = 1 if executing act_batch[i] resulted in
#                        the end of an episode and 0 otherwise.
#           - weights: Array of shape (batch_size,) and dtype np.float32
#                      denoting importance weight of each sampled transition
#           - idxes: Array of shape (batch_size,) and dtype np.int32
#                    idexes in buffer of sampled experiences
#         '''
#         assert beta > 0
#
#         idxes = self._sample_proportional(batch_size)
#
#         weights = []
#         p_min = self._it_min.min() / self._it_sum.sum()
#         max_weight = (p_min * len(self._storage)) ** (-beta)
#
#         for idx in idxes:
#             p_sample = self._it_sum[idx] / self._it_sum.sum()
#             weight = (p_sample * len(self._storage)) ** (-beta)
#             weights.append(weight / max_weight)
#         weights = np.array(weights)
#         batch = [self._storage[idx] for idx in idxes]
#         return batch, weights, idxes
#
#     def update_priorities(self, idxes, priorities):
#         '''
#         Update priorities of sampled transitions.
#
#         sets priority of transition at index idxes[i] in buffer
#         to priorities[i].
#
#         Args:
#           - idxes: List of idxes of sampled transitions
#           - priorities: List of updated priorities corresponding to
#                         transitions at the sampled idxes denoted by
#                         variable `idxes`.
#         '''
#         assert len(idxes) == len(priorities)
#         for idx, priority in zip(idxes, priorities):
#
#             if priority <= 0:
#                 print("Invalid priority:", priority)
#                 print("All priorities:", priorities)
#
#             assert priority > 0
#             assert 0 <= idx < len(self._storage)
#             self._it_sum[idx] = priority ** self._alpha
#             self._it_min[idx] = priority ** self._alpha
#
#             self._max_priority = max(self._max_priority, priority)
#
#     def getSaveState(self):
#         state = super().getSaveState()
#         state.update(
#             {
#                 'alpha': self._alpha,
#                 'it_sum': self._it_sum,
#                 'it_min': self._it_min,
#                 'max_priority': self._max_priority
#             }
#         )
#         return state
#
#     def loadFromState(self, save_state):
#         super().loadFromState(save_state)
#         self._alpha = save_state['alpha']
#         self._it_sum = save_state['it_sum']
#         self._it_min = save_state['it_min']
#         self._max_priority = save_state['max_priority']
