import numpy as np
from collections import namedtuple
from collections import deque


Rollout = namedtuple('Rollout', 'rewards actions states')


class ReplayBuffer(object):
    def __init__(self, max_size=1000000):
        self.max_size = max_size
        self.Buffer = namedtuple(
            'Buffer', 'state, action, reward, next_state, terminal')
        self.mem = deque()

    def append(self, s, a, r, next_s, t):
        b = self.Buffer(s, a, r, next_s, t)
        if self.size >= self.max_size:
            self.mem.popleft()
        self.mem.append(b)

    @property
    def size(self):
        return len(self.mem)

    def get_random_minibatch(self, batch_size, flatten=True):
        assert batch_size <= self.size

        # O(N) runtime on get_random_minibatch, which I think is fine
        mem = list(self.mem)
        idx = np.random.choice(self.size, size=batch_size, replace=False)
        minibatch = [mem[i] for i in idx]

        if flatten:
            return self._flatten_batch(minibatch)
        return minibatch

    def _flatten_batch(self, minibatch):
        states = np.array([m.state for m in minibatch])
        actions = np.array([m.action for m in minibatch])
        rewards = np.array([m.reward for m in minibatch])
        next_states = np.array([m.next_state for m in minibatch])
        terminal = np.array([m.terminal for m in minibatch])

        return self.Buffer(states, actions, rewards, next_states, terminal)
