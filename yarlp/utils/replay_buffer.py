import numpy as np


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.mem = []
        self._next_idx = 0
        self.size = 0

    def add(self, s, a, r, next_s, done):
        b = (s, a, r, next_s, done)
        if self.size < self.max_size:
            self.mem.append(b)
        else:
            self.mem[self._next_idx] = b
        self._next_idx += 1
        self.size = min(self.size + 1, self.max_size)
        self._next_idx = self._next_idx % self.max_size

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        assert batch_size <= self.size

        idx = np.random.choice(self.size, size=batch_size, replace=False)
        minibatch = [self.mem[i] for i in idx]

        return self._flatten_batch(minibatch)

    def _flatten_batch(self, minibatch):
        states = np.array([np.array(m[0]) for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([np.array(m[3]) for m in minibatch])
        done = np.array([m[4] for m in minibatch])

        return states, actions, rewards, next_states, done

    def return_most_recent(self):
        return self._flatten_batch(self.mem[-1])


# class ReplayBuffer(object):
#     def __init__(self, size):
#         """Create Replay buffer.

#         Parameters
#         ----------
#         size: int
#             Max number of transitions to store in the buffer. When the buffer
#             overflows the old memories are dropped.
#         """
#         self._storage = []
#         self._maxsize = size
#         self._next_idx = 0

#     def __len__(self):
#         return len(self._storage)

#     def add(self, obs_t, action, reward, obs_tp1, done):
#         data = (obs_t, action, reward, obs_tp1, done)

#         if self._next_idx >= len(self._storage):
#             self._storage.append(data)
#         else:
#             self._storage[self._next_idx] = data
#         self._next_idx = (self._next_idx + 1) % self._maxsize

#     def _encode_sample(self, idxes):
#         obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
#         for i in idxes:
#             data = self._storage[i]
#             obs_t, action, reward, obs_tp1, done = data
#             obses_t.append(np.array(obs_t, copy=False))
#             actions.append(np.array(action, copy=False))
#             rewards.append(reward)
#             obses_tp1.append(np.array(obs_tp1, copy=False))
#             dones.append(done)
#         return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

#     def sample(self, batch_size):
#         """Sample a batch of experiences.

#         Parameters
#         ----------
#         batch_size: int
#             How many transitions to sample.

#         Returns
#         -------
#         obs_batch: np.array
#             batch of observations
#         act_batch: np.array
#             batch of actions executed given obs_batch
#         rew_batch: np.array
#             rewards received as results of executing act_batch
#         next_obs_batch: np.array
#             next set of observations seen after executing act_batch
#         done_mask: np.array
#             done_mask[i] = 1 if executing act_batch[i] resulted in
#             the end of an episode and 0 otherwise.
#         """
#         idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
#         return self._encode_sample(idxes)
