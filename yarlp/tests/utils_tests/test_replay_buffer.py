import unittest
from yarlp.utils.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):

    def test_append(self):
        r = ReplayBuffer(max_size=2)
        r.append(1, 1, 1, 1, False)
        self.assertEqual(r.size, 1)

        r.append(1, 1, 1, 1, False)
        r.append(1, 1, 1, 1, False)
        self.assertEqual(r.size, 2)

    def test_get_random_minibatch(self):
        r = ReplayBuffer(max_size=5)
        r.append(1, 1, 1, 1, False)
        r.append(1, 1, 1, 1, False)
        r.append(1, 1, 1, 1, False)

        b = r.get_random_minibatch(batch_size=2, flatten=False)
        self.assertEqual(len(b), 2)

        b = r.get_random_minibatch(batch_size=2, flatten=True)
        self.assertEqual(b.state.shape[0], 2)
