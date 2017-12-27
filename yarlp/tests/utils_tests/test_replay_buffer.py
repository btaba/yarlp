import pytest
from yarlp.utils.replay_buffer import ReplayBuffer


def test_append():
    r = ReplayBuffer(max_size=2)
    r.append(1, 1, 1, 1, False)
    assert r.size == 1

    r.append(1, 1, 1, 1, False)
    r.append(1, 1, 1, 1, False)
    assert r.size == 2


def test_get_random_minibatch():
    r = ReplayBuffer(max_size=5)
    r.append(1, 1, 1, 1, False)
    r.append(1, 1, 1, 1, False)
    r.append(1, 1, 1, 1, False)

    b = r.get_random_minibatch(batch_size=2, flatten=False)
    assert len(b) == 2

    b = r.get_random_minibatch(batch_size=2, flatten=True)
    assert b.state.shape[0] == 2
