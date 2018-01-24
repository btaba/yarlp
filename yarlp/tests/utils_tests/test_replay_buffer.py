import pytest
from yarlp.utils.replay_buffer import ReplayBuffer


def test_append():
    r = ReplayBuffer(max_size=2)
    r.add(1, 1, 1, 1, False)
    assert r.size == 1

    r.add(1, 1, 1, 1, False)
    r.add(1, 1, 1, 1, False)
    assert r.size == 2


def test_sample():
    r = ReplayBuffer(max_size=5)
    r.add(1, 1, 1, 1, False)
    r.add(1, 1, 1, 1, False)
    r.add(1, 1, 1, 1, False)

    b = r.sample(batch_size=2)
    assert b[0].shape[0] == 2
