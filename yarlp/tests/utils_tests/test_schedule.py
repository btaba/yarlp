import pytest
from yarlp.utils.schedules import LinearSchedule


def test_linear_schedule():
    ls = LinearSchedule(10, 0, 1)
    assert ls.value(0) == 1
    assert ls.value(10) == 0
    assert ls.value(5) == 0.5
