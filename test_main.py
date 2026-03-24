import pytest


def test_addition(sample_data):
    assert 1 + 1 == 2

def test_error(sample_data):
    with pytest.raises(AssertionError):
        assert 1+1==3