import pytest

@pytest.fixture(scope="module")
def sample_data():
    print("Sample Data")
    # return {"name": "John", "age": 30, "city": "New York"}