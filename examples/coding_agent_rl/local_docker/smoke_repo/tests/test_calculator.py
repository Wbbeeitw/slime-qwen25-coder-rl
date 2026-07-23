from calculator import add


def test_adds_two_positive_numbers():
    assert add(2, 3) == 5


def test_adds_a_negative_number():
    assert add(5, -2) == 3
