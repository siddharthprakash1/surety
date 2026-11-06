from surety.types import ChoiceAnswer


def test_choice_answer_top():
    assert ChoiceAnswer({"a": 0.2, "b": 0.8}, [], 0, 0).top() == ("b", 0.8)
