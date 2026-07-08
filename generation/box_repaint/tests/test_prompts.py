from generation.box_repaint.prompts import prompt_for_category


def test_prompt_names_the_category():
    assert prompt_for_category("bird") == (
        "a bird, flat-color cartoon clipart illustration, bold black outlines")


def test_prompt_handles_multiword_category_name():
    assert prompt_for_category("dining table") == (
        "a dining table, flat-color cartoon clipart illustration, "
        "bold black outlines")
