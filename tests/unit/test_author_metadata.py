import buildml


def test_package_author_metadata() -> None:
    assert buildml.__author__ == "Leonard Onyiriuba"
    assert buildml.__email__ == "leonard.c.onyiriuba@gmail.com"
