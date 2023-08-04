from pytest import CaptureFixture

from pygb.test_module import print_stuff


def test_print_stuff(capsys: CaptureFixture) -> None:
    print_stuff()
    assert capsys.readouterr().out == "stuff\n"
