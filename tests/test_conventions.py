"""Geometry naming conventions."""

from evacusim.conventions import is_platform_zone, is_train_exit, platform_zone, train_exit


def test_train_exit_round_trip():
    assert train_exit(3) == "train_platform_3"
    assert is_train_exit("train_platform_3")
    assert not is_train_exit("escalator_a_down")
    assert not is_train_exit(None)


def test_platform_zone_of_a_target():
    assert platform_zone("train_platform_3") == "platform_3"
    assert platform_zone("Platform_3") == "platform_3"
    assert platform_zone("grey_street") == "grey_street"
    assert platform_zone(None) == ""


def test_platform_zones_are_numbered():
    assert is_platform_zone("platform_12")
    assert not is_platform_zone("platform_abc")
