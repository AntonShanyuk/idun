import idun
from idun._version import __version__


def test_the_package_reports_the_leaf_version() -> None:
    assert idun.__version__ == __version__ == "2.0.0"
