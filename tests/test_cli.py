import pytest

from lensuq import __version__
from lensuq.cli import main


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_missing_subcommand():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_demo_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr("lensuq.demo.run_demo", lambda: calls.append("demo"))
    assert main(["demo"]) == 0
    assert calls == ["demo"]
