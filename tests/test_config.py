"""Tests for the env-var based server configuration."""

import pytest

from gemma4_delimiter_proxy.__main__ import get_config


def test_defaults(monkeypatch):
    for var in ("GEMMA_PROXY_HOST", "GEMMA_PROXY_PORT", "GEMMA_PROXY_LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)
    config = get_config()
    assert config.host == "0.0.0.0"
    assert config.port == 4001
    assert config.log_level == "info"


def test_env_vars_respected(monkeypatch):
    monkeypatch.setenv("GEMMA_PROXY_HOST", "127.0.0.1")
    monkeypatch.setenv("GEMMA_PROXY_PORT", "8080")
    monkeypatch.setenv("GEMMA_PROXY_LOG_LEVEL", "debug")
    config = get_config()
    assert config.host == "127.0.0.1"
    assert config.port == 8080
    assert config.log_level == "debug"


def test_invalid_port_exits_with_clear_error(monkeypatch):
    monkeypatch.setenv("GEMMA_PROXY_PORT", "not-a-port")
    with pytest.raises(SystemExit) as excinfo:
        get_config()
    assert "GEMMA_PROXY_PORT must be an integer" in str(excinfo.value)


@pytest.mark.parametrize("bad_port", ["0", "65536", "99999"])
def test_out_of_range_port_exits_with_clear_error(monkeypatch, bad_port):
    monkeypatch.setenv("GEMMA_PROXY_PORT", bad_port)
    with pytest.raises(SystemExit) as excinfo:
        get_config()
    assert "must be 1-65535" in str(excinfo.value)
