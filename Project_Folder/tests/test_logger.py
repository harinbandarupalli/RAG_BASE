"""Unit tests for the structured logger and secret redactor."""

from src.core.logger import get_logger, redact_secrets, set_log_context, timed_block


def test_redact_openai_secret():
    text = "Authorization: sk-proj-1234567890abcdef12345678"
    redacted = redact_secrets(text)
    assert "[REDACTED]" in redacted
    assert "sk-proj-" not in redacted


def test_redact_google_api_key():
    text = "key=AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q"
    redacted = redact_secrets(text)
    assert "[REDACTED]" in redacted
    assert "AIzaSy" not in redacted


def test_timed_block():
    log = get_logger("test_timer")
    with timed_block("quick_operation", log) as m:
        assert m["name"] == "quick_operation"
    assert "latency_ms" in m
    assert m["latency_ms"] >= 0.0
