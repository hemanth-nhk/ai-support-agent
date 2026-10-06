"""
Logging configuration. Logs are informational/operational only -- never
log full chain-of-thought or raw LLM prompts containing customer data,
per the guardrails around not exposing internal reasoning.
"""
import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
