"""
Shared pytest configuration and fixtures for ai-support-agent tests.
Provides SessionAwareTestClient to establish valid signed session tokens
for legitimate dataset test workflows without weakening security protections.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.session_security import sign_session_id, HEADER_NAME, COOKIE_NAME


class SessionAwareTestClient(TestClient):
    """
    Shared TestClient fixture for legitimate test workflows.
    Automatically establishes and attaches a valid signed X-Session-Token
    and session cookie whenever a test request payload specifies session_id
    and no explicit X-Session-Token header has been provided.
    """

    def request(self, method: str, url, **kwargs):
        headers = dict(kwargs.get("headers") or {})
        if HEADER_NAME not in headers:
            sess_id = None
            if "json" in kwargs and isinstance(kwargs["json"], dict):
                sess_id = kwargs["json"].get("session_id")
            elif "data" in kwargs and isinstance(kwargs["data"], dict):
                sess_id = kwargs["data"].get("session_id")
            elif "params" in kwargs and isinstance(kwargs["params"], dict):
                sess_id = kwargs["params"].get("session_id")
            if sess_id:
                token = sign_session_id(str(sess_id))
                headers[HEADER_NAME] = token
                kwargs["headers"] = headers
                self.cookies.set(COOKIE_NAME, token)
        return super().request(method, url, **kwargs)


@pytest.fixture
def client():
    """Shared client fixture delivering SessionAwareTestClient."""
    return SessionAwareTestClient(app)
