"""
Salesforce authentication for server-side scripts.

Supports two modes (auto-detected):

1. simple-salesforce (recommended for local dev)
   Only needs: SALESFORCE_USERNAME, SALESFORCE_PASSWORD, SALESFORCE_SECURITY_TOKEN

2. OAuth2 Connected App (for production / CI)
   Also needs: SALESFORCE_CLIENT_ID, SALESFORCE_CLIENT_SECRET

Environment:
  SALESFORCE_USERNAME        Salesforce login email
  SALESFORCE_PASSWORD        User password (not including security token)
  SALESFORCE_SECURITY_TOKEN  From Salesforce -> Settings -> Reset My Security Token

Optional:
  SALESFORCE_CLIENT_ID       Connected App Consumer Key  (enables OAuth2 flow)
  SALESFORCE_CLIENT_SECRET   Connected App Consumer Secret
  SALESFORCE_LOGIN_HOST      default login.salesforce.com (use test.salesforce.com for sandbox)
"""
from __future__ import annotations

import os
from typing import Tuple

import requests


def get_access_token() -> Tuple[str, str]:
    """
    Return (access_token, instance_url).

    Uses simple-salesforce (SOAP login) when client_id/client_secret are absent,
    falls back to OAuth2 password flow when they are present.
    """
    username = os.environ.get("SALESFORCE_USERNAME", "").strip()
    password = os.environ.get("SALESFORCE_PASSWORD", "").strip()
    token = os.environ.get("SALESFORCE_SECURITY_TOKEN", "").strip()
    host = os.environ.get("SALESFORCE_LOGIN_HOST", "login.salesforce.com").strip()

    missing = [
        n for n, v in [
            ("SALESFORCE_USERNAME", username),
            ("SALESFORCE_PASSWORD", password),
            ("SALESFORCE_SECURITY_TOKEN", token),
        ]
        if not v
    ]
    if missing:
        raise RuntimeError(
            "Missing environment variables: " + ", ".join(missing) + ". "
            "Set them in .env or export in your shell."
        )

    client_id = os.environ.get("SALESFORCE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("SALESFORCE_CLIENT_SECRET", "").strip()

    if client_id and client_secret:
        return _oauth2_flow(username, password, token, client_id, client_secret, host)

    return _simple_salesforce_flow(username, password, token, host)


def _simple_salesforce_flow(
    username: str, password: str, token: str, host: str,
) -> Tuple[str, str]:
    """Authenticate via simple-salesforce SOAP login (no Connected App needed)."""
    try:
        from simple_salesforce import Salesforce
    except ImportError:
        raise RuntimeError(
            "simple-salesforce is not installed. Run: pip install simple-salesforce"
        )

    domain = "test" if "test.salesforce.com" in host else None
    sf = Salesforce(username=username, password=password, security_token=token, domain=domain)
    instance_url = f"https://{sf.sf_instance}"
    return sf.session_id, instance_url


def _oauth2_flow(
    username: str, password: str, token: str,
    client_id: str, client_secret: str, host: str,
) -> Tuple[str, str]:
    """Authenticate via OAuth2 password flow (requires Connected App)."""
    url = f"https://{host}/services/oauth2/token"
    data = {
        "grant_type": "password",
        "client_id": client_id,
        "client_secret": client_secret,
        "username": username,
        "password": password + token,
    }
    r = requests.post(url, data=data, timeout=60)
    if not r.ok:
        try:
            err = r.json()
            msg = err.get("error_description") or err.get("error") or r.text
        except Exception:
            msg = r.text
        raise RuntimeError(f"Salesforce token request failed ({r.status_code}): {msg}")

    payload = r.json()
    access = payload.get("access_token")
    instance = payload.get("instance_url")
    if not access or not instance:
        raise RuntimeError("Salesforce token response missing access_token or instance_url")
    return access, instance
