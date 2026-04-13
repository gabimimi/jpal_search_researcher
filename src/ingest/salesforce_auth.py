"""
Salesforce OAuth (password + security token) for server-side scripts.

Requires a Connected App with OAuth enabled and the “API (Enable OAuth Settings)”
flow. Your admin must allow the integration user’s profile to use the Connected App.

Environment:
  SALESFORCE_CLIENT_ID       Connected App Consumer Key
  SALESFORCE_CLIENT_SECRET   Connected App Consumer Secret
  SALESFORCE_USERNAME        Integration user login
  SALESFORCE_PASSWORD        User password only (not including security token)
  SALESFORCE_SECURITY_TOKEN  From Salesforce user settings (email if reset)

Optional:
  SALESFORCE_LOGIN_HOST      default login.salesforce.com (use test.salesforce.com for sandbox)
"""
from __future__ import annotations

import os
from typing import Tuple

import requests


def get_access_token() -> Tuple[str, str]:
    """
    Return (access_token, instance_url) using the OAuth2 password flow.
    """
    client_id = os.environ.get("SALESFORCE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("SALESFORCE_CLIENT_SECRET", "").strip()
    username = os.environ.get("SALESFORCE_USERNAME", "").strip()
    password = os.environ.get("SALESFORCE_PASSWORD", "").strip()
    token = os.environ.get("SALESFORCE_SECURITY_TOKEN", "").strip()
    host = os.environ.get("SALESFORCE_LOGIN_HOST", "login.salesforce.com").strip()

    missing = [
        n
        for n, v in [
            ("SALESFORCE_CLIENT_ID", client_id),
            ("SALESFORCE_CLIENT_SECRET", client_secret),
            ("SALESFORCE_USERNAME", username),
            ("SALESFORCE_PASSWORD", password),
            ("SALESFORCE_SECURITY_TOKEN", token),
        ]
        if not v
    ]
    if missing:
        raise RuntimeError(
            "Missing environment variables: " + ", ".join(missing) + ". "
            "See src/ingest/salesforce_auth.py docstring."
        )

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
