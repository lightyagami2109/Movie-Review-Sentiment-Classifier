"""Check Firebase sign-in tokens (ID tokens) using Google's public certificates.

No secret is needed. A token is valid if Google's published certificate matches its
signature, it has not expired, and it was issued for this Firebase project. The only
setting is the project id, which is public (it is also in the website's settings).
"""
import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

import jwt
from cryptography import x509

CERTS_URL = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"


class KeysUnavailable(Exception):
    """Google's certificates could not be fetched, so no token can be checked right now."""


def fetch_certificates():
    """Download the current certificates. Returns ({key id: PEM text}, seconds they stay valid)."""
    try:
        with urllib.request.urlopen(CERTS_URL, timeout=5) as response:
            certs = json.load(response)
            match = re.search(r"max-age=(\d+)", response.headers.get("Cache-Control", ""))
    except (OSError, ValueError) as error:
        raise KeysUnavailable(str(error)) from error
    return certs, int(match.group(1)) if match else 3600


class FirebaseVerifier:
    def __init__(self, project_id, fetch=fetch_certificates):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{3,}", project_id):
            raise ValueError("That does not look like a Firebase project id (for example my-app-1a2b3)")
        self.project_id = project_id
        self.issuer = f"https://securetoken.google.com/{project_id}"
        self._fetch = fetch
        self._lock = threading.Lock()
        self._certs = {}
        self._expires = 0.0
        self._last_refresh = 0.0

    def _refresh(self):
        certs, max_age = self._fetch()
        self._certs = certs
        self._last_refresh = time.time()
        self._expires = self._last_refresh + max(60, min(max_age, 86400))

    def _public_key(self, key_id):
        with self._lock:
            now = time.time()
            if not self._certs or now >= self._expires:
                self._refresh()
            elif key_id not in self._certs and now - self._last_refresh > 60:
                self._refresh()  # Google rotated its keys; look again (at most once a minute)
            pem = self._certs.get(key_id)
        if pem is None:
            raise jwt.InvalidTokenError("Token was signed with an unknown key")
        return x509.load_pem_x509_certificate(pem.encode()).public_key()

    def verify(self, token):
        """Return the token's claims, or raise a jwt exception if it is not valid."""
        header = jwt.get_unverified_header(token)
        if header.get("alg") != "RS256" or not header.get("kid"):
            raise jwt.InvalidTokenError("Unsupported token")
        claims = jwt.decode(
            token,
            self._public_key(header["kid"]),
            algorithms=["RS256"],
            audience=self.project_id,
            issuer=self.issuer,
            leeway=10,
            options={"require": ["exp", "iat", "sub", "aud"]},
        )
        if not isinstance(claims["sub"], str) or not claims["sub"]:
            raise jwt.InvalidTokenError("Token has no user")
        return claims


def read_dotenv(path):
    """Read KEY=VALUE lines from a .env file (comments and blank lines are skipped)."""
    values = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip().removeprefix("export ").strip()] = value.strip().strip("\"'")
    return values


def from_environment(dotenv_path=None):
    """Build a verifier from FIREBASE_PROJECT_ID (or the website's VITE_FIREBASE_PROJECT_ID).
    Looks in the environment first, then in the project's .env file. Returns None if not set."""
    saved = read_dotenv(dotenv_path or Path(__file__).resolve().parent.parent / ".env")

    def setting(name):
        return (os.getenv(name) or saved.get(name) or "").strip()

    project_id = setting("FIREBASE_PROJECT_ID") or setting("VITE_FIREBASE_PROJECT_ID")
    return FirebaseVerifier(project_id) if project_id else None
