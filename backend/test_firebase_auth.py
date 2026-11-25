"""Run with python -m unittest -v test_firebase_auth from backend/. Needs no network."""
import datetime
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import jwt
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

import firebase_auth

PROJECT = "review-reader-1a2b3"


def make_pair(kid):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(private.public_key())
            .serial_number(1).not_valid_before(now).not_valid_after(now + datetime.timedelta(days=1)).sign(private, hashes.SHA256()))
    return private, cert.public_bytes(serialization.Encoding.PEM).decode()


class Fixture(unittest.TestCase):
    def setUp(self):
        self.private, pem = make_pair("k1")
        self.certs = {"k1": pem}
        self.fetches = 0

        def fetch():
            self.fetches += 1
            return dict(self.certs), 3600

        self.verifier = firebase_auth.FirebaseVerifier(PROJECT, fetch=fetch)

    def token(self, key=None, kid="k1", **changes):
        now = int(time.time())
        claims = {"iss": f"https://securetoken.google.com/{PROJECT}", "aud": PROJECT, "sub": "user_123",
                  "iat": now, "exp": now + 3600, "auth_time": now, "email": "a@example.com"}
        claims.update(changes)
        claims = {name: value for name, value in claims.items() if value is not None}
        return jwt.encode(claims, key or self.private, algorithm="RS256", headers={"kid": kid})


class VerifierTests(Fixture):
    def test_a_valid_token_is_accepted(self):
        self.assertEqual(self.verifier.verify(self.token())["sub"], "user_123")

    def test_certificates_are_cached(self):
        self.verifier.verify(self.token())
        self.verifier.verify(self.token())
        self.assertEqual(self.fetches, 1)

    def test_an_expired_token_is_rejected(self):
        with self.assertRaises(jwt.ExpiredSignatureError):
            self.verifier.verify(self.token(exp=int(time.time()) - 300))

    def test_a_token_for_another_project_is_rejected(self):
        with self.assertRaises(jwt.InvalidAudienceError):
            self.verifier.verify(self.token(aud="someone-elses-app"))
        with self.assertRaises(jwt.InvalidIssuerError):
            self.verifier.verify(self.token(iss="https://securetoken.google.com/someone-elses-app"))

    def test_a_token_without_a_user_is_rejected(self):
        with self.assertRaises(jwt.MissingRequiredClaimError):
            self.verifier.verify(self.token(sub=None))
        with self.assertRaises(jwt.InvalidTokenError):
            self.verifier.verify(self.token(sub=""))

    def test_a_token_signed_with_another_key_is_rejected(self):
        other, _ = make_pair("k1")
        with self.assertRaises(jwt.InvalidSignatureError):
            self.verifier.verify(self.token(key=other))

    def test_unknown_key_ids_and_unsigned_tokens_are_rejected(self):
        with self.assertRaises(jwt.InvalidTokenError):
            self.verifier.verify(self.token(kid="nope"))
        unsigned = jwt.encode({"iss": self.verifier.issuer, "aud": PROJECT, "sub": "u", "iat": int(time.time()), "exp": int(time.time()) + 60},
                              None, algorithm="none")
        with self.assertRaises(jwt.InvalidTokenError):
            self.verifier.verify(unsigned)

    def test_rotated_keys_are_picked_up(self):
        self.verifier.verify(self.token())
        new_private, new_pem = make_pair("k2")
        self.certs["k2"] = new_pem
        self.verifier._last_refresh -= 120  # pretend the last look was a while ago
        self.assertEqual(self.verifier.verify(self.token(key=new_private, kid="k2"))["sub"], "user_123")

    def test_unreachable_google_is_reported_separately(self):
        def broken():
            raise firebase_auth.KeysUnavailable("offline")
        verifier = firebase_auth.FirebaseVerifier(PROJECT, fetch=broken)
        with self.assertRaises(firebase_auth.KeysUnavailable):
            verifier.verify(self.token())


class EnvironmentTests(unittest.TestCase):
    def test_no_project_means_sign_in_is_off(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(firebase_auth.from_environment(dotenv_path="/nonexistent/.env"))

    def test_project_from_environment_or_env_file(self):
        with patch.dict(os.environ, {"FIREBASE_PROJECT_ID": PROJECT}, clear=True):
            self.assertEqual(firebase_auth.from_environment(dotenv_path="/nonexistent/.env").project_id, PROJECT)
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / ".env"
            env.write_text(f'# settings\nVITE_FIREBASE_PROJECT_ID="{PROJECT}"\n')
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(firebase_auth.from_environment(dotenv_path=env).project_id, PROJECT)

    def test_a_bad_project_id_raises_a_clear_error(self):
        with patch.dict(os.environ, {"FIREBASE_PROJECT_ID": "Not A Project"}, clear=True):
            with self.assertRaises(ValueError):
                firebase_auth.from_environment(dotenv_path="/nonexistent/.env")


if __name__ == "__main__":
    unittest.main()
