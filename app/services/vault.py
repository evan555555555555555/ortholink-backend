"""
OrthoLink Vault — AES-256-GCM Authenticated Encryption
NIST SP 800-38D compliant symmetric encryption for sensitive payloads.

Key specification:
  - AES-256-GCM (AESGCM from cryptography.hazmat)
  - 96-bit random IV per operation (NIST recommended for GCM §8.2.1)
  - GCM authentication tag: 128-bit (16 bytes, default for AESGCM)
  - No Additional Authenticated Data (payload integrity covers it)

Key derivation:
  - PBKDF2-HMAC-SHA256, 100,000 iterations
  - Domain-specific static salt (not random; KDF salt ≠ IV)
  - Input: VAULT_KEY env var, fallback to SUPABASE_JWT_SECRET
  - If neither configured: vault is disabled (v0 = base64 JSON, not encrypted)

Wire format:
  - Encrypted:  "v1:<base64url(iv[12] || ciphertext+tag[*+16])>"
  - Unencrypted: "v0:<base64(json)>"  (development fallback only)

Usage:
  from app.services.vault import encrypt, decrypt
  token = encrypt({"claim": "value", "score": 0.95})
  data  = decrypt(token)
"""

import base64
import json
import logging
import os
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_KEY_BYTES: int = 32          # AES-256 = 32 bytes
_IV_BYTES: int = 12           # 96-bit IV (NIST SP 800-38D §8.2.1 recommendation)
_GCM_TAG_BYTES: int = 16      # 128-bit GCM authentication tag
_KDF_ITERATIONS: int = 100_000
_KDF_SALT: bytes = b"ortholink-vault-v1-kdf-salt-2026"  # domain-specific, static


# ── Key derivation ────────────────────────────────────────────────────────────

def _derive_key(secret: str) -> bytes:
    """
    PBKDF2-HMAC-SHA256: derive 256-bit AES key from a passphrase.
    100,000 iterations provides ~0.1s per attempt on modern hardware (brute-force resistant).
    """
    kdf = PBKDF2HMAC(
        algorithm=SHA256(),
        length=_KEY_BYTES,
        salt=_KDF_SALT,
        iterations=_KDF_ITERATIONS,
    )
    return kdf.derive(secret.encode("utf-8"))


def _get_key() -> bytes | None:
    """
    Resolve the vault encryption key from environment.
    Returns None if no key is configured (vault disabled — development only).
    Priority: VAULT_KEY → SUPABASE_JWT_SECRET → None
    """
    raw = (
        os.environ.get("VAULT_KEY", "").strip()
        or os.environ.get("SUPABASE_JWT_SECRET", "").strip()
    )
    if not raw or raw in {"change_me_in_production", "your-secret-here"}:
        logger.debug("Vault: no key configured — encryption disabled (v0 mode)")
        return None
    return _derive_key(raw)


# ── Public API ────────────────────────────────────────────────────────────────

def encrypt(payload: dict) -> str:
    """
    AES-256-GCM encrypt a dictionary.

    Returns:
      "v1:<base64(iv || ciphertext+tag)>"  — encrypted
      "v0:<base64(json)>"                  — unencrypted fallback (no key configured)

    The GCM tag (16 bytes) is automatically appended by AESGCM.encrypt().
    """
    key = _get_key()
    plaintext = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    if key is None:
        # No key — base64 JSON without encryption (development fallback)
        return "v0:" + base64.b64encode(plaintext).decode("ascii")

    iv = secrets.token_bytes(_IV_BYTES)          # cryptographically random 96-bit IV
    aesgcm = AESGCM(key)
    ciphertext_and_tag = aesgcm.encrypt(iv, plaintext, None)   # nonce, data, aad=None
    blob = iv + ciphertext_and_tag               # pack IV || ciphertext+tag
    return "v1:" + base64.b64encode(blob).decode("ascii")


def decrypt(token: str) -> dict:
    """
    Decrypt a vault token produced by encrypt().

    Raises:
      ValueError — bad format, authentication failure, key unavailable, or wrong version
    """
    if not isinstance(token, str):
        raise ValueError("Vault token must be a string")

    if token.startswith("v0:"):
        raw = base64.b64decode(token[3:])
        return json.loads(raw)

    if not token.startswith("v1:"):
        raise ValueError(f"Unknown vault token version (prefix: {token[:3]!r})")

    key = _get_key()
    if key is None:
        raise ValueError("Vault key not configured; cannot decrypt v1 token")

    try:
        blob = base64.b64decode(token[3:])
    except Exception:
        raise ValueError("Vault token is not valid base64")

    if len(blob) < _IV_BYTES + _GCM_TAG_BYTES:
        raise ValueError(
            f"Vault token too short (got {len(blob)} bytes, need ≥ {_IV_BYTES + _GCM_TAG_BYTES})"
        )

    iv = blob[:_IV_BYTES]
    ciphertext_and_tag = blob[_IV_BYTES:]

    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(iv, ciphertext_and_tag, None)
    except Exception:
        # Do NOT leak internal details (timing / oracle attacks)
        raise ValueError("Vault decryption failed — authentication tag mismatch")

    return json.loads(plaintext)


def is_vault_enabled() -> bool:
    """Return True if a vault key is configured (encryption is active)."""
    return _get_key() is not None
