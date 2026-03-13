"""
CryptoSigner — HMAC-SHA256 tamper-proof signing for all agent responses.

Every completed agent job response is cryptographically signed:
  - content_hash : SHA-256 of the canonical JSON payload
  - signature    : HMAC-SHA256(content_hash:timestamp, SUPABASE_JWT_SECRET)
  - signed_at    : Unix epoch timestamp
  - algorithm    : "HMAC-SHA256"

This creates an auditable, tamper-evident chain for all regulatory decisions.
Regulatory affairs teams can cryptographically prove:
  - The exact advice given at any point in time
  - That the payload has not been altered after signing
  - The time the analysis was completed

Usage:
    signed = sign_payload(result_dict)
    verification = verify_signature(signed)
    # → {"valid": True, "message": "Signature valid…"}

Verify endpoint: POST /api/v1/integrity/verify-signature
"""

import hashlib
import hmac
import json
import logging
import time

logger = logging.getLogger(__name__)

_ALGORITHM = "HMAC-SHA256"
_VERSION = "1.0"


def _get_secret() -> str:
    """Retrieve signing secret from SUPABASE_JWT_SECRET. Raises if missing."""
    from app.core.config import get_settings
    s = get_settings().supabase_jwt_secret
    if s and len(s) >= 32:
        return s
    raise EnvironmentError(
        "SUPABASE_JWT_SECRET is missing or too short (need ≥32 chars). "
        "Set it in backend/.env — crypto signing is disabled until fixed."
    )


def _canonical(payload: dict) -> str:
    """Produce canonical JSON string of a payload (no _signed block)."""
    clean = {k: v for k, v in payload.items() if k != "_signed"}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str)


def sign_payload(payload: dict) -> dict:
    """
    Sign a result payload and inject a `_signed` metadata block.

    Returns a new dict — does NOT mutate the original.
    Never raises — returns the original dict unsigend if any error occurs.
    """
    try:
        secret = _get_secret()
        canonical = _canonical(payload)

        # SHA-256 content hash
        content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        # Timestamp
        ts = int(time.time())

        # HMAC-SHA256 over "hash:timestamp"
        message = f"{content_hash}:{ts}"
        sig = hmac.new(
            secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        signed = dict(payload)
        signed["_signed"] = {
            "hash": content_hash[:16],       # display prefix shown in UI
            "hash_full": content_hash,        # full hash for offline verification
            "signature": sig,
            "signed_at": ts,
            "algorithm": _ALGORITHM,
            "version": _VERSION,
        }
        return signed

    except Exception as e:
        logger.warning("CryptoSigner.sign_payload failed: %s", e)
        return payload  # Never fail silently — return unsigned payload


def verify_signature(payload: dict) -> dict:
    """
    Verify a previously signed payload.

    Args:
        payload: Complete job result dict including `_signed` block.

    Returns:
        {
            "valid": bool,
            "hash_match": bool,
            "sig_match": bool,
            "signed_at": int,
            "age_seconds": int,
            "message": str,
        }
    """
    signed_meta = payload.get("_signed")
    if not signed_meta:
        return {
            "valid": False,
            "hash_match": False,
            "sig_match": False,
            "signed_at": 0,
            "age_seconds": 0,
            "message": "No _signed block found in payload.",
        }

    try:
        secret = _get_secret()
        canonical = _canonical(payload)

        # Recompute content hash
        computed_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        stored_hash = signed_meta.get("hash_full", "")
        hash_match = hmac.compare_digest(computed_hash, stored_hash)

        # Recompute signature
        ts = signed_meta.get("signed_at", 0)
        message = f"{computed_hash}:{ts}"
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        stored_sig = signed_meta.get("signature", "")
        sig_match = hmac.compare_digest(expected_sig, stored_sig)

        valid = hash_match and sig_match
        age = int(time.time()) - ts

        return {
            "valid": valid,
            "hash_match": hash_match,
            "sig_match": sig_match,
            "signed_at": ts,
            "age_seconds": age,
            "hash_prefix": signed_meta.get("hash", ""),
            "algorithm": signed_meta.get("algorithm", _ALGORITHM),
            "message": (
                f"Signature valid -- payload unmodified. Signed {age}s ago."
                if valid
                else "Signature INVALID -- payload may have been tampered with."
            ),
        }

    except Exception as e:
        return {
            "valid": False,
            "hash_match": False,
            "sig_match": False,
            "signed_at": 0,
            "age_seconds": 0,
            "message": f"Verification error: {e}",
        }
