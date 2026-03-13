#!/usr/bin/env python3
"""
OrthoLink — Dev JWT Generator
Generates a signed JWT for local development and testing.

Usage:
    # Reviewer token (default)
    python scripts/make_dev_token.py

    # Admin token (required for POST /api/v1/alerts/check-changes)
    python scripts/make_dev_token.py --role admin

    # Custom org and expiry
    python scripts/make_dev_token.py --role admin --org my-org --days 30

Set ORTHOLINK_DEV_TOKEN in your shell or pass it in the Authorization header:
    export ORTHOLINK_DEV_TOKEN=$(python scripts/make_dev_token.py --role admin)
    curl -H "Authorization: Bearer $ORTHOLINK_DEV_TOKEN" http://localhost:8000/api/v1/alerts
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Ensure we can import the backend package ──────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_token(
    role: str = "reviewer",
    org_id: str = "dev-org",
    user_id: str = "dev-user",
    days: int = 7,
    secret: str | None = None,
) -> str:
    """Sign and return a JWT with the given claims."""
    try:
        import jwt as _jwt
    except ImportError:
        print(
            "ERROR: PyJWT not installed. Run: pip install PyJWT --break-system-packages",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve secret: CLI arg → env var → hard-coded dev default
    _secret = (
        secret
        or os.getenv("JWT_SECRET")
        or "dev-secret-change-me"
    )

    now = int(time.time())
    payload = {
        "sub": user_id,
        "org_id": org_id,
        "role": role,
        "iat": now,
        "exp": now + (days * 86400),
    }

    algorithm = os.getenv("JWT_ALGORITHM", "HS256")
    token = _jwt.encode(payload, _secret, algorithm=algorithm)
    return token


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a signed dev JWT for OrthoLink local development."
    )
    parser.add_argument(
        "--role",
        choices=["reviewer", "admin", "viewer"],
        default="reviewer",
        help="Token role (default: reviewer). Use 'admin' for check-changes endpoint.",
    )
    parser.add_argument(
        "--org",
        default="dev-org",
        help="org_id claim (default: dev-org)",
    )
    parser.add_argument(
        "--user",
        default="dev-user",
        help="sub / user_id claim (default: dev-user)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Token expiry in days (default: 7)",
    )
    parser.add_argument(
        "--secret",
        default=None,
        help="JWT signing secret (overrides JWT_SECRET env var and dev default)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full JSON payload alongside the token",
    )
    args = parser.parse_args()

    token = make_token(
        role=args.role,
        org_id=args.org,
        user_id=args.user,
        days=args.days,
        secret=args.secret,
    )

    if args.json:
        print(json.dumps({"token": token, "role": args.role, "org_id": args.org, "days": args.days}, indent=2))
    else:
        print(token)

    if args.role == "admin":
        print(
            f"\n[OK] Admin token generated (expires in {args.days} days).\n"
            "   Set it in your shell:\n"
            f"   export ORTHOLINK_DEV_TOKEN=$(python scripts/make_dev_token.py --role admin)\n"
            "   Then call:\n"
            "   curl -X POST http://localhost:8000/api/v1/alerts/check-changes \\\n"
            '        -H "Authorization: Bearer $ORTHOLINK_DEV_TOKEN" \\\n'
            '        -H "Content-Type: application/json" \\\n'
            '        -d \'{"country": "US"}\'\n',
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
