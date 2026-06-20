#!/usr/bin/env python3
"""Create a dashboard user (password is bcrypt-hashed in the database)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sqlalchemy import select

from app.core.database import SessionLocal
from app.core.security import hash_password
from app.models.user import User


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a dashboard user")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    if len(args.password) < 8:
        print("Password must be at least 8 characters", file=sys.stderr)
        sys.exit(1)

    email = args.email.strip().lower()
    db = SessionLocal()
    try:
        existing = db.scalar(select(User).where(User.email == email))
        if existing is not None:
            print(f"User already exists: {email}", file=sys.stderr)
            sys.exit(1)
        user = User(email=email, hashed_password=hash_password(args.password), is_active=True)
        db.add(user)
        db.commit()
        print(f"Created user id={user.id} email={email}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
