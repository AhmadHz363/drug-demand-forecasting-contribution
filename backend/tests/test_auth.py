"""Auth API tests (register, login, protected routes)."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import SessionLocal, engine
from app.main import app
from app.models.user import User


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return "users" in set(inspect(engine).get_table_names())
    except SQLAlchemyError:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(),
    reason="PostgreSQL with users table required",
)


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def unique_email():
    return f"auth-test-{uuid.uuid4().hex[:12]}@example.com"


def _delete_user(email: str) -> None:
    db = SessionLocal()
    try:
        db.query(User).filter(User.email == email.lower()).delete()
        db.commit()
    finally:
        db.close()


@pytest.fixture(autouse=True)
def cleanup_user(unique_email):
    yield
    _delete_user(unique_email)


def test_register_login_and_me(client: TestClient, unique_email: str):
    password = "secure-pass-1"
    reg = client.post(
        "/auth/register",
        json={"email": unique_email, "password": password},
    )
    assert reg.status_code == 201
    assert reg.json()["email"] == unique_email.lower()

    unauth = client.get("/drugs")
    assert unauth.status_code == 401

    login = client.post(
        "/auth/login",
        json={"email": unique_email, "password": password},
    )
    assert login.status_code == 200
    body = login.json()
    token = body["access_token"]
    assert body["token_type"] == "bearer"
    assert body["user"]["email"] == unique_email.lower()

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == unique_email.lower()

    drugs = client.get("/drugs", headers={"Authorization": f"Bearer {token}"})
    assert drugs.status_code == 200


def test_login_rejects_wrong_password(client: TestClient, unique_email: str):
    password = "secure-pass-1"
    client.post("/auth/register", json={"email": unique_email, "password": password})
    res = client.post(
        "/auth/login",
        json={"email": unique_email, "password": "wrong-password"},
    )
    assert res.status_code == 401
