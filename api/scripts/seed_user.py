from pathlib import Path
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select
import os, asyncio

# ── optional .env load (UTF-8 only) ───────────────────────────────
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        print("⚠️  .env not UTF-8 – skipped")

# ── model import (kept same) ──────────────────────────────────────
import sys; sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.models import Base, User

USERNAME = os.getenv("USERNAME_KEY", "alice")
PASSWORD = os.getenv("USER_PASSWORD", "supersecretvalue")

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
engine = create_async_engine("sqlite+aiosqlite:///./app.db")
session_factory = async_sessionmaker(engine, expire_on_commit=False)

async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as db:
        result = await db.execute(select(User).where(User.username == USERNAME))
        user = result.scalar_one_or_none()
        hashed = pwd.hash(PASSWORD)

        if user:
            user.hashed_password = hashed
            action = "Updated"
        else:
            db.add(User(username=USERNAME, hashed_password=hashed))
            action = "Created"
        await db.commit()
        print(f"{action} user {USERNAME}")

if __name__ == "__main__":
    asyncio.run(main())


