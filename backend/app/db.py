from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    pass


class RunRecord(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    teacher_id: Mapped[str] = mapped_column(String(80))
    task_prompt: Mapped[str] = mapped_column(Text)
    preset: Mapped[str | None] = mapped_column(String(80), nullable=True)
    target_params: Mapped[float] = mapped_column(Float)
    target_label: Mapped[str] = mapped_column(String(20))
    output_type: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(30), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    tokens_generated: Mapped[int] = mapped_column(Integer, default=0)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


def make_session_factory():
    settings = get_settings()
    database_url = settings.database_url
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    if database_url.startswith("sqlite:///") and database_url != "sqlite:///:memory:":
        Path(database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(database_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


SessionFactory = make_session_factory()


def get_session() -> Generator[Session, None, None]:
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()
