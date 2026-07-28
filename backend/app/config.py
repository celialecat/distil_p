from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./distil.db"
    artifact_dir: Path = Path("./artifacts")
    teacher_api_base: str | None = None
    teacher_api_key: str | None = None
    teacher_api_model: str | None = None
    cors_origins: str = "http://localhost:3000"
    pipeline_delay: float = 1.5

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    return settings
