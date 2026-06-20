from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Same DB as postgresql://postgres@localhost:5432/DrugForecastingDb — +psycopg driver matches requirements (psycopg v3).
    database_url: str = "postgresql+psycopg://postgres@localhost:5432/DrugForecastingDb"
    app_name: str = "Drug Receipts API"

    jwt_secret_key: str = "change-me-in-production-use-a-long-random-secret"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    auth_enabled: bool = True


settings = Settings()
