"""Alembic environment configuration."""

import sys
from logging.config import fileConfig
from pathlib import Path

# Alembic splits prepend_sys_path on spaces; project paths with spaces break.
_src_root = Path(__file__).resolve().parents[2]
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

from alembic import context
from sqlalchemy import create_engine, pool

from app.core.config import settings
from app.core.database import Base
from app.models.forecast_training_data import ForecastTrainingData  # noqa: F401
from app.models.hospital_daily_demand_enriched import HospitalDailyDemandEnriched  # noqa: F401
from app.models.hospital_receipt_raw import HospitalReceiptRaw  # noqa: F401
from app.models.model_performance import ModelPerformance  # noqa: F401
from app.models.user import User  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = settings.database_url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(settings.database_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
