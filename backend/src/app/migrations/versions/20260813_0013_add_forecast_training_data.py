"""add forecast_training_data table for SHIELD-XR cleaned panel

Revision ID: 20260813_0013
Revises: 20260715_0012
Create Date: 2026-08-13
"""

from alembic import op
import sqlalchemy as sa


revision = "20260813_0013"
down_revision = "20260715_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "forecast_training_data",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("training_run_id", sa.String(length=36), nullable=False),
        sa.Column("drug_code", sa.String(length=255), nullable=False),
        sa.Column("demand_date", sa.Date(), nullable=False),
        sa.Column("demand_raw", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("demand_clean", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("drug_name", sa.String(length=1024), nullable=True),
        sa.Column("drug_category", sa.String(length=255), nullable=True),
        sa.Column("sb_class", sa.String(length=32), nullable=False),
        sa.Column("is_anomaly_day", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_dropout_day", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("ceiling", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "training_run_id",
            "drug_code",
            "demand_date",
            name="uq_forecast_training_data_run_drug_date",
        ),
    )
    op.create_index(
        "ix_forecast_training_data_run",
        "forecast_training_data",
        ["training_run_id"],
    )
    op.create_index(
        "ix_forecast_training_data_drug_date",
        "forecast_training_data",
        ["drug_code", "demand_date"],
    )


def downgrade() -> None:
    op.drop_index("ix_forecast_training_data_drug_date", table_name="forecast_training_data")
    op.drop_index("ix_forecast_training_data_run", table_name="forecast_training_data")
    op.drop_table("forecast_training_data")
