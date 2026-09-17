"""restore SHIELD-XR training persistence tables

Revision ID: 20260814_0018
Revises: 20260813_0017
Create Date: 2026-08-14
"""

from alembic import op
import sqlalchemy as sa


revision = "20260814_0018"
down_revision = "20260813_0017"
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
        sa.Column("is_anomaly_day", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("is_dropout_day", sa.Boolean(), server_default=sa.false(), nullable=False),
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
    op.create_index("ix_forecast_training_data_run", "forecast_training_data", ["training_run_id"])
    op.create_index(
        "ix_forecast_training_data_drug_date",
        "forecast_training_data",
        ["drug_code", "demand_date"],
    )

    op.create_table(
        "model_performance",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_code", sa.String(length=255), nullable=False),
        sa.Column("model_name", sa.String(length=32), nullable=False),
        sa.Column("smape", sa.Numeric(precision=8, scale=4), nullable=False),
        sa.Column("smape_normal_supply", sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column("mase", sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column("mase_normal_supply", sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column("coverage_90", sa.Numeric(precision=6, scale=4), server_default="0", nullable=False),
        sa.Column(
            "evaluated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("training_run_id", sa.String(length=36), nullable=True),
        sa.Column("demand_segment", sa.String(length=16), nullable=True),
        sa.Column("data_quality_status", sa.String(length=16), nullable=True),
        sa.Column("weight_sarima", sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column("weight_lgbm", sa.Numeric(precision=6, scale=4), nullable=True),
        sa.Column("weight_classical", sa.Numeric(precision=6, scale=4), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_performance_drug_code", "model_performance", ["drug_code"])
    op.create_index("ix_model_performance_training_run_id", "model_performance", ["training_run_id"])


def downgrade() -> None:
    op.drop_index("ix_model_performance_training_run_id", table_name="model_performance")
    op.drop_index("ix_model_performance_drug_code", table_name="model_performance")
    op.drop_table("model_performance")
    op.drop_index("ix_forecast_training_data_drug_date", table_name="forecast_training_data")
    op.drop_index("ix_forecast_training_data_run", table_name="forecast_training_data")
    op.drop_table("forecast_training_data")
