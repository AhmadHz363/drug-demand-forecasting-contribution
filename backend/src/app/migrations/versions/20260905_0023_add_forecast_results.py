"""add forecast_results table for overview KPIs

Revision ID: 20260905_0023
Revises: 20260905_0022
Create Date: 2026-09-05
"""

from alembic import op
import sqlalchemy as sa


revision = "20260905_0023"
down_revision = "20260905_0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "forecast_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_code", sa.String(length=255), nullable=False),
        sa.Column("center_syn_id", sa.String(length=255), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("forecast_date", sa.Date(), nullable=False),
        sa.Column("p5", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("p10", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("p50", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("p90", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("p95", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("model_weight_sarima", sa.Numeric(precision=6, scale=4), nullable=False),
        sa.Column("model_weight_lgbm", sa.Numeric(precision=6, scale=4), nullable=False),
        sa.Column("model_weight_tft", sa.Numeric(precision=6, scale=4), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "drug_code",
            "center_syn_id",
            "generated_at",
            "forecast_date",
            name="uq_forecast_results_drug_center_generated_date",
        ),
    )
    op.create_index("ix_forecast_results_drug_code", "forecast_results", ["drug_code"])
    op.create_index("ix_forecast_results_generated_at", "forecast_results", ["generated_at"])
    op.create_index("ix_forecast_results_forecast_date", "forecast_results", ["forecast_date"])


def downgrade() -> None:
    op.drop_index("ix_forecast_results_forecast_date", table_name="forecast_results")
    op.drop_index("ix_forecast_results_generated_at", table_name="forecast_results")
    op.drop_index("ix_forecast_results_drug_code", table_name="forecast_results")
    op.drop_table("forecast_results")
