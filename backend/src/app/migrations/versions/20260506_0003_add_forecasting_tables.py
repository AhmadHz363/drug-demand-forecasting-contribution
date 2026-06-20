"""add forecasting tables

Revision ID: 20260506_0003
Revises: 20260505_0003
Create Date: 2026-05-06
"""

from alembic import op
import sqlalchemy as sa


revision = "20260506_0003"
down_revision = "20260505_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "stockout_flags",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_code", sa.String(length=255), nullable=False),
        sa.Column("center_syn_id", sa.String(length=255), nullable=True),
        sa.Column("flag_date", sa.Date(), nullable=False),
        sa.Column("observed_quantity", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("estimated_true_demand", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("correction_method", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_stockout_flags_drug_code", "stockout_flags", ["drug_code"])
    op.create_index("ix_stockout_flags_flag_date", "stockout_flags", ["flag_date"])

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

    op.create_table(
        "model_performance",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_code", sa.String(length=255), nullable=False),
        sa.Column("model_name", sa.String(length=32), nullable=False),
        sa.Column("smape", sa.Numeric(precision=8, scale=4), nullable=False),
        sa.Column("coverage_90", sa.Numeric(precision=6, scale=4), nullable=False),
        sa.Column(
            "evaluated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_performance_drug_code", "model_performance", ["drug_code"])


def downgrade() -> None:
    op.drop_index("ix_model_performance_drug_code", table_name="model_performance")
    op.drop_table("model_performance")

    op.drop_index("ix_forecast_results_forecast_date", table_name="forecast_results")
    op.drop_index("ix_forecast_results_generated_at", table_name="forecast_results")
    op.drop_index("ix_forecast_results_drug_code", table_name="forecast_results")
    op.drop_table("forecast_results")

    op.drop_index("ix_stockout_flags_flag_date", table_name="stockout_flags")
    op.drop_index("ix_stockout_flags_drug_code", table_name="stockout_flags")
    op.drop_table("stockout_flags")
