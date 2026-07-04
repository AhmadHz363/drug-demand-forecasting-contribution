"""extend model_performance for Phase 3 monitoring

Revision ID: 20260621_0008
Revises: 20260531_0007
Create Date: 2026-06-21
"""

from alembic import op
import sqlalchemy as sa


revision = "20260621_0008"
down_revision = "20260531_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "model_performance",
        sa.Column("mase", sa.Numeric(precision=8, scale=4), nullable=True),
    )
    op.add_column(
        "model_performance",
        sa.Column("training_run_id", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "model_performance",
        sa.Column("demand_segment", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "model_performance",
        sa.Column("data_quality_status", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "model_performance",
        sa.Column("weight_sarima", sa.Numeric(precision=6, scale=4), nullable=True),
    )
    op.add_column(
        "model_performance",
        sa.Column("weight_lgbm", sa.Numeric(precision=6, scale=4), nullable=True),
    )
    op.add_column(
        "model_performance",
        sa.Column("weight_tft", sa.Numeric(precision=6, scale=4), nullable=True),
    )
    op.create_index(
        "ix_model_performance_training_run_id",
        "model_performance",
        ["training_run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_model_performance_training_run_id", table_name="model_performance")
    op.drop_column("model_performance", "weight_tft")
    op.drop_column("model_performance", "weight_lgbm")
    op.drop_column("model_performance", "weight_sarima")
    op.drop_column("model_performance", "data_quality_status")
    op.drop_column("model_performance", "demand_segment")
    op.drop_column("model_performance", "training_run_id")
    op.drop_column("model_performance", "mase")
