"""split validation metrics by supply regime

Revision ID: 20260715_0011
Revises: 20260713_0010
Create Date: 2026-07-15
"""

from alembic import op
import sqlalchemy as sa


revision = "20260715_0011"
down_revision = "20260713_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "model_performance",
        sa.Column("smape_normal_supply", sa.Numeric(precision=8, scale=4), nullable=True),
    )
    op.add_column(
        "model_performance",
        sa.Column("mase_normal_supply", sa.Numeric(precision=8, scale=4), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("model_performance", "mase_normal_supply")
    op.drop_column("model_performance", "smape_normal_supply")
