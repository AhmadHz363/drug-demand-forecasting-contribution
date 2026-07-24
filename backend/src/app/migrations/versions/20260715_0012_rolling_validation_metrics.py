"""rolling validation metrics

Revision ID: 20260715_0012
Revises: 20260715_0011
Create Date: 2026-07-15
"""

from alembic import op
import sqlalchemy as sa


revision = "20260715_0012"
down_revision = "20260715_0011"
branch_labels = None
depends_on = None

_ROLLING_COLUMNS = (
    "smape_7day_full",
    "smape_30day_full",
    "mase_7day_full",
    "mase_30day_full",
    "smape_7day_normal",
    "smape_30day_normal",
    "mase_7day_normal",
    "mase_30day_normal",
)


def upgrade() -> None:
    for column in _ROLLING_COLUMNS:
        op.add_column(
            "model_performance",
            sa.Column(column, sa.Numeric(precision=8, scale=4), nullable=True),
        )


def downgrade() -> None:
    for column in reversed(_ROLLING_COLUMNS):
        op.drop_column("model_performance", column)
