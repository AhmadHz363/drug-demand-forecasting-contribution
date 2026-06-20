"""add daily_drug_demand

Revision ID: 20260505_0003
Revises: 20260505_0002
Create Date: 2026-05-05
"""

from alembic import op
import sqlalchemy as sa


revision = "20260505_0003"
down_revision = "20260505_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "daily_drug_demand",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_code", sa.String(length=255), nullable=False),
        sa.Column("demand_date", sa.Date(), nullable=False),
        sa.Column("total_quantity", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "drug_code",
            "demand_date",
            name="uq_daily_drug_demand_drug_code_date",
        ),
    )
    op.create_index("ix_daily_drug_demand_drug_code", "daily_drug_demand", ["drug_code"])
    op.create_index("ix_daily_drug_demand_demand_date", "daily_drug_demand", ["demand_date"])


def downgrade() -> None:
    op.drop_index("ix_daily_drug_demand_demand_date", table_name="daily_drug_demand")
    op.drop_index("ix_daily_drug_demand_drug_code", table_name="daily_drug_demand")
    op.drop_table("daily_drug_demand")
