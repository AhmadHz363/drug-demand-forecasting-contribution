"""drop legacy per-drug forecasting tables (pre SHIELD-XR)

Revision ID: 20260813_0014
Revises: 20260813_0013
Create Date: 2026-08-13
"""

from alembic import op


revision = "20260813_0014"
down_revision = "20260813_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_stockout_flags_flag_date", table_name="stockout_flags")
    op.drop_index("ix_stockout_flags_drug_code", table_name="stockout_flags")
    op.drop_table("stockout_flags")

    op.drop_index("ix_supplier_lead_times_drug_code", table_name="supplier_lead_times")
    op.drop_table("supplier_lead_times")

    op.drop_index("ix_daily_drug_demand_demand_date", table_name="daily_drug_demand")
    op.drop_index("ix_daily_drug_demand_drug_code", table_name="daily_drug_demand")
    op.drop_table("daily_drug_demand")


def downgrade() -> None:
    import sqlalchemy as sa

    op.create_table(
        "daily_drug_demand",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_code", sa.String(length=255), nullable=False),
        sa.Column("demand_date", sa.Date(), nullable=False),
        sa.Column("total_quantity", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("drug_code", "demand_date", name="uq_daily_drug_demand_drug_code_date"),
    )
    op.create_index("ix_daily_drug_demand_drug_code", "daily_drug_demand", ["drug_code"])
    op.create_index("ix_daily_drug_demand_demand_date", "daily_drug_demand", ["demand_date"])

    op.create_table(
        "supplier_lead_times",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_code", sa.String(length=255), nullable=False),
        sa.Column("avg_lead_time_days", sa.Numeric(precision=6, scale=2), nullable=False),
        sa.Column("lead_time_std_days", sa.Numeric(precision=6, scale=2), nullable=False),
        sa.Column("reliability_score", sa.Numeric(precision=4, scale=3), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("drug_code", name="uq_supplier_lead_times_drug_code"),
    )
    op.create_index("ix_supplier_lead_times_drug_code", "supplier_lead_times", ["drug_code"])

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
