"""add feature engineering stub tables

Revision ID: 20260506_0004
Revises: 20260506_0003
Create Date: 2026-05-06
"""

from alembic import op
import sqlalchemy as sa


revision = "20260506_0004"
down_revision = "20260506_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hospital_census",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("census_date", sa.Date(), nullable=False),
        sa.Column("bed_occupancy_rate", sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column("weekly_surgery_count", sa.Integer(), nullable=False),
        sa.Column("center_syn_id", sa.String(length=255), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_hospital_census_census_date", "hospital_census", ["census_date"])
    op.create_index("ix_hospital_census_center_syn_id", "hospital_census", ["center_syn_id"])

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


def downgrade() -> None:
    op.drop_index("ix_supplier_lead_times_drug_code", table_name="supplier_lead_times")
    op.drop_table("supplier_lead_times")

    op.drop_index("ix_hospital_census_center_syn_id", table_name="hospital_census")
    op.drop_index("ix_hospital_census_census_date", table_name="hospital_census")
    op.drop_table("hospital_census")
