"""add hospital_daily_demand_enriched training panel table

Revision ID: 20260813_0017
Revises: 20260813_0016
Create Date: 2026-08-13
"""

from alembic import op
import sqlalchemy as sa


revision = "20260813_0017"
down_revision = "20260813_0016"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hospital_daily_demand_enriched",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_file", sa.String(length=512), nullable=False),
        sa.Column("demand_date", sa.Date(), nullable=False),
        sa.Column("drug_code", sa.String(length=64), nullable=False),
        sa.Column("article", sa.String(length=512), nullable=True),
        sa.Column("cat", sa.Integer(), nullable=True),
        sa.Column("demand", sa.Integer(), server_default="0", nullable=False),
        sa.Column("n_patients_drug", sa.Integer(), server_default="0", nullable=False),
        sa.Column("n_doctors_drug", sa.Integer(), server_default="0", nullable=False),
        sa.Column("top_cr_drug", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("top_cs_drug", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("top_dr_drug", sa.String(length=255), nullable=True),
        sa.Column("n_unique_patients", sa.Integer(), server_default="0", nullable=False),
        sa.Column("n_admissions", sa.Integer(), server_default="0", nullable=False),
        sa.Column("n_unique_doctors", sa.Integer(), server_default="0", nullable=False),
        sa.Column("n_unique_cr", sa.Integer(), server_default="0", nullable=False),
        sa.Column("n_unique_cs", sa.Integer(), server_default="0", nullable=False),
        sa.Column("n_transactions", sa.Integer(), server_default="0", nullable=False),
        sa.Column("n_demand_txns", sa.Integer(), server_default="0", nullable=False),
        sa.Column("hospital_total_demand", sa.Integer(), server_default="0", nullable=False),
        sa.Column("age_mean", sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column("age_median", sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column("top1_cr", sa.Integer(), nullable=True),
        sa.Column("top1_cr_share", sa.Numeric(precision=10, scale=6), server_default="0", nullable=False),
        sa.Column("top2_cr", sa.Integer(), nullable=True),
        sa.Column("top2_cr_share", sa.Numeric(precision=10, scale=6), server_default="0", nullable=False),
        sa.Column("top3_cr", sa.Integer(), nullable=True),
        sa.Column("top3_cr_share", sa.Numeric(precision=10, scale=6), server_default="0", nullable=False),
        sa.Column(
            "imported_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_file",
            "demand_date",
            "drug_code",
            name="uq_hospital_daily_demand_enriched_source_drug_date",
        ),
    )
    op.create_index(
        "ix_hospital_daily_demand_enriched_source_file",
        "hospital_daily_demand_enriched",
        ["source_file"],
    )
    op.create_index(
        "ix_hospital_daily_demand_enriched_demand_date",
        "hospital_daily_demand_enriched",
        ["demand_date"],
    )
    op.create_index(
        "ix_hospital_daily_demand_enriched_drug_code",
        "hospital_daily_demand_enriched",
        ["drug_code"],
    )


def downgrade() -> None:
    op.drop_index("ix_hospital_daily_demand_enriched_drug_code", table_name="hospital_daily_demand_enriched")
    op.drop_index("ix_hospital_daily_demand_enriched_demand_date", table_name="hospital_daily_demand_enriched")
    op.drop_index("ix_hospital_daily_demand_enriched_source_file", table_name="hospital_daily_demand_enriched")
    op.drop_table("hospital_daily_demand_enriched")
