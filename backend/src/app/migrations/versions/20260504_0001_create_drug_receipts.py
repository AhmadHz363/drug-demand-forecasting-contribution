"""create drug_receipts

Revision ID: 20260504_0001
Revises:
Create Date: 2026-05-04
"""

from alembic import op
import sqlalchemy as sa


revision = "20260504_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "drug_receipts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("receipt_id", sa.String(length=255), nullable=False),
        sa.Column("line_count", sa.Integer(), nullable=True),
        sa.Column("drug_category", sa.String(length=255), nullable=True),
        sa.Column("center_receipt", sa.String(length=255), nullable=True),
        sa.Column("receipt_date", sa.Date(), nullable=False),
        sa.Column("movement_number", sa.String(length=255), nullable=True),
        sa.Column("movement_type", sa.String(length=255), nullable=True),
        sa.Column("drug_code", sa.String(length=255), nullable=False),
        sa.Column("drug_name", sa.String(length=1024), nullable=True),
        sa.Column("month", sa.Integer(), nullable=True),
        sa.Column("center_syn_id", sa.String(length=255), nullable=True),
        sa.Column("quantity", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("unit_price", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("total_price", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("admission_date", sa.Date(), nullable=True),
        sa.Column("room_number", sa.String(length=64), nullable=True),
        sa.Column("bed_number", sa.String(length=64), nullable=True),
        sa.Column("doctor_name", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_drug_receipts_receipt_id", "drug_receipts", ["receipt_id"])
    op.create_index("ix_drug_receipts_receipt_date", "drug_receipts", ["receipt_date"])
    op.create_index("ix_drug_receipts_drug_code", "drug_receipts", ["drug_code"])


def downgrade() -> None:
    op.drop_index("ix_drug_receipts_drug_code", table_name="drug_receipts")
    op.drop_index("ix_drug_receipts_receipt_date", table_name="drug_receipts")
    op.drop_index("ix_drug_receipts_receipt_id", table_name="drug_receipts")
    op.drop_table("drug_receipts")
