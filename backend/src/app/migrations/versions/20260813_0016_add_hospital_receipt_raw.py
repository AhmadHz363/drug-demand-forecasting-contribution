"""add hospital_receipt_raw table for unparsed Excel exports

Revision ID: 20260813_0016
Revises: 20260813_0015
Create Date: 2026-08-13
"""

from alembic import op
import sqlalchemy as sa


revision = "20260813_0016"
down_revision = "20260813_0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "hospital_receipt_raw",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_file", sa.String(length=512), nullable=False),
        sa.Column("source_sheet", sa.String(length=255), nullable=False),
        sa.Column("excel_row_number", sa.Integer(), nullable=False),
        sa.Column("doc", sa.BigInteger(), nullable=True),
        sa.Column("line", sa.Integer(), nullable=True),
        sa.Column("cat", sa.Integer(), nullable=True),
        sa.Column("c_r", sa.Integer(), nullable=True),
        sa.Column("date_raw", sa.Text(), nullable=True),
        sa.Column("mov_num", sa.Integer(), nullable=True),
        sa.Column("mov_des", sa.String(length=255), nullable=True),
        sa.Column("code", sa.String(length=64), nullable=True),
        sa.Column("article", sa.String(length=512), nullable=True),
        sa.Column("m", sa.Integer(), nullable=True),
        sa.Column("c_s", sa.Integer(), nullable=True),
        sa.Column("qty", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("u_p", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("t_p", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("mrn", sa.String(length=64), nullable=True),
        sa.Column("ad_date_raw", sa.Text(), nullable=True),
        sa.Column("r", sa.String(length=64), nullable=True),
        sa.Column("u", sa.String(length=64), nullable=True),
        sa.Column("age_raw", sa.Text(), nullable=True),
        sa.Column("dr", sa.String(length=255), nullable=True),
        sa.Column(
            "imported_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_file",
            "source_sheet",
            "excel_row_number",
            name="uq_hospital_receipt_raw_source_row",
        ),
    )
    op.create_index(
        "ix_hospital_receipt_raw_source_file",
        "hospital_receipt_raw",
        ["source_file"],
    )
    op.create_index(
        "ix_hospital_receipt_raw_code",
        "hospital_receipt_raw",
        ["code"],
    )


def downgrade() -> None:
    op.drop_index("ix_hospital_receipt_raw_code", table_name="hospital_receipt_raw")
    op.drop_index("ix_hospital_receipt_raw_source_file", table_name="hospital_receipt_raw")
    op.drop_table("hospital_receipt_raw")
