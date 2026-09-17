"""add import_coverage table for receipt ledger tracking

Revision ID: 20260905_0022
Revises: 20260905_0021
Create Date: 2026-09-05
"""

from alembic import op
import sqlalchemy as sa


revision = "20260905_0022"
down_revision = "20260905_0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "import_coverage",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("min_receipt_date", sa.Date(), nullable=False),
        sa.Column("max_receipt_date", sa.Date(), nullable=False),
        sa.Column("row_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("center_scope", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), server_default="success", nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
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
        sa.UniqueConstraint("content_hash", name="uq_import_coverage_content_hash"),
    )
    op.create_index("ix_import_coverage_content_hash", "import_coverage", ["content_hash"])
    op.create_index("ix_import_coverage_min_receipt_date", "import_coverage", ["min_receipt_date"])
    op.create_index("ix_import_coverage_max_receipt_date", "import_coverage", ["max_receipt_date"])


def downgrade() -> None:
    op.drop_index("ix_import_coverage_max_receipt_date", table_name="import_coverage")
    op.drop_index("ix_import_coverage_min_receipt_date", table_name="import_coverage")
    op.drop_index("ix_import_coverage_content_hash", table_name="import_coverage")
    op.drop_table("import_coverage")
