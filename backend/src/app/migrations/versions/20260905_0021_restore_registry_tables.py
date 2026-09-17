"""restore receipt registry tables and rename CAMEO drugs table

Revision ID: 20260905_0021
Revises: 20260824_0020
Create Date: 2026-09-05
"""

from alembic import op
import sqlalchemy as sa


revision = "20260905_0021"
down_revision = "20260824_0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # CAMEO attribute rows live in cameo_drugs; drugs is reserved for receipt registry.
    op.rename_table("drugs", "cameo_drugs")
    op.execute("ALTER INDEX IF EXISTS ix_drugs_drug_code RENAME TO ix_cameo_drugs_drug_code")
    op.execute("ALTER INDEX IF EXISTS ix_drugs_match_status RENAME TO ix_cameo_drugs_match_status")
    op.execute("ALTER INDEX IF EXISTS uq_drugs_drug_code RENAME TO uq_cameo_drugs_drug_code")

    op.create_table(
        "categories",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("category_code", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("receipt_count", sa.Integer(), server_default="0", nullable=False),
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
        sa.UniqueConstraint("category_code", name="uq_categories_category_code"),
    )
    op.create_index("ix_categories_category_code", "categories", ["category_code"])

    op.create_table(
        "drugs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_code", sa.String(length=255), nullable=False),
        sa.Column("drug_name", sa.String(length=1024), nullable=True),
        sa.Column("drug_category", sa.String(length=255), nullable=True),
        sa.Column("receipt_count", sa.Integer(), server_default="0", nullable=False),
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
        sa.UniqueConstraint("drug_code", name="uq_drugs_drug_code"),
    )
    op.create_index("ix_drugs_drug_code", "drugs", ["drug_code"])

    op.create_table(
        "drug_receipts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_id", sa.Integer(), nullable=False),
        sa.Column("category_id", sa.Integer(), nullable=True),
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
        sa.ForeignKeyConstraint(["category_id"], ["categories.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["drug_id"], ["drugs.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_drug_receipts_receipt_id", "drug_receipts", ["receipt_id"])
    op.create_index("ix_drug_receipts_receipt_date", "drug_receipts", ["receipt_date"])
    op.create_index("ix_drug_receipts_drug_code", "drug_receipts", ["drug_code"])
    op.create_index("ix_drug_receipts_drug_id", "drug_receipts", ["drug_id"])
    op.create_index("ix_drug_receipts_category_id", "drug_receipts", ["category_id"])


def downgrade() -> None:
    op.drop_index("ix_drug_receipts_category_id", table_name="drug_receipts")
    op.drop_index("ix_drug_receipts_drug_id", table_name="drug_receipts")
    op.drop_index("ix_drug_receipts_drug_code", table_name="drug_receipts")
    op.drop_index("ix_drug_receipts_receipt_date", table_name="drug_receipts")
    op.drop_index("ix_drug_receipts_receipt_id", table_name="drug_receipts")
    op.drop_table("drug_receipts")

    op.drop_index("ix_drugs_drug_code", table_name="drugs")
    op.drop_table("drugs")

    op.drop_index("ix_categories_category_code", table_name="categories")
    op.drop_table("categories")

    op.execute("ALTER INDEX IF EXISTS uq_cameo_drugs_drug_code RENAME TO uq_drugs_drug_code")
    op.execute("ALTER INDEX IF EXISTS ix_cameo_drugs_match_status RENAME TO ix_drugs_match_status")
    op.execute("ALTER INDEX IF EXISTS ix_cameo_drugs_drug_code RENAME TO ix_drugs_drug_code")
    op.rename_table("cameo_drugs", "drugs")
