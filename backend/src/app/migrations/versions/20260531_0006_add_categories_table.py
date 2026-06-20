"""add categories table and link drug_receipts

Revision ID: 20260531_0006
Revises: 20260531_0005
Create Date: 2026-05-31
"""

from alembic import op
import sqlalchemy as sa


revision = "20260531_0006"
down_revision = "20260531_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
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

    op.execute(
        """
        INSERT INTO categories (category_code, name, receipt_count, created_at, updated_at)
        SELECT
            drug_category AS category_code,
            NULL AS name,
            COUNT(*) AS receipt_count,
            now() AS created_at,
            now() AS updated_at
        FROM drug_receipts
        WHERE drug_category IS NOT NULL AND TRIM(drug_category) <> ''
        GROUP BY drug_category
        """
    )

    op.add_column("drug_receipts", sa.Column("category_id", sa.Integer(), nullable=True))
    op.execute(
        """
        UPDATE drug_receipts dr
        SET category_id = c.id
        FROM categories c
        WHERE dr.drug_category = c.category_code
        """
    )
    op.create_index("ix_drug_receipts_category_id", "drug_receipts", ["category_id"])
    op.create_foreign_key(
        "fk_drug_receipts_category_id_categories",
        "drug_receipts",
        "categories",
        ["category_id"],
        ["id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_drug_receipts_category_id_categories",
        "drug_receipts",
        type_="foreignkey",
    )
    op.drop_index("ix_drug_receipts_category_id", table_name="drug_receipts")
    op.drop_column("drug_receipts", "category_id")
    op.drop_index("ix_categories_category_code", table_name="categories")
    op.drop_table("categories")
