"""add drugs table and link drug_receipts

Revision ID: 20260531_0005
Revises: 20260506_0004
Create Date: 2026-05-31
"""

from alembic import op
import sqlalchemy as sa


revision = "20260531_0005"
down_revision = "20260506_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
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

    op.execute(
        """
        INSERT INTO drugs (drug_code, drug_name, drug_category, receipt_count, created_at, updated_at)
        SELECT
            drug_code,
            MAX(drug_name) AS drug_name,
            MAX(drug_category) AS drug_category,
            COUNT(*) AS receipt_count,
            now() AS created_at,
            now() AS updated_at
        FROM drug_receipts
        GROUP BY drug_code
        """
    )

    op.add_column("drug_receipts", sa.Column("drug_id", sa.Integer(), nullable=True))
    op.execute(
        """
        UPDATE drug_receipts dr
        SET drug_id = d.id
        FROM drugs d
        WHERE dr.drug_code = d.drug_code
        """
    )
    op.alter_column("drug_receipts", "drug_id", nullable=False)
    op.create_index("ix_drug_receipts_drug_id", "drug_receipts", ["drug_id"])
    op.create_foreign_key(
        "fk_drug_receipts_drug_id_drugs",
        "drug_receipts",
        "drugs",
        ["drug_id"],
        ["id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint("fk_drug_receipts_drug_id_drugs", "drug_receipts", type_="foreignkey")
    op.drop_index("ix_drug_receipts_drug_id", table_name="drug_receipts")
    op.drop_column("drug_receipts", "drug_id")
    op.drop_index("ix_drugs_drug_code", table_name="drugs")
    op.drop_table("drugs")
