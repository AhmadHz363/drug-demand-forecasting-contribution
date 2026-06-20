"""add drug_catalog

Revision ID: 20260505_0002
Revises: 20260504_0001
Create Date: 2026-05-05
"""

from alembic import op
import sqlalchemy as sa


revision = "20260505_0002"
down_revision = "20260504_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "drug_catalog",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_code", sa.String(length=255), nullable=False),
        sa.Column("drug_name", sa.String(length=512), nullable=False),
        sa.Column("therapeutic_class", sa.String(length=128), nullable=False),
        sa.Column("atc_category", sa.String(length=32), nullable=False),
        sa.Column("pharmaceutical_form", sa.String(length=64), nullable=False),
        sa.Column("ven_class", sa.String(length=1), nullable=False),
        sa.Column("abc_class", sa.String(length=1), nullable=False),
        sa.Column("unit_price_tier", sa.Integer(), nullable=False),
        sa.Column("requires_refrigeration", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("is_controlled_substance", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("average_shelf_life_days", sa.Integer(), nullable=False),
        sa.Column("route_of_administration", sa.String(length=32), nullable=False),
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
        sa.UniqueConstraint("drug_code", name="uq_drug_catalog_drug_code"),
    )
    op.create_index("ix_drug_catalog_drug_code", "drug_catalog", ["drug_code"])


def downgrade() -> None:
    op.drop_index("ix_drug_catalog_drug_code", table_name="drug_catalog")
    op.drop_table("drug_catalog")
