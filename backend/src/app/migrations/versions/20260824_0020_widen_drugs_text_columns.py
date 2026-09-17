"""widen drugs text columns for long resolved_generic values

Revision ID: 20260824_0020
Revises: 20260824_0019
Create Date: 2026-08-24
"""

from alembic import op
import sqlalchemy as sa


revision = "20260824_0020"
down_revision = "20260824_0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("drugs", "resolved_generic", type_=sa.Text(), existing_nullable=True)
    op.alter_column("drugs", "matched_source_generic", type_=sa.Text(), existing_nullable=True)
    op.alter_column("drugs", "generic_name", type_=sa.Text(), existing_nullable=True)
    op.alter_column("drugs", "drug_class", type_=sa.Text(), existing_nullable=True)


def downgrade() -> None:
    op.alter_column("drugs", "drug_class", type_=sa.String(length=512), existing_nullable=True)
    op.alter_column("drugs", "generic_name", type_=sa.String(length=512), existing_nullable=True)
    op.alter_column("drugs", "matched_source_generic", type_=sa.String(length=512), existing_nullable=True)
    op.alter_column("drugs", "resolved_generic", type_=sa.String(length=512), existing_nullable=True)
