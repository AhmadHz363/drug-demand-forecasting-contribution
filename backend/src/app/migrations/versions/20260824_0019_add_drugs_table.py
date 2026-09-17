"""add drugs table for CAMEO cold-start attributes

Revision ID: 20260824_0019
Revises: 20260814_0018
Create Date: 2026-08-24
"""

from alembic import op
import sqlalchemy as sa


revision = "20260824_0019"
down_revision = "20260814_0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "drugs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_code", sa.String(length=64), nullable=False),
        sa.Column("generic_name", sa.String(length=512), nullable=True),
        sa.Column("drug_class", sa.String(length=512), nullable=True),
        sa.Column("indications", sa.Text(), nullable=True),
        sa.Column("dosage_form", sa.String(length=255), nullable=True),
        sa.Column("strength", sa.String(length=255), nullable=True),
        sa.Column("route_of_administration", sa.String(length=255), nullable=True),
        sa.Column("side_effects", sa.Text(), nullable=True),
        sa.Column("contraindications", sa.Text(), nullable=True),
        sa.Column("interaction_warnings_precautions", sa.Text(), nullable=True),
        sa.Column("storage_conditions", sa.Text(), nullable=True),
        sa.Column("pregnancy_category", sa.String(length=128), nullable=True),
        sa.Column("reference", sa.Text(), nullable=True),
        sa.Column("availability", sa.String(length=128), nullable=True),
        sa.Column("input_id", sa.Integer(), nullable=True),
        sa.Column("input_drug_name", sa.String(length=1024), nullable=True),
        sa.Column("resolved_generic", sa.String(length=512), nullable=True),
        sa.Column("matched_source_generic", sa.String(length=512), nullable=True),
        sa.Column("match_status", sa.String(length=64), nullable=True),
        sa.Column("match_method", sa.String(length=128), nullable=True),
        sa.Column("receipt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("synthetic_cells_filled", sa.Integer(), server_default="0", nullable=False),
        sa.Column("training_quality_note", sa.Text(), nullable=True),
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
    op.create_index("ix_drugs_match_status", "drugs", ["match_status"])


def downgrade() -> None:
    op.drop_index("ix_drugs_match_status", table_name="drugs")
    op.drop_index("ix_drugs_drug_code", table_name="drugs")
    op.drop_table("drugs")
