"""Add import_coverage and rename tft weight columns back to classical.

Revision ID: 20260713_0010
Revises: 20260713_0009
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260713_0010"
down_revision: Union[str, None] = "20260713_0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _rename_column_if_needed(table: str, old: str, new: str) -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if table not in inspector.get_table_names():
        return
    columns = {col["name"] for col in inspector.get_columns(table)}
    if old in columns and new not in columns:
        op.alter_column(table, old, new_column_name=new)


def upgrade() -> None:
    op.create_table(
        "import_coverage",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("min_receipt_date", sa.Date(), nullable=False),
        sa.Column("max_receipt_date", sa.Date(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("center_scope", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="success"),
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

    _rename_column_if_needed("forecast_results", "model_weight_tft", "model_weight_classical")
    _rename_column_if_needed("model_performance", "weight_tft", "weight_classical")


def downgrade() -> None:
    _rename_column_if_needed("model_performance", "weight_classical", "weight_tft")
    _rename_column_if_needed("forecast_results", "model_weight_classical", "model_weight_tft")
    op.drop_index("ix_import_coverage_max_receipt_date", table_name="import_coverage")
    op.drop_index("ix_import_coverage_min_receipt_date", table_name="import_coverage")
    op.drop_index("ix_import_coverage_content_hash", table_name="import_coverage")
    op.drop_table("import_coverage")
