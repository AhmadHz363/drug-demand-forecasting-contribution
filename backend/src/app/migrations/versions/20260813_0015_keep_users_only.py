"""keep only users table — drop all other application tables

Revision ID: 20260813_0015
Revises: 20260813_0014
Create Date: 2026-08-13
"""

from alembic import op

revision = "20260813_0015"
down_revision = "20260813_0014"
branch_labels = None
depends_on = None

_TABLES_TO_DROP = (
    "forecast_training_data",
    "forecast_results",
    "model_performance",
    "import_coverage",
    "hospital_census",
    "drug_receipts",
    "drugs",
    "categories",
    "drug_catalog",
)


def upgrade() -> None:
    for table in _TABLES_TO_DROP:
        op.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')


def downgrade() -> None:
    raise NotImplementedError(
        "Downgrade is not supported for 20260813_0015; re-run earlier migrations instead."
    )
