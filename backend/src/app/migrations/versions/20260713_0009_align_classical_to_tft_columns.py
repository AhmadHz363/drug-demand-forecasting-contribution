"""align classical ensemble columns to tft naming

Revision ID: 20260713_0009
Revises: 20260621_0008
Create Date: 2026-07-13
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "20260713_0009"
down_revision = "20260621_0008"
branch_labels = None
depends_on = None


def _rename_column_if_needed(table: str, old_name: str, new_name: str) -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in inspect(bind).get_columns(table)}
    if old_name in columns and new_name not in columns:
        op.alter_column(table, old_name, new_column_name=new_name)


def _create_feature_engineering_tables_if_missing() -> None:
    bind = op.get_bind()
    tables = set(inspect(bind).get_table_names())

    if "hospital_census" not in tables:
        op.create_table(
            "hospital_census",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("census_date", sa.Date(), nullable=False),
            sa.Column("bed_occupancy_rate", sa.Numeric(precision=5, scale=4), nullable=False),
            sa.Column("weekly_surgery_count", sa.Integer(), nullable=False),
            sa.Column("center_syn_id", sa.String(length=255), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_hospital_census_census_date", "hospital_census", ["census_date"])
        op.create_index("ix_hospital_census_center_syn_id", "hospital_census", ["center_syn_id"])

    if "supplier_lead_times" not in tables:
        op.create_table(
            "supplier_lead_times",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("drug_code", sa.String(length=255), nullable=False),
            sa.Column("avg_lead_time_days", sa.Numeric(precision=6, scale=2), nullable=False),
            sa.Column("lead_time_std_days", sa.Numeric(precision=6, scale=2), nullable=False),
            sa.Column("reliability_score", sa.Numeric(precision=4, scale=3), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("drug_code", name="uq_supplier_lead_times_drug_code"),
        )
        op.create_index("ix_supplier_lead_times_drug_code", "supplier_lead_times", ["drug_code"])


def upgrade() -> None:
    _rename_column_if_needed(
        "forecast_results",
        "model_weight_classical",
        "model_weight_tft",
    )
    _rename_column_if_needed(
        "model_performance",
        "weight_classical",
        "weight_tft",
    )
    _create_feature_engineering_tables_if_missing()


def downgrade() -> None:
    _rename_column_if_needed(
        "forecast_results",
        "model_weight_tft",
        "model_weight_classical",
    )
    _rename_column_if_needed(
        "model_performance",
        "weight_tft",
        "weight_classical",
    )
