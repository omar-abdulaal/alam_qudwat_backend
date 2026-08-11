"""multi-dataset provenance: printed_volume, dataset_id, source_content_hash

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-12

Additive/nullable only — existing Rashidun rows are unaffected (they get
NULL for all three new columns; nothing about them is rewritten).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("printed_volume", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("dataset_id", sa.String(length=64), nullable=True))
    op.add_column("documents", sa.Column("source_content_hash", sa.String(length=64), nullable=True))

    op.add_column("chunks", sa.Column("printed_volume", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("dataset_id", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "dataset_id")
    op.drop_column("chunks", "printed_volume")

    op.drop_column("documents", "source_content_hash")
    op.drop_column("documents", "dataset_id")
    op.drop_column("documents", "printed_volume")
