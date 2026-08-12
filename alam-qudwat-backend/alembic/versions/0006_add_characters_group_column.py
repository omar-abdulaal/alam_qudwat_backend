"""add characters.group, backfilled from the current category values

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-12

`category` was, until now, doing double duty as both the broad
classification used for grouping/filtering ("الخلفاء الراشدون",
"الصحابة") and the per-person specific role. Splitting them: `group`
takes over the broad-classification role with today's `category` values
as its starting point, freeing `category` to later hold a specific
per-person label (e.g. "خليفة", "فقيه", "طبيب") without breaking
grouping/filtering.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("characters", sa.Column("group", sa.Text(), nullable=True))
    op.execute('UPDATE characters SET "group" = category')
    op.alter_column("characters", "group", nullable=False)


def downgrade() -> None:
    op.drop_column("characters", "group")
