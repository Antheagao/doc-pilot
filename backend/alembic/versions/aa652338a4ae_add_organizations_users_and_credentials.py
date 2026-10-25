"""add organizations, users, memberships, sessions, and API keys

Revision ID: aa652338a4ae
Revises: b4e7c1a9d625
Create Date: 2026-10-25 10:00:00.000000

Retention: sign-in sessions are deleted at logout and when their user's
password changes, and expired ones are swept at each login and by
`scripts/accounts.py prune-sessions`; user emails are kept until an operator
removes the user; API keys are never deleted by the app (a revoked key keeps
its row for audit).

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'aa652338a4ae'
down_revision: str | Sequence[str] | None = 'b4e7c1a9d625'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Copied from app.models.DEFAULT_ORGANIZATION_ID, not imported: a migration
# keeps meaning what it meant when it was written.
DEFAULT_ORGANIZATION_ID = '270ad65a-150a-498e-bbb1-584f6d21d86e'


def upgrade() -> None:
    """Upgrade schema."""
    # users.email is citext so that "A@x.com" and "a@x.com" are one account.
    op.execute('CREATE EXTENSION IF NOT EXISTS citext')

    op.create_table(
        'organizations',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('slug', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint('char_length(name) BETWEEN 1 AND 100', name='ck_organizations_name'),
        sa.CheckConstraint("slug ~ '^[a-z0-9][a-z0-9-]{1,62}$'", name='ck_organizations_slug'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('slug'),
    )
    op.create_table(
        'users',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('email', postgresql.CITEXT(), nullable=False),
        # The encoded argon2id hash, never the password.
        sa.Column('password_hash', sa.Text(), nullable=False),
        # NULL while the user may sign in.
        sa.Column('disabled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint('char_length(email) BETWEEN 1 AND 254', name='ck_users_email'),
        # Exactly one "@" with something on each side, and no whitespace or
        # control characters. The cast keeps the match case-sensitive and
        # independent of citext's own operators.
        sa.CheckConstraint(
            "email::text ~ '^[^@[:space:][:cntrl:]]+@[^@[:space:][:cntrl:]]+$'",
            name='ck_users_email_shape',
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
    )
    # The primary key leads with organization_id, so it also serves that
    # foreign key; the user_id index serves the other one.
    op.create_table(
        'memberships',
        sa.Column('organization_id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('role', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("role IN ('owner', 'member')", name='ck_memberships_role'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('organization_id', 'user_id'),
    )
    op.create_index('ix_memberships_user_id', 'memberships', ['user_id'], unique=False)
    # Only the SHA-256 digest of the cookie token is stored. The composite
    # foreign key ends a user's sessions in an organization when the
    # membership goes.
    op.create_table(
        'auth_sessions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('token_digest', sa.LargeBinary(), nullable=False),
        sa.Column('organization_id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint('octet_length(token_digest) = 32', name='ck_auth_sessions_token_digest'),
        sa.ForeignKeyConstraint(
            ['organization_id', 'user_id'],
            ['memberships.organization_id', 'memberships.user_id'],
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token_digest'),
    )
    op.create_index(
        'ix_auth_sessions_organization_id_user_id',
        'auth_sessions',
        ['organization_id', 'user_id'],
        unique=False,
    )
    op.create_index('ix_auth_sessions_user_id', 'auth_sessions', ['user_id'], unique=False)
    # The sweep of expired sessions filters on it.
    op.create_index('ix_auth_sessions_expires_at', 'auth_sessions', ['expires_at'], unique=False)
    # prefix is stored in the clear to find the row and to show in a list; the
    # digest is of the whole token, which is shown once and never stored.
    op.create_table(
        'api_keys',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('organization_id', sa.UUID(), nullable=False),
        # NULL when an operator made the key without naming a user, or after
        # that user was removed.
        sa.Column('created_by_user_id', sa.UUID(), nullable=True),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('prefix', sa.Text(), nullable=False),
        sa.Column('token_digest', sa.LargeBinary(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        # NULL until first use.
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        # NULL while the key is active.
        sa.Column('revoked_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint('char_length(name) BETWEEN 1 AND 100', name='ck_api_keys_name'),
        sa.CheckConstraint("prefix ~ '^[a-z2-7]{12}$'", name='ck_api_keys_prefix'),
        sa.CheckConstraint('octet_length(token_digest) = 32', name='ck_api_keys_token_digest'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('prefix'),
    )
    op.create_index('ix_api_keys_organization_id', 'api_keys', ['organization_id'], unique=False)
    op.create_index('ix_api_keys_created_by_user_id', 'api_keys', ['created_by_user_id'], unique=False)

    # Every row that exists before multi-tenancy belongs here.
    op.execute(
        "INSERT INTO organizations (id, name, slug) "
        f"VALUES ('{DEFAULT_ORGANIZATION_ID}', 'Default', 'default')"
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Dropping the tables discards every account, session, and key.
    op.drop_table('api_keys')
    op.drop_table('auth_sessions')
    op.drop_table('memberships')
    op.drop_table('users')
    op.drop_table('organizations')
    # users.email was the only citext column; the extension goes with it.
    op.execute('DROP EXTENSION IF EXISTS citext')
