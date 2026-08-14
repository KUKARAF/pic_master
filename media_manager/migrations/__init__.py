"""Numbered, hand-written schema migrations for manual.db, applied via
`media migrate` (media.py) and tracked in the `migration_state` table (one row
per migration id, see manual_db.py's ManualDB._create_tables).

Deliberately NOT auto-generated the way Django's makemigrations diffs model
classes to write migration files: this codebase has no ORM/model layer to
diff against, just hand-written CREATE TABLE SQL, so there's nothing to
generate from. Each migration here is instead a small hand-written module
exposing:

    ID              — a stable string key, used as the migration_state row key
                       (module filenames can't start with a digit, hence the
                       `m0001_...` filename vs. the `0001_...` id string)
    describe_pending(conn) -> str | None
                    — a human-readable description of what would change, or
                      None if there's nothing to do (already migrated, or a
                      fresh install that was never in the old shape to begin
                      with)
    apply(conn)     — performs the migration. Assumes `conn` is already inside
                      a transaction (see manual_db.run_migrations) — issues
                      DDL/DML only, verifies its own result, raises on
                      mismatch so the caller can roll back. Returns a short
                      human-readable summary string on success.

`conn` is a raw sqlite3 connection (or cursor — both support the
.execute(...).fetchone()/.fetchall() calls these use), never a ManualDB
instance: ManualDB._create_tables() itself refuses to run against a database
with any pending migration, so a ManualDB can't be constructed until
`media migrate --apply` has been run.

List migrations here in the order they must run.
"""
from . import m0001_tags_table

MIGRATIONS = [
    m0001_tags_table,
]
