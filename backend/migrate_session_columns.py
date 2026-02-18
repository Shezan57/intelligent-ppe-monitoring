"""
Database Migration: Add Session Tracking Columns to violations table

Adds 5 new columns for session-based violation tracking:
  - session_start        DATETIME  - When violation session first started
  - last_seen            DATETIME  - Last time worker was detected violating
  - occurrence_count     INTEGER   - How many times re-detected in this session
  - total_duration_minutes REAL    - Total violation duration in minutes
  - is_active_session    BOOLEAN   - Is worker still in frame?

Run: python migrate_session_columns.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "ppe_detection.db")

NEW_COLUMNS = [
    ("session_start",           "DATETIME",  None),
    ("last_seen",               "DATETIME",  None),
    ("occurrence_count",        "INTEGER",   1),
    ("total_duration_minutes",  "REAL",      0.0),
    ("is_active_session",       "BOOLEAN",   1),   # 1 = True in SQLite
]

def run_migration():
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found at: {DB_PATH}")
        return False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(violations)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    print(f"📋 Existing columns: {sorted(existing_cols)}")

    added = []
    skipped = []

    for col_name, col_type, default in NEW_COLUMNS:
        if col_name in existing_cols:
            skipped.append(col_name)
            continue

        # Build ALTER TABLE statement
        if default is not None:
            sql = f"ALTER TABLE violations ADD COLUMN {col_name} {col_type} DEFAULT {default}"
        else:
            sql = f"ALTER TABLE violations ADD COLUMN {col_name} {col_type}"

        cursor.execute(sql)
        added.append(col_name)
        print(f"  ✅ Added column: {col_name} {col_type}")

    conn.commit()

    # Backfill existing rows: set session_start = timestamp, last_seen = timestamp
    if "session_start" in added or "last_seen" in added:
        cursor.execute("""
            UPDATE violations
            SET session_start = timestamp,
                last_seen = timestamp
            WHERE session_start IS NULL
        """)
        updated = cursor.rowcount
        conn.commit()
        print(f"  📝 Backfilled {updated} existing row(s) with session_start/last_seen = timestamp")

    conn.close()

    print(f"\n✅ Migration complete!")
    print(f"   Added:   {added if added else 'none (all already existed)'}")
    print(f"   Skipped: {skipped if skipped else 'none'}")
    return True


if __name__ == "__main__":
    run_migration()
